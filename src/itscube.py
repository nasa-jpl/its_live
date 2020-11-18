import copy
import dask
from dask.distributed import Client
from datetime import datetime, timedelta
import glob
import os
import numpy  as np
import matplotlib.pyplot as plt
import s3fs
import xarray as xr
import xarray.plot as xplt


from itslive import itslive_ui


class ITSCube:
    """
    Class to build ITS_LIVE cube: time series of velocity pairs within a
    polygon of interest.
    """

    class Bounds:
        """
        Class to store min/max pair for a variable.
        """
        def __init__(self, values):
            self.min = min(values)
            self.max = max(values)

        def __str__(self):
            return f"min={self.min} max={self.max}"

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = '4326'

    S3_PREFIX = 's3://'
    HTTP_PREFIX = 'http://'

    # Token within granule's URL that needs to be removed to get file location within S3 bucket:
    # if URL is of the 'http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/velocity_image_pair/landsat/v00.0/32628/file.nc' format,
    # S3 bucket location of the file is 's3://its-live-data.jpl.nasa.gov/velocity_image_pair/landsat/v00.0/32628/file.nc'
    PATH_URL = ".s3.amazonaws.com"

    NC_ENGINE = 'h5netcdf'

    def __init__(self, polygon: tuple, projection: str):
        """
        polygon: tuple
            Polygon for the tile.
        projection: str
            Projection in which polygon is defined.
        """
        self.projection = projection

        # Set min/max x/y values to filter region by
        self.x = ITSCube.Bounds([each[0] for each in polygon])
        self.y = ITSCube.Bounds([each[1] for each in polygon])

        # Convert polygon from its target projection to longitude/latitude coordinates
        # which are used by granule search API
        self.polygon_coords = []

        for each in polygon:
            coords = itslive_ui.transform_coord(projection, ITSCube.LON_LAT_PROJECTION, each[0], each[1])
            self.polygon_coords.extend(coords)

        print(f"Longitude/latitude coords for polygon: {self.polygon_coords}")

        # Dictionary to store filtered by region/start_date/end_date velocity pairs
        # in the following format:
        #    mid_date: velocity values
        self.velocities = {}

        self.layers = None

    def create(self, api_params: dict, num_granules=None):
        """
        Create velocity cube.

        api_params: dict
            Search API required parameters.
        num: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        # Re-set filtered velocities
        self.velocities = {}
        self.layers = None

        # Append polygon information to API's parameters
        params = copy.deepcopy(api_params)
        params['polygon'] = ",".join([str(each) for each in self.polygon_coords])

        found_urls = [each['url'] for each in itslive_ui.get_granule_urls(params)]
        print("Originally found urls: ", len(found_urls))

        if len(found_urls) == 0:
            print(f"No granules are found for the search API parameters: {params}")
            return

        # Keep track of skipped granules due to the other than target projection
        skipped_proj_granules = {}
        # Keep track of skipped granules due to the no data for the polygon of interest
        skipped_empty_granules = []

        s3 = s3fs.S3FileSystem(anon=True)

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        if num_granules:
            found_urls = found_urls[:num_granules]
            print(f"Examining only {len(found_urls)} first granules")

        for each_url in found_urls:
            s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
            s3_path = s3_path.replace(ITSCube.PATH_URL, '')

            with s3.open(s3_path, mode='rb') as fhandle:
                with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                    cube_v, is_empty = self.preprocess_dataset(ds, each_url)

                    if cube_v is not None:
                        # There can be multiple layers for the mid_date, collect all
                        self.velocities.setdefault(cube_v.coords['mid_date'].values, []).append(cube_v)

                    else:
                        if is_empty:
                            skipped_empty_granules.append(each_url)

                        else:
                            skipped_proj_granules.setdefault(int(ds.UTM_Projection.spatial_epsg), []).append(each_url)

        self.combine_layers()
        ITSCube.format_stats(skipped_proj_granules, skipped_empty_granules, len(found_urls))

        return found_urls, skipped_proj_granules

    def create_parallel(self, api_params: dict, num_granules=None):
        """
        Create velocity cube by reading and pre-processing cube layers in parallel.

        api_params: dict
            Search API required parameters.
        num: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules. Should not be used
                  when running the code at AWS.
        """
        # Re-set filtered velocities
        self.velocities = {}
        self.layers = None

        # Append polygon information to API's parameters
        params = copy.deepcopy(api_params)
        params['polygon'] = ",".join([str(each) for each in self.polygon_coords])

        found_urls = [each['url'] for each in itslive_ui.get_granule_urls(params)]
        print("Originally found urls: ", len(found_urls))

        if len(found_urls) == 0:
            print(f"No granules are found for the search API parameters: {params}")
            return

        # Keep track of skipped granules due to the other than target projection
        skipped_proj_granules = {}
        # Keep track of skipped granules due to the no data for the polygon of interest
        skipped_empty_granules = []

        # Parallelize layer collection
        s3 = s3fs.S3FileSystem(anon=True)

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        if num_granules:
            found_urls = found_urls[:num_granules]
            print(f"Examining only {len(found_urls)} first granules")

        # Create Dask client for processing: using "threads" scheduler
        # client = Client(processes=False, n_workers=8)
        tasks = [dask.delayed(self.read_s3_dataset)(each_file, s3) for each_file in found_urls]
        results = dask.compute(tasks, scheduler='processes', num_workers=8)

        for each_ds in results[0]:
            cube_v, is_empty = each_ds

            if cube_v is not None:
                # There can be multiple layers for the mid_date, collect all
                self.velocities.setdefault(cube_v.coords['mid_date'].values, []).append(cube_v)

            else:
                if is_empty:
                    skipped_empty_granules.append(cube_v.attrs['url'])

                else:
                    skipped_proj_granules.setdefault(int(each_ds.UTM_Projection.spatial_epsg), []).append(cube_v.attrs['url'])

        self.combine_layers()
        ITSCube.format_stats(skipped_proj_granules, skipped_empty_granules, len(found_urls))

        return found_urls, skipped_proj_granules


    def create_from_local(self, dirpath='data'):
        """
        Create velocity cube by accessing local data from dirpath.

        api_params: dict
            Search API required parameters.
        num: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules. Should not be used
                  when running the code at AWS.
        """
        # Re-set filtered velocities
        self.velocities = {}
        self.layers = None

        # Keep track of skipped granules due to the other than target projection
        skipped_proj_granules = {}
        # Keep track of skipped granules due to the no data for the polygon of interest
        skipped_empty_granules = []

        found_urls = glob.glob(dirpath + os.sep + '*.nc')

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        for each_url in found_urls:
            with xr.open_dataset(each_url) as ds:
                cube_v, is_empty = self.preprocess_dataset(ds, each_url)

                if cube_v is not None:
                    # There can be multiple layers for the mid_date, collect all
                    self.velocities.setdefault(cube_v.coords['mid_date'].values, []).append(cube_v)

                else:
                    if is_empty:
                         skipped_empty_granules.append(each_url)

                    else:
                        skipped_proj_granules.setdefault(int(ds.UTM_Projection.spatial_epsg), []).append(each_url)

        self.combine_layers()
        ITSCube.format_stats(skipped_proj_granules, skipped_empty_granules, len(found_urls))

        return found_urls, skipped_proj_granules


    def preprocess_dataset(self, ds: xr.Dataset, ds_url: str):
        """
        Pre-process ITS_LIVE dataset in preparation for the cube layer.

        ds: xarray dataset
            Dataset to pre-process.
        ds_url: str
            URL that corresponds to the dataset.

        Returns:
        mid_date: Middle date for the layer
        cube_v:   Filtered data array for the layer.
        empty:    Flag to indicate if dataset does not contain any data for the cube region.
        """
        # Flag if layer data is empty.
        empty  = False

        # Layer velocity data
        cube_v = None

        # Consider granules with data only within target projection
        if str(int(ds.UTM_Projection.spatial_epsg)) == self.projection:
            mid_date = datetime.strptime(ds.img_pair_info.date_center, '%Y%m%d')

            # Add date separation in days as milliseconds for the middle date
            # (avoid resolution issues for layers with the same middle date).
            mid_date += timedelta(milliseconds=int(ds.img_pair_info.date_dt))

            # Define which points are within target polygon.
            mask_lon = (ds.x >= self.x.min) & (ds.x <= self.x.max)
            mask_lat = (ds.y >= self.y.min) & (ds.y <= self.y.max)
            cube_v = ds.where(mask_lon & mask_lat, drop=True).v
            # Another way to filter: cube_v = ds.v.sel(x=slice(self.x.min, self.x.max),y=slice(self.y.max, self.y.min)).copy()

            # If it's a valid velocity layer, add it to the cube.
            if np.any(cube_v.notnull()):
                cube_v.load()

                # Add middle date as a new coordinate
                cube_v = cube_v.assign_coords({'mid_date': mid_date})

                # Add file URL as its source for traceability
                cube_v.attrs['url'] = ds_url

            else:
                empty = True

        return cube_v, empty


    def combine_layers(self):
        """
        Combine selected layers into one array.
        """
        # Construct xarray to hold layers by concatenating layer objects along 'mid_date' dimension
        layers_urls = []
        for each_index, each_date in enumerate(sorted(self.velocities.keys())):
            start_index = 0
            if each_index == 0:
                # This is very first layer
                self.layers = self.velocities[each_date][0]
                layers_urls.append(self.velocities[each_date][0].attrs['url'])
                start_index = 1

            if len(self.velocities[each_date]) > start_index:
                for each_layer in self.velocities[each_date][start_index:]:
                    self.layers = xr.concat([self.layers, each_layer], 'mid_date')
                    layers_urls.append(each_layer.attrs['url'])

        self.layers.attrs['url'] = layers_urls
        self.layers.attrs['projection'] = str(self.projection)

        # No need for collected velocities pairs anymore
#         self.velocities = None


    @staticmethod
    def format_stats(skipped_proj_granules, skipped_empty_granules, num_urls):

        # Total number of skipped granules due to wrong projection
        sum_projs = sum([len(each) for each in skipped_proj_granules.values()])

        print( "Skipped granules:")
        print(f"      empty data: {len(skipped_empty_granules)} ({100.0 * len(skipped_empty_granules)/num_urls}%)")
        print(f"      wrong proj: {sum_projs} ({100.0 * sum_projs/num_urls}%)")


    def create_from_local_parallel(self, dirpath='data'):
        """
        Create velocity cube from local data in parallel.

        api_params: dict
            Search API required parameters.
        num: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules. Should not be used
                  when running the code at AWS.
        """
        # Re-set filtered velocities
        self.velocities = {}
        self.layers = None

        # Keep track of skipped granules due to the other than target projection
        skipped_proj_granules = {}
        # Keep track of skipped granules due to the no data for the polygon of interest
        skipped_empty_granules = []

        found_urls = glob.glob(dirpath + os.sep + '*.nc')

        tasks = [dask.delayed(self.read_dataset)(each_file) for each_file in found_urls]
        results = dask.compute(tasks, scheduler='threads', num_workers=8)

        for each_ds in results[0]:
            cube_v, is_empty = each_ds

            if cube_v is not None:
                    # There can be multiple layers for the mid_date, collect all
                    self.velocities.setdefault(cube_v.coords['mid_date'].values, []).append(cube_v)

            else:
                if is_empty:
                     skipped_empty_granules.append(each_url)

                else:
                    skipped_proj_granules.setdefault(int(ds.UTM_Projection.spatial_epsg), []).append(each_url)

        self.combine_layers()
        ITSCube.format_stats(skipped_proj_granules, skipped_empty_granules, len(found_urls))

        return found_urls, skipped_proj_granules


    def read_dataset(self, url):
        """
        Read Dataset from the file and pre-process for the cube layer.
        """
        with xr.open_dataset(url) as ds:
            return self.preprocess_dataset(ds, url)


    def read_s3_dataset(self, each_url, s3):
        """
        Read Dataset from the S3 bucket and pre-process for the cube layer.
        """
        s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
        s3_path = s3_path.replace(ITSCube.PATH_URL, '')

        with s3.open(s3_path, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                return self.preprocess_dataset(ds, each_url)


    def plot_layers(self):
        """
        Plot cube's velocities in date order. Each layer has its own x/y coordinate labels based on data values
        present in the layer. This method provides a better insight into data variation within each layer.
        """
        num_granules = sum([len(each) for each in self.velocities.values()])

        num_cols = 5
        num_rows = int(num_granules / num_cols)
        print(f"rows={num_rows} cols={num_cols}")

        if (num_granules % num_cols) != 0:
            num_rows += 1

        fig, axes = plt.subplots(ncols=num_cols, nrows=num_rows, figsize=(num_cols*4, num_rows*4))
        col_index = 0
        row_index = 0
        for each_index, each_date in enumerate(sorted(self.velocities.keys())):
            if col_index == num_cols:
                col_index = 0
                row_index += 1

            for each_layer in self.velocities[each_date]:
                each_layer.plot(ax=axes[row_index, col_index])
                axes[row_index, col_index].title.set_text(str(each_date))

                col_index += 1
                if col_index == num_cols:
                    col_index = 0
                    row_index += 1

        plt.tight_layout()
        plt.draw()


    def plot_num_layers(self, num):
        """
        Plot specified number of first cube's velocities in date order. Each layer has its own x/y coordinate labels based on data values
        present in the layer. This method provides a better insight into data variation within each layer.
        """
        num_granules = sum([len(each) for each in self.velocities.values()])
#         num_granules = len(self.velocities)

        fig, axes = plt.subplots(ncols=num, figsize=(num*4, 4))
        col_index = 0
        num_index = 0
        for each_date in sorted(self.velocities.keys()):
            if num_index == num:
                break

            for each_v in self.velocities[each_date]:
                each_v.plot(ax=axes[num_index])
                axes[num_index].title.set_text(str(each_date))
                num_index += 1

        plt.tight_layout()
        plt.draw()



    def plot(self):
        """
        Plot cube's layers in date order. All layers share the same x/y coordinate labels.
        """
        # Does not work now: can't use non-unique faceted values for plotting (mid_date)
        # self.layers.plot(x='x', y = 'y', col='mid_date', col_wrap=5, levels=100)
        
        print("Not supported anymore since xarray can't use non-unique faceted values (mid_date) for plotting ")
