import copy
import dask
from dask.distributed import Client, performance_report
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta
import glob
import os
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd
import s3fs
import timeit
from tqdm import tqdm
import xarray as xr

from itslive import itslive_ui
from grid import Bounds


class Coords:
    """
    Coordinates for the data cube.
    """
    MID_DATE = 'mid_date'
    X = 'x'
    Y = 'y'


class DataVars:
    """
    Data variables for the data cube.
    """
    V = 'v'
    VX = 'vx'
    VY = 'vy'
    URL = 'url'


class ITSCube:
    """
    Class to build ITS_LIVE cube: time series of velocity pairs within a
    polygon of interest.
    """
    # Number of threads for parallel processing
    NUM_THREADS = 4

    # Dask scheduler for parallel processing
    DASK_SCHEDULER = "processes"

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = '4326'

    S3_PREFIX = 's3://'
    HTTP_PREFIX = 'http://'

    # Token within granule's URL that needs to be removed to get file location within S3 bucket:
    # if URL is of the 'http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/velocity_image_pair/landsat/v00.0/32628/file.nc' format,
    # S3 bucket location of the file is 's3://its-live-data.jpl.nasa.gov/velocity_image_pair/landsat/v00.0/32628/file.nc'
    PATH_URL = ".s3.amazonaws.com"

    # Engine to read xarray data into from NetCDF file
    NC_ENGINE = 'h5netcdf'

    # Date format as it appears in granules filenames:
    # (LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc)
    DATE_FORMAT = "%Y%m%d"

    def __init__(self, polygon: tuple, projection: str):
        """
        Initialize object.

        polygon: tuple
            Polygon for the tile.
        projection: str
            Projection in which polygon is defined.
        """
        self.projection = projection

        # Set min/max x/y values to filter region by
        self.x = Bounds([each[0] for each in polygon])
        self.y = Bounds([each[1] for each in polygon])

        # Convert polygon from its target projection to longitude/latitude coordinates
        # which are used by granule search API
        self.polygon_coords = []

        for each in polygon:
            coords = itslive_ui.transform_coord(
                projection,
                ITSCube.LON_LAT_PROJECTION,
                each[0], each[1]
            )
            self.polygon_coords.extend(coords)

        print(f"Longitude/latitude coords for polygon: {self.polygon_coords}")

        # Lists to store filtered by region/start_date/end_date velocity pairs
        # and corresponding metadata (middle dates (+ date separation in days as milliseconds),
        # original granules URLs)
        self.v = []
        self.vx = []
        self.vy = []
        self.dates = []
        self.urls = []

        # Keep track of skipped granules due to the other than target projection
        self.skipped_proj_granules = {}
        # Keep track of skipped granules due to no data for the polygon of interest
        self.skipped_empty_granules = []

        # Constructed cube
        self.layers = None

    def clear(self):
        """
        Reset all internal data structures.
        """
        self.v = []
        self.vx = []
        self.vy = []
        self.dates = []
        self.urls = []
        self.skipped_proj_granules = {}
        self.skipped_empty_granules = []
        self.layers = None

    def request_granules(self, api_params: dict, num_granules: int):
        """
        Send request to ITS_LIVE API to get a list of granules to satisfy polygon request.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """

        # Append polygon information to API's parameters
        params = copy.deepcopy(api_params)
        params['polygon'] = ",".join([str(each) for each in self.polygon_coords])

        start_time = timeit.default_timer()
        found_urls = [each['url'] for each in itslive_ui.get_granule_urls(params)]
        total_num = len(found_urls)
        time_delta = timeit.default_timer() - start_time
        print(f"Number of found by API granules: {total_num} (took {time_delta} seconds)")

        if len(found_urls) == 0:
            raise RuntimeError(f"No granules are found for the search API parameters: {params}")

        # Number of granules to examine is specified
        # TODO: just a workaround for now as it's very slow to examine all granules
        #       sequentially at this point.
        if num_granules:
            found_urls = found_urls[:num_granules]
            print(f"Examining only first {len(found_urls)} out of {total_num} found granules")

        return found_urls

    def add_layer(self, is_empty, layer_projection, v_layer, vx_layer, vy_layer, mid_date, url):
        """
        Examine the layer if it qualifies to be added as a cube layer.
        """
        if v_layer is not None:
            # If there's a layer for the mid_date already, pick the one
            # with newer processing date
            if mid_date in self.dates:
                print(f"Found another granule {url} for existing {mid_date} in layers")
                index = self.dates.index(mid_date)
                found_url = self.urls[index]

                # Get acquisition and processing date for both images from url and index_url
                url_tokens = os.path.basename(url).split('_')
                url_acq_date_1 = datetime.strptime(url_tokens[3], ITSCube.DATE_FORMAT).date()
                url_proc_date_1 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT).date()
                url_acq_date_2 = datetime.strptime(url_tokens[11], ITSCube.DATE_FORMAT).date()
                url_proc_date_2 = datetime.strptime(url_tokens[12], ITSCube.DATE_FORMAT).date()

                found_url_tokens = os.path.basename(found_url).split('_')
                found_url_acq_date_1 = datetime.strptime(found_url_tokens[3],ITSCube.DATE_FORMAT).date()
                found_url_proc_date_1 = datetime.strptime(found_url_tokens[4], ITSCube.DATE_FORMAT).date()
                found_url_acq_date_2 = datetime.strptime(found_url_tokens[11], ITSCube.DATE_FORMAT).date()
                found_url_proc_date_2 = datetime.strptime(found_url_tokens[12],ITSCube.DATE_FORMAT).date()

                # It is allowed for the same image pair only
                if url_acq_date_1 != found_url_acq_date_1 or \
                    url_acq_date_2 != found_url_acq_date_2:
                    raise RuntimeError(
                        f"Found duplicate granule for {mid_date} that differs in image acquisition time: "
                        "{url_acq_date_1} != {found_url_acq_date_1} or "
                        "{url_acq_date_2} != {found_url_acq_date_2} "
                        "({url} vs. {found_url})"
                    )
                if url_proc_date_1 >= found_url_proc_date_1 and \
                   url_proc_date_2 >= found_url_proc_date_2:
                    # Replace the granule with newer processed one
                    print(f"Replacing data for {mid_date} layer: {found_url} by {url}")
                    self.v[index] = v_layer
                    self.vx[index] = vx_layer
                    self.vy[index] = vy_layer
                    self.urls[index] = url
                else:
                    print(f"Keeping data for {mid_date} layer: {found_url} instead of new granule {url}")

            else:
                # print(f"Adding {url} for {mid_date}")
                self.dates.append(mid_date)
                self.v.append(v_layer)
                self.vx.append(vx_layer)
                self.vy.append(vy_layer)
                self.urls.append(url)

        else:
            if is_empty:
                # Layer does not contain valid data for the region
                self.skipped_empty_granules.append(url)

            else:
                # Layer corresponds to other than target projection
                self.skipped_proj_granules.setdefault(layer_projection, []).append(url)

    def create(self, api_params: dict, num_granules=None):
        """
        Create velocity cube.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        self.clear()

        found_urls = self.request_granules(api_params, num_granules)

        # Open S3FS access to S3 bucket with input granules
        s3 = s3fs.S3FileSystem(anon=True)

        for each_url in tqdm(found_urls, ascii=True, desc='Reading and processing S3 granules'):
            s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
            s3_path = s3_path.replace(ITSCube.PATH_URL, '')

            with s3.open(s3_path, mode='rb') as fhandle:
                with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                    results = self.preprocess_dataset(ds, each_url)
                    self.add_layer(*results)

        self.combine_layers()

        # Report statistics for skipped granules
        self.format_stats(len(found_urls))

        return found_urls

    def create_parallel(self, api_params: dict, num_granules=None):
        """
        Create velocity cube by reading and pre-processing cube layers in parallel.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules. Should not be used
                  when running the code at AWS.
        """
        self.clear()

        found_urls = self.request_granules(api_params, num_granules)

        # Parallelize layer collection
        s3 = s3fs.S3FileSystem(anon=True)

        # In order to enable Dask profiling, need to create Dask client for
        # processing: using "processes" or "threads" scheduler
        # processes_scheduler = True if ITSCube.DASK_SCHEDULER == 'processes' else False
        # client = Client(processes=processes_scheduler, n_workers=ITSCube.NUM_THREADS)
        # # Use client to collect profile information
        # client.profile(filename=f"dask-profile-{num_granules}-parallel.html")
        tasks = [dask.delayed(self.read_s3_dataset)(each_file, s3) for each_file in found_urls]

        with ProgressBar():
            # Display Dask's progress bar

            # If to collect performance report (need to define global Client - see above)
            # with performance_report(filename=f"dask-report-{num_granules}.html"):
            #     results = dask.compute(tasks)
            results = dask.compute(
                tasks,
                scheduler=ITSCube.DASK_SCHEDULER,
                num_workers=ITSCube.NUM_THREADS
            )

        for each_ds in results[0]:
            self.add_layer(*each_ds)

        self.combine_layers()
        self.format_stats(len(found_urls))

        return found_urls

    def to_netcdf(self, filename: str):
        """
        Write datacube to the NetCDF file.
        """
        if self.layers is not None:
            self.layers.to_netcdf(filename, engine=ITSCube.NC_ENGINE, unlimited_dims=(Coords.MID_DATE))

        else:
            raise RuntimeError(f"No datacube data available to write to {filename} file.")

    def create_from_local(self, api_params: dict, num_granules=None, local_path=''):
        """
        Create velocity cube by quering its_live API, but using local copy of the
        granules which are downloaded apriori.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        local_path: str
            Directory where granules files are downloaded to.
        """
        self.clear()

        found_urls = self.request_granules(api_params, num_granules)

        for each_url in tqdm(found_urls, ascii=True, desc='Reading and processing S3 granules'):
            s3_file = os.path.basename(each_url)
            s3_path = os.path.join(local_path, s3_file)

            with xr.open_dataset(s3_path) as ds:
                results = self.preprocess_dataset(ds, each_url)
                self.add_layer(*results)

        self.combine_layers()
        self.format_stats(len(found_urls))

        return found_urls

    def create_from_local_parallel(self, api_params: dict, num_granules=None, dirpath='data'):
        """
        Create velocity cube from local data in parallel.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        self.clear()

        found_urls = self.request_granules(api_params, num_granules)

        tasks = [dask.delayed(self.read_dataset)(os.path.join(dirpath, os.path.basename(each_file))) for each_file in found_urls]

        with ProgressBar():
            # Display progress bar
            results = dask.compute(tasks, scheduler=ITSCube.DASK_SCHEDULER, num_workers=ITSCube.NUM_THREADS)

        for each_ds in results[0]:
            self.add_layer(*each_ds)

        self.combine_layers()
        self.format_stats(len(found_urls))

        return found_urls

    def create_from_local_no_api(self, dirpath='data'):
        """
        Create velocity cube by accessing local data stored in "dirpath" directory.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        self.clear()

        found_urls = glob.glob(dirpath + os.sep + '*.nc')

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        for each_url in tqdm(found_urls, ascii=True, desc='Processing local granules'):
            with xr.open_dataset(each_url) as ds:
                results = self.preprocess_dataset(ds, each_url)
                self.add_layer(*results)

        self.combine_layers()
        self.format_stats(len(found_urls))

        return found_urls

    def create_from_local_parallel_no_api(self, dirpath='data'):
        """
        Create velocity cube from local data stored in "dirpath" in parallel.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        self.clear()

        found_urls = glob.glob(dirpath + os.sep + '*.nc')

        tasks = [dask.delayed(self.read_dataset)(each_file) for each_file in found_urls]
        with ProgressBar():
            # Display progress bar
            results = dask.compute(tasks,
                                   scheduler=ITSCube.DASK_SCHEDULER,
                                   num_workers=ITSCube.NUM_THREADS)

        for each_ds in results[0]:
            self.add_layer(*each_ds)

        self.combine_layers()
        self.format_stats(len(found_urls))

        return found_urls

    def preprocess_dataset(self, ds: xr.Dataset, ds_url: str):
        """
        Pre-process ITS_LIVE dataset in preparation for the cube layer.

        ds: xarray dataset
            Dataset to pre-process.
        ds_url: str
            URL that corresponds to the dataset.

        Returns:
        cube_v:     Filtered data array for the layer.
        mid_date:   Middle date that corresponds to the velicity pair (uses date
                    separation as milliseconds)
        empty:      Flag to indicate if dataset does not contain any data for
                    the cube region.
        projection: Source projection for the dataset.
        url:        Original URL for the granule (have to return for parallel
                    processing: no track of inputs for each task, but have output
                    available for each task).
        """
        # Try to load the whole dataset into memory to avoid penalty for random read access
        # when accessing S3 bucket (?)
        # ds.load()

        # Flag if layer data is empty.
        empty = False

        # Layer velocity data
        cube_v = None
        cube_vx = None
        cube_vy = None

        # Layer middle date
        mid_date = None

        # Consider granules with data only within target projection
        if str(int(ds.UTM_Projection.spatial_epsg)) == self.projection:
            mid_date = datetime.strptime(ds.img_pair_info.date_center, '%Y%m%d')

            # Add date separation in days as milliseconds for the middle date
            # (avoid resolution issues for layers with the same middle date).
            mid_date += timedelta(milliseconds=int(ds.img_pair_info.date_dt))

            # Define which points are within target polygon.
            mask_lon = (ds.x >= self.x.min) & (ds.x <= self.x.max)
            mask_lat = (ds.y >= self.y.min) & (ds.y <= self.y.max)
            mask_data = ds.where(mask_lon & mask_lat, drop=True)
            cube_v = mask_data.v
            cube_vx = mask_data.vx
            cube_vy = mask_data.vy
            # Another way to filter:
            # cube_v = ds.v.sel(x=slice(self.x.min, self.x.max),y=slice(self.y.max, self.y.min)).copy()

            # If it's a valid velocity layer, add it to the cube.
            if np.any(cube_v.notnull()):
                # Uncomment if to use random access read of filtered data only
                cube_v.load()
                cube_vx.load()
                cube_vy.load()

            else:
                # Reset cube back to None as it does not contain any valid data
                cube_v = None
                cube_vx = None
                cube_vy = None
                mid_date = None
                empty = True

        # Have to return URL for the dataset, which is provided as an input to the method,
        # to track URL per granule in parallel processing
        return empty, int(ds.UTM_Projection.spatial_epsg), cube_v, cube_vx, cube_vy, mid_date, ds_url

    def combine_layers(self):
        """
        Combine selected layers into one xr.Dataset object.
        """
        self.layers = {}

        # Construct xarray to hold layers by concatenating layer objects along 'mid_date' dimension
        print('Combining layers by date...')
        start_time = timeit.default_timer()
        v_layers = xr.concat(self.v, pd.Index(self.dates, name=Coords.MID_DATE))
        vx_layers = xr.concat(self.vx, pd.Index(self.dates, name=Coords.MID_DATE))
        vy_layers = xr.concat(self.vy, pd.Index(self.dates, name=Coords.MID_DATE))

        time_delta = timeit.default_timer() - start_time
        print(f"Combined {len(self.dates)} layers by date (took {time_delta} seconds)")

        # TODO: Should keep attributes per each layer's array or create
        #       attributes at the top level?
        # TODO: Cludgy way of introducing x and y coordinates to the Dataset?
        #       Ask the group if there is another known way to do it.
        self.layers = xr.Dataset(
            data_vars = {
                DataVars.V: ([Coords.MID_DATE, Coords.Y, Coords.X], v_layers.data),
                DataVars.VX: ([Coords.MID_DATE, Coords.Y, Coords.X], vx_layers.data),
                DataVars.VY: ([Coords.MID_DATE, Coords.Y, Coords.X], vy_layers.data),
                DataVars.URL: ([Coords.MID_DATE], self.urls)
            },
            coords = {Coords.MID_DATE: self.dates,
                      Coords.X: v_layers.coords[Coords.X],
                      Coords.Y: v_layers.coords[Coords.Y]},
            attrs = {'title': 'ITS_LIVE datacube of velocity pairs',
                     'author': 'Alex S. Gardner, JPL/NASA',
                     'institution': 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology',
                     'GDAL_AREA_OR_POINT': 'Area',
                     'projection': str(self.projection)}
        )

        # TODO: Sort data by date?
        # self.layers = self.layers.sortby(Coords.MID_DATE)

        # No need for collected velocities pairs anymore (disable if need
        # to examine collected layers in Jupyter notebook)
        self.v = None
        self.vx = None
        self.vy = None

    def format_stats(self, num_urls: int):
        """
        Format statistics of the run.
        """
        # Total number of skipped granules due to wrong projection
        sum_projs = sum([len(each) for each in self.skipped_proj_granules.values()])

        print( "Skipped granules:")
        print(f"      empty data       : {len(self.skipped_empty_granules)} ({100.0 * len(self.skipped_empty_granules)/num_urls}%)")
        print(f"      wrong projection : {sum_projs} ({100.0 * sum_projs/num_urls}%)")
        if len(self.skipped_proj_granules):
            print(f"      wrong projections: {sorted(self.skipped_proj_granules.keys())}")

    def read_dataset(self, url: str):
        """
        Read Dataset from the file and pre-process for the cube layer.
        """
        with xr.open_dataset(url) as ds:
            return self.preprocess_dataset(ds, url)

    def read_s3_dataset(self, each_url: str, s3):
        """
        Read Dataset from the S3 bucket and pre-process for the cube layer.
        """
        s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
        s3_path = s3_path.replace(ITSCube.PATH_URL, '')

        with s3.open(s3_path, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                return self.preprocess_dataset(ds, each_url)

    def plot(self, variable, boundaries: tuple = None):
        """
        Plot cube's layers data. All layers share the same x/y coordinate labels.
        There is an option to display only a subset of layers by specifying
        start and end index through "boundaries" input parameter.
        """
        if boundaries is not None:
            start, end = boundaries
            self.layers[variable][start:end].plot(
                x=Coords.X,
                y=Coords.Y,
                col=Coords.MID_DATE,
                col_wrap=5,
                levels=100)

        else:
            self.layers[variable].plot(
                x=Coords.X,
                y=Coords.Y,
                col=Coords.MID_DATE,
                col_wrap=5,
                levels=100)


if __name__ == '__main__':
    # Since port forwarding is not working on EC2 to run jupyter lab for now,
    # allow to run test case from itscube.ipynb in standalone mode
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=ITSCube.__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='number of threads to use for parallel processing.')
    parser.add_argument('-s', '--scheduler', type=str, default="processes",
                        help="Dask scheduler to use. One of ['threads', 'processes'].")
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='enable parallel processing')
    parser.add_argument('-n', '--numberGranules', type=int, required=False, default=100,
                        help='number of ITS_LIVE granules to consider for the cube (due to runtime limitations)')
    parser.add_argument('-l', '--localPath', type=str, default=None,
                        help='Local path that stores ITS_LIVE granules')

    args = parser.parse_args()
    ITSCube.NUM_THREADS = args.threads
    ITSCube.DASK_SCHEDULER = args.scheduler

    # Test Case from itscube.ipynb:
    # =============================
    # Create polygon as a square around the centroid in target '32628' UTM projection
    # Projection for the polygon coordinates
    projection = '32628'

    # Centroid for the tile in target projection
    c_x, c_y = (487462, 9016243)

    # Offset in meters (1 pixel=240m): 100 km square (with offset=50km)
    off = 50000
    polygon = (
        (c_x - off, c_y + off),
        (c_x + off, c_y + off),
        (c_x + off, c_y - off),
        (c_x - off, c_y - off),
        (c_x - off, c_y + off))
    print("Polygon: ", polygon)

    # Create cube object
    cube = ITSCube(polygon, projection)

    # Parameters for the search granule API
    API_params = {
        'start'               : '2010-01-05',
        'end'                 : '2020-01-01',
        'percent_valid_pixels': 1
    }

    skipped_projs = {}
    if not args.parallel:
        # Process ITS_LIVE granules sequentially, look at provided number of granules only
        print("Processing granules sequentially...")
        if args.localPath:
            # Granules are downloaded locally
            cube.create_from_local(API_params, args.numberGranules, args.localPath)

        else:
            cube.create(API_params, args.numberGranules)

    else:
        # Process ITS_LIVE granules in parallel, look at 100 first granules only
        print("Processing granules in parallel...")
        if args.localPath:
            # Granules are downloaded locally
            cube.create_from_local_parallel(API_params, args.numberGranules, args.localPath)

        else:
            cube.create_parallel(API_params, args.numberGranules)

    # Write cube data to the NetCDF file
    cube.to_netcdf('test_v_cube.nc')
