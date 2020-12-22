import copy
import dask
from dask.distributed import Client, performance_report
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta
import gc
import glob
import os
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd
import s3fs
import shutil
import timeit
from tqdm import tqdm
import xarray as xr

from itslive import itslive_ui
from grid import Bounds, Grid


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
    # Original data variables per ITS_LIVE granules.
    V                = 'v'
    VX               = 'vx'
    VY               = 'vy'
    CHIP_SIZE_HEIGHT = 'chip_size_height'
    CHIP_SIZE_WIDTH  = 'chip_size_width'
    INTERP_MASK      = 'interp_mask'
    V_ERROR          = 'v_error'

    # Added for the datacube
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

    # Granules are written to the file in chunks to avoid out of memory issues.
    # Number of granules to write to the file at a time.
    NUM_GRANULES_TO_WRITE = 1000

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
        self.chip_size_height = []
        self.chip_size_width = []
        self.interp_mask = []
        self.v_error = []

        self.dates = []
        self.urls = []
        self.num_urls_from_api = None

        # Keep track of skipped granules due to the other than target projection
        self.skipped_proj_granules = {}
        # Keep track of skipped granules due to no data for the polygon of interest
        self.skipped_empty_granules = []
        # Keep track of "double" granules with older processing date which are
        # not included into the cube
        self.skipped_double_granules = []

        # Constructed cube
        self.layers = None

    def clear_vars(self):
        """
        Clear current set of cube layers.
        """
        self.v = None
        self.vx = None
        self.vy = None
        self.chip_size_height = None
        self.chip_size_width = None
        self.interp_mask = None
        self.v_error = None

        self.layers = None
        self.dates = []
        self.urls = []

        gc.collect()

        self.v = []
        self.vx = []
        self.vy = []
        self.chip_size_height = []
        self.chip_size_width = []
        self.interp_mask = []
        self.v_error = []

    def clear(self):
        """
        Reset all internal data structures.
        """
        self.clear_vars()

        self.num_urls_from_api = None
        self.skipped_proj_granules = {}
        self.skipped_empty_granules = []
        self.skipped_double_granules = []

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

        return self.skip_duplicate_granules(found_urls)

    def skip_duplicate_granules(self, found_urls):
        """
        Skip duplicate granules (the ones that have earlier processing date(s)).
        """
        self.num_urls_from_api = len(found_urls)

        # Need to remove duplicate granules for the middle date: some granules
        # have newer processing date, keep those.
        url_mid_dates = []
        keep_urls = []
        self.skipped_double_granules = []

        for each_url in tqdm(found_urls, ascii=True, desc='Skipping duplicate granules...'):
            # Extract acquisition and processing dates
            url_acq_1, url_proc_1, url_acq_2, url_proc_2 = \
                ITSCube.get_dates_from_filename(each_url)

            day_separation = (url_acq_1 - url_acq_2).days
            mid_date = url_acq_2 + timedelta(days=day_separation/2, milliseconds=day_separation)

            # There is a granule for the mid_date already, check which processing
            # time is newer, keep the one with newer processing date
            if mid_date in url_mid_dates:
                index = url_mid_dates.index(mid_date)
                found_url = keep_urls[index]

                found_acq_1, found_proc_1, found_acq_2, found_proc_2 = \
                    ITSCube.get_dates_from_filename(found_url)

                # It is allowed for the same image pair only
                if url_acq_1 != found_acq_1 or \
                    url_acq_2 != found_acq_2:
                    raise RuntimeError(f"Found duplicate granule for {mid_date} that differs in acquisition time: {url_acq_1} != {found_acq_1} or {url_acq_2} != {found_acq_2} ({each_url} vs. {found_url})")

                if url_proc_1 >= found_proc_1 and \
                   url_proc_2 >= found_proc_2:
                    # Replace the granule with newer processed one
                    keep_urls[index] = each_url
                    self.skipped_double_granules.append(found_url)

                else:
                    # New granule has older processing date, don't include
                    self.skipped_double_granules.append(each_url)

            else:
                # This is new mid_date, append information
                url_mid_dates.append(mid_date)
                keep_urls.append(each_url)

        print (f"Keeping {len(keep_urls)} unique granules")
        return keep_urls

    @staticmethod
    def get_dates_from_filename(filename):
        """
        Extract acquisition and processing dates for two images from the filename.
        """
        # Get acquisition and processing date for both images from url and index_url
        url_tokens = os.path.basename(filename).split('_')
        url_acq_date_1 = datetime.strptime(url_tokens[3], ITSCube.DATE_FORMAT)
        url_proc_date_1 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT)
        url_acq_date_2 = datetime.strptime(url_tokens[11], ITSCube.DATE_FORMAT)
        url_proc_date_2 = datetime.strptime(url_tokens[12], ITSCube.DATE_FORMAT)

        return url_acq_date_1, url_proc_date_1, url_acq_date_2, url_proc_date_2

    def add_layer(self, is_empty, layer_projection, mid_date, url, data):
        """
        Examine the layer if it qualifies to be added as a cube layer.
        """
        v, vx, vy, chip_size_height, chip_size_width, interp_mask, v_error = data
        if v is not None:
            # TODO: Handle "duplicate" granules for the mid_date if concatenating
            #       to existing cube.
            #       "Duplicate" granules are handled apriori for newly constructed
            #       cubes (see self.request_granules() method).
            # print(f"Adding {url} for {mid_date}")
            self.dates.append(mid_date)
            self.v.append(v)
            self.vx.append(vx)
            self.vy.append(vy)
            self.chip_size_height.append(chip_size_height)
            self.chip_size_width.append(chip_size_width)
            self.interp_mask.append(interp_mask)
            self.v_error.append(v_error)

            self.urls.append(url)

        else:
            if is_empty:
                # Layer does not contain valid data for the region
                self.skipped_empty_granules.append(url)

            else:
                # Layer corresponds to other than target projection
                self.skipped_proj_granules.setdefault(layer_projection, []).append(url)

    def create(self, api_params: dict, output_dir: str, num_granules=None):
        """
        Create velocity cube.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.clear()

        found_urls = self.request_granules(api_params, num_granules)

        # Open S3FS access to S3 bucket with input granules
        s3 = s3fs.S3FileSystem(anon=True)

        is_first_write = True
        for each_url in tqdm(found_urls, ascii=True, desc='Reading and processing S3 granules'):
            s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
            s3_path = s3_path.replace(ITSCube.PATH_URL, '')

            with s3.open(s3_path, mode='rb') as fhandle:
                with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                    results = self.preprocess_dataset(ds, each_url)
                    self.add_layer(*results)

                    # Check if need to write to the file accumulated number of granules
                    if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
                        self.combine_layers(output_dir, is_first_write)
                        is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(output_dir, is_first_write)

        # Report statistics for skipped granules
        self.format_stats()

        return found_urls

    def create_parallel(self, api_params: dict, output_dir: str, num_granules=None):
        """
        Create velocity cube by reading and pre-processing cube layers in parallel.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules. Should not be used
                  when running the code at AWS.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

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
        is_first_write = True
        start = 0
        num_to_process = len(found_urls)

        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            tasks = [dask.delayed(self.read_s3_dataset)(each_file, s3) for each_file in found_urls[start:start+num_tasks]]
            print("Processing NUM tasks: ", len(tasks))

            results = None
            with ProgressBar():
                # If to collect performance report (need to define global Client - see above)
                # with performance_report(filename=f"dask-report-{num_granules}.html"):
                #     results = dask.compute(tasks)
                results = dask.compute(
                    tasks,
                    scheduler=ITSCube.DASK_SCHEDULER,
                    num_workers=ITSCube.NUM_THREADS
                )

            del tasks
            gc.collect()

            for each_ds in results[0]:
                self.add_layer(*each_ds)

            del results
            gc.collect()

            self.combine_layers(output_dir, is_first_write)

            if start == 0:
                is_first_write = False

            num_to_process -= num_tasks
            start += num_tasks

        self.format_stats()
        return found_urls

    # def to_netcdf(self, filename: str):
    #     """
    #     Write datacube to the NetCDF file.
    #     """
    #     if self.layers is not None:
    #         self.layers.to_netcdf(filename, engine=ITSCube.NC_ENGINE, unlimited_dims=(Coords.MID_DATE))
    #
    #     else:
    #         raise RuntimeError(f"Datacube data does not exist.")

    # def create_from_local(self, api_params: dict, output_dir: str, num_granules=None, local_path=''):
    #     """
    #     Create velocity cube by quering its_live API, but using local copy of the
    #     granules which are downloaded apriori.
    #
    #     api_params: dict
    #         Search API required parameters.
    #     num_granules: int
    #         Number of first granules to examine.
    #         TODO: This is a temporary solution to a very long time to open remote granules.
    #               Should not be used when running the code in production mode.
    #     local_path: str
    #         Directory where granules files are downloaded to.
    #     """
    #     if os.path.exists(output_dir):
    #         shutil.rmtree(output_dir)
    #
    #     self.clear()
    #     found_urls = self.request_granules(api_params, num_granules)
    #
    #     is_first_write = True
    #     for each_url in tqdm(found_urls, ascii=True, desc='Reading and processing S3 granules'):
    #         s3_file = os.path.basename(each_url)
    #         s3_path = os.path.join(local_path, s3_file)
    #
    #         with xr.open_dataset(s3_path) as ds:
    #             results = self.preprocess_dataset(ds, each_url)
    #             self.add_layer(*results)
    #
    #             # Check if need to write to the file accumulated number of granules
    #             if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
    #                 self.combine_layers(output_dir, is_first_write)
    #                 is_first_write = False
    #
    #     # Check if there are remaining layers to be written to the file
    #     if len(self.urls):
    #         self.combine_layers(output_dir, is_first_write)
    #
    #     self.format_stats()
    #
    #     return found_urls
    #
    # def create_from_local_parallel(self, api_params: dict, output_dir: str, num_granules=None, dirpath='data'):
    #     """
    #     Create velocity cube from local data in parallel.
    #
    #     api_params: dict
    #         Search API required parameters.
    #     num_granules: int
    #         Number of first granules to examine.
    #         TODO: This is a temporary solution to a very long time to open remote granules.
    #               Should not be used when running the code in production mode.
    #     dirpath: str
    #         Directory that stores granules files. Default is 'data' sub-directory
    #         accessible from the directory the code is running from.
    #     """
    #     self.clear()
    #     found_urls = self.request_granules(api_params, num_granules)
    #     tasks = [dask.delayed(self.read_dataset)(os.path.join(dirpath, os.path.basename(each_file))) for each_file in found_urls]
    #
    #     results = None
    #     with ProgressBar():
    #         # Display progress bar
    #         results = dask.compute(tasks, scheduler=ITSCube.DASK_SCHEDULER, num_workers=ITSCube.NUM_THREADS)
    #
    #     for each_ds in results[0]:
    #         self.add_layer(*each_ds)
    #
    #     self.combine_layers()
    #     self.format_stats()
    #
    #     return found_urls

    def create_from_local_no_api(self, output_dir: str, dirpath='data'):
        """
        Create velocity cube by accessing local data stored in "dirpath" directory.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.clear()
        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        found_urls = self.skip_duplicate_granules(found_urls)
        is_first_write = True

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        for each_url in tqdm(found_urls, ascii=True, desc='Processing local granules'):
            with xr.open_dataset(each_url) as ds:
                results = self.preprocess_dataset(ds, each_url)
                self.add_layer(*results)
                # Check if need to write to the file accumulated number of granules
                if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
                    self.combine_layers(output_dir, is_first_write)
                    is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(output_dir, is_first_write)

        self.format_stats()

        return found_urls

    def create_from_local_parallel_no_api(self, output_dir: str, dirpath='data'):
        """
        Create velocity cube from local data stored in "dirpath" in parallel.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.clear()
        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        found_urls = self.skip_duplicate_granules(found_urls)

        num_to_process = len(found_urls)

        is_first_write = True
        start = 0
        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            print("NUM to process: ", num_tasks)

            tasks = [dask.delayed(self.read_dataset)(each_file) for each_file in found_urls[start:start+num_tasks]]
            assert len(tasks) == num_tasks
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler=ITSCube.DASK_SCHEDULER,
                                       num_workers=ITSCube.NUM_THREADS)

            for each_ds in results[0]:
                self.add_layer(*each_ds)

            self.combine_layers(output_dir, is_first_write)

            if start == 0:
                is_first_write = False

            num_to_process -= num_tasks
            start += num_tasks

        self.format_stats()

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
        cube_chip_size_height = None
        cube_chip_size_width = None
        cube_interp_mask = None
        cube_v_error = None

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

            # Another way to filter:
            # cube_v = ds.v.sel(x=slice(self.x.min, self.x.max),y=slice(self.y.max, self.y.min)).copy()

            # Get data variables for the polygon
            cube_v = mask_data.v
            cube_vx = mask_data.vx
            cube_vy = mask_data.vy
            cube_chip_size_height = mask_data.chip_size_height
            cube_chip_size_width = mask_data.chip_size_width
            cube_interp_mask = mask_data.interp_mask
            if DataVars.V_ERROR in mask_data:
                cube_v_error = mask_data.v_error

            # If it's a valid velocity layer, add it to the cube.
            if np.any(cube_v.notnull()):
                # Uncomment if to use random access read of filtered data only
                cube_v.load()
                cube_vx.load()
                cube_vy.load()
                cube_chip_size_height.load()
                cube_chip_size_width.load()
                cube_interp_mask.load()
                if cube_v_error is not None:
                    cube_v_error.load()

                else:
                    # Create empty array as it is not provided in the granule,
                    # use the same coordinates as for any cube's data variables.
                    cube_v_error = xr.DataArray(
                        data=None,
                        coords=[cube_v.coords[Coords.Y], cube_v.coords[Coords.X]],
                        dims=[Coords.Y, Coords.X])

            else:
                # Reset cube back to None as it does not contain any valid data
                cube_v = None
                cube_vx = None
                cube_vy = None
                cube_chip_size_height = None
                cube_chip_size_width = None
                cube_interp_mask = None
                cube_v_error = None

                mid_date = None
                empty = True

        # Have to return URL for the dataset, which is provided as an input to the method,
        # to track URL per granule in parallel processing
        return empty, int(ds.UTM_Projection.spatial_epsg), mid_date, ds_url, \
            (cube_v, cube_vx, cube_vy, cube_chip_size_height, cube_chip_size_width, \
             cube_interp_mask, cube_v_error)

    def combine_layers(self, output_dir, is_first_write=False):
        """
        Combine selected layers into one xr.Dataset object and write (append) it
        to the Zarr store.
        """
        self.layers = {}

        # Construct xarray to hold layers by concatenating layer objects along 'mid_date' dimension
        print(f'Combine {len(self.urls)} layers to the {output_dir}...')
        start_time = timeit.default_timer()
        mid_date_coord = pd.Index(self.dates, name=Coords.MID_DATE)

        v_layers = xr.concat(self.v, mid_date_coord)

        # TODO: Should keep attributes per each layer's array or create
        #       attributes at the top level?
        self.layers = xr.Dataset(
            data_vars = {DataVars.URL: ([Coords.MID_DATE], self.urls)},
            coords = {
                Coords.MID_DATE: self.dates,
                Coords.X: v_layers.coords[Coords.X],
                Coords.Y: v_layers.coords[Coords.Y]
            },
            attrs = {
                'title': 'ITS_LIVE datacube of velocity pairs',
                'author': 'Alex S. Gardner, JPL/NASA',
                'institution': 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology',
                'GDAL_AREA_OR_POINT': 'Area',
                'projection': str(self.projection)
            }
        )

        # Assign one data variable at a time to avoid running out of memory
        self.layers[DataVars.V] = v_layers
        del v_layers
        self.v = None
        gc.collect()

        # vx_layers = xr.concat(self.vx, mid_date_coord)
        self.layers[DataVars.VX] = xr.concat(self.vx, mid_date_coord)
        self.vx = None
        gc.collect()

        # vy_layers = xr.concat(self.vy, mid_date_coord)
        self.layers[DataVars.VY] = xr.concat(self.vy, mid_date_coord)
        self.vy = None
        gc.collect()

        self.layers[DataVars.CHIP_SIZE_HEIGHT] = xr.concat(self.chip_size_height, mid_date_coord)
        self.chip_size_height = None
        gc.collect()

        self.layers[DataVars.CHIP_SIZE_WIDTH] = xr.concat(self.chip_size_width, mid_date_coord)
        self.chip_size_width = None
        gc.collect()

        self.layers[DataVars.INTERP_MASK] = xr.concat(self.interp_mask, mid_date_coord)
        self.interp_mask = None
        gc.collect()

        self.layers[DataVars.V_ERROR] = xr.concat(self.v_error, mid_date_coord)
        self.v_error = None
        gc.collect()

        time_delta = timeit.default_timer() - start_time
        print(f"Combined {len(self.urls)} layers (took {time_delta} seconds)")

        start_time = timeit.default_timer()
        # Write to the Zarr store
        if is_first_write:
            # This is first write, create Zarr store
            self.layers.to_zarr(output_dir)

        else:
            # Append layers to existing Zarr store
            self.layers.to_zarr(output_dir, append_dim=Coords.MID_DATE)

        time_delta = timeit.default_timer() - start_time
        print(f"Wrote {len(self.urls)} layers to {output_dir} (took {time_delta} seconds)")

        # Free up memory
        self.clear_vars()

        # TODO: Sort data by date?
        # self.layers = self.layers.sortby(Coords.MID_DATE)

    def format_stats(self):
        """
        Format statistics of the run.
        """
        num_urls = self.num_urls_from_api
        # Total number of skipped granules due to wrong projection
        sum_projs = sum([len(each) for each in self.skipped_proj_granules.values()])

        print( "Skipped granules:")
        print(f"      empty data       : {len(self.skipped_empty_granules)} ({100.0 * len(self.skipped_empty_granules)/num_urls}%)")
        print(f"      wrong projection : {sum_projs} ({100.0 * sum_projs/num_urls}%)")
        print(f"      double mid_date  : {len(self.skipped_double_granules)} ({100.0 * len(self.skipped_double_granules)/num_urls}%)")
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

    @staticmethod
    def plot(cube, variable, boundaries: tuple = None):
        """
        Plot cube's layers data. All layers share the same x/y coordinate labels.
        There is an option to display only a subset of layers by specifying
        start and end index through "boundaries" input parameter.
        """
        if boundaries is not None:
            start, end = boundaries
            cube[variable][start:end].plot(
                x=Coords.X,
                y=Coords.Y,
                col=Coords.MID_DATE,
                col_wrap=5,
                levels=100)

        else:
            cube[variable].plot(
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
                        help="Dask scheduler to use. One of ['threads', 'processes'] (effective only when -p option is specified).")
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='enable parallel processing')
    parser.add_argument('-n', '--numberGranules', type=int, required=False, default=None,
                        help="number of ITS_LIVE granules to consider for the cube (due to runtime limitations). "
                             " If none is provided, process all found granules.")
    parser.add_argument('-l', '--localPath', type=str, default=None,
                        help='Local path that stores ITS_LIVE granules')
    parser.add_argument('-o', '--outputDir', type=str, default="cubedata.zarr",
                        help="Zarr output directory to write cube data to. Default is 'cubedata.zarr'.")
    parser.add_argument('-c', '--chunks', type=int, default=1000,
                        help="Number of granules to write at a time. Default is 1000.")

    args = parser.parse_args()
    ITSCube.NUM_THREADS = args.threads
    ITSCube.DASK_SCHEDULER = args.scheduler
    ITSCube.NUM_GRANULES_TO_WRITE = args.chunks

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
            cube.create_from_local(API_params, args.outputDir, args.numberGranules, args.localPath)

        else:
            cube.create(API_params, args.outputDir, args.numberGranules)

    else:
        # Process ITS_LIVE granules in parallel, look at 100 first granules only
        print("Processing granules in parallel...")
        if args.localPath:
            # Granules are downloaded locally
            cube.create_from_local_parallel(API_params, args.outputDir, args.numberGranules, args.localPath)

        else:
            cube.create_parallel(API_params, args.outputDir, args.numberGranules)

    # Write cube data to the NetCDF file
    # cube.to_netcdf('test_v_cube.nc')
