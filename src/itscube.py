"""
ITSCube class creates ITS_LIVE datacube based on target projection,
bounding polygon and date period provided by the caller.
"""
import copy
from datetime import datetime, timedelta
import gc
import glob
import logging
import os
import shutil
import timeit

# Extra installed packages
import dask
# from dask.distributed import Client, performance_report
from dask.diagnostics import ProgressBar
import numpy  as np
import pandas as pd
import s3fs
from tqdm import tqdm
import xarray as xr

# Local modules
from itslive import itslive_ui
from grid import Bounds, Grid
from itscube_types import Coords, DataVars


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)


class ITSCube:
    """
    Class to build ITS_LIVE cube: time series of velocity pairs within a
    polygon of interest.
    """
    # Current ITSCube software version
    Version = '1.0'

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

    # Directory to write verbose information about skipped granules
    GRANULE_REPORT_DIR = 'logs'

    # Grid cell size for the datacube.
    CELL_SIZE = 240.0

    def __init__(self, polygon: tuple, projection: str):
        """
        Initialize object.

        polygon: tuple
            Polygon for the tile.
        projection: str
            Projection in which polygon is defined.
        """
        self.logger = logging.getLogger("datacube")
        self.logger.info(f"Polygon: {polygon}")
        self.logger.info(f"Projection: {projection}")

        self.projection = projection

        # Set min/max x/y values to filter region by
        self.x = Bounds([each[0] for each in polygon])
        self.y = Bounds([each[1] for each in polygon])

        # Grid for the datacube based on its bounding polygon
        self.grid_x, self.grid_y = Grid.create(self.x, self.y, ITSCube.CELL_SIZE)

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

        self.logger.info(f"Polygon's longitude/latitude coordinates: {self.polygon_coords}")

        # Lists to store filtered by region/start_date/end_date velocity pairs
        # and corresponding metadata (middle dates (+ date separation in days as milliseconds),
        # original granules URLs)
        self.ds = []

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

        # Create log directory
        os.makedirs(ITSCube.GRANULE_REPORT_DIR, exist_ok=True)

    def clear_vars(self):
        """
        Clear current set of cube layers.
        """
        self.ds = None

        self.layers = None
        self.dates = []
        self.urls = []

        gc.collect()

        self.ds = []

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

        self.logger.info(f"ITS_LIVE search API params: {params}")
        start_time = timeit.default_timer()
        found_urls = [each['url'] for each in itslive_ui.get_granule_urls(params)]
        total_num = len(found_urls)
        time_delta = timeit.default_timer() - start_time
        self.logger.info(f"Number of found by API granules: {total_num} (took {time_delta} seconds)")

        if len(found_urls) == 0:
            raise RuntimeError(f"No granules are found for the search API parameters: {params}")

        # Number of granules to examine is specified
        # TODO: just a workaround for now as it's very slow to examine all granules
        #       sequentially at this point.
        if num_granules:
            found_urls = found_urls[:num_granules]
            self.logger.info(f"Examining only first {len(found_urls)} out of {total_num} found granules")

        return self.skip_duplicate_granules(found_urls)

    def skip_duplicate_granules(self, found_urls):
        """
        Skip duplicate granules (the ones that have earlier processing date(s)).
        """
        self.num_urls_from_api = len(found_urls)

        # Need to remove duplicate granules for the middle date: some granules
        # have newer processing date, keep those.
        keep_urls = {}
        self.skipped_double_granules = []

        for each_url in tqdm(found_urls, ascii=True, desc='Skipping duplicate granules...'):
            # Extract acquisition and processing dates
            url_acq_1, url_proc_1, path_row_1, url_acq_2, url_proc_2, path_row_2 = \
                ITSCube.get_tokens_from_filename(each_url)

            # Acquisition time and path/row of both images should be identical
            granule_id = '_'.join([
                url_acq_1.strftime(ITSCube.DATE_FORMAT),
                path_row_1,
                url_acq_2.strftime(ITSCube.DATE_FORMAT),
                path_row_2
            ])

            # There is a granule for the mid_date already, check which processing
            # time is newer, keep the one with newer processing date
            if granule_id in keep_urls:
                # Flag if newly found URL should be kept
                keep_found_url = False

                for found_url in keep_urls[granule_id]:
                    # Check already found URLs for processing time
                    _, found_proc_1, _, _, found_proc_2, _ = \
                        ITSCube.get_tokens_from_filename(found_url)

                    # If both granules have identical processing time,
                    # keep them both - granules might be in different projections,
                    # any other than target projection will be handled later
                    if url_proc_1 == found_proc_1 and \
                       url_proc_2 == found_proc_2:
                        keep_urls[granule_id].append(each_url)
                        keep_found_url = True
                        break

                # There are no "identical" granules to the "each_url", check if
                # new granule has newer processing dates
                if not keep_found_url:
                    # Check if any of the found URLs have older processing time
                    # than newly found URL
                    remove_urls = []
                    for found_url in keep_urls[granule_id]:
                        # Check already found URL for processing time
                        _, found_proc_1, _, _, found_proc_2, _ = \
                            ITSCube.get_tokens_from_filename(found_url)

                        if url_proc_1 >= found_proc_1 and \
                           url_proc_2 >= found_proc_2:
                            # The granule will need to be replaced with a newer
                            # processed one
                            remove_urls.append(found_url)

                        elif url_proc_1 > found_proc_1:
                            # There are few cases when proc_1 is newer in
                            # each_url and proc_2 is newer in found_url, then
                            # keep the granule with newer proc_1
                            remove_urls.append(found_url)

                    if len(remove_urls):
                        # Some of the URLs need to be removed due to newer
                        # processed granule
                        self.logger.info(f"Skipping {remove_urls} in favor of new {each_url}")
                        self.skipped_double_granules.extend(remove_urls)

                        # Remove older processed granules
                        keep_urls[granule_id][:] = [each for each in keep_urls[granule_id] if each not in remove_urls]
                        # Add new granule with newer processing date
                        keep_urls[granule_id].append(each_url)

                    else:
                        # New granule has older processing date, don't include
                        self.logger.info(f"Skipping new {each_url} in favor of {keep_urls[granule_id]}")
                        self.skipped_double_granules.append(each_url)

            else:
                # This is a granule for new ID, append it to URLs to keep
                keep_urls.setdefault(granule_id, []).append(each_url)

        granules = []
        for each in keep_urls.values():
            granules.extend(each)

        self.logger.info(f"Keeping {len(granules)} unique granules")

        return granules

    @staticmethod
    def get_tokens_from_filename(filename):
        """
        Extract acquisition/processing dates and path/row for two images from the filename.
        """
        # Get acquisition and processing date for both images from url and index_url
        url_tokens = os.path.basename(filename).split('_')
        url_acq_date_1 = datetime.strptime(url_tokens[3], ITSCube.DATE_FORMAT)
        url_proc_date_1 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT)
        url_path_row_1 = url_tokens[2]
        url_acq_date_2 = datetime.strptime(url_tokens[11], ITSCube.DATE_FORMAT)
        url_proc_date_2 = datetime.strptime(url_tokens[12], ITSCube.DATE_FORMAT)
        url_path_row_2 = url_tokens[10]

        return url_acq_date_1, url_proc_date_1, url_path_row_1, url_acq_date_2, url_proc_date_2, url_path_row_2

    def add_layer(self, is_empty, layer_projection, mid_date, url, data):
        """
        Examine the layer if it qualifies to be added as a cube layer.
        """

        if data is not None:
            # TODO: Handle "duplicate" granules for the mid_date if concatenating
            #       to existing cube.
            #       "Duplicate" granules are handled apriori for newly constructed
            #       cubes (see self.request_granules() method).
            # print(f"Adding {url} for {mid_date}")
            self.dates.append(mid_date)
            self.ds.append(data)
            self.urls.append(url)

        else:
            if is_empty:
                # Layer does not contain valid data for the region
                self.skipped_empty_granules.append(url)

            else:
                # Layer corresponds to other than target projection
                self.skipped_proj_granules.setdefault(layer_projection, []).append(url)

    @staticmethod
    def init_output_store(output_dir: str):
        """
        Initialize output store to the datacube. The method detects if S3 bucket
        store or local Zarr archive is requested. It removes existing local
        store if it exists already, and opens file-like access to the Zarr store
        in the S3 bucket.
        """
        cube_store = output_dir
        s3_out = None
        if ITSCube.S3_PREFIX not in output_dir:
            # If writing to the local directory, remove datacube store if it exists
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

        else:
            # When writing to the AWS S3 bucket, assume it's a new datacube.
            # Open S3FS access to S3 bucket with output granules
            s3_out = s3fs.S3FileSystem()
            cube_store = s3fs.S3Map(root=output_dir, s3=s3_out, check=False)
            # cube_store = s3_out.open(output_dir, mode='wb')

        # Don't use s3_out, keep it in scope only to guarantee valid file system
        # like access.
        return s3_out, cube_store

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
        s3_out, cube_store = ITSCube.init_output_store(output_dir)

        self.clear()

        found_urls = self.request_granules(api_params, num_granules)

        # Open S3FS access to public S3 bucket with input granules
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
                        self.combine_layers(cube_store, is_first_write)
                        is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(cube_store, is_first_write)

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
        # s3_out, cube_store = ITSCube.init_output_store(output_dir)

        cube_store = output_dir
        s3_out = None
        if ITSCube.S3_PREFIX not in output_dir:
            # If writing to the local directory, remove datacube store if it exists
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

        else:
            # When writing to the AWS S3 bucket, assume it's a new datacube.
            # Open S3FS access to S3 bucket with output granules
            s3_out = s3fs.S3FileSystem()
            cube_store = s3fs.S3Map(root=output_dir, s3=s3_out, check=False)
            # cube_store = s3_out.open(output_dir, mode='wb')

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
            self.logger.info(f"Processing {len(tasks)} tasks")

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

            self.combine_layers(cube_store, is_first_write)
            self.format_stats()

            if start == 0:
                is_first_write = False

            num_to_process -= num_tasks
            start += num_tasks

        return found_urls

    @staticmethod
    def ds_to_netcdf(ds: xr.Dataset, filename: str):
        """
        Write datacube xarray.Dataset to the NetCDF file.
        """
        if ds is not None:
            ds.to_netcdf(filename, engine=ITSCube.NC_ENGINE, unlimited_dims=(Coords.MID_DATE))

        else:
            raise RuntimeError(f"Datacube data does not exist.")

    def create_from_local_no_api(self, output_dir: str, dirpath='data', num_granules=None):
        """
        Create velocity cube by accessing local data stored in "dirpath" directory.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        s3_out, cube_store = ITSCube.init_output_store(output_dir)

        self.clear()

        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        if num_granules is not None:
            found_urls = found_urls[0: num_granules]

        self.num_urls_from_api = len(found_urls)
        found_urls = self.skip_duplicate_granules(found_urls)
        is_first_write = True

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        for each_url in tqdm(found_urls, ascii=True, desc='Processing local granules'):
            with xr.open_dataset(each_url) as ds:
                results = self.preprocess_dataset(ds, each_url)
                self.add_layer(*results)

                # Check if need to write to the file accumulated number of granules
                if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
                    self.combine_layers(cube_store, is_first_write)
                    is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(cube_store, is_first_write)

        self.format_stats()

        return found_urls

    def create_from_local_parallel_no_api(self, output_dir: str, dirpath='data', num_granules=None):
        """
        Create velocity cube from local data stored in "dirpath" in parallel.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        s3_out, cube_store = ITSCube.init_output_store(output_dir)

        self.clear()
        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        if num_granules is not None:
            found_urls = found_urls[0: num_granules]

        found_urls = self.skip_duplicate_granules(found_urls)
        self.num_urls_from_api = len(found_urls)

        num_to_process = len(found_urls)

        is_first_write = True
        start = 0
        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            self.logger.info(f"Number of granules to process: {num_tasks}")

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

            self.combine_layers(cube_store, is_first_write)

            if start == 0:
                is_first_write = False

            num_to_process -= num_tasks
            start += num_tasks

        self.format_stats()

        return found_urls

    def get_data_var(self, ds: xr.Dataset, var_name: str):
        """
        Return xr.DataArray that corresponds to the data variable if it exists
        in the 'ds' dataset, or empty xr.DataArray if it is not present in the 'ds'.
        Empty xr.DataArray assumes the same dimensions as ds.v data array.
        """

        if var_name in ds:
            return ds[var_name]

        # Create empty array as it is not provided in the granule,
        # use the same coordinates as for any cube's data variables.
        # ATTN: Can't use None as data to create xr.DataArray - won't be able
        # to set dtype='short' in encoding for writing to the file.
        data_values = np.empty((len(self.grid_y), len(self.grid_x)))
        data_values[:, :] = np.nan

        return xr.DataArray(
            data=data_values,
            coords=[self.grid_y, self.grid_x],
            dims=[Coords.Y, Coords.X]
        )

    @staticmethod
    def get_data_var_attr(
        ds: xr.Dataset,
        ds_url: str,
        var_name: str,
        attr_name: str,
        missing_value: int = None,
        to_date=False):
        """
        Return a list of attributes for the data variable in data set if it exists,
        or missing_value if it is not present.
        If missing_value is set to None, than specified attribute is expected
        to exist for the data variable "var_name" and exception is raised if
        it does not.
        """
        if var_name in ds and attr_name in ds[var_name].attrs:
            value = ds[var_name].attrs[attr_name]
            # print(f"Read value for {var_name}.{attr_name}: {value}")

            # Check if type has "length"
            if hasattr(type(value), '__len__') and len(value) == 1:
                value = value[0]

            if to_date is True:
                try:
                    tokens = value.split('T')
                    if len(tokens) == 3:
                        # Handle malformed datetime in Sentinel 2 granules:
                        # img_pair_info.acquisition_date_img1 = "20190215T205541T00:00:00"
                        value = tokens[0] + 'T' + tokens[1][0:2] + ':' + tokens[1][2:4]+ ':' + tokens[1][4:6]
                        value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')

                    elif len(value) == 8:
                        # Only date is provided
                        value = datetime.strptime(value[0:8], '%Y%m%d')

                    elif len(value) > 8:
                        # Extract date and time (20200617T00:00:00)
                        value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')

                except ValueError as exc:
                    raise RuntimeError(f"Error converting {value} to date format '%Y%m%d': {exc} for {var_name}.{attr_name} in {ds_url}")

            # print(f"Return value for {var_name}.{attr_name}: {value}")
            return value

        if missing_value is None:
            # If missing_value is not provided, attribute is expected to exist always
            raise RuntimeError(f"{attr_name} is expected within {var_name} for {ds_url}")

        return missing_value

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
        # Tried to load the whole dataset into memory to avoid penalty for random read access
        # when accessing S3 bucket (?) - does not make any difference.
        # ds.load()

        # Flag if layer data is empty
        empty = False

        # Layer data
        mask_data = None

        # Layer middle date
        mid_date = None

        # Detect projection
        ds_projection = None
        if DataVars.UTM_PROJECTION in ds:
            ds_projection = ds.UTM_Projection.spatial_epsg

        elif DataVars.POLAR_STEREOGRAPHIC in ds:
            ds_projection = ds.Polar_Stereographic.spatial_epsg

        else:
            # Unknown type of granule is provided
            raise RuntimeError("Unsupported projection is detected for {ds_url}. One of [{ITSCube.UTM_PROJECTION}, {ITSCube.POLAR_STEREOGRAPHIC}] is supported.")

        # Consider granules with data only within target projection
        if str(int(ds_projection)) == self.projection:
            mid_date = datetime.strptime(ds.img_pair_info.date_center, '%Y%m%d')

            # Add date separation in days as milliseconds for the middle date
            # (avoid resolution issues for layers with the same middle date).
            mid_date += timedelta(milliseconds=int(ds.img_pair_info.date_dt))

            # Define which points are within target polygon.
            mask_lon = (ds.x >= self.x.min) & (ds.x <= self.x.max)
            mask_lat = (ds.y >= self.y.min) & (ds.y <= self.y.max)
            mask = (mask_lon & mask_lat)
            if mask.values.sum() == 0:
                # One or both masks resulted in no coverage
                mask_data = None
                mid_date = None
                empty = True

            else:
                mask_data = ds.where(mask_lon & mask_lat, drop=True)

                # Another way to filter (have to put min/max values in the order
                # corresponding to the grid)
                # cube_v = ds.v.sel(x=slice(self.x.min, self.x.max),y=slice(self.y.max, self.y.min)).copy()

                # If it's a valid velocity layer, add it to the cube.
                if np.any(mask_data.v.notnull()):
                    mask_data.load()

                else:
                    # Reset cube back to None as it does not contain any valid data
                    mask_data = None
                    mid_date = None
                    empty = True

        # Have to return URL for the dataset, which is provided as an input to the method,
        # to track URL per granule in parallel processing
        return empty, int(ds_projection), mid_date, ds_url, mask_data

    def process_v_attributes(self, var_name: str, mid_date_coord):
        """
        Helper method to clean up attributes for v-related data variables.
        """
        _stable_rmse_vars = [DataVars.VX, DataVars.VY]

        # Dictionary of attributes values for new v*_error data variables:
        # std_name, description
        _attrs = {
            'vx_error': ("x_velocity_error", "error for velocity component in x direction"),
            'vy_error': ("y_velocity_error", "error for velocity component in y direction"),
            'va_error': ("azimuth_velocity_error", "error for velocity in radar azimuth direction"),
            'vr_error': ("range_velocity_error", "error for velocity in radar range direction"),
            'vxp_error': ("projected_x_velocity_error", "error for x-direction velocity determined by projecting radar range measurements onto an a priori flow vector"),
            'vyp_error': ("projected_y_velocity_error", "error for y-direction velocity determined by projecting radar range measurements onto an a priori flow vector")
        }

        # Process attributes
        if DataVars.STABLE_APPLY_DATE in self.layers[var_name].attrs:
            # Remove optical legacy attribute if it propagated to the cube data
            del self.layers[var_name].attrs[DataVars.STABLE_APPLY_DATE]

        # If attribute is propagated as cube's data var attribute, delete it.
        # These attributes were collected based on 'v' data variable
        if DataVars.MAP_SCALE_CORRECTED in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.MAP_SCALE_CORRECTED]
        if DataVars.STABLE_SHIFT_APPLIED in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT_APPLIED]

        _name_sep = '_'
        error_name = _name_sep.join([var_name, 'error'])

        # Special care must be taken of v[xy].stable_rmse in
        # optical legacy format vs. v[xy].v[xy]_error in radar format as these
        # are the same
        error_data = None
        if var_name in _stable_rmse_vars:
            error_data = [
                ITSCube.get_data_var_attr(ds, url, var_name, error_name, DataVars.MISSING_VALUE) if error_name in ds[var_name].attrs else
                ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_RMSE, DataVars.MISSING_VALUE)
                for ds, url in zip(self.ds, self.urls)
            ]

            # If attribute is propagated as cube's data var attribute, delete it
            if DataVars.STABLE_RMSE in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[DataVars.STABLE_RMSE]

        else:
            error_data = [ITSCube.get_data_var_attr(ds, url, var_name, error_name, DataVars.MISSING_VALUE)
                          for ds, url in zip(self.ds, self.urls)]

        self.layers[error_name] = xr.DataArray(
            data=error_data,
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.UNITS: DataVars.M_Y_UNITS,
                DataVars.STD_NAME: _attrs[error_name][0],
                DataVars.DESCRIPTION_ATTR: _attrs[error_name][1]
            }
        )

        # If attribute is propagated as cube's data var attribute, delete it
        if error_name in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[error_name]

        # This attribute appears for all v* data variables, capture it only once
        if DataVars.STABLE_COUNT not in self.layers:
            self.layers[DataVars.STABLE_COUNT] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_COUNT)
                      for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.UNITS: DataVars.COUNT_UNITS,
                    DataVars.STD_NAME: 'stable_count',
                    DataVars.DESCRIPTION_ATTR: f'number of stable points used to determine {var_name} shift'
            }
        )

        if DataVars.STABLE_COUNT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_COUNT]

        # This flag appears for vx and vy data variables, capture it only once.
        # "stable_shift_applied" was incorrectly set in the optical legacy dataset
        # and should be set to the no data value
        if DataVars.FLAG_STABLE_SHIFT not in self.layers:
            missing_stable_shift_value = 0.0
            self.layers[DataVars.FLAG_STABLE_SHIFT] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.FLAG_STABLE_SHIFT, missing_stable_shift_value)
                      for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.STD_NAME: DataVars.FLAG_STABLE_SHIFT,
                    DataVars.FLAG_STABLE_SHIFT_MEANINGS: DataVars.DESCRIPTION[DataVars.FLAG_STABLE_SHIFT_MEANINGS]
                }
            )

        # Create 'stable_shift' specific to the data variable,
        # for example, 'vx_stable_shift' for 'vx' data variable
        shift_var_name = _name_sep.join([var_name, DataVars.STABLE_SHIFT])
        self.layers[shift_var_name] = xr.DataArray(
            data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_SHIFT, DataVars.MISSING_VALUE)
                  for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.UNITS: DataVars.M_Y_UNITS,
                DataVars.STD_NAME: f'{var_name}_stable_shift',
                DataVars.DESCRIPTION_ATTR: f'applied {var_name} shift'
            }
        )

        # If attribute is propagated as cube's vx attribute, delete it
        if DataVars.STABLE_SHIFT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT]

        # Return names of new data variables - to be included into "encoding" settings
        # for writing to the file store.
        return [error_name, shift_var_name]

    def set_grid_mapping_attr(self, var_name: str, ds_grid_mapping_value: str):
        """
        Check on existence of "grid_mapping" attribute for the variable, set it
        if not present.
        """
        if DataVars.GRID_MAPPING in self.layers[var_name]:
            # Attribute is already set, nothing to do
            return

        grid_mapping_values = []
        for each_ds in self.ds:
            if var_name in each_ds and DataVars.GRID_MAPPING in each_ds[var_name].attrs:
                grid_mapping_values.append(each_ds[var_name].attrs[DataVars.GRID_MAPPING])

        # Flag if attribute needs to be set manually
        set_grid_mapping = False
        if len(grid_mapping_values) != len(self.ds):
            # None or some of the granules provide grid_mapping attribute
            # ("var_name" data variable might be present only in Radar format),
            # need to set it manually as xr.concat won't preserve the attribute
            set_grid_mapping = True

        unique_values = list(set(grid_mapping_values))
        if len(unique_values) > 1:
            raise RuntimeError(
                f"Inconsistent '{var_name}.{DataVars.GRID_MAPPING}' values are "
                "detected for current {len(self.ds)} layers: {unique_values}")

        if len(unique_values) and unique_values[0] != ds_grid_mapping_value:
            # Make sure the value is the same as previously detected
            raise RuntimeError(
                f"Inconsistent '{DataVars.GRID_MAPPING}' value in "
                "{var_name}: {self.layers[var_name].attrs[DataVars.GRID_MAPPING]} vs. {ds_grid_mapping_value}")

        if set_grid_mapping:
            self.layers[var_name].attrs[DataVars.GRID_MAPPING] = ds_grid_mapping_value

    def combine_layers(self, output_dir, is_first_write=False):
        """
        Combine selected layers into one xr.Dataset object and write (append) it
        to the Zarr store.
        """
        self.layers = {}

        # Construct xarray to hold layers by concatenating layer objects along 'mid_date' dimension
        self.logger.info(f'Combine {len(self.urls)} layers to the {output_dir}...')
        if len(self.ds) == 0:
            self.logger.info('No layers to combine, continue')
            return

        start_time = timeit.default_timer()
        mid_date_coord = pd.Index(self.dates, name=Coords.MID_DATE)

        # for each in self.ds:
        #     self.logger.info(f'Layer v: {each.v.values}')

        v_layers = xr.concat([each_ds.v for each_ds in self.ds], mid_date_coord)

        now_date = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.layers = xr.Dataset(
            data_vars = {DataVars.URL: ([Coords.MID_DATE], self.urls)},
            coords = {
                Coords.MID_DATE: (
                    Coords.MID_DATE,
                    self.dates,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.MID_DATE],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.MID_DATE]
                    }
                ),
                Coords.X: (
                    Coords.X,
                    self.grid_x,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X]
                    }
                ),
                Coords.Y: (
                    Coords.Y,
                    self.grid_y,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y]
                    }
                )
            },
            attrs = {
                'title': 'ITS_LIVE datacube of image_pair velocities',
                'author': 'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)',
                'institution': 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology',
                'date_created': now_date,
                'date_updated': now_date,
                'datacube_software_version': ITSCube.Version,
                'GDAL_AREA_OR_POINT': 'Area',
                'projection': str(self.projection)
            }
        )

        # Set attributes for 'url' data variable
        self.layers[DataVars.URL].attrs[DataVars.STD_NAME] = DataVars.URL
        self.layers[DataVars.URL].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.URL]

        # Set projection information once for the whole datacube
        if is_first_write:
            proj_data = None
            if DataVars.POLAR_STEREOGRAPHIC in self.ds[0]:
                proj_data = DataVars.POLAR_STEREOGRAPHIC

            elif DataVars.UTM_PROJECTION in self.ds[0]:
                proj_data = DataVars.UTM_PROJECTION

            # Should never happen - just in case :)
            if proj_data is None:
                raise RuntimeError(f"Missing {DataVars.POLAR_STEREOGRAPHIC} or {DataVars.UTM_PROJECTION} in {self.urls[0]}")

            # Can't copy the whole data variable, as it introduces obscure coordinates.
            # Just copy all attributes for the scalar type of the xr.DataArray.
            self.layers[proj_data] = xr.DataArray(
                data='',
                attrs=self.ds[0][proj_data].attrs,
                coords={},
                dims=[]
            )

        # ATTN: Assign one data variable at a time to avoid running out of memory.
        #       Delete each variable after it has been processed to free up the
        #       memory.

        # Process 'v' (all formats have v variable - its attributes are inherited,
        # so no need to set them manually)
        self.layers[DataVars.V] = v_layers
        self.layers[DataVars.V].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.V]
        new_v_vars = [DataVars.V]

        # Make sure grid_mapping attribute has the same value for all layers
        grid_mapping_values = [ds.v.attrs[DataVars.GRID_MAPPING] for ds in self.ds]
        unique_values = list(set(grid_mapping_values))
        if len(unique_values) > 1:
            raise RuntimeError("Multiple 'grid_mapping' ('v' attribute) values are detected for current {len(self.ds)} layers: {unique_values}")

        # Remember the value as all 3D data variables need to have this attribute
        # set with the same value
        ds_grid_mapping_value = unique_values[0]
        del grid_mapping_values

        # Collect 'v' attributes: these repeat for v* variables, keep only one copy
        # per datacube

        # Create new data var to store map_scale_corrected v's attribute
        self.layers[DataVars.MAP_SCALE_CORRECTED] = xr.DataArray(
            data=[ITSCube.get_data_var_attr(ds, url, DataVars.V, DataVars.MAP_SCALE_CORRECTED, DataVars.MISSING_BYTE)
                  for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE]
        )

        # If attribute is propagated as cube's v attribute, delete it
        if DataVars.MAP_SCALE_CORRECTED in self.layers[DataVars.V].attrs:
            del self.layers[DataVars.V].attrs[DataVars.MAP_SCALE_CORRECTED]

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [each.drop_vars(DataVars.V) for each in self.ds]
        del v_layers
        gc.collect()

        # Process 'vx'
        self.layers[DataVars.VX] = xr.concat([ds.vx for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VX].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VX]
        new_v_vars.append(DataVars.VX)
        new_v_vars.extend(self.process_v_attributes(DataVars.VX, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VX, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.VX) for ds in self.ds]
        gc.collect()

        # Process 'vy'
        self.layers[DataVars.VY] = xr.concat([ds.vy for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VY].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VY]
        new_v_vars.append(DataVars.VY)
        new_v_vars.extend(self.process_v_attributes(DataVars.VY, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VY, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.VY) for ds in self.ds]
        gc.collect()

        # Process 'va'
        self.layers[DataVars.VA] = xr.concat([self.get_data_var(ds, DataVars.VA) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VA].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VA]
        self.layers[DataVars.VA].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VA]
        self.layers[DataVars.VA].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        self.set_grid_mapping_attr(DataVars.VA, ds_grid_mapping_value)

        new_v_vars.append(DataVars.VA)
        new_v_vars.extend(self.process_v_attributes(DataVars.VA, mid_date_coord))

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VA) if DataVars.VA in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vr'
        self.layers[DataVars.VR] = xr.concat([self.get_data_var(ds, DataVars.VR) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VR].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VR]
        self.layers[DataVars.VR].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VR]
        self.layers[DataVars.VR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.VR)
        new_v_vars.extend(self.process_v_attributes(DataVars.VR, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VR, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VR) if DataVars.VR in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vxp'
        self.layers[DataVars.VXP] = xr.concat([self.get_data_var(ds, DataVars.VXP) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VXP].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VXP]
        self.layers[DataVars.VXP].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VXP]
        self.layers[DataVars.VXP].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.VXP)
        new_v_vars.extend(self.process_v_attributes(DataVars.VXP, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VXP, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VXP) if DataVars.VXP in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vyp'
        self.layers[DataVars.VYP] = xr.concat([self.get_data_var(ds, DataVars.VYP) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VYP].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VYP]
        self.layers[DataVars.VYP].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VYP]
        self.layers[DataVars.VYP].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.VYP)
        new_v_vars.extend(self.process_v_attributes(DataVars.VYP, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VYP, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VYP) if DataVars.VYP in ds else ds for ds in self.ds]
        gc.collect()

        # Process chip_size_height: dtype=ushort
        self.layers[DataVars.CHIP_SIZE_HEIGHT] = xr.concat([ds.chip_size_height for ds in self.ds], mid_date_coord)
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.CHIP_SIZE_COORDS] = DataVars.DESCRIPTION[DataVars.CHIP_SIZE_COORDS]
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.CHIP_SIZE_HEIGHT]

        self.set_grid_mapping_attr(DataVars.CHIP_SIZE_HEIGHT, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.CHIP_SIZE_HEIGHT) for ds in self.ds]
        gc.collect()

        # Process chip_size_width: dtype=ushort
        self.layers[DataVars.CHIP_SIZE_WIDTH] = xr.concat([ds.chip_size_width for ds in self.ds], mid_date_coord)
        self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.CHIP_SIZE_COORDS] = DataVars.DESCRIPTION[DataVars.CHIP_SIZE_COORDS]
        self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.CHIP_SIZE_WIDTH]

        self.set_grid_mapping_attr(DataVars.CHIP_SIZE_WIDTH, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.CHIP_SIZE_WIDTH) for ds in self.ds]
        gc.collect()

        # Process interp_mask: dtype=ubyte
        self.layers[DataVars.INTERP_MASK] = xr.concat([ds.interp_mask for ds in self.ds], mid_date_coord)
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.INTERP_MASK]
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.INTERP_MASK]
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.UNITS] = DataVars.BINARY_UNITS

        self.set_grid_mapping_attr(DataVars.INTERP_MASK, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.INTERP_MASK) for ds in self.ds]
        gc.collect()

        # Process 'v_error'
        self.layers[DataVars.V_ERROR] = xr.concat([self.get_data_var(ds, DataVars.V_ERROR) for ds in self.ds] , mid_date_coord)
        self.layers[DataVars.V_ERROR].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.V_ERROR]
        self.layers[DataVars.V_ERROR].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.V_ERROR]
        self.layers[DataVars.V_ERROR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS
        new_v_vars.append(DataVars.V_ERROR)

        self.set_grid_mapping_attr(DataVars.V_ERROR, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.V_ERROR) if DataVars.V_ERROR in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vp'
        self.layers[DataVars.VP] = xr.concat([self.get_data_var(ds, DataVars.VP) for ds in self.ds] , mid_date_coord)
        self.layers[DataVars.VP].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VP]
        self.layers[DataVars.VP].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VP]
        self.layers[DataVars.VP].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS
        new_v_vars.append(DataVars.VP)

        self.set_grid_mapping_attr(DataVars.VP, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VP) if DataVars.VP in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vp_error'
        self.layers[DataVars.VP_ERROR] = xr.concat([self.get_data_var(ds, DataVars.VP_ERROR) for ds in self.ds] , mid_date_coord)
        self.layers[DataVars.VP_ERROR].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VP_ERROR]
        self.layers[DataVars.VP_ERROR].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VP_ERROR]
        self.layers[DataVars.VP_ERROR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS
        new_v_vars.append(DataVars.VP_ERROR)

        self.set_grid_mapping_attr(DataVars.VP_ERROR, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VP_ERROR) if DataVars.VP_ERROR in ds else ds for ds in self.ds]
        gc.collect()

        for each in DataVars.ImgPairInfo.ALL:
            # Add new variables that correspond to attributes of 'img_pair_info'
            # (only selected ones)
            self.layers[each] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, each, to_date=DataVars.ImgPairInfo.CONVERT_TO_DATE[each]
                ) for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.STD_NAME: DataVars.ImgPairInfo.STD_NAME[each],
                    DataVars.DESCRIPTION_ATTR: DataVars.ImgPairInfo.DESCRIPTION[each]
                }
            )
            if each in DataVars.ImgPairInfo.UNITS:
                # Units attribute exists for the variable
                self.layers[each].attrs[DataVars.UNITS] = DataVars.ImgPairInfo.UNITS[each]

        # Handle acquisition time separately as it has different names in
        # optical and radar formats
        var_name = DataVars.ImgPairInfo.ACQUISITION_IMG1
        self.layers[var_name] = xr.DataArray(
            data=[
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, var_name, to_date = True
                ) if var_name in ds[DataVars.ImgPairInfo.NAME].attrs else
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, DataVars.ImgPairInfo.AQUISITION_DATE_IMG1, to_date = True
                ) for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.STD_NAME: DataVars.ImgPairInfo.STD_NAME[var_name],
                DataVars.DESCRIPTION_ATTR: DataVars.ImgPairInfo.DESCRIPTION[var_name],
            }
        )
        var_name = DataVars.ImgPairInfo.ACQUISITION_IMG2
        self.layers[var_name] = xr.DataArray(
            data=[
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, var_name, to_date = True
                ) if var_name in ds[DataVars.ImgPairInfo.NAME].attrs else
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, DataVars.ImgPairInfo.AQUISITION_DATE_IMG2, to_date = True
                ) for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.STD_NAME: DataVars.ImgPairInfo.STD_NAME[var_name],
                DataVars.DESCRIPTION_ATTR: DataVars.ImgPairInfo.DESCRIPTION[var_name],
            }
        )

        time_delta = timeit.default_timer() - start_time
        self.logger.info(f"Combined {len(self.urls)} layers (took {time_delta} seconds)")

        start_time = timeit.default_timer()
        # Write to the Zarr store
        if is_first_write:
            # Set missing_value only on first write to the disk store, otherwise
            # will get "ValueError: failed to prevent overwriting existing key missing_value in attrs."
            for each in [DataVars.MAP_SCALE_CORRECTED,
                         DataVars.CHIP_SIZE_HEIGHT,
                         DataVars.CHIP_SIZE_WIDTH,
                         DataVars.INTERP_MASK]:
                self.layers[each].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE

            # ATTN: Must set '_FillValue' for each data variable that has
            #       its missing_value attribute set
            encoding_settings = {}
            for each in [DataVars.MAP_SCALE_CORRECTED,
                         DataVars.INTERP_MASK,
                         DataVars.CHIP_SIZE_HEIGHT,
                         DataVars.CHIP_SIZE_WIDTH,
                         DataVars.FLAG_STABLE_SHIFT]:
                encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE}

            # Explicitly set dtype to 'byte' for some data variables
            for each in [DataVars.CHIP_SIZE_HEIGHT,
                         DataVars.CHIP_SIZE_WIDTH]:
                encoding_settings[each]['dtype'] = 'ushort'

            # Explicitly set dtype for some variables
            encoding_settings[DataVars.MAP_SCALE_CORRECTED]['dtype'] = 'byte'
            encoding_settings[DataVars.INTERP_MASK]['dtype'] = 'ubyte'
            encoding_settings[DataVars.FLAG_STABLE_SHIFT]['dtype'] = 'long'

            for each in new_v_vars:
                encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE}

                # Set missing_value only on first write to the disk store, otherwise
                # will get "ValueError: failed to prevent overwriting existing key
                # missing_value in attrs."
                if DataVars.MISSING_VALUE_ATTR not in self.layers[each].attrs:
                    self.layers[each].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_VALUE

            # Explicitly set dtype to 'short' for v* data variables
            for each in [DataVars.V,
                         DataVars.VX,
                         DataVars.VY,
                         DataVars.VA,
                         DataVars.VR,
                         DataVars.VXP,
                         DataVars.VYP,
                         DataVars.VP,
                         DataVars.V_ERROR,
                         DataVars.VP_ERROR]:
                encoding_settings[each]['dtype'] = 'short'

            # Explicitly desable _FillValue for some variables
            for each in [Coords.MID_DATE,
                         DataVars.STABLE_COUNT,
                         DataVars.ImgPairInfo.AUTORIFT_SOFTWARE_VERSION,
                         DataVars.ImgPairInfo.DATE_DT,
                         DataVars.ImgPairInfo.DATE_CENTER]:
                encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})

            # Set units for all datetime objects
            for each in [DataVars.ImgPairInfo.ACQUISITION_IMG1,
                         DataVars.ImgPairInfo.ACQUISITION_IMG2,
                         DataVars.ImgPairInfo.DATE_CENTER,
                         Coords.MID_DATE]:
                encoding_settings[each] = {DataVars.UNITS: DataVars.ImgPairInfo.DATE_UNITS}

            self.logger.info(f"Encoding writing to Zarr: {encoding_settings}")

            # This is first write, create Zarr store
            # self.layers.to_zarr(output_dir, encoding=encoding_settings, consolidated=True)
            self.layers.to_zarr(output_dir, encoding=encoding_settings)

        else:
            # Append layers to existing Zarr store
            # self.layers.to_zarr(output_dir, append_dim=Coords.MID_DATE, consolidated=True)
            self.layers.to_zarr(output_dir, append_dim=Coords.MID_DATE)

        time_delta = timeit.default_timer() - start_time
        self.logger.info(f"Wrote {len(self.urls)} layers to {output_dir} (took {time_delta} seconds)")

        # Free up memory
        self.clear_vars()

        # No need to sort data by date as we will be appending layers to the datacubes

    def format_stats(self):
        """
        Format statistics of the run.
        """
        num_urls = self.num_urls_from_api
        # Total number of skipped granules due to wrong projection
        sum_projs = sum([len(each) for each in self.skipped_proj_granules.values()])

        all_projs = []
        for each in self.skipped_proj_granules.values():
            all_projs.extend(each)

        print("Skipped granules:")
        print(f"      empty data       : {len(self.skipped_empty_granules)} ({100.0 * len(self.skipped_empty_granules)/num_urls}%)")
        print(f"      wrong projection : {sum_projs} ({100.0 * sum_projs/num_urls}%)")
        print(f"      double mid_date  : {len(self.skipped_double_granules)} ({100.0 * len(self.skipped_double_granules)/num_urls}%)")
        if len(self.skipped_proj_granules):
            print(f"      wrong projections: {sorted(self.skipped_proj_granules.keys())}")

        self.logger.info(f"Writing skipped granule informaton to {ITSCube.GRANULE_REPORT_DIR}")

        # Write skipped granules due to double middle date to the file
        with open(os.path.join(ITSCube.GRANULE_REPORT_DIR, 'double_middle_date.txt'), 'w') as fh:
            fh.write(f"# Skipped granules due to double middle date: {len(self.skipped_double_granules)}\n")
            fh.write('\n'.join(self.skipped_double_granules))

        # Write skipped granules due to no spacial coverage to the file
        with open(os.path.join(ITSCube.GRANULE_REPORT_DIR, 'empty_granules.txt'), 'w') as fh:
            fh.write(f"# Skipped granules due to no data: {len(self.skipped_empty_granules)}\n")
            fh.write('\n'.join(self.skipped_empty_granules))

        # Write skipped granules due to the wrong projection to the file
        with open(os.path.join(ITSCube.GRANULE_REPORT_DIR, 'wrong_proj_granules.txt'), 'w') as fh:
            fh.write(f"# Skipped granules due to wrong projection: {len(all_projs)}\n")
            fh.write('\n'.join(all_projs))

        self.logger.info(f"Done.")

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
    import sys
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
                        help='Local path that stores ITS_LIVE granules.')
    parser.add_argument('-o', '--outputDir', type=str, default="cubedata.zarr",
                        help="Zarr output directory to write cube data to. Default is 'cubedata.zarr'.")
    parser.add_argument('-c', '--chunks', type=int, default=1000,
                        help="Number of granules to write at a time. Default is 1000.")
    parser.add_argument('--targetProjection', type=str, required=True,
                        help="UTM target projection.")
    parser.add_argument('--dimSize', type=float, default=100000,
                        help="Cube dimension in meters.")
    parser.add_argument('--centroid', nargs=2, metavar=('x', 'y'), type=float,
                        action='store',
                        help="Centroid point for the datacube in UTM projection.")


    args = parser.parse_args()
    ITSCube.NUM_THREADS = args.threads
    ITSCube.DASK_SCHEDULER = args.scheduler
    ITSCube.NUM_GRANULES_TO_WRITE = args.chunks

    # Test Case from itscube.ipynb:
    # =============================
    # Create polygon as a square around the centroid in target '32628' UTM projection
    # Projection for the polygon coordinates
    # projection = '32628'
    projection = args.targetProjection

    # Centroid for the tile in target projection
    # c_x, c_y = (487462, 9016243)
    c_x, c_y = args.centroid

    # Offset in meters (1 pixel=240m): 100 km square (with offset=50km)
    # off = 50000
    off = args.dimSize / 2.0
    polygon = (
        (c_x - off, c_y + off),
        (c_x + off, c_y + off),
        (c_x + off, c_y - off),
        (c_x - off, c_y - off),
        (c_x - off, c_y + off))

    # Create cube object
    cube = ITSCube(polygon, projection)

    cube.logger.info(f"Command: {sys.argv}")

    # Parameters for the search granule API
    API_params = {
        'start'               : '1984-01-01',
        'end'                 : '2021-01-01',
        'percent_valid_pixels': 1
    }
    cube.logger.info("ITS_LIVE API parameters: %s" %API_params)

    skipped_projs = {}
    if not args.parallel:
        # Process ITS_LIVE granules sequentially, look at provided number of granules only
        cube.logger.info("Processing granules sequentially...")
        if args.localPath:
            # Granules are downloaded locally
            cube.create_from_local_no_api(args.outputDir, args.localPath, args.numberGranules)

        else:
            cube.create(API_params, args.outputDir, args.numberGranules)

    else:
        # Process ITS_LIVE granules in parallel, look at 100 first granules only
        cube.logger.info("Processing granules in parallel...")
        if args.localPath:
            # Granules are downloaded locally
            cube.create_from_local_parallel_no_api(args.outputDir, args.localPath, args.numberGranules)

        else:
            cube.create_parallel(API_params, args.outputDir, args.numberGranules)

    # Write cube data to the NetCDF file
    # cube.to_netcdf('test_v_cube.nc')
