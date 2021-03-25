"""
Reprojection tool for ITS_LIVE granule to new target projection.

Examples:
$ python reproject.py -i input_filename -p target_projection -o output_filename

    Reproject "input_filename" into 'target_projection' and output new granule into
'output_filename' in NetCDF format.

$ python ./reproject.py -i  LC08_L1TP_042242_20180721_20180731_01_T1_X_LC08_L1TP_042242_20170702_20170702_01_RT_G0240V01_P065.nc -o out_noNN_P065.nc -p 32627
"""
import argparse
from datetime import datetime
import gc
import logging
import math
import numpy as np
from osgeo import osr
from osgeo import gdal
import xarray as xr

from grid import Grid, Bounds
from itscube_types import Coords, DataVars


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


__spatial_ref_3031 = "PROJCS[\"WGS 84 / Antarctic Polar Stereographic\"," \
    "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\"," \
    "6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]]," \
    "AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0," \
    "AUTHORITY[\"EPSG\",\"8901\"]],UNIT[“degree\",0.0174532925199433," \
    "AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]," \
    "PROJECTION[\"Polar_Stereographic\"],PARAMETER[\"latitude_of_origin\",-71]," \
    "PARAMETER[\"central_meridian\",0],PARAMETER[\"false_easting\",0]," \
    "PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]" \
    ",AXIS[\"Easting\",NORTH],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"3031\"]]"

__spatial_ref_3413 = "PROJCS[\"WGS 84 / NSIDC Sea Ice Polar Stereographic North\"," \
    "\"GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563," \
    "AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]]" \
    ",PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]]," \
    "UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]]," \
    "AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Polar_Stereographic\"]," \
    "PARAMETER[\"latitude_of_origin\",70],PARAMETER[\"central_meridian\",-45]," \
    "PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0]," \
    "UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",SOUTH]," \
    "AXIS[\"Northing\",SOUTH],AUTHORITY[\"EPSG\",\"3413\"]]"

# Mapping of UTM zone to the central meridian
_epsg_to_central_meridian = {
    27: -21,
    28: -15,
    29: -9
}

class ItsLiveReproject:
    """
    Class to store input ITS_LIVE granule, and functionality to re-project
    its data into a new target projection.

    The following steps must be taken to re-project ITS_LIVE granule to new
    projection:

    1. Compute bounding box for input granule in original P_in projection ("ij" naming convention)
    2. Re-project P_in bounding box to P_out projection ("xy" naming convention)
    3. Compute grid in P_out projection based on its bounding bbox
    4. Project each cell center in P_out grid to original P_in projection: (x0, y0)
    5. Add unit length (240m) to x0 of (x0, y0) and project to P_out: (x1, y1)
    6. Add unit length (240m) to y0 of (x0, y0) and project to P_out: (x2, y2)
    7. In Geogrid code, set normal = (0, 0, 1)
    8. Compute transformation matrix using Geogrid equations
    9. Re-project v* values: gdal.warp(original_granule, P_out_grid) --> P_out_v*
       Apply tranformation matrix to P_out_v* per cell to get "true" v value
    """
    # Number of years in one day: any period would do as long as it's
    # the same time period used to convert v(elocity) component to corresponding
    # d(isplacement), and use the same time period in transformation matrix
    # computations.
    TIME_DELTA = 1.0/365.0

    V_ERROR_ATTRS = {
        DataVars.STD_NAME:         DataVars.NAME[DataVars.V_ERROR],
        DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.V_ERROR],
        DataVars.UNITS:            DataVars.M_Y_UNITS
    }
    def __init__(self, data, output_projection: int):
        """
        Initialize object.
        """
        self.logger = logging.getLogger("ItsLiveReproject")

        self.ds = data
        self.input_file = None
        if isinstance(data, str):
            # Filename for the dataset is provided, read it in
            self.input_file = data
            self.ds = xr.open_dataset(data)
            self.ds.load()

        # Input and output projections
        self.ij_epsg = int(self.ds.UTM_Projection.spatial_epsg)
        self.xy_epsg = output_projection

        if self.ij_epsg == self.xy_epsg:
            raise RuntimeError("Done: original data is in the target {self.xy_epsg} projection already.")

        self.logger.info(f"Reprojecting from {self.ij_epsg} to {self.xy_epsg}")

        # Grid spacing
        self.x_size = self.ds.x.values[1] - self.ds.x.values[0]
        self.y_size = self.ds.y.values[1] - self.ds.y.values[0]

        self.i_limits = Bounds(self.ds.x.values)
        self.j_limits = Bounds(self.ds.y.values)
        self.logger.info(f"P_in:                  x: {self.i_limits} y: {self.j_limits}")
        self.logger.info(f"Grid in P_in:          num_x={len(self.ds.x.values)} num_y={len(self.ds.y.values)}")

        # Placeholders for:
        # bounding box in output projection
        self.x0_bbox = None
        self.y0_bbox = None

        # grid coordinates in output projection
        self.x0_grid = None
        self.y0_grid = None

        # Indices for original cells that correspond to the re-projected cells:
        # to find corresponding values
        self.original_ij_index = None

        # Transformation matrix to rotate warped velocity components (vx* and vy*)
        # in output projection
        self.transformation_matrix = None

    def bounding_box(self):
        """
        Identify bounding box for original dataset.
        """
        # ATTN: Assuming that X and Y cell spacings are the same
        assert np.abs(self.x_size) == np.abs(self.y_size), \
            f"Cell dimensions differ: x={np.abs(self.x_size)} y={np.abs(self.y_size)}"

        center_off_X = self.x_size/2
        center_off_Y = self.y_size/2

        # Compute cell boundaries as ITS_LIVE grid stores x/y for the cell centers
        xmin = self.i_limits.min - center_off_X
        xmax = self.i_limits.max + center_off_X

        # Y coordinate calculations are based on the fact that dy < 0
        ymin = self.j_limits.min + center_off_Y
        ymax = self.j_limits.max - center_off_Y

        return Grid.bounding_box(
            Bounds(min_value=xmin, max_value=xmax),
            Bounds(min_value=ymin, max_value=ymax),
            self.x_size
        )

    def run(self, output_file: str = None):
        """
        Run reprojection of ITS_LIVE granule into target projection.

        This methods warps X and Y components of v and vp velocities, and
        adjusts them by rotation for new projection.
        """
        self.create_transformation_matrix()

        # outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS
        warp_options = gdal.WarpOptions(
            # format='netCDF',
            format='vrt',   # Use virtual memory format to avoid writing warped dataset to the file
            outputBounds=(self.x0_bbox.min, self.y0_bbox.min, self.x0_bbox.max, self.y0_bbox.max),
            xRes=self.x_size,
            yRes=self.y_size,
            srcSRS=f'EPSG:{self.ij_epsg}',
            dstSRS=f'EPSG:{self.xy_epsg}',
            # srcNodata=DataVars.MISSING_VALUE,
            # dstNodata=DataVars.MISSING_VALUE,
            resampleAlg=gdal.GRA_NearestNeighbour
        )

        # Check if v_error is present in original data
        v_error_np = None
        if DataVars.V_ERROR in self.ds:
            v_error_np = self.ds[DataVars.V_ERROR].values

        else:
            # v_error does not exist, compute as
            # (abs(vx)*stable_rmse_vx + abs(vy)*stable_rmse_vy)/v
            num_i = len(self.ds.x.values)
            num_j = len(self.ds.y.values)
            v_error_np = np.zeros((num_j, num_i))
            v_error_np.fill(DataVars.MISSING_VALUE)

            v_np = self.ds[DataVars.V].values
            vx_np = self.ds[DataVars.VX].values
            vy_np = self.ds[DataVars.VY].values

            vx_stable_rmse = self.ds[DataVars.VX].attrs[DataVars.STABLE_RMSE]
            vy_stable_rmse = self.ds[DataVars.VY].attrs[DataVars.STABLE_RMSE]

            for j_ind in range(num_j):
                for i_ind in range(num_i):
                    if v_np[j_ind, i_ind] != 0:
                        v_error_np[j_ind, i_ind] = (np.abs(vx_np[j_ind, i_ind])*vx_stable_rmse + \
                                                    np.abs(vy_np[j_ind, i_ind])*vy_stable_rmse) / v_np[j_ind, i_ind]

        # Compute new vx, vy and v
        vx, vy, v, v_error = self.reproject_velocity(
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            v_error_np,
            warp_options
        )

        # Create new granule in target projection
        ds_coords=[
            (Coords.Y, self.y0_grid, self.ds.y.attrs),
            (Coords.X, self.x0_grid, self.ds.x.attrs)
        ]

        reproject_ds = xr.Dataset(
            data_vars={
                DataVars.VX: xr.DataArray(
                    data=vx,
                    coords=ds_coords,
                    attrs=self.ds[DataVars.VX].attrs),
                DataVars.VY: xr.DataArray(
                    data=vy,
                    coords=ds_coords,
                    attrs=self.ds[DataVars.VY].attrs),
                DataVars.V: xr.DataArray(
                    data=v,
                    coords=ds_coords,
                    attrs=self.ds[DataVars.V].attrs),
                DataVars.V_ERROR: xr.DataArray(
                    data=v_error,
                    coords=ds_coords,
                    attrs=self.ds[DataVars.V_ERROR].attrs \
                          if DataVars.V_ERROR in self.ds \
                          else ItsLiveReproject.V_ERROR_ATTRS),
            },
            coords={
                Coords.Y: (Coords.Y, self.y0_grid, self.ds[Coords.Y].attrs),
                Coords.X: (Coords.X, self.x0_grid, self.ds[Coords.X].attrs),
            },
            attrs=self.ds.attrs
        )

        # Use garbage collection
        vx = None
        vy = None
        v = None
        v_error = None
        gc.collect()

        # Update vx.vx_error (vx.stable_rmse in Optical legacy format) and
        # vy.vy_error (vy.stable_rmse in Optical legacy format):
        # rotate by transformation matrix that corresponds to the center of the grid
        vx_error = self.ds[DataVars.VX].attrs[DataVars.STABLE_RMSE] \
            if DataVars.STABLE_RMSE in self.ds[DataVars.VX].attrs else \
            self.ds[DataVars.VX].attrs[DataVars.VX_ERROR]

        vy_error = self.ds[DataVars.VY].attrs[DataVars.STABLE_RMSE] \
            if DataVars.STABLE_RMSE in self.ds[DataVars.VY].attrs else \
            self.ds[DataVars.VY].attrs[DataVars.VY_ERROR]

        # Get transformation matrix for the center of polygon in target projection
        mid_x_index = int(len(self.x0_grid)/2)
        mid_y_index = int(len(self.y0_grid)/2)

        v_error = np.matmul(self.transformation_matrix[mid_y_index, mid_x_index], [vx_error, vy_error])
        reproject_ds[DataVars.VX].attrs[DataVars.VX_ERROR] = v_error[0]
        reproject_ds[DataVars.VY].attrs[DataVars.VY_ERROR] = v_error[1]

        if DataVars.STABLE_RMSE in reproject_ds[DataVars.VX].attrs:
            # Delete legacy attribute for vx and vy if any
            del reproject_ds[DataVars.VX].attrs[DataVars.STABLE_RMSE]
            del reproject_ds[DataVars.VY].attrs[DataVars.STABLE_RMSE]

        # Use garbage collection
        vx = None
        vy = None
        v = None
        v_error = None
        gc.collect()

        # Set reprojection information
        reproject_ds.attrs['date_reprojected'] = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        reproject_ds.attrs['reprojected_from'] = self.input_file

        # Add projection data variable
        proj_data = None
        proj_name = DataVars.POLAR_STEREOGRAPHIC
        if self.xy_epsg == 3031:
            proj_data = xr.DataArray(
                data='',
                coords={},
                dims=[],
                attrs={
                    DataVars.GRID_MAPPING: 'polar_stereographic',
                    'straight_vertical_longitude_from_pole': 0,
                    'latitude_of_projection_origin': -90.0,
                    'latitude_of_origin': -71.0,
                    'scale_factor_at_projection_origin': 1,
                    'false_easting': 0.0,
                    'false_northing': 0.0,
                    'semi_major_axis': 6378.137,
                    'semi_minor_axis': 6356.752,
                    'inverse_flattening': 298.257223563,
                    'spatial_ref': __spatial_ref_3031,
                    'spatial_proj4': "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
                }
            )

        elif self.xy_epsg == 3413:
            proj_data = xr.DataArray(
                data='',
                coords={},
                dims=[],
                attrs={
                    DataVars.GRID_MAPPING: 'polar_stereographic',
                    'straight_vertical_longitude_from_pole': -45,
                    'latitude_of_projection_origin': 90.0,
                    'latitude_of_origin': 70.0,
                    'scale_factor_at_projection_origin': 1,
                    'false_easting': 0.0,
                    'false_northing': 0.0,
                    'semi_major_axis': 6378.137,
                    'semi_minor_axis': 6356.752,
                    'inverse_flattening': 298.257223563,
                    'spatial_ref': __spatial_ref_3413,
                    'spatial_proj4': "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
                }
            )

        else:
            proj_name = DataVars.UTM_PROJECTION
            zone, spacial_ref_value = self.spatial_ref_32x()
            proj_data = xr.DataArray(
                data='',
                coords={},
                dims=[],
                attrs={
                    DataVars.GRID_MAPPING: 'universal_transverse_mercator',
                    'utm_zone_number': zone,
                    'semi_major_axis': 6378137,
                    'inverse_flattening': 298.257223563,
                    'CoordinateTransformType': 'Projection',
                    'CoordinateAxisTypes': 'GeoX GeoY',
                    'spatial_ref': spacial_ref_value,
                    'spatial_proj4': f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
                }
            )

        proj_data.attrs['spatial_epsg'] = self.xy_epsg
        # Format GeoTransform:
        # x top left (cell left most boundary), grid size, 0, y top left (cell upper most boundary), 0, -grid size
        half_x_cell = self.x_size/2.0
        half_y_cell = self.y_size/2.0
        proj_data.attrs['GeoTransform'] = f"{self.x0_grid[0] - half_x_cell} {self.x_size} 0 {self.y0_grid[0] - half_y_cell} 0 {self.y_size}"
        reproject_ds[proj_name] = proj_data

        # Set grid_mapping for vx, vy:
        reproject_ds[DataVars.VX].attrs[DataVars.GRID_MAPPING] = proj_name
        reproject_ds[DataVars.VY].attrs[DataVars.GRID_MAPPING] = proj_name
        reproject_ds[DataVars.V].attrs[DataVars.GRID_MAPPING] = proj_name
        reproject_ds[DataVars.V_ERROR].attrs[DataVars.GRID_MAPPING] = proj_name

        if DataVars.VP in self.ds:
            # This is Radar format, projected velocity is provided
            vxp, vyp, vp, vp_error = self.reproject_velocity(
                DataVars.VXP,
                DataVars.VYP,
                DataVars.VP_ERROR,
                self.ds[DataVars.VP_ERROR].values,
                warp_options)

            reproject_ds[DataVars.VXP] = xr.DataArray(
                data=vxp,
                coords=ds_coords,
                attrs=self.ds[DataVars.VXP].attrs
            )
            reproject_ds[DataVars.VXP].attrs[DataVars.GRID_MAPPING] = proj_name

            reproject_ds[DataVars.VYP] = xr.DataArray(
                data=vyp,
                coords=ds_coords,
                attrs=self.ds[DataVars.VYP].attrs
            )
            reproject_ds[DataVars.VYP].attrs[DataVars.GRID_MAPPING] = proj_name

            reproject_ds[DataVars.VP] = xr.DataArray(
                data=vp,
                coords=ds_coords,
                attrs=self.ds[DataVars.VP].attrs
            )
            reproject_ds[DataVars.VP].attrs[DataVars.GRID_MAPPING] = proj_name

            reproject_ds[DataVars.VP_ERROR] = xr.DataArray(
                data=vp_error,
                coords=ds_coords,
                attrs=self.ds[DataVars.VP_ERROR].attrs
            )
            reproject_ds[DataVars.VP_ERROR].attrs[DataVars.GRID_MAPPING] = proj_name

            vxp = None
            vyp = None
            vp = None
            vp_error = None
            gc.collect()

            # Process VA, VR
            reproject_ds[DataVars.VA] = xr.DataArray(
                data=self.warp_var(DataVars.VA, warp_options),
                coords=ds_coords,
                attrs=self.ds[DataVars.VA].attrs
            )
            reproject_ds[DataVars.VA].attrs[DataVars.GRID_MAPPING] = proj_name

            reproject_ds[DataVars.VR] = xr.DataArray(
                data=self.warp_var(DataVars.VR, warp_options),
                coords=ds_coords,
                attrs=self.ds[DataVars.VR].attrs
            )
            reproject_ds[DataVars.VR].attrs[DataVars.GRID_MAPPING] = proj_name

        # All formats have the following data variables
        reproject_ds[DataVars.CHIP_SIZE_HEIGHT] = xr.DataArray(
            data=self.warp_var(DataVars.CHIP_SIZE_HEIGHT, warp_options),
            coords=ds_coords,
            attrs=self.ds[DataVars.CHIP_SIZE_HEIGHT].attrs
        )
        reproject_ds[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.GRID_MAPPING] = proj_name

        reproject_ds[DataVars.CHIP_SIZE_WIDTH] = xr.DataArray(
            data=self.warp_var(DataVars.CHIP_SIZE_WIDTH, warp_options),
            coords=ds_coords,
            attrs=self.ds[DataVars.CHIP_SIZE_WIDTH].attrs
        )
        reproject_ds[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.GRID_MAPPING] = proj_name

        reproject_ds[DataVars.INTERP_MASK] = xr.DataArray(
            data=self.warp_var(DataVars.INTERP_MASK, warp_options),
            coords=ds_coords,
            attrs=self.ds[DataVars.INTERP_MASK].attrs
        )
        reproject_ds[DataVars.INTERP_MASK].attrs[DataVars.GRID_MAPPING] = proj_name

        ItsLiveReproject.write_to_netCDF(reproject_ds, output_file)

    @staticmethod
    def write_to_netCDF(ds, output_file: str):
        """
        Write dataset to the netCDF format file.
        """
        if output_file is None:
            # Output filename is not provided, don't write to the file
            return

        encoding_settings = {}
        compression = {"zlib": True, "complevel": 1}

        # Set missing_value
        for each in [DataVars.CHIP_SIZE_HEIGHT,
                     DataVars.CHIP_SIZE_WIDTH,
                     DataVars.INTERP_MASK]:
            ds[each].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE
            # ATTN: Must set '_FillValue' for each data variable that has
            #       its missing_value attribute set
            encoding_settings[each] = {
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
                DataVars.FILL_VALUE_ATTR: None
            }
            encoding_settings[each].update(compression)

        # Disable FillValue for coordinates
        for each in [Coords.X, Coords.Y]:
            encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: None}

        DataVars.FILL_VALUE_ATTR: None
        # Explicitly set dtype to 'byte' for some data variables
        for each in [DataVars.CHIP_SIZE_HEIGHT,
                     DataVars.CHIP_SIZE_WIDTH]:
            encoding_settings[each]['dtype'] = 'ushort'

        # Explicitly set dtype for some variables
        encoding_settings[DataVars.INTERP_MASK]['dtype'] = 'ubyte'

        for each in [DataVars.V, DataVars.VX, DataVars.VY, DataVars.VA, DataVars.VR,
            DataVars.VXP, DataVars.VYP, DataVars.VP, DataVars.V_ERROR, DataVars.VP_ERROR]:
            if each in ds:
                encoding_settings[each] = {
                    DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                    'dtype': 'short'
                }
                encoding_settings[each].update(compression)

                # Set missing_value only on first write to the disk store, otherwise
                # will get "ValueError: failed to prevent overwriting existing key
                # missing_value in attrs."
                if DataVars.MISSING_VALUE_ATTR not in ds[each].attrs:
                    ds[each].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_VALUE

        # print(f"Using encodings: {encoding_settings}")

        # write re-projected data to the file
        ds.to_netcdf(output_file, engine="h5netcdf", encoding = encoding_settings)

    def warp_var(self, var: str, warp_options: gdal.WarpOptions):
        """
        Warp variable into new projection.
        """
        dataset = gdal.Open(f'NETCDF:"{self.input_file}":{var}')

        # Warp data variable
        var_ds = gdal.Warp('', dataset, options=warp_options)
        np_var = var_ds.ReadAsArray()
        self.logger.info(f"Read with GDAL {var}.shape = {np_var.shape}")

        return np_var

    def reproject_velocity(self,
        vx: str,
        vy: str,
        v_var: str,
        v_error_np: np.ndarray,   # v_error in original projection
        warp_options: gdal.WarpOptions):
        """
        Re-project velocity X and Y components and compute velocity magnitude.
        """
        # Warp x component
        dataset = gdal.Open(f'NETCDF:"{self.input_file}":{vx}')
        vx_ds = gdal.Warp('', dataset, options=warp_options)
        np_vx = vx_ds.ReadAsArray()
        self.logger.info(f"Read with GDAL {vx}.shape = {np_vx.shape}")

        # Warp y component
        dataset = gdal.Open(f'NETCDF:"{self.input_file}":{vy}')
        vy_ds = gdal.Warp('', dataset, options=warp_options)
        np_vy = vy_ds.ReadAsArray()
        self.logger.info(f"Read with GDAL {vy}.shape = {np_vy.shape}")

        # Convert velocity components to displacement (per transformation matrix requirement)
        # NOTE: displacement values are in pixel units
        # original_dtype = np_vx.dtype

        np_vx = np_vx.astype(type(self.x_size))
        np_vx[np_vx==DataVars.MISSING_VALUE] = np.nan
        np_vx *= ItsLiveReproject.TIME_DELTA/self.x_size

        np_vy = np_vx.astype(type(self.y_size))
        np_vy[np_vy==DataVars.MISSING_VALUE] = np.nan
        np_vy *= ItsLiveReproject.TIME_DELTA/self.y_size

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)

        vx = np.zeros((num_y, num_x))
        vx.fill(DataVars.MISSING_VALUE)
        vy = np.zeros((num_y, num_x))
        vy.fill(DataVars.MISSING_VALUE)
        v = np.zeros((num_y, num_x))
        v.fill(DataVars.MISSING_VALUE)

        v_error = np.zeros((num_y, num_x))
        v_error.fill(DataVars.MISSING_VALUE)

        # Transform each (dx, dy) to (vx, vy) in output projection
        for y_index in range(num_y):
            for x_index in range(num_x):
                dv = np.array([
                    np_vx[y_index, x_index],
                    np_vy[y_index, x_index]
                ])

                # There is no transformation matrix available for the point -->
                # keep it as NODATA
                if not np.isscalar(self.transformation_matrix[y_index, x_index]) and \
                   not np.any(np.isnan(dv)):  # some warped points get NODATA for vx but valid vy
                    # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                    xy_v = np.matmul(self.transformation_matrix[y_index, x_index], dv)

                    vx[y_index, x_index] = xy_v[0]
                    vy[y_index, x_index] = xy_v[1]

                    # Compute v: sqrt(vx^2 + vy^2)
                    v[y_index, x_index] = np.sqrt(xy_v[0]**2 + xy_v[1]**2)

                    # Look up original velocity value to compute the scale factor
                    # for v_error: scale_factor = v_old / v_new
                    v_i, v_j = self.original_ij_index[y_index, x_index]
                    if v[y_index, x_index] != 0:
                        v_error[y_index, x_index] = (float)(v_error_np[v_j, v_i])*float(self.ds[v_var].isel(y=v_j, x=v_i).values.item())/float(v[y_index, x_index])

        return (vx, vy, v, v_error)

    def create_transformation_matrix(self):
        """
        Reproject variables in ITS_LIVE granule into new projection.
        """
        # Project the bounding box into output projection
        input_projection = osr.SpatialReference()
        input_projection.ImportFromEPSG(self.ij_epsg)

        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(self.xy_epsg)

        ij_to_xy_transfer = osr.CoordinateTransformation(input_projection, output_projection)
        xy_to_ij_transfer = osr.CoordinateTransformation(output_projection, input_projection)

        # Compute bounding box in source projection
        # ij_x_bbox, ij_y_bbox = self.bounding_box()
        # self.logger.info(f"P_in bounding box: x: {ij_x_bbox} y: {ij_y_bbox}")
        # Re-project bounding box to output projection
        # points_in = np.array([
        #     [ij_x_bbox.min, ij_y_bbox.max],
        #     [ij_x_bbox.max, ij_y_bbox.max],
        #     [ij_x_bbox.max, ij_y_bbox.min],
        #     [ij_x_bbox.min, ij_y_bbox.min]
        # ])
        # TODO: confirm if should use corner cells or bounding polygon
        # Re-project corner cells of the grid to output projection
        points_in = np.array([
            [self.i_limits.min, self.j_limits.max],
            [self.i_limits.max, self.j_limits.max],
            [self.i_limits.max, self.j_limits.min],
            [self.i_limits.min, self.j_limits.min],
        ])
        points_out = ij_to_xy_transfer.TransformPoints(points_in)

        bbox_out_x = Bounds([each[0] for each in points_out])
        bbox_out_y = Bounds([each[1] for each in points_out])

        # Get corresponding bounding box in output projection based on edge points of
        # bounding polygon in P_in projection
        self.x0_bbox, self.y0_bbox = Grid.bounding_box(bbox_out_x, bbox_out_y, self.x_size)
        self.logger.info(f"P_out bounding box:    x: {self.x0_bbox} y: {self.y0_bbox}")

        # Output grid will be used as input to the gdal.warp() and to identify
        # corresponding grid cells in original P_in projection when computing
        # transformation matrix
        self.x0_grid, self.y0_grid = Grid.create(self.x0_bbox, self.y0_bbox, self.x_size)
        self.logger.info(f"Grid in P_out:         num_x={len(self.x0_grid)} num_y={len(self.y0_grid)}")
        self.logger.info(f"Cell centers in P_out: x_min={self.x0_grid[0]} x_max={self.x0_grid[-1]} y_max={self.y0_grid[0]} y_min={self.y0_grid[-1]}")

        xy0_points = ItsLiveReproject.dims_to_grid(self.x0_grid, self.y0_grid)
        # self.logger.info(f"xy0_points: {xy0_points}")

        ij0_points = xy_to_ij_transfer.TransformPoints(xy0_points)

        # Calculate x unit vector: add unit length to ij0_points.x
        ij_x_unit = np.array(ij0_points)
        ij_x_unit[:, 0] += self.x_size
        ij_x_list = ij_x_unit.tolist()
        xy_points = ij_to_xy_transfer.TransformPoints(ij_x_unit.tolist())

        num_xy0_points = len(xy0_points)

        # Compute X unit vector based on xy0_points, xy_points
        # in output projection
        xunit_v = np.zeros((num_xy0_points, 3))

        # Compute unit vector for each cell of the output grid
        for index in range(num_xy0_points):
            diff = np.array(xy_points[index]) - np.array(xy0_points[index])
            xunit_v[index] = diff / np.linalg.norm(diff)

        # Calculate Y unit vector: add unit length to ij0_points.y
        ij_y_unit = np.array(ij0_points)
        ij_y_unit[:, 1] += self.y_size
        xy_points = ij_to_xy_transfer.TransformPoints(ij_y_unit.tolist())

        yunit_v = np.zeros((num_xy0_points, 3))

        # Compute X unit vector based on xy0_points and xy_points
        # in output projection
        for index in range(num_xy0_points):
            diff = np.array(xy_points[index]) - np.array(xy0_points[index])
            yunit_v[index] = diff / np.linalg.norm(diff)

        # Local normal vector
        normal = np.array([0.0, 0.0, 1.0])

        # Compute transformation matrix per cell
        self.transformation_matrix = np.zeros((num_xy0_points), dtype=np.object)
        self.transformation_matrix.fill(DataVars.MISSING_VALUE)

        # Store indices of original cells that correspond to re-projected cells
        self.original_ij_index = np.zeros((num_xy0_points), dtype=np.object)

        # Counter of how many points don't have transformation matrix
        no_value_counter = 0

        # CONTINUE: double check on values
        scale_factor_x = self.x_size/ItsLiveReproject.TIME_DELTA
        scale_factor_y = self.y_size/ItsLiveReproject.TIME_DELTA

        num_i = len(self.ds.x.values)
        num_j = len(self.ds.y.values)

        # For each point on the output grid:
        for each_index in range(num_xy0_points):
            # Find corresponding point in source P_in projection
            ij_point = ij0_points[each_index]

            # Find indices for the original point on its grid
            x_ind = int((ij_point[0] - self.i_limits.min) / self.x_size)
            y_ind = int((ij_point[1] - self.j_limits.max) / self.y_size)

            self.original_ij_index[each_index] = [x_ind, y_ind]

            if (x_ind >= num_i) or (x_ind < 0) or (y_ind >= num_j) or (y_ind < 0):
                no_value_counter += 1
                # self.logger.info('Skipping out of range point')
                continue

            # Check if velocity=NODATA_VALUE for original point -->
            # don't compute the matrix
            v_value = self.ds.v.isel(y=y_ind, x=x_ind).values

            if np.isnan(v_value) or v_value.item() == DataVars.MISSING_VALUE:
                no_value_counter += 1
                continue

            # # Check if the point in P_in projection is within original granule's
            # # X/Y range
            # if ij_point[0] < self.i_limits.min or ij_point[0] > self.i_limits.max or \
            #    ij_point[1] < self.j_limits.min or ij_point[1] > self.j_limits.max:
            #     no_value_counter += 1
            #     self.logger.info('Skipping out of range point')
            #     continue

            # self.logger.info(f"Calculate matrix for v={v_value}")
            xunit = xunit_v[each_index]
            yunit = yunit_v[each_index]

            # Computed normal vector for xunit and yunit at the point
            cross = np.cross(yunit, xunit)
            cross = cross / np.linalg.norm(cross)
            cross_check = np.abs(180.0*np.arccos(np.dot(normal, cross))/np.pi)

            # Allow for angular separation less than 1 degree
            if cross_check > 1.0:
                # self.transformation_matrix[each_index] = DataVars.MISSING_VALUE
                no_value_counter += 1
                self.logger.info(f"No value due to cross check: {cross} for xunit={xunit} yunit={yunit} vs. normal={normal}")

            else:
                # See (A9)-(A15) in Yang's autoRIFT paper:
                a = normal[2]*yunit[0]-normal[0]*yunit[2]
                b = normal[2]*yunit[1]-normal[1]*yunit[2]
                c = normal[2]*xunit[0]-normal[0]*xunit[2]
                d = normal[2]*xunit[1]-normal[1]*xunit[2]
                e = normal[2]*scale_factor_y
                f = normal[2]*scale_factor_x
                div_factor = a*d - b*c
                raster1a = -b*f/div_factor
                raster1b = d*e/div_factor
                raster2a = a*f/div_factor
                raster2b = -c*e/div_factor

                self.transformation_matrix[each_index] = np.array([[raster1a, raster1b], [raster2a, raster2b]])

        # Reshape transformation matrix and original cell indices into 2D matrix: (y, x)
        self.transformation_matrix = self.transformation_matrix.reshape((len(self.y0_grid), len(self.x0_grid)))
        self.original_ij_index = self.original_ij_index.reshape((len(self.y0_grid), len(self.x0_grid)))
        self.logger.info(f"Number of points with no transformation matrix: {no_value_counter} out of {num_xy0_points} points ({no_value_counter/num_xy0_points*100.0}%)")

    def spatial_ref_32x(self):
        """
        Format spatial_ref attribute value for the UTM_Projection.
        """
        epsg = math.floor(self.xy_epsg/100)*100
        zone = self.xy_epsg - epsg
        hemisphere = None
        # We only worry about the following EPSG and zone:
        # 32600 + zone in the northern hemisphere
        # 32700 + zone in the southern hemisphere
        if epsg == 32700:
            hemisphere = 'S'

        elif epsg == 32600:
            hemisphere = 'N'

        else:
            raise RuntimeError(f"Unsupported target projection {self.xy_epsg} is provided.")

        return zone, f"PROJCS[\"WGS 84 / UTM zone {zone}{hemisphere}\"," \
            "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\"," \
            "6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]]," \
            "AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0," \
            "AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433," \
            "AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]," \
            "PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0]," \
            f"PARAMETER[\"central_meridian\",{_epsg_to_central_meridian[zone]}]," \
            "PARAMETER[\"scale_factor\",0.9996]," \
            "PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0]," \
            "UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]," \
            "AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH]," \
            f"AUTHORITY[\"EPSG\",\"{self.xy_epsg}\"]]"

    @staticmethod
    def dims_to_grid(x, y):
        """
        Convert x, y dimensions of the dataset into numpy grid array in (y, x) order.
        """
        # Use z=0 as osr.CoordinateTransformation.TransformPoints() returns 3d point coordinates
        grid = np.zeros((len(x)*len(y), 3))

        num_row = 0
        for each_y in y:
            for each_x in x:
                grid[num_row][0] = each_x
                grid[num_row][1] = each_y
                num_row += 1

        return grid


if __name__ == '__main__':
    """
    Re-project ITS_LIVE granule to the target projection.
    """
    parser = argparse.ArgumentParser(description='Re-project ITS_LIVE granule to new projection.')
    parser.add_argument(
        '-i', '--input',
        dest='input_file',
        type=str,
        required=True,
        help='Input file name for ITS_LIVE granule')
    parser.add_argument(
        '-p', '--projection',
        dest='output_proj',
        type=int,
        required=True, help='Output projection')
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        type=str,
        default=None,
        required=False, help='Output filename to store re-projected granule in target projection')

    command_args = parser.parse_args()

    its_data = ItsLiveReproject(command_args.input_file, command_args.output_proj)
    its_data.run(command_args.output_file)
