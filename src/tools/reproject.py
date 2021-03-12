"""
Reprojection tool for ITS_LIVE granule data to new target projection.

Examples:
$ python reproject.py -i input_filename -p target_projection -o output_filename

    Reproject "input_filename" into 'target_projection' and output new granule into
'output_filename' in NetCDF format.
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
    9. Re-project v* values: gdal.warp(original_granule, P_out_grid) --> P_out_v
       Apply tranformation matrix to P_out_v per cell to get "true" v value
    """
    NODATA_VALUE = -32767

    # Number of seconds in one day: any period would do as long as it's
    # the same time period used to convert v(elocity) to d(istance), and
    # then use the same value to compute transformation matrix
    TIME_DELTA = 24 * 3600

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

        # Input and output projections
        self.ij_epsg = int(self.ds.UTM_Projection.spatial_epsg)
        self.xy_epsg = output_projection

        if self.ij_epsg == self.xy_epsg:
            raise RuntimeError("Done: original data is in the target {self.xy_epsg} projection already.")

        self.logger.info(f"Reprojecting from {self.ij_epsg} to {self.xy_epsg}")

        # Grid spacing
        self.XSize = self.ds.x.values[1] - self.ds.x.values[0]
        self.YSize = self.ds.y.values[1] - self.ds.y.values[0]

        # Compute bounding box in source projection
        self.ij_x_bbox, self.ij_y_bbox = ItsLiveReproject.bounding_box(
            self.ds,
            self.XSize,
            self.YSize
        )
        self.logger.info(f"P_in bounding box: x: {self.ij_x_bbox} y: {self.ij_y_bbox}")

        # Placeholders for:
        # bounding box in output projection
        self.x0_bbox = None
        self.y0_bbox = None

        # grid coordinates in output projection
        self.x0_grid = None
        self.y0_grid = None

        # Transformation matrix to rotate warped velocity components (vx* and vy*)
        # in output projection
        self.transformation_matrix = None

    @staticmethod
    def bounding_box(ds, dx, dy):
        """
        Select bounding box for the dataset.
        """
        # ATTN: Assuming that X and Y cell dimensions are the same
        assert np.abs(dx) == np.abs(dy), f"Cell dimensions differ: x={np.abs(dx)} y={np.abs(dy)}"

        center_off_X = dx/2
        center_off_Y = dy/2

        # Compute cell boundaries as ITS_LIVE grid stores x/y for the cell centers
        xmin = ds.x.values.min() - center_off_X
        xmax = ds.x.values.max() + center_off_X

        # Y coordinate calculations are based on the fact that dy < 0
        ymin = ds.y.values.min() + center_off_Y
        ymax = ds.y.values.max() - center_off_Y

        return Grid.bounding_box(
            Bounds(min_value=xmin, max_value=xmax),
            Bounds(min_value=ymin, max_value=ymax),
            dx
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
            outputBounds=(self.x0_bbox.min, self.y0_bbox.max, self.x0_bbox.max, self.y0_bbox.min),
            xRes=self.XSize,
            yRes=self.YSize,
            srcSRS=f'EPSG:{self.ij_epsg}',
            dstSRS=f'EPSG:{self.xy_epsg}',
            resampleAlg=gdal.GRA_NearestNeighbour
        )

        # Compute new vx, vy and v
        vx, vy, v = self.reproject_velocity(DataVars.VX, DataVars.VY, warp_options)

        # Create new granule in target projection
        reproject_ds = xr.Dataset(
            data_vars={
                DataVars.VX: xr.DataArray(
                    data=vx,
                    coords=[self.y0_grid, self.x0_grid],
                    dims=[Coords.Y, Coords.X],
                    attrs=self.ds[DataVars.VX].attrs),
                DataVars.VY: xr.DataArray(
                    data=vy,
                    coords=[self.y0_grid, self.x0_grid],
                    dims=[Coords.Y, Coords.X],
                    attrs=self.ds[DataVars.VY].attrs),
                DataVars.V: xr.DataArray(
                    data=v,
                    coords=[self.y0_grid, self.x0_grid],
                    dims=[Coords.Y, Coords.X],
                    attrs=self.ds[DataVars.V].attrs),
            },
            coords={
                Coords.X: self.x0_grid,
                Coords.Y: self.y0_grid
            },
            attrs=self.ds.attrs
        )

        # Use garbage collection
        vx = None
        vy = None
        v = None
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
                    'spatial_proj4': "+proj=utm +zone=28 +datum=WGS84 +units=m +no_defs"
                }
            )

        proj_data.attrs['spatial_epsg'] = self.xy_epsg
        # Format GeoTransform:
        # x top left (cell left most boundary), grid size, 0, y top left (cell upper most boundary), 0, -grid size
        half_x_cell = self.XSize/2.0
        half_y_cell = self.YSize/2.0
        proj_data.attrs['GeoTransform'] = f"{self.x0_grid[0] - half_x_cell} {self.XSize} 0 {self.y0_grid[0] - half_y_cell} 0 {self.YSize}"
        reproject_ds[proj_name] = proj_data

        # Set grid_mapping for vx, vy:
        reproject_ds[DataVars.VX].attrs[DataVars.GRID_MAPPING] = proj_name
        reproject_ds[DataVars.VY].attrs[DataVars.GRID_MAPPING] = proj_name

        if DataVars.VP in self.ds:
            # This is Radar format, projected velocity is provided
            vxp, vyp, vp = self.reproject_velocity(DataVars.VXP, DataVars.VYP, warp_options)

            reproject_ds[DataVars.VXP] = xr.DataArray(
                data=vxp,
                coords=[self.y0_grid, self.x0_grid],
                dims=[Coords.Y, Coords.X],
                attrs=self.ds[DataVars.VXP].attrs
            )
            reproject_ds[DataVars.VXP].attrs[DataVars.GRID_MAPPING] = proj_name

            reproject_ds[DataVars.VYP] = xr.DataArray(
                data=vyp,
                coords=[self.y0_grid, self.x0_grid],
                dims=[Coords.Y, Coords.X],
                attrs=self.ds[DataVars.VYP].attrs
            )
            reproject_ds[DataVars.VYP].attrs[DataVars.GRID_MAPPING] = proj_name

            reproject_ds[DataVars.VP] = xr.DataArray(
                data=vp,
                coords=[self.y0_grid, self.x0_grid],
                dims=[Coords.Y, Coords.X],
                attrs=self.ds[DataVars.VP].attrs
            )

            vxp = None
            vyp = None
            vp = None
            gc.collect()

            # Process VA
            # TODO

        reproject_ds[DataVars.CHIP_SIZE_HEIGHT] = xr.DataArray(
            data=self.warp_var(DataVars.CHIP_SIZE_HEIGHT, warp_options),
            coords=[self.y0_grid, self.x0_grid],
            dims=[Coords.Y, Coords.X],
            attrs=self.ds[DataVars.CHIP_SIZE_HEIGHT].attrs
        )
        reproject_ds[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.GRID_MAPPING] = proj_name

        reproject_ds[DataVars.CHIP_SIZE_WIDTH] = xr.DataArray(
            data=self.warp_var(DataVars.CHIP_SIZE_WIDTH, warp_options),
            coords=[self.y0_grid, self.x0_grid],
            dims=[Coords.Y, Coords.X],
            attrs=self.ds[DataVars.CHIP_SIZE_WIDTH].attrs
        )
        reproject_ds[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.GRID_MAPPING] = proj_name

        reproject_ds[DataVars.INTERP_MASK] = xr.DataArray(
            data=self.warp_var(DataVars.INTERP_MASK, warp_options),
            coords=[self.y0_grid, self.x0_grid],
            dims=[Coords.Y, Coords.X],
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
            encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE}
            encoding_settings[each].update(compression)

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

        print(f"Using encodings: {encoding_settings}")

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


    def reproject_velocity(self, vx: str, vy: str, warp_options: gdal.WarpOptions):
        """
        Re-project velocity X and Y components and compute velocity magnitude.
        """
        dataset = gdal.Open(f'NETCDF:"{self.input_file}":{vx}')

        # Warp data variable
        vx_ds = gdal.Warp('', dataset, options=warp_options)
        np_vx = vx_ds.ReadAsArray()
        self.logger.info(f"Read with GDAL {vx}.shape = {np_vx.shape}")
        del dataset

        dataset = gdal.Open(f'NETCDF:"{self.input_file}":{vy}')
        # Warp data variable
        vy_ds = gdal.Warp('', dataset, options=warp_options)
        np_vy = vy_ds.ReadAsArray()
        self.logger.info(f"Read with GDAL {vy}.shape = {np_vy.shape}")

        # Transpose np_ds as it's in (y, x) order
        # np_vx = np_vx.transpose()
        # np_vy = np_vy.transpose()
        # self.logger.info(f"Transpose np_vx: {np_vx.shape}")
        # self.logger.info(f"Transpose np_vy: {np_vy.shape}")
        #
        # Convert velocity value to distance (per transformation matrix requirement)
        np_vx *= ItsLiveReproject.TIME_DELTA
        np_vy *= ItsLiveReproject.TIME_DELTA

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)

        vx = np.zeros((num_y, num_x))
        vy = np.zeros((num_y, num_x))
        v = np.zeros((num_y, num_x))

        # Transform each (dx, dy) to (vx, vy) in output projection
        for y_index in range(num_y):
            for x_index in range(num_x):
                dv = np.array([
                    np_vx[y_index, x_index],
                    np_vy[y_index, x_index]
                ])

                if np.isscalar(self.transformation_matrix[y_index, x_index]):
                    # There is no transformation matrix available for the point -->
                    # NODATA
                    vx[y_index, x_index] = ItsLiveReproject.NODATA_VALUE
                    vy[y_index, x_index] = ItsLiveReproject.NODATA_VALUE

                else:
                    # Apply transformation matrix to (vx, vy) values converted to distance
                    xy_v = np.matmul(self.transformation_matrix[y_index, x_index], dv)
                    vx[y_index, x_index] = xy_v[0]
                    vy[y_index, x_index] = xy_v[1]

                    # Compute v: sqrt(vx^2 + vy^2)
                    v[y_index, x_index] = np.sqrt(xy_v[0]**2 + xy_v[1]**2)

        return (vx, vy, v)

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

        # Re-project bounding box to output projection
        points_in = np.array([
            [self.ij_x_bbox.min, self.ij_y_bbox.max],
            [self.ij_x_bbox.max, self.ij_y_bbox.max],
            [self.ij_x_bbox.max, self.ij_y_bbox.min],
            [self.ij_x_bbox.min, self.ij_y_bbox.min]
        ])
        points_out = ij_to_xy_transfer.TransformPoints(points_in)

        bbox_out_x = Bounds([each[0] for each in points_out])
        bbox_out_y = Bounds([each[1] for each in points_out])

        # Get corresponding bounding box in output projection based on edge points of
        # bounding polygon in P_in projection
        self.x0_bbox, self.y0_bbox = Grid.bounding_box(bbox_out_x, bbox_out_y, self.XSize)
        self.logger.info(f"P_out bounding box: x: {self.x0_bbox} y: {self.y0_bbox}")

        # Output grid will be used as input to the gdal.warp() and to identify
        # corresponding grid cells in original P_in projection when computing
        # transformation matrix
        self.x0_grid, self.y0_grid = Grid.create(self.x0_bbox, self.y0_bbox, self.XSize)
        self.logger.info(f"Grid in P_out: num_x={len(self.x0_grid)} num_y={len(self.y0_grid)}")

        xy0_points = ItsLiveReproject.dims_to_grid(self.x0_grid, self.y0_grid)
        ij0_points = xy_to_ij_transfer.TransformPoints(xy0_points)
        # self.logger.info(f"Len of (x0, y0) points in P_out: {xy0_points.shape}")
        # self.logger.info(f"Len of (i0, j0) points in P_in:  {len(ij0_points)}")

        # Calculate x unit vector: add unit length to ij0_points.x
        ij_x_unit = np.array(ij0_points.copy())
        ij_x_unit[:, 0] += self.XSize
        xy_points = ij_to_xy_transfer.TransformPoints(ij_x_unit.tolist())
        # x1 = [each[0] for each in points_out]
        # y1 = [each[1] for each in points_out]

        num_xy0_points = len(xy0_points)

        # Compute X unit vector based on xy0_points, xy_points
        # in output projection
        xunit_v = np.zeros((num_xy0_points, 3))
        # Compute unit vector for each cell of the output grid
        for index in range(num_xy0_points):
            diff = np.array(xy_points[index]) - np.array(xy0_points[index])
            xunit_v[index] = diff / np.linalg.norm(diff)

        # Calculate Y unit vector: add unit length to ij0_points.y
        ij_y_unit = np.array(ij0_points.copy())
        ij_y_unit[:, 1] += self.YSize
        xy_points = ij_to_xy_transfer.TransformPoints(ij_y_unit.tolist())

        yunit_v = np.zeros((num_xy0_points, 3))

        # Compute X unit vector based on xy0_points and xy_points
        # in output projection
        for index in range(num_xy0_points):
            diff = np.array(xy_points[index]) - np.array(xy0_points[index])
            yunit_v[index] = diff / np.linalg.norm(diff)

        print("x_unit[0]: ", xunit_v[0])
        print("y_unit[0]: ", yunit_v[0])

        # Local normal vector
        normal = np.array([0.0, 0.0, -1.0])

        # Compute transformation matrix per cell
        self.transformation_matrix = np.zeros((num_xy0_points), dtype=np.object)

        # Counter of how many points don't have transformation matrix
        no_value_counter = 0

        # For each point on the output grid:
        for each_index in range(num_xy0_points):
            # Find corresponding point in P_in projection
            ij_point = ij0_points[each_index]
            xunit = xunit_v[each_index]
            yunit = yunit_v[each_index]

            # Check if the point in P_in projection is within original granule's
            # X/Y range
            if ij_point[0] < self.ij_x_bbox.min or ij_point[0] > self.ij_x_bbox.max or \
               ij_point[1] < self.ij_y_bbox.min or ij_point[1] > self.ij_y_bbox.max:
                self.transformation_matrix[each_index] = ItsLiveReproject.NODATA_VALUE
                no_value_counter += 1
                continue

            # Computed normal vector for xunit and yunit at the point
            cross = np.cross(xunit, yunit)
            cross = cross / np.linalg.norm(cross)
            cross_check = np.abs(180.0*np.arccos(np.dot(normal, cross))/np.pi)

            # Allow for angular separation less than 1 degree
            if cross_check > 1.0:
                self.transformation_matrix[each_index] = ItsLiveReproject.NODATA_VALUE
                no_value_counter += 1
                self.logger.info(f"No value due to cross check: {cross} for xunit={xunit} yunit={yunit} vs. normal={normal}")

            else:
                raster1a = normal[2]/(ItsLiveReproject.TIME_DELTA/self.XSize/365.0/24.0/3600.0)*(normal[2]*yunit[1]-normal[1]*yunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]))
                raster1b = -normal[2]/(ItsLiveReproject.TIME_DELTA/self.YSize/365.0/24.0/3600.0)*(normal[2]*xunit[1]-normal[1]*xunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]))
                raster2a = -normal[2]/(ItsLiveReproject.TIME_DELTA/self.XSize/365.0/24.0/3600.0)*(normal[2]*yunit[0]-normal[0]*yunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                raster2b = normal[2]/(ItsLiveReproject.TIME_DELTA/self.YSize/365.0/24.0/3600.0)*(normal[2]*xunit[0]-normal[0]*xunit[2])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));

                self.transformation_matrix[each_index] = np.array([[raster1a, raster1b], [raster2a, raster2b]])
                # self.logger.info(f"Got M: {self.transformation_matrix[each_index].shape}")

        # Reshape transformation matrix into 2D matrix: (x, y)
        self.transformation_matrix = self.transformation_matrix.reshape((len(self.y0_grid), len(self.x0_grid)))
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
            "PARAMETER[\"central_meridian\",-15],PARAMETER[\"scale_factor\",0.9996]," \
            "PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0]," \
            "UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]," \
            "AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"{self.xy_epsg}\"]]"

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