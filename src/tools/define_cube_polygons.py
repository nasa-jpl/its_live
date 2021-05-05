"""
Script to define polygons for the datacube production.
"""
import geopandas as gpd
import json
import logging
import numpy as np
import os
from osgeo import gdal

# import s3fs

import itslive_utils
from grid import Bounds

# GDAL settings
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'tif')
gdal.SetConfigOption('VSI_CACHE', 'TRUE')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')

LON_LAT_PROJECTION = '4326'

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("define_datacube")

def define_cubes(shapefile: str, grid_size: int = 100000, cube_dir: str = 'datacube_definitions'):
    """
    Function to define ITS_LIVE data cubes based on provided shape file and grid
    size in meters.
    """
    # File is in s3 bucket
    # if shapefile.startswith('s3'):
    #     s3 = s3fs.S3FileSystem(anon=True)

    shapefile = gpd.read_file('/Users/mliukis/Documents/ITS_LIVE/fromAlex/autorift_landice_0120m/autorift_landice_0120m.shp')
    # print("Shape: ", shapefile.columns)


    for index, each_row in shapefile.iterrows():
        # Process each polygon from the shapefile
        epsg_code = each_row.epsg
        logger.info(f"Processing {epsg_code}")

        roi_file = each_row.ROI

        # If ROI does not have any data, skip the whole polygon as there are
        # no cubes to define
        roi_ds = gdal.Open(f'/vsicurl/{roi_file}')
        # ROI data is stored in [Y; X] order:
        # Type ROI_array:  <class 'numpy.ndarray'>
        # ROI data.shape: (55072, 8880)
        # ROI.GetGeoTransform: (-32647.5, 120.0, 0.0, 10199047.5, 0.0, -120.0)
        # ROI.XSize: 8880 ROI.YSize: 55072
        roi_data = roi_ds.GetRasterBand(1).ReadAsArray()

        if roi_data.sum() == 0:
            logger.info(f"Skipping {epsg_code} due to no data in ROI {roi_file}")
            continue

        envelope = shapefile.envelope[index]

        # Convert polygon to EPSG coords
        epsg_polygon = []

        for each_lon, each_lat in envelope.exterior.coords:
            # print(f"Each lon/lat: {each_lon} {each_lat}")
            epsg_polygon.append(itslive_utils.transform_coord(LON_LAT_PROJECTION, str(epsg_code), each_lon, each_lat))

        # print(f"epsg_code={epsg_code}: {os.linesep.join(str(each) for each in epsg_polygon)}")

        # Find bounding box for the polygon in EPSG regional code
        x_bounds = Bounds([each[0] for each in epsg_polygon])
        y_bounds = Bounds([each[1] for each in epsg_polygon])

        # Round up max and round down min values of bounds
        new_x = x_bounds.extend_to_grid(grid_size)
        new_y = y_bounds.extend_to_grid(grid_size)

        # print("After extends:")
        # str(new_x)
        # str(new_y)

        # Define grid for the data cubes
        x_range = list(range(new_x.min,  new_x.max, grid_size))
        y_range = list(range(new_y.min,  new_y.max, grid_size))

        # Capture longitude and latitude for original polygon
        lon, lat = envelope.exterior.coords.xy
        cube_definition = {'epsg': int(epsg_code),
                           'cubes_x_y': [],
                           'epsg_polygon': epsg_polygon,
                           'cube_size_m': grid_size,
                           'lon_lat_polygon': list(zip(lon, lat)),
                           'num_cubes': 0}

        # GeoTransform: (-32647.5, 120.0, 0.0, 10199047.5, 0.0, -120.0)
        roi_geo_trans = roi_ds.GetGeoTransform()
        roi_xsize = roi_ds.GetRasterBand(1).XSize
        roi_ysize = roi_ds.GetRasterBand(1).YSize

        print(f"ROI data.shape: {roi_data.shape}")
        print(f"ROI.GetGeoTransform: {roi_ds.GetGeoTransform()}")
        print(f"ROI.XSize: {roi_ds.GetRasterBand(1).XSize} ROI.YSize: {roi_ds.GetRasterBand(1).YSize}")

        roi_x = np.array(np.arange(roi_geo_trans[0], roi_geo_trans[0] + roi_geo_trans[1]*roi_xsize, roi_geo_trans[1]))
        roi_y = np.array(np.arange(roi_geo_trans[3], roi_geo_trans[3] + roi_geo_trans[5]*roi_ysize, roi_geo_trans[5]))
        print(f"ROI x: {roi_x.min()} {roi_x.max()}")
        print(f"ROI y: {roi_y.min()} {roi_y.max()}")

        assert len(roi_x) == roi_xsize, f"Inconsistent x dimension size: {len(roi_x)} vs. {roi_xsize}"
        assert len(roi_y) == roi_ysize, f"Inconsistent y dimension size: {len(roi_y)} vs. {roi_ysize}"

        # Create x/y ranges for each of the datacubes
        for each_x in range(new_x.min, new_x.max, grid_size):
            for each_y in range(new_y.min, new_y.max, grid_size):
                cube_x_min, cube_x_max = each_x, each_x + grid_size
                cube_y_min, cube_y_max = each_y, each_y + grid_size

                # Find ROI x and y indices that correspond to the cube polygon:
                sel = np.where((roi_x >= cube_x_min) & (roi_x < cube_x_max))
                # print(f"Selection x: {len(sel[0])}")
                # Inclusive indices of ROI X (pass one for the last index)
                min_x_ind, max_x_ind = sel[0][0], sel[0][-1]+1
                # print(f"Selection x: num={len(sel[0])}: min={min_x_ind} max={max_x_ind}")

                sel = np.where((roi_y >= cube_y_min) & (roi_y < cube_y_max))
                # Inclusive indices of ROI Y (pass one for the last index)
                min_y_ind, max_y_ind = sel[0][0], sel[0][-1]+1
                # print(f"Selection y: num={len(sel[0])}: min={min_y_ind} max={max_y_ind}")

                # TODO: Apply ROI for each of the cubes to see if it contains any ROI pixels
                if roi_data[min_y_ind:max_y_ind, min_x_ind:max_x_ind].sum() == 0:
                    logger.info(f"Skipping cube x=[{cube_x_min}, {cube_x_max}] y=[{cube_y_min}, {cube_y_max}] due to no ROI data")

                else:
                    # TODO: Should "revert" y values: max, min
                    cube_definition['cubes_x_y'].append([(cube_x_min, cube_x_max), (cube_y_min, cube_y_max)])

        cube_definition['num_cubes'] = len(cube_definition['cubes_x_y'])

        if cube_definition['num_cubes'] == 0:
            # There were no cubes with ROI
            logger.info(f"No cubes defined for {epsg_code} due to no data in ROI {roi_file}")

        # Write cube definitions to the file
        with open(f'{cube_dir}/cubeGrid_{epsg_code}.json', 'w+') as fhandle:
            json.dump(cube_definition, fhandle, indent=4, sort_keys=True)


if __name__ == '__main__':
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--shapeFile', type=str, default='/Users/mliukis/Documents/ITS_LIVE/fromAlex/autorift_landice_0120m/autorift_landice_0120m.shp',
                        help='Regional shape file that defines each of the EPSG polygons.')
    args = parser.parse_args()

    define_cubes(args.shapeFile)
