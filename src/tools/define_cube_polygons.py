"""
Define datacube polygons for production.

References:
* To handle crossing of the antimeridian:
    https://towardsdatascience.com/around-the-world-in-80-lines-crossing-the-antimeridian-with-python-and-shapely-c87c9b6e1513
"""
import geopandas as gpd
import json
import geojson
import logging
import math
import numpy as np
import os
from osgeo import gdal
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import Polygon, LineString, GeometryCollection, mapping, MultiPolygon
from shapely.ops import split
from tqdm import tqdm

# import s3fs

import itslive_utils
from grid import Bounds

# GDAL settings
gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'tif')
gdal.SetConfigOption('VSI_CACHE', 'TRUE')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'TRUE')

LON_LAT_PROJECTION = 'EPSG:4326'

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

def translate_polygons(geometry_collection: GeometryCollection) -> list:
    """
    Translate Cartesian polygons back to geospacial coordinates
    """
    polygons = []
    for polygon in geometry_collection:
        (minx, _, maxx, _) = polygon.bounds
        if minx < -180:
            geo_polygon = affinity.translate(polygon, xoff = 360)

        elif maxx > 180:
            geo_polygon = affinity.translate(polygon, xoff = -360)

        else: geo_polygon = polygon

        polygons.append(geo_polygon)

    return polygons

def define_cubes(shape_filename: str, cube_filename: str, target_epsg_codes: list = None, grid_size: int = 100000):
    """
    Function to define ITS_LIVE data cubes based on provided shape file and grid
    size in meters.
    """
    # TODO: Read the file from s3 bucket
    # if shape_filename.startswith('s3'):
    #     s3 = s3fs.S3FileSystem(anon=True)

    shapefile = gpd.read_file(shape_filename)
    # print("Shape: ", shapefile.columns)

    features = []

    for index, each_row in shapefile.iterrows():
        # Process each polygon from the shapefile
        epsg_code = int(each_row.epsg)

        if target_epsg_codes is not None and epsg_code not in target_epsg_codes:
            continue

        # Format the code for the pyproj.Transformer
        epsg_code = f"EPSG:{epsg_code}"
        logging.info(f"Processing {epsg_code}")

        # Read ROI data in
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
            logging.info(f"{epsg_code}'s ROI {roi_file} has no data")

        # GeoTransform: (-32647.5, 120.0, 0.0, 10199047.5, 0.0, -120.0)
        roi_geo_trans = roi_ds.GetGeoTransform()
        roi_xsize = roi_ds.GetRasterBand(1).XSize
        roi_ysize = roi_ds.GetRasterBand(1).YSize

        # logging.info(f"ROI data.shape: {roi_data.shape}")
        logging.info(f"ROI.GetGeoTransform: {roi_ds.GetGeoTransform()}")
        logging.info(f"ROI.XSize: {roi_ds.GetRasterBand(1).XSize} ROI.YSize: {roi_ds.GetRasterBand(1).YSize}")

        # Get the cube polygon size in pixels
        cube_num_cells = int(grid_size/roi_ds.GetGeoTransform()[1])**2
        logging.info(f"Number of cube cells: {cube_num_cells}")

        roi_x = np.array(np.arange(roi_geo_trans[0], roi_geo_trans[0] + roi_geo_trans[1]*roi_xsize, roi_geo_trans[1]))
        roi_y = np.array(np.arange(roi_geo_trans[3], roi_geo_trans[3] + roi_geo_trans[5]*roi_ysize, roi_geo_trans[5]))
        # logging.info(f"ROI x: {roi_x.min()} {roi_x.max()}")
        # logging.info(f"ROI y: {roi_y.min()} {roi_y.max()}")

        assert len(roi_x) == roi_xsize, f"Inconsistent x dimension size: {len(roi_x)} vs. {roi_xsize}"
        assert len(roi_y) == roi_ysize, f"Inconsistent y dimension size: {len(roi_y)} vs. {roi_ysize}"

        envelope = shapefile.geometry[index]

        # For the polar polygons, where opposite edge vertices
        # (for example,
        #   (-179.999; 55) re-projected to (-2764198.4528116877, 2764198.35632296)
        #   ( 179.999; 55) re-projected to (-2764198.3563229595, 2764198.452811688)
        # )
        # are projected almost to the same x/y values in UTM coordinates,
        # need to split the polygon along center meridian (long=0) to get
        # full area coverage. Otherwise, we are getting reduced coverage in
        # UTM projection.
        split_polygons = [Polygon(envelope)]
        if epsg_code in ['3413', '3031']:
            split_meridian = 0
            meridian = [[split_meridian, -90.0], [split_meridian, 90.0]]
            splitter = LineString(meridian)
            split_polygons = split(Polygon(envelope), splitter)
            logging.info(f"Number of split polygons: {len(split_polygons)}")

        # Process each polygon (if it was split, otherwise there is only one
        # polygon)
        transformer = Transformer.from_crs(LON_LAT_PROJECTION, epsg_code, always_xy=True)
        back_transformer = Transformer.from_crs(epsg_code, LON_LAT_PROJECTION, always_xy=True)

        for each_polygon in split_polygons:
            # Convert polygon to target EPSG coords
            epsg_polygon = []
            lon, lat = map(np.array, each_polygon.exterior.coords.xy)
            logging.info(f"Polygon lon: {lon.shape} lat: {lat.shape}")
            # This is to prevent corner cases when
            # all original coordinates at latitude=+-90 or longitude=+-180 get
            # projected to the same x/y=(0;0) in target EPSG
            lon[np.isclose(lon, -180.0)] = -179.999999
            lon[np.isclose(lon, 180.0)] = 179.999999
            lat[np.isclose(lat, -90.0)] = -89.999999
            lat[np.isclose(lat, 90.0)] = 89.999999

            # Convert coordinates
            # epsg_polygon = [pyproj.transform(proj1, proj2, e_lon, e_lat) for e_lon, e_lat in zip(lon, lat)]
            epsg_polygon = [transformer.transform(e_lon, e_lat) for e_lon, e_lat in zip(lon, lat)]
            logging.info(f"EPGS region: {len(epsg_polygon)}")

            # # Convert back to lon/lat to debug coverage
            # debug_lonlat = []
            # for each_x, each_y in epsg_polygon:
            #     debug_lonlat.append(back_transformer.transform(each_x, each_y))
            #     print(f"Converted x/y: {each_x}; {each_y} to lon/lat: {debug_lonlat[-1]}")
            #
            # logging.info(f"EPGS region lon/lat debug: {debug_lonlat}")

            # Find bounding box for the polygon in EPSG regional code
            x_bounds = Bounds([each[0] for each in epsg_polygon])
            y_bounds = Bounds([each[1] for each in epsg_polygon])

            logging.info(f"EPGS region: x: {x_bounds} y: {y_bounds}")

            # Round up max and round down min values of bounds
            new_x = x_bounds.extend_to_grid(grid_size)
            new_y = y_bounds.extend_to_grid(grid_size)

            logging.info(f"EPGS region expanded to {grid_size}m: x: {new_x} y: {new_y}")

            # Define grid for the data cubes
            x_range = list(range(new_x.min,  new_x.max, grid_size))
            y_range = list(range(new_y.min,  new_y.max, grid_size))

            # Create x/y ranges for each of the datacubes
            for each_x in tqdm(range(new_x.min, new_x.max, grid_size), ascii=True, desc="Processing X axis..."):
                for each_y in tqdm(range(new_y.min, new_y.max, grid_size), ascii=True, desc="Processing Y axis..."):
                    cube_x_min, cube_x_max = each_x, each_x + grid_size
                    cube_y_min, cube_y_max = each_y, each_y + grid_size

                    # Find ROI x and y indices that correspond to the cube polygon:
                    sel = np.where((roi_x >= cube_x_min) & (roi_x < cube_x_max))
                    # print(f"Selection x: {len(sel[0])}")
                    # Inclusive indices of ROI X (pass one for the last index)
                    min_x_ind, max_x_ind = sel[0][0], sel[0][-1]+1

                    sel = np.where((roi_y >= cube_y_min) & (roi_y < cube_y_max))
                    # Inclusive indices of ROI Y (pass one for the last index)
                    min_y_ind, max_y_ind = sel[0][0], sel[0][-1]+1

                    # Convert cube polygon to lon/lat coordinates
                    lonlat_coords = []
                    for each_x, each_y in [
                        (cube_x_min, cube_y_min),
                        (cube_x_max, cube_y_min),
                        (cube_x_max, cube_y_max),
                        (cube_x_min, cube_y_max),
                        (cube_x_min, cube_y_min)]:
                        lonlat_coords.append(list(back_transformer.transform(each_x, each_y)))

                    # Initialize min/max of the split line
                    minx = maxx = lonlat_coords[0][0]
                    crosses_antimeridian = False
                    split_meridian = None

                    for coord_index, (lon, _) in enumerate(lonlat_coords[1:], start=1):
                        lon_prev, _ = lonlat_coords[coord_index - 1]

                        # Assuming a minimum travel distance between two provided longitude coordinates,
                        # checks if the 180th meridian (antimeridian) is crossed.
                        delta_lon = lon - lon_prev
                        if abs(delta_lon) > 180.0:
                            # Shift current longitude if antimeridian cross occurs
                            direction = math.copysign(1, delta_lon)

                            cross_shift = direction * 360.0
                            lonlat_coords[coord_index][0] = lon - cross_shift
                            crosses_antimeridian = True

                        x_shift = lonlat_coords[coord_index][0]
                        if x_shift < minx: minx = x_shift
                        if x_shift > maxx: maxx = x_shift

                    geometry_obj = Polygon(lonlat_coords)
                    if crosses_antimeridian:
                        # Define meridian to split on
                        split_meridian = -180 if minx < -180 else 180

                        meridian = [[split_meridian, -90.0], [split_meridian, 90.0]]
                        splitter = LineString(meridian)
                        split_polygons = split(Polygon(lonlat_coords), splitter)

                        geo_polygons = translate_polygons(split_polygons)

                        # Check if cube overlaps original polygon in lon/lat coordinates
                        keep_polygons = []
                        for each in geo_polygons:
                            if Polygon(each).intersects(each_polygon):
                                keep_polygons.append(each)

                        if len(keep_polygons) == 0:
                            # None of the cube polygons intersect original polygon
                            geometry_obj = None

                        elif len(keep_polygons) == 1:
                            geometry_obj = Polygon(keep_polygons[0])

                        else:
                            geometry_obj = MultiPolygon(keep_polygons)

                    else:
                        if not geometry_obj.intersects(each_polygon):
                            geometry_obj = None

                    if geometry_obj is None:
                        # There is no valid cube to record
                        continue

                    # Region Of Interest coverage within the cube
                    roi_coverage = roi_data[min_y_ind:max_y_ind, min_x_ind:max_x_ind].sum()/cube_num_cells

                    # For each cube capture lon/lat coordinates as geometry
                    # and UTM coordinates as a property to be accessed "manually"
                    features.append(
                        geojson.Feature(
                            geometry=geometry_obj,
                            properties={
                                # "stroke-width": 2,
                                # "stroke-opacity": 1,
                                "fill-opacity": 1.0 - roi_coverage,
                                "fill": "red",
                                'roi_percent_coverage': roi_coverage*100,
                                'data_epsg':    epsg_code,
                                'geometry_epsg': geojson.Polygon([[
                                    [cube_x_min, cube_y_min],
                                    [cube_x_max, cube_y_min],
                                    [cube_x_max, cube_y_max],
                                    [cube_x_min, cube_y_max],
                                    [cube_x_min, cube_y_min]]]),
                            }
                            # Adding "style" does not make opacity work in QGIS either:
                            # style = {
                            #     "opacity": 1.0 - roi_coverage,
                            #     # "fill-opacity": 1.0 - roi_coverage,
                            #     # "opacity": 1.0 - roi_coverage,
                            #     "fill": "green" if roi_coverage > 0.5 else "yellow",
                            #     # "fill": "green" if roi_coverage > 0.5 else "yellow",
                            #
                            # }
                        )
                    )

    feature_collection = geojson.FeatureCollection(features)

    # Write cube definitions to the file
    with open(cube_filename, 'w+') as fhandle:
        json.dump(feature_collection, fhandle, indent=4)

    features = None

if __name__ == '__main__':
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--shapeFile', type=str, default='/Users/mliukis/Documents/ITS_LIVE/fromAlex/autorift_landice_0120m/autorift_landice_0120m.shp',
                        help='Regional shape file that defines each of the EPSG polygons.')
    parser.add_argument('-o', '--outputFile', type=str, default='cubeGrid.json',
                        help='Geojson file to store cube polygon definitions.')
    parser.add_argument('-c', '--epsgCode', type=str, default=None,
                        help='EPGS code(s) as json list to create datacube grid for. Default is None meaning to generate complete datacube grid.')

    args = parser.parse_args()

    # Map provided EPSG codes to the list of int codes
    epsg_codes = list(map(int, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    logging.info(f"Got EPGS codes: {epsg_codes}")

    define_cubes(args.shapeFile, args.outputFile, epsg_codes)
