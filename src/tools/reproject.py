"""
Reprojection tool for ITS_LIVE granules.
"""

import argparse
import logging
import numpy as np
from osgeo import osr
import xarray as xr

from grid import Grid, Bounds


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)


class ItsLiveReproject:
    """
    Class to store input ITS_LIVE granule, and functionality to re-project
    its data into a new target projection.
    """
    def __init__(self, data):
        """
        Initialize object.
        """
        self.logger = logging.getLogger("ItsLiveReproject")

        self.ds = data
        if isinstance(data, str):
            # Filename for the dataset is provided, read it in
            self.ds = xr.open_dataset(data)

        self.x = self.ds.x.values
        self.y = self.ds.y.values

        # Image related parameters
        self.startingX = self.ds.x.values[0]
        self.startingY = self.ds.y.values[0]

        self.XSize = self.ds.x.values[1] - self.ds.x.values[0]
        self.YSize = self.ds.y.values[1] - self.ds.y.values[0]

        self.X_res = np.abs(self.XSize)
        self.Y_res = np.abs(self.YSize)

        self.numberOfSamples = len(self.ds.x)
        self.numberOfLines = len(self.ds.y)

        self.epsg = int(self.ds.UTM_Projection.spatial_epsg)

        # Compute bounding box in source projection
        self.bbox_x, self.bbox_y = ItsLiveReproject.bounding_box(
            self.ds,
            self.XSize,
            self.YSize
        )
        self.logger.info(f"P_in bounding box: x: {self.bbox_x} y: {self.bbox_y}")

    @staticmethod
    def bounding_box(ds, dx, dy):
        """
        Select bounding box for the dataset.
        """
        center_off_X = dx/2
        center_off_Y = dy/2

        # Compute cell boundaries as ITS_LIVE grid stores x/y for the cell centers
        xmin = ds.x.values.min() - center_off_X
        xmax = ds.x.values.max() + center_off_X

        # Y coordinate calculations are based on the fact that dy < 0
        ymin = ds.y.values.min() + center_off_Y
        ymax = ds.y.values.max() - center_off_Y

        # ATTN: Assuming that X and Y cell dimensions are the same
        assert np.abs(dx) == np.abs(dy), f"Cell dimensions differ: x={np.abs(dx)} y={np.abs(dy)}"

        return Grid.bounding_box(
            Bounds(min_value=xmin, max_value=xmax),
            Bounds(min_value=ymin, max_value=ymax),
            dx
        )

    def run(self, projection: int, output_file: str):
        """
        Reproject variables in ITS_LIVE granule into new projection.
        """
        # Project the bounding box into output projection
        input_projection = osr.SpatialReference()
        input_projection.ImportFromEPSG(int(self.ds.UTM_Projection.spatial_epsg))

        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(projection)

        transfer = osr.CoordinateTransformation(input_projection, output_projection)

        # Re-project bounding box to output projection
        self.logger.info(f"Reprojecting from {int(self.ds.UTM_Projection.spatial_epsg)} to {projection}")
        points_in = np.array([
            [self.bbox_x.min, self.bbox_y.max],
            [self.bbox_x.max, self.bbox_y.max],
            [self.bbox_x.max, self.bbox_y.min],
            [self.bbox_x.min, self.bbox_y.min]
        ])
        points_out = transfer.TransformPoints(points_in)

        bbox_out_x = Bounds([each[0] for each in points_out])
        bbox_out_y = Bounds([each[1] for each in points_out])

        # x_coords = np.arange(x1.min + self.XSize/2, x1.max, self.XSize)
        # y_coords = np.arange(y1.max + self.YSize/2, y1.min, self.YSize)

        # Get a bounding box in output projection based on edge points of
        # bounding polygon in P_in projection
        x0_bbox, y0_bbox = Grid.bounding_box(bbox_out_x, bbox_out_y, self.XSize)
        self.logger.info(f"P_out bounding box: {x0_bbox} {y0_bbox}")

        # Output grid will be used as input to the gdal.warp()
        x0_grid, y0_grid = Grid.create(x0_bbox, y0_bbox, self.XSize)
        self.logger.info(f"Grid in P_out (cell centers): x={x0_bbox} y={y0_bbox}")

        # Re-project input (x, y) cell centers into output projection: x0, y0
        points_in = ItsLiveReproject.dims_to_grid(self.ds.x.values, self.ds.y.values)
        points_out_0 = transfer.TransformPoints(points_in)
        self.logger.info(f"Len of points in Pin:           {points_in.shape}")
        self.logger.info(f"Len of (x0, y0) points in Pout: {len(points_out_0)}")

        # Get unique x and y points for x0, y0 grid - do we need them?
        x0 = list(set([each[0] for each in points_out]))
        y0 = list(set([each[1] for each in points_out]))

        # Add unit length to ds.x.values
        x_plus_unit = self.ds.x.values + self.XSize
        points_in = ItsLiveReproject.dims_to_grid(x_plus_unit, self.ds.y.values)
        points_out_1 = transfer.TransformPoints(points_in)
        # x1 = [each[0] for each in points_out]
        # y1 = [each[1] for each in points_out]

        # Add unit length to ds.y.values
        y_plus_unit = self.ds.y.values + self.YSize
        points_in = ItsLiveReproject.dims_to_grid(self.ds.x.values, y_plus_unit)
        points_out_2 = transfer.TransformPoints(points_in)
        # x2 = [each[0] for each in points_out]
        # y2 = [each[1] for each in points_out]

        # Compute unit vectors based on points_out_0, points_out_1 and points_out_2
        x_unit = np.zeros((len(points_out_0), 3))
        y_unit = np.zeros((len(points_out_0), 3))

        # For each cell of the grid
        for index in range(len(points_out_0)):
            diff = np.array(points_out_1[index]) - np.array(points_out_0[index])
            x_unit[index] = diff / np.linalg.norm(diff)

            diff = np.array(points_out_2[index]) - np.array(points_out_0[index])
            y_unit[index] = diff / np.linalg.norm(diff)

        print("x_unit[0]: ", x_unit[0])
        print("y_unit[0]: ", y_unit[0])

        # Local normal vector
        normal_vector = np.array([0.0, 0.0, 1.0])

        # Compute transformation matrix per cell



    @staticmethod
    def dims_to_grid(x, y):
        """
        Convert x, y dimensions of the dataset into numpy grid array.
        """
        grid = np.zeros((len(x)*len(y), 2))

        num_row = 0
        for each_x in x:
            for each_y in y:
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
        help='Input file name for ITS_LIVE format data')
    parser.add_argument(
        '-p', '--projection',
        dest='output_proj',
        type=int,
        required=True, help='Output projection')
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        type=str,
        required=True, help='Output filename to store re-projected granule in target projection')

    command_args = parser.parse_args()

    its_data = ItsLiveReproject(command_args.input_file)
    its_data.run(command_args.output_proj, command_args.output_file)
