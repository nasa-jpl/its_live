import numpy as np


class Bounds:
    """
    Class to store min/max pair for a variable.
    """
    def __init__(self, values=None, min_value=None, max_value=None):
        """
        Initialize object based on the list of values or provided min and max
        values.
        """
        if values is not None:
            # If list if provided, use it to determine min and max values
            self.min = min(values)
            self.max = max(values)

        else:
            self.min = min_value
            self.max = max_value

    def __str__(self):
        """
        String representation of the object.
        """
        return f"min={self.min} max={self.max}"


class Grid:
    """
    Grid specific helper functions.
    """
    # Supported grid sizes
    _SUPPORTED_SIZES = [60, 120, 240, 480, 960, 1920, 3840]

    # Original band 8 Landsat pixel size.
    # This ensures that the offset grid aligns with the pixel edges used in the
    # Landsat gridding convention
    L8B8_pix = 15

    @staticmethod
    def bounding_box(x: Bounds, y: Bounds, grid_spacing: int) -> (Bounds, Bounds):
        """
        Define bounding box for provided coordinates.
        """
        # Check if requested grid size is allowable
        if grid_spacing not in Grid._SUPPORTED_SIZES:
            raise RuntimeError(f'Grid spacing should be one of {Grid._SUPPORTED_SIZES} to keep grids of different spacing aligned')

        if x.min >= x.max:
            raise RuntimeError(f'x.min ({x.min}) must be < x.max ({x.max})')

        if y.min >= y.max:
            raise RuntimeError(f'y.min ({y.min}) must be < y.max ({y.max})')

        # Determine grid edges
        x0_min = np.ceil(x.min/grid_spacing)*grid_spacing - Grid.L8B8_pix/2
        x0_max = np.ceil(x.max/grid_spacing)*grid_spacing - Grid.L8B8_pix/2
        y0_min = np.floor(y.min/grid_spacing)*grid_spacing + Grid.L8B8_pix/2
        y0_max = np.floor(y.max/grid_spacing)*grid_spacing + Grid.L8B8_pix/2

        # print("bounding_box: x_in: ", x)
        # print("bounding_box: y_in: ", y)
        #
        return Bounds(min_value=x0_min, max_value=x0_max), \
               Bounds(min_value=y0_min, max_value=y0_max)

    @staticmethod
    def create(x: Bounds, y: Bounds, grid_spacing):
        """
        Create new grid given the spacing and bounding box for the region.
        """
        # Calculate grid bounds
        x0, y0 = Grid.bounding_box(x, y, grid_spacing)
        print(f"Grid.create: bounding box: x: {x0} y: {y0}" )

        # Generate vectors of grid centers
        # Cell center offset
        cell_center_offset = grid_spacing/2
        x_vals = np.arange(x0.min + cell_center_offset, x0.max, grid_spacing)
        y_vals = np.arange(y0.max - cell_center_offset, y0.min, -grid_spacing)

        return x_vals, y_vals
