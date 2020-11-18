import numpy as np


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
    def bounds(x_min, x_max, y_min, y_max, grid_spacing):
        """
        Define bounding box for provided coordinates.
        """
        # Check if requested grid size is allowable
        if grid_spacing not in Grid._SUPPORTED_SIZES:
            raise RuntimeError(f'Grid spacing should be one of {Grid._SUPPORTED_SIZES} to keep grids of different spacing aligned')

        if x_min >= x_max:
            raise RuntimeError(f'xmin ({x_min}) must be < xmax ({x_max})')

        if y_min >= y_max:
            raise RuntimeError(f'y_min ({y_min}) must be < y_max ({y_max})')

        # Determine grid edges
        x0_min = np.ceil(x_min/grid_spacing)*grid_spacing - Grid.L8B8_pix/2
        y0_min = np.floor(y_min/grid_spacing)*grid_spacing + Grid.L8B8_pix/2
        x0_max = np.ceil(x_max/grid_spacing)*grid_spacing - Grid.L8B8_pix/2
        y0_max = np.floor(y_max/grid_spacing)*grid_spacing + Grid.L8B8_pix/2

        return x0_min, x0_max, y0_min, y0_max


    @staticmethod
    def create_grid(x_min, x_max, y_min, y_max, grid_spacing):
        """
        Create new grid given the spacing and bounding box for the region.
        """
        # Calculate grid bounds
        x0_min, x0_max, y0_min, y0_max = Grid.bounds(x_min, x_max, y_min, y_max, grid_spacing)
        print(f"create_grid: {x0_min} {x0_max}, {y0_min}, {y0_max}" )

        # Generate vectors of grid centers
        # Cell center offset
        cell_center_offset = grid_spacing/2
        x_vals = np.arange(x0_min + cell_center_offset, x0_max, grid_spacing)
        y_vals = np.arange(y0_max - cell_center_offset, y0_min, -grid_spacing)

        return x_vals, y_vals
