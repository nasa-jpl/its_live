"""
Classes that define data variables and attributes for the ITS_LIVE datacube.
"""


class Coords:
    """
    Coordinates for the data cube.
    """
    MID_DATE = 'mid_date'
    X = 'x'
    Y = 'y'

    STD_NAME = {
        MID_DATE: "image_pair_center_date_with_time_separation"
    }

    DESCRIPTION = {
        MID_DATE: "midpoint of image 1 and image 2 acquisition date with time " \
            "separation (in days) between acquisition of image 1 and image 2 as " \
            "milliseconds"
    }


class DataVars:
    """
    Data variables for the data cube.
    """
    # Attributes that appear for multiple data variables
    MISSING_VALUE_ATTR         = 'missing_value'
    FILL_VALUE_ATTR            = '_FillValue'
    DESCRIPTION_ATTR           = 'description'  # v, vx, vy
    GRID_MAPPING               = 'grid_mapping' # v, vx, vy - store only one
    STABLE_COUNT               = 'stable_count' # vx, vy    - store only one
    STABLE_SHIFT               = 'stable_shift' # vx, vy
    FLAG_STABLE_SHIFT_MEANINGS = 'flag_stable_shift_meanings' # vx, vy

    # Optical Legacy format only:
    STABLE_SHIFT_APPLIED = 'stable_shift_applied' # vx, vy - remove from attributes
    STABLE_APPLY_DATE    = 'stable_apply_date' # vx, vy - remove from attributes

    STD_NAME    = 'standard_name'
    UNITS       = 'units'
    M_Y_UNITS   = 'm/y'
    COUNT_UNITS = 'count'
    BINARY_UNITS = 'binary'

    # Original data variables and their attributes per ITS_LIVE granules.
    V = 'v'
    # Attributes
    MAP_SCALE_CORRECTED = 'map_scale_corrected'

    VX = 'vx'
    # Attributes
    VX_ERROR          = 'vx_error'          # In Radar and updated Optical formats
    FLAG_STABLE_SHIFT = 'flag_stable_shift' # In Radar and updated Optical formats
    STABLE_RMSE       = 'stable_rmse'       # In Optical legacy format only

    VY                = 'vy'
    # Attributes
    VY_ERROR          = 'vy_error'          # In Radar and updated Optical formats

    CHIP_SIZE_HEIGHT  = 'chip_size_height'
    CHIP_SIZE_WIDTH   = 'chip_size_width'
    # Attributes
    CHIP_SIZE_COORDS  = 'chip_size_coordinates'

    INTERP_MASK      = 'interp_mask'
    V_ERROR          = 'v_error' # Optical and Radar formats only
    # Radar format only:
    VA               = 'va'
    VP               = 'vp'
    VP_ERROR         = 'vp_error'
    VR               = 'vr'
    VX               = 'vx'
    VXP              = 'vxp'
    VYP              = 'vyp'

    # Specific to the datacube
    URL = 'original_url_path'

    # Data variable specific to the epsg code:
    # * Polar_Stereographic when epsg code of 3031 or 3413
    # * UTM_Projection when epsg code of 326** or 327**
    POLAR_STEREOGRAPHIC = 'Polar_Stereographic'
    UTM_PROJECTION = 'UTM_Projection'

    # Missing values for data variables
    MISSING_BYTE      = 0.0
    MISSING_UBYTE     = 0.0
    MISSING_VALUE     = -32767.0
    MISSING_POS_VALUE = 32767.0

    NAME = {
        INTERP_MASK: "interpolated_value_mask",
        VA: "azimuth_velocity",
        VP: "projected_velocity",
        VR: "range_velocity",
        VXP: "projected_x_velocity",
        VYP: "projected_y_velocity",
        V_ERROR: 'velocity_error',
        VP_ERROR: "projected_velocity_error",
    }


    # Description strings for all data variables and some
    # of their attributes.
    DESCRIPTION = {
        V: "velocity magnitude",
        VX: "velocity component in x direction",
        VY: "velocity component in y direction",

        # These descriptions are based on Radar granule format. Have to set them
        # manually since there are no Radar format granules are available for
        # processing just yet (otherwise these attributes would be automatically
        # picked up from the granules).
        VA: "velocity in radar azimuth direction",
        VR: "velocity in radar range direction",
        VP: "velocity magnitude determined by projecting radar " \
            "range measurements onto an a priori flow vector. Where projected " \
            "errors are larger than those determined from range and azimuth " \
            "measurements, unprojected v estimates are used",
        VXP: "x-direction velocity determined by projecting radar " \
            "range measurements onto an a priori flow vector. Where projected " \
            "errors are larger than those determined from range and azimuth " \
            "measurements, unprojected vx estimates are used",
        VYP: "y-direction velocity determined by projecting radar " \
            "range measurements onto an a priori flow vector. Where projected " \
            "errors are larger than those determined from range and azimuth " \
            "measurements, unprojected vy estimates are used",
        V_ERROR: "velocity magnitude error",
        VP_ERROR: "velocity magnitude error determined by projecting " \
            "radar range measurements onto an a priori flow vector. " \
            "Where projected errors are larger than those determined from range " \
            "and azimuth measurements, unprojected v_error estimates are used",
        INTERP_MASK: "light interpolation mask",
        CHIP_SIZE_COORDS: "Optical data: chip_size_coordinates = " \
            "'image projection geometry: width = x, height = y'. Radar data: " \
            "chip_size_coordinates = 'radar geometry: width = range, height = azimuth'",
        CHIP_SIZE_HEIGHT: "height of search window",
        CHIP_SIZE_WIDTH: "width of search window",
        FLAG_STABLE_SHIFT_MEANINGS: "flag for applying velocity bias correction " \
            " over stable surfaces " \
            "(stationary or slow-flowing surfaces with velocity < 15 m/yr): " \
            "0 = there is no stable surface available and no correction is applied; " \
            "1 = there are stable surfaces and velocity bias is corrected;",
        URL: "original granule URL"
    }


    class ImgPairInfo:
        """
        Class to represent attributes of the "img_pair_info" data variable,
        which become new data variables in the datacube to represent these
        attributes for all layers in the datacube.
        """
        NAME = 'img_pair_info'

        DATE_UNITS = 'days since 1970-01-01'

        # Attributes
        MISSION_IMG1              = 'mission_img1'
        SENSOR_IMG1               = 'sensor_img1'
        SATELLITE_IMG1            = 'satellite_img1'
        ACQUISITION_IMG1          = 'acquisition_img1'
        MISSION_IMG2              = 'mission_img2'
        SENSOR_IMG2               = 'sensor_img2'
        SATELLITE_IMG2            = 'satellite_img2'
        ACQUISITION_IMG2          = 'acquisition_img2'
        DATE_DT                   = 'date_dt'
        # Rename mid_date to date_center as they are the same, don't collect this
        DATE_CENTER               = 'date_center'
        ROI_VALID_PERCENTAGE      = 'roi_valid_percentage'
        AUTORIFT_SOFTWARE_VERSION = 'autoRIFT_software_version'

        # Optical and optical legacy formats define them as:
        AQUISITION_DATE_IMG1 = 'aquisition_date_img1'
        AQUISITION_DATE_IMG2 = 'aquisition_date_img2'
        ALL = [
            MISSION_IMG1,
            SENSOR_IMG1,
            SATELLITE_IMG1,
            MISSION_IMG2,
            SENSOR_IMG2,
            SATELLITE_IMG2,
            DATE_DT,
            DATE_CENTER,
            ROI_VALID_PERCENTAGE,
            AUTORIFT_SOFTWARE_VERSION
        ]

        # Description strings for data variables.
        DESCRIPTION = {
            MISSION_IMG1: "id of the mission that acquired image 1",
            SENSOR_IMG1:  "id of the sensor that acquired image 1",
            SATELLITE_IMG1: "id of the satellite that acquired image 1",
            ACQUISITION_IMG1: "acquisition date and time of image 1",
            MISSION_IMG2: "id of the mission that acquired image 2",
            SENSOR_IMG2:  "id of the sensor that acquired image 2",
            SATELLITE_IMG2: "id of the satellite that acquired image 2",
            ACQUISITION_IMG2: "acquisition date and time of image 2",
            DATE_DT: "time separation between acquisition of image 1 and image 2",
            DATE_CENTER: "midpoint of image 1 and image 2 acquisition date",
            ROI_VALID_PERCENTAGE: "percentage of pixels with a valid velocity " \
                "estimate determined for the intersection of the full image " \
                "pair footprint and the region of interest (roi) that defines " \
                "where autoRIFT tried to estimate a velocity",
            AUTORIFT_SOFTWARE_VERSION: "version of autoRIFT software"
        }

        # Flag if data variable values are to be converted to the date objects
        CONVERT_TO_DATE = {
            MISSION_IMG1: False,
            SENSOR_IMG1:  False,
            SATELLITE_IMG1: False,
            ACQUISITION_IMG1: True,
            MISSION_IMG2: False,
            SENSOR_IMG2:  False,
            SATELLITE_IMG2: False,
            ACQUISITION_IMG2: True,
            DATE_DT: False,
            DATE_CENTER: True,
            ROI_VALID_PERCENTAGE: False,
            AUTORIFT_SOFTWARE_VERSION: False
        }

        STD_NAME = {
            MISSION_IMG1: "image1_mission",
            SENSOR_IMG1: "image1_sensor",
            SATELLITE_IMG1: "image1_satellite",
            ACQUISITION_IMG1: "image1_acquition_date",
            MISSION_IMG2: "image2_mission",
            SENSOR_IMG2: "image2_sensor",
            SATELLITE_IMG2: "image2_satellite",
            ACQUISITION_IMG2: "image2_acquition_date",
            DATE_DT: "image_pair_time_separation",
            DATE_CENTER: "image_pair_center_date",
            ROI_VALID_PERCENTAGE: "region_of_interest_valid_pixel_percentage",
            AUTORIFT_SOFTWARE_VERSION: "autoRIFT_software_version"
        }

        UNITS = {
            DATE_DT: 'days',
            # ACQUISITION_IMG1: DATE_UNITS,
            # ACQUISITION_IMG2: DATE_UNITS,
            # DATE_CENTER: DATE_UNITS
        }
