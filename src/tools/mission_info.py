"""
Module to capture mission specific information used by multiple Python scripts.
"""

class Encoding:
    """
    Encoding settings for writing ITS_LIVE granule to the file
    """
    LANDSAT = {
        'interp_mask':      {'_FillValue': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
        'chip_size_height': {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'chip_size_width':  {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'v_error':          {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'v':                {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vx':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vy':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'x':                {'_FillValue': None},
        'y':                {'_FillValue': None}
    }
