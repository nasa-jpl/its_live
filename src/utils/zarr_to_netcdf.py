"""
Script to convert Zarr store to the NetCDF format file.

Usage:
python zarr_to_netcdf.py -i ZarrStoreName -o NetCDFFileName

Convert Zarr data stored in ZarrStoreName to the NetCDF file NetCDFFileName.
"""

import argparse
import timeit
import warnings
import xarray as xr

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(epilog='\n'.join(__doc__.split('\n')[1:]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input Zarr store directory.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="NetCDF filename to store data to.")
    parser.add_argument('-e', '--engine', type=str, required=False, default='netcdf4',
                        help="NetCDF engine to use to store NetCDF data to the file. Default is 'h5netcdf'.")

    args = parser.parse_args()

    start_time = timeit.default_timer()
    # Don't decode time delta's as it does some internal conversion based on
    # provided units
    ds_zarr = xr.open_zarr(args.input, decode_timedelta=False)
    time_delta = timeit.default_timer() - start_time
    print(f"Read Zarr {args.input} (took {time_delta} seconds)")

    compression = {"zlib": True, "complevel": 1}
    encoding = {}

    encode_data_vars = (
        'v',
        'v_error',
        'map_scale_corrected',
        'vx',
        'vx_error',
        'vx_stable_shift',
        'flag_stable_shift',
        'vy',
        'vy_error',
        'vy_stable_shift',
        'chip_size_height',
        'chip_size_width',
        'interp_mask',
        'va',
        'va_error',
        'va_stable_shift',
        'vp',
        'vp_error',
        'vr',
        'vr_error',
        'vr_stable_shift',
        'vxp',
        'vxp_error',
        'vxp_stable_shift',
        'vyp',
        'vyp_error',
        'vyp_stable_shift',
        'mission_img1',
        'sensor_img1',
        'satellite_img1',
        'acquisition_img1',
        'mission_img2',
        'sensor_img2',
        'satellite_img2',
        'acquisition_img2',
        'date_dt',
        'date_center',
        'roi_valid_percentage',
        'autoRIFT_software_version'
    )

    # Set up compression for each of the data variables
    for each in encode_data_vars:
        encoding[each] = compression

    start_time = timeit.default_timer()
    ds_zarr.to_netcdf(
        args.output,
        engine=args.engine,
        encoding = encoding
    )
    time_delta = timeit.default_timer() - start_time
    print(f"Wrote dataset to NetCDF file {args.output} (took {time_delta} seconds)")
