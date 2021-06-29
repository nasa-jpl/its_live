#!/usr/bin/env python
"""
Fix attributes of ITS_LIVE granules with information that was not available
during granule production.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Joe Kennedy
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import dask
import datetime
from dask.diagnostics import ProgressBar
import json
import logging
import os
import s3fs
import time
from tqdm import tqdm
import xarray as xr


class Encoding:
    """
    Encoding settings for writing ITS_LIVE granule to the file
    """
    LANDSAT = {
        'interp_mask':      {'_FillValue': 0.0, 'dtype': 'ubyte'},
        'chip_size_height': {'_FillValue': 0.0, 'dtype': 'ushort'},
        'chip_size_width':  {'_FillValue': 0.0, 'dtype': 'ushort'},
        'v_error':          {'_FillValue': -32767.0, 'dtype': 'short'},
        'v':                {'_FillValue': -32767.0, 'dtype': 'short'},
        'vx':               {'_FillValue': -32767.0, 'dtype': 'short'},
        'vy':               {'_FillValue': -32767.0, 'dtype': 'short'},
        'x':                {'_FillValue': None},
        'y':                {'_FillValue': None}
    }

class LandSat:
    BUCKET = 'usgs-landsat'

    # Datetime format for the metadata in LandSat data
    DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

    @staticmethod
    def get_lc2_stac_json_key(scene_name):
        """
        Get path to the scene info file.
        """
        year = scene_name[17:21]
        path = scene_name[10:13]
        row = scene_name[13:16]
        return f'collection02/level-1/standard/oli-tirs/{year}/{path}/{row}/{scene_name}/{scene_name}_stac.json'

    @staticmethod
    def get_lc2_metadata_acquisition_time(scene_name):
        """
        Directly load the STAC item json from the AWS bucket, and extract
        acquisition time for the scene.
        """
        # logging.info(f"Get time for: {scene_name}")
        key = LandSat.get_lc2_stac_json_key(scene_name)
        # logging.info(f"Got key: {key}")

        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=LandSat.BUCKET, Key=key, RequestPayer='requester')
        return json.load(obj['Body'])['properties']['datetime']


def get_granule_acquisition_times(filename):
    """
    Extract scenes names from the granule filename.
    """
    scenes = os.path.basename(filename).split('_X_')

    # Get mission + sensor from the first image name
    mission_instrument = scenes[0][0:4]

    if mission_instrument in ('LC08', 'LO08'):
        # Landsat8 granule filename. For example:
        # LC08_L1TP_231011_20210319_20210328_02_T2_X_LC08_L1TP_231011_20210404_20210409_02_T1_G0120V02_P039_IL_ASF_OD.nc
        return (
            LandSat.get_lc2_metadata_acquisition_time(scenes[0]),
            LandSat.get_lc2_metadata_acquisition_time(scenes[1][:-27])
        )
    # TODO: handle other missions granules
    # elif mission_instrument == 'Sxxx'

    return None, None


class FixGranulesAttributes:
    """
    Class to fix some attributes of ITS_LIVE granules (that were transferred
    from ASF to ITS_LIVE bucket).
    """
    # Date and time format used by ITS_LIVE granules
    DATETIME_FORMAT = '%Y%m%dT%H:%M:%S'

    # Default "zero" (not set) time for the granule acquisition date/time
    ZERO_TIME = datetime.time(0,0,0)

    def __init__(self, bucket: str, bucket_dir: str, glob_pattern: dir):
        """
        Initialize the object.
        """
        # use a glob to list directory
        logging.info(f"Reading {bucket_dir}")
        self.all_granules = s3fs.S3FileSystem(skip_instance_cache=True).glob(f'{os.path.join(bucket, bucket_dir)}/{glob_pattern}')
        self.all_granules.sort()

        self.bucket = bucket

    def parallel(self, local_dir: str, chunk_size: int, num_dask_workers: int, start_index: int):
        """
        Fix acquisition date and time attributes of ITS_LIVE granules stored
        in the bucket.
        """
        num_to_fix = len(self.all_granules) - start_index
        start = start_index
        logging.info(f"{num_to_fix} granules to fix...")

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixGranulesAttributes.acquisition_datetime)(self.bucket, each, local_dir) for each in self.all_granules[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    def __call__(self, local_dir: str, chunk_size: int, start_index: int=0, stop_index: int=None):
        """
        Fix acquisition date and time attributes of ITS_LIVE granules stored
        in the bucket.
        """
        num_to_fix = len(self.all_granules) - start_index
        if stop_index is not None:
            num_to_fix = stop_index - start_index

        start = start_index
        logging.info(f"{num_to_fix} granules to fix (start={start_index} stop={stop_index})...")

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            for each in self.all_granules[start:start+num_tasks]:
                each_result = FixGranulesAttributes.acquisition_datetime(self.bucket, each, local_dir)
                logging.info("-->".join(each_result))
                time.sleep(3)

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def acquisition_datetime(bucket_name: str, granule_url: str, local_dir: str):
        """
        Fix acquisition datetime attribute for image1 and image2 of the granule
        """
        msgs = [f'Processing {granule_url}']

        # When there is not much work to be done over the s3fs, some of the reads
        # seem to be stuck when the same cached s3 instance is reused. Disable
        # cache of s3fs instances (fixes the problem).
        s3 = s3fs.S3FileSystem(skip_instance_cache=True)

        with s3.open(granule_url, 'rb') as fhandle:
            with xr.open_dataset(fhandle) as ds:
                img1_datetime = ds['img_pair_info'].attrs['acquisition_date_img1']
                img2_datetime = ds['img_pair_info'].attrs['acquisition_date_img2']

                # Convert to datetime objects
                img1_obj = datetime.datetime.strptime(img1_datetime, FixGranulesAttributes.DATETIME_FORMAT)
                img2_obj = datetime.datetime.strptime(img2_datetime, FixGranulesAttributes.DATETIME_FORMAT)

                # Check if acquisition time is already set (non-zero).
                # If LandSat acquisition time happens to be zero, then it will get reset
                # through the time metadata as acquired from the LandSat S3 bucket.
                # (most likely there won't be many occurences of it)
                if img1_obj.time() != FixGranulesAttributes.ZERO_TIME and \
                   img2_obj.time() != FixGranulesAttributes.ZERO_TIME:
                    msgs.append("Time is already set.")
                    return msgs

                granule_basename = os.path.basename(granule_url)
                time1_str, time2_str = get_granule_acquisition_times(granule_basename)

                if (time1_str is None) or (time2_str is None):
                    msgs.append(f"CRITICAL: unexpected filename format for {granule_basename}")
                    return msgs

                time1 = datetime.datetime.strptime(time1_str, LandSat.DATETIME_FORMAT)
                time2 = datetime.datetime.strptime(time2_str, LandSat.DATETIME_FORMAT)

                if img1_obj.date() != time1.date():
                    msgs.append(f"ERROR: Inconsistent img1 date: {img1_datetime} vs. {time1_str}")
                    return msgs

                if img2_obj.date() != time2.date():
                    msgs.append(f"ERROR: Inconsistent img2 date: {img2_datetime} vs. {time2_str}")
                    return msgs

                # If time happends to be the same as in the granule, skip
                # overwriting the file
                if img1_obj.time() == time1.time() and img2_obj.time() == time2.time():
                    msgs.append("Time stamps match, skip overwrite.")
                    return msgs

                msgs.append(f'Replace img1 time: {img1_datetime} vs. {time1_str}')
                msgs.append(f'Replace img2 time: {img2_datetime} vs. {time2_str}' )

                ds['img_pair_info'].attrs['acquisition_date_img1'] = time1.strftime(FixGranulesAttributes.DATETIME_FORMAT)
                ds['img_pair_info'].attrs['acquisition_date_img2'] = time2.strftime(FixGranulesAttributes.DATETIME_FORMAT)

                # Write the granule locally, upload it to the bucket, remove file
                fixed_file = os.path.join(local_dir, granule_basename)
                ds.to_netcdf(fixed_file, engine='h5netcdf', encoding = Encoding.LANDSAT)

                # Upload corrected granule to the bucket
                s3_client = boto3.client('s3')
                try:
                    bucket_granule = granule_url.replace(bucket_name+'/', '')
                    msgs.append(f"Uploading {bucket_granule} to {bucket_name}")
                    s3_client.upload_file(fixed_file, bucket_name, bucket_granule)

                    msgs.append(f"Removing local {fixed_file}")
                    os.unlink(fixed_file)

                except ClientError as exc:
                    msgs.append(f"ERROR: {exc}")

                return msgs


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--chunk_size', type=int,
        default=100, help='Number of granules to fix in parallel [%(default)d]'
    )
    parser.add_argument(
        '-b', '--bucket', type=str,
        default='its-live-data.jpl.nasa.gov',
        help='AWS S3 that stores ITS_LIVE granules to fix attributes for'
    )
    parser.add_argument(
        '-d', '--bucket_dir', type=str,
        default='velocity_image_pair/landsat/v02',
        help='AWS S3 bucket and directory that store the granules'
    )
    parser.add_argument(
        '-l', '--local_dir', type=str,
        default='sandbox',
        help='Directory to store fixed granules before uploading them to the S3 bucket'
    )
    parser.add_argument(
        '-glob', action='store', type=str,
        default='*/*.nc',
        help='Glob pattern for the granule search under "s3://bucket/dir/" [%(default)s]')

    parser.add_argument(
        '-w', '--dask-workers', type=int,
        default=None,
        help='Number of Dask parallel workers [%(default)d]'
    )

    parser.add_argument(
        '--start-granule', type=int,
        default=0,
        help='Index for the start granule to process (inclusive)[%(default)d]'
    )
    parser.add_argument(
        '--stop-granule', type=int,
        default=None,
        help='Index for the stop granule to process (exclusive) [%(default)d]'
    )

    args = parser.parse_args()
    logging.info(f"Args: {args}")

    fix_attributes = FixGranulesAttributes(args.bucket, args.bucket_dir, args.glob)

    if args.dask_workers is not None:
        fix_attributes.parallel(args.local_dir, args.chunk_size, args.dask_workers, args.start_granule)

    else:
        fix_attributes(args.local_dir, args.chunk_size, args.start_granule, args.stop_granule)


if __name__ == '__main__':
    import sys

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info(f"Command: {sys.argv}")

    main()

    logging.info("Done.")
