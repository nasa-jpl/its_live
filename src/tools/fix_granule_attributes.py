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
from datetime import datetime
from dask.diagnostics import ProgressBar
import json
import logging
import os
from pathlib import Path
import s3fs
from tqdm import tqdm
import xarray as xr

from mission_info import Encoding


class LandSat:
    BUCKET = 'usgs-landsat'

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
    return None, None

    # TODO: handle other missions granules
    # elif mission_instrument == 'Sxxx'


class FixGranulesAttributes:
    """
    Class to fix some attributes of ITS_LIVE granules (that were transferred
    from ASF to ITS_LIVE bucket).
    """
    # Date and time format used by ITS_LIVE granules
    DATETIME_FORMAT = '%Y%m%dT%H:%M:%S'

    def __init__(self, bucket: str, bucket_dir: str, glob_pattern: dir, exclude_granule_file: str=None, granule_prefix: str=None):
        self.s3 = s3fs.S3FileSystem()

        # use a glob to list directory
        logging.info(f"Reading {bucket_dir}")
        self.all_granules = self.s3.glob(f'{os.path.join(bucket, bucket_dir)}/{glob_pattern}')
        logging.info(f"Number of all granules: {len(self.all_granules)}")

        # Temporary fix to allow only for specific granule prefix ("LO08" were processed separately)
        if granule_prefix is not None:
            self.all_granules = [each for each in self.all_granules if granule_prefix in each]
            logging.info(f"Number of {granule_prefix} granules: {len(self.all_granules)}")

        if exclude_granule_file is not None:
            exclude_granules = json.loads(exclude_granule_file.read_text())

            # Remove exclude_ids from the jobs to process
            self.all_granules = list(set(self.all_granules).difference(exclude_granules))
            logging.info(f"Number of granules to process: {len(self.all_granules)}")

        # self.all_granules = [each for each in self.all_granules if 'LC08' in each]
        self.bucket = bucket

    def __call__(self, local_dir: str, chunk_size: int, num_dask_workers: int, start_index: int):
        """
        Fix acquisition date and time attributes of ITS_LIVE granules stored
        in the bucket.
        """
        num_to_fix = len(self.all_granules) - start_index

        start = start_index
        logging.info(f"{num_to_fix} granules to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixGranulesAttributes.acquisition_datetime)(self.bucket, each, local_dir, self.s3) for each in self.all_granules[start:start+num_tasks]]
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

    @staticmethod
    def acquisition_datetime(bucket_name: str, granule_url: str, local_dir: str, s3):
        """
        Fix acquisition datetime attribute for image1 and image2 of the granule
        """
        msgs = [f'Processing {granule_url}']

        # get center lat lon
        with s3.open(granule_url) as fhandle:
            with xr.open_dataset(fhandle) as ds:
                img1_datetime = ds['img_pair_info'].attrs['acquisition_date_img1']
                img2_datetime = ds['img_pair_info'].attrs['acquisition_date_img2']

                granule_basename = os.path.basename(granule_url)
                time1, time2 = get_granule_acquisition_times(granule_basename)

                if time1 is None or time2 is None:
                    msgs.append(f"CRITICAL: unexpected filename format for {granule_basename}")
                    return msgs

                if datetime.strptime(img1_datetime, FixGranulesAttributes.DATETIME_FORMAT).date() != \
                   datetime.strptime(time1, '%Y-%m-%dT%H:%M:%S.%fZ').date():
                    msgs.append(f"ERROR: Inconsistent img1 date: {img1_datetime} vs. {time1}")
                    return msgs

                if datetime.strptime(img2_datetime, FixGranulesAttributes.DATETIME_FORMAT).date() != \
                   datetime.strptime(time2, '%Y-%m-%dT%H:%M:%S.%fZ').date():
                    msgs.append(f"ERROR: Inconsistent img2 date: {img2_datetime} vs. {time2}")
                    return msgs

                # msgs.append(f'Replace time: {img1_datetime} vs. {time1}; {img2_datetime} vs. {time2}' )
                msgs.append(f'Replace img1 time: {img1_datetime} vs. {time1}')
                msgs.append(f'Replace img2 time: {img2_datetime} vs. {time2}' )

                time1 = datetime.strptime(time1, '%Y-%m-%dT%H:%M:%S.%fZ')
                time2 = datetime.strptime(time2, '%Y-%m-%dT%H:%M:%S.%fZ')

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
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
        '-glob', action='store', type=str, default='*/*.nc',
        help='Glob pattern for the granule search under "s3://bucket/dir/" [%(default)s]')

    parser.add_argument('-w', '--dask-workers', type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument('-s', '--start-granule', type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument('-e', '--exclude-granule-file', type=Path,
        default=None,
        help='JSON list of granules (previously processed) to exclude from processing [%(default)s]')
    parser.add_argument('-i', '--granule-prefix', type=str,
        default=None,
        help='Granule prefix to include into processing [%(default)s]')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    fix_attributes = FixGranulesAttributes(args.bucket, args.bucket_dir, args.glob, args.exclude_granule_file, args.granule_prefix)
    fix_attributes(args.local_dir, args.chunk_size, args.dask_workers, args.start_granule)


if __name__ == '__main__':
    main()

    logging.info("Done.")
