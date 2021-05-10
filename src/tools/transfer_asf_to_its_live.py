#!/usr/bin/env python
"""
To run the script, you need to have credentials for the https://urs.earthdata.nasa.gov
(register for free is you don't have an account).
Place credentials into the file:
echo 'machine urs.earthdata.nasa.gov login USERNAME password PASSWORD' >& ~/.netrc
"""

import argparse
import json
import logging
import os
from pathlib import Path

import boto3
import fsspec
import xarray as xr
from botocore.exceptions import ClientError

import hyp3_sdk as sdk
import numpy as np

HYP3_AUTORIFT_API = 'https://hyp3-autorift.asf.alaska.edu'

#
# Author: Mark Fahnestock
#
def point_to_prefix(dir_path: str, lat: float, lon: float) -> str:
    """
    Returns a string (for example, N78W124) for directory name based on
    granule centerpoint lat,lon
    """
    NShemi_str = 'N' if lat >= 0.0 else 'S'
    EWhemi_str = 'E' if lon >= 0.0 else 'W'

    outlat = int(10*np.trunc(np.abs(lat/10.0)))
    if outlat == 90: # if you are exactly at a pole, put in lat = 80 bin
        outlat = 80

    outlon = int(10*np.trunc(np.abs(lon/10.0)))

    if outlon >= 180: # if you are at the dateline, back off to the 170 bin
        outlon = 170

    dirstring = os.path.join(dir_path, f'{NShemi_str}{outlat:02d}{EWhemi_str}{outlon:03d}')
    return dirstring

def object_exists(bucket, key: str) -> bool:
    try:
        bucket.Object(key).load()

    except ClientError:
        return False

    return True

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--job-ids', type=Path, help='JSON list of HyP3 Job IDs')
    parser.add_argument('-t', '--target-bucket', help='Upload the autoRIFT products to this AWS bucket')
    parser.add_argument('-d', '--dir', help='Upload the autoRIFT products to this sub-directory of AWS bucket')
    parser.add_argument('-u', '--user', help='Username for https://urs.earthdata.nasa.gov login')
    parser.add_argument('-p', '--password', help='Password for https://urs.earthdata.nasa.gov login')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    hyp3 = sdk.HyP3(HYP3_AUTORIFT_API, args.user, args.password)
    job_ids = json.loads(args.job_ids.read_text())
    target_bucket = boto3.resource('s3').Bucket(args.target_bucket)
    for ii in job_ids:
        job = hyp3.get_job_by_id(ii)
        logging.info(f'Processing {job}')
        if job.running():
            logging.warning(f'Job is still running! Skipping {job}')
            continue

        elif job.succeeded():
            # get center lat lon
            with fsspec.open(job.files[0]['url']) as f:
                with xr.open_dataset(f) as ds:
                    lat = ds.img_pair_info.latitude[0]
                    lon = ds.img_pair_info.longitude[0]
            logging.info(f'Image center (lat, lon): ({lat}, {lon})')

            source = {'Bucket': job.files[0]['s3']['bucket'],
                      'Key': job.files[0]['s3']['key']}

            target_prefix = point_to_prefix(args.dir, lat, lon)
            target_key = f'{target_prefix}/{job.files[0]["filename"]}'

            if object_exists(target_bucket, target_key):
                logging.warning(f'{target_bucket.name}/{target_key} already exists! skipping {job}')
                continue

            logging.info(f'Copying {source["Bucket"]}/{source["Key"]} to {target_bucket.name}/{target_key}')
            target_bucket.copy(source, target_key)
            # TODO: need browse and/or thumbnail images too?

        else:
            logging.warning(f'{job} failed!')
            # TODO: handle failures


if __name__ == '__main__':
    main()
