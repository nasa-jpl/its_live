#!/usr/bin/env python
"""
Fix attributes of ITS_LIVE granules with information that was not available
during granule production.

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
import s3fs
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

    if mission_instrument == 'LC08':
        # Landsat8 granule filename. For example:
        # LC08_L1TP_231011_20210319_20210328_02_T2_X_LC08_L1TP_231011_20210404_20210409_02_T1_G0120V02_P039_IL_ASF_OD.nc
        return (
            LandSat.get_lc2_metadata_acquisition_time(scenes[0]),
            LandSat.get_lc2_metadata_acquisition_time(scenes[1][:-27])
        )
    # TODO: handle other missions granules
    # elif mission_instrument == 'Sxxx'


class FixGranulesAttributes:
    """
    Class to fix some attributes of ITS_LIVE granules (that were transferred
    from ASF to ITS_LIVE bucket).
    """
    POLAR_STEREOGRAPHIC = 'Polar_Stereographic'
    UTM_PROJECTION = 'UTM_Projection'

    # Date and time format used by ITS_LIVE granules
    DATETIME_FORMAT = '%Y%m%dT%H:%M:%S'

    def __init__(self, bucket: str, bucket_dir: str, glob_pattern: dir):
        self.s3 = s3fs.S3FileSystem()

        # use a glob to list directory
        logging.info(f"Reading {bucket_dir}")
        self.all_granules = self.s3.glob(f'{os.path.join(bucket, bucket_dir)}/{glob_pattern}')

        self.bucket = bucket

    def __call__(self, local_dir: str, chunk_size: int):
        """
        Fix acquisition date and time attributes of ITS_LIVE granules stored
        in the bucket.
        """
        num_to_fix = len(self.all_granules)

        start = 0
        logging.info(f"{num_to_fix} granules to fix...")

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixGranulesAttributes.acquisition_datetime)(self.bucket, each, local_dir, self.s3) for each in self.all_granules[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=8)

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    def dont__call__(self, local_dir: str, chunk_size: int):
        """
        Fix acquisition date and time attributes of ITS_LIVE granules stored
        in the bucket.
        """
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        logging.info(f"{len(self.all_granules)} granules to fix...")
        start = 0

        for each in tqdm(self.all_granules, ascii=True, desc="Fixing granules attributes..."):
            FixGranulesAttributes.acquisition_datetime_with_logging(self.bucket, each, local_dir, self.s3)

    @staticmethod
    def acquisition_datetime(bucket_name: str, granule_url: str, local_dir: str, s3):
        """
        Fix acquisition datetime attribute for image1 and image2 of the granule
        """
        msgs = [f'Processing {granule_url}']

        # get center lat lon
        with s3.open(granule_url) as fhandle:
            with xr.open_dataset(fhandle) as ds:
                # Original granules specify '|S1' dtype for data-less
                # variables, so need to convert them to String type to avoid
                # xarray adding "string1" new dimension to such dataa-less
                # char-type variables.
                proj_data = None
                if FixGranulesAttributes.POLAR_STEREOGRAPHIC in ds:
                    proj_data = FixGranulesAttributes.POLAR_STEREOGRAPHIC

                elif FixGranulesAttributes.UTM_PROJECTION in ds:
                    proj_data = FixGranulesAttributes.UTM_PROJECTION

                # Just copy all attributes for the scalar type of the xr.DataArray.
                ds[proj_data] = xr.DataArray(
                    data='',
                    attrs=ds[proj_data].attrs,
                    coords={},
                    dims=[]
                )

                # Just copy all attributes for the scalar type of the xr.DataArray.
                ds['img_pair_info'] = xr.DataArray(
                    data='',
                    attrs=ds.img_pair_info.attrs,
                    coords={},
                    dims=[]
                )

                img1_datetime = ds['img_pair_info'].attrs['acquisition_date_img1']
                img2_datetime = ds['img_pair_info'].attrs['acquisition_date_img2']

                granule_basename = os.path.basename(granule_url)
                time1, time2 = get_granule_acquisition_times(granule_basename)

                if datetime.strptime(img1_datetime, FixGranulesAttributes.DATETIME_FORMAT).date() != \
                   datetime.strptime(time1, '%Y-%m-%dT%H:%M:%S.%fZ').date():
                    msgs.append(f"ERROR: Inconsistent img1 date: {img1_datetime} vs. {time1}")

                if datetime.strptime(img2_datetime, FixGranulesAttributes.DATETIME_FORMAT).date() != \
                   datetime.strptime(time2, '%Y-%m-%dT%H:%M:%S.%fZ').date():
                    msgs.append(f"ERROR: Inconsistent img2 date: {img2_datetime} vs. {time2}")

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

    @staticmethod
    def acquisition_datetime_with_logging(bucket_name: str, granule_url: str, local_dir: str, s3):
        """
        Fix acquisition datetime attribute for image1 and image2 of the granule
        """
        logging.info(f'Processing {granule_url}')

        # get center lat lon
        with s3.open(granule_url) as fhandle:
            with xr.open_dataset(fhandle) as ds:
                # Original granules specify '|S1' dtype for data-less
                # variables, so need to convert them to String type to avoid
                # xarray adding "string1" new dimension to such dataa-less
                # char-type variables.
                proj_data = None
                if FixGranulesAttributes.POLAR_STEREOGRAPHIC in ds:
                    proj_data = FixGranulesAttributes.POLAR_STEREOGRAPHIC

                elif FixGranulesAttributes.UTM_PROJECTION in ds:
                    proj_data = FixGranulesAttributes.UTM_PROJECTION

                # Just copy all attributes for the scalar type of the xr.DataArray.
                ds[proj_data] = xr.DataArray(
                    data='',
                    attrs=ds[proj_data].attrs,
                    coords={},
                    dims=[]
                )

                # Just copy all attributes for the scalar type of the xr.DataArray.
                ds['img_pair_info'] = xr.DataArray(
                    data='',
                    attrs=ds.img_pair_info.attrs,
                    coords={},
                    dims=[]
                )

                img1_datetime = ds['img_pair_info'].attrs['acquisition_date_img1']
                img2_datetime = ds['img_pair_info'].attrs['acquisition_date_img2']

                granule_basename = os.path.basename(granule_url)
                time1, time2 = get_granule_acquisition_times(granule_basename)

                if datetime.strptime(img1_datetime, FixGranulesAttributes.DATETIME_FORMAT).date() != \
                   datetime.strptime(time1, '%Y-%m-%dT%H:%M:%S.%fZ').date():
                    logging.error(f"Inconsistent img1 date: {img1_datetime} vs. {time1}")

                if datetime.strptime(img2_datetime, FixGranulesAttributes.DATETIME_FORMAT).date() != \
                   datetime.strptime(time2, '%Y-%m-%dT%H:%M:%S.%fZ').date():
                    logging.error(f"Inconsistent img2 date: {img2_datetime} vs. {time2}")

                # msgs.append(f'Replace time: {img1_datetime} vs. {time1}; {img2_datetime} vs. {time2}' )
                logging.info(f'Replace img1 time: {img1_datetime} vs. {time1}')
                logging.info(f'Replace img2 time: {img2_datetime} vs. {time2}' )

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
                    logging.info(f"Uploading {bucket_granule} to {bucket_name}")

                    s3_client.upload_file(fixed_file, bucket_name, bucket_granule)

                    logging.info(f"Removing local {fixed_file}")
                    os.unlink(fixed_file)

                except ClientError as exc:
                    logging.error(f"{exc}")

def main():
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
        default='velocity_image_pair/landsat/v01.0',
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

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    fix_attributes = FixGranulesAttributes(args.bucket, args.bucket_dir, args.glob)
    fix_attributes(args.local_dir, args.chunk_size)

if __name__ == '__main__':

    # scene = 'LC08_L1TP_009011_20200703_20200913_02_T1'
    # acquisition_time = get_lc2_metadata_acquisition_time(scene)
    # print(f'{scene} acquisition time: {acquisition_time}')

    main()
