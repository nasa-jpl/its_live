"""
Script to drive Batch processing for datacube generation at AWS.

It accepts geojson file with datacube definitions and submits one AWS Batch job
per each datacube which has ROI (region of interest) != 0.
"""
import boto3
import logging


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)


class DataCubeBatch:
    """
    Class to manage one Batch job submission at AWS.
    """
    CLIENT = boto3.client('batch')

    def __init__(self, job_name: str):
        """
        Initialize object.
        """
        self.job_name = job_name

    def __call__(self, params: dict={}):
        """
        Submit job to AWS.
        """
         # -n 100 -o testdata.zarr -b s3://kh9-1/test_datacube --targetProjection 32628 --centroid 487462 9016243
        response = DataCubeBatch.CLIENT.submit_job(
            jobName=self.job_name,
            jobQueue='masha-dave-test',
            # jobDefinition='arn:aws:batch:us-west-2:849259517355:job-definition/its-live-datacube-python:1',
            jobDefinition='arn:aws:batch:us-west-2:849259517355:job-definition/its-live-datacube-with-options:1',
            parameters={
                'numberGranules': '100',
                'outputStore': 'batch_testcube.zarr',
                'outputBucket': 's3://kh9-1/test_datacube',
                'targetProjection': '32628',
                'centroid': '[487462, 9016243]'
            },
            # containerOverrides={
            #     'vcpus': 123,
            #     'memory': ,
            #     'command': [
            #         'string',
            #     ],
            #     'environment': [
            #         {
            #             'name': 'string',
            #             'value': 'string'
            #         },
            #     ]
            # },
            retryStrategy={
                'attempts': 1
            },
            timeout={
                'attemptDurationSeconds': 60
            }
        )

        logging.info(f"Response: {response}")

def main(cube_definition_file: str):
    """
    Driver to submit multiple Batch jobs to AWS.
    """
    # Submit Batch job to AWS for each datacube which has ROI!=0
    batch = DataCubeBatch('itslivecube_batch_by_python_test')
    batch()

if __name__ == '__main__':
    # Since port forwarding is not working on EC2 to run jupyter lab for now,
    # allow to run test case from itscube.ipynb in standalone mode
    import argparse
    import warnings
    import sys
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--cubeDefinitionFile', type=str, default=None,
                        help="GeoJson file that stores cube polygon definitions [%(default)s].")
    parser.add_argument('-b', '--batchJobDefinition', type=str, default='arn:aws:batch:us-west-2:849259517355:job-definition/its-live-datacubes:1',
                        help="Batch job definition to use [%(default)s].")

    args = parser.parse_args()

    main(args.cubeDefinitionFile)
