#!/bin/bash --login
set -e

PROGRAM_DIR=/home/conda/itslive
export PYTHONPATH=$PYTHONPATH:${PROGRAM_DIR}

python /home/conda/itslive/itscube.py "$@"
