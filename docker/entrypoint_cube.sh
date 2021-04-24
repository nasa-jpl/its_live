#!/bin/bash --login
set -e
# conda activate itscube

#PROGRAM_DIR=/home/conda/itslive

#export PYTHONPATH=$PYTHONPATH:${PGE_PROGRAM_DIR}

python ./its_cube.py "$@"
