#!/bin/bash
#
# Script to build ITS_LIVE docker image for the datacube.
#
echo '
=======================================================
Building datacube docker image...
======================================================='

set -ex

IMAGE='its_live/datacube'
TAG=$1
BUILD_DATE_TIME=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

# defaults
[ -z "${WORKSPACE}" ] && WORKSPACE=$(dirname $(realpath $0))/..
[ -z "${TAG}" ] && TAG="${USER}"

echo "WORKSPACE: $WORKSPACE"
echo "IMAGE: $IMAGE"
echo "TAG: $TAG"

# Goal: minimize the number of docker layers
# Therefore:  Avoid multiple COPY commands in the Dockerfile
# To support that, do the work here of gathering all the files to copy to
# the docker image. The first step is to create a directory on the host to
# use as a staging area for those files.
TEMP_STAGING_DIR=$(mktemp -d staging_for_cube_docker_image_XXXXXXXXXX)
function cleanup {
  if [[ -z ${KEEP_TEMP_FILES} ]]; then
    echo "cleaning up..."
    rm -r ${TEMP_STAGING_DIR}
  fi
}

trap cleanup EXIT # clean up on exit regardless of whether the build succeeds

mkdir -p ${TEMP_STAGING_DIR}/env

# Copy files to the staging area and build the PGE docker image
cp -r ${WORKSPACE}/src/itscube_types.py ${WORKSPACE}/src/itscube.py \
      ${WORKSPACE}/src/grid.py ${WORKSPACE}/src/itslive_utils.py \
      ${WORKSPACE}/LICENSE \
      ${TEMP_STAGING_DIR}/

cp -r ${WORKSPACE}/environment/cube_environment.yml \
      ${WORKSPACE}/docker/entrypoint_cube.sh \
      ${TEMP_STAGING_DIR}/env/

# Create VERSION file
printf "build_version: ${TAG}\nbuild_datetime: ${BUILD_DATE_TIME}\n" \
    > ${TEMP_STAGING_DIR}/VERSION \

# remove the old docker image if it exists
docker images | grep ${IMAGE}:${TAG} | xargs docker rmi

# build the docker image
docker build --rm --force-rm -t ${IMAGE}:${TAG} \
    --build-arg BUILD_DATE_TIME=${BUILD_DATE_TIME} \
    --build-arg BUILD_VERSION=${TAG} \
    --build-arg SOURCE_DIR=$(basename ${TEMP_STAGING_DIR}) \
    -f ${WORKSPACE}/docker/Dockerfile_datacube ${WORKSPACE}
