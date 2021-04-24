#!/bin/bash
#
# Script to build ITS_LIVE docker images
#
set -ex

TAG=$1

# defaults
[ -z "${WORKSPACE}" ] && WORKSPACE=$(dirname $(realpath $0))/..
[ -z "${TAG}" ] && TAG="${USER}"

echo "WORKSPACE: $WORKSPACE"
echo "TAG: $TAG"

# check scripts directory exists
if [ ! -d "${WORKSPACE}/docker" ]; then
  echo "Error: the docker directory doesn't exist at ${WORKSPACE}/"
  exit 1
fi

# build all of the docker images
BUILD_SCRIPTS_DIR=${WORKSPACE}/docker
${BUILD_SCRIPTS_DIR}/build_datacube.sh ${TAG}
${BUILD_SCRIPTS_DIR}/build_catalog_geojson.sh ${TAG}
${BUILD_SCRIPTS_DIR}/build_reproject.sh ${TAG}
