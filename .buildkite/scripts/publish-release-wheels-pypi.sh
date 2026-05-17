#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -e

BUCKET="vllm-wheels"
SUBPATH="omni/$BUILDKITE_COMMIT"
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

RELEASE_VERSION=$(buildkite-agent meta-data get release-version | sed 's/^v//')
if [[ -z "$RELEASE_VERSION" ]]; then
  echo "[FATAL] release-version metadata not set."
  exit 1
fi
echo "Release version: $RELEASE_VERSION"

# if [[ -z "$PYPI_TOKEN" ]]; then
#   echo "[FATAL] PYPI_TOKEN is not set."
#   exit 1
# else
#   export TWINE_USERNAME="__token__"
#   export TWINE_PASSWORD="$PYPI_TOKEN"
# fi

set -x

ALL_DIR=/tmp/vllm-omni-release-all
DIST_DIR=/tmp/vllm-omni-release-dist
mkdir -p "$ALL_DIR" "$DIST_DIR"

echo "========================================"
echo "All wheels in S3 (no filters):"
echo "========================================"
aws s3 cp --recursive "$S3_COMMIT_PREFIX" "$ALL_DIR"
ls -la "$ALL_DIR"

echo ""
echo "========================================"
echo "Filtered wheels (include: vllm_omni-${RELEASE_VERSION}*.whl, exclude: *dev*):"
echo "========================================"
aws s3 cp --recursive --exclude "*" --include "vllm_omni-${RELEASE_VERSION}*.whl" --exclude "*dev*" "$S3_COMMIT_PREFIX" "$DIST_DIR"
ls -la "$DIST_DIR"

PYPI_WHEEL_FILES=$(find "$DIST_DIR" -name "vllm_omni-${RELEASE_VERSION}*.whl" -not -name "*+*")
if [[ -z "$PYPI_WHEEL_FILES" ]]; then
  echo "[FATAL] No wheels found for version ${RELEASE_VERSION}"
  exit 1
fi

echo ""
echo "========================================"
echo "Wheels that would be uploaded to PyPI:"
echo "========================================"
echo "$PYPI_WHEEL_FILES"

# python3 -m twine check $PYPI_WHEEL_FILES
# python3 -m twine upload --non-interactive --verbose $PYPI_WHEEL_FILES
# echo "Wheels uploaded to PyPI"
