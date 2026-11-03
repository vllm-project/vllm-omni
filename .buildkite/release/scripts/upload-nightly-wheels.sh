#!/usr/bin/env bash

set -ex

# Upload a single wheel to S3 under the omni/ prefix.
# Index generation is handled separately by generate-and-upload-nightly-index.sh.

BUCKET="vllm-wheels"
SUBPATH="omni/$BUILDKITE_COMMIT"
S3_COMMIT_PREFIX="s3://$BUCKET/$SUBPATH/"

# ========= collect & upload the wheel ==========

# python3 -m build outputs to dist/ by default
wheel_files=(dist/*.whl)

# Check that exactly one wheel is found
if [[ ${#wheel_files[@]} -ne 1 ]]; then
  echo "Error: Expected exactly one wheel file in dist/, but found ${#wheel_files[@]}"
  exit 1
fi
wheel="${wheel_files[0]}"

echo "Uploading wheel: $wheel"

# Extract the version from the wheel
version=$(unzip -p "$wheel" '**/METADATA' | grep '^Version: ' | cut -d' ' -f2)
echo "Version in wheel: $version"

# Upload wheel to S3
aws s3 cp "$wheel" "$S3_COMMIT_PREFIX"

echo "Wheel uploaded to $S3_COMMIT_PREFIX. Index generation is handled by a separate step."
