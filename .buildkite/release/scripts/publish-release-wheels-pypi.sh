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

if [[ -z "$PYPI_TOKEN" ]]; then
  echo "[FATAL] PYPI_TOKEN is not set."
  exit 1
else
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="$PYPI_TOKEN"
fi

set -x

if ! command -v uv &> /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 /tmp/vllm-omni-release-env
source /tmp/vllm-omni-release-env/bin/activate
uv pip install twine

DIST_DIR=/tmp/vllm-omni-release-dist
mkdir -p "$DIST_DIR"

echo "Downloading wheels from S3:"
aws s3 ls "$S3_COMMIT_PREFIX"
aws s3 cp --recursive --exclude "*" --include "vllm_omni-${RELEASE_VERSION}*.whl" --exclude "*dev*" "$S3_COMMIT_PREFIX" "$DIST_DIR"
ls -la "$DIST_DIR"

PYPI_WHEEL_FILES=$(find "$DIST_DIR" -name "vllm_omni-${RELEASE_VERSION}*.whl" -not -name "*+*")
if [[ -z "$PYPI_WHEEL_FILES" ]]; then
  echo "[FATAL] No wheels found for version ${RELEASE_VERSION}"
  exit 1
fi

python3 -m twine check $PYPI_WHEEL_FILES
python3 -m twine upload --non-interactive --verbose $PYPI_WHEEL_FILES
echo "Wheels uploaded to PyPI"
