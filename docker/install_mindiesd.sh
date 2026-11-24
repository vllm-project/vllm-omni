#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
set -e

# Use the upstream indexed gate fix until a release wheel contains it.
mindiesd_revision=0080d12739143f7255f415147bec6f4a2f626700
export MINDIE_SD_VERSION_OVERRIDE="3.3.0+git.${mindiesd_revision:0:7}"
export BUILD_JOB_NUM="${BUILD_JOB_NUM:-$(nproc)}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib"

python3 -m pip install --no-cache-dir "setuptools>=65.5" wheel "packaging>=20.0"
GIT_TERMINAL_PROMPT=0 python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -v \
    "mindiesd @ git+https://gitcode.com/Ascend/MindIE-SD.git@${mindiesd_revision}"
