#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Source before server/client commands to run on the host without a container.
COSMOS3_NATIVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VLLM_PYTHON="${VLLM_PYTHON:-$COSMOS3_NATIVE_ROOT/.venv-native/bin/python}"
export COSMOS3_CLIENT_PYTHON="${COSMOS3_CLIENT_PYTHON:-$VLLM_PYTHON}"
export COSMOS3_CANN_ROOT="${COSMOS3_CANN_ROOT:-$COSMOS3_NATIVE_ROOT/.runtime/cann-9.1.0}"
[[ -d "$COSMOS3_CANN_ROOT/opp" ]] || { echo "Missing CANN runtime: $COSMOS3_CANN_ROOT" >&2; return 2; }
export VLLM_OMNI_VERSION_OVERRIDE=0.33.0rc1
export ASCEND_HOME_PATH="$COSMOS3_CANN_ROOT" ASCEND_TOOLKIT_HOME="$COSMOS3_CANN_ROOT"
export ASCEND_TOOLKIT_LATEST_HOME="$COSMOS3_CANN_ROOT"
export ASCEND_OPP_PATH="$COSMOS3_CANN_ROOT/opp" ASCEND_AICPU_PATH="$COSMOS3_CANN_ROOT"
export TOOLCHAIN_HOME="$COSMOS3_CANN_ROOT/toolkit"
export COSMOS3_MESA_ROOT="${COSMOS3_MESA_ROOT:-$COSMOS3_NATIVE_ROOT/.runtime/mesa-23.2.1}"
[[ -f "$COSMOS3_MESA_ROOT/lib/libOSMesa.so" ]] || { echo "Missing pinned Mesa runtime: $COSMOS3_MESA_ROOT" >&2; return 2; }
export LD_LIBRARY_PATH="$COSMOS3_MESA_ROOT/lib:$COSMOS3_NATIVE_ROOT/.runtime/lib:$COSMOS3_CANN_ROOT/lib64:$COSMOS3_CANN_ROOT/lib64/plugin/opskernel:$COSMOS3_CANN_ROOT/lib64/plugin/nnengine:$COSMOS3_CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$COSMOS3_NATIVE_ROOT:$COSMOS3_NATIVE_ROOT/plugins/cosmos3_client:$COSMOS3_CANN_ROOT/python/site-packages:$COSMOS3_CANN_ROOT/opp/built-in/op_impl/ai_core/tbe${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$COSMOS3_CANN_ROOT/bin:$COSMOS3_CANN_ROOT/compiler/ccec_compiler/bin:$PATH"
export COSMOS3_DEVICE=npu COSMOS3_NPU_DEVICES="${COSMOS3_NPU_DEVICES:-4}"
export ASCEND_DEVICE_ID=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# Let the server/client launchers select their original 2/1 OpenMP defaults.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# The simulation client uses CPU Torch. Workers import torch_npu explicitly.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# Match the legacy control JIT; closed_loop_eval loads LLVM before SciPy/Torch.
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-0}" NUMBA_OPT="${NUMBA_OPT:-3}"
