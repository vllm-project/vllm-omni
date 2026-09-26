# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

_LOCAL_RANK_ENV_NAMES = (
    "LOCAL_RANK",
    "VLLM_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
)
_GLOBAL_RANK_ENV_NAMES = ("RANK", "RANK_ID")


def _visible_device_count() -> int | None:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    return len(devices) or None


def _get_tensor_model_parallel_rank() -> int:
    from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

    return int(get_tensor_model_parallel_rank())


def get_connector_local_rank() -> int:
    try:
        return _get_tensor_model_parallel_rank()
    except Exception:
        pass

    for env_name in _LOCAL_RANK_ENV_NAMES:
        try:
            return int(os.environ[env_name])
        except (KeyError, TypeError, ValueError):
            pass

    visible_count = _visible_device_count()
    for env_name in _GLOBAL_RANK_ENV_NAMES:
        try:
            rank = int(os.environ[env_name])
        except (KeyError, TypeError, ValueError):
            continue
        return rank % visible_count if visible_count else rank

    return 0
