# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero OpenPI upstream parity accuracy test."""

from __future__ import annotations

import abc
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.e2e.accuracy.helpers import (
    DreamZeroUpstreamServer,
    assert_dreamzero_logs_clean,
    assert_dreamzero_upstream_log_matches_vllm_baseline,
    dreamzero_parity_vllm_client,
    load_dreamzero_upstream_modules,
    normalize_dreamzero_metadata,
)
from tests.helpers.mark import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

# Hardcoded case inputs (do not use a DREAMZERO_REPO env var):
# - upstream source checkout of https://github.com/dreamzero0/dreamzero
# - HF weights GEAR-Dreams/DreamZero-DROID (helpers resolve via snapshot_download)
DREAMZERO_MODEL = "GEAR-Dreams/DreamZero-DROID"
DREAMZERO_REPO = Path.home() / ".cache" / "vllm-omni" / "dreamzero_upstream"

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]

EXPECTED_METADATA = {
    "image_resolution": [180, 320],
    "n_external_cameras": 2,
    "needs_wrist_camera": True,
    "needs_stereo_camera": False,
    "needs_session_id": True,
    "action_space": "joint_position",
}


def _ensure_openpi_client_for_case() -> None:
    """PyPI openpi-client pins numpy<2; provide a local stand-in for this case only."""
    try:
        import openpi_client  # noqa: F401

        return
    except ImportError:
        pass

    from vllm_omni.entrypoints.openpi.connection import _pack, _unpack

    class BasePolicy(abc.ABC):
        @abc.abstractmethod
        def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
            raise NotImplementedError

        def reset(self) -> None:
            return None

    class _Packer:
        def pack(self, obj: Any) -> bytes:
            return _pack(obj)

    openpi_client = types.ModuleType("openpi_client")
    base_policy = types.ModuleType("openpi_client.base_policy")
    base_policy.BasePolicy = BasePolicy
    msgpack_numpy = types.ModuleType("openpi_client.msgpack_numpy")
    msgpack_numpy.Packer = _Packer
    msgpack_numpy.packb = _pack
    msgpack_numpy.unpackb = _unpack
    openpi_client.base_policy = base_policy
    openpi_client.msgpack_numpy = msgpack_numpy
    openpi_client.BasePolicy = BasePolicy
    sys.modules["openpi_client"] = openpi_client
    sys.modules["openpi_client.base_policy"] = base_policy
    sys.modules["openpi_client.msgpack_numpy"] = msgpack_numpy


_ensure_openpi_client_for_case()


@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_dreamzero_openpi_upstream_parity(tmp_path: Path) -> None:
    """Assert vLLM OpenPI actions match the upstream DreamZero reference server."""
    load_dreamzero_upstream_modules(repo=DREAMZERO_REPO, model_id=DREAMZERO_MODEL)

    upstream_log = tmp_path / "dreamzero_upstream.log"
    with DreamZeroUpstreamServer(log_path=upstream_log) as upstream_client:
        upstream_metadata, upstream_outputs = upstream_client.collect_parity_outputs()
    assert_dreamzero_logs_clean(upstream_log)
    assert_dreamzero_upstream_log_matches_vllm_baseline(upstream_log)

    with dreamzero_parity_vllm_client() as vllm_client:
        vllm_metadata, vllm_outputs = vllm_client.collect_parity_outputs()

    assert normalize_dreamzero_metadata(upstream_metadata) == EXPECTED_METADATA
    assert normalize_dreamzero_metadata(vllm_metadata) == EXPECTED_METADATA

    for idx, (vllm_action, upstream_action) in enumerate(zip(vllm_outputs, upstream_outputs, strict=True)):
        np.testing.assert_allclose(
            vllm_action,
            upstream_action,
            rtol=1e-2,
            atol=1e-3,
            err_msg=f"OpenPI step {idx} output mismatch",
        )
