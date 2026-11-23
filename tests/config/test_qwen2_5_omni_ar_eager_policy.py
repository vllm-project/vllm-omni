# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pin the Qwen2.5-Omni AR-stage CUDA-graph policy.

The talker runs with CUDA graphs on CUDA (issue #7960): it emits 1.5-2k codec
tokens per request, so removing the per-token launch overhead is what moves its
TPOT. The thinker stays eager because compiling its decode changes
image-conditioned greedy text and is slower for image prefill. Graph capture is
not qualified on ROCm, on the NPU talker, or on XPU, so those platforms keep
eager.

This test keeps the shared deploy YAML, its CI overlay and that intent in sync
without needing a GPU.
"""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "qwen2_5_omni.yaml"


def _eager_flags(platform: str, rel_path: str = _DEPLOY) -> dict[int, bool | None]:
    config = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path(rel_path)),
        platform=platform,
    )
    return {stage.stage_id: stage.enforce_eager for stage in config.stages}


def test_cuda_runs_talker_with_cuda_graphs() -> None:
    # The talker captures graphs; the thinker keeps eager so image-conditioned
    # greedy text stays byte-identical to eager. Code2Wav keeps eager too.
    assert _eager_flags("cuda") == {0: True, 1: False, 2: True}


def test_rocm_keeps_ar_stages_eager() -> None:
    assert _eager_flags("rocm") == {0: True, 1: True, 2: True}


def test_npu_keeps_talker_eager() -> None:
    flags = _eager_flags("npu")
    # The NPU thinker graph is validated; the talker graph is not qualified yet.
    assert flags[0] is False
    assert flags[1] is True
    assert flags[2] is True


def test_xpu_keeps_every_stage_eager() -> None:
    assert _eager_flags("xpu") == {0: True, 1: True, 2: True}


def test_ci_overlay_inherits_cuda_graphs() -> None:
    # The CUDA e2e test runs the CI overlay, so the overlay must pick up the
    # same AR-stage policy as the shared YAML it inherits from.
    flags = _eager_flags("cuda", "ci/qwen2_5_omni.yaml")
    assert flags[0] is True
    assert flags[1] is False
