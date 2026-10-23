# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CAM++ capture/replay output must match the eager output on real NPU.

Guards the two regressions the review rounds caught: captured kernels are
recorded, not executed (fixed in 31e9b208d, replay once after capture),
and the bucket must only be published after that replay+clone succeeds
(fixed in 9778802d3). The first call exercises capture+replay, the second
the steady-state replay path; both are compared against a plain eager
forward on the same input. Random weights suffice: same architecture,
same kernels, same comparison, so no pretrained asset is needed.
"""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import (
    CampplusTorch,
)
from vllm_omni.model_executor.models.indextts2.utils.campplus.dtdnn import CAMPPlus


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except ImportError:
        return False


_HAS_NPU = _npu_available()

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not _HAS_NPU, reason="requires an Ascend NPU (torch_npu)"),
]


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("n_frames", (97, 200))
def test_capture_and_replay_match_eager(n_frames: int) -> None:
    # No manual torch.npu.config tweak here on purpose: the capture path
    # must self-configure the graph-capturable ACLNN runtime via
    # _prepare_speaker_embedding_graph_runtime, and this test would fail
    # if that self-configuration regressed.
    torch.manual_seed(0)
    wrapper = CampplusTorch(CAMPPlus(feat_dim=80, embedding_size=192).eval(), "npu")
    feat = torch.randn(n_frames, 80)

    with torch.inference_mode():
        out_capture = wrapper(feat)  # first call: capture + mandatory replay
        assert n_frames in wrapper._graphs, "capture failed; parity check is vacuous"
        eager_ref = wrapper.model(feat.unsqueeze(0).float().to("npu"))
        out_replay = wrapper(feat)  # second call: steady-state replay

    assert torch.allclose(out_capture.cpu(), eager_ref.cpu(), atol=2e-4, rtol=0)
    assert torch.allclose(out_replay.cpu(), eager_ref.cpu(), atol=2e-4, rtol=0)
    assert wrapper._replay_calls >= 2
    assert wrapper._eager_calls == 0
