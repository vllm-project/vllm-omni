# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""S3 quantize capture/replay output must match the eager path on real NPU.

Mirrors ``test_speaker_embedding_graph_npu.py`` for the S3 short-audio
graph cache: the first ``_quantize_s3_graph`` call exercises capture plus
the mandatory post-capture replay (fixed in 31e9b208d), the second the
steady-state replay path; both must return the same discrete codes and
lengths as ``S3TokenizerV2.quantize``'s short-audio branch on the same
input (the CPU suite pins the inlined eager copy; this pins the graph).
"""

import threading

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    StepAudio2Token2WavCore,
)

s3tokenizer = pytest.importorskip("s3tokenizer")


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


def _core_on_npu(model) -> StepAudio2Token2WavCore:
    # Skip the full model bootstrap (yaml, campplus, vocoder weights); only
    # audio_tokenizer and device take part in the quantize graph paths.
    core = StepAudio2Token2WavCore.__new__(StepAudio2Token2WavCore)
    torch.nn.Module.__init__(core)
    core._models_loaded = True
    core._audio_tokenizer = model
    core.device = torch.device("npu")
    # Graph bookkeeping normally declared in __init__ (skipped above).
    core._s3_graphs = {}
    core._s3_failed = set()
    core._s3_graph_lock = threading.Lock()
    core._s3_graph_done = None
    core._s3_replay_calls = 0
    core._s3_eager_calls = 0
    return core


@hardware_test(res={"npu": "A2"}, num_cards=1)
def test_s3_capture_and_replay_match_eager() -> None:
    # No manual torch.npu.config tweak here on purpose: the capture path
    # must self-configure the graph-capturable ACLNN runtime via
    # _prepare_s3_quantize_graph_runtime, and this test would fail
    # if that self-configuration regressed.
    torch.manual_seed(0)
    model = s3tokenizer.model_v2.S3TokenizerV2("speech_tokenizer_v2_25hz").eval().to("npu")
    core = _core_on_npu(model)

    g = torch.Generator().manual_seed(0)
    mels = torch.randn(1, 128, 50, generator=g).to("npu")
    mels_lens = torch.tensor([50], dtype=torch.long, device="npu")

    with torch.inference_mode():
        capture_code, capture_len = core._quantize_s3_graph(mels, mels_lens)
        assert 50 in core._s3_graphs, "capture failed; parity check is vacuous"
        lib_code, lib_len = model.quantize(mels, mels_lens)
        replay_code, replay_len = core._quantize_s3_graph(mels, mels_lens)

    assert torch.equal(capture_code.cpu(), lib_code.cpu())
    assert torch.equal(capture_len.cpu(), lib_len.cpu())
    assert torch.equal(replay_code.cpu(), lib_code.cpu())
    assert torch.equal(replay_len.cpu(), lib_len.cpu())
    assert core._s3_replay_calls >= 2
    assert core._s3_eager_calls == 0
