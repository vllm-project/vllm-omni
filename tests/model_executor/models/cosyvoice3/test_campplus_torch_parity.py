# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU onnxruntime vs on-device torch parity for the CAM++ speaker embedding.

Loads the shipped ``campplus.onnx`` twice — once as a CPU onnxruntime session,
once through :func:`get_campplus_torch` — and compares the L2-normalized
192-d embeddings (cosine similarity, max abs diff). The input goes through the
exact production transform of ``StepAudio2Token2WavCore._prepare_prompt``:
``kaldi.fbank(num_mel_bins=80, dither=0)`` minus the per-utterance mean, fed
with a deterministic synthetic 16 kHz waveform so the test needs no audio
asset. Requires a real Ascend NPU and the onnx asset, so the default CI run
skips it; ``CAMPPLUS_ONNX_PATH`` overrides the asset path.

CI note: no ``test-npu-*.yml`` workflow collects this file yet and the
default asset path is machine-local, so in CI it currently always skips.
It is meant as a device-side regression gate — run it on any machine with
an A2 (single-card, 910B) queue slot and the model snapshot. Wiring a CI
job that points ``CAMPPLUS_ONNX_PATH`` at the HF snapshot is left for a
follow-up once the NPU queue layout settles.
"""

import hashlib
import math
import os

import onnxruntime
import pytest
import torch
from torchaudio.compliance import kaldi

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import get_campplus_torch

_DEFAULT_ONNX = "/dataset/models/MiniCPM-o-4_5/assets/token2wav/campplus.onnx"
ONNX_PATH = os.environ.get("CAMPPLUS_ONNX_PATH", _DEFAULT_ONNX)

# Token lengths mirroring the PR description's parity runs.
_TOKEN_LENGTHS = (97, 200, 501)

# 25 ms window / 10 ms shift at 16 kHz, matching kaldi.fbank defaults.
_FRAME_SHIFT = 160
_WINDOW = 400


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
    pytest.mark.skipif(not os.path.isfile(ONNX_PATH), reason=f"missing {ONNX_PATH}"),
]


def _make_fbank(n_frames: int) -> torch.Tensor:
    """Deterministic waveform -> the exact serving-time fbank transform."""
    n_samples = (n_frames - 1) * _FRAME_SHIFT + _WINDOW
    t = torch.arange(n_samples, dtype=torch.float32) / 16000
    gen = torch.Generator().manual_seed(0)
    wav = 0.3 * torch.sin(2 * math.pi * 220 * t) + 0.2 * torch.sin(2 * math.pi * 1500 * t)
    wav = wav + 0.05 * torch.randn(n_samples, generator=gen)
    feat = kaldi.fbank(wav.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    return feat - feat.mean(dim=0, keepdim=True)


def _asset_revision(path: str) -> str:
    """Short sha256 of the onnx asset so results pin the exact weights version."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def _run_ort(feat: torch.Tensor) -> torch.Tensor:
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(ONNX_PATH, sess_options=option, providers=["CPUExecutionProvider"])
    (out,) = session.run(None, {session.get_inputs()[0].name: feat.numpy().astype("float32")[None]})
    return torch.from_numpy(out[0])


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("n_frames", _TOKEN_LENGTHS)
def test_campplus_torch_matches_onnx_cpu(n_frames: int) -> None:
    feat = _make_fbank(n_frames)

    ort_emb = _run_ort(feat)

    campplus = get_campplus_torch(ONNX_PATH, "npu")
    with torch.inference_mode():
        torch_emb = campplus(feat).float().cpu()[0]

    ort_n = torch.nn.functional.normalize(ort_emb, dim=0)
    torch_n = torch.nn.functional.normalize(torch_emb, dim=0)
    cosine = float(torch.dot(ort_n, torch_n))
    maxdiff = float(torch.max(torch.abs(ort_n - torch_n)))

    print(f"campplus onnx[{_asset_revision(ONNX_PATH)}] T={n_frames}: cosine={cosine:.6f} maxdiff={maxdiff:.2e}")
    assert cosine > 0.9999
    assert maxdiff < 1e-3
