# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""UTMOS input-amplitude regressions; no accelerator or model download needed."""

from __future__ import annotations

import io

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

torch = pytest.importorskip("torch")

from vllm_omni.benchmarks.data_modules import utmos_jit  # noqa: E402


class _Export(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class _BakedCudaInput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(device="cuda:0", dtype=torch.float32) / 32768).mean().reshape(1)


@pytest.mark.parametrize(
    "fn, scale",
    [
        (lambda x: x.to(dtype=torch.float32) / torch.tensor(32768.0), 32768.0),
        (lambda x: x, 1.0),
        (lambda x: x / 2.0, 1.0),
        (lambda x: x.square().mean() / 32768.0, 1.0),
        (lambda x: x / 32768.0 + x, 1.0),
    ],
    ids=["int16-input", "no-division", "other-divisor", "feature-division", "mixed-input"],
)
def test_input_contract(fn, scale):
    audio = torch.tensor([[0.25, 0.5]])
    # Frozen traces carry tensor Constants, like the real balacoon export.
    model = torch.jit.freeze(torch.jit.trace(_Export(fn).eval(), audio))
    buffer = io.BytesIO()
    torch.jit.save(model, buffer)
    buffer.seek(0)
    model = torch.jit.load(buffer, map_location="cpu")
    assert utmos_jit._detect_utmos_input_scale(model) == scale
    torch.testing.assert_close(utmos_jit._UTMOSScorer(model)(audio), fn(audio * scale), rtol=0, atol=0)


def test_loaded_scorer_restores_amplitude_without_mutating_or_accumulating(tmp_path):
    path = tmp_path / "utmos.jit"
    torch.jit.save(torch.jit.script(_BakedCudaInput()), str(path))
    # Exercise real load, device retargeting, scale detection and warmup.
    scorer = utmos_jit.load_utmos_jit(str(path), "cpu")
    audio = torch.tensor([[0.25, 0.5]], dtype=torch.float32)
    original = audio.clone()
    for _ in range(2):
        with torch.no_grad():
            actual = scorer(audio)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, original.mean().reshape(1), rtol=0, atol=0)
        torch.testing.assert_close(audio, original, rtol=0, atol=0)
