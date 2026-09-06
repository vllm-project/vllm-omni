# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for the Seed-TTS UTMOS scorer.

Two things make the published ``balacoon/utmos`` TorchScript archive unusable as
shipped: it was traced on CUDA and hard-codes ``cuda:0`` inside its methods, and
it normalizes its input by 32768 (it wants int16-valued samples) while the WER
pipeline hands it a ``[-1, 1]`` array.  Both are checked here on CPU only —
no CUDA, no NPU, and no network access to Hugging Face.

vllm stubs are installed by tests/benchmarks/conftest.py before collection.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

torch = pytest.importorskip("torch")


def _round_trip(module: torch.nn.Module) -> torch.jit.ScriptModule:
    """Script, serialize and reload — the shape a downloaded archive arrives in."""
    buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(module), buffer)
    buffer.seek(0)
    return torch.jit.load(buffer, map_location="cpu")


class _TracedOnCuda(torch.nn.Module):
    """Mimics the archive's root ``forward``: cast to cuda, then normalize."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moved = x.to(torch.device("cuda:0"))
        return torch.mean(torch.div(moved, 32768.0)).reshape(1)


class _Inner(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.device("cuda:0"))


class _Outer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.inner(x)).reshape(1)


class _NoNormalization(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x).reshape(1)


class _NormalizesByTwo(torch.nn.Module):
    """A hypothetical custom export: it normalizes, but not by the int16 factor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.div(x, 2.0)).reshape(1)


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="the baked cuda:0 constant resolves on a CUDA build; the defect is a no-CUDA one",
)
def test_the_archive_as_shipped_is_unusable_without_cuda():
    """A fresh instance, never retargeted: this is what the loader returns today.

    ``balacoon/utmos`` itself raises ``NotImplementedError`` here (verified on an
    Ascend build), and that is a ``RuntimeError`` subclass; a locally scripted
    stand-in surfaces the same failed ``aten::to`` as the interpreter's plain
    ``RuntimeError``. Matching on the device name is what pins the cause.
    """
    model = _round_trip(_TracedOnCuda())
    with pytest.raises(RuntimeError, match="cuda"):
        model(torch.zeros(1, 16))


def test_retargeting_makes_it_run_on_the_loaded_device():
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    # A separate instance on purpose: TorchScript caches an execution plan on the
    # first call, so the rewrite has to happen before the module is ever run --
    # which is what ``_ensure_utmos_jit_model`` does.
    model = _round_trip(_TracedOnCuda())
    assert seed_tts_eval._utmos_retarget_device_constants(model, "cpu") == 1
    assert float(model(torch.zeros(1, 16)).item()) == 0.0


def test_retarget_reaches_nested_submodules():
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    model = _round_trip(_Outer())
    assert seed_tts_eval._utmos_retarget_device_constants(model, "cpu") == 1
    assert float(model(torch.ones(1, 4)).item()) == pytest.approx(1.0)


def test_retarget_is_a_noop_when_the_graph_already_names_the_target():
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    model = _round_trip(_TracedOnCuda())
    assert seed_tts_eval._utmos_retarget_device_constants(model, "cuda:0") == 0


def test_resolve_device_uses_the_documented_env_var(monkeypatch):
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    monkeypatch.setenv("SEED_TTS_UTMOS_DEVICE", "cuda:1")
    assert seed_tts_eval._utmos_resolve_device(torch) == "cuda:1"


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch):
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    monkeypatch.delenv("SEED_TTS_UTMOS_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert seed_tts_eval._utmos_resolve_device(torch) == "cpu"


def test_input_scale_is_read_off_the_graph():
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    model = _round_trip(_TracedOnCuda())
    assert seed_tts_eval._utmos_detect_input_scale(model) == pytest.approx(32768.0)


def test_input_scale_ignores_an_export_that_normalizes_by_something_else():
    """A custom ``SEED_TTS_UTMOS_HF_REPO`` graph must not be silently rescaled."""
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    model = _round_trip(_NormalizesByTwo())
    assert seed_tts_eval._utmos_detect_input_scale(model) == pytest.approx(1.0)


def test_input_scale_is_one_when_the_export_does_not_normalize():
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    model = _round_trip(_NoNormalization())
    assert seed_tts_eval._utmos_detect_input_scale(model) == pytest.approx(1.0)


def test_predict_hands_the_model_the_amplitude_it_normalizes_by(monkeypatch):
    """With the scale applied, the export sees the waveform, not 1/32768 of it."""
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    model = _round_trip(_TracedOnCuda())
    seed_tts_eval._utmos_retarget_device_constants(model, "cpu")
    scale = seed_tts_eval._utmos_detect_input_scale(model)
    monkeypatch.setattr(seed_tts_eval, "_ensure_utmos_jit_model", lambda: model)
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_input_scale", scale)

    rng = np.random.default_rng(0)
    wav = rng.uniform(-1.0, 1.0, size=16000).astype(np.float32)

    got = seed_tts_eval._utmos_predict_f32_16k(wav)
    assert got == pytest.approx(float(np.mean(wav)), abs=1e-6)

    # Without the scale the model is fed a waveform 32768x too quiet.
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_input_scale", 1.0)
    attenuated = seed_tts_eval._utmos_predict_f32_16k(wav)
    assert attenuated == pytest.approx(float(np.mean(wav)) / scale, abs=1e-9)


def test_predict_returns_none_when_the_archive_is_unavailable(monkeypatch):
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    monkeypatch.setattr(seed_tts_eval, "_ensure_utmos_jit_model", lambda: None)
    assert seed_tts_eval._utmos_predict_f32_16k(np.zeros(16, dtype=np.float32)) is None
