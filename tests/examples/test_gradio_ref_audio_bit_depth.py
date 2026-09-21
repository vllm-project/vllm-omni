# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-depth handling in the Gradio demos' reference-audio encoders.

``gr.Audio(type="numpy")`` decodes an upload with
``pydub.AudioSegment.from_file()`` and hands the callback
``np.array(segment.get_array_of_samples())``, i.e. samples at the *source* bit
depth: int8 for 8-bit, int16 for 16-bit, and int32 padded to full int32 scale
for 24- and 32-bit sources. The demo helpers that turn that array into a base64
WAV data URL therefore have to rescale it. A bare ``astype(np.int16)`` is a
narrowing integer cast, which wraps modulo 2**16 and destroys the waveform
before it ever leaves the demo.

Regression test for https://github.com/vllm-project/vllm-omni/issues/7899.
"""

import base64
import contextlib
import importlib.machinery
import importlib.util
import io
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]

_ONLINE_SERVING = Path(__file__).resolve().parents[2] / "examples" / "online_serving"

_SAMPLE_RATE = 24000


def _sr_first(fn: Callable) -> Callable:
    """Helper takes a single ``(sample_rate, samples)`` tuple."""
    return lambda samples, sample_rate: fn((sample_rate, samples))


def _samples_first(fn: Callable) -> Callable:
    """Helper takes a single ``(samples, sample_rate)`` tuple."""
    return lambda samples, sample_rate: fn((samples, sample_rate))


def _two_args(fn: Callable) -> Callable:
    """Helper takes ``samples`` and ``sample_rate`` as separate arguments."""
    return lambda samples, sample_rate: fn(samples, sample_rate)


# Every demo that turns a ``gr.Audio(type="numpy")`` upload into a base64 WAV
# data URL: (test id, path under examples/online_serving, attribute, calling convention).
_ENCODERS = [
    ("qwen3_tts", "text_to_speech/qwen3_tts/tts_common.py", "encode_audio_to_base64", _sr_first),
    ("fish_speech", "text_to_speech/fish_speech/gradio_demo.py", "encode_audio_to_base64", _sr_first),
    ("glm_tts", "text_to_speech/glm_tts/gradio_demo.py", "encode_audio_to_base64", _sr_first),
    ("moss_tts_nano", "text_to_speech/moss_tts_nano/gradio_demo.py", "encode_audio_to_base64", _sr_first),
    ("voxcpm2", "text_to_speech/voxcpm2/gradio_demo.py", "_encode_audio", _sr_first),
    ("audio8_tts", "text_to_speech/audio8_tts/gradio_demo.py", "encode_audio_to_base64", _sr_first),
    ("qwen3_omni", "qwen3_omni/gradio_demo.py", "audio_to_base64_data_url", _samples_first),
    ("qwen2_5_omni", "qwen2_5_omni/gradio_demo.py", "audio_to_base64_data_url", _samples_first),
    ("minicpmo", "minicpmo/gradio_demo.py", "audio_to_base64_data_url", _two_args),
    ("aura_omni", "aura_omni/gradio_demo.py", "_audio_to_data_url", _sr_first),
]

# Stubbed for every demo, not just when missing. A missing ``gradio`` is re-raised by the
# demos as a plain ``ImportError`` with an install hint, so the loop below cannot learn its
# name; and importing ``vllm_omni`` -- which one demo does for a default-path helper the
# encoder never calls -- drags the whole package into a CPU-only example test.
_ALWAYS_STUBBED = ("gradio", "vllm_omni")


class _StubModule(ModuleType):
    """A stand-in module whose every attribute is a mock.

    The empty ``__path__`` means a submodule import (``from fastapi.responses import ...``)
    fails with its own dotted name, which the loop below then stubs in turn.
    """

    __path__: list[str] = []

    def __getattr__(self, attr: str) -> MagicMock:
        return MagicMock(name=f"{self.__name__}.{attr}")


def _stub(name: str) -> ModuleType:
    stub = _StubModule(name)
    stub.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    return stub


@contextlib.contextmanager
def _demo_module(rel_path: str) -> Iterator[ModuleType]:
    """Execute a demo script with its module-scope imports stubbed out.

    The demos import a UI toolkit and API clients the encoder never touches: ``gradio``
    lives in the optional ``[demo]`` extra, and the omni ones add ``openai``, ``torch``,
    ``httpx`` and ``fastapi``. Skipping on those would leave this module a green no-op
    wherever the extras are absent, which is exactly how CI installs the project. So stub
    the names in ``_ALWAYS_STUBBED`` plus whatever else turns out to be missing -- learned
    one name per failed import -- and leave anything else that is installed alone.
    """
    path = _ONLINE_SERVING / rel_path
    # Several demos share the basename ``gradio_demo``; give each a unique key.
    name = f"_vllm_omni_demo_{rel_path.replace('/', '_').removesuffix('.py')}"
    saved: dict[str, ModuleType | None] = {}

    def install_stub(dotted: str) -> None:
        saved.setdefault(dotted, sys.modules.get(dotted))
        sys.modules[dotted] = _stub(dotted)

    try:
        for dotted in _ALWAYS_STUBBED:
            install_stub(dotted)
        while True:
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            saved.setdefault(name, sys.modules.get(name))
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as exc:
                if exc.name is None or exc.name in saved:
                    raise
                install_stub(exc.name)
                continue
            yield module
            return
    finally:
        for dotted, original in saved.items():
            if original is None:
                sys.modules.pop(dotted, None)
            else:
                sys.modules[dotted] = original


def _decode_data_url(data_url: str) -> tuple[int, np.ndarray]:
    """Decode a ``data:audio/wav;base64,...`` URL back to int16 PCM."""
    prefix = "data:audio/wav;base64,"
    assert data_url.startswith(prefix), data_url[:64]
    wav_bytes = base64.b64decode(data_url[len(prefix) :])
    samples, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    return sample_rate, samples


def _reference_waveform(num_samples: int = 2048) -> np.ndarray:
    """A deterministic tone at 0.8 full scale."""
    t = np.arange(num_samples) / _SAMPLE_RATE
    return 0.8 * np.sin(2 * np.pi * 220.0 * t)


def _quantize(waveform: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Quantize a [-1, 1] waveform to ``dtype`` the way a decoder of that bit depth would."""
    if np.issubdtype(dtype, np.floating):
        return waveform.astype(dtype)
    full_scale = np.iinfo(dtype).max + 1
    return np.clip(np.rint(waveform * full_scale), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def _normalize(samples: np.ndarray) -> np.ndarray:
    """Map int or float samples onto a common [-1, 1] scale for comparison."""
    if np.issubdtype(samples.dtype, np.floating):
        return np.clip(samples.astype(np.float64), -1.0, 1.0)
    return samples.astype(np.float64) / (np.iinfo(samples.dtype).max + 1)


_ENCODER_PARAMS = [pytest.param(*spec[1:], id=spec[0]) for spec in _ENCODERS]

# pydub reports int8 for 8-bit sources, int16 for 16-bit, and int32 for 24- and 32-bit ones;
# float arrays reach the same helpers from the local-file paths.
_UPLOAD_DTYPES = [np.int32, np.int8, np.int16, np.float32, np.float64]


@pytest.mark.parametrize(("rel_path", "attr", "wrapper"), _ENCODER_PARAMS)
@pytest.mark.parametrize("dtype", _UPLOAD_DTYPES, ids=lambda d: np.dtype(d).name)
def test_reference_audio_survives_encoding(rel_path, attr, wrapper, dtype):
    """The encoded data URL must carry the uploaded waveform, whatever its bit depth."""
    samples = _quantize(_reference_waveform(), np.dtype(dtype))

    with _demo_module(rel_path) as demo:
        sample_rate, decoded = _decode_data_url(wrapper(getattr(demo, attr))(samples, _SAMPLE_RATE))

    assert sample_rate == _SAMPLE_RATE
    np.testing.assert_allclose(_normalize(decoded), _normalize(samples), atol=1e-3)


@pytest.mark.parametrize(("rel_path", "attr", "wrapper"), _ENCODER_PARAMS)
def test_int16_upload_is_bit_exact(rel_path, attr, wrapper):
    """A 16-bit upload already matches the wire format and must pass through untouched."""
    samples = _quantize(_reference_waveform(), np.dtype(np.int16))

    with _demo_module(rel_path) as demo:
        _, decoded = _decode_data_url(wrapper(getattr(demo, attr))(samples, _SAMPLE_RATE))

    np.testing.assert_array_equal(decoded, samples)


@pytest.mark.parametrize(("rel_path", "attr", "wrapper"), _ENCODER_PARAMS)
def test_wide_integer_upload_does_not_wrap(rel_path, attr, wrapper):
    """Pins the exact failure in #7899: truncating a full-scale int32 inverts loud samples."""
    # Values chosen so that the low 16 bits have the opposite sign of the sample.
    samples = np.array([2_000_000_000, -2_000_000_000, 1_717_986_817], dtype=np.int32)

    with _demo_module(rel_path) as demo:
        _, decoded = _decode_data_url(wrapper(getattr(demo, attr))(samples, _SAMPLE_RATE))

    assert np.sign(decoded).tolist() == np.sign(samples).tolist()
    assert abs(int(decoded[0])) > 16_000, "a near-full-scale input must stay near full scale"
