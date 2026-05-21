# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline E2E smoke tests for FunCineForge zero-shot dubbing inference.

Uses official FunCineForge test data (reference audio, face embeddings)
from https://github.com/FunAudioLLM/FunCineForge/tree/main/exps/data

Verifies:
- Model loads and generates audio output with reference audio
- Full dubbing with face embeddings produces valid audio
- Audio duration is within expected bounds
- Sample rate matches expected 24kHz
"""

from __future__ import annotations

import functools
import io
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import soundfile as sf
from huggingface_hub import snapshot_download

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.core_model, pytest.mark.tts]
from vllm_omni.model_executor.models.funcineforge.config import (
    FunCineForgeConfig,
)

MODEL = "FunAudioLLM/Fun-CineForge"
MODEL_DIR_ENV = "VLLM_OMNI_FUNCINEFORGE_MODEL_DIR"

# Official test data from FunCineForge GitHub repo
_GITHUB_DATA_BASE = "https://raw.githubusercontent.com/FunAudioLLM/FunCineForge/main/exps/data"

# Reference audio for voice cloning
REF_AUDIO_URL = f"{_GITHUB_DATA_BASE}/ref.wav"

# Vocal audio (source speaker audio for dubbing)
VOCAL_AUDIO_URL = f"{_GITHUB_DATA_BASE}/clipped/en_monologue_1.wav"

# Face embedding for lip-sync conditioning
FACE_EMB_URL = f"{_GITHUB_DATA_BASE}/embs_video/en_monologue_1.pkl"

# Official demo text and clue from demo.jsonl entry
SYNTH_TEXT = (
    "Every closet on a Carnival cruise ship. To make the numbers work, I needed a lot of cedar, fast and cheap."
)
CLUE_TEXT = (
    "A single middle-aged male speaker describes a business or "
    "construction requirement with a practical and matter-of-fact tone. "
    "His voice is deep and slightly gravelly, maintaining a professional "
    "and informative demeanor throughout the segment."
)

ASYNC_CHUNK_MODES = [
    pytest.param(False, id="sync"),
    pytest.param(True, id="async_chunk"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_ref_audio() -> tuple[np.ndarray, int]:
    """Download and cache official reference audio."""
    with urlopen(REF_AUDIO_URL, timeout=60) as resp:
        data = resp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


@functools.lru_cache(maxsize=1)
def _load_vocal_audio() -> tuple[np.ndarray, int]:
    """Download and cache official vocal audio (source speaker)."""
    with urlopen(VOCAL_AUDIO_URL, timeout=60) as resp:
        data = resp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


@functools.lru_cache(maxsize=1)
def _load_face_embedding() -> bytes:
    """Download and cache official face embedding pkl file."""
    with urlopen(FACE_EMB_URL, timeout=60) as resp:
        return resp.read()


@functools.lru_cache(maxsize=1)
def _resolve_model_dir() -> Path:
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(snapshot_download(MODEL, allow_patterns=["*"]))


def _concat_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        tensors = []
        for t in audio_val:
            if t is None:
                continue
            if hasattr(t, "detach"):
                t = t.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            if hasattr(t, "float"):
                t = t.float()
            if isinstance(t, torch.Tensor):
                tensors.append(t.reshape(-1))
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)

    if hasattr(audio_val, "detach"):
        audio_val = audio_val.detach()
    if hasattr(audio_val, "cpu"):
        audio_val = audio_val.cpu()
    if hasattr(audio_val, "float"):
        audio_val = audio_val.float()
    if hasattr(audio_val, "numpy"):
        audio_val = audio_val.numpy()
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _assert_valid_audio(outputs, *, min_dur: float = 0.5, max_dur: float = 65.0):
    """Common audio output assertions."""
    config = FunCineForgeConfig()

    assert outputs, "No outputs returned"
    audio_mm = outputs[0].multimodal_output
    assert "audio" in audio_mm, "No audio output found"

    audio = _concat_audio(audio_mm["audio"])
    assert audio.size > 0, "Generated audio is empty"

    sr_val = audio_mm.get("sr", config.sample_rate)
    if isinstance(sr_val, list) and sr_val:
        sr_val = sr_val[-1]
    if hasattr(sr_val, "item"):
        sr_val = sr_val.item()
    sr = int(sr_val)
    assert sr == 24000, f"Unexpected sample_rate={sr}"

    duration_s = audio.size / sr
    assert min_dur <= duration_s <= max_dur, f"Unexpected duration={duration_s:.3f}s (samples={audio.size}, sr={sr})"
    return audio, sr


# ---------------------------------------------------------------------------
# Test: basic TTS with reference audio (no face embedding)
# ---------------------------------------------------------------------------


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("async_chunk", ASYNC_CHUNK_MODES)
def test_funcineforge_offline_tts(async_chunk: bool) -> None:
    """FunCineForge TTS with reference audio should produce valid audio."""
    ref_audio, ref_sr = _load_ref_audio()
    model_dir = _resolve_model_dir()

    with OmniRunner(
        str(model_dir),
        seed=42,
        stage_configs_path=get_deploy_config_path("funcineforge.yaml"),
        async_chunk=async_chunk,
        stage_init_timeout=300,
    ) as omni_runner:
        sampling_params_list = omni_runner.get_default_sampling_params_list()

        inputs = [
            {
                "prompt": SYNTH_TEXT,
                "multi_modal_data": {"audio": (ref_audio, ref_sr)},
                "modalities": ["audio"],
                "mm_processor_kwargs": {
                    "prompt_text": CLUE_TEXT,
                    "sample_rate": ref_sr,
                },
            }
        ]

        outputs = omni_runner.omni.generate(inputs, sampling_params_list)
        _assert_valid_audio(outputs)


# ---------------------------------------------------------------------------
# Test: full dubbing with face embedding
# ---------------------------------------------------------------------------


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("async_chunk", ASYNC_CHUNK_MODES)
def test_funcineforge_offline_dubbing_with_face(async_chunk: bool) -> None:
    """FunCineForge dubbing with face embedding should produce valid audio.

    This test exercises the full dubbing pipeline:
    text + reference audio (voice cloning) + face embedding (lip-sync).
    Uses official en_monologue_1 test data from FunCineForge repo.
    """
    import tempfile

    import torch

    from vllm_omni.model_executor.models.funcineforge.utils import (
        load_face_embedding,
    )

    ref_audio, ref_sr = _load_ref_audio()
    model_dir = _resolve_model_dir()

    # Download face embedding to temp file (load_face_embedding expects path)
    face_data = _load_face_embedding()
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        f.write(face_data)
        face_path = f.name

    try:
        # Estimate speech length from vocal audio duration
        vocal_audio, vocal_sr = _load_vocal_audio()
        config = FunCineForgeConfig()
        speech_len = int(len(vocal_audio) / vocal_sr * config.token_rate)

        face_emb = load_face_embedding(face_path, speech_len, face_size=config.face_size)
        assert isinstance(face_emb, torch.Tensor)
        assert face_emb.shape == (1, speech_len, config.face_size)

        with OmniRunner(
            str(model_dir),
            seed=42,
            stage_configs_path=get_deploy_config_path("funcineforge.yaml"),
            async_chunk=async_chunk,
            stage_init_timeout=300,
        ) as omni_runner:
            sampling_params_list = omni_runner.get_default_sampling_params_list()

            inputs = [
                {
                    "prompt": SYNTH_TEXT,
                    "multi_modal_data": {"audio": (ref_audio, ref_sr)},
                    "modalities": ["audio"],
                    "mm_processor_kwargs": {
                        "prompt_text": CLUE_TEXT,
                        "sample_rate": ref_sr,
                        "face_embedding": face_emb,
                    },
                }
            ]

            outputs = omni_runner.omni.generate(inputs, sampling_params_list)
            _assert_valid_audio(outputs)
    finally:
        os.unlink(face_path)
