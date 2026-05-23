# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for FunCineForge dubbing/TTS model.

Tests verify the /v1/audio/speech endpoint works correctly with
FunCineForge, using official reference audio and face embeddings from
the FunCineForge repo.  FunCineForge requires reference audio + ref_text
for voice cloning, and optionally face embeddings for lip-sync dubbing.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import functools
import io
import pickle
from urllib.request import urlopen

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.core_model, pytest.mark.tts]

MODEL = "FunAudioLLM/Fun-CineForge"

# Official test data from FunCineForge GitHub repo
_GITHUB_DATA_BASE = "https://raw.githubusercontent.com/FunAudioLLM/FunCineForge/main/exps/data"

# Reference audio for voice cloning
REF_AUDIO_URL = f"{_GITHUB_DATA_BASE}/ref.wav"

# Vocal audio (source speaker) — used to estimate speech_len
VOCAL_AUDIO_URL = f"{_GITHUB_DATA_BASE}/clipped/en_monologue_1.wav"

# Face embedding for lip-sync conditioning (pickle format from official repo)
FACE_EMB_URL = f"{_GITHUB_DATA_BASE}/embs_video/en_monologue_1.pkl"

# Clue text matching official demo.jsonl en_monologue_1
REF_TEXT = (
    "A single middle-aged male speaker describes a business or "
    "construction requirement with a practical and matter-of-fact tone."
)

# FunCineForge token rate: 25 Hz (codec frames per second)
_TOKEN_RATE = 25


@functools.lru_cache(maxsize=1)
def _load_vocal_audio() -> tuple[np.ndarray, int]:
    """Download official vocal audio to estimate speech_len."""
    with urlopen(VOCAL_AUDIO_URL, timeout=60) as resp:
        data = resp.read()
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


@functools.lru_cache(maxsize=1)
def _load_face_embedding_pkl() -> dict:
    """Download official face embedding pkl and parse it."""
    with urlopen(FACE_EMB_URL, timeout=60) as resp:
        return pickle.loads(resp.read())  # noqa: S301


def _create_face_npz(dest: str) -> int:
    """Convert official pkl face embedding to .npz and return speech_len.

    ``load_face_embedding()`` in serving layer uses ``np.load(allow_pickle=False)``
    which requires .npz format, while the official repo ships .pkl files.
    """
    face_dict = _load_face_embedding_pkl()
    embeddings = np.asarray(face_dict["embeddings"])
    face_indices = np.asarray(face_dict["faceI"])
    np.savez(dest, embeddings=embeddings, faceI=face_indices)

    vocal_audio, vocal_sr = _load_vocal_audio()
    speech_len = int(len(vocal_audio) / vocal_sr * _TOKEN_RATE)
    return speech_len


def get_stage_config(name: str = "funcineforge.yaml"):
    """Get the deploy config path for FunCineForge."""
    return get_deploy_config_path(name)


def get_prompt(prompt_type="en"):
    """Official demo texts from FunCineForge repo."""
    prompts = {
        "en": (
            "Every closet on a Carnival cruise ship. To make the numbers work, I needed a lot of cedar, fast and cheap."
        ),
        "zh": "这是一个电影配音模型的测试。",
    }
    return prompts.get(prompt_type, prompts["en"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config(),
            server_args=[
                "--trust-remote-code",
                "--disable-log-stats",
                "--no-async-chunk",
            ],
        ),
        id="funcineforge",
    )
]

tts_async_chunk_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_stage_config(),
            server_args=[
                "--trust-remote-code",
                "--disable-log-stats",
            ],
        ),
        id="funcineforge_async_chunk",
    )
]


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_en_sync(omni_server, openai_client) -> None:
    """
    Test FunCineForge dubbing with English text via OpenAI API (sync).
    Uses official ref.wav from FunCineForge repo for voice cloning.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_async_chunk_server_params, indirect=True)
def test_funcineforge_dubbing_en_async(omni_server, openai_client) -> None:
    """
    Test FunCineForge dubbing with async_chunk streaming.
    Uses official ref.wav from FunCineForge repo for voice cloning.

    Deploy Setting: funcineforge.yaml with async_chunk: true
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio (streamed)
    Input Setting: stream=True
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": True,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_zh_sync(omni_server, openai_client) -> None:
    """
    Test FunCineForge dubbing with Chinese text via OpenAI API (sync).
    Verifies bilingual (zh_en) model handles Chinese input.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


# ---------------------------------------------------------------------------
# FunCineForge cinematic dubbing parameter tests
# ---------------------------------------------------------------------------


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_speech_type(omni_server, openai_client) -> None:
    """
    Test FunCineForge dubbing with speech_type parameter.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text + speech_type
    Output Modal: audio
    Input Setting: stream=False, speech_type='旁白' (narration)
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "speech_type": "旁白",
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_with_face(omni_server, openai_client, tmp_path) -> None:
    """
    Test FunCineForge dubbing with face_path for lip-sync conditioning.

    Downloads official en_monologue_1 face embedding from FunCineForge repo,
    converts from pkl to npz format, and passes the path to the server.
    Verifies the full face_path → load_face_embedding → model pipeline.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text + face_path + speech_len
    Output Modal: audio
    Input Setting: stream=False
    """
    face_file = str(tmp_path / "face.npz")
    speech_len = _create_face_npz(face_file)

    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "face_path": face_file,
        "speech_len": speech_len,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_with_face_pkl(omni_server, openai_client, tmp_path) -> None:
    """
    Test FunCineForge dubbing with pkl-format face embedding (official repo format).

    The serving layer's load_face_embedding supports both .npz and .pkl files.
    This test uses the raw pkl from the official repo to verify pkl support.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text + face_path (pkl) + speech_len
    Output Modal: audio
    Input Setting: stream=False
    """
    face_dict = _load_face_embedding_pkl()
    face_pkl_file = str(tmp_path / "face.pkl")
    with open(face_pkl_file, "wb") as f:
        pickle.dump(face_dict, f)

    vocal_audio, vocal_sr = _load_vocal_audio()
    speech_len = int(len(vocal_audio) / vocal_sr * _TOKEN_RATE)

    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "face_path": face_pkl_file,
        "speech_len": speech_len,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_with_dialogue_metadata(omni_server, openai_client, tmp_path) -> None:
    """
    Test FunCineForge dubbing with dialogue metadata and speech_type.

    Exercises the HF demo-style parameters: face embedding + speech_type
    tag + multi-speaker dialogue metadata (start/duration/spk/gender/age).
    Uses official en_monologue_1 test data from FunCineForge repo.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text + face_path + speech_type + dialogue + speech_len
    Output Modal: audio
    Input Setting: stream=False
    """
    face_file = str(tmp_path / "face.npz")
    speech_len = _create_face_npz(face_file)

    dialogue = [
        {"start": 0.0, "duration": 5.74, "spk": 1, "gender": "male", "age": "middle-aged"},
    ]

    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "face_path": face_file,
        "speech_len": speech_len,
        "speech_type": "独白",
        "dialogue": dialogue,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_funcineforge_dubbing_full_cinematic(omni_server, openai_client, tmp_path) -> None:
    """
    Full cinematic dubbing: face + speech_type + dialogue + speech_len.

    Exercises the complete FunCineForge dubbing pipeline with all cinematic
    parameters using official test data: face embedding (lip-sync), speech
    type tag, and dialogue metadata (multi-speaker time/gender/age tags).

    Deploy Setting: funcineforge.yaml
    Input Modal: text + ref_audio + ref_text + face_path + speech_type + dialogue + speech_len
    Output Modal: audio
    Input Setting: stream=False
    """
    face_file = str(tmp_path / "face.npz")
    speech_len = _create_face_npz(face_file)

    dialogue = [
        {"start": 0.0, "duration": 3.5, "spk": 1, "gender": "男", "age": "中年"},
    ]

    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "face_path": face_file,
        "speech_len": speech_len,
        "speech_type": "独白",
        "dialogue": dialogue,
    }
    openai_client.send_audio_speech_request(request_config)


# ---------------------------------------------------------------------------
# FunCineForge video preprocessing tests (require demo checkout + moviepy)
# ---------------------------------------------------------------------------


def _has_video_preprocess() -> bool:
    """Check if video preprocessing dependencies are available."""
    try:
        import moviepy  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("FUNCINEFORGE_DEMO_ROOT") is not None


# Official test video clip from FunCineForge GitHub repo
VIDEO_CLIP_URL = f"{_GITHUB_DATA_BASE}/clipped/en_monologue_1.mp4"


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
@pytest.mark.skipif(
    not _has_video_preprocess(),
    reason="Video preprocessing requires moviepy + FUNCINEFORGE_DEMO_ROOT",
)
def test_funcineforge_video_dubbing_sync(omni_server, openai_client) -> None:
    """
    Test FunCineForge end-to-end video dubbing via /v1/audio/speech.

    Sends a video URL with start/end timestamps, speaker metadata, and
    lets the server perform video preprocessing (clip extraction, face
    embedding extraction, reference audio extraction). Matches the
    HuggingFace Space demo flow.

    Requires: moviepy + FUNCINEFORGE_DEMO_ROOT environment variable.

    Deploy Setting: funcineforge.yaml
    Input Modal: text + video + video_start + video_end + speaker metadata
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "video": VIDEO_CLIP_URL,
        "video_start": 0.0,
        "video_end": 5.0,
        "ref_text": REF_TEXT,
        "speech_type": "独白",
        "speaker_gender": "male",
        "speaker_age": "middle-aged",
    }
    openai_client.send_audio_speech_request(request_config)
