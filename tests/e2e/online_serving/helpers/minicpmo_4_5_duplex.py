"""Shared fixtures and argument builders for MiniCPM-o 4.5 duplex E2E tests."""

from __future__ import annotations

import hashlib
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub import snapshot_download

from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = "openbmb/MiniCPM-o-4_5"
DEPLOY_CONFIG = modify_stage_config(
    get_deploy_config_path("minicpmo_4_5_duplex.yaml"),
    updates={
        "base_config": get_deploy_config_path("minicpmo_4_5.yaml"),
        # Cap per-stage KV so Thinker/Talker/Code2Wav share one GPU with
        # max_sessions=2. Talker must stay within its 4096 context so the
        # 0.5 GiB budget passes vLLM's min-KV check at init.
        "stages": {
            0: {"kv_cache_memory_bytes": 6 * 1024 * 1024 * 1024},
            1: {
                "max_model_len": 4096,
                "kv_cache_memory_bytes": 512 * 1024 * 1024,
            },
            2: {"kv_cache_memory_bytes": 256 * 1024 * 1024},
        },
        # Platform overrides apply after ordinary stage settings and would
        # otherwise reinstate the base CUDA 2 GiB Talker default.
        "platforms": {
            "cuda": {
                "stages": [
                    {
                        "stage_id": 1,
                        "kv_cache_memory_bytes": 512 * 1024 * 1024,
                    }
                ]
            }
        },
    },
)
CORE_DEPLOY_CONFIG = modify_stage_config(
    DEPLOY_CONFIG,
    updates={
        "stages": {
            0: {"enforce_eager": True},
            1: {"enforce_eager": True},
        }
    },
)
ASSET_DIR = Path(__file__).resolve().parents[3] / "assets" / "minicpmo_4_5"
RESPONSE_REQUIRED_WAV = ASSET_DIR / "response_required_16k.wav"
RESPONSE_REQUIRED_SHA256 = "2e5fd4eb3ee434ce107ee3a0591fa624a33f7683c7462f45fe651c443c9af941"
SOFT_INTERRUPT_WAV = ASSET_DIR / "soft_interrupt_16k.wav"
SOFT_INTERRUPT_SHA256 = "cadae6d0ddc510310f16f8775d6379f30a5195369ccb54ff10a8e3f9a1f4a2ea"
REF_AUDIO_RELATIVE_PATH = Path("assets") / "HT_ref_audio.wav"

SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=["--trust-remote-code"],
        ),
        id="three-stage-single-gpu",
    )
]
CORE_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=CORE_DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=["--trust-remote-code"],
        ),
        id="three-stage-single-gpu",
    )
]


def validated_wav(path: Path, expected_sha256: str) -> Path:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise AssertionError(
            f"MiniCPM-o duplex fixture SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with wave.open(str(path), "rb") as wav_file:
        actual_format = (
            wav_file.getframerate(),
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getcomptype(),
        )
    expected_format = (16_000, 1, 2, "NONE")
    if actual_format != expected_format:
        raise AssertionError(f"MiniCPM-o duplex fixture must be 16 kHz mono PCM16, got {actual_format}")
    return path


def validated_input_wav() -> Path:
    return validated_wav(RESPONSE_REQUIRED_WAV, RESPONSE_REQUIRED_SHA256)


def validated_soft_interrupt_wav() -> Path:
    return validated_wav(SOFT_INTERRUPT_WAV, SOFT_INTERRUPT_SHA256)


def resolve_ref_audio(model_prefix: str) -> Path:
    if model_prefix:
        model_root = Path(model_prefix) / MODEL
    else:
        model_root = Path(snapshot_download(MODEL, local_files_only=True))
    ref_audio = model_root / REF_AUDIO_RELATIVE_PATH
    if not ref_audio.is_file():
        raise FileNotFoundError(f"MiniCPM-o checkpoint ref audio is missing: {ref_audio}")
    return ref_audio


def realtime_url(omni_server) -> str:
    return f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1"


def demo_args(
    *,
    omni_server,
    input_wav: Path,
    ref_audio: Path,
    output_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        url=realtime_url(omni_server),
        model=omni_server.model,
        session_id=f"duplex-ci-single-{uuid.uuid4().hex}",
        input_wav=str(input_wav),
        ref_audio=str(ref_audio),
        turn_input_wav=[],
        output_dir=str(output_dir),
        output_audio_format="pcm16",
        chunk_ms=200,
        realtime_input=True,
        first_turn_ms=1400,
        turn_duration_ms=[],
        first_turn_transcript="duplex CI speech",
        omit_transcript_hints=False,
        validation_mode="response-required",
        temperature=0.0,
        scenario="sequential",
        require_audio=True,
        require_distinct_inputs=False,
        expect_empty_turn=[],
        short_ack_ms=350,
        turns=1,
        timeout_s=120.0,
        model_policy_settle_ms=2000,
    )


def multi_session_args(
    *,
    omni_server,
    input_wav: Path,
    ref_audio: Path,
    output_dir: Path,
    response_required: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        url=realtime_url(omni_server),
        model=omni_server.model,
        sessions=2,
        input_wav=str(input_wav),
        ref_audio=str(ref_audio),
        session_input_wav=[],
        session_expected_token=[],
        turn_input_wav=[],
        output_dir=str(output_dir),
        realtime_input=True,
        chunk_ms=200,
        turns=1,
        first_turn_ms=1400,
        turn_duration_ms=[],
        response_required=response_required,
        temperature=0.0 if response_required else None,
        disconnect_session_index=0,
        takeover_session_index=1,
        resume_after_ms=100,
        expire_session_index=None,
        expire_after_s=40.0,
        verify_admission_limit=None,
        model_policy_settle_ms=2000,
        timeout_s=180.0,
    )
