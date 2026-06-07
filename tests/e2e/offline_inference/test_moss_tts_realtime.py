# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for MOSS-TTS-Realtime (MossTTSRealtime, 1.7B).

Uses the standard omni_runner + pytestmark pattern (one module-scoped engine
per file, skill invariant I4).  The delay-model variant lives in the sibling
file test_moss_tts.py.

The request format mirrors end2end.py:_build_realtime_prompt exactly:
  1. snapshot_download to locate the MossTTSRealtimeProcessor module.
  2. MOSS-Audio-Tokenizer (separate HF repo) to encode the reference clip.
  3. Build the (L, 17) int grid; split into prompt_token_ids + codes.ref +
     ids.all (remaining text tokens fed one-per-step during decode).

Set MOSS_TTS_SKIP_ON_NET_FAIL=1 to skip in air-gapped environments.

No determinism test: MossTTSRealtime uses a local depth transformer that is
sensitive to async scheduling timing; waveform reproducibility is not
guaranteed across back-to-back calls.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "OpenMOSS-Team/MOSS-TTS-Realtime"
SAMPLE_RATE = 24_000

REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/main/assets/audio/reference_zh_1.wav"

_STAGE0_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    max_tokens=512,
    seed=42,
    detokenize=False,
)
_STAGE1_PARAMS = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_tokens=65536,
    seed=42,
    detokenize=False,
)
_SAMPLING = [_STAGE0_PARAMS, _STAGE1_PARAMS]

# ---------------------------------------------------------------------------
# Deploy config
# ---------------------------------------------------------------------------


def _get_test_config() -> str:
    """Derive a CI-friendly config from moss_tts_realtime.yaml.

    Reduces Stage 0 gpu_memory_utilization from 0.60 → 0.45 to leave headroom
    for Stage 1 (0.12) on a shared L4/A10G.  max_num_seqs=1 keeps per-test
    peak memory predictable.
    """
    return modify_stage_config(
        get_deploy_config_path("moss_tts_realtime.yaml"),
        updates={
            "stages": {
                0: {
                    "max_num_seqs": 1,
                    "gpu_memory_utilization": 0.45,
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# pytestmark — one engine for the whole module
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", [(MODEL, _get_test_config())], indirect=True),
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref_audio_path(tmp_path_factory) -> str:
    """Download the upstream reference clip once per test module.

    Set MOSS_TTS_SKIP_ON_NET_FAIL=1 to skip in air-gapped environments.
    """
    target = tmp_path_factory.mktemp("moss_tts_realtime_ref") / "zh_1.wav"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            target.write_bytes(resp.read())
    except Exception as exc:
        msg = f"Cannot fetch reference clip {REF_AUDIO_URL}: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not target.exists() or target.stat().st_size == 0:
        pytest.fail(f"Reference clip empty after download: {target}")
    return str(target)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ref_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load and resample reference audio to (1, T) float32 at target_sr."""
    data, sr = sf.read(path, always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T.astype("float32"))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav  # (1, T)


def _build_request(ref_audio_path: str, text: str) -> dict:
    """Build a MOSS-TTS-Realtime request.

    Mirrors end2end.py:_build_realtime_prompt exactly:
      1. Locate MossTTSRealtimeProcessor in the HF snapshot.
      2. Encode ref audio via MOSS-Audio-Tokenizer (separate repo).
      3. Build the (L, 17) int grid and return prompt_token_ids +
         additional_information: {codes.ref, ids.all}.

    Frees the codec model before returning to avoid competing with the
    running vllm engine for GPU/CPU memory.
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    # Step 1: locate the realtime processor module in the snapshot.
    try:
        snap_dir = Path(snapshot_download(repo_id=MODEL))
    except Exception as exc:
        msg = f"Cannot locate snapshot for {MODEL}: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)

    proc_module_path = snap_dir / "processing_mossttsrealtime.py"
    if not proc_module_path.exists():
        pytest.fail(f"Realtime processor module missing at {proc_module_path}")

    mod_name = "_moss_tts_realtime_proc"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, proc_module_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        mod = sys.modules[mod_name]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    realtime_processor = mod.MossTTSRealtimeProcessor(tokenizer=tokenizer)

    # Step 2: encode the reference audio clip via MOSS-Audio-Tokenizer.
    try:
        codec = AutoModel.from_pretrained("OpenMOSS-Team/MOSS-Audio-Tokenizer", trust_remote_code=True).to("cpu").eval()
    except Exception as exc:
        msg = f"Cannot load MOSS-Audio-Tokenizer: {exc}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)

    ref_wav = _load_ref_audio(ref_audio_path)
    wav_2d = ref_wav if ref_wav.dim() == 2 else ref_wav.unsqueeze(0)
    with torch.no_grad():
        enc = codec.batch_encode([wav_2d.squeeze(0)], num_quantizers=16)
    codes = enc.audio_codes[:, 0, : int(enc.audio_codes_lengths[0].item())]
    audio_tokens = codes.transpose(0, 1).contiguous().cpu().numpy()  # (T, 16)
    del codec
    gc.collect()

    # Step 3: build the (L, 17) prefill grid.
    system_grid = realtime_processor.make_ensemble(prompt_audio_tokens=audio_tokens)

    assistant_header_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    assistant_grid = np.full((len(assistant_header_ids), 17), 1024, dtype=np.int64)
    assistant_grid[:, 0] = assistant_header_ids

    text_ids_only = tokenizer.encode(text, add_special_tokens=False)
    if not text_ids_only:
        pytest.fail(f"Empty text after tokenization: {text!r}")

    PREFILL_MAX_TEXT = 12
    cur_len = min(len(text_ids_only), PREFILL_MAX_TEXT)
    text_grid = np.full((cur_len, 17), 1024, dtype=np.int64)
    text_grid[:, 0] = text_ids_only[:cur_len]
    text_grid[-1, 1] = 1025  # audio_bos at the last prefilled text token

    grid = np.concatenate([system_grid, assistant_grid, text_grid], axis=0)
    prompt_token_ids = grid[:, 0].tolist()
    audio_codes_tensor = torch.from_numpy(grid[:, 1:].astype(np.int64).copy())  # (L, 16)
    remaining_text_ids = list(text_ids_only[cur_len:])

    additional_info: dict = {"codes": {"ref": audio_codes_tensor}}
    if remaining_text_ids:
        additional_info["ids"] = {"all": remaining_text_ids}

    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": additional_info,
    }


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    """Run one request and return (waveform_cpu, sample_rate)."""
    chunks: list[torch.Tensor] = []
    sr_final = SAMPLE_RATE
    for out in omni_runner.omni.generate(request, _SAMPLING):
        mm = out.multimodal_output
        if not mm:
            continue
        audio = mm.get("audio")
        if audio is None:
            audio = mm.get("model_outputs")
        if audio is None:
            continue
        sr = mm.get("sr")
        if sr is not None:
            sr_final = int(sr.item() if hasattr(sr, "item") else sr)
        if isinstance(audio, list):
            audio = torch.cat(
                [t.reshape(-1) for t in audio if isinstance(t, torch.Tensor) and t.numel() > 0],
                dim=0,
            )
        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            chunks.append(audio.reshape(-1).cpu())
    if not chunks:
        raise AssertionError("No audio output received from generate()")
    return torch.cat(chunks, dim=0), sr_final


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_realtime_english(omni_runner: OmniRunner, ref_audio_path: str) -> None:
    """MossTTSRealtime: English voice_clone produces non-empty 24 kHz audio."""
    req = _build_request(ref_audio_path, "This is a real-time TTS streaming test.")
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_moss_tts_realtime_chinese(omni_runner: OmniRunner, ref_audio_path: str) -> None:
    """MossTTSRealtime: Chinese input produces non-empty audio."""
    req = _build_request(ref_audio_path, "你好，这是语音合成测试。")
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)
