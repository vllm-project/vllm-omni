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

A determinism test is included: the talker's local depth transformer samples
audio codes with ``torch.multinomial`` outside vLLM's seeded sampler, so the
per-request seed re-seeds a generator (``_seeded_generator`` in the talker) and
two same-seed runs reproduce.  Prefix caching is disabled in ``_get_test_config``
so the cached-prefill path can't perturb the logits.
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

from tests.helpers.audio_output import audio_from_mm, collect_audio_from_outputs
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
    for Stage 1 (0.12) on a shared L4/A10G.  ``max_num_seqs`` is capped at 2
    rather than the YAML default of 8: it keeps per-test peak memory
    predictable while still letting ``test_moss_tts_realtime_batch`` co-schedule
    its two requests, which is the only way that test can observe the
    cross-request row misalignment it guards (at 1 the scheduler serialises
    them and the guard cannot fail).  Prefix caching is disabled so two identical-seed
    runs are bit-reproducible (the cached-prefill path can perturb logits enough
    to shift sampling/EOS) — required by ``test_moss_tts_realtime_deterministic``.
    """
    return modify_stage_config(
        get_deploy_config_path("moss_tts_realtime.yaml"),
        updates={
            "stages": {
                0: {
                    "max_num_seqs": 2,
                    "gpu_memory_utilization": 0.45,
                    "enable_prefix_caching": False,
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# pytestmark — one engine for the whole module
# ---------------------------------------------------------------------------

# Same marks as the sibling delay file test_moss_tts_expansion.py: selected by
# the nightly "TTS · Function Test" (-m "full_model and L4 and tts"). The
# module-wide skip stays until #4700 (moss_tts_realtime architecture is not
# registered with Transformers in CI) is resolved; it is not fixed here.
pytestmark = [
    pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4700"),
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_runner",
        [(MODEL, _get_test_config(), {"trust_remote_code": True})],
        indirect=True,
    ),
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


def _build_request(ref_audio_path: str, text: str, seed: int = 42) -> dict:
    """Build a MOSS-TTS-Realtime request.

    Mirrors end2end.py:_build_realtime_prompt exactly:
      1. Locate MossTTSRealtimeProcessor in the HF snapshot.
      2. Encode ref audio via MOSS-Audio-Tokenizer (separate repo).
      3. Build the (L, 17) int grid and return prompt_token_ids +
         additional_information: {codes.ref, seed, ids.all}.

    ``seed`` is forwarded under ``additional_information["seed"]`` so the talker's
    audio sampler re-seeds its generator per request (reproducible audio).

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

    additional_info: dict = {"codes": {"ref": audio_codes_tensor}, "seed": [seed]}
    if remaining_text_ids:
        additional_info["ids"] = {"all": remaining_text_ids}

    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": additional_info,
    }


def _collect_audio(omni_runner: OmniRunner, request: dict) -> tuple[torch.Tensor, int]:
    """Run one request and return (waveform_cpu, sample_rate)."""
    return collect_audio_from_outputs(omni_runner.omni.generate(request, _SAMPLING), SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_realtime_english(omni_runner: OmniRunner, ref_audio_path: str) -> None:
    """MossTTSRealtime: English voice_clone produces non-empty 24 kHz audio."""
    req = _build_request(ref_audio_path, "This is a real-time TTS streaming test.")
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.numel() > 0, "Audio tensor is empty"
    assert not torch.all(audio == 0), "Audio is silence"


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_realtime_chinese(omni_runner: OmniRunner, ref_audio_path: str) -> None:
    """MossTTSRealtime: Chinese input produces non-empty audio."""
    req = _build_request(ref_audio_path, "你好，这是语音合成测试。")
    audio, sr = _collect_audio(omni_runner, req)

    assert sr == SAMPLE_RATE
    assert audio.numel() > 0
    assert not torch.all(audio == 0)


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_realtime_deterministic(omni_runner: OmniRunner, ref_audio_path: str) -> None:
    """MossTTSRealtime: same seed yields identical waveforms.

    The local-transformer samples audio codes with ``torch.multinomial`` outside
    vLLM's seeded sampler; the per-request seed re-seeds a generator so two
    same-seed runs reproduce exactly (prefix caching is disabled in
    ``_get_test_config`` so the cached-prefill path can't perturb the logits).
    """
    sp_audio1, _ = _collect_audio(
        omni_runner, _build_request(ref_audio_path, "Determinism check for realtime.", seed=7)
    )
    sp_audio2, _ = _collect_audio(
        omni_runner, _build_request(ref_audio_path, "Determinism check for realtime.", seed=7)
    )

    assert sp_audio1.shape == sp_audio2.shape, f"Shapes differ: {sp_audio1.shape} vs {sp_audio2.shape}"
    assert torch.allclose(sp_audio1, sp_audio2, atol=1e-4), "Same seed produced different audio"


@hardware_test(res={"cuda": "L4"})
def test_moss_tts_realtime_batch(omni_runner: OmniRunner, ref_audio_path: str) -> None:
    """MossTTSRealtime: two concurrent requests with different text produce
    distinct audio.

    Regression guard for the cross-request audio corruption fixed in #4415
    (broadcast / row-misalignment): with that bug, request 1 would receive
    request 0's audio. Audio sampling is seeded (deterministic), so distinctness
    here reflects genuine per-request content, not RNG noise.

    The guard only bites if the two requests actually share a batch — run
    sequentially they each get their own correct audio and the assertion passes
    whether or not the bug is present. ``_get_test_config`` therefore sets
    ``max_num_seqs: 2`` on Stage 0; do not lower it back to 1.

    Coverage limitation: synchronous ``generate(requests, ...)`` prefills then
    decodes both requests in lockstep, so every step is all-prefill or
    all-decode — it does not exercise the mixed prefill+decode row-misalignment
    path (which needs staggered submission the batch API can't express).
    """
    requests = [
        _build_request(ref_audio_path, "Hello, how are you today?", seed=42),
        _build_request(ref_audio_path, "The weather is sunny and warm.", seed=42),
    ]
    # generate() returns a flat list of OmniRequestOutput; each carries
    # .request_id ("<idx>_<uuid>"). Group audio chunks per request index so a
    # per-request mix-up shows up as identical waveforms across the two requests.
    chunks_by_req: dict[int, list[torch.Tensor]] = {}
    for out in omni_runner.omni.generate(requests, _SAMPLING):
        idx = int(out.request_id.split("_", 1)[0])
        chunk = audio_from_mm(out.multimodal_output)
        if chunk is not None and chunk.numel() > 0:
            chunks_by_req.setdefault(idx, []).append(chunk)

    assert set(chunks_by_req) == {0, 1}, f"Expected audio for requests 0 and 1, got {sorted(chunks_by_req)}"
    a = torch.cat(chunks_by_req[0], dim=0)
    b = torch.cat(chunks_by_req[1], dim=0)
    assert a.numel() > 0 and b.numel() > 0, "An output is empty"
    if a.numel() == b.numel():
        assert not torch.equal(a, b), "Both requests returned identical audio (batch concurrency bug)"
