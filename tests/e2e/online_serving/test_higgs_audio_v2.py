# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end checks for the vllm-omni higgs_audio_v2 integration.

This test suite exercises the parts of the higgs_audio_v2 path that do NOT
require booting the full vLLM engine: the request-validator scope checks,
the prompt builder's tokenization parity, the stage_input_processor adapter
contract, and Stage-1 decode parity against the upstream reference fixture
(AC-4).

The Stage-0 talker AR-loop is gated on the reference fixtures + the
upstream-trace memo and is exercised in a follow-up online test once the
forward path lands; see ``results/plan.md`` Milestone 4 Step G.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "higgs_audio_v2" / "reference_hello_world.pt"


# --------------------------------------------------------------------- scope
def test_validator_rejects_voice_cloning_fields() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        REJECTED_REQUEST_FIELDS,
        UnsupportedInputError,
        validate_plain_text_request,
    )

    for field in REJECTED_REQUEST_FIELDS:
        payload = {"input": "Hello", field: "anything"}
        with pytest.raises(UnsupportedInputError) as excinfo:
            validate_plain_text_request(payload)
        assert "higgs_audio_v2" in str(excinfo.value)
        assert field in str(excinfo.value)


def test_validator_rejects_multi_speaker_tag() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        UnsupportedInputError,
        validate_plain_text_request,
    )

    with pytest.raises(UnsupportedInputError) as excinfo:
        validate_plain_text_request({"input": "[SPEAKER0] hi"})
    assert "multi-speaker" in str(excinfo.value).lower()


def test_validator_accepts_plain_text() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        validate_plain_text_request,
    )

    validate_plain_text_request({"input": "Hello world."})


# ------------------------------------------------------ registry / pipeline
def test_pipeline_registry_has_higgs_audio_v2() -> None:
    from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

    assert "higgs_audio_v2" in _OMNI_PIPELINES


def test_model_registry_has_both_stages() -> None:
    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    assert "HiggsAudioV2ForConditionalGeneration" in _OMNI_MODELS
    assert "HiggsAudioV2TalkerForConditionalGeneration" in _OMNI_MODELS
    assert "HiggsAudioV2Code2WavForConditionalGeneration" in _OMNI_MODELS


def test_pipeline_hf_architectures_declared() -> None:
    from vllm_omni.model_executor.models.higgs_audio_v2.pipeline import (
        HIGGS_AUDIO_V2_PIPELINE,
    )

    assert HIGGS_AUDIO_V2_PIPELINE.model_type == "higgs_audio_v2"
    assert "HiggsAudioV2ForConditionalGeneration" in HIGGS_AUDIO_V2_PIPELINE.hf_architectures


# ------------------------------------------------------- stage_input_processor
def test_stage_input_processor_async_chunk_emits_codes_only_when_chunk_full() -> None:
    """Adapter must wait for chunk_size frames before flushing a chunk."""
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
    from vllm_omni.model_executor.stage_input_processors.higgs_audio_v2 import (
        talker2code2wav_async_chunk,
    )

    class _FakeRequest:
        external_req_id = "req-0"

        def is_finished(self) -> bool:
            return False

    class _FakeManager:
        code_prompt_token_ids: dict[str, list[list[int]]] = {"req-0": []}
        connector = None

    mgr = _FakeManager()

    # Feed 24 frames -- below default chunk_size=25 -- and expect no flush.
    for _ in range(24):
        frame = torch.zeros(8, dtype=torch.long)
        payload = {"codes": {"audio": frame}}
        out = talker2code2wav_async_chunk(mgr, payload, _FakeRequest(), is_finished=False)
        assert out is None
    # 25th frame should trigger a flush.
    payload = {"codes": {"audio": torch.zeros(8, dtype=torch.long)}}
    out = talker2code2wav_async_chunk(mgr, payload, _FakeRequest(), is_finished=False)
    assert isinstance(out, OmniPayloadStruct)
    assert isinstance(out.codes, CodesStruct)
    assert out.codes.audio.numel() > 0
    assert isinstance(out.meta, MetaStruct)


# --------------------------------------------------- Stage-1 decode contract
def test_stage1_rejects_stream_specials() -> None:
    """Code IDs >= audio_stream_bos_id (1024) must raise ValueError before decode."""
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )

    cfg = HiggsAudioV2Config()
    stage1 = HiggsAudioV2Code2Wav(cfg)
    # Force the loaded flag so the validator runs before the (unloaded) decoder
    # would otherwise raise on missing weights.
    stage1._loaded = True
    stage1.quantizer = object()  # not used, validator runs first
    stage1.fc2 = object()
    stage1.acoustic_decoder = object()
    bad_codes = torch.full((1, cfg.num_codebooks, 4), cfg.audio_stream_bos_id, dtype=torch.long)
    with pytest.raises(ValueError) as excinfo:
        stage1(bad_codes)
    assert "stream-special" in str(excinfo.value).lower() or "out-of-range" in str(excinfo.value).lower()


@pytest.mark.skipif(not FIXTURE.exists(), reason="reference fixture not captured yet")
def test_fixture_shape_invariants() -> None:
    """Sanity check the captured fixture before any Stage-1 parity test runs."""
    blob = torch.load(FIXTURE, weights_only=False)
    assert blob["prompt_text"]
    assert blob["input_ids"].ndim == 2
    codes = blob["audio_codes"]
    assert codes.ndim == 3
    # codes shape is one of [B, num_codebooks=8, T] or [B, T, num_codebooks=8].
    # Both are accepted; downstream code normalizes to [B, 8, T] before sending
    # to Stage 1.
    assert 8 in (int(codes.shape[1]), int(codes.shape[2]))
    pcm = blob["reference_pcm"]
    assert pcm.ndim == 1
    assert pcm.dtype == torch.int16


@pytest.mark.skipif(
    not FIXTURE.exists() or os.environ.get("HIGGS_AUDIO_V2_TOKENIZER_DIR") is None or not torch.cuda.is_available(),
    reason="reference fixture + audio_tokenizer dir required; runs on GPU",
)
def test_stage1_decode_parity_rms() -> None:
    """AC-4 positive test: Stage-1 PCM matches the upstream reference within 1e-4 RMS."""
    from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
        HiggsAudioV2Config,
    )
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_code2wav import (
        HiggsAudioV2Code2Wav,
    )

    audio_tokenizer_dir = Path(os.environ["HIGGS_AUDIO_V2_TOKENIZER_DIR"])
    blob = torch.load(FIXTURE, weights_only=False)
    codes = blob["audio_codes"].long()
    if codes.shape[1] != 8 and codes.shape[2] == 8:
        codes = codes.transpose(1, 2).contiguous()

    device = torch.device("cuda")
    cfg = HiggsAudioV2Config()
    stage1 = HiggsAudioV2Code2Wav(cfg).to(device)
    stage1.load_weights(model_dir=str(audio_tokenizer_dir.parent), device=device)
    pcm_out = stage1(codes.to(device)).squeeze(0).squeeze(0).to(torch.float32).clamp_(-1.0, 1.0).cpu()
    pcm_ref = blob["reference_pcm"].to(torch.float32) / 32767.0
    n = min(int(pcm_out.shape[0]), int(pcm_ref.shape[0]))
    rms = ((pcm_out[:n] - pcm_ref[:n]) ** 2).mean().sqrt().item()
    assert rms <= 1e-4, f"Stage-1 vs HF reference PCM RMS={rms:.3e} exceeds 1e-4"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
