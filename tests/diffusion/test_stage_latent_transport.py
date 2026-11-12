# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


import pytest
import torch

from vllm_omni.diffusion.stage_latent_transport import StageLatentTransport, unpack_stage_latents

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def transport():
    value = StageLatentTransport(2, "video")
    yield value
    value.close()


def _prompt():
    return {
        "prompt": "",
        "additional_information": {
            "minimax_h3_decode": {
                "video_latents": [torch.arange(6).reshape(1, 2, 3), torch.ones(1, 2, 3)],
                "height": 32,
                "width": 48,
                "preencode_mp4": False,
            }
        },
    }


def test_shared_memory_preserves_order_and_single_consumer(transport):
    original = _prompt()
    packed = transport.pack("request", original)
    assert "video_latents" not in packed["additional_information"]["minimax_h3_decode"]
    restored = unpack_stage_latents(packed, "request", 2, "MiniMaxH3VideoDecoderPipeline")
    options = restored["additional_information"]["minimax_h3_decode"]
    for expected, received in zip(
        original["additional_information"]["minimax_h3_decode"]["video_latents"], options["video_latents"], strict=True
    ):
        assert torch.equal(expected, received)
    assert options["stage_latent_transfer"]["output_indices"] == [0, 1]
    with pytest.raises(ValueError, match="missing or already consumed"):
        unpack_stage_latents(packed, "request", 2, "MiniMaxH3VideoDecoderPipeline")
    assert transport.release("request")["latent_shm_bytes"] > 0
    assert not transport.pending


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", "other"),
        ("modality", "audio"),
        ("payload_version", 999),
        ("parent_stage_id", 0),
        ("stage_id", 3),
    ],
)
def test_wrong_envelope_is_rejected(transport, field, value):
    packed = transport.pack("request", _prompt())
    packed["_minimax_h3_stage_latent_shm"][field] = value
    with pytest.raises(ValueError, match="mismatch"):
        unpack_stage_latents(packed, "request", 2, "MiniMaxH3VideoDecoderPipeline")
    transport.release("request")
    assert not transport.pending


def test_cancel_unlinks_unconsumed_segment(transport):
    packed = transport.pack("request", _prompt())
    transport.release("request")
    with pytest.raises(ValueError, match="missing or already consumed"):
        unpack_stage_latents(packed, "request", 2, "MiniMaxH3VideoDecoderPipeline")


def test_failed_put_is_cleaned_up(transport, monkeypatch):
    cleanup = []
    monkeypatch.setattr(transport.connector, "put", lambda *args: (False, 0, None))
    monkeypatch.setattr(transport.connector, "cleanup", lambda key: cleanup.append(key))
    with pytest.raises(RuntimeError, match="Failed to write"):
        transport.pack("request", _prompt())
    assert len(cleanup) == 1
    assert not transport.pending


def test_duplicate_submission_is_rejected(transport):
    transport.pack("request", _prompt())
    with pytest.raises(ValueError, match="Duplicate"):
        transport.pack("request", _prompt())


def test_reference_cannot_claim_a_different_output_count(transport):
    packed = transport.pack("request", _prompt())
    packed["_minimax_h3_stage_latent_shm"]["output_count"] = 1
    with pytest.raises(ValueError, match="count mismatch"):
        unpack_stage_latents(packed, "request", 2, "MiniMaxH3VideoDecoderPipeline")


def test_legacy_prompt_has_no_transport():
    prompt = _prompt()
    assert unpack_stage_latents(prompt, "request", 2, "MiniMaxH3DecoderPipeline") is prompt
