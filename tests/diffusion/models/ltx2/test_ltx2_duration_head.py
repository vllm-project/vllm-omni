# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.ltx2_duration_head import LTX2DurationHead

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_duration_head() -> LTX2DurationHead:
    return LTX2DurationHead(
        video_cross_attention_dim=12,
        audio_cross_attention_dim=8,
        pooler_hidden_dim=16,
        num_queries=1,
        num_pooler_heads=4,
        mlp_hidden_dim=10,
    )


def test_duration_head_accepts_video_and_audio_connector_tokens():
    head = _tiny_duration_head()

    duration = head(torch.randn(2, 5, 12), torch.randn(2, 7, 8))

    assert duration.shape == (2,)
    assert torch.all(duration > 0)


def test_duration_head_requires_at_least_one_modality():
    with pytest.raises(ValueError, match="video_tokens and/or audio_tokens"):
        _tiny_duration_head()()


def test_duration_head_state_dict_matches_converted_checkpoint_names():
    names = set(_tiny_duration_head().state_dict())

    assert {
        "video_input_proj.weight",
        "audio_input_proj.weight",
        "attention_pooler.query_tokens",
        "attention_pooler.to_q.weight",
        "attention_pooler.to_k.weight",
        "attention_pooler.to_v.weight",
        "attention_pooler.to_out.weight",
        "mlp_hidden.weight",
        "mlp_out.weight",
    } <= names


@pytest.mark.parametrize(
    ("seconds", "expected_frames"),
    [
        (5.2, 121),
        (0.1, 25),
        (30.0, 473),
    ],
)
def test_predict_num_frames_clamps_and_snaps_to_vae_grid(monkeypatch, seconds, expected_frames):
    head = _tiny_duration_head()
    monkeypatch.setattr(head, "forward", lambda *_args, **_kwargs: torch.tensor([seconds]))

    frames = head.predict_num_frames(
        torch.empty(1, 1, 12),
        frame_rate=24.0,
        temporal_compression_ratio=8,
    )

    assert frames == expected_frames
    assert (frames - 1) % 8 == 0


def test_predict_num_frames_rejects_multiple_prompts(monkeypatch):
    head = _tiny_duration_head()
    monkeypatch.setattr(head, "forward", lambda *_args, **_kwargs: torch.tensor([2.0, 3.0]))

    with pytest.raises(ValueError, match="one prompt"):
        head.predict_num_frames(
            torch.empty(2, 1, 12),
            frame_rate=24.0,
            temporal_compression_ratio=8,
        )
