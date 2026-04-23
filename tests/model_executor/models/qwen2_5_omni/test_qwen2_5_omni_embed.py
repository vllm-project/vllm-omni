# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for Qwen2.5-Omni embed_input_ids to verify embeddings are
correctly assigned to audio/image/video token positions.

Regression test for: https://github.com/vllm-project/vllm/issues/34506
  - Non-interleaved mixed modalities (audio + image + video) should correctly
    assign audio embeddings to audio positions, image to image, video to video.
  - Interleaved (use_audio_in_video) should also work correctly.

Pure embedding helpers below are inlined from upstream
``vllm.model_executor.models.qwen2_5_omni_thinker`` / ``interfaces`` / ``utils``
so this module does not import ``qwen2_5_vl`` / ``conv`` (avoids duplicate
``CustomOp`` registration when multiple vLLM copies are on ``sys.path``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Fake token IDs
AUDIO_TOKEN_ID = 1001
IMAGE_TOKEN_ID = 1002
VIDEO_TOKEN_ID = 1003
TEXT_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Inlined embedding logic (no vLLM model graph import)
# ---------------------------------------------------------------------------


def check_interleaved_audio_video(
    is_video: torch.Tensor,
    is_audio: torch.Tensor,
    num_video: int,
    num_audio: int,
) -> bool:
    """Return True only for true audio-in-video interleaving (no text gaps in span)."""
    if num_video == 0 or num_audio == 0:
        return False

    video_pos = is_video.nonzero(as_tuple=True)[0]
    audio_pos = is_audio.nonzero(as_tuple=True)[0]

    if not (video_pos[0].item() < audio_pos[-1].item() and audio_pos[0].item() < video_pos[-1].item()):
        return False

    combined_start = min(video_pos[0].item(), audio_pos[0].item())
    combined_end = max(video_pos[-1].item(), audio_pos[-1].item())
    total_in_range = combined_end - combined_start + 1
    return (num_video + num_audio) == total_in_range


def merge_interleaved_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: Sequence[Tensor] | Tensor | tuple[Tensor, ...],
    is_video: torch.Tensor,
    is_audio: torch.Tensor,
    is_multimodal: torch.Tensor,
    num_video: int,
    num_audio: int,
) -> torch.Tensor:
    """Scatter video/audio (and other) embeddings for interleaved audio-in-video."""
    video_embeds: list[torch.Tensor] = []
    audio_embeds: list[torch.Tensor] = []
    other_embeds: list[torch.Tensor] = []
    video_remaining = num_video
    audio_remaining = num_audio

    for emb in multimodal_embeddings:
        n = emb.shape[0]
        if video_remaining > 0 and n <= video_remaining:
            video_embeds.append(emb)
            video_remaining -= n
        elif audio_remaining > 0 and n <= audio_remaining:
            audio_embeds.append(emb)
            audio_remaining -= n
        else:
            other_embeds.append(emb)

    if video_embeds:
        inputs_embeds[is_video] = torch.cat(video_embeds, dim=0)
    if audio_embeds:
        inputs_embeds[is_audio] = torch.cat(audio_embeds, dim=0)
    if other_embeds:
        other_mask = is_multimodal & ~is_video & ~is_audio
        inputs_embeds[other_mask] = torch.cat(other_embeds, dim=0)

    return inputs_embeds


def _flatten_embeddings(embeddings: Sequence[Tensor] | Tensor | tuple[Tensor, ...]) -> Tensor:
    """Match vLLM ``_flatten_embeddings`` (``vllm/model_executor/models/utils.py``)."""
    if isinstance(embeddings, torch.Tensor):
        return embeddings.flatten(0, -2)
    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: Sequence[Tensor] | Tensor | tuple[Tensor, ...],
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """In-place scatter of flattened multimodal embeddings (same contract as vLLM utils)."""
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype
    inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
    return inputs_embeds


def _embed_text_input_ids(
    model: Any,
    input_ids: Tensor,
    embed_input_ids: Any,
    *,
    is_multimodal: Tensor | None,
) -> Tensor:
    if is_multimodal is not None and getattr(model, "_has_oov_mm_tokens", False):
        in_vocab_ids = input_ids.masked_fill(
            is_multimodal.to(device=input_ids.device, non_blocking=True),
            0,
        )
        return embed_input_ids(in_vocab_ids)
    return embed_input_ids(input_ids)


def qwen25_omni_thinker_embed_input_ids(
    model: Any,
    input_ids: torch.Tensor,
    multimodal_embeddings: Sequence[Tensor] | Tensor | tuple[Tensor, ...] | None = None,
    *,
    is_multimodal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same control flow as ``Qwen2_5OmniThinkerForConditionalGeneration.embed_input_ids`` (subset used in tests)."""
    if multimodal_embeddings is None or is_multimodal is None:
        return model.get_language_model().embed_input_ids(input_ids)

    inputs_embeds = _embed_text_input_ids(
        model,
        input_ids,
        model.get_language_model().embed_input_ids,
        is_multimodal=is_multimodal,
    )

    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    video_token_id = model.config.video_token_index
    audio_token_id = model.config.audio_token_index

    input_ids_cpu = input_ids.cpu()
    is_video = is_multimodal & (input_ids_cpu == video_token_id)
    is_audio = is_multimodal & (input_ids_cpu == audio_token_id)

    num_video = is_video.sum().item()
    num_audio = is_audio.sum().item()

    if check_interleaved_audio_video(is_video, is_audio, num_video, num_audio):
        inputs_embeds = _embed_text_input_ids(
            model,
            input_ids,
            model.get_language_model().embed_input_ids,
            is_multimodal=is_multimodal,
        )
        return merge_interleaved_embeddings(
            inputs_embeds,
            multimodal_embeddings,
            is_video,
            is_audio,
            is_multimodal,
            num_video,
            num_audio,
        )

    inputs_embeds = _embed_text_input_ids(
        model,
        input_ids,
        model.get_language_model().embed_input_ids,
        is_multimodal=is_multimodal,
    )
    if is_multimodal is None:
        raise ValueError("`embed_input_ids` requires `is_multimodal` when multimodal_embeddings is set.")
    return merge_multimodal_embeddings(
        inputs_embeds,
        multimodal_embeddings,
        is_multimodal,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_token_seq(audio_n: int, image_n: int, video_n: int, text_prefix: int = 3, text_sep: int = 2):
    """
    Build a flat token sequence:
      [text_prefix] [AUDIO * audio_n] [text_sep] [IMAGE * image_n]
      [text_sep] [VIDEO * video_n] [text_sep]
    Returns (input_ids tensor, is_multimodal mask, positions dict).
    """
    tokens = (
        [TEXT_TOKEN_ID] * text_prefix
        + [AUDIO_TOKEN_ID] * audio_n
        + [TEXT_TOKEN_ID] * text_sep
        + [IMAGE_TOKEN_ID] * image_n
        + [TEXT_TOKEN_ID] * text_sep
        + [VIDEO_TOKEN_ID] * video_n
        + [TEXT_TOKEN_ID] * text_sep
    )
    input_ids = torch.tensor(tokens)
    is_multimodal = (input_ids == AUDIO_TOKEN_ID) | (input_ids == IMAGE_TOKEN_ID) | (input_ids == VIDEO_TOKEN_ID)
    return input_ids, is_multimodal


def make_interleaved_seq(video_chunks: list[int], audio_chunks: list[int], text_prefix: int = 2):
    """
    Build an interleaved sequence like use_audio_in_video:
      [text] [V*v0] [A*a0] [V*v1] [A*a1] ...
    """
    tokens = [TEXT_TOKEN_ID] * text_prefix
    for v, a in zip(video_chunks, audio_chunks):
        tokens += [VIDEO_TOKEN_ID] * v + [AUDIO_TOKEN_ID] * a
    input_ids = torch.tensor(tokens)
    is_multimodal = (input_ids == VIDEO_TOKEN_ID) | (input_ids == AUDIO_TOKEN_ID)
    return input_ids, is_multimodal


# ---------------------------------------------------------------------------
# Tests for check_interleaved_audio_video
# ---------------------------------------------------------------------------


class TestCheckInterleavedAudioVideo:
    def test_non_interleaved_audio_then_video(self):
        """Audio entirely before video -> not interleaved."""
        input_ids, is_multimodal = make_token_seq(5, 0, 4)
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert not check_interleaved_audio_video(is_video, is_audio, is_video.sum().item(), is_audio.sum().item())

    def test_non_interleaved_with_image(self):
        """Audio + image + video (the mixed_modalities case) -> not interleaved."""
        input_ids, is_multimodal = make_token_seq(5, 4, 6)
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert not check_interleaved_audio_video(is_video, is_audio, is_video.sum().item(), is_audio.sum().item())

    def test_no_audio(self):
        """Video only -> not interleaved."""
        input_ids, is_multimodal = make_token_seq(0, 0, 6)
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert not check_interleaved_audio_video(is_video, is_audio, is_video.sum().item(), is_audio.sum().item())

    def test_interleaved(self):
        """V A V A interleaved -> True."""
        input_ids, is_multimodal = make_interleaved_seq([4, 4], [3, 3])
        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        assert check_interleaved_audio_video(is_video, is_audio, is_video.sum().item(), is_audio.sum().item())


# ---------------------------------------------------------------------------
# Tests for embed_input_ids via a minimal mock
# ---------------------------------------------------------------------------


def make_mock_model(mocker: MockerFixture, hidden: int = 8):
    """
    Return a minimal mock of Qwen2_5OmniThinkerForConditionalGeneration
    that has enough structure to run embed_input_ids.
    """
    model = mocker.Mock()

    # Config with token IDs
    cfg = mocker.Mock()
    cfg.video_token_index = VIDEO_TOKEN_ID
    cfg.audio_token_index = AUDIO_TOKEN_ID
    model.config = cfg
    model._has_oov_mm_tokens = False

    def fake_lm_embed(ids: torch.Tensor) -> torch.Tensor:
        # Use .clone() so the tensor is contiguous (expand() creates a strided
        # view with shared memory, which masked_scatter_ cannot handle).
        return ids.float().unsqueeze(-1).expand(-1, hidden).clone()

    lang_model = mocker.Mock()
    lang_model.embed_input_ids = fake_lm_embed
    model.get_language_model = mocker.Mock(return_value=lang_model)

    model.embed_input_ids = lambda *a, **kw: qwen25_omni_thinker_embed_input_ids(model, *a, **kw)

    return model, hidden


def build_mm_embeds(audio_n, image_n, video_n, hidden, audio_val=10.0, image_val=20.0, video_val=30.0):
    """
    Build multimodal_embeddings list in position order (audio, image, video).
    Each embedding is filled with a distinct constant so we can verify placement.
    """
    embs = []
    if audio_n:
        embs.append(torch.full((audio_n, hidden), audio_val))
    if image_n:
        embs.append(torch.full((image_n, hidden), image_val))
    if video_n:
        embs.append(torch.full((video_n, hidden), video_val))
    return embs


class TestEmbedInputIds:
    def _run(self, mocker: MockerFixture, audio_n, image_n, video_n, hidden=8):
        """
        Run embed_input_ids for a non-interleaved mixed-modality sequence.
        Returns (result_embeds, input_ids, is_multimodal).
        """
        input_ids, is_multimodal = make_token_seq(audio_n, image_n, video_n)
        mm_embeds = build_mm_embeds(audio_n, image_n, video_n, hidden)

        model, _ = make_mock_model(mocker, hidden)
        result = model.embed_input_ids(input_ids, mm_embeds, is_multimodal=is_multimodal)
        return result, input_ids, is_multimodal

    def test_audio_only(self, mocker: MockerFixture):
        """Audio-only: audio positions get audio embeddings."""
        audio_n, hidden = 5, 8
        audio_val = 10.0
        result, input_ids, is_multimodal = self._run(mocker, audio_n, 0, 0, hidden)

        audio_pos = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
        assert result[audio_pos].allclose(torch.full((audio_n, hidden), audio_val)), (
            "Audio positions should get audio embeddings"
        )

    def test_video_only(self, mocker: MockerFixture):
        """Video-only: video positions get video embeddings."""
        video_n, hidden = 6, 8
        video_val = 30.0
        result, input_ids, is_multimodal = self._run(mocker, 0, 0, video_n, hidden)

        video_pos = (input_ids == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
        assert result[video_pos].allclose(torch.full((video_n, hidden), video_val)), (
            "Video positions should get video embeddings"
        )

    def test_mixed_modalities_audio_goes_to_audio_pos(self, mocker: MockerFixture):
        """
        Regression test for GitHub issue #34506:
        With audio + image + video (non-interleaved), audio positions must
        receive audio embeddings (not image or video embeddings).
        """
        audio_n, image_n, video_n, hidden = 5, 4, 6, 8
        audio_val, image_val, video_val = 10.0, 20.0, 30.0

        result, input_ids, is_multimodal = self._run(mocker, audio_n, image_n, video_n, hidden)

        audio_pos = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
        image_pos = (input_ids == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        video_pos = (input_ids == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]

        mean_a = result[audio_pos].mean().item()
        assert result[audio_pos].allclose(torch.full((audio_n, hidden), audio_val)), (
            f"Audio emb wrong: expected {audio_val}, got mean={mean_a:.1f}"
        )

        mean_i = result[image_pos].mean().item()
        assert result[image_pos].allclose(torch.full((image_n, hidden), image_val)), (
            f"Image emb wrong: expected {image_val}, got mean={mean_i:.1f}"
        )

        mean_v = result[video_pos].mean().item()
        assert result[video_pos].allclose(torch.full((video_n, hidden), video_val)), (
            f"Video emb wrong: expected {video_val}, got mean={mean_v:.1f}"
        )

    def test_text_positions_unchanged(self, mocker: MockerFixture):
        """Text positions should keep their text embeddings."""
        audio_n, image_n, video_n, hidden = 3, 2, 4, 8
        result, input_ids, is_multimodal = self._run(mocker, audio_n, image_n, video_n, hidden)

        text_pos = (~is_multimodal).nonzero(as_tuple=True)[0]
        # Text tokens have value TEXT_TOKEN_ID=0, so embed -> 0.0
        assert result[text_pos].allclose(torch.zeros(len(text_pos), hidden)), (
            "Text positions should keep text embeddings"
        )

    def test_interleaved_use_audio_in_video(self, mocker: MockerFixture):
        """
        Interleaved (use_audio_in_video): video chunks interleaved with audio.
        Video embeddings must go to video positions, audio to audio positions.
        """
        hidden = 8
        audio_val, video_val = 10.0, 30.0
        video_chunks = [4, 4]
        audio_chunks = [3, 3]
        input_ids, is_multimodal = make_interleaved_seq(video_chunks, audio_chunks)

        video_n = sum(video_chunks)  # 8
        audio_n = sum(audio_chunks)  # 6

        mm_embeds = [
            torch.full((video_n, hidden), video_val),
            torch.full((audio_n, hidden), audio_val),
        ]

        model, _ = make_mock_model(mocker, hidden)
        result = model.embed_input_ids(input_ids, mm_embeds, is_multimodal=is_multimodal)

        video_pos = (input_ids == VIDEO_TOKEN_ID).nonzero(as_tuple=True)[0]
        audio_pos = (input_ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]

        assert result[video_pos].allclose(torch.full((video_n, hidden), video_val)), (
            "Interleaved: video positions should get video embeddings"
        )

        assert result[audio_pos].allclose(torch.full((audio_n, hidden), audio_val)), (
            "Interleaved: audio positions should get audio embeddings"
        )


# ---------------------------------------------------------------------------
# Tests for merge_interleaved_embeddings helper
# ---------------------------------------------------------------------------


class TestMergeInterleavedEmbeddings:
    def test_basic_interleaved(self):
        """Video chunks + audio chunks scattered to correct positions."""
        hidden = 4
        input_ids, is_multimodal = make_interleaved_seq([3, 3], [2, 2])

        is_video = is_multimodal & (input_ids == VIDEO_TOKEN_ID)
        is_audio = is_multimodal & (input_ids == AUDIO_TOKEN_ID)
        num_video = is_video.sum().item()  # 6
        num_audio = is_audio.sum().item()  # 4

        inputs_embeds = torch.zeros(len(input_ids), hidden)
        mm_embeds = [
            torch.full((num_video, hidden), 30.0),
            torch.full((num_audio, hidden), 10.0),
        ]

        result = merge_interleaved_embeddings(
            inputs_embeds,
            mm_embeds,
            is_video,
            is_audio,
            is_multimodal,
            num_video,
            num_audio,
        )

        video_pos = is_video.nonzero(as_tuple=True)[0]
        audio_pos = is_audio.nonzero(as_tuple=True)[0]
        assert result[video_pos].allclose(torch.full((num_video, hidden), 30.0))
        assert result[audio_pos].allclose(torch.full((num_audio, hidden), 10.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
