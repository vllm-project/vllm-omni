# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniMax Music 3 batched payload partitioning.

The stage-0 payload is split per request by indexing list values: a list is
indexed by batch row, anything else is handed to every request unchanged. So
per-request metadata has to be carried as lists. Merging it into one dict made
every request in a batch receive the last request's seed, window offset and
terminal flag, which crosses seeds between songs, misplaces acoustic windows,
and lets one request's terminal flag end another request's stream early.

These drive ``make_omni_output`` directly with a hand-built batch, so no GPU
and no engine are needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.minimax_music3.frame_state import RequestARState
from vllm_omni.model_executor.models.minimax_music3.prompt import SPECIAL_TOKEN_IDS
from vllm_omni.model_executor.models.minimax_music3.talker import (
    MiniMaxMusic3TalkerForConditionalGeneration as Talker,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_WIDTH = 4


def _info(*, seed: int, start: int, frames: int, last: bool) -> dict:
    """One batch row's intermediate buffer, as ``postprocess`` leaves it."""
    return {
        "latent": torch.full((frames, _WIDTH), float(seed)),
        "codes": {"audio": torch.full((frames, 8), seed)},
        "meta": {
            "audio_seed": seed,
            "num_processed_tokens": start,
            "codec_chunk_frames": frames,
            "last_chunk": last,
        },
    }


def _payload_for(output, row: int) -> dict:
    """Partition the batch payload down to one request, as the runner does."""
    flat = {}
    for key, value in output.multimodal_outputs.items():
        if isinstance(value, dict):
            for qual, val in value.items():
                flat[f"{key}.{qual}"] = val
        else:
            flat[key] = value
    return {key: to_payload_element(value, row, 1, 1) for key, value in flat.items()}


class TestBatchedMetadataPartitioning:
    """Two requests in one batch must keep their own metadata."""

    @staticmethod
    def _two_request_batch():
        infos = [
            _info(seed=111, start=100, frames=3, last=False),
            _info(seed=222, start=200, frames=5, last=True),
        ]
        talker = Talker.__new__(Talker)
        return talker.make_omni_output(torch.zeros(2, _WIDTH), model_intermediate_buffer=infos)

    def test_seeds_are_not_crossed(self) -> None:
        """A song decoded with another request's seed is a different song."""
        output = self._two_request_batch()
        assert _payload_for(output, 0)["meta.audio_seed"] == 111
        assert _payload_for(output, 1)["meta.audio_seed"] == 222

    def test_window_offsets_stay_with_their_request(self) -> None:
        """The start frame decides where a window lands in the song."""
        output = self._two_request_batch()
        assert _payload_for(output, 0)["meta.num_processed_tokens"] == 100
        assert _payload_for(output, 1)["meta.num_processed_tokens"] == 200

    def test_terminal_flag_does_not_leak_across_requests(self) -> None:
        """One request finishing must not end another request's stream."""
        output = self._two_request_batch()
        assert _payload_for(output, 0)["meta.last_chunk"] is False
        assert _payload_for(output, 1)["meta.last_chunk"] is True

    def test_span_length_matches_its_own_latent(self) -> None:
        output = self._two_request_batch()
        for row, frames in ((0, 3), (1, 5)):
            entry = _payload_for(output, row)
            assert entry["meta.codec_chunk_frames"] == frames
            assert entry["latent"].shape[0] == frames

    def test_latent_and_codes_follow_the_same_row(self) -> None:
        """The tensors were already per-request; pin that they stay aligned."""
        output = self._two_request_batch()
        for row, seed in ((0, 111), (1, 222)):
            entry = _payload_for(output, row)
            assert entry["latent"][0, 0].item() == pytest.approx(float(seed))
            assert int(entry["codes.audio"][0, 0]) == seed


class TestPartialBatches:
    """Only some rows carry a span on any given step."""

    def test_row_without_a_span_does_not_take_another_row_metadata(self) -> None:
        infos = [{}, _info(seed=777, start=300, frames=2, last=True)]
        talker = Talker.__new__(Talker)
        output = talker.make_omni_output(torch.zeros(2, _WIDTH), model_intermediate_buffer=infos)

        idle = _payload_for(output, 0)
        assert idle["latent"].numel() == 0
        assert idle["meta.audio_seed"] is None
        assert idle["meta.last_chunk"] is None

        active = _payload_for(output, 1)
        assert active["meta.audio_seed"] == 777
        assert active["meta.last_chunk"] is True

    def test_batch_with_no_spans_emits_no_payload(self) -> None:
        talker = Talker.__new__(Talker)
        output = talker.make_omni_output(torch.zeros(2, _WIDTH), model_intermediate_buffer=[{}, {}])
        assert output.multimodal_outputs is None

    def test_consumed_entries_are_cleared_from_the_buffer(self) -> None:
        """The runner never clears this buffer, so a span left behind would be
        re-emitted on every later step."""
        infos = [_info(seed=1, start=0, frames=2, last=False)]
        talker = Talker.__new__(Talker)
        talker.make_omni_output(torch.zeros(1, _WIDTH), model_intermediate_buffer=infos)
        assert "latent" not in infos[0]
        assert "meta" not in infos[0]


class TestWholeSongWithoutAsyncChunk:
    """With ``async_chunk: false`` the engine collects stage 0's outputs and
    ``ar2acoustic`` hands stage 1 the whole song, so the talker must publish it
    once, from frame 0."""

    @staticmethod
    def _talker(total_frames: int) -> tuple[Talker, RequestARState]:
        talker = Talker.__new__(Talker)
        talker.publishes_windows = False
        state = RequestARState(seed=5, max_frames=total_frames)
        talker._states = {"song": state}
        return talker, state

    def test_the_song_is_published_once_when_it_ends(self) -> None:
        talker, state = self._talker(450)
        for index in range(450):
            state.frames.append(torch.full((1, _WIDTH), float(index)))
            state.codes.append(torch.full((1, 8), index))
            state.generated_frames += 1
            # Past several window boundaries, and nothing is published.
            assert talker.postprocess(None, None, global_request_id="song") == {}

        state.finished = True
        span = talker.postprocess(None, None, global_request_id="song")
        assert span["latent"][:, 0].tolist() == [float(index) for index in range(450)]
        assert span["codes"]["audio"].shape == (450, 8)
        assert span["meta"]["num_processed_tokens"] == 0
        assert span["meta"]["last_chunk"] is True
        assert span["meta"]["audio_seed"] == 5
        assert talker.postprocess(None, None, global_request_id="song") == {}

    def test_idle_rows_get_placeholders_of_the_published_dtypes(self) -> None:
        """The engine joins each request's outputs, so a float32 placeholder
        would turn the bf16 song it is later joined with into float32."""
        talker = Talker.__new__(Talker)
        infos = [{}, _info(seed=777, start=0, frames=2, last=True)]
        output = talker.make_omni_output(torch.zeros(2, _WIDTH, dtype=torch.bfloat16), model_intermediate_buffer=infos)

        idle = _payload_for(output, 0)
        assert idle["latent"].numel() == 0
        assert idle["latent"].dtype == torch.bfloat16
        assert idle["codes.audio"].dtype == torch.long


@pytest.mark.parametrize(("prompt_tokens", "frame_limit", "expected_frames"), [(7, 9, 2), (5, 2, 2), (8, 9, 1)])
@pytest.mark.parametrize("async_chunk", [False, True])
def test_final_audio_fits_in_context(prompt_tokens, frame_limit, expected_frames, async_chunk, monkeypatch):
    # Force a non-EOS draw on every step so only the length limits can end the song.
    max_model_len = 10
    talker = Talker.__new__(Talker)
    talker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len))
    talker.publishes_windows = async_chunk
    talker._states = {}
    talker._row_seeds = [5, 5]
    talker._row_pair_ids = ["song", "song"]
    talker._row_roles = ["cond", "uncond"]
    talker._row_sample_eligible = [True, True]
    talker._row_prompt_tokens = [prompt_tokens, prompt_tokens]
    talker._row_max_frames = [frame_limit, frame_limit]
    talker.c0_logit_ids = torch.tensor([0, 1])
    logits = torch.tensor([[0.0, 1000.0], [0.0, 1000.0]])
    hidden = torch.ones(2, _WIDTH)

    # Use CPU indices and fixed depth outputs in place of GPU staging and model weights.
    def cpu_vectors(cond, uncond, seeds, positions, device):
        return torch.tensor([cond, uncond, seeds, positions], device=device)

    monkeypatch.setattr(talker, "_stage_step_vectors", cpu_vectors)
    monkeypatch.setattr(talker, "_depth_decode", Mock(return_value=(torch.zeros(1, 8, dtype=torch.long), hidden[:1])))

    for step in range(1, max_model_len - prompt_tokens + 1):
        info = talker.postprocess(None, global_request_id="song")
        output = talker.make_omni_output(hidden, model_intermediate_buffer=[info, {}])
        talker._last_logits_hidden = hidden
        tokens = talker.sample(logits, None).sampled_token_ids[:, 0].tolist()
        if tokens == [SPECIAL_TOKEN_IDS["<|audio_end|>"]] * 2:
            assert step == expected_frames + 1
            payload = _payload_for(output, 0)
            assert payload["meta.last_chunk"] is True
            assert payload["meta.num_processed_tokens"] == 0
            assert payload["latent"].shape[0] == payload["meta.codec_chunk_frames"] == expected_frames
            assert payload["codes.audio"].shape == (expected_frames, 8)
            return
        assert output.multimodal_outputs is None
    pytest.fail("Engine reached max_model_len before the talker delivered the final audio")
