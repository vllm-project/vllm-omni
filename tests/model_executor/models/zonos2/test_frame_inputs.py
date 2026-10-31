# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for frozen frames and speaker slots at the talker boundary.

Use two-dimensional embeddings and a recording decoder instead of constructing
the checkpoint-sized backbone or initializing vLLM attention/KV caches.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_talker import (
    Zonos2MultiEmbedder,
    Zonos2TalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RecordingDecoder(nn.Module):
    def forward(self, hidden, positions, router_state, residual=None):
        self.hidden = hidden.detach().clone()
        self.positions = positions.detach().clone()
        assert router_state is None
        return hidden, residual, router_state


@pytest.fixture
def talker():
    """Allocate only the components used by the input boundary."""
    model = Zonos2TalkerForConditionalGeneration.__new__(Zonos2TalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = Zonos2Config(
        dim=2,
        n_layers=1,
        head_dim=2,
        codebook_size=2,
        eoa_id=2,
        audio_pad_id=3,
        text_vocab=3,
        speaker_embedding_dim=2048,
        speaker_lda_dim=2,
        norm_eps=0.5,
    )
    model.multi_embedder = Zonos2MultiEmbedder(model.config)
    model.layers = nn.ModuleList([_RecordingDecoder()])
    model.out_norm = nn.RMSNorm(2, eps=model.config.norm_eps)
    model.speaker_lda_projection = nn.Linear(2048, 2)
    model.speaker_projection = nn.Linear(2, 2)
    with torch.no_grad():
        for table in model.multi_embedder.embedders:
            table.weight.zero_()
        model.speaker_lda_projection.weight.zero_()
        model.speaker_lda_projection.weight[0, 0] = 1
        model.speaker_lda_projection.weight[1, 1] = 1
        model.speaker_lda_projection.bias.copy_(torch.tensor([1.0, -2.0]))
        model.speaker_projection.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, -3.0]]))
        model.speaker_projection.bias.copy_(torch.tensor([-1.0, 2.0]))
    return model


def _frames(rows=1, dtype=torch.int64):
    return torch.full((rows, 10), 3, dtype=dtype)


def test_columns_select_cb0_through_cb8_then_text_without_normalizing(talker):
    # Each column selects its own distinct ID and table. The final vector is a
    # concrete sum (1+...+10, 2+...+20), independent of the production loop.
    frames = _frames()
    with torch.no_grad():
        for column, table in enumerate(talker.multi_embedder.embedders):
            token = column % 3
            frames[0, column] = token
            table.weight[token] = torch.tensor([column + 1.0, 2.0 * (column + 1)])
    before = frames.clone()
    expected = torch.tensor([[55.0, 110.0]])
    torch.testing.assert_close(talker._embed_input_ids(frames), expected)
    torch.testing.assert_close(talker.embed_input_ids(frames), expected)
    assert torch.equal(frames, before)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_loaded_nonzero_padding_rows_are_not_masked(talker, dtype):
    with torch.no_grad():
        talker.multi_embedder.embedders[0].weight[3] = torch.tensor([1.0, 2.0])
        talker.multi_embedder.embedders[8].weight[3] = torch.tensor([4.0, 8.0])
        talker.multi_embedder.embedders[9].weight[3] = torch.tensor([16.0, 32.0])
    torch.testing.assert_close(talker.embed_input_ids(_frames(dtype=dtype)), torch.tensor([[21.0, 42.0]]))


def test_audio_and_text_vocabulary_limits_are_independent():
    config = Zonos2Config(dim=2, codebook_size=4, eoa_id=4, audio_pad_id=5, text_vocab=3)
    embedder = Zonos2MultiEmbedder(config)
    with torch.no_grad():
        for table in embedder.embedders:
            table.weight.zero_()
        embedder.embedders[8].weight[4] = torch.tensor([2.0, 0.0])
        embedder.embedders[9].weight[3] = torch.tensor([0.0, 3.0])
    frames = torch.full((1, 10), 5, dtype=torch.long)
    frames[0, 8] = 4  # Audio EOA is valid, but outside the text table.
    frames[0, 9] = 3  # Text padding is independent of audio padding.
    torch.testing.assert_close(embedder(frames), torch.tensor([[2.0, 3.0]]))
    for column, upper in ((0, 6), (8, 6), (9, 4)):
        for invalid in (-1, upper):
            bad = frames.clone()
            bad[0, column] = invalid
            with pytest.raises(IndexError):
                embedder(bad)


@pytest.mark.parametrize("shape", [(), (2, 9), (2, 11), (1, 2, 10)])
def test_rejects_wrong_frame_shapes(talker, shape):
    with pytest.raises(ValueError, match="shape|input ids"):
        talker.embed_input_ids(torch.zeros(shape, dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int16, torch.bool])
def test_rejects_non_integer_frame_dtypes(talker, dtype):
    with pytest.raises(TypeError, match="int32 or torch.int64"):
        talker.embed_input_ids(torch.zeros((2, 10), dtype=dtype))


def test_legacy_text_ids_keep_padding_and_clamping_behavior(talker):
    with torch.no_grad():
        talker.multi_embedder.embedders[0].weight[3] = torch.tensor([1.0, 2.0])
        talker.multi_embedder.embedders[9].weight.copy_(torch.tensor([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))
    expected = torch.tensor([[4.0, 6.0], [6.0, 8.0], [10.0, 12.0]])
    torch.testing.assert_close(talker.embed_input_ids(torch.tensor([-1, 1, 99])), expected)


def test_prefill_and_single_frame_decode_chunks_embed_identically(talker):
    frames = torch.tensor(
        [
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 1],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 3, 3, 3, 3, 3, 3, 3, 3],
            [2, 1, 0, 3, 3, 3, 3, 3, 3, 3],
        ]
    )
    with torch.no_grad():
        talker.multi_embedder.embedders[0].weight[:3] = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        talker.multi_embedder.embedders[1].weight[:3] = torch.tensor([[0.0, 4.0], [0.0, 5.0], [0.0, 6.0]])
        talker.multi_embedder.embedders[9].weight[1] = torch.tensor([7.0, 8.0])
    expected = torch.tensor([[7.0, 8.0], [1.0, 0.0], [2.0, 6.0], [3.0, 5.0]])
    whole = talker.embed_input_ids(frames)
    chunks = torch.cat(
        [talker.embed_input_ids(frames[:2]), *(talker.embed_input_ids(row) for row in frames[2:].split(1))]
    )
    torch.testing.assert_close(whole, expected)
    torch.testing.assert_close(chunks, expected)


def test_ids_and_raw_inputs_embeds_share_one_emb_norm_boundary(talker):
    frames = _frames()
    frames[0, 9] = 1
    with torch.no_grad():
        talker.multi_embedder.embedders[9].weight[1] = torch.tensor([3.0, 4.0])
    raw = talker.embed_input_ids(frames)
    expected = torch.tensor([[3.0 / math.sqrt(13.0), 4.0 / math.sqrt(13.0)]])
    original = torch.nn.functional.rms_norm
    for embeds in (None, raw):
        with patch("torch.nn.functional.rms_norm", wraps=original) as norm:
            talker(frames, torch.tensor([42]), inputs_embeds=embeds)
        # One embedding norm before the decoder, plus the distinct output norm.
        assert norm.call_count == 2
        assert norm.call_args_list[0].kwargs.get("weight") is None
        assert norm.call_args_list[1].args[2] is talker.out_norm.weight
        torch.testing.assert_close(talker.layers[0].hidden, expected)
    torch.testing.assert_close(raw, torch.tensor([[3.0, 4.0]]))


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_speaker_replaces_entire_local_row_before_emb_norm(talker, position_dtype):
    raw = torch.tensor([[3.0, 4.0], [100.0, 200.0], [-3.0, 4.0]])
    before = raw.clone()
    speaker = torch.zeros((1, 2048), dtype=torch.float64)
    speaker[0, :2] = torch.tensor([3.0, 4.0])
    positions = torch.tensor([101, 102, 103])
    talker(
        _frames(3),
        positions,
        inputs_embeds=raw,
        speaker_embeddings=speaker,
        speaker_positions=torch.tensor([1], dtype=position_dtype),
    )
    # The two affine projections produce [7,-4]; the original [100,200] is
    # replaced completely, and row 1 means the local input row (not position 1).
    expected = torch.tensor(
        [
            [3.0 / math.sqrt(13), 4.0 / math.sqrt(13)],
            [7.0 / math.sqrt(33), -4.0 / math.sqrt(33)],
            [-3.0 / math.sqrt(13), 4.0 / math.sqrt(13)],
        ]
    )
    torch.testing.assert_close(talker.layers[0].hidden, expected)
    assert torch.equal(talker.layers[0].positions, positions)
    assert torch.equal(raw, before)


@pytest.mark.parametrize("missing", ["speaker_embeddings", "speaker_positions"])
def test_speaker_arguments_must_be_paired(talker, missing):
    kwargs = {"speaker_embeddings": torch.zeros((1, 2048)), "speaker_positions": torch.tensor([0])}
    kwargs[missing] = None
    with pytest.raises(ValueError, match="supplied together"):
        talker(_frames(), torch.tensor([0]), **kwargs)


@pytest.mark.parametrize("shape", [(2048,), (1, 2047), (1, 1, 2048)])
def test_rejects_invalid_speaker_shapes(talker, shape):
    with pytest.raises(ValueError, match="speaker_embeddings must have shape"):
        talker(_frames(), torch.tensor([0]), speaker_embeddings=torch.zeros(shape), speaker_positions=torch.tensor([0]))


@pytest.mark.parametrize("speaker_positions", [torch.tensor(0), torch.tensor([[0]]), torch.tensor([0, 1])])
def test_rejects_invalid_speaker_position_shapes(talker, speaker_positions):
    with pytest.raises(ValueError, match="one input row index"):
        talker(
            _frames(), torch.tensor([0]), speaker_embeddings=torch.zeros((1, 2048)), speaker_positions=speaker_positions
        )


def test_rejects_float_speaker_positions(talker):
    with pytest.raises(TypeError, match="int32 or torch.int64"):
        talker(
            _frames(),
            torch.tensor([0]),
            speaker_embeddings=torch.zeros((1, 2048)),
            speaker_positions=torch.tensor([0.0]),
        )


@pytest.mark.parametrize("invalid_position", [-1, 1])
def test_speaker_positions_cannot_address_outside_local_rows(talker, invalid_position):
    with pytest.raises((IndexError, RuntimeError)):
        talker(
            _frames(),
            torch.tensor([99]),
            speaker_embeddings=torch.zeros((1, 2048)),
            speaker_positions=torch.tensor([invalid_position]),
        )


@pytest.mark.parametrize("shape", [(2,), (1, 3), (1, 1, 2)])
def test_rejects_invalid_raw_embedding_shapes(talker, shape):
    with pytest.raises(ValueError, match="raw embeddings must have shape"):
        talker(_frames(), torch.tensor([0]), inputs_embeds=torch.zeros(shape))


@pytest.mark.parametrize("positions", [torch.tensor(0), torch.tensor([[0]]), torch.tensor([0, 1])])
def test_forward_positions_must_match_input_rows(talker, positions):
    with pytest.raises(ValueError, match="one position per input frame"):
        talker(_frames(), positions)


def test_bf16_text_convenience_uses_canonical_audio_then_text_order(talker):
    talker.to(dtype=torch.bfloat16)
    with torch.no_grad():
        for table in talker.multi_embedder.embedders[:9]:
            table.weight[3].fill_(1)
        talker.multi_embedder.embedders[9].weight[1].fill_(256)
    frames = _frames()
    frames[0, 9] = 1
    expected = torch.tensor([[264.0, 264.0]], dtype=torch.bfloat16)
    # Starting from text=256 loses each individual +1 in BF16. The official
    # order sums the nine audio columns first, then adds the text column.
    torch.testing.assert_close(talker.embed_input_ids(frames), expected, rtol=0, atol=0)
    torch.testing.assert_close(talker.embed_input_ids(torch.tensor([1])), expected, rtol=0, atol=0)


def test_runner_prefill_chunks_use_absolute_offsets_and_inject_speaker_once(talker):
    frames = _frames(3)
    frames[:, 9] = torch.tensor([1, 2, 1])
    with torch.no_grad():
        talker.multi_embedder.embedders[9].weight.copy_(torch.tensor([[0.0, 0.0], [3.0, 4.0], [5.0, 6.0], [0.0, 0.0]]))
    speaker = torch.zeros(2048)
    speaker[:2] = torch.tensor([3.0, 4.0])
    info = {
        "zonos2_frames": frames,
        "zonos2_speaker_embedding": speaker,
        "zonos2_speaker_position": 0,
        "_omni_is_prefill": True,
    }
    _, first, update = talker.preprocess(torch.tensor([1]), None, _omni_num_computed_tokens=0, **info)
    _, rest, _ = talker.preprocess(torch.tensor([2, 1]), None, _omni_num_computed_tokens=1, **info)
    torch.testing.assert_close(first, torch.tensor([[7.0, -4.0]]))
    torch.testing.assert_close(rest, torch.tensor([[5.0, 6.0], [3.0, 4.0]]))
    assert update == {}
    # The preprocessor returns raw embeddings. Only forward normalizes them.
    talker(frames, torch.arange(3), inputs_embeds=torch.cat((first, rest)))
    expected = torch.tensor([[7.0, -4.0]]) / math.sqrt(33.0)
    torch.testing.assert_close(talker.layers[0].hidden[:1], expected)
    assert torch.equal(frames[:, 9], torch.tensor([1, 2, 1]))


def test_runner_decode_uses_all_nine_previous_codes_and_text_padding(talker):
    frames = _frames()
    with torch.no_grad():
        for col, table in enumerate(talker.multi_embedder.embedders):
            table.weight[1] = torch.tensor([col + 1.0, 0.0])
            table.weight[3] = torch.tensor([0.0, col + 1.0])
    info = {
        "zonos2_frames": frames,
        "_omni_num_computed_tokens": 1,
        "_omni_is_prefill": False,
        "codes": {"audio": torch.ones((1, 9), dtype=torch.int32)},
    }
    lifecycle = torch.tensor([2])
    ids, embeds, updates = talker.preprocess(lifecycle, None, **info)
    torch.testing.assert_close(embeds, torch.tensor([[45.0, 10.0]]))
    assert torch.equal(ids, lifecycle) and updates == {}


def test_runner_decode_requires_complete_codes_instead_of_using_lifecycle_token(talker):
    for codes in (None, torch.ones(8, dtype=torch.int32)):
        with pytest.raises(ValueError, match="nine"):
            talker.preprocess(
                torch.tensor([1]),
                None,
                zonos2_frames=_frames(),
                _omni_num_computed_tokens=1,
                _omni_is_prefill=False,
                codes={"audio": codes},
            )
