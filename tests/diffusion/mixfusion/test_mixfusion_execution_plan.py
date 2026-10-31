# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework-level tests for the MixFusion layout primitives (CPU).

``execution_plan`` is model-agnostic: it turns sequences plus a model-declared
``AttentionLayout`` into the packed varlen layout (cu_seqlens / max_seq_len,
gather rows out of the packed output, scatter rows back into the padded
per-request space). These tests pin the happy path and every validation
error with hand-computed layouts, so the primitives can be relied on before a
model adapter is involved.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout, StreamSpec
from vllm_omni.diffusion.mixfusion.execution_plan import (
    build_execution_plan,
    expand_per_chunk,
    gcd_chunk_size,
    plan_chunk_size,
    reassemble_chunk_rows,
    split_sequence_chunks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

DEVICE = torch.device("cpu")

LAYOUT = AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))


def _plan_kwargs(**overrides) -> dict:
    """Two requests, chunk size 4: 2 chunks then 3 chunks (5 total).

    Request 0 is text=3 + image=8 tokens, request 1 is text=5 + image=12;
    the text stream is padded to 6 columns, the image stream needs none.
    """
    kwargs = dict(
        layout=LAYOUT,
        chunk_size=4,
        request_chunk_ranges=[(0, 2), (2, 5)],
        image_row_counts=(1, 1),
        request_stream_lens={"text": (3, 5), "image": (8, 12)},
        stream_padded_lens={"text": 6},
        chunk_to_request=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        device=DEVICE,
    )
    kwargs.update(overrides)
    return kwargs


# ── gcd_chunk_size / plan_chunk_size ──


def test_gcd_chunk_size_is_the_largest_common_divisor():
    assert gcd_chunk_size([torch.zeros(1, 8, 4), torch.zeros(2, 12, 4)]) == 4
    assert gcd_chunk_size([torch.zeros(1, 8, 4)]) == 8


def test_plan_chunk_size_accepts_an_admissible_batch():
    accepted, reason, chunk_size, chunk_count = plan_chunk_size(
        [torch.zeros(1, 8, 4), torch.zeros(1, 12, 4)],
        min_chunk_tokens=1,
        max_chunks=8,
    )

    assert (accepted, reason) == (True, "ok")
    assert (chunk_size, chunk_count) == (4, 5)


def test_plan_chunk_size_counts_chunks_per_image():
    # Two images per request multiply the chunk count in the stacked stream.
    latents = [torch.zeros(2, 8, 4), torch.zeros(1, 12, 4)]

    accepted, reason, chunk_size, chunk_count = plan_chunk_size(
        latents,
        min_chunk_tokens=1,
        max_chunks=8,
    )

    assert accepted
    assert (chunk_size, chunk_count) == (4, 7)


def test_plan_chunk_size_rejects_chunks_below_the_minimum():
    accepted, reason, chunk_size, chunk_count = plan_chunk_size(
        [torch.zeros(1, 4, 4)],
        min_chunk_tokens=8,
        max_chunks=8,
    )

    assert not accepted
    assert reason == "chunk_size=4 < min_chunk_tokens=8"
    assert (chunk_size, chunk_count) == (4, 1)


def test_plan_chunk_size_rejects_too_many_chunks():
    accepted, reason, chunk_size, chunk_count = plan_chunk_size(
        [torch.zeros(1, 8, 4), torch.zeros(1, 12, 4)],
        min_chunk_tokens=1,
        max_chunks=4,
    )

    assert not accepted
    assert reason == "chunk_count=5 > max_chunks=4"
    assert (chunk_size, chunk_count) == (4, 5)


# ── build_execution_plan: happy path ──


def test_build_execution_plan_lays_streams_out_in_declared_order():
    plan = build_execution_plan(**_plan_kwargs())

    # text(3) image(8) | text(5) image(12) -> cu_seqlens [0, 11, 28].
    assert plan.cu_seqlens == [0, 11, 28]
    assert plan.cu_seqlens_tensor.tolist() == [0, 11, 28]
    assert plan.cu_seqlens_tensor.dtype == torch.int32
    assert plan.max_seq_len == 17
    assert plan.num_chunks == 5
    assert plan.request_chunk_ranges == [(0, 2), (2, 5)]
    assert plan.image_row_counts == (1, 1)
    assert plan.request_stream_lens == {"text": (3, 5), "image": (8, 12)}
    assert plan.stream_padded_lens == {"text": 6}


def test_build_execution_plan_gathers_each_streams_rows():
    plan = build_execution_plan(**_plan_kwargs())

    text_gather, text_scatter = plan.output_indices["text"]
    image_gather, image_scatter = plan.output_indices["image"]

    # Stream order inside request 1 starts after request 0's 11 tokens.
    assert text_gather.tolist() == [0, 1, 2, 11, 12, 13, 14, 15]
    assert image_gather.tolist() == list(range(3, 11)) + list(range(16, 28))
    # text is padded from 3/5 to 6 columns per request -> scatter into it.
    assert text_scatter.tolist() == [0, 1, 2, 6, 7, 8, 9, 10]
    # image is chunked: rows are already contiguous, no scatter needed.
    assert image_scatter is None


def test_build_execution_plan_skips_scatter_when_no_padding_is_needed():
    plan = build_execution_plan(
        **_plan_kwargs(
            request_stream_lens={"text": (6, 6), "image": (8, 12)},
            stream_padded_lens={"text": 6},
        )
    )

    text_gather, text_scatter = plan.output_indices["text"]

    assert text_gather.tolist() == [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19]
    assert text_scatter is None
    assert plan.cu_seqlens == [0, 14, 32]
    assert plan.max_seq_len == 18


# ── build_execution_plan: validation ──


def test_build_execution_plan_rejects_mismatched_stream_keys():
    with pytest.raises(ValueError, match=r"request_stream_lens keys \['text'\] do not match"):
        build_execution_plan(**_plan_kwargs(request_stream_lens={"text": (3, 5)}))


def test_build_execution_plan_rejects_short_image_row_counts():
    with pytest.raises(ValueError, match=r"image_row_counts \(1,\) span 2 chunked rows"):
        build_execution_plan(**_plan_kwargs(image_row_counts=(1,)))


def test_build_execution_plan_rejects_wrong_stream_length_count():
    with pytest.raises(ValueError, match=r"request_stream_lens\['text'\] has 1 entries, expected 2"):
        build_execution_plan(**_plan_kwargs(request_stream_lens={"text": (3,), "image": (8, 12)}))


def test_build_execution_plan_rejects_an_inverted_chunk_range():
    with pytest.raises(ValueError, match=r"request 0 has an inverted chunk range \(2, 0\)"):
        build_execution_plan(
            **_plan_kwargs(
                request_chunk_ranges=[(2, 0)],
                image_row_counts=(1,),
                request_stream_lens={"text": (3,), "image": (0,)},
            )
        )


def test_build_execution_plan_rejects_a_chunked_stream_of_the_wrong_length():
    with pytest.raises(ValueError, match=r"Stream 'image' of request 1 has 8 tokens, expected 12 from 3 chunk\(s\)"):
        build_execution_plan(**_plan_kwargs(request_stream_lens={"text": (3, 5), "image": (8, 8)}))


def test_build_execution_plan_requires_a_padded_width_for_request_streams():
    with pytest.raises(ValueError, match=r"Stream 'text' needs an entry in stream_padded_lens"):
        build_execution_plan(**_plan_kwargs(stream_padded_lens={}))


def test_build_execution_plan_rejects_a_stream_wider_than_its_padded_width():
    with pytest.raises(ValueError, match=r"Stream 'text' of request 1 has 5 tokens, padded to 4"):
        build_execution_plan(**_plan_kwargs(stream_padded_lens={"text": 4}))


def test_build_execution_plan_copies_the_padded_lens_it_was_given():
    plan = build_execution_plan(**_plan_kwargs(stream_padded_lens={"text": 6, "image": 12}))

    assert plan.stream_padded_lens == {"text": 6, "image": 12}


# ── chunk splitting / reassembly ──


def test_split_sequence_chunks_keeps_states_and_freqs_aligned():
    rows = [torch.arange(8.0).reshape(1, 8, 1), torch.arange(12.0).reshape(1, 12, 1)]
    freqs = [torch.arange(8.0).reshape(8, 1), torch.arange(12.0).reshape(12, 1)]

    chunks, freq_chunks, ranges, chunk_to_request = split_sequence_chunks(rows, freqs, chunk_size=4)

    assert len(chunks) == len(freq_chunks) == 5
    assert ranges == [(0, 2), (2, 5)]
    assert chunk_to_request == [0, 0, 1, 1, 1]
    assert [chunk.shape for chunk in chunks] == [(1, 4, 1)] * 5
    assert torch.equal(chunks[2][0, :, 0], rows[1][0, :4, 0])
    assert torch.equal(freq_chunks[2][:, 0], freqs[1][:4, 0])


def test_split_sequence_chunks_rejects_a_non_divisible_sequence():
    with pytest.raises(ValueError, match=r"seq_len=10 is not divisible by chunk_size=4"):
        split_sequence_chunks(
            [torch.zeros(1, 10, 1)],
            [torch.zeros(10, 1)],
            chunk_size=4,
        )


def test_expand_per_chunk_duplicates_request_conditioning():
    temb = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    expanded = expand_per_chunk(temb, torch.tensor([0, 0, 1, 1, 1]))

    assert expanded.tolist() == [[0.0, 1.0], [0.0, 1.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]


def test_reassemble_chunk_rows_rebuilds_one_tensor_per_request():
    chunk_rows = torch.arange(5 * 4 * 2, dtype=torch.float32).reshape(5, 4, 2)

    outputs = reassemble_chunk_rows(chunk_rows, [(0, 2), (2, 5)], row_counts=(1, 1))

    assert [output.shape for output in outputs] == [(1, 8, 2), (1, 12, 2)]
    assert torch.equal(outputs[0], chunk_rows[0:2].reshape(1, 8, 2))
    assert torch.equal(outputs[1], chunk_rows[2:5].reshape(1, 12, 2))


def test_reassemble_chunk_rows_concatenates_images_of_one_request():
    chunk_rows = torch.arange(4 * 4 * 2, dtype=torch.float32).reshape(4, 4, 2)

    outputs = reassemble_chunk_rows(chunk_rows, [(0, 2), (2, 4)], row_counts=(2,))

    assert [output.shape for output in outputs] == [(2, 8, 2)]
    assert torch.equal(outputs[0][0], chunk_rows[0:2].reshape(1, 8, 2).squeeze(0))
    assert torch.equal(outputs[0][1], chunk_rows[2:4].reshape(1, 8, 2).squeeze(0))
