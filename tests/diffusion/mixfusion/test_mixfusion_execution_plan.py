# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework-level tests for the MixFusion layout primitives (CPU).

``execution_plan`` is model-agnostic: it turns sequences plus a model-declared
``AttentionLayout`` into the packed varlen layout (cu_seqlens / max_seq_len,
gather rows out of the packed output, scatter rows back into the padded
per-request space). These tests pin it with hand-computed layouts, so the
primitives can be relied on before a model adapter is involved.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout, StreamSpec
from vllm_omni.diffusion.mixfusion.execution_plan import (
    build_execution_plan,
    expand_per_chunk,
    reassemble_chunk_rows,
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


# ── build_execution_plan: happy path ──


def test_build_execution_plan_lays_streams_out_in_declared_order():
    plan = build_execution_plan(**_plan_kwargs())

    # text(3) image(8) | text(5) image(12) -> cu_seqlens [0, 11, 28].
    assert plan.layout is LAYOUT
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
    assert plan.cu_seqlens_tensor.tolist() == [0, 14, 32]
    assert plan.max_seq_len == 18


# ── chunk splitting / reassembly ──


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
