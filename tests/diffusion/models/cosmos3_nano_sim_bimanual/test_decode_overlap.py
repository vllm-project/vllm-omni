# SPDX-License-Identifier: Apache-2.0
"""CUDA ordering, storage lifetime and failure cleanup for overlapped decode."""

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.decode_overlap import CausalDecodeQueue

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_causal_decode_preserves_producer_order_and_pinned_outputs():
    state = None
    seen_streams = []

    def decode(x):
        nonlocal state
        seen_streams.append(torch.cuda.current_stream().cuda_stream)
        state = x.clone() if state is None else state + x
        return state

    queue = CausalDecodeQueue(decode, torch.device("cuda"))
    for i in range(1, 5):
        # Allocate/write on the producer, release immediately after submission,
        # then exercise the allocator while decode may still be consuming it.
        x = torch.full((256, 256), i / 32, device="cuda")
        queue.submit(x)
        del x
        torch.empty((256, 256), device="cuda").fill_(-1)
    outputs = queue.finish()
    assert seen_streams == [queue.stream.cuda_stream] * 4
    assert queue.stream.cuda_stream != torch.cuda.current_stream().cuda_stream
    for i, output in enumerate(outputs, 1):
        assert output.is_pinned()
        assert torch.equal(output, torch.full_like(output, i * (i + 1) / 64))
    queue.close()  # idempotent cleanup
    with pytest.raises(RuntimeError, match="closed"):
        queue.submit(torch.zeros(1, device="cuda"))


def test_error_cleanup_drains_decoder_work():
    complete = torch.cuda.Event()

    def fail(x):
        _ = x @ x
        complete.record()
        raise ValueError("decode failure")

    queue = CausalDecodeQueue(fail, torch.device("cuda"))
    with pytest.raises(ValueError, match="decode failure"):
        try:
            queue.submit(torch.ones((256, 256), device="cuda"))
        finally:
            queue.close()
    assert queue.closed and complete.query()


def test_drained_stream_can_be_reused_across_requests():
    stream = torch.cuda.Stream()
    for value in (0.25, 0.75):
        queue = CausalDecodeQueue(lambda x: x * 2, torch.device("cuda"), stream=stream)
        queue.submit(torch.full((8, 8), value, device="cuda"))
        output = queue.finish()[0]
        assert queue.stream is stream
        assert torch.equal(output, torch.full_like(output, min(1.0, value * 2)))
