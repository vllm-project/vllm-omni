# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Captured Qwen2.5-Omni image encoder graphs replay the eager tower under TP=2."""

import socket

import pytest
import torch
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.helpers.distributed import start_collective_workers
from tests.helpers.mark import hardware_marks
from tests.model_executor.models.qwen2_5_omni.test_qwen2_5_omni_vision_encoder_cudagraph_cuda import (
    BUDGETS,
    _build_encoder,
    _eager,
    _images,
    _tol,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.accelerator.device_count() < 2, reason="Two CUDA devices required"
    ),
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=2),
]


def _assert_matches_eager_on_both_ranks(encoder, manager, grids, seed):
    from vllm.distributed import tensor_model_parallel_all_gather

    inputs = _images(grids, seed)
    expected = _eager(encoder, inputs)
    with torch.inference_mode():
        actual = manager.execute(inputs)
    assert actual is not None, grids
    assert [tuple(x.shape) for x in actual] == [tuple(x.shape) for x in expected]
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, **_tol(reference), msg=lambda detail: f"{grids}: {detail}")
        # The row-parallel merger all-reduces, so both ranks hold the same rows.
        both = tensor_model_parallel_all_gather(output.unsqueeze(0).contiguous(), dim=0)
        torch.testing.assert_close(both[0], both[1], rtol=0, atol=0)


def _worker(rank, port, backend):
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.worker.encoder_cudagraph import SingleReplayEncoderCudaGraphManager

    torch.accelerator.set_device_index(rank)
    # The captured graphs hold NCCL all-reduces; custom all-reduce needs
    # peer-to-peer links that two-card test hosts may not have.
    parallel_state.set_custom_all_reduce(False)
    parallel_state.init_distributed_environment(
        world_size=2, rank=rank, local_rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl"
    )
    try:
        with set_current_vllm_config(VllmConfig()):
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
        encoder = _build_encoder(backend, tp_size=2)
        assert supports_encoder_cudagraph(encoder)
        manager = SingleReplayEncoderCudaGraphManager(
            encoder.vllm_config, torch.device("cuda"), encoder.visual.dtype, encoder
        )
        try:
            with torch.inference_mode(), graph_capture(device=torch.device("cuda")):
                set_cudagraph_capturing_enabled(True)
                try:
                    manager.capture(graph_pool=torch.cuda.graph_pool_handle())
                finally:
                    set_cudagraph_capturing_enabled(False)
            assert manager.get_cumulative_stats()["num_budgets"] == len(BUDGETS)
            # Consecutive groups alternate between the 8- and 64-token graphs:
            # one image, two images, two images filling budget 8, one image
            # with partial windows, one long image.
            cases = [[[1, 4, 8]], [[1, 8, 8], [1, 4, 8]], [[1, 4, 4], [1, 4, 4]], [[1, 6, 10]], [[1, 16, 16]]]
            for seed, grids in enumerate(cases):
                _assert_matches_eager_on_both_ranks(encoder, manager, grids, seed)
            stats = manager.get_cumulative_stats()
            assert stats["graph_hits"] == 7 and stats["graph_misses"] == 0, stats

            # Larger than every budget, or more than one replay: the whole group
            # goes back to the runner's single eager call.
            with torch.inference_mode():
                assert manager.execute(_images([[1, 16, 32]], seed=9)) is None
                assert manager.execute(_images([[1, 16, 16], [1, 4, 8]], seed=10)) is None
            assert manager.get_cumulative_stats()["graph_misses"] == 3
            # Graphs still replay correctly after a fallback.
            _assert_matches_eager_on_both_ranks(encoder, manager, [[1, 8, 8], [1, 4, 8]], seed=11)
        finally:
            manager.clear()
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()


@pytest.mark.parametrize("backend", [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN])
def test_tp2_captured_graphs_match_eager_on_every_rank(backend):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    start_collective_workers(_worker, args=(port, backend), nprocs=2, device_kind="cuda")
