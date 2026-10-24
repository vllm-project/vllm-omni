# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded decoder batches preserve rank-local tile and sample order."""

import importlib.util
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Load the leaf module without importing the GPU pipeline package.
_path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vae_tiling.py"
_spec = importlib.util.spec_from_file_location("h3_tile_batch", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
TileBatchExecutor = _module.TileBatchExecutor


class Checkpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_batches = []
        self.encoder_calls = []
        self.fail = False

    def decode(self, x):
        if self.fail:
            raise torch.OutOfMemoryError("injected decoder failure")
        self.forward_batches.append(x.shape[0])
        return x * 2 + 1

    def encode(self, x):
        return x

    def _run_tile_tasks(self, tiles, indices, forward_fn, stack_tiling, cls_agg=None):
        self.encoder_calls.append((indices, stack_tiling, cls_agg))
        return [forward_fn(tiles[i]) for i in indices]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("sample_batch", [1, 2])
def test_eight_ranks_tail_batches_and_sample_order(k, rank, sample_batch):
    model = Checkpoint().eval()
    executor = TileBatchExecutor(model, k)
    tiles = [torch.arange(sample_batch * 8).reshape(sample_batch, 2, 2, 2) + i * 100 for i in range(28)]
    indices = list(range(rank, 28, 8))
    outputs = executor.run(tiles, indices, model.decode, True)
    expected_groups = [min(k, len(indices) - start) for start in range(0, len(indices), k)]
    assert model.forward_batches == [q * sample_batch for q in expected_groups]
    assert len(outputs) == len(indices)
    for output, index in zip(outputs, indices):
        torch.testing.assert_close(output, tiles[index] * 2 + 1)


def test_control_status_stays_on_cpu_under_non_cpu_default_device():
    model = Checkpoint().eval()
    with torch.device("meta"):
        executor = TileBatchExecutor(model, 4)
    assert executor.status.device == torch.device("cpu")
    assert executor.status.dtype == torch.int32
    assert executor.status.item() == 0


def test_encoder_is_delegated_and_empty_decoder_is_valid():
    model = Checkpoint().eval()
    executor = TileBatchExecutor(model, 4)
    agg = object()
    tile = torch.zeros(1, 2, 2)
    executor.run([tile], [0], model.encode, True, agg)
    assert model.encoder_calls == [([0], True, agg)]
    assert executor.run([], [], model.decode, False) == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4", None])
def test_invalid_batch_size(value):
    with pytest.raises(ValueError, match="positive integer"):
        TileBatchExecutor(Checkpoint().eval(), value)


def test_mixed_shapes_and_decoder_failure_propagate():
    model = Checkpoint().eval()
    executor = TileBatchExecutor(model, 4)
    with pytest.raises(ValueError, match="matching shapes"):
        executor.run([torch.zeros(1, 2), torch.zeros(1, 3)], [0, 1], model.decode, False)
    model.fail = True
    with pytest.raises(torch.OutOfMemoryError, match="injected"):
        executor.run([torch.zeros(1, 2)], [0], model.decode, False)


def _distributed_worker(rank, rendezvous, results, k, failure):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=10)
    )
    try:
        model = Checkpoint().eval()
        executor = TileBatchExecutor(model, k, dist.group.WORLD, control_timeout=timedelta(seconds=2))
        if rank == 1 and failure == "exit":
            os._exit(0)
        if rank == 1 and failure == "timeout":
            time.sleep(4)
        if rank == 1 and failure == "oom":
            model.fail = True
        if rank == 1 and failure == "error":
            # Incorrect shape forces a normal local validation error.
            tiles = [torch.zeros(1, 2), torch.zeros(1, 3)]
        else:
            tiles = [torch.full((1, 2), i, dtype=torch.float32) for i in range(4)]
        try:
            indices = [0, 1] if rank == 1 and failure == "error" else list(range(4 if rank == 0 else 3))
            outputs = executor.run(tiles, indices, model.decode, False)
            result = {"ok": True, "tiles": len(outputs), "batches": model.forward_batches}
        except (RuntimeError, ValueError) as exc:
            result = {"ok": False, "error": str(exc), "fatal": getattr(exc, "requires_stage_shutdown", False)}
        Path(results, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("k,failure", [(3, None), (4, None), (4, "oom"), (4, "error"), (4, "exit"), (4, "timeout")])
def test_distributed_completion_and_failures(tmp_path, k, failure):
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=_distributed_worker, args=(rank, str(tmp_path / "rendezvous"), str(tmp_path), k, failure)
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 40
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert all(not p.is_alive() for p in processes), "a peer was left waiting after a rank failure"
        assert all(p.exitcode == 0 for p in processes)
        for rank in range(2):
            if rank == 1 and failure == "exit":
                continue
            result = json.loads((tmp_path / f"rank{rank}.json").read_text())
            assert result["ok"] is (failure is None)
            if failure is not None:
                assert result["fatal"] is True
            if failure is None:
                assert result["tiles"] == (4 if rank == 0 else 3)
                assert result["batches"] == ([3, 1] if rank == 0 and k == 3 else [4 if rank == 0 else 3])
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
