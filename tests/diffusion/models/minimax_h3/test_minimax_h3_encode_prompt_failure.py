# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression test for the MiniMax-H3 rank-0 encoder failure path.

``encode_prompt`` runs the text encoder on the first ``text_encoder_tp_size``
DiT ranks and broadcasts the result to every rank. When the encoder forward
raises on rank 0 (an Ascend operator tiling error, an OOM, a shape mismatch),
rank 0 used to leave the function before the broadcast while the other ranks
entered it, so they blocked on a collective nobody would join and the worker
wedged: every later request failed while ``/health`` kept answering.

The failure now travels through ``_broadcast_rank0_exception``, so all ranks
raise the same error together.
"""

import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_ENCODER_FAILURE = "npu_rms_norm: NPU function error: call aclnnRmsNorm failed, error code is 561002"


def _encode_prompt_worker(rank, rendezvous, backend):
    """Drive ``encode_prompt`` on two ranks, with and without a failing encoder."""
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.set_device(device)
    dist.init_process_group(
        backend, init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        module._dit_rank_world = lambda: (dist.group.WORLD, rank, 2)

        model = object.__new__(module.MiniMaxH3Pipeline)
        torch.nn.Module.__init__(model)
        model.device = device
        # The encoder runs on rank 0 only; the other rank waits for the
        # broadcast. This is the configuration that wedged.
        model.text_encoder_tp_size = 1
        model.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        model.tokenizer = lambda text, **kwargs: {"input_ids": [1, 2, 3]}
        model._distribute_encode_inputs = lambda ids, kwargs: ids

        prepared = SimpleNamespace(
            prompt="test",
            media=SimpleNamespace(task="t2va"),
            condition_labels=(),
            images=None,
            qwen_videos=None,
            video_timestamps=None,
        )

        def encode_text(ids, kwargs):
            if rank == 0:
                raise RuntimeError(_ENCODER_FAILURE)
            return torch.ones(3, 5120, device=device, dtype=torch.bfloat16)

        model._encode_text_hidden = encode_text

        started = time.monotonic()
        try:
            model.encode_prompt(prepared)
            outcome = "returned"
            detail = ""
        except Exception as exc:  # noqa: BLE001
            outcome = "raised"
            detail = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started

        print(f"RANK {rank} OUTCOME {outcome} ELAPSED {elapsed:.3f} DETAIL {detail}", flush=True)
        assert outcome == "raised", f"rank {rank} expected the encoder failure to propagate"
        # The other rank rebuilds the error with a "[rank 0]" prefix; both must
        # carry the operator failure so the runner records one error per rank.
        assert _ENCODER_FAILURE in detail, f"rank {rank} lost the operator error: {detail}"
        # A wedged rank would sit in the collective until the timeout.
        assert elapsed < 30, f"rank {rank} took {elapsed:.1f}s, which suggests it waited on a collective"
    finally:
        dist.destroy_process_group()


def _run_encode_prompt(tmp_path, backend):
    context = mp.spawn(
        _encode_prompt_worker,
        args=(str(tmp_path / "rendezvous"), backend),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("MiniMax H3 encode_prompt did not finish within 150 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.cpu
def test_two_rank_encoder_failure_reaches_every_rank(tmp_path):
    """A rank-0 encoder failure must not leave the peers waiting in a collective."""
    _run_encode_prompt(tmp_path, "gloo")


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parallel
@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_MINIMAX_H3_COLLECTIVES") != "1",
    reason="set VLLM_TEST_MINIMAX_H3_COLLECTIVES=1 with two reserved GPUs",
)
def test_two_rank_encoder_failure_reaches_every_rank_nccl(tmp_path):
    if torch.accelerator.device_count() < 2:
        pytest.skip("requires two GPUs")
    _run_encode_prompt(tmp_path, "nccl")
