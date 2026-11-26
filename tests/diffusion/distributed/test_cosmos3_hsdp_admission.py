# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded Cosmos3 mode admission smoke with real FSDP weight collectives."""

import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]


def _run_cosmos3_hsdp_modes(rank: int, rendezvous: str) -> None:
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    from tests.diffusion.models.cosmos3.test_cosmos3_pipeline import _make_cosmos3_pipeline
    from vllm_omni.diffusion.data import DiffusionOutput, DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched import RequestScheduler
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        config = OmniDiffusionConfig(
            model="cosmos3",
            model_config={"guardrails": False},
            num_gpus=2,
            max_num_seqs=2,
            parallel_config=DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=2, hsdp_data_parallel=True),
        )
        pipeline = _make_cosmos3_pipeline()
        pipeline.od_config = config
        pipeline.device = device
        # Tiny model doubles keep this checkpoint-free. Actual Cosmos3 forward
        # resolves the defaults and runs every denoising step; FSDP gathers this
        # parameter on each transformer forward and reshards it afterward.
        pipeline.transformer.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 2, device=device)))
        fully_shard(pipeline.transformer, mesh=init_device_mesh("cuda", (2,)), reshard_after_forward=True)
        assert isinstance(pipeline.transformer.weight, DTensor)
        pipeline._format_and_tokenize_prompts = lambda *args, **kwargs: (
            torch.ones(1, 1, dtype=torch.long, device=device),
            torch.ones(1, 1, dtype=torch.long, device=device),
            None,
            None,
        )
        preprocess = get_cosmos3_pre_process_func(config)
        scheduler = RequestScheduler()
        scheduler.initialize(config)
        for request_id, mode in (
            ("image", "image"),
            ("video", "video"),
            ("image-a", "image"),
            ("image-b", "image"),
            ("video-a", "video"),
            ("video-b", "video"),
        ):
            scheduler.add_request(
                preprocess(
                    OmniDiffusionRequest(
                        request_id=request_id,
                        prompt={"prompt": "A mountain lake.", "modalities": [mode]},
                        sampling_params=OmniDiffusionSamplingParams(
                            height=16,
                            width=16,
                            num_frames=5,
                            num_inference_steps=None,
                            guidance_scale=1.0,
                            output_type="latent",
                        ),
                    )
                )
            )

        completed = set()
        for expected_ids in (["image"], ["video"], ["image-a", "image-b"], ["video-a", "video-b"]):
            wave = scheduler.schedule()
            assert wave.scheduled_request_ids == expected_ids
            # Match the worker's short-wave padding: every shard rank executes
            # a request, including when the wave contains only one request.
            request = wave.scheduled_new_reqs[rank % len(wave.scheduled_new_reqs)].req
            mode = request.prompt["modalities"][0]
            expected_steps = 50 if mode == "image" else 35
            before = len(pipeline.transformer.calls)
            with torch.no_grad():
                result = pipeline.forward(DiffusionRequestBatch(requests=[request]))
            assert len(pipeline.transformer.calls) - before == expected_steps
            assert result.output[mode].numel() > 0

            results = [None, None]
            dist.all_gather_object(results, request.request_id)
            assert set(results) == set(expected_ids)
            completed.update(
                scheduler.update_from_output(
                    wave,
                    BatchRunnerOutput.from_list(
                        [
                            RunnerOutput(request_id=r, finished=True, result=DiffusionOutput(output="done"))
                            for r in expected_ids
                        ]
                    ),
                )
            )
        assert completed == {"image", "video", "image-a", "image-b", "video-a", "video-b"}
        assert scheduler.num_waiting_requests() == scheduler.num_running_requests() == 0
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=2)
@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.accelerator.device_count() < 2,
    reason="requires two CUDA GPUs and NCCL",
)
def test_cosmos3_hsdp_mixed_modes_complete(tmp_path) -> None:
    context = torch.multiprocessing.spawn(
        _run_cosmos3_hsdp_modes, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=False
    )
    deadline = time.monotonic() + 120
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Cosmos3 HSDP mode waves did not complete within 120 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
