# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit real-model archived sampling rollout, on one GPU or SP2."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import psutil
import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"}, num_cards=1)]


def load_pipeline(sp_size=1):
    from vllm.config import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline

    config = OmniDiffusionConfig(
        model=os.environ["HY_NOMAS_MODEL"],
        parallel_config={"ulysses_degree": sp_size},
        dtype=torch.bfloat16,
        distributed_executor_backend="mp",
        enforce_eager=True,
        custom_pipeline_args={
            "ar_height": 384,
            "ar_width": 640,
            "ar_max_latent_frames": 32,
            "ar_max_segments": 4,
            "ar_history_mode": "recent",
            "ar_history_latent_frames": 18,
        },
    )
    with set_default_torch_dtype(torch.bfloat16):
        pipeline = HyNomasPipeline(od_config=config)
    pipeline.load_weights(DiffusersPipelineLoader(LoadConfig(), config).get_all_weights(pipeline))
    return config, pipeline


def request(session_id, index, *, close=False, reset=False):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionControlInput, ARDiffusionTickRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompts = [
        "A forest beside a clear river, daylight.",
        "The camera approaches a wooden bridge beside the river.",
        "Birds fly above the river as the camera moves toward the bridge.",
    ]
    prompt = prompts[index % len(prompts)]
    tick = ARDiffusionTickRequest(
        session_id,
        f"{session_id}-{index}",
        index,
        prompt=prompt,
        close_session=close,
        reset=reset,
        controls=(ARDiffusionControlInput("camera", "hy_nomas.camera.v1", {"camera_trajectory": "pan_left"}),),
    )
    return OmniDiffusionRequest(
        prompt=prompt,
        request_id=tick.request_id,
        sampling_params=OmniDiffusionSamplingParams(
            height=384,
            width=640,
            num_frames=49,
            seed=42,
            num_inference_steps=4,
            guidance_scale=1,
            extra_args=tick.to_extra_args(),
        ),
    )


def check_released(runner, archive, free_blocks):
    assert not runner._sessions and not runner.pipeline._ar_sessions
    assert archive.records == () and archive.frames == archive.payload_bytes == 0
    assert runner.kv_cache.manager.block_pool.get_num_free_blocks() == free_blocks
    with pytest.raises(RuntimeError, match="no pending"):
        archive.publish()


@torch.no_grad()
def run_worker(args):
    import imageio.v2 as imageio

    from vllm_omni.diffusion.distributed.utils import get_local_device
    from vllm_omni.diffusion.models.hy_nomas.history_sampling import (
        history_frame_indices,
        select_history,
        selected_layers,
        selected_mask,
    )
    from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
    from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
    from vllm_omni.experimental.ar_diffusion.tick_protocol import AR_DIFFUSION_TICK_KEY
    from vllm_omni.platforms import current_omni_platform

    rank = int(os.environ.get("RANK", 0))
    config, pipeline = load_pipeline(int(os.environ.get("WORLD_SIZE", 1)))
    pipeline.ar_config = replace(
        pipeline.ar_config,
        height=args.height,
        width=args.width,
        history_mode=args.mode,
        max_total_segments=max(args.turns, 4),
        history_latent_frames=args.budget,
        history_archive_max_bytes=args.archive_gib * 1024**3,
        history_archive_max_frames=1024,
    )
    runner = ARDiffusionModelRunner(None, config, get_local_device())
    runner.pipeline = pipeline
    runner.ar_diffusion_kv_config = ARDiffusionKVConfig(enable=True, gpu_memory_fraction=0.01, warmup_cudagraph=False)
    runner._preallocate_kv_cache(
        available_bytes=min(current_omni_platform.mem_get_info()[0], args.admission_gib * 1024**3)
    )
    free = runner.kv_cache.manager.block_pool.get_num_free_blocks()
    records, frames, first_hash = [], 0, None
    archive = None
    latent_frames = (args.frames - 1) // 4 + 1
    prepare_chunk = pipeline.prepare_next_chunk

    def check_camera(state):
        prepare_chunk(state)
        assert state.extra["hy_nomas_camera_override"] == camera
        assert (state.extra["hy_nomas_camera"] is None) == (camera is None)

    pipeline.prepare_next_chunk = check_camera
    writer = imageio.get_writer(str(args.output / f"{args.mode}.mp4"), fps=24) if rank == 0 else None
    try:
        for index in range(args.turns):
            if psutil.virtual_memory().available < 256 * 1024**3:
                raise RuntimeError("Stop test: shared host has less than 256 GiB available")
            started = time.monotonic()
            item = request("sampling", index, close=index == args.turns - 1)
            item.sampling_params.height = args.height
            item.sampling_params.width = args.width
            item.sampling_params.num_frames = args.frames
            item.sampling_params.num_inference_steps = args.steps
            camera = ("pan_left", "pan_right", None, "forward")[index % 4] if args.camera_cycle else "pan_left"
            tick = item.sampling_params.extra_args[AR_DIFFUSION_TICK_KEY]
            tick["controls"][0]["data"]["camera_trajectory"] = camera
            tick["applied_event_ids"] = [index * 2, index * 2 + 1]
            output = runner.execute_model(item)
            assert not output.error, output.error
            video = output.output["payload"]["video"]
            meta = output.output["metadata"]["hy_nomas"]
            assert output.output["metadata"]["ar_diffusion"]["applied_event_ids"] == tick["applied_event_ids"]
            assert meta["frame_start"] == frames and meta["num_frames"] == len(video)
            frames += len(video)
            if writer is not None:
                for frame in video:
                    writer.append_data(frame)
            positions = history_frame_indices(latent_frames * (index + 1), args.budget, args.mode)
            if index < args.turns - 1:
                session = pipeline._ar_sessions["sampling"]
                archive = session.history_archive
                assert tuple(frame.position for frame in session.history_frames) == positions
                assert archive.frames == latent_frames * (index + 1)
                digest = hashlib.sha256(archive.records[0].layers[0][0].view(torch.uint8).numpy().tobytes()).hexdigest()
                assert first_hash is None or digest == first_hash
                first_hash = digest
                # Early and final working sets: check every layer's restored rows.
                if index in (0, 2, args.turns - 2):
                    cache = runner._sessions["sampling"].segmented("cond")
                    selected = select_history(archive.records, positions)
                    torch.testing.assert_close(session.history_mask.cpu(), selected_mask(selected), rtol=0, atol=0)
                    cache.prepare_layer_views(1, 0)
                    shape = (1, cache.cache.num_kv_heads, cache.cache.head_size)
                    current = torch.zeros(shape, device=pipeline.device, dtype=pipeline.target_dtype)
                    for layer, (expected_k, expected_v) in enumerate(
                        selected_layers(selected, pipeline.num_double_blocks)
                    ):
                        key, value = cache.assemble(layer, current[:0], current[:0], current, current)
                        torch.testing.assert_close(key[:-1].cpu(), expected_k, rtol=0, atol=0)
                        torch.testing.assert_close(value[:-1].cpu(), expected_v, rtol=0, atol=0)
                    del selected, expected_k, expected_v
            assert meta["history_tokens"] <= pipeline.ar_diffusion_kv_cache_spec().segmented.max_retained_tokens
            record = {
                "turn": index,
                "frame_ids": positions,
                "elapsed_seconds": time.monotonic() - started,
                "rss_bytes": psutil.Process().memory_info().rss,
                "camera": camera,
                **meta,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
        assert archive is not None and frames == args.turns * args.frames and meta["withheld_frames"] == 0
        check_released(runner, archive, free)
        result = {
            "policy": args.mode,
            "budget": args.budget,
            "rank": rank,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "frames_per_turn": args.frames,
            "max_total_segments": pipeline.ar_config.max_total_segments,
            "archive_quota_gib_per_rank": args.archive_gib,
            "video_seconds": frames / 24,
            "free_blocks_after_close": free,
            "records": records,
            "scope": "seed42, prompt/camera controls; functional rollout, not W6 quality acceptance",
        }
        (args.output / f"{args.mode}-rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"PASS: {args.mode} rank={rank}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        for session_id in tuple(runner._sessions):
            runner.close_session(session_id)


@pytest.mark.parametrize("mode", ["uniform", "recent_mix_uniform", "temporal_context"])
def test_sampling_rollout(tmp_path, mode):
    with (tmp_path / "sampling.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.e2e.accuracy.test_hy_nomas_sampling",
                "--mode",
                mode,
                "--output",
                str(tmp_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["uniform", "recent_mix_uniform", "temporal_context"], required=True)
    parser.add_argument("--budget", type=int, default=18)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--frames", type=int, choices=(49, 61), default=49)
    parser.add_argument("--archive-gib", type=int, default=128)
    parser.add_argument("--admission-gib", type=int, default=100)
    parser.add_argument("--camera-cycle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        from vllm_omni.diffusion.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            world_size=world, rank=int(os.environ["RANK"]), local_rank=int(os.environ["LOCAL_RANK"])
        )
        initialize_model_parallel(sequence_parallel_size=world, ulysses_degree=world)
    try:
        run_worker(args)
    finally:
        if world > 1:
            from vllm_omni.diffusion.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
