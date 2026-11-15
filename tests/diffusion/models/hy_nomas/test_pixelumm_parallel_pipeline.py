# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.diffusion.ar_diffusion.test_capability_runner import make_runner
from tests.diffusion.models.hy_nomas.test_ar import PipelineWithCPUComponents
from tests.diffusion.models.hy_nomas.test_pixelumm_parallel import make_decoder
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


def make_request(index, *, frames=9, session="s", close=False):
    tick = ARDiffusionTickRequest(session, f"request-{index}", index, (index,), "forest", (), close_session=close)
    params = OmniDiffusionSamplingParams(
        num_frames=frames,
        height=128,
        width=128,
        seed=7,
        num_inference_steps=4,
        guidance_scale=1,
        extra_args=tick.to_extra_args(),
    )
    return DiffusionRequestBatch(
        requests=[OmniDiffusionRequest(prompt="forest", sampling_params=params, request_id=tick.request_id)]
    )


def _pipeline_worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        torch.set_num_threads(1)

        def all_reduce(value):
            dist.all_reduce(value)
            return value

        group = SimpleNamespace(
            ulysses_rank=rank,
            ulysses_group=dist.group.WORLD,
            world_size=world_size,
            cpu_group=dist.group.WORLD,
            all_reduce=all_reduce,
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas.get_sp_group", lambda: group)
            patch.setattr("vllm_omni.diffusion.distributed.parallel_state.get_sp_group", lambda: group)
            patch.setattr(
                "vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas.action_conditioning",
                lambda *args: (2, None, None),
            )
            patch.setattr(
                "vllm_omni.experimental.ar_diffusion.runner.current_omni_platform",
                SimpleNamespace(synchronize=lambda: None),
            )
            outputs, positions, offline_outputs = {}, {}, {}
            for mode in ("none", "window"):
                pipeline = PipelineWithCPUComponents()
                pipeline.sp_size = world_size
                pipeline.heads_num = 2 * world_size
                pipeline.hidden_size = 8 * world_size
                pipeline.in_channels = 32
                pipeline.transformer.config.out_channels = 32
                pipeline.decoder_parallel_mode = mode
                pipeline.vae = make_decoder()
                if mode == "window":
                    pipeline.vae.configure_distributed_decode(dist.group.WORLD)
                pipeline.initialize_ar(
                    {
                        "ar_height": 128,
                        "ar_width": 128,
                        "ar_chunk_latent_frames": 3,
                        "ar_max_latent_frames": 12,
                        "ar_max_segments": 4,
                        "ar_session_capacity": 1,
                    }
                )
                runner = make_runner(pipeline)
                videos, starts, committed = [], [], []
                for index in range(3):
                    request = make_request(index, close=index == 2)
                    with runner._bound_ar_session("s", description="window integration"):
                        output = pipeline.forward(request)
                    videos.append(output.output["payload"]["video"])
                    starts.append(output.output["metadata"]["stream"]["frame_start"])
                    committed.append(pipeline._ar_sessions["s"].emitted_frames)
                outputs[mode] = np.concatenate(videos, axis=0)
                positions[mode] = {"starts": starts, "committed": committed}
                runner.close_session("s")
                request = make_request(0)
                request.requests[0].sampling_params.extra_args = {}
                offline = pipeline.forward(request)
                offline_outputs[mode] = None if offline.media is None else offline.media.video.tensor
                if mode == "window":
                    failed = make_request(0, frames=25, session="failed", close=True)
                    failed.requests[0].sampling_params.extra_args = {"session_id": "failed", "close_session": True}
                    pipeline.vae.decoder.fail = rank == 1
                    with pytest.raises(RuntimeError, match="window failure|another decode rank|another SP rank"):
                        with runner._bound_ar_session("failed", description="window failure"):
                            pipeline.forward(failed)
                    pipeline.vae.decoder.fail = False
                    assert "failed" not in runner._sessions and "failed" not in pipeline._ar_sessions
                    with runner._bound_ar_session("failed", description="window recovery"):
                        recovered = pipeline.forward(failed)
                    assert len(recovered.output["payload"]["video"]) == (25 if rank == 0 else 0)
                    runner.close_session("failed")

                    if world_size == 2:
                        for failure_rank in (0, 1):
                            oom_triggered = False
                            original_tensor = torch.tensor

                            def fail_decoder(*args, **kwargs):
                                nonlocal oom_triggered
                                oom_triggered = True
                                raise torch.OutOfMemoryError("injected decoder OOM")

                            def fail_status_allocation(*args, **kwargs):
                                if oom_triggered and kwargs.get("dtype") == torch.int32:
                                    raise torch.OutOfMemoryError("injected status allocation OOM")
                                return original_tensor(*args, **kwargs)

                            session_id = f"oom-{failure_rank}"
                            oom_request = make_request(0, frames=25, session=session_id, close=True)
                            oom_request.requests[0].sampling_params.extra_args = {
                                "session_id": session_id,
                                "close_session": True,
                            }
                            with pytest.MonkeyPatch.context() as oom_patch:
                                if rank == failure_rank:
                                    oom_patch.setattr(pipeline.vae.decoder, "forward", fail_decoder)
                                    oom_patch.setattr(torch, "tensor", fail_status_allocation)
                                with pytest.raises(
                                    RuntimeError, match="decoder OOM|another decode rank|another SP rank"
                                ):
                                    with runner._bound_ar_session(session_id, description="window OOM"):
                                        pipeline.forward(oom_request)
                            assert session_id not in runner._sessions and session_id not in pipeline._ar_sessions
                            with runner._bound_ar_session(session_id, description="window OOM recovery"):
                                recovered = pipeline.forward(oom_request)
                            assert len(recovered.output["payload"]["video"]) == (25 if rank == 0 else 0)
                            runner.close_session(session_id)

                    @contextmanager
                    def failing_context(component):
                        yield
                        if rank == 1 and component is pipeline.vae:
                            raise RuntimeError("injected component release failure")

                    with pytest.MonkeyPatch.context() as failure_patch:
                        failure_patch.setattr(pipeline, "_component_context", failing_context)
                        with pytest.raises(RuntimeError, match="component release failure|another SP rank"):
                            pipeline.forward(request)
                    recovered_offline = pipeline.forward(request)
                    assert (recovered_offline.media is not None) == (rank == 0)
            if rank == 0:
                assert np.array_equal(outputs["none"], outputs["window"])
                assert outputs["window"].shape == (27, 128, 128, 3)
                assert torch.equal(offline_outputs["none"], offline_outputs["window"])
            else:
                assert outputs["window"].shape == (0, 128, 128, 3)
                assert offline_outputs["window"] is None
            assert positions["none"] == positions["window"]
            assert positions["window"]["committed"][-1] == 27
            peers = [None] * world_size
            dist.all_gather_object(peers, positions["window"])
            assert all(peer == peers[0] for peer in peers)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_pipeline_window_parallel_matches_serial_and_recovers(tmp_path, monkeypatch, world_size):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    mp.spawn(_pipeline_worker, args=(world_size, (tmp_path / "rendezvous").as_uri()), nprocs=world_size, join=True)
