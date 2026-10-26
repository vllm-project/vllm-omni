# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise MP4 prestart with real Gloo collectives and spawned ranks."""

import json
import multiprocessing
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _prestart_worker(rank, ranks, init_method, result_dir, fail_start):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
    from vllm_omni.diffusion.utils import chunked_video

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=max(ranks) + 1, init_method=init_method, timeout=timedelta(seconds=10)
    )
    try:
        group = dist.new_group(ranks=list(ranks), backend="gloo", timeout=timedelta(seconds=10))
        if rank not in ranks:
            return
        events = []

        class AudioVAE:
            def decode_latent(self, latent):
                return torch.zeros(1, 1, 8)

        class VideoVAE(MiniMaxH3VideoVAE):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.parallel_size = len(ranks)
                self.decode_dtype = torch.float32

            @property
            def decoder_component(self):
                return self

            def _native_parallel_state(self):
                return {"sp_process_group": group}

            def decode_with_chunks(self, latent, *, on_chunk):
                events.append("decode")
                # Stand in for the first collective in the native VAE decode.
                dist.all_reduce(torch.ones(1), group=group)
                if self.owns_chunk_output:
                    on_chunk(torch.zeros(1, 3, 1, 2, 2))

        class Session:
            def __init__(self, **kwargs):
                self.has_frames = False

            def start(self, **kwargs):
                events.append("start")
                if fail_start:
                    raise ValueError("owner codec initialization failed")

            def push(self, frames):
                self.has_frames = True

            def finish(self):
                return [b"mp4"] if self.has_frames else []

            def transfer_stats(self):
                return None

            def abort(self):
                events.append("abort")

        pipeline = object.__new__(mod.MiniMaxH3Pipeline)
        torch.nn.Module.__init__(pipeline)
        pipeline.video_vae = VideoVAE()
        pipeline.audio_vae = AudioVAE()
        pipeline.device = torch.device("cpu")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mod.MiniMaxH3Pipeline, "_component_on_device", lambda *args: nullcontext())
            patch.setattr(mod.current_omni_platform, "create_autocast_context", lambda **kwargs: nullcontext())
            patch.setattr(chunked_video, "ChunkedVideoMP4Session", Session)
            result = {"rank": rank, "events": events}
            try:
                output = pipeline.decode_to_mp4(torch.zeros(1), torch.zeros(1), height=2, width=2)
                result["output"] = output.decode()
            except Exception as exc:
                result.update(error_type=type(exc).__name__, error=str(exc))
            (Path(result_dir) / f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
@pytest.mark.parametrize("ranks", [(0,), (0, 1), (1, 2)], ids=["single", "two-rank", "subgroup-owner-global-1"])
@pytest.mark.parametrize("fail_start", [False, True], ids=["success", "owner-start-failure"])
def test_preencode_start_keeps_vae_ranks_consistent(tmp_path, ranks, fail_start):
    context = multiprocessing.get_context("spawn")
    init_method = (tmp_path / "rendezvous").as_uri()
    workers = [
        context.Process(target=_prestart_worker, args=(rank, ranks, init_method, str(tmp_path), fail_start))
        for rank in range(max(ranks) + 1)
    ]
    try:
        for worker in workers:
            worker.start()
        deadline = time.monotonic() + 90
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        assert not any(worker.is_alive() for worker in workers), "MP4 prestart left a VAE rank hanging"
        assert all(worker.exitcode == 0 for worker in workers)
        for rank in ranks:
            result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
            owner = rank == ranks[0]
            assert result["events"].count("start") == int(owner)
            if fail_start:
                assert "decode" not in result["events"], result
                assert "abort" in result["events"]
                assert result["error_type"] == ("ValueError" if owner else "RuntimeError")
                assert ("owner codec initialization failed" if owner else "MP4 encoder startup failed") in result[
                    "error"
                ]
            else:
                assert "error" not in result, result
                assert "decode" in result["events"]
                assert "abort" not in result["events"]
                assert result["output"] == ("mp4" if owner else "")
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=5)
