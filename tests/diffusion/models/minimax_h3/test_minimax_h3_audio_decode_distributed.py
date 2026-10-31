# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _decode_worker(rank, init_method):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=init_method, timeout=timedelta(seconds=60))
    try:
        expected = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
        decoder = Mock(return_value=expected) if rank == 0 else Mock(side_effect=AssertionError("peer decoded audio"))
        pipeline = SimpleNamespace(
            device=torch.device("cpu"),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            video_vae=Mock(decode_latent=Mock(return_value=torch.ones(1, 1, 4, 4))),
            audio_vae=Mock(decode_latent=decoder),
            _component_on_device=lambda component: nullcontext(),
            _offload_model_cpu_stage_output=lambda value: value,
            _release_stage_cache=lambda: None,
        )
        with (
            patch.object(module, "_dit_rank_world", return_value=(None, rank, 2)),
            patch.object(module.current_omni_platform, "create_autocast_context", return_value=nullcontext()),
        ):
            _, audio = module.MiniMaxH3Pipeline.decode(pipeline, torch.zeros(1), torch.zeros(1), height=2, width=3)
            torch.testing.assert_close(audio, expected, rtol=0, atol=0)
            assert decoder.call_count == (1 if rank == 0 else 0)

            if rank == 0:
                decoder.side_effect = RuntimeError("audio decode failed")
            with pytest.raises(RuntimeError, match="audio decode failed"):
                module.MiniMaxH3Pipeline.decode(pipeline, torch.zeros(1), torch.zeros(1), height=2, width=3)
            assert decoder.call_count == (2 if rank == 0 else 0)

            # Different DP replicas own different requests and waveforms.
            pipeline.parallel_config.data_parallel_size = 2
            expected = torch.full((1, 2, 8), float(rank + 10))
            decoder.reset_mock(side_effect=True)
            decoder.return_value = expected
            _, audio = module.MiniMaxH3Pipeline.decode(pipeline, torch.zeros(1), torch.zeros(1), height=2, width=3)
            torch.testing.assert_close(audio, expected, rtol=0, atol=0)
            decoder.assert_called_once()
    finally:
        dist.destroy_process_group()


def test_audio_decode_broadcasts_waveform_and_owner_failure(tmp_path):
    mp.spawn(_decode_worker, args=((tmp_path / "audio-decode-group").as_uri(),), nprocs=2, join=True)
