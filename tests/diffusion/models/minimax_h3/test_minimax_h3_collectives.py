# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _collective_worker(rank, rendezvous, backend):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    from vllm_omni.errors import OmniClientError
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        current_omni_platform.set_device(device)
    dist.init_process_group(
        backend, init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        module._dit_rank_world = lambda: (dist.group.WORLD, rank, 2)

        class VisualCodec:
            fail = False

            def is_distributed_enabled(self):
                return True

            def collective(self):
                value = torch.ones(1, device=device)
                dist.all_reduce(value)
                assert value.item() == 2

            def encode_image(self, image):
                self.collective()
                assert image.size == (32, 32)
                return torch.ones(1, 96)

            def encode_video(self, frames):
                self.collective()
                assert frames.shape == (4, 32, 32, 3)
                if self.fail and rank == 1:
                    raise ValueError("codec failed on rank 1")
                return torch.full((1, 96), 2.0), (1, 2, 2)

        class AudioCodec:
            def encode_waveform(self, waveform, sample_rate):
                assert rank == 0
                assert sample_rate == 32000
                assert waveform.shape == (64000,)
                return torch.full((160, 32), 3.0), 80

        model = object.__new__(module.MiniMaxH3Pipeline)
        torch.nn.Module.__init__(model)
        model.device = device
        model.od_config = SimpleNamespace(enable_layerwise_offload=False)
        model.video_vae = VisualCodec()
        model.audio_vae = AudioCodec()
        model.partition = "combined"
        model.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
        model.text_encoder_tp_size = 2
        model.text_encoder_group = SimpleNamespace(
            world_size=2, ranks=[0, 1], rank_in_group=rank, device_group=dist.group.WORLD
        )
        model.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        model.tokenizer = lambda text, **kwargs: {"input_ids": [1, 2, 3]}

        def encode_text(ids, kwargs):
            assert ids.tolist() == [1, 2, 3]
            assert not kwargs
            hidden = torch.ones(3, 5120, device=device, dtype=torch.bfloat16)
            dist.all_reduce(hidden)
            return hidden

        model._encode_text_hidden = encode_text
        sampling = OmniDiffusionSamplingParams(
            height=32, width=32, num_frames=96, extra_args={"task": "t2va", "aspect_ratio": "1:1"}
        )
        text_conditioning = model._prepare_local_conditioning({"prompt": "test"}, sampling)
        torch.testing.assert_close(text_conditioning.hidden_states, torch.full_like(text_conditioning.hidden_states, 2))
        assert text_conditioning.visual_condition is None

        media = (
            MiniMaxH3EncoderMediaInput(
                task="ref2va",
                height=32,
                width=32,
                num_frames=107,
                latent_t=32,
                audio_t=178,
                images=(torch.zeros(32, 32, 3, dtype=torch.uint8),),
                videos=(torch.zeros(4, 32, 32, 3, dtype=torch.uint8),),
                video_audios=(None,),
                audios=((torch.zeros(64000), 32000),),
            )
            if rank == 0
            else None
        )
        conditioning = model._encode_local_media(media)
        assert [block["kind"] for block in conditioning.ref_blocks] == ["image", "video", "audio"]
        torch.testing.assert_close(
            conditioning.visual_condition.cpu(), torch.cat([torch.ones(1, 96), torch.full((1, 96), 2.0)])
        )
        torch.testing.assert_close(conditioning.audio_condition.cpu(), torch.full((160, 32), 3.0))
        assert conditioning.audio_condition_lengths == (80,)

        model.video_vae.fail = True
        with pytest.raises(OmniClientError, match="codec failed on rank 1"):
            model._encode_local_media(media)
        with pytest.raises(OmniClientError, match="non-empty prompt"):
            model._prepare_local_conditioning({"prompt": ""}, sampling)
        model.video_vae.fail = False
        assert model._encode_local_media(media).visual_condition_shapes == ((1, 2, 2), (1, 2, 2))
    finally:
        dist.destroy_process_group()


def _run_collectives(tmp_path, backend):
    context = mp.spawn(_collective_worker, args=(str(tmp_path / "rendezvous"), backend), nprocs=2, join=False)
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("MiniMax H3 collectives did not finish within 150 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.cpu
def test_two_rank_conditioning_and_errors_gloo(tmp_path):
    _run_collectives(tmp_path, "gloo")


@pytest.mark.cuda
@pytest.mark.parallel
@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_MINIMAX_H3_COLLECTIVES") != "1",
    reason="set VLLM_TEST_MINIMAX_H3_COLLECTIVES=1 with two reserved GPUs",
)
def test_two_rank_conditioning_and_errors_nccl(tmp_path):
    if torch.accelerator.device_count() < 2:
        pytest.skip("requires two GPUs")
    _run_collectives(tmp_path, "nccl")
