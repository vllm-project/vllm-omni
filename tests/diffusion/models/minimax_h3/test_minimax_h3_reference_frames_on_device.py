# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MiniMaxH3EncoderMediaInput,
    MiniMaxH3TimelineGuide,
)
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

VIDEO = torch.arange(4 * 32 * 32 * 3).remainder(256).to(torch.uint8).view(4, 32, 32, 3)
EDITS_AND_GUIDES = {
    "video_audios": (None, None),
    "audios": (),
    "video_edit": torch.zeros(107, 32, 32, 3, dtype=torch.uint8),
    "video_edit_mask": torch.zeros(32, 2, 2),
    "timeline_guides": (MiniMaxH3TimelineGuide(start=0, frames=torch.ones(1, 32, 32, 3, dtype=torch.uint8)),),
}


class _BroadcastBuffer(torch.Tensor):
    def cpu(self, *args, **kwargs):
        return self.as_subclass(torch.Tensor).clone()


class _VideoVAE:
    def __init__(self, prepares_on_device=True):
        self.prepares_on_device = prepares_on_device
        self.queries = 0
        self.received = []

    def prepares_video_on_device(self):
        self.queries += 1
        return self.prepares_on_device

    def encode_video(self, frames):
        self.received.append(frames)
        return torch.zeros(4, 96), (1, 4, 4)


def _media(**overrides):
    media = MiniMaxH3EncoderMediaInput(
        task="ref2va",
        height=32,
        width=32,
        num_frames=107,
        latent_t=32,
        audio_t=178,
        images=(torch.full((32, 32, 3), 7, dtype=torch.uint8),),
        videos=(VIDEO, torch.full((3, 32, 32, 3), 9, dtype=torch.uint8)),
        video_audios=((torch.ones(48000), 16000), None),
        audios=((torch.zeros(64000), 32000),),
    )
    return replace(media, **overrides)


@pytest.mark.parametrize(
    ("rank", "prepares", "overrides", "copied"),
    [
        (1, True, EDITS_AND_GUIDES, [True, False, False, True, True, True]),
        (1, True, {"videos": (VIDEO.float(), VIDEO)}, [True, True, False, True, True]),
        (1, False, {}, [True] * 5),
        (1, True, {"videos": (), "video_audios": ()}, [True, True]),
        (0, True, {}, None),
    ],
)
def test_distribute_media_inputs_keeps_only_uint8_videos_on_device(monkeypatch, rank, prepares, overrides, copied):
    media = _media(**overrides)
    tensors = media.to_mm_tensors()
    header = (media.to_metadata(), [value.dtype for value in tensors])
    broadcast: list[torch.Tensor] = []

    def fake_broadcast(tensor, *, dtype, device):
        broadcast.append(tensors[len(broadcast)].clone().as_subclass(_BroadcastBuffer))
        return broadcast[-1]

    monkeypatch.setattr(pipeline_module, "_dit_rank_world", lambda: (None, rank, 2))
    monkeypatch.setattr(
        pipeline_module.dist, "broadcast_object_list", lambda values, src, group: values.__setitem__(0, header)
    )
    monkeypatch.setattr(pipeline_module, "_broadcast_tensor", fake_broadcast)
    model = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.video_vae = _VideoVAE(prepares)
    result = model._distribute_media_inputs(media if rank == 0 else None)
    assert model.video_vae.queries == (rank != 0 and bool(media.videos))
    if copied is None:
        assert result is media
        return
    received = result.to_mm_tensors()
    assert [value is not buffer for value, buffer in zip(received, broadcast, strict=True)] == copied
    assert result.to_metadata() == media.to_metadata()
    assert all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(received, tensors, strict=True))


def test_encode_media_hands_device_videos_to_the_vae_as_tensors():
    resident = torch.empty(4, 32, 32, 3, dtype=torch.uint8, device="meta")
    media = _media(images=(), videos=(resident, VIDEO), video_audios=(None, None), audios=())
    video_vae = _VideoVAE()
    assert encode_media(media, video_vae=video_vae, audio_vae=None, emit_conditioning=False) is None
    device_frames, host_frames = video_vae.received
    assert device_frames.device.type == "meta" and device_frames.shape == resident.shape
    assert isinstance(host_frames, np.ndarray) and np.array_equal(host_frames, VIDEO.numpy())
