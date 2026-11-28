# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from tests.diffusion.models.minimax_h3.test_minimax_h3_reference_frames_on_device import VIDEO, _media, _VideoVAE
from tests.diffusion.models.minimax_h3.test_minimax_h3_vae_encode_prep import _aligned_model, _vae, _video
from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.parametrize("model", [None, _aligned_model(5)])
def test_resident_frames_normalize_bitwise_like_host_upload(model):
    vae = _vae(model)
    device = torch.device("cuda", torch.accelerator.current_device_index())
    frames = _video(10, height=18, width=30)
    resident = torch.from_numpy(frames).to(device)
    expected = vae._stream_prepare_video_tensor(frames, device)
    assert torch.equal(vae._stream_prepare_video_tensor(resident, device), expected)
    assert torch.equal(resident.cpu(), torch.from_numpy(frames))


def test_encode_media_moves_float_device_videos_to_host():
    device = torch.device("cuda", torch.accelerator.current_device_index())
    media = _media(images=(), videos=(VIDEO.float().to(device),), video_audios=(None,), audios=())
    video_vae = _VideoVAE()
    encode_media(media, video_vae=video_vae, audio_vae=None, emit_conditioning=False)
    assert isinstance(video_vae.received[0], np.ndarray) and np.array_equal(video_vae.received[0], VIDEO.numpy())
