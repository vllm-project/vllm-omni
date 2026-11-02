# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode overlap must preserve outputs and component residency on failures."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as mod

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _pipeline(device, overlap):
    pipe = object.__new__(mod.MiniMaxH3Pipeline)
    pipe.device = torch.device(device)
    pipe.od_config = OmniDiffusionConfig(additional_config={"minimax_h3_vae_audio_overlap": overlap})
    pipe._release_stage_cache = lambda: None
    pipe._component_on_device = lambda component: nullcontext()
    pipe._offload_model_cpu_stage_output = lambda value: value
    pipe.audio_vae = SimpleNamespace(decode_latent=lambda value: value * 2 + 1)
    pipe.video_vae = SimpleNamespace(
        decoder_component=object(), decode_dtype=torch.float16, decode_latent=lambda value: value * 3 - 1
    )
    return pipe


@pytest.mark.cpu
@pytest.mark.parametrize("fail", [False, True])
def test_side_stream_drains_before_components_offload(monkeypatch, fail):
    pipe = _pipeline("cuda", True)
    events = []
    stream = SimpleNamespace(
        wait_stream=lambda caller: events.append("wait"), synchronize=lambda: events.append("drain")
    )
    module = SimpleNamespace(
        current_stream=lambda device: object(), Stream=lambda **kwargs: stream, stream=lambda stream: nullcontext()
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: module)
    monkeypatch.setattr(mod.current_omni_platform, "create_autocast_context", lambda **kwargs: nullcontext())

    @contextmanager
    def resident(component):
        name = "audio" if component is pipe.audio_vae else "video"
        events.append("load " + name)
        try:
            yield
        finally:
            events.append("offload " + name)

    def video_decode(value):
        events.append("video decode")
        if fail:
            raise RuntimeError("video decoder failure")
        return value

    pipe._component_on_device = resident
    pipe.video_vae.decode_latent = video_decode
    pipe.audio_vae.decode_latent = lambda value: Mock()
    if fail:
        with pytest.raises(RuntimeError, match="video decoder failure"):
            pipe.decode(torch.ones(1, 3, 4, 4), Mock(), height=2, width=2)
    else:
        video, _ = pipe.decode(torch.ones(1, 3, 4, 4), Mock(), height=2, width=2)
        assert video.shape == (1, 3, 2, 2)
    assert events.index("drain") < events.index("offload video") < events.index("offload audio")


@pytest.mark.cpu
def test_overlap_rejects_non_cuda():
    with pytest.raises(ValueError, match="requires CUDA"):
        _pipeline("cpu", True).decode(torch.ones(1), torch.ones(1), height=2, width=2)


@hardware_test(res={"cuda": ["L4", "H100", "B200", "RTX5090"]}, num_cards=1)
@torch.no_grad()
def test_cuda_overlap_matches_serial_decode():
    pipe = _pipeline("cuda", False)
    video = torch.randn(1, 3, 8, 16, 16, device=pipe.device)
    audio = torch.randn(1, 2, 4096, device=pipe.device)
    expected = pipe.decode(video, audio, height=12, width=12)
    pipe.od_config.additional_config["minimax_h3_vae_audio_overlap"] = True
    actual = pipe.decode(video, audio, height=12, width=12)
    for serial, overlap in zip(expected, actual, strict=True):
        torch.testing.assert_close(serial, overlap, rtol=0, atol=0)
