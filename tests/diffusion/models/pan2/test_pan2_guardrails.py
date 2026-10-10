# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.errors import GuardrailViolationError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCKED_PROMPT = "unsafe prompt"


class _FakePlatform:
    """The platform attribute the guardrails read: the compute device type."""

    device_type = "compute"


class _FakeGuard(nn.Module):
    def __init__(self, devices: list[tuple[str, str]], name: str):
        super().__init__()
        self.devices = devices
        self.name = name

    def to(self, device):
        self.devices.append((self.name, str(device)))
        return self


class _FakeChecker(nn.Module):
    """Same interface as `PAN2SafetyChecker`, without weights."""

    def __init__(self, block_video: bool = False):
        super().__init__()
        self.devices: list[tuple[str, str]] = []
        self.text_guard = _FakeGuard(self.devices, "text")
        self.video_guard = _FakeGuard(self.devices, "video")
        self.block_video = block_video
        self.prompts: list[str] = []
        self.videos: list[np.ndarray] = []

    def to(self, device):
        self.devices.append(("checker", str(device)))
        return self

    def check_text_safety(self, prompt: str) -> bool:
        self.prompts.append(prompt)
        return prompt != BLOCKED_PROMPT

    def check_video_safety(self, frames: np.ndarray) -> np.ndarray | None:
        self.videos.append(frames)
        return None if self.block_video else frames


@pytest.fixture
def guardrails(monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.diffusion.models.pan2 import guardrails

    monkeypatch.setattr(guardrails, "_text_guardrail", None)
    monkeypatch.setattr(guardrails, "_video_guardrail", None)
    monkeypatch.setattr(guardrails, "current_omni_platform", _FakePlatform())
    return guardrails


@pytest.fixture
def make_checker(guardrails, monkeypatch: pytest.MonkeyPatch):
    def _make(**kwargs) -> _FakeChecker:
        checker = _FakeChecker(**kwargs)
        monkeypatch.setattr(guardrails.PAN2SafetyChecker, "from_pretrained", staticmethod(lambda: checker))
        return checker

    return _make


def _od_config(**model_config) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(model_config=model_config)


def _request(prompt, **extra_args) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt, sampling_params=OmniDiffusionSamplingParams(extra_args=extra_args), request_id="pan2-test"
    )


def _video(seed: int = 0) -> torch.Tensor:
    # Values slightly outside [-1, 1] exercise the clamping, as a VAE decode can produce them.
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 3, 5, 8, 12, generator=generator) * 2.2 - 1.1).to(torch.bfloat16)


def _pre_process(od_config):
    from vllm_omni.diffusion.models.pan2.pipeline_pan2 import get_pan2_pre_process_func

    return get_pan2_pre_process_func(od_config)


def _post_process(od_config):
    from vllm_omni.diffusion.models.pan2.pipeline_pan2 import get_pan2_post_process_func

    return get_pan2_post_process_func(od_config)


def test_blocked_prompt_raises_before_generation(make_checker):
    checker = make_checker()
    pre_process = _pre_process(_od_config())

    with pytest.raises(GuardrailViolationError, match="Input was blocked by PAN2 guardrails"):
        pre_process(_request(BLOCKED_PROMPT))
    with pytest.raises(GuardrailViolationError):
        pre_process(_request({"prompt": BLOCKED_PROMPT, "negative_prompt": "blurry"}))
    assert checker.prompts == [BLOCKED_PROMPT, BLOCKED_PROMPT]


def test_safe_prompt_passes_and_negative_prompt_is_not_checked(make_checker):
    checker = make_checker()
    pre_process = _pre_process(_od_config())

    request = _request({"prompt": "a cat walks on the grass", "negative_prompt": BLOCKED_PROMPT})
    assert pre_process(request) is request
    assert checker.prompts == ["a cat walks on the grass"]


def test_blocked_video_raises(make_checker):
    make_checker(block_video=True)
    _pre_process(_od_config())
    post_process = _post_process(_od_config())

    with pytest.raises(GuardrailViolationError, match="generated video was blocked by PAN2 guardrails"):
        post_process(_video(), sampling_params=OmniDiffusionSamplingParams())


def test_passing_video_is_unchanged_and_checked_as_delivered(make_checker):
    checker = make_checker()
    _pre_process(_od_config())
    video = _video()

    frames = _post_process(_od_config())(video, sampling_params=OmniDiffusionSamplingParams())
    expected = _post_process(_od_config(guardrails=False))(video)

    assert len(checker.videos) == 1
    checked = checker.videos[0]
    assert checked.dtype == np.uint8
    assert checked.shape == (5, 8, 12, 3)
    np.testing.assert_array_equal(checked, np.stack([np.asarray(frame) for frame in expected]))
    assert len(frames) == len(expected)
    for frame, expected_frame in zip(frames, expected):
        np.testing.assert_array_equal(np.asarray(frame), np.asarray(expected_frame))


def test_every_video_of_a_batch_is_checked(make_checker, guardrails):
    checker = make_checker()
    _pre_process(_od_config())
    batch = [_video(0), _video(1)]
    videos = _post_process(_od_config())(torch.cat(batch), sampling_params=OmniDiffusionSamplingParams())

    # Each video is checked once, in order, as the frames that are delivered for it.
    assert len(checker.videos) == 2
    for checked, video, delivered in zip(checker.videos, batch, videos):
        expected = guardrails.video_to_uint8_frames(video[0])
        np.testing.assert_array_equal(checked, expected)
        np.testing.assert_array_equal(np.stack([np.asarray(frame) for frame in delivered]), expected)
    assert not np.array_equal(checker.videos[0], checker.videos[1])


def test_latent_output_is_not_checked(make_checker):
    checker = make_checker(block_video=True)
    _pre_process(_od_config())
    latents = torch.zeros(1, 4, 2, 2, 2)
    assert _post_process(_od_config())(latents, output_type="latent") is latents
    assert checker.videos == []


def test_request_output_type_latent_is_returned_unchecked(make_checker):
    checker = make_checker(block_video=True)
    _pre_process(_od_config())
    latents = torch.zeros(1, 4, 2, 2, 2)
    sampling_params = OmniDiffusionSamplingParams(output_type="latent")
    assert _post_process(_od_config())(latents, sampling_params=sampling_params) is latents
    assert checker.videos == []


def test_server_switch_off_loads_and_checks_nothing(guardrails, monkeypatch: pytest.MonkeyPatch):
    def from_pretrained():
        raise AssertionError("the guardrails must not load when they are disabled")

    monkeypatch.setattr(guardrails.PAN2SafetyChecker, "from_pretrained", staticmethod(from_pretrained))
    od_config = _od_config(guardrails=False)

    request = _request(BLOCKED_PROMPT)
    assert _pre_process(od_config)(request) is request
    # A request cannot turn the checks back on when the server did not load the guardrails.
    assert _pre_process(od_config)(_request(BLOCKED_PROMPT, guardrails=True)).prompt == BLOCKED_PROMPT
    _post_process(od_config)(_video(), sampling_params=OmniDiffusionSamplingParams(extra_args={"guardrails": True}))


def test_per_request_override_skips_both_checks(make_checker):
    checker = make_checker(block_video=True)
    od_config = _od_config()
    pre_process = _pre_process(od_config)

    request = _request(BLOCKED_PROMPT, guardrails=False)
    assert pre_process(request) is request
    _post_process(od_config)(_video(), sampling_params=OmniDiffusionSamplingParams(extra_args={"guardrails": False}))
    assert checker.prompts == []
    assert checker.videos == []


def test_guardrails_load_once_on_the_compute_device(make_checker):
    checker = make_checker()
    _pre_process(_od_config())
    _pre_process(_od_config())
    assert checker.devices == [("checker", "compute")]


def test_offloaded_guards_visit_the_compute_device_per_check(make_checker):
    checker = make_checker()
    od_config = _od_config(offload_guardrail_models=True)
    _pre_process(od_config)(_request("a cat"))
    _post_process(od_config)(_video(), sampling_params=OmniDiffusionSamplingParams())
    assert checker.devices == [("text", "compute"), ("text", "cpu"), ("video", "compute"), ("video", "cpu")]


def test_missing_package_fails_only_when_enabled(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.models.pan2 as pan2

    # Re-import the adapter without the package, and restore the original module afterwards.
    monkeypatch.setattr(pan2, "guardrails", getattr(pan2, "guardrails", None), raising=False)
    monkeypatch.setitem(sys.modules, "pan2_guardrail", None)
    monkeypatch.delitem(sys.modules, "vllm_omni.diffusion.models.pan2.guardrails", raising=False)

    request = _request("a cat")
    assert _pre_process(_od_config(guardrails=False))(request) is request
    with pytest.raises(ValueError, match="pan2-guardrail package is not installed"):
        _pre_process(_od_config())
