# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for where the LTX-2 vocoder and its BWE filter are built."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.ltx2 import ltx2_components, ltx2_runtime
from vllm_omni.diffusion.models.ltx2.ltx2_components import LTX23_COMPONENT_PROFILE, _load_ltx_vocoder
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import _restore_ltx_bwe_resampler_filter
from vllm_omni.platforms.interface import OmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _set_signal_window_ops(monkeypatch, supported: bool) -> None:
    # Both modules bind ``current_omni_platform`` at import time, so replace
    # that module attribute with a platform stub reporting ``supported``.
    platform = SimpleNamespace(supports_signal_window_ops=lambda: supported)
    monkeypatch.setattr(ltx2_components, "current_omni_platform", platform)
    monkeypatch.setattr(ltx2_runtime, "current_omni_platform", platform)


def test_platform_supports_signal_window_ops_by_default():
    assert OmniPlatform.supports_signal_window_ops() is True


class _PrimaryVocoder:
    pass


class _FallbackVocoder:
    pass


@pytest.mark.parametrize("primary_fails", [False, True])
@pytest.mark.parametrize(("supported", "expected_device"), [(False, "cpu"), (True, "meta")])
def test_ltx_vocoder_construction_device_follows_platform_flag(monkeypatch, primary_fails, supported, expected_device):
    _set_signal_window_ops(monkeypatch, supported)
    profile = replace(
        LTX23_COMPONENT_PROFILE,
        vocoder_cls=_PrimaryVocoder,
        vocoder_fallback_cls=_FallbackVocoder,
    )
    construction_devices = []

    def fake_load_component(component_cls, model, subfolder, **kwargs):
        # A tensor created without an explicit device lands on the active
        # default device, which is where Diffusers builds the vocoder filters.
        construction_devices.append((component_cls, subfolder, torch.empty(1).device))
        if primary_fails and component_cls is _PrimaryVocoder:
            raise OSError("no BWE vocoder in this checkpoint")
        return component_cls()

    monkeypatch.setattr(ltx2_components, "_load_component", fake_load_component)

    # The pipeline loader builds components under an accelerator default
    # device; "meta" stands in for it on CPU-only runners.
    with torch.device("meta"):
        vocoder = _load_ltx_vocoder(
            profile,
            "unused-model",
            local_files_only=True,
            dtype=torch.bfloat16,
            revision=None,
        )

    expected_cls = _FallbackVocoder if primary_fails else _PrimaryVocoder
    assert isinstance(vocoder, expected_cls)
    assert construction_devices[-1][:2] == (expected_cls, "vocoder")
    assert len(construction_devices) == (2 if primary_fails else 1)
    # Flag False forces CPU; flag True keeps the ambient default device.
    assert [device.type for _, _, device in construction_devices] == [expected_device] * len(construction_devices)


def _make_bwe_vocoder(filter_value: torch.Tensor, ratio: int, kernel_size: int) -> torch.nn.Module:
    resampler = torch.nn.Module()
    resampler.ratio = ratio
    resampler.kernel_size = kernel_size
    resampler.register_buffer("filter", filter_value, persistent=False)
    vocoder = torch.nn.Module()
    vocoder.resampler = resampler
    return vocoder


def _reference_hann_filter(kernel_size: int, ratio: int) -> torch.Tensor:
    # The formula previously evaluated on the filter's device.
    rolloff = 0.99
    lowpass_filter_width = 6
    width = math.ceil(lowpass_filter_width / rolloff)
    time_axis = (torch.arange(kernel_size, dtype=torch.float32) / ratio - width) * rolloff
    time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2).square()
    return (torch.sinc(time_axis) * window * rolloff / ratio).reshape(1, 1, -1)


_RATIO = 2
_KERNEL_SIZE = 2 * math.ceil(6 / 0.99) * _RATIO + 1


@pytest.mark.parametrize("supported", [False, True])
def test_ltx_bwe_filter_restore_matches_fp32_reference(monkeypatch, supported):
    _set_signal_window_ops(monkeypatch, supported)
    vocoder = _make_bwe_vocoder(torch.zeros((1, 1, _KERNEL_SIZE), dtype=torch.bfloat16), _RATIO, _KERNEL_SIZE)

    _restore_ltx_bwe_resampler_filter(vocoder)

    restored = vocoder.resampler.filter
    assert restored.dtype == torch.float32
    assert torch.equal(restored, _reference_hann_filter(_KERNEL_SIZE, _RATIO))
    assert restored is vocoder.resampler._ltx_restored_filter


@pytest.mark.parametrize(
    ("supported", "expected_sinc_device", "expected_to_calls"),
    [(False, "cpu", ["meta"]), (True, "meta", [])],
)
def test_ltx_bwe_filter_restore_device_follows_platform_flag(
    monkeypatch, supported, expected_sinc_device, expected_to_calls
):
    _set_signal_window_ops(monkeypatch, supported)
    # "meta" stands in for an accelerator; create the buffer before recording.
    vocoder = _make_bwe_vocoder(
        torch.zeros((1, 1, _KERNEL_SIZE), device="meta", dtype=torch.bfloat16),
        _RATIO,
        _KERNEL_SIZE,
    )
    sinc_devices = []
    to_calls = []
    original_sinc = torch.sinc
    original_to = torch.Tensor.to

    def recording_sinc(value):
        sinc_devices.append(value.device)
        return original_sinc(value)

    def recording_to(self, *args, **kwargs):
        to_calls.append(str(args[0]) if args else str(kwargs))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch, "sinc", recording_sinc)
    monkeypatch.setattr(torch.Tensor, "to", recording_to)

    _restore_ltx_bwe_resampler_filter(vocoder)

    restored = vocoder.resampler.filter
    assert [device.type for device in sinc_devices] == [expected_sinc_device]
    assert to_calls == expected_to_calls
    assert restored.device.type == "meta"
    assert restored.dtype == torch.float32
    assert tuple(restored.shape) == (1, 1, _KERNEL_SIZE)
    assert restored is vocoder.resampler._ltx_restored_filter

    _restore_ltx_bwe_resampler_filter(vocoder)

    assert len(sinc_devices) == 1
    assert to_calls == expected_to_calls
    assert vocoder.resampler.filter is restored
