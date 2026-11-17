# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU residency of the streaming HiFT vocoder state.

``_stream_hift_from_feat`` keeps each request's mel window and phase
accumulator on the HiFT device between chunks. These tests pin that property
and prove it is bit-exact against the previous behavior (state round-tripped
through host memory between chunks), on the SineGen2 config CosyVoice3 ships
and the legacy SineGen config.
"""

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import (
    CausalConvRNNF0Predictor,
    CausalHiFTGenerator,
    SineGen,
    SineGen2,
)
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]

WINDOW_LEN = 64
# First chunk sized like a prompt prefill, then fixed steps; history crosses
# the bounded window (40 + 24*3 = 112 > 64) so the overlap path runs.
FIRST_LEN = 40
STEP_LEN = 24
TOTAL_MEL = 112

REAL_CONFIG = {"sampling_rate": 24000, "nb_harmonics": 8}
LEGACY_CONFIG = {"sampling_rate": 22050, "nb_harmonics": 4}

CONFIGS = [
    pytest.param(REAL_CONFIG, id="sinegen2_real_config"),
    pytest.param(LEGACY_CONFIG, id="sinegen1_legacy_config"),
]


def _make_hift(config: dict) -> CausalHiFTGenerator:
    torch.manual_seed(0)
    f0_predictor = CausalConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=16)
    return CausalHiFTGenerator(
        in_channels=80,
        base_channels=32,
        nb_harmonics=config["nb_harmonics"],
        sampling_rate=config["sampling_rate"],
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=f0_predictor,
    ).eval()


def _make_model(hift: CausalHiFTGenerator) -> CosyVoice3Code2Wav:
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model.hift = hift
    model._hift_window_len = WINDOW_LEN
    return model


def _chunks() -> list[torch.Tensor]:
    torch.manual_seed(1)
    mel = torch.randn(1, 80, TOTAL_MEL)
    return [mel[:, :, :FIRST_LEN], *(mel[:, :, p : p + STEP_LEN] for p in range(FIRST_LEN, TOTAL_MEL, STEP_LEN))]


def _to_host(cache: dict[str, torch.Tensor | int] | None) -> dict[str, torch.Tensor | int] | None:
    """Round-trip a cache through host memory, as the pre-change code did."""
    if cache is None:
        return None
    return {key: value.cpu().contiguous() if isinstance(value, torch.Tensor) else value for key, value in cache.items()}


def _stream(model: CosyVoice3Code2Wav, chunks: list[torch.Tensor], *, host_cache: bool):
    torch.manual_seed(0)
    emitted: list[torch.Tensor] = []
    cache = None
    for chunk in chunks:
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        if host_cache:
            cache = _to_host(cache)
        emitted.append(speech.reshape(speech.shape[0], -1))
    speech, cache = model._stream_hift_from_feat(
        torch.zeros(1, 80, 0, device=chunks[0].device), cache_state=cache, finalize=True
    )
    emitted.append(speech.reshape(speech.shape[0], -1))
    return emitted, cache


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("config", CONFIGS)
def test_streaming_state_stays_on_hift_device(config: dict) -> None:
    hift = _make_hift(config)
    assert isinstance(hift.m_source.l_sin_gen, SineGen2 if config is REAL_CONFIG else SineGen)
    # Resolve with an explicit index: tensors report ``cuda:0``, and
    # ``torch.device("cuda") != torch.device("cuda:0")`` under ``==``.
    device = torch.device("cuda", torch.accelerator.current_device_index())
    model = _make_model(hift.to(device))

    cache = None
    for chunk in (c.to(device) for c in _chunks()):
        _, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        assert isinstance(cache, dict)
        assert cache["mel"].device == device, "mel history must stay on the HiFT device"
        assert isinstance(cache["mel"], torch.Tensor) and cache["mel"].is_contiguous()
        if cache["phase_acc"] is not None:
            assert cache["phase_acc"].device == device, "phase accumulator must stay on the HiFT device"
    _, cache = model._stream_hift_from_feat(torch.zeros(1, 80, 0, device=device), cache_state=cache, finalize=True)
    assert cache is None, "finalize must drop the streaming cache"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("config", CONFIGS)
def test_gpu_resident_state_matches_host_round_trip_bit_exactly(config: dict) -> None:
    """Device-resident cache must reproduce the host round-trip bit-for-bit."""
    # Resolve with an explicit index: tensors report ``cuda:0``, and
    # ``torch.device("cuda") != torch.device("cuda:0")`` under ``==``.
    device = torch.device("cuda", torch.accelerator.current_device_index())
    model = _make_model(_make_hift(config).to(device))
    chunks = [c.to(device) for c in _chunks()]

    resident, resident_cache = _stream(model, chunks, host_cache=False)
    host, _ = _stream(model, chunks, host_cache=True)

    assert len(resident) == len(host)
    for index, (a, b) in enumerate(zip(resident, host, strict=True)):
        assert a.device == b.device == device
        assert torch.equal(a, b), f"chunk {index} diverged between device-resident and host cache"
    # The cache the finalize pass would have consumed is also identical.
    assert resident_cache is None


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("config", CONFIGS)
def test_mixed_residency_caches_are_equivalent(config: dict) -> None:
    """A cache created device-resident then bounced to host must still work."""
    # Resolve with an explicit index: tensors report ``cuda:0``, and
    # ``torch.device("cuda") != torch.device("cuda:0")`` under ``==``.
    device = torch.device("cuda", torch.accelerator.current_device_index())
    model = _make_model(_make_hift(config).to(device))
    chunks = [c.to(device) for c in _chunks()]

    torch.manual_seed(0)
    emitted, cache = [], None
    for index, chunk in enumerate(chunks):
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        if index == 0:
            cache = _to_host(cache)  # old-format cache mid-stream
        emitted.append(speech.reshape(speech.shape[0], -1))
    mixed = torch.cat(emitted, dim=-1)

    torch.manual_seed(0)
    emitted, cache = [], None
    for chunk in chunks:
        speech, cache = model._stream_hift_from_feat(chunk, cache_state=cache, finalize=False)
        emitted.append(speech.reshape(speech.shape[0], -1))
    resident = torch.cat(emitted, dim=-1)

    assert torch.equal(mixed, resident)
