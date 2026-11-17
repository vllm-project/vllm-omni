# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Qwen2.5 token2wav vocoder CUDA-graph wiring.

Covers the config-flag plumbing (default off), the ``_vocode`` routing
(graph wrapper vs raw module), the derived streaming capture sizes, and the
``amplitude_to_db`` min-level cache that unblocks CUDA-graph capture. No
checkpoints or GPU required.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_tiny_vocoder_cpu():
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
        Qwen2_5OmniBigVGANConfig,
    )

    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        Qwen2_5OmniToken2WavBigVGANModel,
    )

    config = Qwen2_5OmniBigVGANConfig(
        upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4),
        upsample_initial_channel=64,
        mel_dim=80,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
    )
    torch.manual_seed(11)
    return Qwen2_5OmniToken2WavBigVGANModel(config).eval()


def _make_token2wav_stub(vocoder, config=None):
    """A real-method ``Qwen2_5OmniToken2WavModel`` shell without running its
    heavy ``__init__`` (DiT construction); only the vocoder-routing attrs."""
    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        Qwen2_5OmniToken2WavModel,
    )

    stub = object.__new__(Qwen2_5OmniToken2WavModel)
    torch.nn.Module.__init__(stub)  # enough for attribute assignment; no HF init
    stub.code2wav_bigvgan_model = vocoder
    stub._bigvgan_graph = None
    stub.config = config if config is not None else SimpleNamespace()
    return stub


def _wiring_stub(vocoder, **config_attrs):
    return SimpleNamespace(
        config=SimpleNamespace(**config_attrs),
        code2wav_bigvgan_model=vocoder,
        chunk_size=72,
        future_size=20,
        vocoder_hop=240,
    )


# ---------------------------------------------------------------------------
# amplitude_to_db min-level cache
# ---------------------------------------------------------------------------


def test_amplitude_to_db_min_level_cached():
    """The min-level tensor is built once per (device, dtype, level) instead
    of a pageable H2D per vocode (a CUDA-graph capture blocker); the value
    is unchanged and the cache is a plain attribute, not state."""
    model = _make_tiny_vocoder_cpu()
    x = torch.rand(1, 80, 33) + 0.5

    a1 = model.amplitude_to_db(x, -115)
    a2 = model.amplitude_to_db(x, -115)
    assert torch.equal(a1, a2)
    assert len(model._min_level_cache) == 1

    fresh = 20 * torch.log10(torch.clamp(x, min=torch.exp(torch.tensor(-115 / 20.0 * np.log(10)))))
    torch.testing.assert_close(a1, fresh, rtol=0.0, atol=0.0)

    # A different level (or dtype/device) gets its own entry.
    model.amplitude_to_db(x, -100)
    assert len(model._min_level_cache) == 2
    model.amplitude_to_db(x.double(), -115)
    assert len(model._min_level_cache) == 3

    assert not any("min_level" in key for key in model.state_dict())


# ---------------------------------------------------------------------------
# capture-size derivation and flag plumbing
# ---------------------------------------------------------------------------


def test_vocoder_graph_capture_sizes_derivation():
    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        _vocoder_graph_capture_sizes,
    )

    # Streaming widths are exact hits: first chunk = chunk_size, middle
    # chunk = chunk_size + 2*future_size (past/future cache slices cancel).
    assert _vocoder_graph_capture_sizes(72, 20, [128, 256]) == [72, 112, 128, 256]
    # Deduplicated and sorted when a ladder entry coincides.
    assert _vocoder_graph_capture_sizes(72, 28, [72, 256]) == [72, 128, 256]
    assert _vocoder_graph_capture_sizes(64, 20, []) == [64, 104]


def test_wiring_flag_default_off():
    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        _maybe_enable_vocoder_cuda_graph,
    )

    stub = _wiring_stub(_make_tiny_vocoder_cpu())
    _maybe_enable_vocoder_cuda_graph(stub)
    assert stub._bigvgan_graph is None


def test_wiring_flag_on_builds_wrapper():
    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
        _maybe_enable_vocoder_cuda_graph,
    )

    vocoder = _make_tiny_vocoder_cpu()
    stub = _wiring_stub(vocoder, vocoder_cuda_graph=True)
    _maybe_enable_vocoder_cuda_graph(stub)
    wrapper = stub._bigvgan_graph
    assert wrapper is not None
    assert wrapper.capture_sizes == [72, 112, 128, 256, 512, 1024, 2048]
    assert wrapper.upsample_factor == 240
    assert not wrapper._warmed_up  # capture is lazy: no CUDA work at wiring time

    # Explicit ladder from config wins over the default.
    stub2 = _wiring_stub(vocoder, vocoder_cuda_graph=True, vocoder_cuda_graph_capture_sizes=[64])
    _maybe_enable_vocoder_cuda_graph(stub2)
    assert stub2._bigvgan_graph.capture_sizes == [64, 72, 112]


def test_vocate_passthrough_matches_forward():
    """With the graph disabled (the default), ``_vocode`` must be exactly
    today's ``forward`` contract: [T] CPU waveform."""
    vocoder = _make_tiny_vocoder_cpu()
    stub = _make_token2wav_stub(vocoder)
    torch.manual_seed(1)
    mel = torch.randn(1, 80, 40) - 5.0

    with torch.no_grad():
        via_vocate = stub._vocode(mel)
        via_forward = vocoder(mel)
    torch.testing.assert_close(via_vocate, via_forward, rtol=0.0, atol=0.0)


def test_vocate_routes_through_wrapper_when_present():
    """With a wrapper attached, ``_vocode`` returns the wrapper's GPU output
    squeezed and moved to CPU (today's forward contract)."""
    from vllm_omni.model_executor.models.common.ops.bigvgan_cuda_graph import (
        BigVGANCUDAGraphWrapper,
    )

    vocoder = _make_tiny_vocoder_cpu()
    wrapper = BigVGANCUDAGraphWrapper(
        vocoder,
        capture_fn=vocoder.vocode,
        num_mels=80,
        upsample_factor=4,
        capture_sizes=[40],
    )
    stub = _make_token2wav_stub(vocoder)
    stub._bigvgan_graph = wrapper
    torch.manual_seed(2)
    mel = torch.randn(1, 80, 40) - 5.0

    with torch.no_grad():
        out = stub._vocode(mel)
        eager = vocoder(mel)
    assert out.device.type == "cpu"
    torch.testing.assert_close(out, eager, rtol=0.0, atol=0.0)
