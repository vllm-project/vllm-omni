# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for OmniVoiceDecoder's fused DAC decoder.

OmniVoiceDecoder.load_weights builds its acoustic decoder through the
shared higgs-audio v2 builder (``build_higgs_audio_acoustic_decoder``),
which swaps the transformers DAC Snake1d modules for the fused
bit-identical Triton kernel and folds conv biases + residual adds into
it. These tests pin the contract:

- the builder output keeps the stock param names (the load_weights copy
  loop and real checkpoints depend on it), applies the HiggsAudioV2
  output-padding fix and tanh -> Identity,
- fused output is bit-identical to the stock transformers decoder with
  the same weights, on CPU (eager fallback) and CUDA (fused kernel),
- the real load_weights path builds the fused decoder and stays
  bit-identical end to end,
- fp16 checkpoint weights still upcast to fp32 (ConvTranspose overflow
  pin survives the shared builder).
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import (
    _fused_dac_decoder_forward,
    _FusedDacSnake1d,
    build_higgs_audio_acoustic_decoder,
)
from vllm_omni.model_executor.models.omnivoice.omnivoice_decoder import OmniVoiceDecoder
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = pytest.mark.core_model

_requires_cuda = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="needs CUDA")

NUM_QUANTIZERS = 8
# Real k2-fsa/OmniVoice audio_tokenizer geometry.
ACOUSTIC_MODEL_CONFIG = {
    "codebook_dim": 8,
    "codebook_size": 1024,
    "decoder_hidden_size": 1024,
    "downsampling_ratios": [8, 5, 4, 2, 3],
    "encoder_hidden_size": 64,
    "hidden_size": 256,
    "hop_length": 960,
    "model_type": "dac",
    "n_codebooks": 9,
    "quantizer_dropout": 0,
    "sampling_rate": 16000,
    "upsampling_ratios": [8, 5, 4, 2, 3],
}


def _tokenizer_config() -> dict:
    return {
        "codebook_dim": 64,
        "codebook_size": 1024,
        "acoustic_model_config": dict(ACOUSTIC_MODEL_CONFIG),
    }


def _stock_decoder(device: str) -> torch.nn.Module:
    """The pre-change omnivoice construction: stock DacModel decoder with the
    HiggsAudioV2 output-padding fix and tanh -> Identity, no fusing."""
    from transformers import DacConfig, DacModel

    decoder = DacModel(DacConfig(**ACOUSTIC_MODEL_CONFIG)).decoder.to(device=device, dtype=torch.float32)
    for module in decoder.modules():
        if isinstance(module, torch.nn.ConvTranspose1d):
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            module.output_padding = (stride % 2,)
    if hasattr(decoder, "tanh"):
        decoder.tanh = torch.nn.Identity()
    return decoder.eval()


def _fused_decoder(device: str) -> torch.nn.Module:
    return build_higgs_audio_acoustic_decoder(_tokenizer_config(), torch.device(device)).eval()


def _wrap(quantizer, fc2, acoustic_decoder) -> OmniVoiceDecoder:
    model = OmniVoiceDecoder(OmniVoiceConfig())
    model.quantizer = quantizer
    model.fc2 = fc2
    model.acoustic_decoder = acoustic_decoder
    model._loaded = True
    return model


def _random_rvq_fc2(device: str, seed: int = 0):
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_decoder import HiggsAudioRVQ

    torch.manual_seed(seed)
    quantizer = HiggsAudioRVQ(num_quantizers=NUM_QUANTIZERS, codebook_size=1024, codebook_dim=64, hidden_size=1024).to(
        device=device, dtype=torch.float32
    )
    fc2 = torch.nn.Linear(1024, 256).to(device=device, dtype=torch.float32)
    return quantizer, fc2


def _codes(t_frames: int, device: str, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1024, (1, NUM_QUANTIZERS, t_frames), generator=gen).to(device)


def _checkpoint_state_dict(stock_decoder: torch.nn.Module, *, fp16: bool = False, scale: float = 1.0) -> dict:
    """Random weights in the k2-fsa/OmniVoice audio_tokenizer layout."""
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(NUM_QUANTIZERS):
        sd[f"quantizer.quantizers.{i}.codebook.embed"] = torch.randn(1024, 64) * 0.1
        sd[f"quantizer.quantizers.{i}.project_out.weight"] = torch.randn(1024, 64) * 0.05
        sd[f"quantizer.quantizers.{i}.project_out.bias"] = torch.randn(1024) * 0.05
    fc2_w = torch.randn(256, 1024) * 0.05
    fc2_b = torch.randn(256) * 0.05
    for name, param in stock_decoder.named_parameters():
        sd[f"acoustic_decoder.{name}"] = param.data.clone() * scale
    if fp16:
        fc2_w, fc2_b = fc2_w.half(), fc2_b.half()
        for k in list(sd):
            if k.startswith("acoustic_decoder."):
                sd[k] = sd[k].half()
    sd["fc2.weight"] = fc2_w
    sd["fc2.bias"] = fc2_b
    return sd


def _write_checkpoint(root, state_dict: dict) -> None:
    token_dir = root / "audio_tokenizer"
    token_dir.mkdir()
    with open(token_dir / "config.json", "w") as f:
        json.dump(_tokenizer_config(), f)
    save_file(state_dict, str(token_dir / "model.safetensors"))


# ---------------------------------------------------------------------------
# Builder contract (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.cpu
def test_builder_keeps_stock_param_names_and_fixes():
    fused = _fused_decoder("cpu")
    stock = _stock_decoder("cpu")

    # Snake modules swapped, fused forward override installed, and the stock
    # layout checks that gate it.
    assert sum(1 for m in fused.modules() if isinstance(m, _FusedDacSnake1d)) > 0
    assert fused.forward.__func__ is _fused_dac_decoder_forward

    # HiggsAudioV2 fixes the builder must apply.
    assert isinstance(fused.tanh, torch.nn.Identity)
    for module in fused.modules():
        if isinstance(module, torch.nn.ConvTranspose1d):
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            assert module.output_padding == (stride % 2,)

    # Param names identical -> load_weights' copy loop and strict loads keep
    # working against real checkpoints.
    fused.load_state_dict(stock.state_dict(), strict=True)


@pytest.mark.cpu
def test_fused_matches_stock_on_cpu_fallback():
    """On CPU the fused modules fall back to the eager Snake expression, so
    the whole decoder must stay bit-identical to the stock construction."""
    stock = _stock_decoder("cpu")
    fused = _fused_decoder("cpu")
    fused.load_state_dict(stock.state_dict(), strict=True)
    for m in fused.modules():
        if isinstance(m, _FusedDacSnake1d):
            assert not m.can_fuse(torch.zeros(1, 4, 8))

    quantizer, fc2 = _random_rvq_fc2("cpu")
    codes = _codes(25, "cpu")
    ref = _wrap(quantizer, fc2, stock)
    got = _wrap(quantizer, fc2, fused)
    assert torch.equal(ref(codes), got(codes))


# ---------------------------------------------------------------------------
# Bit-exactness and the real load path (CUDA)
# ---------------------------------------------------------------------------


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fused_bit_exact_vs_stock_cuda():
    device = "cuda:0"
    stock = _stock_decoder(device)
    fused = _fused_decoder(device)
    fused.load_state_dict(stock.state_dict(), strict=True)

    quantizer, fc2 = _random_rvq_fc2(device)
    ref = _wrap(quantizer, fc2, stock)
    got = _wrap(quantizer, fc2, fused)
    for t_frames in (25, 375):  # small and the default 15 s chunk
        codes = _codes(t_frames, device)
        assert torch.equal(ref(codes), got(codes)), f"T={t_frames} diverged"


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_load_weights_builds_fused_bit_exact_decoder(tmp_path):
    """The real load_weights path (fake checkpoint in the OmniVoice layout)
    must build the fused decoder and decode bit-identically to the stock
    construction loaded with the same weights."""
    device = torch.device("cuda:0")
    sd = _checkpoint_state_dict(_stock_decoder("cpu"))
    _write_checkpoint(tmp_path, sd)

    decoder = OmniVoiceDecoder(OmniVoiceConfig())
    decoder.load_weights(str(tmp_path), device)

    assert sum(1 for m in decoder.acoustic_decoder.modules() if isinstance(m, _FusedDacSnake1d)) > 0
    assert decoder.acoustic_decoder.forward.__func__ is _fused_dac_decoder_forward

    stock = _stock_decoder("cpu").to(device)
    stock.load_state_dict(
        {k[len("acoustic_decoder.") :]: v for k, v in sd.items() if k.startswith("acoustic_decoder.")},
        strict=True,
    )
    ref = _wrap(decoder.quantizer, decoder.fc2, stock)
    codes = _codes(50, "cuda:0")
    assert torch.equal(ref(codes), decoder(codes))


@_requires_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_fp16_checkpoint_upcasts_to_fp32(tmp_path):
    """Checkpoint weights may be fp16; the acoustic decoder must end up fp32
    (ConvTranspose upsampling overflows fp16) and produce finite audio."""
    device = torch.device("cuda:0")
    sd = _checkpoint_state_dict(_stock_decoder("cpu"), fp16=True, scale=8.0)
    _write_checkpoint(tmp_path, sd)

    decoder = OmniVoiceDecoder(OmniVoiceConfig())
    decoder.load_weights(str(tmp_path), device)

    assert all(p.dtype == torch.float32 for p in decoder.acoustic_decoder.parameters())
    out = decoder(_codes(25, "cuda:0"))
    assert torch.isfinite(out).all()
