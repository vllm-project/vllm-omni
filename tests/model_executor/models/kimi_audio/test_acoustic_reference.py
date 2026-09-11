# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official reduced-network oracle; CPU SDPA substitutes only FlashAttention.

The fixture contains official parameter layouts and outputs from the pinned
upstream implementation, not pretrained weights or evidence of GPU parity.
"""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

from vllm_omni.model_executor.models.kimi_audio.detokenizer import PrefixStreamingFlowMatchingDetokenizer
from vllm_omni.model_executor.models.kimi_audio.detokenizer.bigvgan_wrapper import BigVGANWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
FIXTURES = Path(__file__).parent / "fixtures"


def cpu_flash_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
    """Evaluate the official noncausal attention math on each packed sequence."""
    outputs = []
    for i in range(len(cu_seqlens_q) - 1):
        qi = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].transpose(0, 1).unsqueeze(0)
        ki = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1).unsqueeze(0)
        vi = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].transpose(0, 1).unsqueeze(0)
        outputs.append(F.scaled_dot_product_attention(qi, ki, vi).squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


@pytest.fixture
def acoustic_files(tmp_path):
    reference = json.loads((FIXTURES / "acoustic_reference.json").read_text(encoding="utf-8"))
    tensors = load_file(str(FIXTURES / "acoustic_reference.safetensors"))
    fm_dir, vocoder_dir = tmp_path / "audio_detokenizer", tmp_path / "vocoder"
    fm_dir.mkdir()
    vocoder_dir.mkdir()
    (fm_dir / "config.yaml").write_text(yaml.safe_dump(reference["fm_config"]), encoding="utf-8")
    (vocoder_dir / "config.json").write_text(json.dumps(reference["vocoder_config"]), encoding="utf-8")
    torch.save({"state_dict": {k[3:]: v for k, v in tensors.items() if k.startswith("fm.")}}, fm_dir / "model.pt")
    torch.save(
        {"generator": {k[8:]: v for k, v in tensors.items() if k.startswith("vocoder.")}}, vocoder_dir / "model.pt"
    )
    return reference, tensors, fm_dir, vocoder_dir


def test_internal_bigvgan_restores_official_checkpoint(acoustic_files):
    _, tensors, _, vocoder_dir = acoustic_files
    wrapper = BigVGANWrapper.from_pretrained(vocoder_dir / "config.json", vocoder_dir / "model.pt", "cpu")
    with torch.inference_mode():
        actual = wrapper.decode_mel(tensors["mel_input"])
    torch.testing.assert_close(actual, tensors["vocoder_output"], rtol=2e-5, atol=2e-6)


def test_internal_streaming_acoustics_match_official(acoustic_files, monkeypatch):
    pytest.importorskip("torchdyn")
    pytest.importorskip("timm")
    reference, tensors, fm_dir, vocoder_dir = acoustic_files
    flash = ModuleType("flash_attn")
    flash.flash_attn_varlen_func = cpu_flash_attention
    flash.flash_attn_varlen_qkvpacked_func = None  # Not used by the streaming path.
    monkeypatch.setitem(sys.modules, "flash_attn", flash)
    decoder = PrefixStreamingFlowMatchingDetokenizer.from_pretrained(
        fm_config=fm_dir / "config.yaml",
        fm_ckpt=fm_dir / "model.pt",
        vocoder_config=vocoder_dir / "config.json",
        vocoder_ckpt=vocoder_dir / "model.pt",
        device="cpu",
        max_kv_cache_tokens=reference["max_kv_cache_tokens"],
        look_ahead_tokens=reference["look_ahead_tokens"],
        use_cfg=False,
    )
    mels = []
    decode_mel = decoder.vocoder.decode_mel

    def record_mel(mel):
        mels.append(mel.clone())
        return decode_mel(mel)

    monkeypatch.setattr(decoder.vocoder, "decode_mel", record_mel)
    for case in reference["cases"]:
        decoder.clear_states()
        decoder.max_pos_size = case["max_pos_size"]
        torch.manual_seed(case["seed"])
        for i, codes in enumerate(case["chunks"]):
            actual = decoder.detokenize_streaming(
                torch.tensor([codes]),
                ode_step=reference["ode_steps"],
                upsample_factor=4,
                is_final=i == len(case["chunks"]) - 1,
            )
            expected = tensors[f"{case['name']}.wave.{i}"]
            # The migrated FM layers, solver and prefix handling must remain
            # exact under the same CPU attention kernel, before the vocoder.
            torch.testing.assert_close(mels[-1], tensors[f"{case['name']}.mel.{i}"], rtol=0, atol=0)
            # bf16 common activations cache reciprocals, unlike upstream's
            # eager division. FP32 vocoder parity is checked separately above.
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
            assert decoder.semantic_fm.start_position_id == case["positions"][i]
            assert decoder.semantic_fm.ode_wrapper.kv_cache_tokens == case["cache_lengths"][i]
        assert decoder.previous_chunk_left is decoder.pre_mel is decoder.pre_wav is None
        assert decoder.semantic_fm.ode_wrapper.incremental_state == {}
        assert decoder.semantic_fm.ode_wrapper.x_cond is None
