# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA bit-parity tests for CSM-1B against the HuggingFace reference weights.

These need a GPU and the gated ``sesame/csm-1b`` checkpoint; they skip cleanly
without either (so CI on a GPU-less / un-gated host is green). On a host that
has both they MUST run -- they are the real correctness guarantee behind the
2-stage port:

  * Depth decoder: our incremental, KV-cached 31-step loop
    (``CsmDepthDecoder.run``) must be BIT-EXACT under greedy against an
    independent cacheless full-recompute reference driving the SAME real HF
    ``CsmDepthDecoderForCausalLM`` module. This proves the cache reset, the
    position-0 ``backbone_last_hidden_state`` seeding, and the per-step
    codebook-head indexing are all correct (the "incremental == full recompute"
    equivalence).
  * Mimi vocoder: ``CsmMimiVocoder._mimi_decode`` must be bit-exact against the
    raw ``MimiModel.decode`` for in-range codes (the "Mimi diff 0.0" guarantee),
    and the reserved-id clamp must route 2048/2049/2050 to the clamped decode
    (never feeding the codec an out-of-range id) while still producing finite
    audio.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

_MODEL = "sesame/csm-1b"

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytestmark = [pytest.mark.core_model, cuda]


def _model_is_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(_MODEL, "config.json") is not None
    except Exception:
        return False


requires_weights = pytest.mark.skipif(not _model_is_cached(), reason=f"{_MODEL} not in local HF cache (gated)")


@pytest.fixture(scope="module")
def hf_csm():
    pytest.importorskip("transformers")
    from transformers.models.csm.modeling_csm import CsmForConditionalGeneration

    model = CsmForConditionalGeneration.from_pretrained(_MODEL, torch_dtype=torch.float32)
    model = model.to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# --------------------------------------------------------------------------
# Depth decoder: incremental KV loop == cacheless full recompute (greedy)
# --------------------------------------------------------------------------


@torch.inference_mode()
def _reference_depth_full_recompute(depth, cb0, backbone_hidden, num_codebooks):
    """Independent reference: greedily decode cb1..cb31 by re-feeding the WHOLE
    growing sequence each step with NO KV cache. Shares no code with
    ``CsmDepthDecoder.run`` -- HF seeds the backbone hidden at position 0 only
    when ``past_seen == 0`` (verified in CsmDepthDecoderModel.forward), which is
    always the case here, so this is exactly equivalent to a correct incremental
    decode."""
    codes = [cb0]  # list of (B,) Long
    for _k in range(1, num_codebooks):
        prefix = torch.stack(codes, dim=1)  # (B, k)
        ids = torch.nn.functional.pad(prefix, (1, 0), value=0)  # pos-0 placeholder
        out = depth(
            input_ids=ids,
            backbone_last_hidden_state=backbone_hidden,
            use_cache=False,
            logits_to_keep=1,
        )
        logits = out.logits[:, -1, :].float()
        codes.append(torch.argmax(logits, dim=-1))
    return torch.stack(codes, dim=1)  # (B, num_codebooks)


@torch.inference_mode()
def _real_backbone_hidden(model, text="The quick brown fox."):
    """A real backbone last-hidden-state vector, so depth logits are confident
    (no argmax ties that would make a greedy comparison flaky)."""
    tok = getattr(model.config, "text_vocab_size", 128256)
    # Deterministic short token sequence inside the text vocab.
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda") % tok
    embeds = model.embed_text_tokens(ids)
    bb = model.backbone_model(inputs_embeds=embeds)
    h = bb.last_hidden_state[:, -1, :]  # (1, hidden)
    return h.to(torch.float32)


@requires_weights
def test_depth_incremental_matches_full_recompute_bitexact(hf_csm):
    from vllm_omni.model_executor.models.csm.csm_depth import CsmDepthDecoder

    num_codebooks = int(hf_csm.config.num_codebooks)
    hidden_size = int(hf_csm.config.hidden_size)
    h_t = _real_backbone_hidden(hf_csm)

    wrapper = CsmDepthDecoder(num_codebooks=num_codebooks, hidden_size=hidden_size, aux_dtype=torch.float32)
    wrapper.set_module(hf_csm.depth_decoder)

    for cb0_val in (0, 100, 1000, 2047):
        cb0 = torch.tensor([cb0_val], device="cuda", dtype=torch.long)
        ours = wrapper.run(cb0=cb0, backbone_last_hidden_state=h_t, temperature=0.0, top_k=0)
        ref = _reference_depth_full_recompute(hf_csm.depth_decoder, cb0, h_t, num_codebooks)
        assert ours.shape == (1, num_codebooks)
        assert int(ours[0, 0]) == cb0_val  # cb0 passthrough
        assert torch.equal(ours, ref), f"depth mismatch at cb0={cb0_val}: {ours} vs {ref}"


@requires_weights
def test_depth_run_is_deterministic_greedy(hf_csm):
    from vllm_omni.model_executor.models.csm.csm_depth import CsmDepthDecoder

    h_t = _real_backbone_hidden(hf_csm)
    wrapper = CsmDepthDecoder(
        num_codebooks=int(hf_csm.config.num_codebooks),
        hidden_size=int(hf_csm.config.hidden_size),
        aux_dtype=torch.float32,
    )
    wrapper.set_module(hf_csm.depth_decoder)
    cb0 = torch.tensor([100], device="cuda", dtype=torch.long)
    a = wrapper.run(cb0=cb0, backbone_last_hidden_state=h_t, temperature=0.0, top_k=0)
    b = wrapper.run(cb0=cb0, backbone_last_hidden_state=h_t, temperature=0.0, top_k=0)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# Mimi vocoder: wrapper decode == raw MimiModel.decode + reserved-id clamp
# --------------------------------------------------------------------------


def _make_vocoder(codec, num_codebooks, sample_rate):
    from vllm_omni.model_executor.models.csm.csm_mimi import CsmMimiVocoder

    v = object.__new__(CsmMimiVocoder)
    nn.Module.__init__(v)
    v.num_codebooks = num_codebooks
    v.sample_rate = sample_rate
    v._device = torch.device("cuda")
    v.config = SimpleNamespace(codec_samples_per_frame=1920)
    v._stream_state_by_req = {}
    v._mimi_codec = codec
    return v


@torch.inference_mode()
def _raw_mimi_decode(codec, codes_qf):
    out = codec.decode(codes_qf.unsqueeze(0))
    av = out.audio_values
    return av.reshape(av.shape[0], -1)[0].to(torch.float32)


@requires_weights
def test_mimi_decode_bitexact_vs_raw_for_in_range_codes(hf_csm):
    codec = hf_csm.codec_model
    num_codebooks = int(hf_csm.config.num_codebooks)
    sr = int(getattr(hf_csm.config, "codec_sample_rate", 24000) or 24000)
    v = _make_vocoder(codec, num_codebooks, sr)

    g = torch.Generator(device="cuda").manual_seed(0)
    for _ in range(8):  # the "diff 0.0 across 8 frames" guarantee
        frames = int(torch.randint(1, 6, (1,), generator=g, device="cuda").item())
        codes = torch.randint(0, 2048, (num_codebooks, frames), generator=g, device="cuda", dtype=torch.long)
        ours = v._mimi_decode(codes)
        ref = _raw_mimi_decode(codec, codes)
        assert torch.equal(ours, ref)


@requires_weights
def test_mimi_decode_clamps_reserved_ids_to_codec_range(hf_csm):
    codec = hf_csm.codec_model
    num_codebooks = int(hf_csm.config.num_codebooks)
    sr = int(getattr(hf_csm.config, "codec_sample_rate", 24000) or 24000)
    v = _make_vocoder(codec, num_codebooks, sr)

    codes = torch.randint(0, 2048, (num_codebooks, 3), device="cuda", dtype=torch.long)
    codes[0, 0] = 2048
    codes[1, 1] = 2049
    codes[2, 2] = 2050
    clamped = codes.clamp(min=0, max=2047)

    ours = v._mimi_decode(codes)
    ref = _raw_mimi_decode(codec, clamped)
    # Reserved ids must be clamped before reaching the codec -> identical to the
    # clamped decode, and the caller's tensor is not mutated.
    assert torch.equal(ours, ref)
    assert torch.isfinite(ours).all()
    assert int(codes[0, 0]) == 2048  # original tensor untouched (clamp on a copy)
