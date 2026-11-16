# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MOSS-TTS-Realtime fused code-predictor loading.

The MOSS-TTS-Realtime depth transformer shares ``CodePredictorBaseModel``, whose
q/k/v and gate/up projections are fused. Before the fix, the talker loaded those
shards with a per-tensor ``default_weight_loader``, which silently dropped every
q/k/v/gate/up shard and left the fused params randomly initialized -- no
exception, just a corrupted model.

These tests assert the property that failure mode violates: after
``load_weights``, every fused parameter is both reported loaded and numerically
equal to the concatenation of the shards that fed it.
"""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.configuration_moss_tts import (
    MossTTSLocalTransformerConfig,
)
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local import (
    MossTTSRealtimeLocalTransformer,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_cfg():
    return MossTTSLocalTransformerConfig(
        head_dim=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=8,
    )


def _hf_shards(cfg, *, drop: set[str] | None = None):
    """Synthetic HF-style checkpoint tensors for the depth transformer."""
    drop = drop or set()
    h, inter = cfg.hidden_size, cfg.intermediate_size
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    out = {}
    for layer in range(cfg.num_hidden_layers):
        p = f"model.layers.{layer}"
        for nm, rows in (("q_proj", q_out), ("k_proj", kv_out), ("v_proj", kv_out)):
            key = f"{p}.self_attn.{nm}.weight"
            if nm not in drop:
                out[key] = torch.randn(rows, h)
        out[f"{p}.self_attn.o_proj.weight"] = torch.randn(h, q_out)
        for nm in ("gate_proj", "up_proj"):
            key = f"{p}.mlp.{nm}.weight"
            if nm not in drop:
                out[key] = torch.randn(inter, h)
        out[f"{p}.mlp.down_proj.weight"] = torch.randn(h, inter)
        out[f"{p}.input_layernorm.weight"] = torch.randn(h)
        out[f"{p}.post_attention_layernorm.weight"] = torch.randn(h)
        for nm in ("q_norm", "k_norm"):
            out[f"{p}.self_attn.{nm}.weight"] = torch.randn(cfg.head_dim)
    out["model.norm.weight"] = torch.randn(h)
    return out


def test_every_fused_param_is_loaded_and_correctly_packed():
    cfg = _tiny_cfg()
    lt = MossTTSRealtimeLocalTransformer(cfg)
    shards = _hf_shards(cfg)

    loaded = lt.load_weights(iter([(k, v) for k, v in shards.items()]))

    params = dict(lt.named_parameters())
    fused = [n for n in params if ".self_attn.qkv_proj." in n or ".mlp.gate_up_proj." in n]
    assert fused, "model declares no fused params -- test would be vacuous"

    # 1. Reported as loaded (this is what the old path silently failed).
    unreported = [n for n in fused if n not in loaded]
    assert not unreported, f"fused params not reported loaded: {unreported}"

    # 2. Numerically equal to the concatenation of their shards, in order.
    for layer in range(cfg.num_hidden_layers):
        p = f"model.layers.{layer}"
        want_qkv = torch.cat([shards[f"{p}.self_attn.{n}.weight"] for n in ("q_proj", "k_proj", "v_proj")], dim=0)
        got_qkv = params[f"{p}.self_attn.qkv_proj.weight"].detach()
        assert torch.equal(got_qkv, want_qkv), f"layer {layer}: qkv packing mismatch"

        want_gu = torch.cat([shards[f"{p}.mlp.{n}.weight"] for n in ("gate_proj", "up_proj")], dim=0)
        got_gu = params[f"{p}.mlp.gate_up_proj.weight"].detach()
        assert torch.equal(got_gu, want_gu), f"layer {layer}: gate_up packing mismatch"


def test_incomplete_shard_group_raises_instead_of_corrupting():
    """A checkpoint missing v_proj must fail loudly, not load a random qkv."""
    cfg = _tiny_cfg()
    lt = MossTTSRealtimeLocalTransformer(cfg)
    shards = _hf_shards(cfg, drop={"v_proj"})

    with pytest.raises(RuntimeError, match="incomplete fused shards"):
        lt.load_weights(iter([(k, v) for k, v in shards.items()]))


def test_absent_fused_group_raises():
    """No q/k/v at all must also raise, via the missing-fused-param guard."""
    cfg = _tiny_cfg()
    lt = MossTTSRealtimeLocalTransformer(cfg)
    shards = _hf_shards(cfg, drop={"q_proj", "k_proj", "v_proj"})

    with pytest.raises(RuntimeError, match="missing fused parameters"):
        lt.load_weights(iter([(k, v) for k, v in shards.items()]))


def test_returned_names_keep_the_model_prefix():
    """Returned names must be ``model.*`` for talker re-prefixing."""
    cfg = _tiny_cfg()
    lt = MossTTSRealtimeLocalTransformer(cfg)
    loaded = lt.load_weights(iter([(k, v) for k, v in _hf_shards(cfg).items()]))
    assert loaded, "nothing reported loaded"
    assert all(n.startswith("model.") for n in loaded), sorted(n for n in loaded if not n.startswith("model."))
