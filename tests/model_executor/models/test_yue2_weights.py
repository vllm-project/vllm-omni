# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint name partitioning for YuE2's single-file MoT safetensors.

The checkpoint interleaves AR-path, NAR-path and projection tensors under one
namespace (``model.layers.N.*`` and ``model.layers.N.nar_*`` plus top-level
``vae2llm``/``llm2vae``/``time_embedder``). ``load_weights`` routes them to
either the vLLM backbone loader or the hand-loaded side modules, remapping
``nar_*`` names onto the ``nar_layers`` layout. These tests pin that routing
with the checkpoint's real tensor names (28 layers x 22 tensors + 12 top
level, per bd90e4c).

A routing mistake fails loud (strict load) or silent (wrong tensor lands in
the wrong module): the strict-load path only guards the side half, so the
AR/side split and the name rewrite are asserted directly here.
"""

import pytest
import torch

from vllm_omni.model_executor.models.yue2.weights import partition_checkpoint_weights

# The 12 AR-path names of one checkpoint layer (no nar_ prefix).
AR_LAYER_NAMES = [
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
# The 10 NAR-path names of one layer; all carry the nar_ prefix.
NAR_LAYER_NAMES = [
    "nar_input_layernorm",
    "nar_self_attn.q_proj",
    "nar_self_attn.k_proj",
    "nar_self_attn.v_proj",
    "nar_self_attn.o_proj",
    "nar_self_attn.q_norm",
    "nar_self_attn.k_norm",
    "nar_pre_mlp_layernorm",
    "nar_mlp.gate_proj",
    "nar_mlp.up_proj",
]
TOP_AR_NAMES = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
TOP_SIDE_NAMES = ["vae2llm.weight", "llm2vae.weight", "time_embedder.mlp.0.weight"]

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _t():
    return torch.zeros(1)


def test_ar_layer_tensors_stay_on_the_backbone_path():
    names = [f"model.layers.0.{n}.weight" for n in AR_LAYER_NAMES]
    ar, side = partition_checkpoint_weights([(n, _t()) for n in names])
    assert sorted(n for n, _ in ar) == sorted(names)
    assert side == []


def test_nar_layer_tensors_are_routed_and_remapped():
    for nar_name, module in [
        ("nar_input_layernorm", "input_layernorm"),
        ("nar_pre_mlp_layernorm", "pre_mlp_layernorm"),
        ("nar_self_attn.q_proj", "self_attn.q_proj"),
        ("nar_self_attn.k_norm", "self_attn.k_norm"),
        ("nar_mlp.gate_proj", "mlp.gate_proj"),
    ]:
        ar, side = partition_checkpoint_weights([(f"model.layers.3.{nar_name}.weight", _t())])
        assert ar == []
        assert side == [(f"nar_layers.3.{module}.weight", side[0][1])]


def test_two_digit_layer_indices_partition_correctly():
    ar, side = partition_checkpoint_weights(
        [
            ("model.layers.27.self_attn.q_proj.weight", _t()),
            ("model.layers.27.nar_self_attn.q_proj.weight", _t()),
        ]
    )
    assert [n for n, _ in ar] == ["model.layers.27.self_attn.q_proj.weight"]
    assert [n for n, _ in side] == ["nar_layers.27.self_attn.q_proj.weight"]


def test_top_level_projection_heads_route_to_side_unchanged():
    ar, side = partition_checkpoint_weights([(n, _t()) for n in TOP_SIDE_NAMES])
    assert ar == []
    assert sorted(n for n, _ in side) == sorted(TOP_SIDE_NAMES)


def test_top_level_backbone_tensors_stay_on_the_backbone_path():
    ar, side = partition_checkpoint_weights([(n, _t()) for n in TOP_AR_NAMES])
    assert sorted(n for n, _ in ar) == sorted(TOP_AR_NAMES)
    assert side == []


def test_latent_position_embedding_buffer_is_dropped():
    # Deterministic sinusoid rebuilt in __init__; a stale checkpoint copy would
    # be rejected by load_state_dict (persistent=False) or silently misbind.
    ar, side = partition_checkpoint_weights([("latent_pos_embed.pe", _t())])
    assert ar == [] and side == []


def test_tensor_identity_is_preserved_through_routing():
    sentinel = torch.tensor([7.0])
    _, side = partition_checkpoint_weights([("vae2llm.weight", sentinel)])
    assert side[0][1] is sentinel


def test_full_checkpoint_shape_routes_without_loss():
    """28 layers x 22 tensors + 12 top-level tensors, the real checkpoint shape."""
    weights = []
    for layer in range(28):
        for name in AR_LAYER_NAMES:
            weights.append((f"model.layers.{layer}.{name}.weight", _t()))
        for name in NAR_LAYER_NAMES:
            weights.append((f"model.layers.{layer}.{name}.weight", _t()))
    weights += [(n, _t()) for n in TOP_AR_NAMES]
    weights += [(n, _t()) for n in TOP_SIDE_NAMES]
    weights.append(("model.layers.0.nar_mlp.down_proj.weight", _t()))
    weights.append(("time_embedder.mlp.2.weight", _t()))
    weights.append(("latent_pos_embed.pe", _t()))
    ar, side = partition_checkpoint_weights(weights)
    assert len(ar) == 28 * 11 + 3  # per-layer AR names + embed/norm/lm_head
    assert len(side) == 28 * 10 + 5  # per-layer NAR names + nar_mlp.down + 3 top side + time mlp.2
    assert len(weights) == len(ar) + len(side) + 1  # + dropped latent_pos_embed.pe


def test_nar_name_without_layer_prefix_is_left_alone():
    # Defensive: only model.layers.N.nar_* is rewritten; a stray nar_ name
    # elsewhere must not crash or be mangled (it will fail strict load loudly).
    ar, side = partition_checkpoint_weights([("other.nar_thing.weight", _t())])
    assert [n for n, _ in side] == ["other.nar_thing.weight"]


def test_load_weights_loads_the_vae_eagerly(monkeypatch):
    """The VAE must load inside load_weights, not lazily at the first
    finishing request: a bad path or failed download then surfaces at
    startup, and the decoder weights sit on the GPU before vLLM's memory
    profiling sizes the KV cache."""
    from types import SimpleNamespace

    import vllm_omni.model_executor.models.yue2.yue2 as yue2_mod

    model = object.__new__(yue2_mod.Yue2ForCausalLM)
    monkeypatch.setattr(
        yue2_mod,
        "partition_checkpoint_weights",
        lambda _w: ([("ar.weight", _t())], [("side.weight", _t())]),
    )
    monkeypatch.setattr(
        yue2_mod,
        "AutoWeightsLoader",
        lambda _m: SimpleNamespace(load_weights=lambda pairs: {n for n, _ in pairs}),
    )
    model.load_state_dict = lambda _sd, strict=False: ([], [])
    vae_calls: list[torch.device] = []

    def _fake_vae_model(device: torch.device) -> object:
        vae_calls.append(device)
        return object()

    model._vae_model = _fake_vae_model

    loaded = model.load_weights(iter([]))
    assert loaded == {"ar.weight", "side.weight"}
    assert len(vae_calls) == 1


def test_vae_stays_out_of_the_module_tree(monkeypatch):
    """The checkpoint has no VAE keys: a registered ``_vae`` submodule would
    be flagged as uninitialized by DefaultModelLoader.track_weights_loading,
    and a model-wide dtype/device pass would silently change VAE numerics."""
    import vllm_omni.model_executor.models.yue2.yue2 as yue2_mod

    model = object.__new__(yue2_mod.Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model._vae = None
    model._vae_device = None
    monkeypatch.setattr(
        yue2_mod.YuE2VAE,
        "from_pretrained",
        lambda *_a, **_k: torch.nn.Linear(2, 2),
    )
    model._vae_model(torch.device("cpu"))
    assert isinstance(model._vae, torch.nn.Linear)
    assert not any(name.startswith("_vae.") for name, _ in model.named_parameters())
