# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The OmniVoice generator packs q/k/v and gate/up into fused projections.

The HF checkpoint stores those five tensors separately, so packing them is a
load-time step -- and the failure mode it introduces is silent. The generic
name-based loader looks up ``q_proj``/``gate_proj`` as modules; once they no
longer exist it finds nothing, logs a warning, and leaves the fused parameters
at their random initialization. The model then serves requests and produces
noise. These tests assert the property that failure mode violates: after a
load, every fused parameter is fully written from the checkpoint, and anything
that would leave one partly written raises instead.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    _FUSED_PROJECTIONS,
    OmniVoiceGenerator,
)
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_LAYERS = 2
HIDDEN = 32
HEAD_DIM = 8
NUM_HEADS = 4
NUM_KV_HEADS = 2
INTERMEDIATE = 64


def _config() -> OmniVoiceConfig:
    return OmniVoiceConfig(
        llm_config={
            "hidden_size": HIDDEN,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTERMEDIATE,
            "vocab_size": 64,
            "max_position_embeddings": 128,
        },
        enable_cuda_graph=False,
    )


def _checkpoint_shards() -> dict[str, torch.Tensor]:
    """The five per-layer tensors an OmniVoice checkpoint actually stores."""
    torch.manual_seed(0)
    shards: dict[str, torch.Tensor] = {}
    for layer in range(NUM_LAYERS):
        prefix = f"llm.layers.{layer}"
        shards[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN)
        shards[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
        shards[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN)
        shards[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
        shards[f"{prefix}.mlp.up_proj.weight"] = torch.randn(INTERMEDIATE, HIDDEN)
    return shards


def test_every_fused_parameter_is_written() -> None:
    generator = OmniVoiceGenerator(_config())
    shards = _checkpoint_shards()

    loaded = generator._load_fused_projections(shards)

    assert loaded == set(shards), "some checkpoint shards were not consumed"
    shards_per_layer = sum(len(paths) for paths in _FUSED_PROJECTIONS.values())
    assert shards_per_layer == 5, "q,k,v and gate,up"
    assert len(loaded) == NUM_LAYERS * shards_per_layer


def test_packed_weight_matches_the_shards_it_came_from() -> None:
    """Packing order is q,k,v and gate,up -- the split in forward() assumes it."""
    generator = OmniVoiceGenerator(_config())
    shards = _checkpoint_shards()
    generator._load_fused_projections(shards)

    for layer in range(NUM_LAYERS):
        prefix = f"llm.layers.{layer}"
        qkv = torch.cat(
            [
                shards[f"{prefix}.self_attn.q_proj.weight"],
                shards[f"{prefix}.self_attn.k_proj.weight"],
                shards[f"{prefix}.self_attn.v_proj.weight"],
            ],
            dim=0,
        )
        gate_up = torch.cat(
            [shards[f"{prefix}.mlp.gate_proj.weight"], shards[f"{prefix}.mlp.up_proj.weight"]],
            dim=0,
        )
        torch.testing.assert_close(generator.layers[layer].self_attn.qkv_proj.weight, qkv)
        torch.testing.assert_close(generator.layers[layer].mlp.gate_up_proj.weight, gate_up)


def test_no_fused_parameter_keeps_its_random_init() -> None:
    """The corruption this guards against: a fused param never written at all."""
    generator = OmniVoiceGenerator(_config())
    before = {
        name: param.detach().clone()
        for name, param in generator.named_parameters()
        if name.endswith(("qkv_proj.weight", "gate_up_proj.weight"))
    }
    assert len(before) == NUM_LAYERS * len(_FUSED_PROJECTIONS)

    generator._load_fused_projections(_checkpoint_shards())

    for name, initial in before.items():
        current = dict(generator.named_parameters())[name]
        assert not torch.equal(current, initial), f"{name} was left at its initialization"


@pytest.mark.parametrize(
    "dropped",
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "mlp.gate_proj", "mlp.up_proj"],
)
def test_a_missing_shard_raises_instead_of_loading_partially(dropped: str) -> None:
    generator = OmniVoiceGenerator(_config())
    shards = _checkpoint_shards()
    del shards[f"llm.layers.0.{dropped}.weight"]

    with pytest.raises(ValueError, match="missing"):
        generator._load_fused_projections(shards)


def test_a_wrong_shaped_shard_raises() -> None:
    generator = OmniVoiceGenerator(_config())
    shards = _checkpoint_shards()
    shards["llm.layers.0.self_attn.q_proj.weight"] = torch.randn(NUM_HEADS * HEAD_DIM + 1, HIDDEN)

    with pytest.raises(ValueError, match="pack to"):
        generator._load_fused_projections(shards)


def test_a_checkpoint_missing_a_whole_layer_raises() -> None:
    """Half-loaded is the dangerous state: it neither errors nor works."""
    generator = OmniVoiceGenerator(_config())
    shards = {k: v for k, v in _checkpoint_shards().items() if not k.startswith("llm.layers.1.")}

    with pytest.raises(ValueError, match="fused projections"):
        generator._load_fused_projections(shards)


def test_a_checkpoint_with_no_fused_shards_is_left_alone() -> None:
    """An unrelated state_dict must not trip the completeness check."""
    generator = OmniVoiceGenerator(_config())
    assert generator._load_fused_projections({"audio_heads.weight": torch.randn(4, 4)}) == set()


def test_load_weights_wires_the_packing_in(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The packing has to be reached from load_weights, not merely available.

    Every test above calls _load_fused_projections directly, so deleting its call
    site from load_weights would leave them all green while the model loaded
    corrupted. This goes through the public entry point instead.
    """
    import safetensors.torch

    generator = OmniVoiceGenerator(_config())
    shards = _checkpoint_shards()
    before = generator.layers[0].self_attn.qkv_proj.weight.detach().clone()

    (tmp_path / "model.safetensors").write_bytes(b"")  # only its existence is checked
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *args, **kwargs: shards)
    generator.load_weights(str(tmp_path), torch.device("cpu"))

    expected = torch.cat(
        [
            shards["llm.layers.0.self_attn.q_proj.weight"],
            shards["llm.layers.0.self_attn.k_proj.weight"],
            shards["llm.layers.0.self_attn.v_proj.weight"],
        ],
        dim=0,
    )
    written = generator.layers[0].self_attn.qkv_proj.weight
    assert not torch.equal(written, before), "load_weights left the fused projection at its init"
    torch.testing.assert_close(written, expected)
