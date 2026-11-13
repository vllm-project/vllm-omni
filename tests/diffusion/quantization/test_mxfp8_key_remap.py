# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for merge_mxfp8_checkpoint.py key-remapping helpers and model metadata.

These are pure-Python unit tests that exercise the transformation functions
without loading any actual checkpoint files or requiring NPU hardware.

Key contracts verified:
- SUPPORTED_MODEL_TYPES includes Wan2.2-TI2V-5B (MXFP8 supports it; MXFP4 does not)
- MXFP8_QUANT_CONFIG structure matches what TransformerConfig.from_dict() reads
- _remap_keys correctly translates msModelSlim naming → Diffusers naming
- _get_transformer_dirs routes cascade vs single-transformer models
- _get_quant_subdir maps high/low noise subdirs for cascade models
"""

import json
import pathlib

import pytest
import torch
from safetensors.torch import load_file, save_file

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# SUPPORTED_MODEL_TYPES and CASCADE_MODEL_TYPES
# ---------------------------------------------------------------------------


def test_supported_model_types_includes_all_wan22():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-T2V-A14B" in SUPPORTED_MODEL_TYPES
    assert "Wan2.2-I2V-A14B" in SUPPORTED_MODEL_TYPES


def test_supported_model_types_includes_ti2v_5b():
    """MXFP8 supports TI2V-5B (contrast: MXFP4 explicitly excludes it)."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import SUPPORTED_MODEL_TYPES

    assert "Wan2.2-TI2V-5B" in SUPPORTED_MODEL_TYPES


def test_cascade_model_types_excludes_ti2v():
    """TI2V-5B is a single-transformer model — must not be in CASCADE_MODEL_TYPES."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import CASCADE_MODEL_TYPES

    assert "Wan2.2-TI2V-5B" not in CASCADE_MODEL_TYPES
    assert "Wan2.2-T2V-A14B" in CASCADE_MODEL_TYPES
    assert "Wan2.2-I2V-A14B" in CASCADE_MODEL_TYPES


def test_supported_model_types_includes_minimax_h3():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import SUPPORTED_MODEL_TYPES

    assert "MiniMax-H3" in SUPPORTED_MODEL_TYPES


# ---------------------------------------------------------------------------
# MXFP8_QUANT_CONFIG — auto-detection contract
# ---------------------------------------------------------------------------


def test_mxfp8_quant_config_has_required_keys():
    """MXFP8_QUANT_CONFIG must carry exactly the keys that auto-detection reads:
    quant_method (selects DiffusionMXFP8Config) and is_checkpoint_mxfp8_serialized
    (selects NPUMxfp8LinearMethod over the online path)."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import MXFP8_QUANT_CONFIG

    assert MXFP8_QUANT_CONFIG["quant_method"] == "mxfp8"
    assert MXFP8_QUANT_CONFIG["is_checkpoint_mxfp8_serialized"] is True


# ---------------------------------------------------------------------------
# _get_transformer_dirs
# ---------------------------------------------------------------------------


def test_get_transformer_dirs_cascade():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_transformer_dirs

    assert _get_transformer_dirs("Wan2.2-T2V-A14B") == ["transformer", "transformer_2"]
    assert _get_transformer_dirs("Wan2.2-I2V-A14B") == ["transformer", "transformer_2"]


def test_get_transformer_dirs_single():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_transformer_dirs

    assert _get_transformer_dirs("Wan2.2-TI2V-5B") == ["transformer"]


def test_get_transformer_dirs_minimax_h3():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_transformer_dirs

    assert _get_transformer_dirs("MiniMax-H3") == ["FL2VA/transformer"]


# ---------------------------------------------------------------------------
# _get_quant_subdir
# ---------------------------------------------------------------------------


def test_get_quant_subdir_cascade_high_noise():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_quant_subdir

    base = pathlib.Path("/quant")
    result = _get_quant_subdir("Wan2.2-T2V-A14B", base, "transformer")
    assert result == base / "high_noise_model"


def test_get_quant_subdir_cascade_low_noise():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_quant_subdir

    base = pathlib.Path("/quant")
    result = _get_quant_subdir("Wan2.2-T2V-A14B", base, "transformer_2")
    assert result == base / "low_noise_model"


def test_get_quant_subdir_non_cascade():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _get_quant_subdir

    base = pathlib.Path("/quant")
    result = _get_quant_subdir("Wan2.2-TI2V-5B", base, "transformer")
    assert result == base


# ---------------------------------------------------------------------------
# _remap_keys — msModelSlim naming → Diffusers naming
# ---------------------------------------------------------------------------


def test_remap_keys_self_attn_q():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"blocks.0.self_attn.q.weight": torch.zeros(1)}
    meta = {"blocks.0.self_attn.q.weight": "W8A8_MXFP8"}
    new_state, new_meta = _remap_keys(state, meta)
    assert "blocks.0.attn1.to_q.weight" in new_state
    assert new_meta.get("blocks.0.attn1.to_q.weight") == "W8A8_MXFP8"


def test_remap_keys_self_attn_all_heads():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    pairs = {
        "self_attn.q": "attn1.to_q",
        "self_attn.k": "attn1.to_k",
        "self_attn.v": "attn1.to_v",
        "self_attn.o": "attn1.to_out.0",
    }
    for src_part, dst_part in pairs.items():
        src_key = f"blocks.0.{src_part}.weight"
        state = {src_key: torch.zeros(1)}
        new_state, _ = _remap_keys(state, {})
        expected = f"blocks.0.{dst_part}.weight"
        assert expected in new_state, f"{src_key} → expected {expected}, got {list(new_state)}"


def test_remap_keys_ffn():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {
        "blocks.1.ffn.0.weight": torch.zeros(1),
        "blocks.1.ffn.2.weight": torch.zeros(1),
    }
    new_state, _ = _remap_keys(state, {})
    assert "blocks.1.ffn.net.0.proj.weight" in new_state
    assert "blocks.1.ffn.net.2.weight" in new_state


def test_remap_keys_norm_order_swap():
    """norm2↔norm3 swap: msModelSlim uses norm1/norm3/norm2 order,
    Diffusers uses norm1/norm2/norm3."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {
        "blocks.0.norm2.weight": torch.zeros(1),
        "blocks.0.norm3.weight": torch.zeros(1),
    }
    new_state, _ = _remap_keys(state, {})
    # norm2 → norm3 and norm3 → norm2
    assert "blocks.0.norm3.weight" in new_state
    assert "blocks.0.norm2.weight" in new_state
    # Both must be present (swap, not collapse)
    assert len([k for k in new_state if "norm" in k and "norm_q" not in k and "norm_k" not in k]) == 2


def test_remap_keys_head():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"head.head.weight": torch.zeros(1)}
    new_state, _ = _remap_keys(state, {})
    assert "proj_out.weight" in new_state


def test_remap_keys_cross_attn():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"blocks.0.cross_attn.q.weight": torch.zeros(1)}
    new_state, _ = _remap_keys(state, {})
    assert "blocks.0.attn2.to_q.weight" in new_state


def test_remap_keys_meta_only_mapped_for_existing_state_keys():
    """quant_meta entries are only emitted for keys present in state_dict."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"blocks.0.self_attn.q.weight": torch.zeros(1)}
    # meta has an extra key not in state_dict
    meta = {
        "blocks.0.self_attn.q.weight": "W8A8_MXFP8",
        "blocks.0.self_attn.q.weight_scale": "W8A8_MXFP8",
    }
    _, new_meta = _remap_keys(state, meta)
    assert "blocks.0.attn1.to_q.weight" in new_meta
    # weight_scale was in meta but not state_dict → must be absent
    assert "blocks.0.attn1.to_q.weight_scale" not in new_meta


def test_remap_keys_preserves_tensors():
    """Tensor values must survive the key rename unchanged."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    t = torch.randn(4, 8)
    state = {"blocks.0.self_attn.q.weight": t}
    new_state, _ = _remap_keys(state, {})
    assert torch.equal(new_state["blocks.0.attn1.to_q.weight"], t)


def test_remap_keys_smoothquant_multiplier():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    scale = torch.tensor([0.5, 2.0])
    new_state, new_meta = _remap_keys(
        {"blocks.0.self_attn.q.div.mul_scale": scale},
        {"blocks.0.self_attn.q.div.mul_scale": "FLOAT"},
    )
    name = "blocks.0.attn1.to_q.mul_scale"
    assert torch.equal(new_state[name], scale)
    assert new_meta[name] == "FLOAT"


def test_remap_keys_unwraps_msmodelslim_smoothquant_linear():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {
        "blocks.0.self_attn.q.linear.weight": torch.zeros(2, 4),
        "blocks.0.self_attn.q.linear.weight_scale": torch.zeros(2, 1, dtype=torch.uint8),
        "blocks.0.self_attn.q.div.mul_scale": torch.ones(4),
    }
    meta = {
        "blocks.0.self_attn.q.linear.weight": "W8A8_MXFP8",
        "blocks.0.self_attn.q.linear.weight_scale": "W8A8_MXFP8",
        "blocks.0.self_attn.q.div.mul_scale": "FLOAT",
    }

    new_state, new_meta = _remap_keys(state, meta)

    assert "blocks.0.attn1.to_q.weight" in new_state
    assert "blocks.0.attn1.to_q.weight_scale" in new_state
    assert "blocks.0.attn1.to_q.mul_scale" in new_state
    assert "blocks.0.attn1.to_q.linear.weight" not in new_state
    assert new_meta["blocks.0.attn1.to_q.weight"] == "W8A8_MXFP8"


def test_remap_keys_preserves_unrelated_linear_segment():
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {"blocks.0.condition_embedder.linear.weight": torch.zeros(1)}
    new_state, _ = _remap_keys(state, {})
    assert "blocks.0.condition_embedder.linear.weight" in new_state


def test_remap_keys_minimax_h3_smoothquant():
    """MiniMax-H3 remap unwraps smoothed linears but keeps other .linear. names."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _remap_keys

    state = {
        "blocks.0.attn.qkv_proj.linear.weight": torch.zeros(2, 4),
        "blocks.0.attn.qkv_proj.linear.weight_scale": torch.zeros(2, 1, dtype=torch.uint8),
        "blocks.0.attn.qkv_proj.div.mul_scale": torch.ones(4),
        "blocks.0.adaln_proj.linear.weight": torch.zeros(6, 4),
    }
    meta = {
        "blocks.0.attn.qkv_proj.linear.weight": "W8A8_MXFP8",
        "blocks.0.attn.qkv_proj.linear.weight_scale": "W8A8_MXFP8",
        "blocks.0.attn.qkv_proj.div.mul_scale": "FLOAT",
        "blocks.0.adaln_proj.linear.weight": "FLOAT",
    }

    new_state, new_meta = _remap_keys(state, meta, rename_dict={})

    assert "blocks.0.attn.qkv_proj.weight" in new_state
    assert "blocks.0.attn.qkv_proj.weight_scale" in new_state
    assert "blocks.0.attn.qkv_proj.mul_scale" in new_state
    assert "blocks.0.attn.qkv_proj.linear.weight" not in new_state
    # The AdaLN linear has no smooth marker, so its .linear. prefix is kept.
    assert "blocks.0.adaln_proj.linear.weight" in new_state
    assert new_meta["blocks.0.attn.qkv_proj.weight"] == "W8A8_MXFP8"


def test_reverse_qkv_grouped_reorder_round_trips():
    """Reverse reorder maps the runtime (Q|K|V) layout back to the grouped checkpoint layout."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _reverse_qkv_grouped_reorder

    num_heads, head_dim = 2, 4
    # Runtime layout: [Q(2*4), K(2*4), V(2*4)] rows.
    runtime = torch.arange(3 * num_heads * head_dim).reshape(3 * num_heads * head_dim, 1).float()
    grouped = _reverse_qkv_grouped_reorder(
        {"blocks.0.attn.qkv_proj.weight": runtime.clone()},
        num_query_groups=num_heads,
        heads_per_group=1,
        head_dim=head_dim,
    )["blocks.0.attn.qkv_proj.weight"]
    # Grouped order interleaves per query group: q[g], k[g], v[g].
    q = runtime[: num_heads * head_dim]
    k = runtime[num_heads * head_dim : 2 * num_heads * head_dim]
    v = runtime[2 * num_heads * head_dim :]
    expected = torch.stack(
        [chunk[g * head_dim : (g + 1) * head_dim, 0] for g in range(num_heads) for chunk in (q, k, v)]
    ).reshape(-1, 1)
    torch.testing.assert_close(grouped, expected)


@pytest.mark.parametrize(
    ("state", "meta", "expected"),
    [
        ({"layer.weight": torch.ones(1), "layer.mul_scale": torch.ones(4)}, {"layer.weight": "W8A8_MXFP8"}, True),
        ({"layer.weight": torch.ones(1)}, {"layer.weight": "W8A8_MXFP8"}, False),
        ({"layer.weight": torch.ones(1)}, {"layer.weight": "FLOAT"}, False),
    ],
)
def test_smoothquant_checkpoint_requirement(state, meta, expected):
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _has_complete_mxfp8_smooth_scales

    assert _has_complete_mxfp8_smooth_scales(state, meta) is expected


@pytest.mark.parametrize("with_smooth_scale", [False, True])
def test_convert_exported_checkpoint_preserves_scales_and_float_layers(tmp_path, with_smooth_scale):
    """Conversion consumes exported tensors without requiring calibration metadata."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _convert_transformer

    original = tmp_path / "original"
    quantized = tmp_path / "quantized"
    output = tmp_path / "output"
    original.mkdir()
    quantized.mkdir()
    base = {
        "blocks.0.attn1.to_q.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "blocks.0.attn1.to_q.bias": torch.randn(8, dtype=torch.bfloat16),
        "proj_out.weight": torch.randn(4, 32, dtype=torch.bfloat16),
    }
    save_file(base, original / "diffusion_pytorch_model.safetensors")
    (original / "config.json").write_text(json.dumps({"hidden_size": 32}))
    prefix = "blocks.0.self_attn.q"
    weight_prefix = prefix + ".linear" if with_smooth_scale else prefix
    weight = torch.randn(8, 32).to(torch.float8_e4m3fn)
    scales = torch.full((8, 1), 127, dtype=torch.uint8)
    exported = {
        weight_prefix + ".weight": weight,
        weight_prefix + ".weight_scale": scales,
        "head.head.weight": base["proj_out.weight"],
    }
    description = {
        weight_prefix + ".weight": "W8A8_MXFP8",
        weight_prefix + ".weight_scale": "W8A8_MXFP8",
        "head.head.weight": "FLOAT",
    }
    if with_smooth_scale:
        exported[prefix + ".div.mul_scale"] = torch.linspace(0.5, 2.0, 32)
        description[prefix + ".div.mul_scale"] = "FLOAT"
    save_file(exported, quantized / "quant_model_weight.safetensors")
    (quantized / "quant_model_description.json").write_text(json.dumps(description))
    # Search diagnostics are optional and have no deployment semantics.
    (quantized / "flex_smooth_search.json").write_text("unused calibration diagnostics")

    _convert_transformer("Wan2.2-TI2V-5B", quantized, output, original)

    merged = load_file(output / "diffusion_pytorch_model.safetensors")
    assert torch.equal(merged["blocks.0.attn1.to_q.weight"].float(), weight.float())
    assert torch.equal(merged["blocks.0.attn1.to_q.weight_scale"], scales)
    assert torch.equal(merged["blocks.0.attn1.to_q.bias"], base["blocks.0.attn1.to_q.bias"])
    assert torch.equal(merged["proj_out.weight"], base["proj_out.weight"])
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["ignored_layers"] == ["proj_out"]
    assert config.get("require_smooth_scale", False) is with_smooth_scale
    assert "alpha" not in config and "beta" not in config
    if with_smooth_scale:
        assert torch.equal(merged["blocks.0.attn1.to_q.mul_scale"], exported[prefix + ".div.mul_scale"])


@pytest.mark.parametrize(
    ("model_type", "checkpoint_name", "expected"),
    [
        # Wan2.2's loader renames these before the layer is built, but
        # ignored_layers is consumed first, so it must already use runtime names.
        ("Wan2.2-T2V-A14B", "blocks.0.ffn.net.0.proj", "blocks.0.ffn.net_0.proj"),
        ("Wan2.2-T2V-A14B", "blocks.0.ffn.net.2", "blocks.0.ffn.net_2"),
        ("Wan2.2-I2V-A14B", "blocks.0.attn1.to_out.0", "blocks.0.attn1.to_out"),
        ("Wan2.2-TI2V-5B", "blocks.0.attn2.to_out.0", "blocks.0.attn2.to_out"),
        # Unrelated names and other model families pass through untouched.
        ("Wan2.2-T2V-A14B", "blocks.0.ffn.net_2", "blocks.0.ffn.net_2"),
        ("Wan2.2-T2V-A14B", "proj_out", "proj_out"),
        ("MiniMax-H3", "blocks.0.adaln_proj.linear", "blocks.0.adaln_proj.linear"),
        ("MiniMax-H3", "condition_proj", "condition_proj"),
    ],
)
def test_runtime_ignored_layer_name(model_type, checkpoint_name, expected):
    """FLOAT layers must reach ignored_layers under their runtime prefix."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _runtime_ignored_layer_name

    assert _runtime_ignored_layer_name(checkpoint_name, model_type) == expected


def test_convert_exported_checkpoint_normalizes_ignored_layers_to_runtime(tmp_path):
    """FLOAT Wan2.2 layers are skipped by the names the runtime actually builds."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import _convert_transformer

    original = tmp_path / "original"
    quantized = tmp_path / "quantized"
    output = tmp_path / "output"
    original.mkdir()
    quantized.mkdir()
    # Checkpoint spellings: the rename table turns these into ffn.net.0.proj,
    # ffn.net.2 and attn1.to_out.0, while the runtime builds ffn.net_0.proj,
    # ffn.net_2 and attn1.to_out.
    base = {
        "blocks.0.ffn.0.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "blocks.0.ffn.2.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "blocks.0.self_attn.o.weight": torch.randn(8, 32, dtype=torch.bfloat16),
        "blocks.0.self_attn.q.weight": torch.randn(8, 32, dtype=torch.bfloat16),
    }
    save_file(base, original / "diffusion_pytorch_model.safetensors")
    (original / "config.json").write_text(json.dumps({"hidden_size": 32}))
    exported = {
        "blocks.0.ffn.0.weight": base["blocks.0.ffn.0.weight"],
        "blocks.0.ffn.2.weight": base["blocks.0.ffn.2.weight"],
        "blocks.0.self_attn.o.weight": base["blocks.0.self_attn.o.weight"],
        "blocks.0.self_attn.q.weight": torch.randn(8, 32).to(torch.float8_e4m3fn),
        "blocks.0.self_attn.q.weight_scale": torch.full((8, 1), 127, dtype=torch.uint8),
    }
    description = {
        "blocks.0.ffn.0.weight": "FLOAT",
        "blocks.0.ffn.2.weight": "FLOAT",
        "blocks.0.self_attn.o.weight": "FLOAT",
        "blocks.0.self_attn.q.weight": "W8A8_MXFP8",
        "blocks.0.self_attn.q.weight_scale": "W8A8_MXFP8",
    }
    save_file(exported, quantized / "quant_model_weight.safetensors")
    (quantized / "quant_model_description.json").write_text(json.dumps(description))

    _convert_transformer("Wan2.2-T2V-A14B", quantized, output, original)

    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["ignored_layers"] == [
        "blocks.0.attn1.to_out",
        "blocks.0.ffn.net_0.proj",
        "blocks.0.ffn.net_2",
    ]
    # Checkpoint tensor keys keep the renamed (diffusers-style) spelling; only
    # ignored_layers is rewritten to the runtime prefix.
    merged = load_file(output / "diffusion_pytorch_model.safetensors")
    assert "blocks.0.ffn.net.2.weight" in merged
    assert "blocks.0.attn1.to_out.0.weight" in merged


def test_repack_minimax_h3_component_links_resolve_with_relative_paths(tmp_path, monkeypatch):
    """A relative --original-model must still produce resolvable component links."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import (
        MINIMAX_H3_SYMLINK_DIRS,
        repack,
    )

    original = tmp_path / "original"
    transformer = original / "FL2VA" / "transformer"
    transformer.mkdir(parents=True)
    (original / "model_index.json").write_text(json.dumps({"partition": "fl2va"}))
    (original / "FL2VA" / "model_index.json").write_text(json.dumps({"partition": "fl2va"}))
    (transformer / "config.json").write_text(
        json.dumps({"hidden_size": 32, "num_attention_heads": 1, "attention_head_dim": 32})
    )
    save_file(
        {"blocks.0.norm1.weight": torch.ones(4, dtype=torch.bfloat16)},
        transformer / "diffusion_pytorch_model.safetensors",
    )
    for sub in MINIMAX_H3_SYMLINK_DIRS:
        (original / "FL2VA" / sub).mkdir()
        (original / "FL2VA" / sub / "marker.txt").write_text(sub)

    quantized = tmp_path / "quantized"
    quantized.mkdir()
    save_file(
        {"blocks.0.mlp.fc2.weight": torch.zeros(2, 2).to(torch.float8_e4m3fn)},
        quantized / "quant_model_weight.safetensors",
    )
    (quantized / "quant_model_description.json").write_text(json.dumps({"blocks.0.mlp.fc2.weight": "W8A8_MXFP8"}))

    monkeypatch.chdir(tmp_path)
    repack("MiniMax-H3", pathlib.Path("original"), pathlib.Path("quantized"), pathlib.Path("output"))

    for sub in MINIMAX_H3_SYMLINK_DIRS:
        link = tmp_path / "output" / "FL2VA" / sub
        assert link.is_symlink(), sub
        assert link.exists(), f"{sub} link is dangling: {link.readlink()}"
        assert pathlib.Path(link).resolve() == (original / "FL2VA" / sub).resolve()
