# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B5 loader routing tests (CPU only).

Pins the two-path behaviour of `SenseNovaU1Pipeline.load_weights`:

1. `stacked_params_mapping` fuses q/k/v -> `qkv_proj` and gate/up ->
   `gate_up_proj`;
2. any other name is looked up directly in `named_parameters()` and copied.

The load-bearing property is that the **first matching entry wins** and the
`*_mot_gen` entries must therefore precede the plain `q_proj` / `k_proj` /
`v_proj` entries: `.q_proj` is a substring of `.q_proj_mot_gen`, so the reverse
order would route generation-tower weights into the understanding tower.

The name list used here is the real layout observed in the SenseNova-U1.5
checkpoint (1116 tensors; 420 routed through the mapping, 126 of them matching
more than one entry and resolved by ordering). If a checkpoint index is
available at `--sensenova-checkpoint`-style env var `SENSENOVA_MODEL_DIR`, the
audit is extended to every real name.
"""

import ast
import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_PIPELINE_SRC = (
    Path(__file__).resolve().parents[4]
    / "vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py"
)


def _load_mapping():
    text = _PIPELINE_SRC.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == "stacked_params_mapping" for t in targets):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple)):
            return ast.literal_eval(node.value)
    raise RuntimeError("stacked_params_mapping not found")


MAPPING = _load_mapping()


def all_matches(name):
    return [(p, w, s) for p, w, s in MAPPING if w in name]


def route(name):
    """The loader's path-1 loop: first matching entry wins."""
    for param_name, weight_name, shard_id in MAPPING:
        if weight_name in name:
            return name.replace(weight_name, param_name), shard_id
    return None, None


def _layer_names(layer: int) -> list[str]:
    """Real checkpoint names for one layer, both towers, all fused groups."""
    lm = f"language_model.model.layers.{layer}"
    names = []
    for proj in ("q", "k", "v"):
        names.append(f"{lm}.self_attn.{proj}_proj.weight")
        names.append(f"{lm}.self_attn.{proj}_proj_mot_gen.weight")
    for proj in ("o_proj", "o_proj_mot_gen"):
        names.append(f"{lm}.self_attn.{proj}.weight")
    for mlp, suffix in (("mlp", ""), ("mlp_mot_gen", "_mot_gen")):
        for proj in ("gate", "up", "down"):
            names.append(f"{lm}.{mlp}.{proj}_proj.weight")
        names.append(f"{lm}.post_attention_layernorm{suffix}.weight")
    for norm in ("input_layernorm", "input_layernorm_mot_gen"):
        names.append(f"{lm}.{norm}.weight")
    for norm in ("q_norm", "q_norm_hw", "k_norm", "k_norm_hw"):
        names.append(f"{lm}.self_attn.{norm}.weight")
        names.append(f"{lm}.self_attn.{norm}_mot_gen.weight")
    return names


# ---------------------------------------------------------------------------
# ordering: the core hazard
# ---------------------------------------------------------------------------


def test_mot_gen_entries_all_precede_plain_entries():
    """Ordering is the mechanism; assert it directly, not just its effect."""
    names = [w for _, w, _ in MAPPING]
    gen = [i for i, n in enumerate(names) if "mot_gen" in n]
    plain = [i for i, n in enumerate(names) if "mot_gen" not in n and n.startswith((".q", ".k", ".v"))]
    assert gen, "no mot_gen entries"
    assert plain, "no plain q/k/v entries"
    assert max(gen) < min(plain), f"mot_gen entries must precede plain ones: {MAPPING}"


def test_substring_hazard_is_real_on_these_names():
    """Guard the premise: `.q_proj` really is a substring of `.q_proj_mot_gen`."""
    assert ".qkv_proj" in ".qkv_proj_mot_gen"
    for p in ("q", "k", "v"):
        assert f".{p}_proj" in f".{p}_proj_mot_gen"


@pytest.mark.parametrize("layer", [0, 1, 20, 41])
def test_first_match_routes_each_tower_correctly(layer):
    """Every real per-layer name resolves to the right tower and fused target."""
    for name in _layer_names(layer):
        matches = all_matches(name)
        if not matches:
            continue  # path-2 name (norm/embedding/down_proj/o_proj)
        first = matches[0]
        target, shard = route(name)
        assert target == name.replace(first[1], first[0]), name
        assert shard == first[2], name
        assert ("mot_gen" in name) == ("mot_gen" in target), name


def test_both_towers_are_represented_in_the_mapping():
    """A loader bug can hide in the tower nobody checked."""
    und = route("language_model.model.layers.0.self_attn.q_proj.weight")
    gen = route("language_model.model.layers.0.self_attn.q_proj_mot_gen.weight")
    # only the projection component is rewritten; `.weight` is preserved
    assert und == ("language_model.model.layers.0.self_attn.qkv_proj.weight", "q")
    assert gen == ("language_model.model.layers.0.self_attn.qkv_proj_mot_gen.weight", "q")


def test_mlp_fusion_shard_ids():
    for layer in (0, 5, 41):
        base = f"language_model.model.layers.{layer}.mlp"
        assert route(f"{base}.gate_proj.weight") == (f"{base}.gate_up_proj.weight", 0)
        assert route(f"{base}.up_proj.weight") == (f"{base}.gate_up_proj.weight", 1)
        # down_proj is not fused -> path 2
        assert route(f"{base}.down_proj.weight") == (None, None)


def test_path2_names_are_left_for_direct_lookup():
    """Norms, embeddings, o_proj, down_proj and lm_head must not be captured."""
    for name in (
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "language_model.model.norm.weight",
        "language_model.model.norm_mot_gen.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.input_layernorm_mot_gen.weight",
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.layers.0.self_attn.k_norm_hw_mot_gen.weight",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.self_attn.o_proj_mot_gen.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
    ):
        assert route(name) == (None, None), f"{name} should use the direct-lookup path"


def test_no_two_sources_collapse_onto_one_target_and_shard():
    seen = {}
    for name in _layer_names(0) + _layer_names(41):
        target, shard = route(name)
        if target is None:
            continue
        key = (target, shard)
        assert key not in seen, f"{name} collides with {seen.get(key)} on {key}"
        seen[key] = name


def test_multi_entry_names_exist_so_ordering_is_load_bearing():
    multi = [n for n in _layer_names(0) if len(all_matches(n)) > 1]
    assert multi, "expected the mot_gen names to match more than one entry"
    assert all("mot_gen" in n for n in multi)


@pytest.mark.skipif(
    not os.environ.get("SENSENOVA_MODEL_DIR"),
    reason="set SENSENOVA_MODEL_DIR to audit every real checkpoint name",
)
def test_every_real_checkpoint_name_routes_or_is_owned_elsewhere():
    """Full audit over the real index when a checkpoint is available.

    Uncovered names must fall under `language_model.*` (path-2 direct lookup) or
    under a namespace this loader does not own (`fm_modules.`, `vision_model.`).
    """
    index = Path(os.environ["SENSENOVA_MODEL_DIR"]) / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip(f"no index at {index}")
    names = sorted(json.loads(index.read_text())["weight_map"])
    wrong = []
    for n in names:
        matches = all_matches(n)
        if not matches:
            assert n.startswith(("language_model.", "fm_modules.", "vision_model.")), n
            continue
        first = matches[0]
        if route(n) != (n.replace(first[1], first[0]), first[2]):
            wrong.append(n)
        if len(matches) > 1 and ("mot_gen" in n) != ("mot_gen" in route(n)[0]):
            wrong.append(n)
    assert not wrong, f"{len(wrong)} mis-routed names, e.g. {wrong[:5]}"
