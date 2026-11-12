# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared HunyuanImage 3.0 backbone layout and AR/DiT worker adapters.

This module is the single home for the "fusion ABI" contract between the AR
(stage 0, weight provider) and DiT (stage 1, weight consumer) backbones.

Both stages attach their backbone as ``.model`` and use the same per-layer
module naming (``embed_tokens``, ``layers.N.self_attn``, ``layers.N.mlp`` and
``layers.N.input_layernorm``/``post_attention_layernorm``, plus a final
``norm``).  Their dense linear weights use the same fused runtime names
(``qkv_proj``, ``gate_up_proj``) and their routed experts use the same upstream
MoeRunner names (``experts.routed_experts.w13_weight`` / ``w2_weight``).

The provider exports the finalized runtime tensors; the consumer reconstructs
the same canonical name space on ``meta``, validates the manifest and binds the
imported views.  The canonical name of a tensor is its backbone-relative
parameter path (relative to ``.model``), which is identical on both stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..backend import ExportSession, ImportedWeights
from ..contracts import CompatibilityError
from ..session import (
    SharedWeightGroupConfig,
    build_session_context,
)
from . import build_weight_spec

if TYPE_CHECKING:
    import torch

BACKBONE_ATTR = "model"

# Backbone-relative runtime parameter suffixes that belong to the shared
# backbone.  The AR and DiT stages must agree on this exact set; anything else
# (KV-cache state, AR/DiT private projections, embedded buffers) stays private.
_SHARED_LAYER_TAILS = frozenset(
    {
        ("input_layernorm", "weight"),
        ("post_attention_layernorm", "weight"),
        ("self_attn", "qkv_proj", "weight"),
        ("self_attn", "qkv_proj", "bias"),
        ("self_attn", "o_proj", "weight"),
        ("self_attn", "o_proj", "bias"),
        ("self_attn", "query_layernorm", "weight"),
        ("self_attn", "key_layernorm", "weight"),
        ("mlp", "gate", "weight"),
        ("mlp", "gate_up_proj", "weight"),
        ("mlp", "gate_up_proj", "bias"),
        ("mlp", "down_proj", "weight"),
        ("mlp", "down_proj", "bias"),
        ("mlp", "shared_mlp", "gate_up_proj", "weight"),
        ("mlp", "shared_mlp", "gate_up_proj", "bias"),
        ("mlp", "shared_mlp", "down_proj", "weight"),
        ("mlp", "shared_mlp", "down_proj", "bias"),
    }
)


def is_shared_param(name: str) -> bool:
    """Return True for a backbone-relative runtime parameter name to share.

    Accepts the canonical paths produced by ``backbone.named_parameters()``
    relative to ``.model`` (for example ``layers.0.self_attn.qkv_proj.weight``).
    """
    parts = name.split(".")
    if parts == ["embed_tokens", "weight"] or parts == ["norm", "weight"]:
        return True
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        tail = tuple(parts[2:])
        if tail in _SHARED_LAYER_TAILS:
            return True
        # Routed MoE experts own w13_weight/w2_weight on the shared upstream
        # MoeRunner.  Their intermediate submodule names are stable but kept
        # out of the contract; match the terminal weight names instead.
        if (
            len(tail) >= 3
            and tail[0] == "mlp"
            and tail[1] == "experts"
            and tail[-1]
            in (
                "w13_weight",
                "w2_weight",
            )
        ):
            return True
    return False


def shared_backbone_names(backbone: Any) -> list[str]:
    """Deterministically ordered canonical names of all shared backbone params."""
    return sorted(name for name, _ in backbone.named_parameters() if is_shared_param(name))


def collect_shared_bindings(backbone: Any) -> dict[str, torch.Tensor]:
    """Collect shared tensors once, in canonical name order (including on meta).

    Values are plain ``torch.Tensor`` views (``detach()``) sharing the original
    storage: the codec rejects tensor subclasses such as vLLM's
    ``ModelWeightParameter``, and a view avoids copying or mutating the model.
    """
    return {
        name: param.detach()
        for name, param in sorted(
            ((name, param) for name, param in backbone.named_parameters() if is_shared_param(name)),
            key=lambda item: item[0],
        )
    }


def attach_imported_weights(backbone: Any, imported: ImportedWeights) -> None:
    """Bind imported CUDA views into a ``meta``-constructed backbone.

    Each canonical name maps back to the same submodule path on the consumer.
    The reported set must match the local shared set exactly so the strict
    unloaded-weight check stays meaningful for private components.
    """
    import torch

    expected = shared_backbone_names(backbone)
    missing = sorted(set(expected) - set(imported))
    extra = sorted(set(imported) - set(expected))
    if missing or extra:
        raise CompatibilityError(
            f"imported views do not match the shared backbone layout (missing={missing}, extra={extra})"
        )

    for name in expected:
        view = imported[name]
        parent_name, _, attr = name.rpartition(".")
        parent = backbone.get_submodule(parent_name) if parent_name else backbone
        if not isinstance(view, torch.Tensor):
            raise CompatibilityError(f"imported view is not a tensor: {name}")
        parameter = torch.nn.Parameter(view, requires_grad=False)
        setattr(parent, attr, parameter)


def shared_param_full_names(model: Any) -> list[str]:
    """Full parameter paths on the pipeline (``model.<backbone-relative-name>``).

    These are the names ``DiffusersPipelineLoader._get_expected_parameter_names``
    reports, so they can be passed straight through as ``planned_weights`` to
    mark the shared backbone as externally loaded.
    """
    backbone = getattr(model, BACKBONE_ATTR)
    return [f"{BACKBONE_ATTR}.{name}" for name in shared_backbone_names(backbone)]


def is_shared_checkpoint_name(name: str) -> bool:
    """Return True if a HunyuanImage3 checkpoint weight feeds a shared param.

    ``name`` is the raw safetensors key (the outer ``model.`` prefix is still
    present for backbone weights).  This mirrors the fusion mapping the DiT
    backbone applies in ``HunyuanImage3Model.load_weights``: q/k/v collapse into
    ``qkv_proj``, gate/up collapse into ``gate_up_proj``/``gate_and_up_proj``,
    and every routed-expert tensor is shared.
    """
    if not name.startswith(f"{BACKBONE_ATTR}."):
        return False
    rel = name[len(f"{BACKBONE_ATTR}.") :]
    parts = rel.split(".")
    if parts and parts[-1] in ("weight", "bias"):
        parts = parts[:-1]
    # Some checkpoints glue bias to the projection name (gate_proj_bias);
    # unglue it back to the projection token so the prefix still matches.
    if parts and parts[-1].endswith("_bias") and parts[-1] != "bias":
        parts[-1] = parts[-1][: -len("_bias")]
    if not parts:
        return False

    if parts in (["embed_tokens"], ["norm"], ["wte"], ["ln_f"]):
        return True
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        tail = parts[2:]
        if not tail:
            return False
        if tail[0] in ("input_layernorm", "post_attention_layernorm"):
            return True
        if tail[0] == "self_attn" and len(tail) >= 2:
            return tail[1] in ("q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj", "query_layernorm", "key_layernorm")
        if tail[0] == "mlp" and len(tail) >= 2:
            if tail[1] == "experts":
                return True
            if tail[1] == "gate":
                return True
            if tail[1] in ("gate_proj", "gate_up_proj", "up_proj", "down_proj", "gate_and_up_proj"):
                return True
            if (
                tail[1] == "shared_mlp"
                and len(tail) >= 3
                and tail[2] in ("gate_proj", "gate_up_proj", "up_proj", "down_proj", "gate_and_up_proj")
            ):
                return True
    return False


def validate_supported_layout(config: Any) -> None:
    """Reject v1-unsupported HunyuanImage3 configuration shapes.

    The v1 shared layout requires a homogeneous, fused-QKV backbone: no CLA
    cross-attention layers and a scalar (non-per-layer) expert count.  Both
    stages must already agree on dtype/TP/moe backend; the manifest catches
    the rest.
    """
    if getattr(config, "use_cla", False):
        raise CompatibilityError("HunyuanImage3 weight sharing does not support CLA cross-attention layers")
    num_experts = getattr(config, "num_experts", None)
    if isinstance(num_experts, (list, tuple)):
        raise CompatibilityError("per-layer expert counts are not supported by the v1 shared layout")


def publish_backbone(
    backend: Any,
    model: Any,
    *,
    group: SharedWeightGroupConfig,
    device_index: int,
) -> ExportSession:
    """Provider entry: export the AR model's finalized backbone weights."""
    backbone = getattr(model, BACKBONE_ATTR)
    bindings = collect_shared_bindings(backbone)
    spec = build_weight_spec(group.model_id, group.checkpoint_revision, bindings, layout=group.layout)
    context = build_session_context(group, device_index)
    return backend.prepare_export(bindings, spec, context)


def open_backbone_import(
    backend: Any,
    model: Any,
    *,
    payload: bytes,
    group: SharedWeightGroupConfig,
    device_index: int,
) -> ImportedWeights:
    """Consumer entry: import and bind the shared backbone into a meta model."""
    backbone = getattr(model, BACKBONE_ATTR)
    bindings = collect_shared_bindings(backbone)
    expected = build_weight_spec(group.model_id, group.checkpoint_revision, bindings, layout=group.layout)
    context = build_session_context(group, device_index)
    imported = backend.open_import(payload, expected, context)
    try:
        attach_imported_weights(backbone, imported)
    except Exception:
        # The import succeeded but binding failed; seal the uncertain mapping
        # so the group can tear it down instead of leaking the IPC views.
        backend.mark_failed(imported)
        raise
    return imported


__all__ = [
    "is_shared_checkpoint_name",
    "open_backbone_import",
    "publish_backbone",
    "shared_param_full_names",
    "validate_supported_layout",
]
