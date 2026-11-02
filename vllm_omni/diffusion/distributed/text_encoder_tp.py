# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor parallelism for Hugging Face text encoders.

Diffusion pipelines usually load their text encoder with
``transformers.PreTrainedModel.from_pretrained`` and keep a full copy on every
DiT rank.  For large VLM encoders (e.g. the ~15 GB Qwen2.5-VL used by
Qwen-Image) that copy dominates per-device memory regardless of the DiT
parallel layout.

:func:`maybe_shard_text_encoder` shards such an encoder in place with the
model's own ``base_model_tp_plan`` (the plan transformers ships for its
``tp_plan="auto"`` support): ``colwise`` linears keep a slice of their output
features and ``rowwise`` linears keep a slice of their input features and
all-reduce their output.  Transformers attention reshapes projections with
``-1`` heads, so the attention modules work on the local heads unchanged.

Requirements for the caller:

- shard before moving the encoder to the device, so the full copy never
  occupies device memory;
- every rank of a text-encoder TP group must run the encoder in lockstep with
  identical inputs (the encoder output is replicated across the group).
"""

from __future__ import annotations

from typing import Any

import regex as re
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import skip_init
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
from vllm_omni.diffusion.distributed.parallel_state import get_text_encoder_tp_group

logger = init_logger(__name__)

_COLWISE = "colwise"
_ROWWISE = "rowwise"


class RowShardedLinear(nn.Linear):
    """``nn.Linear`` holding an input-feature shard; all-reduces its output."""

    tp_group: GroupCoordinator

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = self.tp_group.all_reduce(F.linear(input_, self.weight))
        # Add the bias once, after the reduction.
        if self.bias is not None:
            output = output + self.bias
        return output


def _shard_linear(src: nn.Linear, style: str, group: GroupCoordinator) -> nn.Linear:
    tp_size, tp_rank = group.world_size, group.rank_in_group
    if style == _COLWISE:
        cls, in_features, out_features = nn.Linear, src.in_features, src.out_features // tp_size
        part = slice(tp_rank * out_features, (tp_rank + 1) * out_features)
        weight, bias = src.weight[part], None if src.bias is None else src.bias[part]
    else:
        cls, in_features, out_features = RowShardedLinear, src.in_features // tp_size, src.out_features
        part = slice(tp_rank * in_features, (tp_rank + 1) * in_features)
        weight, bias = src.weight[:, part], src.bias
    dst = skip_init(
        cls,
        in_features,
        out_features,
        bias=src.bias is not None,
        device=src.weight.device,
        dtype=src.weight.dtype,
    )
    with torch.no_grad():
        dst.weight.copy_(weight)
        if bias is not None:
            dst.bias.copy_(bias)
    dst.requires_grad_(src.weight.requires_grad)
    if isinstance(dst, RowShardedLinear):
        dst.tp_group = group
    return dst


def _find_tp_plan(model: nn.Module) -> tuple[dict[str, str], Any]:
    """Return the text model's ``base_model_tp_plan`` and the config owning it."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"{type(model).__name__} has no Hugging Face config; text encoder TP is unsupported")
    candidates = []
    if hasattr(config, "get_text_config"):
        candidates.append(config.get_text_config(decoder=True))
    candidates.append(config)
    for cfg in candidates:
        plan = getattr(cfg, "base_model_tp_plan", None)
        if plan:
            return dict(plan), cfg
    raise ValueError(
        f"{type(model).__name__} does not define a base_model_tp_plan; text encoder TP is unsupported for this model"
    )


def _plan_regex(pattern: str) -> re.Pattern[str]:
    # Plan keys are relative to the base text model ("layers.*.self_attn.q_proj");
    # match them as a dotted suffix of the full module name.
    return re.compile(r"(?:^|\.)" + re.escape(pattern).replace(r"\*", r"\d+") + "$")


def _validate_tp_size(cfg: Any, tp_size: int) -> None:
    for attr in ("num_attention_heads", "num_key_value_heads"):
        value = getattr(cfg, attr, None)
        if value is not None and value % tp_size != 0:
            raise ValueError(f"text_encoder_tp_size ({tp_size}) must divide the text encoder's {attr} ({value})")


def shard_text_encoder(model: nn.Module, group: GroupCoordinator) -> nn.Module:
    """Shard ``model``'s linears in place across ``group`` using its TP plan."""
    tp_size = group.world_size
    if tp_size == 1:
        return model
    plan, plan_config = _find_tp_plan(model)
    unsupported = sorted({style for style in plan.values() if style not in (_COLWISE, _ROWWISE)})
    if unsupported:
        raise NotImplementedError(
            f"{type(model).__name__} TP plan uses unsupported styles {unsupported}; "
            f"text encoder TP supports only {_COLWISE!r} and {_ROWWISE!r}"
        )
    _validate_tp_size(plan_config, tp_size)

    rules = [(_plan_regex(pattern), style) for pattern, style in plan.items()]
    targets: list[tuple[str, str]] = []
    for name, module in model.named_modules():
        for regex, style in rules:
            if regex.search(name):
                if type(module) is not nn.Linear:
                    raise NotImplementedError(
                        f"text encoder TP expects nn.Linear at {name!r}, got {type(module).__name__}"
                    )
                dim = module.out_features if style == _COLWISE else module.in_features
                if dim % tp_size != 0:
                    raise ValueError(f"text_encoder_tp_size ({tp_size}) must divide {name} features ({dim})")
                targets.append((name, style))
                break
    if not targets:
        raise ValueError(f"no linear in {type(model).__name__} matched its TP plan; text encoder TP is unsupported")

    for name, style in targets:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        src = getattr(parent, child_name)
        setattr(parent, child_name, _shard_linear(src, style, group))

    logger.info(
        "Sharded %s across text-encoder TP group %s: %d linears",
        type(model).__name__,
        group.ranks,
        len(targets),
    )
    return model


def maybe_shard_text_encoder(model: nn.Module, od_config: Any) -> nn.Module:
    """Shard ``model`` if ``parallel_config.text_encoder_tp_size`` > 1."""
    parallel_config = getattr(od_config, "parallel_config", None)
    tp_size = int(getattr(parallel_config, "text_encoder_tp_size", 1) or 1)
    if tp_size == 1:
        return model
    return shard_text_encoder(model, get_text_encoder_tp_group(tp_size))
