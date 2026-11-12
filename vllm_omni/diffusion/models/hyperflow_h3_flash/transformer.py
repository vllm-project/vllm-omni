# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Structural deltas follow Video-Rebirth/hyperflow-h3-flash (Apache-2.0):
# dropped blocks, endpoint time embed, low-rank and INT8 checkpoint layout.
"""HyperFlow H3 Flash DiT on Omni's MiniMax-H3 stack.

P0 loads HyperFlow's pruned / low-rank / INT8 convrot checkpoint by densifying
into Omni's fused MiniMax-H3 modules (correctness + serve path). Native W8A8
GEMM reuse is tracked as a follow-up (RFC performance workstream).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.hyperflow_h3_flash.checkpoint import DEFAULT_DROPPED_BLOCKS
from vllm_omni.diffusion.models.hyperflow_h3_flash.int8_convrot import regular_hadamard
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3DiTModel,
    MiniMaxH3TimeEmbedder,
)

logger = init_logger(__name__)


class DroppedBlock(nn.Module):
    """Removed DiT block: hidden states pass through unchanged."""

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x


def _diffusers_block_index(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "transformer_blocks" and parts[1].isdigit():
        return int(parts[1])
    return None


def densify_int8(qweight: torch.Tensor, wscale: torch.Tensor, group: int) -> torch.Tensor:
    """Recover bf16 ``W`` from HyperFlow ``qweight ≈ round(W R / wscale)``."""
    rot = regular_hadamard(group).to(dtype=torch.float32)
    # qweight: [out, in], wscale: [out]
    scaled = qweight.to(torch.float32) * wscale.to(torch.float32)[:, None]
    in_features = scaled.shape[1]
    if in_features % group:
        raise ValueError(f"int8 in_features {in_features} not divisible by group {group}")
    # Apply R on the input axis in groups: W ≈ scaled @ R (R symmetric orthonormal).
    chunks = scaled.reshape(scaled.shape[0], in_features // group, group) @ rot
    return chunks.reshape(scaled.shape[0], in_features).to(torch.bfloat16)


def densify_lowrank(down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``W = up @ down`` with shapes ``[out, rank]`` and ``[rank, in]``."""
    return (up.to(torch.float32) @ down.to(torch.float32)).to(torch.bfloat16)


class HyperFlowH3FlashDiTModel(MiniMaxH3DiTModel):
    """MiniMax-H3 DiT with HyperFlow dropped blocks and two-time embeddings."""

    def __init__(
        self,
        od_config,
        quant_config=None,
        *,
        diffusers_weights: bool | None = True,
        attention_cls: type[MiniMaxH3Attention] = MiniMaxH3Attention,
    ) -> None:
        # HyperFlow checkpoints are Diffusers-layout INT8/low-rank exports.
        super().__init__(
            od_config,
            quant_config=quant_config,
            diffusers_weights=True,
            attention_cls=attention_cls,
        )
        tf_config = od_config.tf_model_config
        config_mapping = tf_config.to_dict() if hasattr(tf_config, "to_dict") else dict(tf_config)
        extras = getattr(od_config, "extras", None) or {}
        if not isinstance(extras, Mapping):
            extras = {}
        partition = str(extras.get("hyperflow_partition") or "fl2va").lower()
        raw_dropped = config_mapping.get("dropped_blocks")
        if raw_dropped is None:
            raw_dropped = extras.get("hyperflow_dropped_blocks") or DEFAULT_DROPPED_BLOCKS.get(partition, ())
        self._hyperflow_dropped_blocks = frozenset(int(x) for x in raw_dropped)
        self._hyperflow_gate = float(config_mapping.get("hyperflow_gate", extras.get("hyperflow_gate", 0.25)))
        quant = config_mapping.get("quantization") or extras.get("hyperflow_quantization") or {}
        self._hyperflow_int8_group = int(quant.get("group", 64)) if isinstance(quant, Mapping) else 64
        self._hyperflow_int8_linears = frozenset(quant.get("linears", ())) if isinstance(quant, Mapping) else frozenset()
        self._hyperflow_lowrank_ranks = {
            str(k): int(v) for k, v in (config_mapping.get("lowrank_ranks") or {}).items()
        }

        for index in self._hyperflow_dropped_blocks:
            if 0 <= index < len(self.blocks):
                self.blocks[index] = DroppedBlock()

        # Twin of ``time_embedder`` for the step endpoint ``r``.
        self.endpoint_time_embedder = MiniMaxH3TimeEmbedder(self.arch, prefix="endpoint_time_embedder")
        logger.info(
            "HyperFlow H3 Flash DiT: dropped_blocks=%s gate=%s densify_int8=%s densify_lowrank=%s",
            sorted(self._hyperflow_dropped_blocks),
            self._hyperflow_gate,
            bool(self._hyperflow_int8_linears),
            bool(self._hyperflow_lowrank_ranks),
        )

    def _embed(self, **kwargs: Any):
        embeddings, t_emb = super()._embed(**kwargs)
        endpoint = getattr(self, "_pending_endpoint_timesteps", None)
        if endpoint is None:
            # Official HyperFlow has no single-time mode; denoise must pass endpoints.
            raise ValueError(
                "HyperFlow H3 Flash needs unique_endpoint_timesteps (r of every (t, r) pair); "
                "got none — the few-step distilled model is undefined with r = t"
            )
        unique_timesteps = kwargs["unique_timesteps"].view(-1)
        endpoint = endpoint.view(-1).to(device=unique_timesteps.device)
        if endpoint.shape != unique_timesteps.shape:
            raise ValueError(
                f"unique_endpoint_timesteps {tuple(endpoint.shape)} != unique_timesteps {tuple(unique_timesteps.shape)}"
            )
        r_emb = self.endpoint_time_embedder(endpoint)
        t_emb = t_emb + self._hyperflow_gate * (r_emb - t_emb)
        return embeddings, t_emb

    def forward(self, *, pdd_step: int | None = None, **kwargs: Any):
        # Accept HyperFlow's endpoint tensor without breaking the H3 kwarg gate.
        endpoint = kwargs.pop("unique_endpoint_timesteps", None)
        if endpoint is None:
            raise ValueError(
                "HyperFlow H3 Flash needs unique_endpoint_timesteps (r of every (t, r) pair); "
                "run through MiniMaxH3ModelBatch HyperFlow path / flash_blocks"
            )
        self._pending_endpoint_timesteps = endpoint
        try:
            return super().forward(pdd_step=pdd_step, **kwargs)
        finally:
            if hasattr(self, "_pending_endpoint_timesteps"):
                delattr(self, "_pending_endpoint_timesteps")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load HyperFlow Diffusers weights, densifying INT8 / low-rank tensors."""
        pending_int8: dict[str, dict[str, torch.Tensor]] = {}
        pending_lowrank: dict[str, dict[str, torch.Tensor]] = {}
        dense_stream: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            block_idx = _diffusers_block_index(name)
            if block_idx is not None and block_idx in self._hyperflow_dropped_blocks:
                continue
            if name.startswith("endpoint_time_embedder."):
                # Loaded via standard path after renaming to Omni time-embedder layout.
                dense_stream.append((name, tensor))
                continue
            if name.endswith(".qweight") or name.endswith(".wscale"):
                module, _, kind = name.rpartition(".")
                pending_int8.setdefault(module, {})[kind] = tensor
                continue
            if name.endswith(".down.weight") or name.endswith(".up.weight"):
                # ``...to_q.down.weight`` → module ``...to_q``, kind down/up
                module_kind, _, weight = name.rpartition(".")
                module, _, kind = module_kind.rpartition(".")
                if weight != "weight" or kind not in {"down", "up"}:
                    raise ValueError(f"unsupported HyperFlow low-rank key: {name}")
                pending_lowrank.setdefault(module, {})[kind] = tensor
                continue
            dense_stream.append((name, tensor))

        for module, parts in pending_lowrank.items():
            if "down" not in parts or "up" not in parts:
                raise ValueError(f"incomplete HyperFlow low-rank pair for {module}: {sorted(parts)}")
            dense = densify_lowrank(parts["down"], parts["up"])
            dense_stream.append((f"{module}.weight", dense))

        for module, parts in pending_int8.items():
            if "qweight" not in parts or "wscale" not in parts:
                raise ValueError(f"incomplete HyperFlow INT8 pair for {module}: {sorted(parts)}")
            dense = densify_int8(parts["qweight"], parts["wscale"], self._hyperflow_int8_group)
            dense_stream.append((f"{module}.weight", dense))

        # Map endpoint_time_embedder Diffusers keys onto Omni TimeEmbedder params.
        remapped: list[tuple[str, torch.Tensor]] = []
        endpoint_alias = {
            "endpoint_time_embedder.linear_1.weight": "endpoint_time_embedder.proj_in.weight",
            "endpoint_time_embedder.linear_1.bias": "endpoint_time_embedder.proj_in.bias",
            "endpoint_time_embedder.linear_2.weight": "endpoint_time_embedder.proj_out.weight",
            "endpoint_time_embedder.linear_2.bias": "endpoint_time_embedder.proj_out.bias",
        }
        endpoint_tensors: list[tuple[str, torch.Tensor]] = []
        for name, tensor in dense_stream:
            aliased = endpoint_alias.get(name, name)
            if aliased.startswith("endpoint_time_embedder."):
                # Not in MiniMax Diffusers native-target map; load after super().
                endpoint_tensors.append((aliased, tensor))
            else:
                remapped.append((aliased, tensor))

        # Instrument param loaders so USP/TP shape mismatches name the target.
        params = dict(self.named_parameters())
        originals: list[tuple[Any, Any]] = []
        for pname, param in params.items():
            orig = getattr(param, "weight_loader", None)
            if orig is None:
                continue

            def _make_loader(p: Any = param, n: str = pname, o: Any = orig):
                def _loader(param_arg: Any, loaded_weight: torch.Tensor, *args: Any, **kwargs: Any):
                    try:
                        return o(param_arg, loaded_weight, *args, **kwargs)
                    except AssertionError as exc:
                        raise RuntimeError(
                            f"HyperFlow weight shape mismatch loading into {n}: "
                            f"param={tuple(p.data.shape)} loaded={tuple(loaded_weight.shape)} "
                            f"args={args}"
                        ) from exc

                return _loader

            originals.append((param, orig))
            param.weight_loader = _make_loader()
        try:
            loaded = super().load_weights(remapped)
        finally:
            for param, orig in originals:
                param.weight_loader = orig
        for name, tensor in endpoint_tensors:
            if name not in params:
                raise ValueError(f"HyperFlow endpoint weight has no parameter: {name}")
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
            loaded.add(name)
        return loaded


__all__ = ["DroppedBlock", "HyperFlowH3FlashDiTModel", "densify_int8", "densify_lowrank"]
