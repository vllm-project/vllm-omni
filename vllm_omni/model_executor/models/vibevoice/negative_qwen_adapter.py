# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen2 execution adapter for the named KV branch graph path.

References the original ``Qwen2Model`` compute modules (no weight copy) and
consumes explicit layer inputs so the forward pass is a pure function of
tensors — capturable by ``torch.compile`` and CUDA Graph.

Layer mapping uses ``attn.layer_name`` (the registered attention prefix) to
match branch layer names, but execution order follows
``language_model.layers`` (the model's original decoder order).  This prevents
registry ordering from changing Qwen execution semantics.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.models.qwen2 import Qwen2Model

from vllm_omni.worker.named_kv import ops as _named_kv_ops  # noqa: F401 -- register custom op
from vllm_omni.worker.named_kv.flash_attention import UnsupportedNamedKVGraphError


class Qwen2KVBranchAdapter:
    """Reference existing Qwen2 compute modules; consume explicit layer inputs."""

    def __init__(
        self,
        language_model: Qwen2Model,
        hidden_size: int,
    ) -> None:
        self.language_model = language_model
        self.hidden_size = int(hidden_size)
        self._layer_pairs: list[tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]] = []
        self._scheduler_metadata: list[torch.Tensor | None] = []
        self._scheduler_metadata_sizes: dict[int, int] = {}
        self._block_size = 16  # default; updated in bind_kv_caches
        self._max_batch_size = 4  # default; updated in bind_kv_caches
        self._closed = False

    def bind_kv_caches(
        self,
        *,
        branch_layer_names: tuple[str, ...],
        k_caches: list[torch.Tensor],
        v_caches: list[torch.Tensor],
    ) -> None:
        """Bind K/V caches to model layers, verifying alignment.

        Execution order follows ``language_model.layers`` (model's original
        order), NOT ``branch_layer_names`` order.  This prevents registry
        ordering from changing Qwen execution semantics.
        """
        if self._closed:
            raise RuntimeError("Qwen2KVBranchAdapter is closed")
        if self._layer_pairs:
            raise RuntimeError("KV caches already bound")
        if not branch_layer_names or len(set(branch_layer_names)) != len(branch_layer_names):
            raise ValueError("Branch registry must contain unique, non-empty layer names")
        if len(k_caches) != len(branch_layer_names):
            raise ValueError(f"K/V cache count {len(k_caches)} != layer count {len(branch_layer_names)}")
        if len(v_caches) != len(branch_layer_names):
            raise ValueError(f"V cache count {len(v_caches)} != layer count {len(branch_layer_names)}")

        # Build name → cache mapping.
        name_to_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, name in enumerate(branch_layer_names):
            name_to_cache[name] = (k_caches[i], v_caches[i])

        # Map by model's original layer order.
        layer_pairs = []
        matched_names: set[str] = set()
        for layer in self.language_model.layers:
            attn = layer.self_attn
            layer_name = attn.attn.layer_name  # Attention.layer_name = prefix
            if layer_name not in name_to_cache:
                raise ValueError(f"Model layer {layer_name!r} not in branch registry")
            k_cache, v_cache = name_to_cache[layer_name]
            inner = attn.attn  # Attention instance
            if layer_name in matched_names:
                raise ValueError(f"Duplicate model attention registration: {layer_name}")
            # Verify the explicit four-dimensional K/V ABI before indexing.
            if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
                raise ValueError(f"K/V must have matching four-dimensional shapes for {layer_name}")
            if k_cache.shape[1] != k_caches[0].shape[1]:
                raise ValueError(f"KV block size mismatch for {layer_name}")
            # Verify head/dtype/device compatibility.
            if k_cache.shape[-2] != inner.num_kv_heads:
                raise ValueError(f"KV head mismatch for {layer_name}: {k_cache.shape[-2]} != {inner.num_kv_heads}")
            if k_cache.shape[-1] != inner.head_size:
                raise ValueError(f"Head dim mismatch for {layer_name}: {k_cache.shape[-1]} != {inner.head_size}")
            if k_cache.dtype != v_cache.dtype:
                raise ValueError(f"K/V dtype mismatch for {layer_name}")
            if k_cache.device != v_cache.device:
                raise ValueError(f"K/V device mismatch for {layer_name}")
            layer_pairs.append((layer, k_cache, v_cache, self._build_layer_config(attn)))
            matched_names.add(layer_name)

        # Verify completeness.
        if len(matched_names) != len(branch_layer_names):
            missing = set(branch_layer_names) - matched_names
            raise ValueError(f"Branch layers not found in model: {missing}")
        if len(layer_pairs) != len(self.language_model.layers):
            raise ValueError(
                f"Layer count mismatch: {len(layer_pairs)} pairs vs {len(self.language_model.layers)} model layers"
            )

        self._layer_pairs = layer_pairs
        # Pre-allocate scheduler_metadata workspace for FA3 AOT scheduling.
        # This is required for CUDA Graph capture.
        # In eager mode, scheduler_metadata is None (computed per-call by FA).
        self._block_size = k_caches[0].shape[1]
        self._scheduler_metadata: list[torch.Tensor | None] = [None for _ in range(len(self._layer_pairs))]

    def init_graph_workspace(self, max_batch_size: int) -> None:
        """Pre-allocate FA3 scheduler_metadata workspace for graph capture.

        Called by the executor before warmup.  Allocates a fixed-address
        workspace tensor per layer, matching vLLM's FA backend pattern.
        """
        from vllm.utils.math_utils import round_up

        self._max_batch_size = int(max_batch_size)
        size = 1 + round_up(self._max_batch_size, 4) * 4
        device = self._layer_pairs[0][1].device if self._layer_pairs else torch.device("cpu")
        self._scheduler_metadata = [torch.zeros(size, dtype=torch.int32, device=device) for _ in self._layer_pairs]

    def _build_layer_config(self, attn: Any) -> dict[str, Any]:
        """Capture per-layer FA parameters at bind time."""
        inner = attn.attn  # Attention instance
        fa_version = inner.impl.vllm_flash_attn_version
        if fa_version != 3:
            raise UnsupportedNamedKVGraphError(f"Named KV graph currently verifies only FA3, got FA{fa_version}")
        return {
            "fa_version": fa_version,
            "kv_cache_dtype": inner.kv_cache_dtype,
            "k_scale": inner._k_scale,
            "v_scale": inner._v_scale,
            "softmax_scale": inner.impl.scale,
            "num_kv_heads": inner.num_kv_heads,
            "head_size": inner.head_size,
            "num_heads_q": inner.num_heads,
        }

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        max_seq_len: int,
    ) -> torch.Tensor:
        """Run negative Qwen2 forward with explicit layer inputs.

        Preserves full Qwen decoder semantics:
        - residual is cross-layer (passed through each layer)
        - input_layernorm, post_attention_layernorm, final norm
        - QKV projection, QK norm, RoPE, MLP
        """
        if self._closed:
            raise RuntimeError("Qwen2KVBranchAdapter is closed")
        if not self._layer_pairs:
            raise RuntimeError("KV caches not bound; call bind_kv_caches first")

        hidden_states = embeddings
        residual: torch.Tensor | None = None

        for layer_idx, (layer, k_cache, v_cache, lc) in enumerate(self._layer_pairs):
            inner = layer.self_attn  # Qwen2Attention instance
            # --- Self Attention ---
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)

            qkv, _ = inner.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    inner.q_size,
                    inner.kv_size,
                    inner.kv_size,
                ],
                dim=-1,
            )

            # QK normalization if enabled.
            if inner.qk_norm:
                total_tokens = q.shape[0]
                q = q.view(total_tokens, inner.num_heads, inner.head_dim)
                k = k.view(total_tokens, inner.num_kv_heads, inner.head_dim)
                q = inner.q_norm(q)
                k = inner.k_norm(k)
                q = q.view(total_tokens, inner.q_size)
                k = k.view(total_tokens, inner.kv_size)

            # RoPE.
            q, k = inner.rotary_emb(positions, q, k)

            # Named KV branch attention via custom op.
            q3 = q.view(-1, inner.num_heads, inner.head_dim)
            k3 = k.view(-1, inner.num_kv_heads, inner.head_dim)
            v3 = v.view(-1, inner.num_kv_heads, inner.head_dim)
            out3 = torch.empty_like(q3)

            torch.ops.vllm_omni.named_kv_branch_attention(
                q3,
                k3,
                v3,
                out3,
                k_cache,
                v_cache,
                slot_mapping,
                block_table,
                query_start_loc,
                seq_lens,
                1,
                max_seq_len,
                lc["kv_cache_dtype"],
                lc["k_scale"],
                lc["v_scale"],
                lc["softmax_scale"],
                lc["num_kv_heads"],
                lc["head_size"],
                lc["num_heads_q"],
                self._block_size,
                (
                    self._scheduler_metadata[layer_idx][: self._scheduler_metadata_sizes[embeddings.shape[0]]]
                    if self._scheduler_metadata[layer_idx] is not None
                    else None
                ),
                lc["fa_version"],
            )

            attn_output = out3.view(-1, inner.q_size)
            output, _ = inner.o_proj(attn_output)

            # --- MLP ---
            hidden_states, residual = layer.post_attention_layernorm(output, residual)
            hidden_states = layer.mlp(hidden_states)

        # Final norm.
        hidden_states, _ = self.language_model.norm(hidden_states, residual)
        return hidden_states

    def close(self) -> None:
        """Release K/V cache view references (not shared Qwen weights)."""
        self._layer_pairs.clear()
        self._scheduler_metadata.clear()
        self._scheduler_metadata_sizes.clear()
        self._closed = True


__all__ = ["Qwen2KVBranchAdapter"]
