# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph capture for Qwen-Image-2.1 KV-cache decode steps.

Decode (denoising steps 1..N) recomputes only the target image's tokens: the
shape is identical every step and the timestep-independent prefix K/V come from
the cross-step KV cache. That is the textbook CUDA-graph case, except the
default cache stores prefix K/V in freshly allocated tensors, so a captured
graph would bind addresses that the next request (or a step-mode cache merge)
no longer owns.

Requests retain ownership of their prefix K/V. Decode refreshes separate
fixed-address graph buffers when the active prefix tensors change. Graphs
are keyed by branch, batch, prefix length, complete image layout,
dtype, device and backend, so equal token counts with different RoPE layouts stay separate.

Each key admits a single live owner, tracked by a weak reference to the
registering request's prefix K tensor. A second in-flight request whose cache
aliases the same key is refused graph decode and stays eager until the
owner's cache is released, so shared static buffers are never overwritten
under a concurrent owner.

Memory note: graph entries own an additional copy of the prefix K/V. Decode
attention concatenates that prefix with target K/V inside the captured region;
transient tensors use the graph memory pool.

The model runner regionally torch.compile's the transformer blocks even when
decode graphs are enabled, so capture records the compiled (fused) kernels;
inductor's own cudagraphs stay off so the two graph layers never stack.

Fallbacks (all logged, all eager): CUDA unavailable, model-level CPU offload
without persistent weight staging, sequence/ring/tensor parallelism,
HSDP/offload hooks, KV-cache quantization, dynamic LoRA wrappers, padded text
masks (the masked attention path branches on mask contents, which cannot be
captured), unknown graph key at decode, a second live request aliasing an
owned graph key, and capture failure.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        QwenImage21Transformer2DModel,
    )

logger = init_logger(__name__)


def _dynamic_lora_wrappers_present(module: torch.nn.Module) -> bool:
    """True once ``DiffusionLoRAManager`` has wrapped layers under ``module``.

    A captured graph records the layers and branches that ran at capture time;
    binding or rescaling an adapter afterwards changes the math without
    changing the shapes the graph was keyed on, so graphs and dynamic LoRA
    must not mix. Same reasoning as ``sensenova_u1.paged_decode``.
    """
    try:
        from vllm.lora.layers import BaseLayerWithLoRA
    except ImportError:  # pragma: no cover - depends on the wheel
        return False
    return any(isinstance(m, BaseLayerWithLoRA) for m in module.modules())


class QwenImage21DecodeGraphEntry:
    """Static state for one captured decode graph.

    Buffers are created outside inference mode so replay-side ``copy_`` is
    legal regardless of the caller's grad context. ``graph`` is None until the
    first decode step captures it. Failed entries are replaced by a ``None``
    marker in the manager so their buffers can be released.
    """

    def __init__(
        self,
        *,
        branch: str,
        batch_size: int,
        prefix_len: int,
        target_tokens: int,
        in_channels: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_shapes: list[torch.Size],
        kv_dtype: torch.dtype,
        freqs: torch.Tensor,
    ):
        self.branch = branch
        self.target_tokens = target_tokens
        self.prefix_len = prefix_len
        with torch.inference_mode(False):
            self.hidden = torch.zeros(batch_size, target_tokens, in_channels, dtype=dtype, device=device)
            self.timestep = torch.zeros(batch_size, dtype=dtype, device=device)
            self.k = [torch.zeros(shape, dtype=kv_dtype, device=device) for shape in kv_shapes]
            self.v = [torch.zeros(shape, dtype=kv_dtype, device=device) for shape in kv_shapes]
            self.target_token_mask = torch.ones(target_tokens, dtype=torch.bool, device=device)
            # RoPE frequencies are layout-dependent but step-independent. Clone
            # into a normal (non-inference) tensor: the prefill runs under
            # inference_mode, and the captured body takes views of this tensor
            # outside it, which is forbidden for inference tensors.
            self.freqs = freqs.clone()
            self.attn_metadata = None
        # Per-block cache dicts with the same shape as the legacy protocol.
        self.block_caches = [{branch: {"key": self.k[i], "value": self.v[i]}} for i in range(len(kv_shapes))]
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: torch.Tensor | None = None
        self.captures = 0
        self.prefix_sources: list[tuple[weakref.ReferenceType[torch.Tensor], int | None]] = []

    def refresh_prefix(self, kv_cache: list[dict[str, dict[str, torch.Tensor]]]) -> None:
        sources = [block[self.branch][name] for block in kv_cache for name in ("key", "value")]
        versions = [None if tensor.is_inference() else tensor._version for tensor in sources]
        if len(sources) == len(self.prefix_sources) and all(
            previous() is tensor and version == current_version
            for (previous, version), tensor, current_version in zip(self.prefix_sources, sources, versions)
        ):
            return
        # Prefill replaces inference-mode tensors; decode treats them as immutable.
        # Weak references avoid retaining completed requests or mistaking a reused
        # allocation for the same prefix. Normal tensors also track in-place writes.
        self.prefix_sources = []
        for i, block in enumerate(kv_cache):
            self.k[i].copy_(block[self.branch]["key"])
            self.v[i].copy_(block[self.branch]["value"])
        self.prefix_sources = [(weakref.ref(tensor), version) for tensor, version in zip(sources, versions)]

    def capture(self, body) -> None:
        """Warm up on a side stream, then capture one decode step.

        Warm-up runs the same computation it will record, allocating cuBLAS
        workspaces and any lazy buffers outside the capture. Decode writes
        nothing to the cache (prefix K/V are read-only), so warm-up is
        idempotent.
        """
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(side), torch.inference_mode(False), torch.no_grad():
                for _ in range(2):
                    body()
        finally:
            torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        # The platform-wide pool is shared with every other captured path in
        # the tree; entries only keep their *output* alive, whose tiny size
        # makes reuse safe because the caller clones it before any other graph
        # on the pool replays.
        with torch.inference_mode(False), torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                self.output = body()
        self.graph = graph
        self.captures += 1


class QwenImage21DecodeGraphManager:
    """Owns the static KV buffers and captured graphs of one transformer."""

    def __init__(self, model: QwenImage21Transformer2DModel, max_entries: int = 8, model_level_offload: bool = False):
        self.model = model
        self.max_entries = max_entries
        self.model_level_offload = model_level_offload
        self.entries: OrderedDict[tuple, QwenImage21DecodeGraphEntry | None] = OrderedDict()
        # Key -> weak reference to the prefix K tensor of the request that owns
        # the entry's static buffers. Dead references mean the owning request
        # finished and its cache was released, so the key is free to adopt.
        self._owners: dict[tuple, weakref.ReferenceType[torch.Tensor]] = {}
        self._static_eligible: bool | None = None

    # ── eligibility ──

    def _offload_reason(self) -> str | None:
        """Reason decode graphs are incompatible with model-level offload, if any.

        Model-level (sequential) offload registers its hook on the top-level
        transformer module, not on individual blocks. With persistent staging
        the hook keeps the DiT on fixed device storage and only rebinds
        ``p.data`` around each generation, and the swap completes in
        ``pre_forward`` before capture or replay begin, so captured weight
        pointers stay valid. Without staging every swap allocates fresh
        storage and replay would read stale pointers, so capture stays
        disabled and decode falls back to eager.
        """
        registry = getattr(self.model, "_hook_registry", None)
        hooks = registry._hooks if registry is not None else {}
        if hooks:
            from vllm_omni.diffusion.offloader.sequential_backend import (
                SequentialOffloadHook,
                sequential_offload_staging_active,
            )

            if set(hooks) <= {SequentialOffloadHook._HOOK_NAME} and sequential_offload_staging_active(self.model):
                return None
            return "model module carries offload/cache hooks"
        if self.model_level_offload:
            return "model-level CPU offload without persistent staging swaps weight storage after capture"
        return None

    def _check_static_eligibility(self) -> bool:
        """One-time checks that cannot change after weights are loaded."""
        model = self.model
        reason = self._offload_reason()
        param = next(model.parameters(), None)
        if reason is None and (param is None or param.device.type != "cuda"):
            reason = f"model is not on CUDA (device={None if param is None else param.device})"
        elif reason is None:
            parallel_config = getattr(model, "parallel_config", None)
            if parallel_config is not None:
                sp = getattr(parallel_config, "sequence_parallel_size", None) or 1
                if sp > 1 or getattr(parallel_config, "ring_degree", 1) > 1:
                    reason = "sequence/ring parallelism changes decode shapes per rank"
                elif getattr(parallel_config, "use_hsdp", False):
                    reason = "HSDP wraps blocks with gather hooks outside the captured region"
            if reason is None:
                from vllm.distributed import get_tensor_model_parallel_world_size

                if get_tensor_model_parallel_world_size() > 1:
                    reason = "tensor parallelism runs collectives inside the captured region"
            if reason is None:
                for block in model.transformer_blocks:
                    if hasattr(block, "_hook_registry") or hasattr(block, "_omni_original_forward"):
                        reason = "transformer blocks carry offload/cache hooks"
                        break
            if reason is None and model.transformer_blocks:
                attn_layer = model.transformer_blocks[0].attn.attn
                if getattr(attn_layer, "_kv_cache_dtype", None) is not None:
                    reason = "KV-cache quantization is enabled"

        if reason is not None:
            logger.warning_once("Qwen-Image-2.1 CUDA graph decode disabled: %s. Falling back to eager decode.", reason)
            return False
        return True

    def eligible(self) -> bool:
        if self._static_eligible is None:
            self._static_eligible = self._check_static_eligibility()
        return self._static_eligible

    def _backend_name(self) -> str:
        backend = self.model.transformer_blocks[0].attn.attn.attn_backend
        return backend.get_name() if backend is not None else "custom"

    # ── prefill registration ──

    def register_prefill(
        self,
        *,
        kv_cache: list[dict[str, dict[str, torch.Tensor]]],
        cache_branch: str,
        prefix_len: int,
        img_shapes: list[tuple[int, int, int]],
        target_freqs: torch.Tensor,
        joint_key_valid: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        """Prepare graph buffers without changing the request's prefix K/V.

        The first in-flight request to register a key owns its static buffers;
        registrations for the same key from another live request are refused
        so their decode stays eager until the owner completes.
        """
        if not self.eligible() or torch.compiler.is_compiling():
            return
        if _dynamic_lora_wrappers_present(self.model):
            if self.entries:
                logger.warning_once(
                    "Qwen-Image-2.1 CUDA graph decode: LoRA wrappers appeared; "
                    "dropping captured graphs and falling back to eager decode."
                )
                self.entries.clear()
            return
        if joint_key_valid is not None and not bool(joint_key_valid.all()):
            # The masked decode path (varlen-unpack or 4D mask) branches on mask
            # contents and cannot be captured; this request stays eager.
            logger.warning_once(
                "Qwen-Image-2.1 CUDA graph decode: padded text mask present; this request falls back to eager decode."
            )
            return

        first = kv_cache[0].get(cache_branch)
        if first is None or "key" not in first:
            return
        if any(name.endswith("_scale") for name in first):
            # Quantized (FP8) prefix storage: the static K/V buffers assume a single
            # dtype and the captured body would not see the dequant scales (a bf16 K +
            # fp8 V mix would even copy_ fp8 into bf16 buffers, silently dropping the
            # scales). Keep quantized-cache requests on the eager decode path.
            logger.warning_once(
                "Qwen-Image-2.1 CUDA graph decode: quantized prefix KV cache "
                "(prefix_kv_cache_dtype) is not graph-capturable; falling back to eager decode."
            )
            return
        batch_size = first["key"].shape[0]
        key = (
            cache_branch,
            batch_size,
            prefix_len,
            tuple(tuple(shape) for shape in img_shapes),
            dtype,
            first["key"].device.index,
            self._backend_name(),
        )

        if key not in self.entries:
            while len(self.entries) >= self.max_entries:
                evicted_key, evicted_entry = self.entries.popitem(last=False)
                self._owners.pop(evicted_key, None)
                del evicted_entry
                logger.debug("Evicting decode graph entry %s (max_entries=%d)", evicted_key, self.max_entries)
            allocation_error = None
            try:
                self.entries[key] = QwenImage21DecodeGraphEntry(
                    branch=cache_branch,
                    batch_size=batch_size,
                    prefix_len=prefix_len,
                    target_tokens=target_freqs.shape[0],
                    in_channels=self.model.in_channels,
                    dtype=dtype,
                    device=first["key"].device,
                    kv_shapes=[block_cache[cache_branch]["key"].shape for block_cache in kv_cache],
                    kv_dtype=first["key"].dtype,
                    freqs=target_freqs,
                )
            except torch.OutOfMemoryError as exc:
                allocation_error = str(exc)
            if allocation_error is not None:
                # Keep a bounded failure marker without retaining tensors or the exception traceback.
                self.entries[key] = None
                current_omni_platform.empty_cache()
                logger.warning(
                    "Qwen-Image-2.1 CUDA graph allocation failed for key=%s: %s. Falling back to eager decode.",
                    key,
                    allocation_error,
                )
                return
        else:
            self.entries.move_to_end(key)

        entry = self.entries[key]
        if entry is not None:
            owner = self._owners.get(key)
            owner_prefix = owner() if owner is not None else None
            if owner_prefix is not None and owner_prefix is not first["key"]:
                # Another in-flight request still owns this entry's static
                # buffers; refreshing them would overwrite that request's
                # prefix mid-generation. This request decodes eagerly until
                # the owner's cache is released and the weak reference dies.
                logger.warning_once(
                    "Qwen-Image-2.1 CUDA graph decode: entry key=%s is owned by another in-flight "
                    "request; this request falls back to eager decode.",
                    key,
                )
                return
            self._owners[key] = weakref.ref(first["key"])
            entry.prefix_sources = []

        logger.debug(
            "Registered decode graph entry key=%s (entries=%d, captured=%d)",
            key,
            len(self.entries),
            sum(1 for e in self.entries.values() if e is not None and e.graph is not None),
        )

    # ── decode replay ──

    def try_decode(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: list[dict[str, dict[str, torch.Tensor]]],
        cache_branch: str,
        img_shapes: list[tuple[int, int, int]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Replay the captured decode step, or return None to fall back to eager."""
        if not self.eligible() or torch.compiler.is_compiling():
            return None
        if _dynamic_lora_wrappers_present(self.model):
            self.clear()
            return None
        if encoder_hidden_states_mask is not None:
            text_positions = ~img_mask[0, : encoder_hidden_states_mask.shape[1]]
            if not bool(encoder_hidden_states_mask[:, text_positions].all()):
                return None
        first = kv_cache[0][cache_branch]
        cached_key = first["key"]
        if cached_key.device.type != "cuda":
            # Registration is device-agnostic (unit tests exercise the buffer
            # machinery on CPU), but capture/replay is CUDA-only: an empty
            # graph over CPU ops would replay as a stale no-op.
            return None
        key = (
            cache_branch,
            cached_key.shape[0],
            cached_key.shape[1],
            tuple(tuple(shape) for shape in img_shapes),
            hidden_states.dtype,
            cached_key.device.index,
            self._backend_name(),
        )
        entry = self.entries.get(key)
        if entry is None:
            return None
        self.entries.move_to_end(key)

        owner = self._owners.get(key)
        owner_prefix = owner() if owner is not None else None
        if owner_prefix is not None and owner_prefix is not cached_key:
            # The entry is owned by a different in-flight request (see
            # register_prefill); this request stays on the eager decode path
            # until the owner's cache is released.
            return None

        entry.hidden.copy_(hidden_states[:, -entry.target_tokens :])
        entry.timestep.copy_(timestep)
        # Graph buffers are scratch space; request caches must never alias them.
        entry.refresh_prefix(kv_cache)

        if entry.graph is None:
            capture_error = None
            try:
                entry.capture(lambda: self.model._decode_graph_forward(entry))
            except Exception as exc:
                capture_error = str(exc)
            if capture_error is not None:
                self.entries[key] = None
                del entry
                current_omni_platform.empty_cache()
                logger.warning(
                    "Qwen-Image-2.1 CUDA graph capture failed for key=%s: %s. "
                    "Falling back to eager decode for this shape.",
                    key,
                    capture_error,
                )
                return None
            logger.info("Captured Qwen-Image-2.1 decode graph for key=%s", key)
        entry.graph.replay()
        assert entry.output is not None
        # Clone before returning: the output lives in the shared graph pool and
        # the next replay (of this or another graph on the pool) overwrites it.
        return entry.output.clone()

    def clear(self) -> None:
        """Drop all entries, freeing buffers and captured graphs."""
        self.entries.clear()
        self._owners.clear()
