# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA graph capture, cache, and replay for VoxCPM2."""

from __future__ import annotations

import copy
import dataclasses
import logging
import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.worker.runner_assisted_metadata import RunnerAssistedFullAttentionMetadataRequest

from .model_state import VoxCPM2BatchSlots
from .runtime_config import _VoxCPM2RuntimeConfig

if TYPE_CHECKING:
    from .model_state import VoxCPM2RequestState
    from .voxcpm2_talker import VoxCPM2TalkerForConditionalGeneration

logger = init_logger(__name__)


class _CFMBufferManager:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        feat_dim: int,
        patch_size: int,
        dit_hidden_size: int,
        max_batch_size: int = 1,
        sway_sampling_coef: float = 1.0,
    ):
        n = 2 * max_batch_size  # CFG doubles the batch
        self.x_in = torch.zeros(n, feat_dim, patch_size, device=device, dtype=dtype)
        self.mu_in = torch.zeros(n, dit_hidden_size, device=device, dtype=dtype)
        self.t_in = torch.zeros(n, device=device, dtype=dtype)
        self.dt_in = torch.zeros(n, device=device, dtype=dtype)
        self.cond_in = torch.zeros(n, feat_dim, patch_size, device=device, dtype=dtype)
        self.noise = torch.zeros(max_batch_size, feat_dim, patch_size, device=device, dtype=dtype)
        self.x_work = torch.zeros_like(self.noise)
        self._zeroed_mu_negative_batch: int | None = None
        self._sway_coef = sway_sampling_coef
        self._device = device
        self._dtype = dtype
        self.t_span_10 = self._make_t_span(10)

    def _make_t_span(self, n: int) -> torch.Tensor:
        t = torch.linspace(1, 0, n + 1, device=self._device, dtype=self._dtype)
        return t + self._sway_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

    def get_t_span(self, n: int) -> torch.Tensor:
        return self.t_span_10 if n == 10 else self._make_t_span(n)


class _AttentionMetadataLike(Protocol):
    scheduler_metadata: Any


class _ForwardContextLike(Protocol):
    attn_metadata: dict[str, _AttentionMetadataLike] | Any


@dataclasses.dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    input_embeds: torch.Tensor
    positions: torch.Tensor
    output: torch.Tensor


@dataclasses.dataclass
class _CapturedVAEGraph:
    graph: torch.cuda.CUDAGraph
    input_feat: torch.Tensor
    output: torch.Tensor


@dataclasses.dataclass
class _CapturedCFMGraph:
    graph: torch.cuda.CUDAGraph
    mu: torch.Tensor
    cond: torch.Tensor
    noise: torch.Tensor
    output: torch.Tensor
    buffers: _CFMBufferManager


@dataclasses.dataclass
class _CapturedUnifiedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    batch_size: int
    # Static input buffers (copied before replay)
    input_embeds: torch.Tensor
    positions: torch.Tensor
    prev_feat_embed: torch.Tensor
    prefix_feat_cond: torch.Tensor
    # Static output buffers (read after replay)
    next_feat_embed: torch.Tensor
    cfm_output: torch.Tensor
    lm_hidden: torch.Tensor
    # Internal buffers for CFM solver
    cfm_buffers: _CFMBufferManager
    cfm_noise: torch.Tensor
    text_mask: torch.Tensor | None = None
    last_token_indices: torch.Tensor | None = None
    patch_indices: torch.Tensor | None = None
    scaffold_hidden: torch.Tensor | None = None
    pcm: torch.Tensor | None = None
    pcm_valid: torch.Tensor | None = None
    stop_logits: torch.Tensor | None = None
    audio_slots: torch.Tensor | None = None
    audio_rows_valid: torch.Tensor | None = None
    audio_copy_event: torch.cuda.Event | None = None


@dataclasses.dataclass
class _UnifiedDecodeGraphStats:
    captures: int = 0
    replays: int = 0
    skips: dict[str, int] = dataclasses.field(default_factory=dict)
    real_batch_sizes: dict[int, int] = dataclasses.field(default_factory=dict)
    graph_bucket_sizes: dict[int, int] = dataclasses.field(default_factory=dict)
    logged_replays: int = 0
    logged_skips: int = 0

    def record_skip(self, reason: str) -> None:
        self.skips[reason] = self.skips.get(reason, 0) + 1

    def record_replay_batch(self, *, real_batch_size: int, graph_bucket_size: int) -> None:
        self.real_batch_sizes[real_batch_size] = self.real_batch_sizes.get(real_batch_size, 0) + 1
        self.graph_bucket_sizes[graph_bucket_size] = self.graph_bucket_sizes.get(graph_bucket_size, 0) + 1

    @property
    def total_skips(self) -> int:
        return sum(self.skips.values())


class VoxCPM2CudaGraphWrapper:
    """Own VoxCPM2 graph captures and static buffers without owning model weights.

    The talker supplies model computation and request-state commits. A weak
    reference keeps this execution helper out of the model's ownership cycle.
    """

    def __init__(
        self, model: VoxCPM2TalkerForConditionalGeneration, config: _VoxCPM2RuntimeConfig, max_batch_size: int
    ) -> None:
        self.model = weakref.proxy(model)
        self._scaffold_graphs: dict[int, _CapturedGraph] = {}
        self._residual_graphs: dict[int, _CapturedGraph] = {}
        self._vae_graphs: dict[tuple[int, int], _CapturedVAEGraph] = {}
        self._cfm_graphs: dict[int, _CapturedCFMGraph] = {}
        self._unified_graphs: dict[int | tuple[int, int], _CapturedUnifiedDecodeGraph] = {}
        self._mixed_graphs: dict[tuple[int, int] | tuple[int, int, int], _CapturedUnifiedDecodeGraph] = {}
        self._unified_graph_stats = _UnifiedDecodeGraphStats()
        self._decode_graph_capture_policy = config.decode_graph_capture_policy
        self._max_cached_graphs = max_batch_size
        self._mixed_graph_cache_limit = config.mixed_graph_cache_limit
        self._unified_decode_graph_max_batch_size = config.unified_decode_graph_max_batch_size
        self._unified_graph_bucket_sizes = self._build_unified_graph_bucket_sizes(
            min(max_batch_size, self._unified_decode_graph_max_batch_size)
        )
        self._cuda_graph_warmup_steps = 0
        self._cuda_graph_warmup_threshold = 3
        self._runner_assisted_unified_decode_graph_active = False
        self._runner_assisted_unified_decode_graph_batch_size = 0

    @staticmethod
    def _voxcpm2_unwrap_torch_compile(module: nn.Module | Callable) -> nn.Module | Callable:
        """Return the eager module behind torch.compile wrappers."""
        seen: set[int] = set()
        while hasattr(module, "_orig_mod") and id(module) not in seen:
            seen.add(id(module))
            module = getattr(module, "_orig_mod")
        return module

    @staticmethod
    def _voxcpm2_compile_without_inductor_cudagraphs(
        module: nn.Module | Callable,
        *,
        mode: str | None,
        fullgraph: bool,
    ) -> Callable:
        """Compile VoxCPM2 LocDiT for capture inside VoxCPM2-owned CUDA graphs."""
        kwargs: dict[str, Any] = {
            "fullgraph": fullgraph,
            "options": {
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            },
        }
        if mode is not None:
            kwargs["mode"] = mode
        return torch.compile(module, **kwargs)

    @staticmethod
    def _raise_if_cuda_runtime_error(error: BaseException) -> None:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise error
        if isinstance(error, RuntimeError):
            message = str(error).lower()
            cuda_markers = (
                "cuda error",
                "device-side assert",
                "illegal memory access",
                "out of memory",
                "cublas",
                "cudnn",
                "cufft",
            )
            if any(marker in message for marker in cuda_markers):
                raise error

    def _voxcpm2_compile_unified_capture_estimator(self, module: nn.Module | Callable) -> Callable:
        eager_module = self._voxcpm2_unwrap_torch_compile(module)
        if getattr(self, "_estimator_for_unified_capture_source", None) is eager_module and hasattr(
            self, "_estimator_for_unified_capture"
        ):
            return self._estimator_for_unified_capture
        compiled = self._voxcpm2_compile_without_inductor_cudagraphs(
            eager_module,
            mode=None,
            fullgraph=False,
        )
        compiled._compiled = True
        self._estimator_for_unified_capture_source = eager_module
        self._estimator_for_unified_capture = compiled
        return compiled

    def _voxcpm2_compile_unified_capture_feat_encoder(self, module: nn.Module | Callable) -> Callable:
        eager_module = self._voxcpm2_unwrap_torch_compile(module)
        if getattr(self, "_feat_encoder_for_unified_capture_source", None) is eager_module and hasattr(
            self, "_feat_encoder_for_unified_capture"
        ):
            return self._feat_encoder_for_unified_capture
        compiled = self._voxcpm2_compile_without_inductor_cudagraphs(
            eager_module,
            mode=None,
            fullgraph=False,
        )
        compiled._compiled = True
        self._feat_encoder_for_unified_capture_source = eager_module
        self._feat_encoder_for_unified_capture = compiled
        return compiled

    @staticmethod
    def _nullify_volatile_metadata(ctx: _ForwardContextLike) -> _ForwardContextLike:
        """Set ``scheduler_metadata`` to None on all attention layers.

        This is the only tensor FA3 reallocates each step (variable shape).
        All other metadata tensors are persistent model-runner buffers.
        Setting it to None makes FA3 use default scheduling (~0.1ms cost).
        """
        if not isinstance(ctx.attn_metadata, dict):
            return ctx

        ctx = copy.copy(ctx)
        new_meta: dict[str, Any] = {}
        for layer_name, meta in ctx.attn_metadata.items():
            if getattr(meta, "scheduler_metadata", None) is not None:
                meta = copy.copy(meta)
                meta.scheduler_metadata = None
            new_meta[layer_name] = meta
        ctx.attn_metadata = new_meta
        return ctx

    def _capture_graph(
        self,
        model: nn.Module,
        batch_size: int,
        label: str,
        is_residual: bool = False,
    ) -> _CapturedGraph:
        """Capture a CUDA Graph for *model* at *batch_size*."""
        talker = self.model

        hidden_size = talker.config.hidden_size
        dtype = talker._side_dtype
        dev = torch.device(talker._device)

        model.precompute_fused_qkv()

        g = _CapturedGraph(
            graph=torch.cuda.CUDAGraph(),
            input_embeds=torch.zeros(batch_size, hidden_size, device=dev, dtype=dtype),
            positions=torch.zeros(batch_size, device=dev, dtype=torch.long),
            output=torch.zeros(batch_size, hidden_size, device=dev, dtype=dtype),
        )

        if is_residual:
            call_kwargs = dict(positions=g.positions, inputs_embeds=g.input_embeds)
        else:
            call_kwargs = dict(input_ids=None, positions=g.positions, inputs_embeds=g.input_embeds)

        ctx = get_forward_context()
        patched_ctx = self._nullify_volatile_metadata(ctx)

        with override_forward_context(patched_ctx):
            for _ in range(3):
                _ = model(**call_kwargs)

            with torch.cuda.graph(g.graph, pool=current_platform.get_global_graph_pool()):
                g.output = model(**call_kwargs)

        logger.info("CUDA Graph captured for %s (batch_size=%d)", label, batch_size)
        return g

    def _replay_graph(
        self,
        g: _CapturedGraph,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        *,
        clone_output: bool = True,
    ) -> torch.Tensor:
        """Copy fresh inputs into static buffers, then replay.

        No metadata copy needed: persistent buffers (seq_lens, slot_mapping,
        etc.) are updated in-place by the model runner.  scheduler_metadata
        was nullified at capture time so no kernel references it.
        """
        g.input_embeds[:batch_size].copy_(inputs_embeds[:batch_size])
        g.positions[:batch_size].copy_(positions[:batch_size])
        g.graph.replay()
        output = g.output[:batch_size]
        return output.clone() if clone_output else output

    def run_scaffold(self, inputs_embeds: torch.Tensor, positions: torch.Tensor, batch_size: int) -> torch.Tensor:
        if batch_size not in self._scaffold_graphs:
            self._scaffold_graphs[batch_size] = self._capture_graph(self.model.model, batch_size, "scaffold")
        return self._replay_graph(self._scaffold_graphs[batch_size], inputs_embeds, positions, batch_size)

    def run_residual(self, inputs_embeds: torch.Tensor, positions: torch.Tensor, batch_size: int) -> torch.Tensor:
        if batch_size not in self._residual_graphs:
            self._residual_graphs[batch_size] = self._capture_graph(
                self.model.residual_model, batch_size, "residual", is_residual=True
            )
        return self._replay_graph(self._residual_graphs[batch_size], inputs_embeds, positions, batch_size)

    def run_vae(self, feat: torch.Tensor) -> torch.Tensor:
        key = (feat.shape[0], feat.shape[-1])
        graph = self._vae_graphs.get(key)
        if graph is None:
            graph = self._capture_vae_graph(feat)
            self._vae_graphs[key] = graph
        graph.input_feat.copy_(feat)
        graph.graph.replay()
        return graph.output.float()

    def run_cfm_for_state(self, state: VoxCPM2RequestState, dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        talker = self.model
        if talker._has_deterministic_cfm_noise(state) and not talker._enable_cfm_prealloc_output:
            graph = self._get_cfm_cuda_graph(dit_h, cond)
            graph.mu.copy_(dit_h)
            graph.cond.copy_(cond)
            talker._fill_deterministic_cfm_noise_for_state(state, graph.noise)
            graph.graph.replay()
            return graph.output.clone().transpose(1, 2)
        if not talker._enable_cfm_prealloc_output:
            return self._run_cfm_cuda_graph(dit_h, cond).transpose(1, 2)
        return self._run_cfm_cuda_graph_to_state_buffer(state, dit_h, cond)

    def _should_use_decode_graph(self, batch_size: int) -> bool:
        if batch_size > self._max_cached_graphs:
            return False
        policy = self._decode_graph_capture_policy
        if policy == "all":
            return True
        if policy == "power2":
            return batch_size > 0 and (batch_size & (batch_size - 1)) == 0
        if policy == "power2_or_cached":
            return (
                batch_size in self._scaffold_graphs
                or batch_size in self._residual_graphs
                or (batch_size > 0 and (batch_size & (batch_size - 1)) == 0)
            )
        return True

    @staticmethod
    def _build_unified_graph_bucket_sizes(max_size: int) -> frozenset[int]:
        if max_size <= 0:
            return frozenset()
        sizes = {1}
        size = 2
        while size <= min(max_size, 16):
            sizes.add(size)
            size <<= 1
        bucket = 24
        while bucket <= max_size:
            sizes.add(bucket)
            bucket += 8
        sizes.add(max_size)
        return frozenset(size for size in sizes if 0 < size <= max_size)

    def _select_unified_graph_bucket_size(self, batch_size: int) -> int | None:
        if (
            batch_size <= 0
            or batch_size > self._unified_decode_graph_max_batch_size
            or batch_size > self._max_cached_graphs
        ):
            return None
        return min((size for size in self._unified_graph_bucket_sizes if size >= batch_size), default=None)

    def _capture_vae_graph(self, feat: torch.Tensor) -> _CapturedVAEGraph:
        talker = self.model
        batch_size = feat.shape[0]
        num_frames = feat.shape[-1]
        input_feat = torch.zeros_like(feat)
        sr_cond = talker._get_vae_decode_sr_cond(feat.device)

        with torch.no_grad():
            for _ in range(3):
                output = talker.tts.audio_vae.decode(input_feat, sr_cond=sr_cond)

            g = _CapturedVAEGraph(
                graph=torch.cuda.CUDAGraph(),
                input_feat=input_feat,
                output=output,
            )
            with torch.cuda.graph(g.graph, pool=current_platform.get_global_graph_pool()):
                g.output = talker.tts.audio_vae.decode(g.input_feat, sr_cond=sr_cond)

        logger.info("CUDA Graph captured for AudioVAE decode (batch_size=%d, frames=%d)", batch_size, num_frames)
        return g

    def _capture_cfm_graph(
        self,
        mu: torch.Tensor | None,
        cond: torch.Tensor,
        *,
        batch_size: int | None = None,
        dit_hidden: int | None = None,
    ) -> _CapturedCFMGraph:
        talker = self.model
        if mu is not None:
            batch_size = mu.shape[0]
            dit_hidden = mu.shape[-1]
            device = mu.device
            dtype = mu.dtype
        else:
            assert batch_size is not None
            assert dit_hidden is not None
            device = cond.device
            dtype = cond.dtype
        graph_buffers = _CFMBufferManager(
            device=device,
            dtype=dtype,
            feat_dim=talker._feat_dim,
            patch_size=talker._patch_size,
            dit_hidden_size=dit_hidden,
            max_batch_size=batch_size,
        )
        static_mu = torch.zeros(batch_size, dit_hidden, device=device, dtype=dtype)
        static_cond = torch.zeros_like(cond)
        static_noise = torch.zeros(batch_size, talker._feat_dim, talker._patch_size, device=device, dtype=dtype)

        with torch.no_grad():
            for _ in range(3):
                output = talker._solve_cfm_with_noise(
                    talker.tts.feat_decoder,
                    static_mu,
                    talker._patch_size,
                    static_cond,
                    static_noise,
                    talker._inference_timesteps,
                    talker._cfg_value,
                    graph_buffers,
                    cfg_cutoff_ratio=talker._cfg_cutoff_ratio,
                )

            graph = _CapturedCFMGraph(
                graph=torch.cuda.CUDAGraph(),
                mu=static_mu,
                cond=static_cond,
                noise=static_noise,
                output=output,
                buffers=graph_buffers,
            )
            with torch.cuda.graph(graph.graph, pool=current_platform.get_global_graph_pool()):
                graph.output = talker._solve_cfm_with_noise(
                    talker.tts.feat_decoder,
                    graph.mu,
                    talker._patch_size,
                    graph.cond,
                    graph.noise,
                    talker._inference_timesteps,
                    talker._cfg_value,
                    graph.buffers,
                    cfg_cutoff_ratio=talker._cfg_cutoff_ratio,
                )

        logger.info("CUDA Graph captured for VoxCPM2 CFM solver (batch_size=%d)", batch_size)
        return graph

    def _get_cfm_cuda_graph(
        self,
        mu: torch.Tensor | None,
        cond: torch.Tensor,
        *,
        batch_size: int | None = None,
        dit_hidden: int | None = None,
    ) -> _CapturedCFMGraph:
        if mu is not None:
            graph_batch_size = mu.shape[0]
        else:
            assert batch_size is not None
            graph_batch_size = batch_size
        graph = self._cfm_graphs.get(graph_batch_size)
        if graph is None:
            graph = self._capture_cfm_graph(mu, cond, batch_size=batch_size, dit_hidden=dit_hidden)
            self._cfm_graphs[graph_batch_size] = graph
        return graph

    def _run_cfm_cuda_graph(self, mu: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        graph = self._get_cfm_cuda_graph(mu, cond)
        graph.mu.copy_(mu)
        return self._run_cfm_cuda_graph_from_static_mu(graph, cond)

    def _run_cfm_cuda_graph_from_static_mu(self, graph: _CapturedCFMGraph, cond: torch.Tensor) -> torch.Tensor:
        graph.cond.copy_(cond)
        graph.noise.normal_()
        graph.graph.replay()
        return graph.output.clone()

    def _run_cfm_cuda_graph_to_state_buffer(
        self,
        state: VoxCPM2RequestState,
        mu: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        talker = self.model
        graph = self._get_cfm_cuda_graph(mu, cond)
        graph.mu.copy_(mu)
        graph.cond.copy_(cond)
        if talker._has_deterministic_cfm_noise(state):
            talker._fill_deterministic_cfm_noise_for_state(state, graph.noise)
        else:
            graph.noise.normal_()
        graph.graph.replay()

        output_shape = graph.output.shape
        if (
            state.cfm_output_gpu is None
            or state.cfm_output_gpu.shape != output_shape
            or state.cfm_output_gpu.device != graph.output.device
            or state.cfm_output_gpu.dtype != graph.output.dtype
        ):
            state.cfm_output_gpu = torch.empty(output_shape, device=graph.output.device, dtype=graph.output.dtype)
        state.cfm_output_gpu.copy_(graph.output)
        return state.cfm_output_gpu.transpose(1, 2)

    def _capture_unified_decode_graph(
        self, batch_size: int, token_capacity: int | None = None, patch_batch_size: int | None = None
    ) -> _CapturedUnifiedDecodeGraph:
        talker = self.model
        mixed = token_capacity is not None
        token_capacity = token_capacity if mixed else batch_size
        patch_count = batch_size if patch_batch_size is None else patch_batch_size
        H = talker.config.hidden_size
        D = talker._feat_dim
        P = talker._patch_size
        dev = torch.device(talker._device)
        dtype = talker._side_dtype

        talker._setup_cfm_buffers()
        talker._setup_execution_optimizations()
        talker.model.precompute_fused_qkv()
        talker.residual_model.precompute_fused_qkv()

        tts = talker.tts
        dit_hidden = tts.lm_to_dit_proj.out_features + tts.res_to_dit_proj.out_features
        cfm_buffers = _CFMBufferManager(
            device=dev,
            dtype=dtype,
            feat_dim=D,
            patch_size=P,
            dit_hidden_size=dit_hidden,
            max_batch_size=max(1, patch_count),
        )

        g = _CapturedUnifiedDecodeGraph(
            graph=torch.cuda.CUDAGraph(),
            batch_size=batch_size,
            input_embeds=torch.zeros(token_capacity, H, device=dev, dtype=dtype),
            positions=torch.zeros(token_capacity, device=dev, dtype=torch.long),
            prev_feat_embed=torch.zeros(token_capacity, H, device=dev, dtype=dtype),
            prefix_feat_cond=torch.zeros(batch_size, P, D, device=dev, dtype=dtype),
            next_feat_embed=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            cfm_output=torch.zeros(batch_size, D, P, device=dev, dtype=dtype),
            lm_hidden=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            cfm_buffers=cfm_buffers,
            cfm_noise=torch.zeros(batch_size, D, P, device=dev, dtype=dtype),
        )

        if patch_batch_size is not None:
            g.patch_indices = torch.zeros(patch_count, device=dev, dtype=torch.long)
        if mixed:
            g.text_mask = torch.ones(token_capacity, 1, device=dev, dtype=dtype)
            g.last_token_indices = torch.zeros(batch_size, device=dev, dtype=torch.long)
        model_state = getattr(talker, "_mrv2_model_state", None)
        per_patch_audio = model_state.per_patch_audio
        if patch_batch_size is not None and (not per_patch_audio or not model_state.incremental_vae):
            raise ValueError("Exact patch graphs require MRV2 incremental per-patch audio")
        if per_patch_audio:
            g.audio_slots = torch.full((batch_size,), len(model_state.slots), device=dev, dtype=torch.long)
            g.audio_rows_valid = torch.zeros(batch_size, device=dev, dtype=torch.bool)
        dit_proj = talker._dit_proj_fn

        def unified_fwd():
            scaffold_h = talker.model(input_ids=None, positions=g.positions, inputs_embeds=g.input_embeds)
            if isinstance(scaffold_h, tuple):
                scaffold_h = scaffold_h[0]
            lm_h = tts.fsq_layer(scaffold_h)
            if mixed:
                lm_h = lm_h * (1 - g.text_mask) + scaffold_h * g.text_mask
            res_input = tts.fusion_concat_proj(torch.cat([lm_h, g.prev_feat_embed], dim=-1))
            res_h = talker.residual_model(positions=g.positions, inputs_embeds=res_input)
            if mixed:
                g.scaffold_hidden = scaffold_h
                lm_h = lm_h.index_select(0, g.last_token_indices)
                res_h = res_h.index_select(0, g.last_token_indices)
            output_lm_h = lm_h
            prefix_cond = g.prefix_feat_cond
            noise = g.cfm_noise
            if g.patch_indices is not None:
                if patch_count == 0:
                    g.stop_logits = talker._stop_fn(output_lm_h)
                    g.pcm = torch.zeros(batch_size, P * int(tts.audio_vae.decode_chunk_size), device=dev)
                    g.pcm_valid = torch.zeros(batch_size, device=dev, dtype=torch.bool)
                    return torch.zeros_like(g.next_feat_embed), torch.zeros_like(g.cfm_output), output_lm_h
                lm_h = lm_h.index_select(0, g.patch_indices)
                res_h = res_h.index_select(0, g.patch_indices)
                prefix_cond = prefix_cond.index_select(0, g.patch_indices)
                noise = noise.index_select(0, g.patch_indices)
            dit_h = dit_proj(lm_h, res_h)
            cond = prefix_cond.transpose(1, 2).contiguous()
            cfm_out = talker._solve_cfm_with_noise(
                tts.feat_decoder,
                dit_h,
                P,
                cond,
                noise,
                talker._inference_timesteps,
                talker._cfg_value,
                cfm_buffers,
                cfg_cutoff_ratio=talker._cfg_cutoff_ratio,
            )
            feat_enc = tts.feat_encoder(cfm_out.transpose(1, 2).unsqueeze(1)).squeeze(1)
            next_embed = tts.enc_to_lm_proj(feat_enc)
            if per_patch_audio:
                g.stop_logits = talker._stop_fn(output_lm_h)
                if getattr(model_state, "incremental_vae", False):
                    slots, rows_valid, scores = g.audio_slots, g.audio_rows_valid, g.stop_logits
                    if g.patch_indices is not None:
                        slots = slots.index_select(0, g.patch_indices)
                        rows_valid = rows_valid.index_select(0, g.patch_indices)
                        scores = scores.index_select(0, g.patch_indices)
                    pcm, valid = model_state.decode_patch(slots, rows_valid, cfm_out.transpose(1, 2))
                    model_state.commit_patch(slots, valid, next_embed, cfm_out.transpose(1, 2), scores)
                    if g.patch_indices is not None:
                        g.pcm = pcm.new_zeros(batch_size, pcm.shape[1]).index_copy_(0, g.patch_indices, pcm)
                        g.pcm_valid = valid.new_zeros(batch_size).index_copy_(0, g.patch_indices, valid)
                    else:
                        g.pcm, g.pcm_valid = pcm, valid
            if g.patch_indices is not None:
                next_embed = next_embed.new_zeros(batch_size, H).index_copy_(0, g.patch_indices, next_embed)
                cfm_out = cfm_out.new_zeros(batch_size, D, P).index_copy_(0, g.patch_indices, cfm_out)
            return next_embed, cfm_out, output_lm_h

        ctx = get_forward_context()
        capture_context = override_forward_context(self._nullify_volatile_metadata(ctx))

        original_estimator = tts.feat_decoder.estimator
        capture_estimator = self._voxcpm2_compile_unified_capture_estimator(original_estimator)
        tts.feat_decoder.estimator = capture_estimator

        original_feat_encoder = tts.feat_encoder
        capture_feat_encoder = self._voxcpm2_compile_unified_capture_feat_encoder(original_feat_encoder)
        tts.feat_encoder = capture_feat_encoder

        try:
            with capture_context:
                with torch.no_grad():
                    for _ in range(3):
                        g.cfm_noise.normal_()
                        unified_fwd()

                    g.cfm_noise.normal_()
                    with torch.cuda.graph(g.graph, pool=current_platform.get_global_graph_pool()):
                        g.next_feat_embed, g.cfm_output, g.lm_hidden = unified_fwd()
        finally:
            tts.feat_decoder.estimator = original_estimator
            tts.feat_encoder = original_feat_encoder
        if patch_batch_size is not None:
            logger.info(
                "Exact patch graph captured: request_bucket=%d token_bucket=%d patch_count=%d",
                batch_size,
                token_capacity,
                patch_count,
            )
        self._unified_graph_stats.captures += 1
        logger.info(
            "CUDA Graph captured for unified decode (batch_size=%d, captures=%d)",
            batch_size,
            self._unified_graph_stats.captures,
        )
        return g

    def _unified_decode_graph_skip_reason(
        self,
        *,
        can_use_graph: bool,
        is_all_decode: bool,
        num_reqs: int,
    ) -> str | None:
        talker = self.model
        if not talker._enable_unified_decode_graph:
            return "disabled"
        if not can_use_graph:
            return "graph_not_ready"
        if not is_all_decode:
            return "not_all_decode"
        if num_reqs > self._unified_decode_graph_max_batch_size:
            return "batch_exceeds_unified_max"
        if num_reqs > self._max_cached_graphs:
            return "batch_too_large"
        graph_size = self._select_unified_graph_bucket_size(num_reqs)
        if graph_size is None:
            return "capture_policy"
        if num_reqs > 1 and not self._runner_assisted_unified_decode_graph_active:
            return "runner_full_metadata_missing"
        for req_id, _, _, _ in talker._pending_requests:
            state = talker._active_states[req_id]
            if not state.prefill_completed:
                return "prefill_incomplete"
            if (
                talker._state_tensor(state, "prev_feat_embed") is None
                or talker._state_tensor(state, "curr_prefix_feat_cond") is None
            ):
                return "state_not_ready"
        return None

    def _maybe_log_unified_graph_stats(self) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        stats = self._unified_graph_stats
        total_skips = stats.total_skips
        if stats.replays - stats.logged_replays < 100 and total_skips - stats.logged_skips < 50:
            return
        stats.logged_replays = stats.replays
        stats.logged_skips = total_skips
        logger.debug(
            "VoxCPM2 unified graph stats: captures=%d replays=%d skips=%s real_batch_sizes=%s graph_bucket_sizes=%s",
            stats.captures,
            stats.replays,
            stats.skips,
            dict(sorted(stats.real_batch_sizes.items())),
            dict(sorted(stats.graph_bucket_sizes.items())),
        )

    # -------------------- vllm hooks --------------------

    def get_runner_assisted_full_attention_metadata_request(
        self,
        *,
        req_ids: Sequence[str],
        num_reqs: int,
        num_scheduled_tokens: Sequence[int],
        num_computed_tokens: Sequence[int],
        max_num_scheduled_tokens: int,
        num_prompt_tokens: Sequence[int] | None = None,
    ) -> RunnerAssistedFullAttentionMetadataRequest | None:
        talker = self.model
        if not talker._enable_unified_decode_graph or not talker._enable_cuda_graph:
            return None
        if len(req_ids) != num_reqs or len(num_computed_tokens) != num_reqs:
            return None
        has_prefill = any(int(n) == 0 for n in num_computed_tokens) or any(
            not getattr(talker._active_states.get(req_id), "prefill_completed", False) for req_id in req_ids
        )
        exact = getattr(getattr(talker, "_runtime_config", None), "enable_exact_patch_batch", False)
        patch_count = None
        if exact:
            if len(num_scheduled_tokens) != num_reqs:
                return None
            patch_count = talker._scheduled_patch_count(
                req_ids, num_scheduled_tokens, num_computed_tokens, num_prompt_tokens
            )
            if patch_count is None:
                return None
            has_prefill |= any(
                int(num_computed_token) < int(num_prompt_token)
                for num_computed_token, num_prompt_token in zip(num_computed_tokens, num_prompt_tokens, strict=True)
            )
        if self._mixed_unified_graph_enabled() and has_prefill:
            tts_compiled = (
                getattr(talker.tts.feat_decoder.estimator, "_compiled", False) if talker._tts is not None else False
            )
            if not tts_compiled or self._cuda_graph_warmup_steps < self._cuda_graph_warmup_threshold:
                return None
            bucket = self._select_unified_graph_bucket_size(num_reqs)
            if (
                bucket is None
                or num_reqs > self._unified_decode_graph_max_batch_size
                or len(req_ids) != num_reqs
                or len(num_scheduled_tokens) != num_reqs
                or any(int(n) <= 0 for n in num_scheduled_tokens)
            ):
                return None
            tokens = self._mixed_token_capacity(sum(int(n) for n in num_scheduled_tokens), bucket)
            if tokens is None:
                return None
            key = (bucket, tokens, patch_count) if exact else (bucket, tokens)
            if (
                key not in self._mixed_graphs
                and self._mixed_graph_cache_limit > 0
                and len(self._mixed_graphs) >= self._mixed_graph_cache_limit
            ):
                return None
            return RunnerAssistedFullAttentionMetadataRequest(bucket, key not in self._mixed_graphs, tokens)
        model_state = getattr(talker, "_mrv2_model_state", None)
        per_patch_single = num_reqs == 1 and getattr(model_state, "per_patch_audio", False)
        if (num_reqs <= 1 and not per_patch_single) or num_reqs > self._unified_decode_graph_max_batch_size:
            return None
        if num_reqs > self._max_cached_graphs or self._select_unified_graph_bucket_size(num_reqs) is None:
            return None
        if max_num_scheduled_tokens != 1 or len(num_scheduled_tokens) != num_reqs:
            return None
        if any(int(n) != 1 for n in num_scheduled_tokens):
            return None
        if len(req_ids) != num_reqs or len(num_computed_tokens) != num_reqs:
            return None
        if any(int(n) <= 0 for n in num_computed_tokens):
            return None
        tts_compiled = (
            getattr(talker.tts.feat_decoder.estimator, "_compiled", False) if talker._tts is not None else False
        )
        if not tts_compiled or self._cuda_graph_warmup_steps < self._cuda_graph_warmup_threshold:
            return None
        for req_id in req_ids:
            state = talker._active_states.get(req_id)
            if (
                state is None
                or not state.prefill_completed
                or talker._state_tensor(state, "prev_feat_embed") is None
                or talker._state_tensor(state, "curr_prefix_feat_cond") is None
            ):
                return None
        bucket_size = self._select_unified_graph_bucket_size(num_reqs)
        if bucket_size is None:
            return None
        key = (bucket_size, patch_count) if exact else bucket_size
        return RunnerAssistedFullAttentionMetadataRequest(
            num_reqs_padded=bucket_size,
            for_cudagraph_capture=key not in self._unified_graphs,
        )

    def set_runner_assisted_full_attention_metadata_context(
        self,
        *,
        enabled: bool,
        num_reqs: int = 0,
    ) -> None:
        enabled = bool(enabled)
        self._runner_assisted_unified_decode_graph_active = enabled
        self._runner_assisted_unified_decode_graph_batch_size = int(num_reqs) if enabled else 0

    def mixed_graph_capacity(self, num_reqs: int, num_tokens: int, patch_count: int | None) -> int | None:
        bucket = self._select_unified_graph_bucket_size(num_reqs)
        if bucket is None:
            return None
        capacity = self._mixed_token_capacity(num_tokens, bucket)
        if capacity is None:
            return None
        if patch_count is not None:
            key = (bucket, capacity, patch_count)
            if (
                key not in self._mixed_graphs
                and self._mixed_graph_cache_limit > 0
                and len(self._mixed_graphs) >= self._mixed_graph_cache_limit
            ):
                return None
        return capacity

    def record_skip(self, reason: str) -> None:
        self._unified_graph_stats.record_skip(reason)
        self._maybe_log_unified_graph_stats()

    def _mixed_token_capacity(self, num_tokens: int, request_bucket: int) -> int | None:
        talker = self.model
        limit = talker.vllm_config.scheduler_config.max_num_batched_tokens
        required = max(num_tokens, request_bucket)
        if required > limit:
            return None
        return min(limit, 1 << (required - 1).bit_length())

    def _mixed_unified_graph_enabled(self) -> bool:
        talker = self.model
        return talker._enable_unified_decode_graph

    def _prepare_mixed_graph_inputs(self, g: _CapturedUnifiedDecodeGraph, num_reqs: int) -> None:
        """Map packed token inputs to per-request patch conditions."""
        talker = self.model
        offset = 0
        last_indices = []
        g.text_mask.zero_()
        g.prev_feat_embed.zero_()
        g.prefix_feat_cond.zero_()
        for row, (req_id, is_prefill, _embeds, count) in enumerate(talker._pending_requests):
            state = talker._active_states[req_id]
            if is_prefill:
                text_mask, audio_mask, audio_feat, feat_embed = state.prefill_masks
                start = state.prefill_chunk_start
                end = start + count
                text_mask = text_mask[:, start:end]
                audio_mask = audio_mask[:, start:end]
                feat_embed = feat_embed[:, start:end]
                g.text_mask[offset : offset + count].copy_(text_mask.reshape(-1, 1))
                g.prev_feat_embed[offset : offset + count].copy_(
                    (audio_mask.unsqueeze(-1) * feat_embed).reshape(count, -1)
                )
                g.prefix_feat_cond[row].copy_(audio_feat[0, end - 1])
            else:
                g.prev_feat_embed[offset].copy_(talker._state_tensor(state, "prev_feat_embed").reshape(-1))
                g.prefix_feat_cond[row].copy_(talker._state_tensor(state, "curr_prefix_feat_cond"))
            offset += count
            last_indices.append(offset - 1)
        last_indices.extend([last_indices[-1]] * (g.batch_size - num_reqs))
        host = torch.tensor(last_indices, dtype=torch.long, pin_memory=g.last_token_indices.is_cuda)
        g.last_token_indices.copy_(host, non_blocking=True)

    def get_unified_graph(
        self, num_reqs: int, mixed_token_capacity: int | None, patch_count: int | None
    ) -> _CapturedUnifiedDecodeGraph:
        model_state = self.model._mrv2_model_state
        if model_state.per_patch_audio and model_state._pcm_copy_event is not None:
            # Shared capture pools can alias outputs across different buckets.
            torch.cuda.current_stream(self.model._device).wait_event(model_state._pcm_copy_event)
        graph_size = self._select_unified_graph_bucket_size(num_reqs)
        if graph_size is None:
            raise RuntimeError(f"No unified decode graph bucket for batch size {num_reqs}")
        graph_cache = self._unified_graphs if mixed_token_capacity is None else self._mixed_graphs
        graph_key = graph_size if mixed_token_capacity is None else (graph_size, mixed_token_capacity)
        if patch_count is not None:
            graph_key = (
                (graph_size, patch_count)
                if mixed_token_capacity is None
                else (graph_size, mixed_token_capacity, patch_count)
            )
        g = graph_cache.get(graph_key)
        if g is None:
            g = (
                self._capture_unified_decode_graph(graph_size, mixed_token_capacity, patch_count)
                if patch_count is not None
                else self._capture_unified_decode_graph(graph_size, mixed_token_capacity)
            )
            graph_cache[graph_key] = g

        return g

    def replay_unified(
        self,
        g: _CapturedUnifiedDecodeGraph,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        states: list[VoxCPM2RequestState],
        commit_mask: list[bool],
        batch_context: VoxCPM2BatchSlots | None,
        mixed_token_capacity: int | None,
        patch_rows: list[int] | None,
    ) -> tuple[torch.Tensor, int]:
        talker = self.model
        num_reqs = len(states)
        graph_size = g.batch_size
        self._unified_graph_stats.replays += 1
        self._unified_graph_stats.record_replay_batch(real_batch_size=num_reqs, graph_bucket_size=graph_size)
        self._maybe_log_unified_graph_stats()
        model_state = getattr(talker, "_mrv2_model_state", None)
        slot_indices = model_state.indices_for(states, batch_context)
        if mixed_token_capacity is not None:
            self._prepare_mixed_graph_inputs(g, num_reqs)
        else:
            g.prev_feat_embed[:num_reqs].copy_(model_state.next_embed.index_select(0, slot_indices))
            g.prefix_feat_cond[:num_reqs].copy_(model_state.prefix_feat.index_select(0, slot_indices))

        num_tokens = sum(count for _, _, _, count in talker._pending_requests) if mixed_token_capacity else num_reqs
        g.input_embeds[:num_tokens].copy_(inputs_embeds[:num_tokens])
        g.positions[:num_tokens].copy_(positions[:num_tokens])
        if mixed_token_capacity is not None:
            g.input_embeds[num_tokens:].zero_()
            g.positions[num_tokens:].zero_()
        elif graph_size > num_reqs:
            # Runner-assisted FULL metadata pads attention to graph_size. Keep
            # padded rows as valid duplicates instead of position-0 zero rows;
            # some attention backends may still touch padded slots before
            # honoring seq_lens=0.
            last = num_reqs - 1
            g.input_embeds[num_reqs:graph_size].copy_(g.input_embeds[last : last + 1].expand(graph_size - num_reqs, -1))
            g.positions[num_reqs:graph_size].copy_(g.positions[last : last + 1].expand(graph_size - num_reqs))
            g.prev_feat_embed[num_reqs:graph_size].copy_(
                g.prev_feat_embed[last : last + 1].expand(graph_size - num_reqs, -1)
            )
            g.prefix_feat_cond[num_reqs:graph_size].copy_(
                g.prefix_feat_cond[last : last + 1].expand(graph_size - num_reqs, -1, -1)
            )
        g.cfm_noise.normal_()
        # Seeded rows overwrite their slice from their own generator, so the
        # noise a seeded request sees does not depend on batch composition.
        for i, state in enumerate(states[:num_reqs]):
            if mixed_token_capacity is not None and talker._has_deterministic_cfm_noise(state):
                talker._fill_deterministic_cfm_noise_for_state(state, g.cfm_noise[i : i + 1])
            elif state.cfm_generator is not None:
                g.cfm_noise[i : i + 1].normal_(generator=state.cfm_generator)

        if g.audio_slots is not None:
            g.audio_slots[:num_reqs].copy_(slot_indices)
            g.audio_slots[num_reqs:].fill_(len(model_state.slots))
            host = torch.tensor(commit_mask + [False] * (graph_size - num_reqs), dtype=torch.bool, pin_memory=True)
            g.audio_rows_valid.copy_(host, non_blocking=True)
        if g.patch_indices is not None:
            host_rows = torch.tensor(patch_rows, dtype=torch.long, pin_memory=True)
            g.patch_indices.copy_(host_rows, non_blocking=True)
        g.graph.replay()

        return slot_indices, num_tokens
