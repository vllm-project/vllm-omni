# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph path for π0.5 ``sample_actions``.

``sample_actions`` has three regions that are captured and replayed separately:

1. the prefix embedding (``embed_prefix``);
2. the prefix forward pass that writes the per-layer prefix K/V into
   ``model.kv_cache`` (``paligemma_with_expert.forward`` with prefix inputs);
3. one denoising step (``denoise_step``), replayed once per step.

One graph per region and batch size is enough. Every other shape is fixed for
a serving instance: ``max_cameras`` image slots (a missing camera only flips
its mask), ``image_resolution`` images, a ``tokenizer_max_length`` text block
and a ``chunk_size`` suffix. With schedule precomputation off, the per-request
step count only changes how often region 3 replays. With it on, the default
schedule supplies a new conditioning/modulation row on every replay; unsupported
schedules use eager denoising. No captured control flow depends on tensor data.

The three regions are captured for batch 1 and for every other size in
``batch_sizes`` (the pipeline declares them from ``max_num_seqs``), each size
over a KV cache and a memory pool of its own; ``kv_cache_for`` hands a formed
batch's cache to ``sample_actions``. A formed batch of a captured size replays
its three graphs as batch 1 does. A batch between two captured sizes runs over
the first rows of the next larger size's cache. Its prefix embedding and prefix
forward run without a graph, at its own size (the prefix forward on the fused
kernels): both are compute-bound, so the rows a padded replay would add cost
their full time, and their cuBLAS GEMMs may round a row differently among more
rows. Its denoising steps replay the larger graph, which is launch-bound: the
rows behind the batch carry stale data and are dropped from the output (every
row is computed on its own). A batch above the largest captured size runs
without a graph.

Optional request-invariant masks, positions and RoPE tables are copied into
stable buffers once per request, before its denoising loop. Optional KV-only
prefix capture is used only by callers explicitly requesting KV-only output;
generic prefix calls retain the eager full-hidden-state semantics.

The graphs are captured once at init (``Pi05CUDAGraphs.capture``), each size's
in replay order and into its memory pool. Regions hand data to each other
without copies: region 1's ``prefix_embs`` output is region 2's static input,
its ``prefix_pad_masks`` output is region 3's, and the size's KV cache
(``model.kv_cache`` for batch 1) is written by region 2 and extended and read
by region 3. A region copies an input into its static buffer only when the
caller passes a different tensor.

Each region method has the signature of the eager call it replaces and falls
back to that call when nothing is captured, its inputs differ from the
captured ones, or torch's default dtype is not the float32 it was captured
under. The graphs capture whatever the regions run: on the optimized path,
the fused Triton kernels ``Pi05Pipeline`` enables before capturing. The eager
baseline is ``Pi05ForActionPrediction.cuda_graphs is None`` with the fused
kernels off.

A replay overwrites the tensors the previous replay of that region returned.
``sample_actions`` consumes them within the call, and the chunk it returns
comes from its own Euler update, so nothing it returns aliases a graph buffer.
``tests/diffusion/models/pi05/test_pi05_cuda_graph_parity.py`` pins replay
bit-exact against the same regions run eagerly, on the real checkpoint.
"""

from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
    Pi05DenoiseContext,
    Pi05DenoiseStepData,
    Pi05KVCache,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from vllm_omni.diffusion.models.pi05.runtime_pi05 import Pi05RuntimeOptions

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.pi05.modeling_pi05 import Pi05ForActionPrediction

logger = init_logger(__name__)

# ``prepare_attention_masks_4d`` builds its float mask in torch's default dtype,
# so a graph equals the eager call only under the default it was captured with.
# ``DiffusersLoader`` constructs the pipeline (and so captures) under the serving
# dtype as the default, while requests run under torch's float32 default.
_CAPTURE_DEFAULT_DTYPE = torch.float32


def _matches(src, static: torch.Tensor) -> bool:
    return (
        isinstance(src, torch.Tensor)
        and src.shape == static.shape
        and src.dtype == static.dtype
        and src.device == static.device
    )


def _load(static: torch.Tensor, src: torch.Tensor) -> None:
    if src is not static:
        static.copy_(src)


def _matches_rows(src, static: torch.Tensor) -> bool:
    """``src`` is the first rows of ``static``'s shape: a batch no larger than the captured one."""
    return (
        isinstance(src, torch.Tensor)
        and src.dim() == static.dim()
        and 0 < src.shape[0] <= static.shape[0]
        and src.shape[1:] == static.shape[1:]
        and src.dtype == static.dtype
        and src.device == static.device
    )


def _load_rows(static: torch.Tensor, src: torch.Tensor) -> None:
    if src.shape[0] == static.shape[0]:
        _load(static, src)
    else:
        static[: src.shape[0]].copy_(src)


def _serves(cache, static: Pi05KVCache) -> bool:
    """``cache`` is the captured cache or a row view of it (``Pi05KVCache.rows``)."""
    return cache is static or (
        isinstance(cache, Pi05KVCache)
        and cache.batch_size <= static.batch_size
        and cache.prefix_len == static.prefix_len
        and cache.key.data_ptr() == static.key.data_ptr()
        and cache.value.data_ptr() == static.value.data_ptr()
    )


def _rotary_matches(src, static: tuple[torch.Tensor, torch.Tensor] | None) -> bool:
    if static is None:
        return src is None
    return (
        isinstance(src, tuple)
        and len(src) == 2
        and all(_matches(tensor, buffer) for tensor, buffer in zip(src, static))
    )


def _step_data_matches(src: Pi05DenoiseStepData | None, static: Pi05DenoiseStepData | None) -> bool:
    if static is None:
        return src is None
    return (
        isinstance(src, Pi05DenoiseStepData)
        and _matches(src.timestep, static.timestep)
        and _matches(src.time_cond, static.time_cond)
        and _matches(src.modulations, static.modulations)
    )


@dataclass
class _EmbedPrefixGraph:
    graph: torch.cuda.CUDAGraph
    images: list[torch.Tensor]
    image_masks: list[torch.Tensor]
    lang_tokens: torch.Tensor
    lang_masks: torch.Tensor
    # ``(prefix_embs, prefix_pad_masks, prefix_att_masks)``
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class _PrefixForwardGraph:
    graph: torch.cuda.CUDAGraph
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    # Region 1's ``prefix_embs`` output.
    inputs_embeds: torch.Tensor
    kv_cache: Pi05KVCache
    # ``([prefix_out, None], kv_cache)``
    outputs: tuple[list[torch.Tensor | None], Pi05KVCache]
    prefix_kv_only: bool = False
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class _DenoiseStepGraph:
    graph: torch.cuda.CUDAGraph
    # Region 1's ``prefix_pad_masks`` output.
    prefix_pad_masks: torch.Tensor
    kv_cache: Pi05KVCache
    x_t: torch.Tensor
    timestep: torch.Tensor
    v_t: torch.Tensor
    context: Pi05DenoiseContext | None = None
    step_data: Pi05DenoiseStepData | None = None


class Pi05CUDAGraphs:
    """Per-region CUDA Graph capture/replay for one ``Pi05ForActionPrediction``."""

    def __init__(self, model: Pi05ForActionPrediction, batch_sizes: Iterable[int] = (1,)):
        # The model owns this object through ``model.cuda_graphs``; a weak
        # reference back keeps that from forming a cycle, so dropping the model
        # frees its GPU memory immediately rather than at the next GC pass.
        self._model = weakref.proxy(model)
        # Capture modes are initialization-only. The model's frozen options
        # determine one graph per region, not a graph for every schedule row.
        self._runtime_options = getattr(model, "runtime_options", Pi05RuntimeOptions())
        self._embed_prefix: _EmbedPrefixGraph | None = None
        self._prefix_forward: _PrefixForwardGraph | None = None
        self._denoise_step: _DenoiseStepGraph | None = None
        # Batch 1 is always captured; ``batch_sizes`` adds the formed-batch
        # sizes the three regions are also captured for.
        sizes = tuple(sorted(set(batch_sizes)))
        if any(not isinstance(size, int) or size < 1 for size in sizes):
            raise ValueError(f"batch_sizes must be positive integers, got {sizes}.")
        self._batch_sizes = sizes
        self._embed_batches: dict[int, _EmbedPrefixGraph] = {}
        self._prefix_batches: dict[int, _PrefixForwardGraph] = {}
        self._denoise_batches: dict[int, _DenoiseStepGraph] = {}
        self.num_replays: Counter[str] = Counter()

    # ── Capture ──────────────────────────────────────────────────────
    def capture(self) -> None:
        """Capture all three regions at the deployed shapes, at batch size 1
        and at every other declared batch size.

        Needs ``model.kv_cache``. Raises on failure: ``enforce_eager=False`` asks
        for the graph path, so a silent eager fallback would hide a slowdown.
        """
        try:
            self._capture()
        except Exception as exc:
            self._embed_prefix = self._prefix_forward = self._denoise_step = None
            self._embed_batches = {}
            self._prefix_batches = {}
            self._denoise_batches = {}
            raise RuntimeError(
                "π0.5 CUDA Graph capture failed. Set `enforce_eager: true` in the deploy "
                "config, or pass --enforce-eager, to serve eagerly."
            ) from exc

    def _capture(self) -> None:
        model = self._model
        kv_cache = model.kv_cache
        if kv_cache is None or kv_cache.batch_size != 1:
            raise ValueError(f"Capture needs a batch-size-1 model.kv_cache, got {kv_cache!r}.")
        device = kv_cache.key.device

        # Override an outer loader inference_mode as well: static buffers must
        # remain normal tensors, which callers in either mode may copy into.
        with torch.inference_mode(False), torch.no_grad(), set_default_torch_dtype(_CAPTURE_DEFAULT_DTYPE):
            # Batch 1 runs over ``model.kv_cache`` and the global pool; a formed
            # batch over a cache and a pool of its own, since batches of
            # different sizes replay in any order.
            regions = self._capture_regions(1, kv_cache, current_platform.get_global_graph_pool(), device)
            batches = {
                batch: self._capture_regions(batch, None, torch.cuda.graph_pool_handle(), device)
                for batch in self._batch_sizes
                if batch > 1
            }

        self._embed_prefix, self._prefix_forward, self._denoise_step = regions
        self._embed_batches = {batch: embed for batch, (embed, _, _) in batches.items()}
        self._prefix_batches = {batch: prefix for batch, (_, prefix, _) in batches.items()}
        self._denoise_batches = {batch: denoise for batch, (_, _, denoise) in batches.items()}

    def _capture_regions(
        self, batch: int, kv_cache: Pi05KVCache | None, pool, device: torch.device
    ) -> tuple[_EmbedPrefixGraph, _PrefixForwardGraph, _DenoiseStepGraph]:
        """The three regions for a batch of ``batch`` requests, over ``kv_cache``
        (a new one of the batch's size when None), captured into ``pool`` in
        replay order.

        The AdaRMS schedule table serves batch 1 only; a formed batch embeds
        its timestep inside its denoising graph.
        """
        model = self._model
        options = self._runtime_options
        config = model.config
        height, width = config.image_resolution
        text_len = int(config.tokenizer_max_length)

        # Dummy inputs; only their shapes, dtypes and addresses are captured.
        images = [torch.zeros(batch, 3, height, width, device=device) for _ in range(int(config.max_cameras))]
        image_masks = [torch.ones(batch, dtype=torch.bool, device=device) for _ in images]
        lang_tokens = torch.zeros(batch, text_len, dtype=torch.long, device=device)
        lang_masks = torch.ones(batch, text_len, dtype=torch.bool, device=device)

        def embed():
            return model.embed_prefix(images, image_masks, lang_tokens, lang_masks)

        # Eager warmup, outside capture: lazy initialization and library
        # workspaces. It also builds regions 2 and 3's inputs with the same
        # calls ``sample_actions`` uses, so their shapes and dtypes match.
        prefix_embs, prefix_pad_masks, prefix_att_masks = embed()
        if kv_cache is None:
            kv_cache = model.new_kv_cache(batch, prefix_embs.shape[1])
            # Dummy prefix K/V; only its addresses are captured.
            kv_cache.key.zero_()
            kv_cache.value.zero_()
        attention_mask = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_kwargs = {}
        rotary_emb = None
        if options.prefix_kv_only:
            prefix_kwargs["prefix_kv_only"] = True
        if options.hoist_denoise_invariants:
            rotary_emb = model.prepare_prefix_rotary(prefix_embs, position_ids)
            prefix_kwargs["rotary_emb"] = rotary_emb

        def prefix(embs):
            return model.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,
                inputs_embeds=[embs, None],
                use_cache=True,
                **prefix_kwargs,
            )

        x_t = torch.zeros(batch, model.action_horizon, model.action_dim, dtype=torch.float32, device=device)
        timestep = torch.ones(batch, dtype=torch.float32, device=device)
        context = None
        step_data = None
        denoise_kwargs = {}
        if options.hoist_denoise_invariants:
            context = model.prepare_denoise_context(prefix_pad_masks)
            denoise_kwargs["context"] = context
        if options.precompute_adarms and batch == 1:
            table = model.runtime_step_table(model.num_inference_steps, 1)
            if table is None:
                raise ValueError("Prepare the default BS=1 AdaRMS schedule after loading weights, before capture.")
            row = table[0]
            # All rows use these same static addresses. A replay loads the
            # selected row in bulk; it never keeps using capture's row 0.
            timestep = row.timestep.clone()
            step_data = Pi05DenoiseStepData(timestep, row.time_cond.clone(), row.modulations.clone())
            denoise_kwargs["step_data"] = step_data

        def step(pad_masks):
            return model.denoise_step(
                prefix_pad_masks=pad_masks,
                past_key_values=kv_cache,
                x_t=x_t,
                timestep=timestep,
                **denoise_kwargs,
            )

        prefix(prefix_embs)
        step(prefix_pad_masks)
        torch.accelerator.synchronize(device)

        # Graphs sharing a pool must replay in the order they were captured,
        # which is the order ``sample_actions`` runs the regions in.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outputs = embed()
        embed_prefix = _EmbedPrefixGraph(graph, images, image_masks, lang_tokens, lang_masks, outputs)
        prefix_embs, prefix_pad_masks, _ = outputs

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            outputs = prefix(prefix_embs)
        prefix_forward = _PrefixForwardGraph(
            graph,
            attention_mask,
            position_ids,
            prefix_embs,
            kv_cache,
            outputs,
            options.prefix_kv_only,
            rotary_emb,
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            v_t = step(prefix_pad_masks)
        denoise_step = _DenoiseStepGraph(graph, prefix_pad_masks, kv_cache, x_t, timestep, v_t, context, step_data)
        return embed_prefix, prefix_forward, denoise_step

    def kv_cache_for(self, batch_size: int, prefix_len: int) -> Pi05KVCache | None:
        """The KV cache a formed batch of ``batch_size`` requests runs over: the
        cache of the graphs that serve the size (the first rows of it when the
        batch is smaller), or None when no graph does. Batch 1 runs over
        ``model.kv_cache``."""
        g = None if batch_size == 1 else self._denoise_graph(batch_size)
        if g is None or not g.kv_cache.fits(g.kv_cache.batch_size, prefix_len):
            return None
        return g.kv_cache if g.kv_cache.batch_size == batch_size else g.kv_cache.rows(batch_size)

    def _embed_graph(self, batch_size: int) -> _EmbedPrefixGraph | None:
        """The prefix embedding graph of exactly ``batch_size``; a batch between
        captured sizes runs its prefix embedding without a graph."""
        return self._embed_prefix if batch_size == 1 else self._embed_batches.get(batch_size)

    def _prefix_graph(self, batch_size: int) -> _PrefixForwardGraph | None:
        """The prefix forward graph of exactly ``batch_size``; a batch between
        captured sizes runs its prefix forward without a graph."""
        return self._prefix_forward if batch_size == 1 else self._prefix_batches.get(batch_size)

    def _denoise_graph(self, batch_size: int) -> _DenoiseStepGraph | None:
        """The denoising graph that serves ``batch_size``: its own, else the smallest larger one."""
        if batch_size == 1:
            return self._denoise_step
        g = self._denoise_batches.get(batch_size)
        if g is None:
            larger = [size for size in self._denoise_batches if size > batch_size]
            g = self._denoise_batches[min(larger)] if larger else None
        return g

    # ── Region 1: prefix embedding ───────────────────────────────────
    def embed_prefix(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self._embed_graph(lang_tokens.shape[0]) if isinstance(lang_tokens, torch.Tensor) else None
        if g is None:
            return self._model.embed_prefix(images, image_masks, lang_tokens, lang_masks)
        if not (
            torch.get_default_dtype() == _CAPTURE_DEFAULT_DTYPE
            and len(images) == len(g.images)
            and len(image_masks) == len(g.image_masks)
            and all(_matches(src, static) for src, static in zip(images, g.images))
            and all(_matches(src, static) for src, static in zip(image_masks, g.image_masks))
            and _matches(lang_tokens, g.lang_tokens)
            and _matches(lang_masks, g.lang_masks)
        ):
            logger.warning_once("π0.5 embed_prefix inputs differ from the captured ones; it runs eagerly.")
            return self._model.embed_prefix(images, image_masks, lang_tokens, lang_masks)

        for static, src in zip(g.images, images):
            _load(static, src)
        for static, src in zip(g.image_masks, image_masks):
            _load(static, src)
        _load(g.lang_tokens, lang_tokens)
        _load(g.lang_masks, lang_masks)
        g.graph.replay()
        self.num_replays["embed_prefix"] += 1
        return g.outputs

    # ── Region 2: prefix forward (prefix KV construction) ────────────
    def prefix_forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool = False,
        adarms_cond: torch.Tensor | None = None,
        *,
        prefix_kv_only: bool = False,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        adarms_modulations: torch.Tensor | None = None,
    ):
        g = None
        if inputs_embeds is not None and len(inputs_embeds) == 2 and isinstance(inputs_embeds[0], torch.Tensor):
            g = self._prefix_graph(inputs_embeds[0].shape[0])
        if g is not None and not (
            torch.get_default_dtype() == _CAPTURE_DEFAULT_DTYPE
            and past_key_values is g.kv_cache
            and use_cache
            and adarms_cond is None
            and adarms_modulations is None
            and prefix_kv_only == g.prefix_kv_only
            and _rotary_matches(rotary_emb, g.rotary_emb)
            and inputs_embeds[1] is None
            and _matches(inputs_embeds[0], g.inputs_embeds)
            and _matches(attention_mask, g.attention_mask)
            and _matches(position_ids, g.position_ids)
        ):
            logger.warning_once("π0.5 prefix forward inputs differ from the captured ones; it runs eagerly.")
            g = None
        if g is None:
            # Omitting the optional flag still requests full prefix hidden
            # states, even if this instance captured a KV-only prefix graph.
            kwargs = {}
            if prefix_kv_only:
                kwargs["prefix_kv_only"] = True
            if rotary_emb is not None:
                kwargs["rotary_emb"] = rotary_emb
            if adarms_modulations is not None:
                kwargs["adarms_modulations"] = adarms_modulations
            return self._model.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
                **kwargs,
            )

        _load(g.inputs_embeds, inputs_embeds[0])
        _load(g.attention_mask, attention_mask)
        _load(g.position_ids, position_ids)
        if g.rotary_emb is not None:
            for static, src in zip(g.rotary_emb, rotary_emb):
                _load(static, src)
        g.graph.replay()
        self.num_replays["prefix_forward"] += 1
        return g.outputs

    # ── Region 3: one denoising step ─────────────────────────────────
    def prepare_denoise_context(self, context: Pi05DenoiseContext) -> Pi05DenoiseContext:
        """Copy this request's invariants once into the captured buffers.

        ``sample_actions`` owns the request context and calls this before its
        denoising loop. The returned static context is valid until the next
        request of its batch size prepares it; graph buffers do not support
        concurrent requests. An incompatible context stays request-local and
        uses eager denoising.
        """
        g = self._denoise_graph(context.attention_mask.shape[0]) if isinstance(context, Pi05DenoiseContext) else None
        if g is None or g.context is None:
            return context
        tensors = ("attention_mask", "position_ids", "cos", "sin")
        if not (
            torch.get_default_dtype() == _CAPTURE_DEFAULT_DTYPE
            and all(_matches_rows(getattr(context, name), getattr(g.context, name)) for name in tensors)
        ):
            logger.warning_once("π0.5 denoising context differs from the captured one; it runs eagerly.")
            return context
        for name in tensors:
            _load_rows(getattr(g.context, name), getattr(context, name))
        self.num_replays["prepare_denoise_context"] += 1
        return g.context

    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        *,
        context: Pi05DenoiseContext | None = None,
        step_data: Pi05DenoiseStepData | None = None,
    ) -> torch.Tensor:
        g = self._denoise_graph(x_t.shape[0])
        if g is not None and not (
            torch.get_default_dtype() == _CAPTURE_DEFAULT_DTYPE
            and _serves(past_key_values, g.kv_cache)
            and _matches_rows(prefix_pad_masks, g.prefix_pad_masks)
            and _matches_rows(x_t, g.x_t)
            and _matches_rows(timestep, g.timestep)
            and context is g.context
            and _step_data_matches(step_data, g.step_data)
        ):
            logger.warning_once("π0.5 denoise_step inputs differ from the captured ones; it runs eagerly.")
            g = None
        if g is None:
            kwargs = {}
            # A static context prepared for a larger graph does not fit the eager call.
            if context is not None and context.attention_mask.shape[0] == x_t.shape[0]:
                kwargs["context"] = context
            if step_data is not None:
                kwargs["step_data"] = step_data
            return self._model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
                **kwargs,
            )

        if g.context is None:
            _load_rows(g.prefix_pad_masks, prefix_pad_masks)
        _load_rows(g.x_t, x_t)
        _load_rows(g.timestep, timestep)
        if g.step_data is not None:
            # ``modulations`` contains every layer's input/post-attention norm
            # and final norm in one tensor: one copy, not a copy per layer.
            _load(g.step_data.time_cond, step_data.time_cond)
            _load(g.step_data.modulations, step_data.modulations)
        g.graph.replay()
        self.num_replays["denoise_step"] += 1
        return g.v_t if x_t.shape[0] == g.v_t.shape[0] else g.v_t[: x_t.shape[0]]
