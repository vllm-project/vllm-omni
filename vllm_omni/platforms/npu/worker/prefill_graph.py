# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-length raw NPU graphs for single-request MiniCPM-o prefill."""

from copy import copy
from pathlib import Path

import torch
from torch.utils._pytree import tree_leaves, tree_map
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm_ascend.attention.attention_v1 import AscendAttentionState

from vllm_omni.platforms.npu.attention.static_shape_backend import OmniStaticShapeAttentionBackendImpl

logger = init_logger(__name__)
THINKER_STARTUP_PREFIX = "minicpmo-thinker-startup-"
THINKER_STARTUP_LENGTHS = range(128, 161)
TALKER_STARTUP_PREFIX = "minicpmo-talker-startup-"
TALKER_STARTUP_LENGTHS = range(7, 24)
PREFILL_CAPTURE_FREE_BYTES = 6 * 1024**3


class PrefillGraph:
    def __init__(
        self,
        runner,
        model,
        embeds,
        positions,
        input_ids,
        metadata,
        layers,
        *,
        thinker=False,
    ):
        self.embeds = torch.empty_like(embeds)
        self.positions = torch.empty_like(positions)
        self.input_ids = torch.empty_like(input_ids) if input_ids is not None else None
        self.slot_groups = {}
        self.layer_slot_group = {}
        self.kv = []
        self.metadata = {}
        groups_by_address = {}
        for name, layer_metadata in metadata.items():
            impl = getattr(layers.get(name), "impl", None)
            if not isinstance(impl, OmniStaticShapeAttentionBackendImpl) or impl.use_bnsd_kv_cache:
                raise ValueError("Prefill graph requires native TND and paged KV")
            if impl.sinks is not None or impl.sliding_window is not None or impl.kv_sharing_target_layer_name:
                raise ValueError("Prefill graph cannot write this attention layer")
            for cache in (impl.key_cache, impl.value_cache):
                if cache is None or cache.ndim != 4 or not cache.is_contiguous():
                    raise ValueError("Prefill graph requires contiguous KV")
                self.kv.append((name, cache.flatten(0, 1)))
            slots = layer_metadata.slot_mapping
            address = slots.data_ptr()
            if address not in groups_by_address:
                groups_by_address[address] = name
                self.slot_groups[name] = torch.empty_like(slots)
            group_name = groups_by_address[address]
            self.layer_slot_group[name] = group_name
            static = copy(layer_metadata)
            static.slot_mapping = self.slot_groups[group_name]
            if not thinker:
                static.talker_prefill_exact = True
                static.static_shape_prefill_bucket = embeds.shape[0]
            self.metadata[name] = static
        self.bind(embeds, positions, input_ids, metadata)
        size = embeds.shape[0]

        def call_model():
            return model(
                input_ids=self.input_ids,
                positions=self.positions,
                inputs_embeds=self.embeds,
                intermediate_tensors=None,
            )

        live_context = get_forward_context() if thinker else None
        if live_context is not None:
            static_context = copy(live_context)
            if isinstance(live_context.attn_metadata, list):
                static_context.attn_metadata = [{**live_context.attn_metadata[0], **self.metadata}]
            else:
                static_context.attn_metadata = {**live_context.attn_metadata, **self.metadata}
            static_slots = {name: entry.slot_mapping for name, entry in self.metadata.items()}
            static_context.slot_mapping = {**live_context.slot_mapping, **static_slots}
            self.skip_compiled = static_context.skip_compiled

        def forward():
            if live_context is not None:
                with override_forward_context(static_context):
                    return call_model()
            with runner._forward_context(
                self.metadata,
                runner.vllm_config,
                num_tokens=size,
                num_actual_tokens=size,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(num_tokens=size),
                skip_compiled=True,
            ):
                return call_model()

        current = torch.npu.current_stream()
        # Use one expandable-pool stream per runner across prefill shapes.
        warmup = runner._prefill_graph_warmup_stream
        warmup.wait_stream(current)
        try:
            with torch.npu.stream(warmup):
                forward()
        finally:
            current.wait_stream(warmup)
        self.graph = torch.npu.NPUGraph()
        if not thinker:
            if not hasattr(runner, "_talker_prefill_capture_pool"):
                runner._talker_prefill_capture_pool = torch.npu.graph_pool_handle()
            # Replay is serialized on the model worker and returns a snapshot.
            with torch.npu.graph(self.graph, pool=runner._talker_prefill_capture_pool, stream=warmup):
                self.hidden = forward()
        else:
            with torch.npu.graph(self.graph):
                self.hidden = forward()
        self.replays = 0
        logger.info("[%s-prefill-graph] captured tokens=%d", "thinker" if thinker else "talker", size)

    def bind(self, embeds, positions, input_ids, metadata):
        if self.input_ids is not None:
            self.input_ids.copy_(input_ids)
        self.embeds.copy_(embeds)
        self.positions.copy_(positions)
        for layer_name, group_name in self.layer_slot_group.items():
            if metadata[layer_name].slot_mapping.data_ptr() != metadata[group_name].slot_mapping.data_ptr():
                raise ValueError("Prefill KV slot groups changed since graph capture")
        for name, slots in self.slot_groups.items():
            slots.copy_(metadata[name].slot_mapping)


def _read_indices(metadata):
    slots_by_address = {}
    indices = {}
    for name, layer in metadata.items():
        address = layer.slot_mapping.data_ptr()
        if address not in slots_by_address:
            slots_by_address[address] = layer.slot_mapping.long()
        indices[name] = slots_by_address[address]
    return indices


def _kv_rows(kv, indices):
    return [cache.index_select(0, indices[name]) for name, cache in kv]


def run_prefill(runner, model_inputs, batch, desc, *, startup_path, thinker=False):
    lengths = THINKER_STARTUP_LENGTHS if thinker else TALKER_STARTUP_LENGTHS
    if batch.num_reqs != 1 or batch.num_tokens not in lengths:
        return None
    prefix = THINKER_STARTUP_PREFIX if thinker else TALKER_STARTUP_PREFIX
    startup = batch.req_ids[0].startswith(prefix) and Path(startup_path).is_file()
    if not startup and batch.num_tokens not in getattr(runner, "_exact_prefill_passed", ()):
        return None
    if (
        not batch.has_prefill
        or batch.is_dummy
        or desc.cg_mode != CUDAGraphMode.NONE
        or runner.lora_config is not None
        or runner.use_aux_hidden_state_outputs != thinker
        or getattr(runner, "_exact_prefill_disabled", False)
        or model_inputs.get("intermediate_tensors") is not None
    ):
        return None
    metadata = get_forward_context().attn_metadata if thinker else runner.model_state.attn_metadata
    if isinstance(metadata, list) and len(metadata) == 1:
        metadata = metadata[0]
    if not isinstance(metadata, dict) or not metadata:
        return None
    if not isinstance(get_forward_context().slot_mapping, dict):
        return None
    if thinker:
        target_layers = getattr(runner, "_thinker_prefill_target_attention", None)
        if target_layers is None:
            from vllm.model_executor.layers.attention import Attention

            target_layers = runner._thinker_prefill_target_attention = {
                layer.layer_name for layer in runner.model.modules() if isinstance(layer, Attention)
            }
        if not target_layers or not target_layers.issubset(metadata):
            return None
        metadata = {name: layer for name, layer in metadata.items() if name in target_layers}
    first = next(iter(metadata.values()))
    size = batch.num_tokens
    if (
        batch.num_tokens_after_padding != size
        or first.slot_mapping.shape != (size,)
        or not all(
            m.attn_state == AscendAttentionState.PrefillNoCache
            and m.causal
            and m.actual_seq_lengths_q == [size]
            and m.num_actual_tokens == size
            and m.slot_mapping.shape == (size,)
            and m.attn_mask is first.attn_mask
            for m in metadata.values()
        )
    ):
        return None
    infos = model_inputs["model_intermediate_buffer"]
    if len(infos) != 1 or not isinstance(infos[0], dict) or infos[0].get("native_duplex"):
        return None
    duplex = infos[0].get("duplex")
    if isinstance(duplex, dict) and duplex.get("data_plane") is True:
        return None
    embeds, positions = model_inputs.get("inputs_embeds"), model_inputs.get("positions")
    if embeds is None or embeds.dtype != torch.bfloat16 or positions is None or positions.shape != (size,):
        return None

    if thinker:
        input_ids = model_inputs.get("input_ids")
        if input_ids is None or input_ids.shape != (size,):
            return None
        model = runner.model
    else:
        talker = getattr(runner.model, "talker", runner.model)
        talker._flush_deferred_cleanup()
        model = talker.tts_model
        input_ids = None
    stage = "thinker" if thinker else "talker"
    graphs = getattr(runner, "_exact_prefill_graphs", None)
    if graphs is None:
        runner._prefill_graph_warmup_stream = torch.npu.Stream()
        graphs = runner._exact_prefill_graphs = {}
        runner._exact_prefill_passed = set()
    try:
        graph = graphs.get(size)
        if graph is None:
            if torch.npu.mem_get_info()[0] < PREFILL_CAPTURE_FREE_BYTES:
                logger.info_once("[%s-prefill-startup] remaining device memory below capture budget", stage)
                return None
            layers = runner.vllm_config.compilation_config.static_forward_context
            graph = graphs[size] = PrefillGraph(
                runner,
                model,
                embeds,
                positions,
                input_ids,
                metadata,
                layers,
                thinker=thinker,
            )
        else:
            if thinker and graph.skip_compiled != get_forward_context().skip_compiled:
                return None
            if any(x.attn_mask is not metadata[name].attn_mask for name, x in graph.metadata.items()):
                return None
            graph.bind(embeds, positions, input_ids, metadata)
        graph.graph.replay()
    except (RuntimeError, ValueError, TypeError, AttributeError):
        runner._exact_prefill_disabled = True
        logger.exception("[%s-prefill-graph] capture or replay failed", stage)
        return None
    if not startup:
        graph.replays += 1
        if graph.replays == 1:
            logger.info("[%s-prefill-graph] active replay tokens=%d", stage, size)
        # A later request can replay the same exact-length graph while
        # output materialization still owns this step's hidden tensor.
        return tree_map(torch.clone, graph.hidden)
    slot_indices = _read_indices(metadata)
    captured_kv = _kv_rows(graph.kv, slot_indices)
    expected = runner.model(**model_inputs)
    actual_leaves, expected_leaves = tree_leaves(graph.hidden), tree_leaves(expected)
    hidden_match = len(actual_leaves) == len(expected_leaves) and all(
        torch.equal(actual, baseline) for actual, baseline in zip(actual_leaves, expected_leaves, strict=True)
    )
    kv_match = all(
        torch.equal(old, cache.index_select(0, slot_indices[name]))
        for (name, cache), old in zip(graph.kv, captured_kv, strict=True)
    )
    logger.info(
        "[%s-prefill-graph] shadow tokens=%d hidden=%s kv=%s leaves=%d",
        stage,
        size,
        hidden_match,
        kv_match,
        len(actual_leaves),
    )
    if hidden_match and kv_match:
        runner._exact_prefill_passed.add(size)
    else:
        runner._exact_prefill_disabled = True
        logger.warning("[%s-prefill-graph] shadow discrepancy", stage)
    return expected
