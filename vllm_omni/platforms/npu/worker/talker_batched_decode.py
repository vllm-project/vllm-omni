# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched single-query graph replay for the MiniCPM-o Talker."""

import os
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm_ascend.attention.attention_v1 import AscendAttentionState

from vllm_omni.platforms.npu.attention import static_shape_decode
from vllm_omni.platforms.npu.attention.static_shape_backend import OmniStaticShapeAttentionBackendImpl
from vllm_omni.platforms.npu.worker.talker_context import (
    SHORT_CONTEXT,
    hybrid_enabled,
    scheduled_length,
    trace_context,
)
from vllm_omni.platforms.npu.worker.talker_exact_rotary import fused as exact_bf16_rotary

logger = init_logger(__name__)
MODE = os.getenv("MINICPMO_NPU_TALKER_NARROW_C8_MODE_FILE")


class CompactGraph:
    def __init__(
        self,
        runner,
        model,
        first,
        metadata,
        embeds,
        positions,
        rows,
        capacity,
        wide_rows,
        *,
        fixed_down_lanes=False,
        fused_rope=False,
    ):
        self.embed = torch.empty_like(embeds[:rows])
        self.positions = torch.empty_like(positions[:rows])
        self.slots = torch.empty_like(first.slot_mapping[:rows])
        self.lengths = torch.empty_like(first.seq_lens_device.reshape(-1)[:rows])
        self.first_length = torch.empty_like(self.lengths)
        self.blocks = torch.empty_like(
            first.block_tables[:rows, : capacity // runner.vllm_config.cache_config.block_size]
        )
        heads = model._tts_config.num_attention_heads
        self.bias = torch.empty((rows, heads, 1, capacity), dtype=embeds.dtype, device=embeds.device)
        self.lse = torch.empty(1, dtype=embeds.dtype, device=embeds.device)
        static = copy(first)
        static.attn_state = AscendAttentionState.DecodeOnly
        static.num_actual_tokens = static.num_decode_tokens = static.num_decodes = rows
        static.num_prefills = 0
        static.slot_mapping, static.block_tables = self.slots, self.blocks
        static.seq_lens_device = self.lengths
        static.actual_seq_lengths_q = list(range(1, rows + 1))
        static.talker_fused_frame = self.bias, self.lse, capacity
        self.metadata = dict.fromkeys(metadata, static)
        self.kv = []
        registered = runner.vllm_config.compilation_config.static_forward_context
        for name in metadata:
            impl = getattr(registered.get(name), "impl", None)
            if not isinstance(impl, OmniStaticShapeAttentionBackendImpl) or impl.use_bnsd_kv_cache:
                raise ValueError("The narrow path needs static paged TND attention")
            for cache in (impl.key_cache, impl.value_cache):
                if cache is None or cache.ndim != 4 or not cache.is_contiguous():
                    raise ValueError("The narrow path needs the layer's actual contiguous paged KV")
                self.kv.append(cache.flatten(0, 1))
        self.runner = runner
        self.model = model
        self.rows = rows
        self.capacity = capacity
        self.desc = BatchDescriptor(num_tokens=rows)
        self.graph = None
        self.down_projections = ()
        self.fixed_down_lanes = fixed_down_lanes
        self.fused_rope = fused_rope
        if rows > 4:
            if runner.vllm_config.parallel_config.tensor_parallel_size != 1:
                raise ValueError("Talker hybrid replay needs TP=1")
            down = tuple(layer.mlp.down_proj for layer in model.tts_model.layers)
            widths = {layer.input_size_per_partition for layer in down}
            if len(widths) != 1 or any(layer.bias is not None for layer in down):
                raise ValueError("Talker hybrid replay needs a uniform, bias-free MLP")
            self.down_projections = down
            down_shape = (wide_rows, widths.pop())
            if self.fixed_down_lanes:
                # Keep all original M rows and the original strides. Only
                # selected outputs escape the projection; inactive rows stay
                # zero throughout capture and replay.
                self.down_input = embeds.new_zeros(down_shape)
            else:
                self.down_input = embeds.new_empty(down_shape)
                self.lane_base = torch.arange(rows, dtype=torch.long, device=embeds.device) * 16
                self.lanes = torch.empty_like(self.lane_base)

    @contextmanager
    def _original_down_shapes(self):
        # The Ascend BF16 down-projection changes numerical results at some
        # M shapes. Keep its original row count and locations while compiling
        # this graph; attention and the other MLP/GEMM operators stay narrow.
        replaced = []
        try:
            for projection in self.down_projections:
                old = projection.forward
                had_local = "forward" in projection.__dict__
                replaced.append((projection, old, had_local))

                def project(x, call=old):
                    if self.fixed_down_lanes:
                        live = slice(0, self.rows * 16, 16)
                        self.down_input[live].copy_(x)
                        result, bias = call(self.down_input)
                        return result[live].contiguous(), bias
                    self.down_input.zero_()
                    self.down_input.index_copy_(0, self.lanes, x)
                    result, bias = call(self.down_input)
                    return result.index_select(0, self.lanes), bias

                projection.forward = project
            yield
        finally:
            for projection, old, had_local in replaced:
                if had_local:
                    projection.forward = old
                else:
                    del projection.forward

    @contextmanager
    def _exact_rotary(self):
        if not self.fused_rope:
            yield
            return
        replaced = []
        try:
            layers = tuple(self.model.tts_model.layers)
            if len(layers) != 20:
                raise ValueError("Exact compact RoPE needs the 20-layer Talker")
            for layer in layers:
                rotary = layer.self_attn.rotary_emb
                if (
                    not rotary.is_neox_style
                    or rotary.head_size != 64
                    or rotary.rotary_dim != 64
                    or getattr(rotary.apply_rotary_emb, "enable_fp32_compute", False)
                ):
                    raise ValueError("Exact compact RoPE needs full-head BF16 NeoX rounding")
                cache = rotary.cos_sin_cache
                if cache.ndim != 2 or cache.shape[1] != 64 or not cache.is_contiguous():
                    raise ValueError("Exact compact RoPE needs a contiguous cosine/sine table")
                old = rotary.forward
                replaced.append((rotary, old, "forward" in rotary.__dict__))

                def project(positions, query, key=None, table=cache):
                    if (
                        key is None
                        or query.shape != (self.rows, 768)
                        or key.shape != query.shape
                        or query.dtype != torch.bfloat16
                        or key.dtype != query.dtype
                        or query.stride(1) != 1
                        or key.stride(1) != 1
                        or positions.ndim != 1
                        or positions.shape[0] != self.rows
                        or positions.dtype not in (torch.int32, torch.int64)
                    ):
                        raise ValueError("Unexpected Q/K shape in compact Talker")
                    if table.device != query.device or table.dtype != query.dtype:
                        table = table.to(device=query.device, dtype=query.dtype)
                    return exact_bf16_rotary(query, key, positions.contiguous(), table)

                rotary.forward = project
            yield
        finally:
            for rotary, old, had_local in reversed(replaced):
                if had_local:
                    rotary.forward = old
                else:
                    del rotary.forward

    def prepare_step(self, first, width):
        torch.sub(first.seq_lens_device.reshape(-1)[: self.rows], width - 1, out=self.first_length)
        self.blocks.copy_(first.block_tables[: self.rows, : self.blocks.shape[1]])

    def bind(self, first, embeds, positions, width, frame):
        end = self.rows * width
        if self.down_projections and not self.fixed_down_lanes:
            torch.add(self.lane_base, frame, out=self.lanes)
        self.embed.copy_(embeds[frame:end:width])
        self.positions.copy_(positions[frame:end:width])
        self.slots.copy_(first.slot_mapping[frame:end:width])
        torch.add(self.first_length, frame, out=self.lengths)

    def forward(self):
        static_shape_decode.emit_mask_refresh(self.bias, self.lengths, self.bias.shape[1], self.capacity)
        with (
            self._original_down_shapes(),
            self._exact_rotary(),
            self.runner._forward_context(
                self.metadata,
                self.runner.vllm_config,
                num_tokens=self.rows,
                num_actual_tokens=self.rows,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=self.desc,
                skip_compiled=True,
            ),
        ):
            return self.model.tts_model(
                input_ids=None,
                positions=self.positions,
                intermediate_tensors=None,
                inputs_embeds=self.embed,
            )

    def replay(self):
        if self.graph is None:
            current = torch.npu.current_stream()
            warmup = self.runner._c8_talker_warmup
            warmup.wait_stream(current)
            try:
                with torch.npu.stream(warmup):
                    template = self.forward()
            finally:
                current.wait_stream(warmup)
            # Keep frame outputs outside the capture pool: variant graphs use
            # common scratch and replay serially on the Talker runner.
            self.output = torch.empty_like(template)
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                self.graph,
                pool=self.runner._c8_talker_shared_capture_pool,
                stream=self.runner._c8_talker_capture_stream,
            ):
                self.output.copy_(self.forward())
        self.graph.replay()
        return self.output


def prepare(runner, talker, model_inputs, input_batch, width):
    if MODE is None or not 2 <= input_batch.num_reqs <= 8 or width != 16 or input_batch.has_prefill:
        return None
    try:
        mode, _, variant = Path(MODE).read_text().strip().partition("_")
    except FileNotFoundError:
        return None
    if (
        mode not in ("shadow", "active")
        or variant not in ("", "fixed", "fixed_rope")
        or model_inputs.get("intermediate_tensors") is not None
    ):
        return None
    rows = input_batch.num_reqs
    fixed_down_lanes = variant in ("fixed", "fixed_rope") and rows > 4
    fused_rope = variant == "fixed_rope"
    spans = runner.model_state._request_token_spans(input_batch)
    if spans != [(i * width, (i + 1) * width) for i in range(rows)]:
        return None
    infos = model_inputs.get("model_intermediate_buffer") or []
    if len(infos) != rows or any(not isinstance(item, dict) or item.get("native_duplex") for item in infos):
        return None
    metadata = runner.model_state.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return None
    first = next(iter(metadata.values()))
    if not all(
        getattr(x, "seq_lens_device", None) is not None
        and x.seq_lens_device is first.seq_lens_device
        and x.block_tables is first.block_tables
        and x.slot_mapping is first.slot_mapping
        for x in metadata.values()
    ):
        return None

    full_capacity = SHORT_CONTEXT
    model_len = runner.model_config.max_model_len
    hybrid = hybrid_enabled(model_len)
    if model_len != full_capacity and not hybrid:
        return None
    length = scheduled_length(input_batch, rows, width)
    if hybrid and (length is None or length > full_capacity):
        trace_context(input_batch, "wide_batch", length)
        return None
    block = runner.vllm_config.cache_config.block_size
    embeds, positions = model_inputs.get("inputs_embeds"), model_inputs.get("positions")
    if (
        block != 128
        or embeds is None
        or embeds.dtype != torch.bfloat16
        or embeds.shape[0] < rows * width
        or positions is None
        or positions.ndim != 1
        or positions.shape[0] < rows * width
        or first.slot_mapping.ndim != 1
        or first.slot_mapping.numel() < rows * width
        or first.block_tables.shape[0] < rows
        or first.block_tables.shape[1] < full_capacity // block
        or first.seq_lens_device.numel() < rows
    ):
        return None
    if not hasattr(runner, "_c8_talker_graphs"):
        runner._c8_talker_graphs = {}
        runner._c8_talker_ready = set()
        runner._c8_talker_rejected = set()
        runner._c8_talker_warmup = torch.npu.Stream()
        runner._c8_talker_capture_stream = torch.npu.Stream()
        runner._c8_talker_shared_capture_pool = torch.npu.graph_pool_handle()
    graphs, ready, rejected = runner._c8_talker_graphs, runner._c8_talker_ready, runner._c8_talker_rejected
    wide_rows = embeds.shape[0]
    # MRv2's host seq_lens_np already counts this entire scheduled K-step
    # window. A bucket that fits it is safe for every frame in the step.
    # Use 512 on any build that cannot provide a host length vector.
    preferred = full_capacity
    if length is not None:
        preferred = static_shape_decode.select_bucket(length, (128, 256, full_capacity))
    candidate_capacities = (preferred, full_capacity) if preferred != full_capacity else (full_capacity,)
    graph_key = next(
        (
            key
            for capacity in candidate_capacities
            if (key := (rows, wide_rows, capacity, fixed_down_lanes, fused_rope)) not in rejected
        ),
        None,
    )
    if graph_key is None:
        return None
    capacity = graph_key[2]
    compact = graphs.get(graph_key)
    if compact is None:
        try:
            compact = graphs[graph_key] = CompactGraph(
                runner,
                talker,
                first,
                metadata,
                embeds,
                positions,
                rows,
                capacity,
                wide_rows,
                fixed_down_lanes=fixed_down_lanes,
                fused_rope=fused_rope,
            )
        except ValueError:
            rejected.add(graph_key)
            logger.exception("[c8-talker-narrow] unsupported geometry rows=%d", rows)
            return None
    if tuple(compact.metadata) != tuple(metadata):
        return None
    # Keep the complete first K-step on the reference path while comparing
    # every frame; only subsequent steps may use the qualified graph.
    shadow = mode == "shadow" or graph_key not in ready

    def frame_forward(frame, wide_forward):
        if frame == 0 and graph_key not in rejected:
            compact.prepare_step(first, width)
        if not shadow and graph_key not in rejected:
            try:
                compact.bind(first, embeds, positions, width, frame)
                logger.info_once(
                    "[c8-talker-narrow] active replay rows=%d capacity=%d fixed=%s rope=%s",
                    rows,
                    capacity,
                    fixed_down_lanes,
                    fused_rope,
                )
                output = compact.replay()
                if hybrid and frame == width - 1:
                    trace_context(input_batch, "compact_batch", length, capacity)
                return output
            except (RuntimeError, TypeError, ValueError):
                rejected.add(graph_key)
                logger.exception("[c8-talker-narrow] replay failed rows=%d", rows)
        reference = wide_forward()[frame : rows * width : width].clone()
        if graph_key in rejected or not shadow:
            return reference
        compact.bind(first, embeds, positions, width, frame)
        slots = compact.slots.long()
        expected_kv = [cache.index_select(0, slots) for cache in compact.kv]
        try:
            actual = compact.replay()
            hidden_ok = torch.equal(actual, reference)
            kv_ok = all(
                torch.equal(old, cache.index_select(0, slots))
                for old, cache in zip(expected_kv, compact.kv, strict=True)
            )
        except (RuntimeError, TypeError, ValueError):
            logger.exception("[c8-talker-narrow] shadow capture failed rows=%d", rows)
            hidden_ok = kv_ok = False
        if not (hidden_ok and kv_ok):
            ready.discard(graph_key)
            rejected.add(graph_key)
            for cache, old in zip(compact.kv, expected_kv, strict=True):
                cache.index_copy_(0, slots, old)
            logger.warning(
                "[c8-talker-narrow] shadow mismatch rows=%d capacity=%d frame=%d hidden=%s kv=%s",
                rows,
                capacity,
                frame,
                hidden_ok,
                kv_ok,
            )
        elif frame == width - 1 and graph_key not in ready:
            ready.add(graph_key)
            logger.info(
                "[c8-talker-narrow] shadow exact all frames rows=%d capacity=%d fixed=%s rope=%s",
                rows,
                capacity,
                fixed_down_lanes,
                fused_rope,
            )
        return reference

    return frame_forward
