# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in one-request Talker graph for Model Runner V2."""

from __future__ import annotations

import copy
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import torch
import torch_npu  # noqa: F401 -- registers torch.npu on Ascend
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger
from vllm_ascend.attention.attention_v1 import AscendAttentionState

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    HISTORY_WINDOW,
    _codec_int_param,
    _request_key,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms.npu.attention import static_shape_decode
from vllm_omni.platforms.npu.attention.static_shape_backend import OmniStaticShapeAttentionBackendImpl
from vllm_omni.platforms.npu.worker.talker_context import (
    SHORT_CONTEXT,
    hybrid_enabled,
    scheduled_length,
    trace_context,
)
from vllm_omni.platforms.npu.worker.talker_fused_sampling import make_device_sampler

logger = init_logger(__name__)
WIDTH = 16
CAPACITY = SHORT_CONTEXT


@dataclass
class LiveStep:
    info: dict
    request_id: str
    state: dict
    sampler: dict
    embeds: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    lengths: torch.Tensor
    blocks: torch.Tensor
    metadata: dict


def prepare_step(runner, talker, batch_desc, model_inputs, batch):
    if (
        batch.num_reqs != 1
        or batch_desc.num_reqs != 1
        or batch_desc.num_tokens != WIDTH
        or batch.has_prefill
        or runner.use_aux_hidden_state_outputs
        or model_inputs.get("intermediate_tensors") is not None
    ):
        logger.info_once(
            "[talker-fused-c1] shape gate: requests=%s desc_requests=%s tokens=%s prefill=%s aux=%s context=%s",
            batch.num_reqs,
            batch_desc.num_reqs,
            batch_desc.num_tokens,
            batch.has_prefill,
            runner.use_aux_hidden_state_outputs,
            runner.model_config.max_model_len,
        )
        return None
    info = model_inputs.get("model_intermediate_buffer", ())
    if len(info) != 1 or not isinstance(info[0], dict):
        return None
    info = info[0]
    if info.get("native_duplex") or info.get("_omni_is_prefill"):
        return None
    request_id = _request_key(info, 0)
    talker._flush_deferred_cleanup()
    state = talker._request_audio_states.get(request_id)
    sampler = getattr(talker, "_request_host_codec_states", {}).get(request_id)
    if not state or not sampler or not sampler.get("history"):
        logger.info_once("[talker-fused-c1] host sampling history unavailable at step entry")
        return None
    eos = talker._codec_eos_id
    if (
        state.get("finished")
        or state.get("turn_end_drain")
        or state.get("codec_repetition_detection")
        or sampler.get("finished")
        or sampler["step"] != state["step"]
        or sampler.get("params") != ((eos,), eos, (), 0.8, 100, 0.8, 0.0, 1.05, 0.0, 0.0, False)
    ):
        logger.info_once(
            "[talker-fused-c1] request gate: model_step=%s sampler_step=%s settings=%s",
            state.get("step"),
            sampler.get("step"),
            sampler.get("params"),
        )
        return None
    if info.get("_omni_finished") or request_id in talker._deferred_cleanup_ids:
        return None
    metadata = runner.model_state.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return None
    first = next(iter(metadata.values()))
    if not hasattr(first, "seq_lens_device"):
        logger.info_once("[talker-fused-c1] static length tensor unavailable")
        return None
    if not all(
        getattr(m, "seq_lens_device", None) is first.seq_lens_device
        and m.block_tables is first.block_tables
        and m.slot_mapping is first.slot_mapping
        for m in metadata.values()
    ):
        logger.info_once("[talker-fused-c1] layer metadata uses multiple cache groups")
        return None
    blocks, slots, lengths = first.block_tables, first.slot_mapping, first.seq_lens_device
    block_size = runner.vllm_config.cache_config.block_size
    if (
        blocks.shape[0] != 1
        or blocks.shape[1] * block_size < CAPACITY
        or slots.ndim != 1
        or slots.numel() < WIDTH
        or lengths.numel() != 1
    ):
        logger.info_once(
            "[talker-fused-c1] metadata gate: blocks=%s slots=%s lengths=%s", blocks.shape, slots.shape, lengths.shape
        )
        return None
    embeds, positions = model_inputs.get("inputs_embeds"), model_inputs.get("positions")
    if embeds is None or positions is None or positions.ndim != 1 or embeds.dtype != torch.bfloat16:
        return None
    return LiveStep(info, request_id, state, sampler, embeds, positions, slots, lengths, blocks, metadata)


class TalkerFusedGraph:
    def __init__(self, runner, talker, live, sampler_factory, *, split_at_eos=False, capacity=CAPACITY):
        self.runner = runner
        self.capacity = capacity
        self.talker = talker
        self.eos = talker._codec_eos_id
        device = live.embeds.device
        self.embeds = torch.empty_like(live.embeds[:1])
        self.positions = torch.empty_like(live.positions[:WIDTH])
        self.slots = torch.empty_like(live.slots[:WIDTH])
        self.lengths = torch.empty_like(live.lengths)
        block_size = runner.vllm_config.cache_config.block_size
        self.blocks = torch.empty_like(live.blocks[:, : capacity // block_size])
        self.offsets = torch.arange(1 - WIDTH, 1, dtype=self.lengths.dtype, device=device)
        self.history = torch.empty((1, HISTORY_WINDOW), dtype=torch.long, device=device)
        self.uniforms = torch.empty(WIDTH, dtype=torch.float32, device=device)
        self.window = torch.empty(3, dtype=torch.int32, device=device)
        self.frame_indices = torch.arange(WIDTH, dtype=torch.int32, device=device)
        self.active = torch.ones(1, dtype=torch.bool, device=device)
        self.bias = torch.empty(
            (1, talker._tts_config.num_attention_heads, 1, capacity), dtype=live.embeds.dtype, device=device
        )
        self.lse = torch.empty(1, dtype=live.embeds.dtype, device=device)
        self.sampler = sampler_factory(self.eos, 0.8, 100, 0.8, 1.05, device)
        self.metadata = []
        self.kv = []
        self.layer_names = tuple(live.metadata)
        no_compile_layers = runner.vllm_config.compilation_config.static_forward_context
        for name in live.metadata:
            layer = no_compile_layers.get(name)
            impl = getattr(layer, "impl", None)
            if not isinstance(impl, OmniStaticShapeAttentionBackendImpl) or impl.use_bnsd_kv_cache:
                raise ValueError(f"Talker fused decode requires static FIA on {name}")
            for cache in (impl.key_cache, impl.value_cache):
                if cache is None or cache.ndim != 4 or not cache.is_contiguous():
                    raise ValueError(f"Talker fused decode requires contiguous paged KV on {name}")
                self.kv.append(cache.flatten(0, 1))
        first = next(iter(live.metadata.values()))
        for frame in range(WIDTH):
            metadata = copy.copy(first)
            metadata.num_actual_tokens = metadata.num_decode_tokens = 1
            metadata.num_decodes, metadata.num_prefills = 1, 0
            metadata.attn_state = AscendAttentionState.DecodeOnly
            metadata.slot_mapping = self.slots[frame : frame + 1]
            metadata.block_tables = self.blocks
            metadata.talker_fused_frame = (self.bias, self.lse, capacity)
            self.metadata.append(dict.fromkeys(live.metadata, metadata))
        self.graph = None
        self.replay_ms = []
        self.split_at_eos = split_at_eos
        self.tail_skips = 0
        self.tail_steps = 0
        self.desc = BatchDescriptor(num_tokens=1)

    def bind(self, live):
        self.embeds.copy_(live.embeds[:1])
        self.positions.copy_(live.positions[:WIDTH])
        self.slots.copy_(live.slots[:WIDTH])
        self.lengths.copy_(live.lengths)
        self.blocks.copy_(live.blocks[:, : self.blocks.shape[1]])
        history = live.sampler["history"][-HISTORY_WINDOW:]
        self.history.copy_(torch.tensor([[-1] * (HISTORY_WINDOW - len(history)) + history], dtype=torch.long))
        rng = random.Random(0)
        rng.setstate(live.sampler["generator"].getstate())
        self.uniforms.copy_(torch.tensor([rng.random() for _ in range(WIDTH)], dtype=torch.float32))
        self.window.copy_(
            torch.tensor(
                [
                    live.state["step"],
                    _codec_int_param(live.state, "min_tokens", self.talker._codec_min_tokens),
                    _codec_int_param(live.state, "max_tokens", self.talker._codec_max_tokens),
                ],
                dtype=torch.int32,
            )
        )

    def snapshot_kv(self):
        slots = self.slots.long()
        return slots, [cache.index_select(0, slots) for cache in self.kv]

    def restore_kv(self, saved):
        slots, rows = saved
        for cache, original in zip(self.kv, rows, strict=True):
            cache.index_copy_(0, slots, original)

    def forward(self, start=0, end=WIDTH, state=None):
        history, active, embeds = state if state is not None else (self.history, self.active, self.embeds)
        hidden_rows, outputs = [], []
        for frame in range(start, end):
            metadata = self.metadata[frame]
            static_shape_decode.emit_mask_refresh(
                self.bias, self.lengths + self.offsets[frame], self.bias.shape[1], self.capacity
            )
            with self.runner._forward_context(
                metadata,
                self.runner.vllm_config,
                num_tokens=1,
                num_actual_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=self.desc,
                skip_compiled=True,
            ):
                hidden = self.talker.tts_model(
                    input_ids=None,
                    positions=self.positions[frame : frame + 1],
                    intermediate_tensors=None,
                    inputs_embeds=embeds,
                )
            logits = self.talker.head_code[0](hidden).float()
            step = self.window[0] + self.frame_indices[frame]
            mask_eos = (step < self.window[1]).reshape(1)
            token, guard = self.sampler(logits, history, mask_eos, self.uniforms[frame : frame + 1])
            stop = ~active | (token == self.eos) | (step + 1 >= self.window[2])
            emit = ~stop
            outputs.append(torch.stack((token, emit.long(), stop.long(), (guard & active).long()), dim=-1))
            hidden_rows.append(hidden)
            history = torch.where(
                emit[:, None],
                torch.cat((history[:, 1:], token[:, None]), dim=1),
                history,
            )
            embeds = self.talker.emb_code[0](token)
            active = emit
        return torch.cat(hidden_rows), torch.cat(outputs), (history, active, embeds)

    def run(self):
        if self.graph is None:
            saved = self.snapshot_kv()
            current = torch.npu.current_stream()
            warmup = torch.npu.Stream()
            try:
                warmup.wait_stream(current)
                with torch.npu.stream(warmup):
                    self.forward()
                current.wait_stream(warmup)
                self.restore_kv(saved)
                graph = torch.npu.NPUGraph()
                midpoint = WIDTH // 2 if self.split_at_eos else WIDTH
                with torch.npu.graph(graph):
                    hidden, packed, state = self.forward(end=midpoint)
                if self.split_at_eos:
                    self.midpoint_state = state
                    self.second_graph = torch.npu.NPUGraph()
                    with torch.npu.graph(self.second_graph):
                        self.second_hidden, self.second_packed, _ = self.forward(start=midpoint, state=state)
            finally:
                current.wait_stream(warmup)
                self.restore_kv(saved)
            self.graph, self.hidden, self.packed = graph, hidden, packed
            logger.info(
                "[talker-fused-c1] captured %d live-weight frames split=%s capacity=%d",
                WIDTH,
                self.split_at_eos,
                self.capacity,
            )
        self.graph.replay()
        rows = self.packed.cpu().tolist()
        if not self.split_at_eos:
            return self.hidden, rows
        self.tail_steps += 1
        if rows[-1][2] or any(row[3] for row in rows):
            self.tail_skips += 1
            logger.info_once("[talker-fused-c1] first half early exit")
            hidden_rows = self.hidden[-1:].expand(WIDTH // 2, -1)
            return torch.cat((self.hidden, hidden_rows)), rows + [[self.eos, 0, 1, 0]] * (WIDTH // 2)
        self.second_graph.replay()
        return torch.cat((self.hidden, self.second_hidden)), rows + self.second_packed.cpu().tolist()


def summarize(packed):
    codes = [token for token, emit, _, _ in packed if emit]
    stops = [bool(stop) for _, _, stop, _ in packed]
    guard = any(flag for _, _, _, flag in packed)
    draws = len(codes) + int(any(stops))
    return codes, stops, guard, draws


def commit(talker, live, hidden, packed, codes, stops, draws):
    state, sampler = live.state, live.sampler
    for _ in range(draws):
        sampler["generator"].random()
    sampler["step"] += draws
    sampler["finished"] = stops[-1]
    sampler["history"].extend(codes)
    del sampler["history"][:-HISTORY_WINDOW]
    state["step"] = sampler["step"]
    state["finished"] = stops[-1]
    full_ids = state.get("codec_full_ids")
    if isinstance(full_ids, list):
        full_ids.extend(token for token, _, _, _ in packed[:draws])
    delta = torch.tensor(codes, dtype=torch.long).reshape(-1, 1)
    if codes:
        state["last_code"] = codes[-1]
        ids = talker._host_sample_device_ids
        state["_kstep_last_code"] = ids.narrow(0, codes[-1], 1)
        talker._request_codec_history.setdefault(live.request_id, []).extend(codes)
        prior_codes = state.get("codes")
        if not isinstance(prior_codes, torch.Tensor):
            prior_codes = delta.new_empty(0)
        state["codes"] = torch.cat((prior_codes.reshape(-1), delta.reshape(-1)))[-HISTORY_WINDOW:]
    live.info["audio_state"] = state
    token = packed[draws - 1][0]
    live.info["audio_codes"] = {
        "current": talker._host_sample_device_ids.narrow(0, token, 1),
        "accumulated": state["codes"],
    }
    row_continue, row_stop, flag_false, flag_true, _ = talker._step_constants(hidden)
    talker.set_batch_stop_logits(torch.stack([row_stop if stop else row_continue for stop in stops]))
    return OmniOutput(
        text_hidden_states=hidden.clone(),
        multimodal_outputs={
            "codes": {"audio": [delta]},
            "meta": {
                "finished": [flag_true if stops[-1] else flag_false],
                "codec_frame_valid": [torch.ones(len(codes), dtype=torch.bool)],
            },
        },
    )


def check_shadow(talker, result, live, packed, kv, shadow_slots, graph):
    output = result.multimodal_outputs
    expected = output["codes"]["audio"][0].reshape(-1).tolist()
    codes, stops, guard, draws = summarize(packed)
    actual_stops = talker.batch_stop_logits.argmax(dim=-1).bool().cpu().tolist()
    last_finished = bool(output["meta"]["finished"][0].item())
    stop_match = stops == actual_stops and stops[-1] == last_finished
    # The wide loop speculatively writes all reserved rows. Compare the
    # prefix that scheduler can commit, including the query producing EOS.
    actual_kv = [cache.index_select(0, shadow_slots)[:draws] for cache in graph.kv]
    kv_max = torch.stack([(a - b[:draws]).abs().amax() for a, b in zip(actual_kv, kv, strict=True)])
    max_error = float(kv_max.max().item())
    logger.info(
        "[talker-fused-c1] shadow codes=%d match=%s stop_match=%s guard=%s kv_max_abs=%.6g req=%s",
        len(codes),
        codes == expected,
        stop_match,
        guard,
        max_error,
        live.request_id,
    )
    return codes == expected and stop_match and max_error == 0.0


def run_fused_step(runner, mode_path, batch_desc, model_inputs, batch, run_model):
    def baseline():
        return runner._run_kstep_frames(model_inputs, batch, run_model=run_model)

    try:
        mode = Path(mode_path).read_text().strip()
    except FileNotFoundError:
        return baseline()
    if mode not in (
        "shadow",
        "active",
        "guard",
        "shadow_reference",
        "reference",
        "shadow_tail",
        "tail",
        "shadow_capacity",
        "tail_capacity",
        "shadow_prefill_capacity",
        "tail_prefill_capacity",
        "shadow_thinker_prefill",
        "active_thinker_prefill",
    ) or getattr(runner, "_talker_fused_disabled", False):
        return baseline()
    reference = mode in ("shadow_reference", "reference")
    bucketed = mode in (
        "shadow_capacity",
        "tail_capacity",
        "shadow_prefill_capacity",
        "tail_prefill_capacity",
        "shadow_thinker_prefill",
        "active_thinker_prefill",
    )
    tail = bucketed or mode in ("shadow_tail", "tail")
    shadow = mode in ("shadow_reference", "shadow", "shadow_tail", "shadow_capacity")
    graph_name = "_talker_fused_reference_graph" if reference else "_talker_fused_graph"
    if tail:
        graph_name = "_talker_fused_tail_graph"
    model_len = runner.model_config.max_model_len
    hybrid = hybrid_enabled(model_len)
    if model_len != CAPACITY and not hybrid:
        return baseline()
    length = None
    if hybrid and batch.num_reqs == 1:
        length = scheduled_length(batch, 1, WIDTH)
        if length is None or length > CAPACITY:
            trace_context(batch, "wide_c1", length)
            return baseline()
    talker = getattr(runner.model, "talker", runner.model)
    live = prepare_step(runner, talker, batch_desc, model_inputs, batch)
    if live is None:
        logger.info_once("[talker-fused-c1] eligibility declined at least one step")
        return baseline()

    capacity = CAPACITY
    if bucketed:
        lengths = getattr(batch, "seq_lens_np", None)
        block_size = runner.vllm_config.cache_config.block_size
        if lengths is not None and len(lengths) >= 1 and block_size == 128:
            length = int(lengths[0])
            if WIDTH <= length <= 256:
                capacity = 128 if length <= 128 else 256
                graph_name = f"_talker_fused_tail_graph_{capacity}"
    accepted = graph_name + "_shadow_passed"
    rejected = graph_name + "_shadow_failed"
    if not shadow:
        if getattr(runner, rejected, False):
            return baseline()
        # Active deployments qualify each graph locally on its first use.
        shadow = not getattr(runner, accepted, False)

    graph = getattr(runner, graph_name, None)
    if graph is None:
        sampler_factory = make_device_sampler
        if reference:
            from vllm_omni.platforms.npu.worker.talker_fused_sampling_reference import (
                make_device_sampler as sampler_factory,
            )
        try:
            graph = TalkerFusedGraph(runner, talker, live, sampler_factory, split_at_eos=tail, capacity=capacity)
        except ValueError:
            runner._talker_fused_disabled = True
            logger.exception("[talker-fused-c1] graph setup declined")
            return baseline()
        setattr(runner, graph_name, graph)
    started = perf_counter_ns()
    if tuple(live.metadata) != graph.layer_names:
        return baseline()
    graph.bind(live)
    saved = graph.snapshot_kv() if shadow else None
    try:
        hidden, packed = graph.run()
        codes, stops, guard, draws = summarize(packed)
        if not shadow:
            if guard or mode == "guard":
                # Every speculative write belongs to this step's reserved
                # range. The baseline overwrites the consumed prefix and
                # the scheduler drops the remainder on EOS.
                logger.info("[talker-fused-c1] numeric fallback req=%s", live.request_id)
                return baseline()
            runner._kstep_ran_last = True
            graph.replay_ms.append((perf_counter_ns() - started) / 1e6)
            if len(graph.replay_ms) == 32:
                logger.info(
                    "[talker-fused-c1] 32 replay+bind+readback ms p50=%.3f mean=%.3f",
                    statistics.median(graph.replay_ms),
                    statistics.mean(graph.replay_ms),
                )
                graph.replay_ms.clear()
                if tail:
                    logger.info("[talker-fused-c1] split steps=%d early_exits=%d", graph.tail_steps, graph.tail_skips)
            logger.info_once("[talker-fused-c1] active replay enabled mode=%s", mode)
            if hybrid:
                trace_context(batch, "fused_c1", length, graph.capacity)
            return commit(talker, live, hidden, packed, codes, stops, draws)
        assert saved is not None
        slots = saved[0]
        shadow_kv = [cache.index_select(0, slots) for cache in graph.kv]
    except (RuntimeError, TypeError, AttributeError):
        runner._talker_fused_disabled = True
        logger.exception("[talker-fused-c1] capture or replay failed")
        if saved is not None:
            graph.restore_kv(saved)
        return baseline()
    graph.restore_kv(saved)
    result = baseline()
    ok = check_shadow(talker, result, live, packed, shadow_kv, slots, graph)
    if not guard:
        if ok:
            setattr(runner, accepted, True)
        else:
            setattr(runner, rejected, True)
            logger.warning("[talker-fused-c1] shadow discrepancy requires investigation")
    return result
