#!/usr/bin/env python3
"""Microbenchmark Ming CFM graph variants without torch profiler overhead."""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import torch
from ming_tts_talker_profile import build_inputs, load_talker, run_ar_profile


@contextmanager
def force_original_sde_graph(generator):
    """Force the pre-optimization graph path for temperature=0 comparisons."""
    original_get_sampler_pool = generator._get_sampler_pool

    def _get_sampler_pool(batch_size, device, deterministic_sde_noise):
        return original_get_sampler_pool(batch_size, device, False)

    generator._get_sampler_pool = _get_sampler_pool
    try:
        yield
    finally:
        generator._get_sampler_pool = original_get_sampler_pool


def _cat_latents(latents: list[list[torch.Tensor]]) -> torch.Tensor:
    return torch.cat(latents[0], dim=1)


def _cat_all_latents(latents: list[list[torch.Tensor]]) -> torch.Tensor:
    return torch.cat([torch.cat(item, dim=1) for item in latents], dim=0)


def _waveform_diff(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    ref = reference.detach().float().flatten()
    cand = candidate.detach().float().flatten()
    overlap = min(ref.numel(), cand.numel())
    if overlap == 0:
        return {
            "reference_samples": int(ref.numel()),
            "candidate_samples": int(cand.numel()),
            "overlap_samples": int(overlap),
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rmse": 0.0,
        }
    diff = (ref[:overlap] - cand[:overlap]).abs()
    return {
        "reference_samples": int(ref.numel()),
        "candidate_samples": int(cand.numel()),
        "overlap_samples": int(overlap),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
    }


def _run(generator, inputs, args, *, force_original: bool):
    ctx = force_original_sde_graph(generator) if force_original else nullcontext()
    with ctx:
        return run_ar_profile(
            generator,
            inputs,
            max_steps=args.steps,
            use_static_cache=True,
            device=args.device,
        )


def _mean(values: list[float]) -> float:
    return float(stats.mean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(stats.stdev(values)) if len(values) > 1 else 0.0


def _summarize_samples(samples: dict[str, dict[str, list[float]]]) -> dict:
    return {
        name: {
            key: {
                "mean": _mean(values),
                "stdev": _stdev(values),
                "samples": values,
            }
            for key, values in metric.items()
        }
        for name, metric in samples.items()
    }


def _iter_qwen2_mlp_modules(model):
    for module in model.modules():
        if all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")):
            yield module


def _ensure_fused_llm_mlp(model) -> int:
    """Pack Qwen2 MLP gate/up projections for a profiling-only A/B probe."""
    count = 0
    for mlp in _iter_qwen2_mlp_modules(model):
        if not hasattr(mlp, "_ming_original_forward"):
            mlp._ming_original_forward = mlp.forward

        if not hasattr(mlp, "_ming_fused_gate_up_proj"):
            gate_proj = mlp.gate_proj
            up_proj = mlp.up_proj
            has_bias = gate_proj.bias is not None or up_proj.bias is not None
            fused = torch.nn.Linear(
                gate_proj.in_features,
                gate_proj.out_features + up_proj.out_features,
                bias=has_bias,
                device=gate_proj.weight.device,
                dtype=gate_proj.weight.dtype,
            )
            with torch.no_grad():
                fused.weight.copy_(torch.cat([gate_proj.weight, up_proj.weight], dim=0))
                if has_bias:
                    gate_bias = (
                        gate_proj.bias
                        if gate_proj.bias is not None
                        else torch.zeros(
                            gate_proj.out_features, device=gate_proj.weight.device, dtype=gate_proj.weight.dtype
                        )
                    )
                    up_bias = (
                        up_proj.bias
                        if up_proj.bias is not None
                        else torch.zeros(up_proj.out_features, device=up_proj.weight.device, dtype=up_proj.weight.dtype)
                    )
                    fused.bias.copy_(torch.cat([gate_bias, up_bias], dim=0))
            fused.requires_grad_(False)
            mlp.add_module("_ming_fused_gate_up_proj", fused)
        count += 1
    return count


def _set_fused_llm_mlp_enabled(model, enabled: bool) -> int:
    count = _ensure_fused_llm_mlp(model)

    def fused_forward(self, x):
        gate_up = self._ming_fused_gate_up_proj(x)
        gate, up = gate_up.split((self.gate_proj.out_features, self.up_proj.out_features), dim=-1)
        return self.down_proj(self.act_fn(gate) * up)

    for mlp in _iter_qwen2_mlp_modules(model):
        if enabled:
            mlp.forward = MethodType(fused_forward, mlp)
        else:
            mlp.forward = mlp._ming_original_forward
    return count


def _run_full_history_recompute(generator, inputs, args):
    """Run the AR loop without KV cache by recomputing the full prefix.

    Unlike the existing use_static_cache=False path, this feeds the prompt plus
    every generated embedding back into the LLM, so it preserves the causal
    context semantics at the cost of more LLM work.
    """
    cfg = generator.cfg_strength
    dtype = next(generator._model.parameters()).dtype
    batch_size = inputs.shape[0]
    device = args.device
    his_lat = torch.zeros(batch_size, generator.his_patch_size, generator.latent_dim, device=device, dtype=dtype)
    generated_embeds: list[torch.Tensor] = []
    latents_by_request: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    llm_s: list[float] = []
    cfm_s: list[float] = []
    collect_s: list[float] = []

    with torch.no_grad():
        for step in range(args.steps):
            torch.accelerator.synchronize(device)
            t0 = time.perf_counter()
            full_inputs = torch.cat([inputs, *generated_embeds], dim=1) if generated_embeds else inputs
            outputs = generator._model(inputs_embeds=full_inputs, use_cache=False)
            last_hs = outputs.last_hidden_state[:, -1:, :]
            torch.accelerator.synchronize(device)
            t1 = time.perf_counter()

            gen_lat, next_inputs, _stop_out = generator.cfm_sample_step(last_hs, his_lat, cfg=cfg)
            torch.accelerator.synchronize(device)
            t2 = time.perf_counter()

            for row in range(batch_size):
                latents_by_request[row].append(gen_lat[row : row + 1])
            his_lat = generator._update_his_lat(his_lat, gen_lat)
            generated_embeds.append(next_inputs)
            torch.accelerator.synchronize(device)
            t3 = time.perf_counter()

            if step > 0:
                llm_s.append(t1 - t0)
                cfm_s.append(t2 - t1)
                collect_s.append(t3 - t2)

    return latents_by_request, {
        "llm_decode_ms": _mean(llm_s) * 1000.0,
        "cfm_step_ms": _mean(cfm_s) * 1000.0,
        "collect_ms": _mean(collect_s) * 1000.0,
    }


def run_full_history_recompute_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=False)

    samples = {
        "static_cache": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        "full_history_recompute": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
    }
    for idx in range(args.repeats):
        torch.manual_seed(1000 + idx)
        _latents, timers = _run(generator, inputs, args, force_original=False)
        for key in samples["static_cache"]:
            samples["static_cache"][key].append(float(timers[key]))

    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run_full_history_recompute(generator, inputs, args)

    torch.manual_seed(1234)
    opt_latents, opt_timers = _run_full_history_recompute(generator, inputs, args)
    diff = (_cat_latents(baseline_latents) - _cat_latents(opt_latents)).float().abs()

    for idx in range(args.repeats):
        torch.manual_seed(1000 + idx)
        _latents, timers = _run_full_history_recompute(generator, inputs, args)
        for key in samples["full_history_recompute"]:
            samples["full_history_recompute"][key].append(float(timers[key]))

    return {
        "mode": "full_history_recompute_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
        },
        "timers": _summarize_samples(samples),
    }


def run_llm_recompute_semantics_probe(generator, inputs, args, llm_config, talker_cfg) -> dict:
    """Compare StaticCache and full-prefix LLM outputs on the same prefixes."""
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()

    dtype = next(generator._model.parameters()).dtype
    batch_size = inputs.shape[0]
    device = torch.device(args.device)

    torch.manual_seed(1234)
    his_lat = torch.zeros(batch_size, generator.his_patch_size, generator.latent_dim, device=device, dtype=dtype)
    past_key_values, max_cache_len = generator._init_batched_kv_cache(batch_size, True, device, dtype)
    current_inputs = inputs
    forced_inputs: list[torch.Tensor] = []

    with torch.no_grad():
        for step in range(min(args.steps, max_cache_len - inputs.shape[1])):
            last_hs = generator.llm_step(
                current_inputs,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=True,
            )
            gen_lat, next_inputs, _stop_out = generator.cfm_sample_step(last_hs, his_lat, cfg=generator.cfg_strength)
            forced_inputs.append(next_inputs.detach())
            his_lat = generator._update_his_lat(his_lat, gen_lat)
            current_inputs = next_inputs

    generator._sampler_pools.clear()
    past_key_values, _ = generator._init_batched_kv_cache(batch_size, True, device, dtype)
    hidden_diffs: list[float] = []
    hidden_mean_diffs: list[float] = []
    static_s: list[float] = []
    full_s: list[float] = []

    with torch.no_grad():
        for step in range(len(forced_inputs)):
            current_inputs = inputs if step == 0 else forced_inputs[step - 1]
            full_inputs = torch.cat([inputs, *forced_inputs[:step]], dim=1) if step > 0 else inputs

            torch.accelerator.synchronize(device)
            t0 = time.perf_counter()
            static_hs = generator.llm_step(
                current_inputs,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=True,
            )
            torch.accelerator.synchronize(device)
            t1 = time.perf_counter()

            full_hs = generator._model(inputs_embeds=full_inputs, use_cache=False).last_hidden_state[:, -1:, :]
            torch.accelerator.synchronize(device)
            t2 = time.perf_counter()

            if step > 0:
                static_s.append(t1 - t0)
                full_s.append(t2 - t1)
            diff = (static_hs - full_hs).float().abs()
            hidden_diffs.append(float(diff.max().item()))
            hidden_mean_diffs.append(float(diff.mean().item()))

    return {
        "mode": "llm_recompute_semantics_probe",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": len(forced_inputs),
        },
        "equivalence": {
            "max_abs_hidden_diff": max(hidden_diffs) if hidden_diffs else 0.0,
            "mean_abs_hidden_diff": _mean(hidden_mean_diffs),
            "per_step_max_abs_hidden_diff": hidden_diffs,
            "per_step_mean_abs_hidden_diff": hidden_mean_diffs,
        },
        "timers": {
            "static_cache_llm_decode_ms": {
                "mean": _mean(static_s) * 1000.0,
                "stdev": _stdev([v * 1000.0 for v in static_s]),
                "samples": [v * 1000.0 for v in static_s],
            },
            "full_prefix_llm_decode_ms": {
                "mean": _mean(full_s) * 1000.0,
                "stdev": _stdev([v * 1000.0 for v in full_s]),
                "samples": [v * 1000.0 for v in full_s],
            },
        },
    }


def _collect_teacher_forced_inputs(generator, inputs, args) -> list[torch.Tensor]:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()

    dtype = next(generator._model.parameters()).dtype
    batch_size = inputs.shape[0]
    device = torch.device(args.device)
    his_lat = torch.zeros(batch_size, generator.his_patch_size, generator.latent_dim, device=device, dtype=dtype)
    past_key_values, max_cache_len = generator._init_batched_kv_cache(batch_size, True, device, dtype)
    current_inputs = inputs
    forced_inputs: list[torch.Tensor] = []

    with torch.no_grad():
        for step in range(min(args.steps, max_cache_len - inputs.shape[1])):
            last_hs = generator.llm_step(
                current_inputs,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=True,
            )
            gen_lat, next_inputs, _stop_out = generator.cfm_sample_step(last_hs, his_lat, cfg=generator.cfg_strength)
            forced_inputs.append(next_inputs.detach())
            his_lat = generator._update_his_lat(his_lat, gen_lat)
            current_inputs = next_inputs
    return forced_inputs


def _run_static_cache_teacher_forced(generator, inputs, forced_inputs, args):
    dtype = next(generator._model.parameters()).dtype
    batch_size = inputs.shape[0]
    device = torch.device(args.device)
    past_key_values, _ = generator._init_batched_kv_cache(batch_size, True, device, dtype)
    hidden_states: list[torch.Tensor] = []
    times: list[float] = []

    with torch.no_grad():
        for step in range(len(forced_inputs)):
            current_inputs = inputs if step == 0 else forced_inputs[step - 1]
            torch.accelerator.synchronize(device)
            t0 = time.perf_counter()
            hidden = generator.llm_step(
                current_inputs,
                step=step,
                past_key_values=past_key_values,
                use_static_cache=True,
            )
            torch.accelerator.synchronize(device)
            t1 = time.perf_counter()
            hidden_states.append(hidden.detach().clone())
            if step > 0:
                times.append(t1 - t0)
    return hidden_states, times


def _snapshot_static_cache(past_key_values):
    snapshot = []
    for layer in getattr(past_key_values, "layers", []):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        cumulative_length = getattr(layer, "cumulative_length", None)
        snapshot.append(
            (
                keys.detach().clone() if keys is not None else None,
                values.detach().clone() if values is not None else None,
                cumulative_length.detach().clone() if cumulative_length is not None else None,
            )
        )
    return snapshot


def _restore_static_cache(past_key_values, snapshot) -> None:
    for layer, (keys, values, cumulative_length) in zip(getattr(past_key_values, "layers", []), snapshot):
        if keys is not None:
            layer.keys.copy_(keys)
        if values is not None:
            layer.values.copy_(values)
        if cumulative_length is not None:
            layer.cumulative_length.copy_(cumulative_length)


def _run_llm_decode_graph_teacher_forced(generator, inputs, forced_inputs, args):
    dtype = next(generator._model.parameters()).dtype
    batch_size = inputs.shape[0]
    device = torch.device(args.device)
    past_key_values, _ = generator._init_batched_kv_cache(batch_size, True, device, dtype)
    hidden_states: list[torch.Tensor] = []
    times: list[float] = []

    with torch.no_grad():
        prefill = generator.llm_step(
            inputs,
            step=0,
            past_key_values=past_key_values,
            use_static_cache=True,
        )
        hidden_states.append(prefill.detach().clone())
        prompt_cache_snapshot = _snapshot_static_cache(past_key_values)

        if len(forced_inputs) <= 1:
            return hidden_states, times

        input_ph = torch.empty_like(forced_inputs[0])
        cache_pos_ph = torch.empty((forced_inputs[0].shape[1],), device=device, dtype=torch.long)
        input_ph.copy_(forced_inputs[0])
        cache_pos_ph.copy_(torch.arange(inputs.shape[1], inputs.shape[1] + forced_inputs[0].shape[1], device=device))

        graph_stream = torch.cuda.Stream(device=device)
        graph_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(graph_stream):
            for _ in range(3):
                graph_outputs = generator._model(
                    past_key_values=past_key_values,
                    inputs_embeds=input_ph,
                    use_cache=True,
                    cache_position=cache_pos_ph,
                )
        torch.cuda.current_stream(device).wait_stream(graph_stream)
        torch.accelerator.synchronize(device)
        _restore_static_cache(past_key_values, prompt_cache_snapshot)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_outputs = generator._model(
                past_key_values=past_key_values,
                inputs_embeds=input_ph,
                use_cache=True,
                cache_position=cache_pos_ph,
            )
            graph_hidden = graph_outputs.last_hidden_state[:, -1:, :]
        torch.accelerator.synchronize(device)
        _restore_static_cache(past_key_values, prompt_cache_snapshot)

        positions = torch.arange(
            inputs.shape[1],
            inputs.shape[1] + len(forced_inputs) * forced_inputs[0].shape[1],
            device=device,
            dtype=torch.long,
        )

        for step in range(1, len(forced_inputs)):
            input_ph.copy_(forced_inputs[step - 1])
            start = (step - 1) * forced_inputs[0].shape[1]
            cache_pos_ph.copy_(positions[start : start + forced_inputs[0].shape[1]])
            torch.accelerator.synchronize(device)
            t0 = time.perf_counter()
            graph.replay()
            torch.accelerator.synchronize(device)
            t1 = time.perf_counter()
            hidden_states.append(graph_hidden.detach().clone())
            times.append(t1 - t0)
    return hidden_states, times


def run_llm_decode_graph_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    torch.manual_seed(1234)
    forced_inputs = _collect_teacher_forced_inputs(generator, inputs, args)
    ref_hidden, ref_times = _run_static_cache_teacher_forced(generator, inputs, forced_inputs, args)
    graph_hidden, graph_times = _run_llm_decode_graph_teacher_forced(generator, inputs, forced_inputs, args)
    pair_count = min(len(ref_hidden), len(graph_hidden))
    diffs = [(ref_hidden[i] - graph_hidden[i]).float().abs() for i in range(pair_count)]
    return {
        "mode": "llm_decode_graph_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": len(forced_inputs),
            "timed_decode_steps": len(graph_times),
        },
        "equivalence": {
            "max_abs_hidden_diff": max((float(d.max().item()) for d in diffs), default=0.0),
            "mean_abs_hidden_diff": _mean([float(d.mean().item()) for d in diffs]),
            "per_step_max_abs_hidden_diff": [float(d.max().item()) for d in diffs],
            "per_step_mean_abs_hidden_diff": [float(d.mean().item()) for d in diffs],
        },
        "timers": {
            "static_cache_llm_decode_ms": {
                "mean": _mean(ref_times) * 1000.0,
                "stdev": _stdev([v * 1000.0 for v in ref_times]),
                "samples": [v * 1000.0 for v in ref_times],
            },
            "cuda_graph_llm_decode_ms": {
                "mean": _mean(graph_times) * 1000.0,
                "stdev": _stdev([v * 1000.0 for v in graph_times]),
                "samples": [v * 1000.0 for v in graph_times],
            },
        },
    }


def run_llm_decode_graph_ar_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._maybe_pack_qkv_projections()
    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()

    original_enabled = getattr(generator, "_llm_decode_graph_enabled", False)
    try:
        torch.manual_seed(1234)
        generator._llm_decode_graph_enabled = False
        generator._llm_decode_graphs.clear()
        baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=False)

        torch.manual_seed(1234)
        generator._llm_decode_graph_enabled = True
        generator._llm_decode_graphs.clear()
        graph_latents, graph_timers = _run(generator, inputs, args, force_original=False)
    finally:
        generator._llm_decode_graph_enabled = original_enabled
        generator._llm_decode_graphs.clear()

    diff = (_cat_latents(baseline_latents) - _cat_latents(graph_latents)).float().abs()
    samples = {
        "static_cache": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        "llm_decode_graph": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
    }
    torch.manual_seed(0)
    generator._llm_decode_graph_enabled = True
    generator._llm_decode_graphs.clear()
    _run(generator, inputs, args, force_original=False)

    for idx in range(args.repeats):
        for name, enabled in (("static_cache", False), ("llm_decode_graph", True)):
            torch.manual_seed(1000 + idx)
            generator._llm_decode_graph_enabled = enabled
            _latents, timers = _run(generator, inputs, args, force_original=False)
            for key in samples[name]:
                samples[name][key].append(float(timers[key]))

    generator._llm_decode_graph_enabled = original_enabled
    generator._llm_decode_graphs.clear()
    return {
        "mode": "llm_decode_graph_ar_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": graph_timers,
        },
        "timers": _summarize_samples(samples),
    }


def run_vae_stream_full_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()

    torch.manual_seed(1234)
    latents_by_request, ar_timers = _run(generator, inputs[:1], args, force_original=False)
    latents = latents_by_request[0]
    sr = int(generator._audio_vae.config.sample_rate)

    with torch.no_grad():
        generator._vae_stream_graph_enabled = False
        stream_wav = generator.decode_to_waveform(latents, stream_decode=True)
        generator._vae_stream_graph_enabled = True
        graph_stream_wav = generator.decode_to_waveform(latents, stream_decode=True)
        generator._vae_stream_graph_enabled = False
        full_raw_wav = generator.decode_to_waveform(latents, stream_decode=False)
        full_trimmed_wav = generator.trim_trailing_silence(full_raw_wav)

    samples = {
        "stream_decode_ms": [],
        "graph_stream_decode_ms": [],
        "full_decode_raw_ms": [],
        "full_decode_trimmed_ms": [],
    }
    for _idx in range(args.repeats):
        torch.accelerator.synchronize(args.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            generator._vae_stream_graph_enabled = False
            generator.decode_to_waveform(latents, stream_decode=True)
        torch.accelerator.synchronize(args.device)
        t1 = time.perf_counter()

        with torch.no_grad():
            generator._vae_stream_graph_enabled = True
            generator.decode_to_waveform(latents, stream_decode=True)
            generator._vae_stream_graph_enabled = False
        torch.accelerator.synchronize(args.device)
        t_graph = time.perf_counter()

        with torch.no_grad():
            raw = generator.decode_to_waveform(latents, stream_decode=False)
        torch.accelerator.synchronize(args.device)
        t2 = time.perf_counter()

        with torch.no_grad():
            generator.trim_trailing_silence(raw)
        torch.accelerator.synchronize(args.device)
        t3 = time.perf_counter()

        samples["stream_decode_ms"].append((t1 - t0) * 1000.0)
        samples["graph_stream_decode_ms"].append((t_graph - t1) * 1000.0)
        samples["full_decode_raw_ms"].append((t2 - t_graph) * 1000.0)
        samples["full_decode_trimmed_ms"].append((t3 - t_graph) * 1000.0)

    return {
        "mode": "vae_stream_full_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
            "sample_rate": sr,
        },
        "workload": {
            "batch_size": 1,
            "input_tokens": inputs.shape[1],
            "profile_steps": len(latents),
            "latent_frames": int(sum(lat.shape[1] for lat in latents)),
            "repeats": args.repeats,
        },
        "ar_timers": ar_timers,
        "audio": {
            "stream_seconds": float(stream_wav.shape[-1]) / sr,
            "graph_stream_seconds": float(graph_stream_wav.shape[-1]) / sr,
            "full_raw_seconds": float(full_raw_wav.shape[-1]) / sr,
            "full_trimmed_seconds": float(full_trimmed_wav.shape[-1]) / sr,
            "stream_vs_graph_stream": _waveform_diff(stream_wav, graph_stream_wav),
            "stream_vs_full_raw": _waveform_diff(stream_wav, full_raw_wav),
            "stream_vs_full_trimmed": _waveform_diff(stream_wav, full_trimmed_wav),
        },
        "timers": {
            name: {
                "mean": _mean(values),
                "stdev": _stdev(values),
                "samples": values,
            }
            for name, values in samples.items()
        },
    }


def run_vae_qwen_graph_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()

    torch.manual_seed(1234)
    latents_by_request, ar_timers = _run(generator, inputs[:1], args, force_original=False)
    latents = latents_by_request[0]
    sr = int(generator._audio_vae.config.sample_rate)
    vae_decoder = generator._audio_vae.decoder
    original_enabled = vae_decoder._qwen_decode_graph_enabled

    try:
        with torch.no_grad():
            generator._vae_stream_graph_enabled = False
            vae_decoder._qwen_decode_graph_enabled = False
            vae_decoder._qwen_decode_graphs.clear()
            baseline_wav = generator.decode_to_waveform(latents, stream_decode=True)

            vae_decoder._qwen_decode_graph_enabled = True
            vae_decoder._qwen_decode_graphs.clear()
            graph_wav = generator.decode_to_waveform(latents, stream_decode=True)

        samples = {
            "stream_decode_ms": [],
            "qwen_graph_stream_decode_ms": [],
        }
        graph_shapes = [str(key[0]) for key in vae_decoder._qwen_decode_graphs]
        for _idx in range(args.repeats):
            vae_decoder._qwen_decode_graph_enabled = False
            torch.accelerator.synchronize(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                generator.decode_to_waveform(latents, stream_decode=True)
            torch.accelerator.synchronize(args.device)
            t1 = time.perf_counter()

            vae_decoder._qwen_decode_graph_enabled = True
            with torch.no_grad():
                generator.decode_to_waveform(latents, stream_decode=True)
            torch.accelerator.synchronize(args.device)
            t2 = time.perf_counter()

            samples["stream_decode_ms"].append((t1 - t0) * 1000.0)
            samples["qwen_graph_stream_decode_ms"].append((t2 - t1) * 1000.0)
    finally:
        vae_decoder._qwen_decode_graph_enabled = original_enabled
        vae_decoder._qwen_decode_graphs.clear()
        generator._vae_stream_graph_enabled = False

    return {
        "mode": "vae_qwen_graph_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
            "sample_rate": sr,
        },
        "workload": {
            "batch_size": 1,
            "input_tokens": inputs.shape[1],
            "profile_steps": len(latents),
            "latent_frames": int(sum(lat.shape[1] for lat in latents)),
            "repeats": args.repeats,
        },
        "ar_timers": ar_timers,
        "audio": {
            "baseline_seconds": float(baseline_wav.shape[-1]) / sr,
            "qwen_graph_seconds": float(graph_wav.shape[-1]) / sr,
            "stream_vs_qwen_graph": _waveform_diff(baseline_wav, graph_wav),
        },
        "graph_shapes": graph_shapes,
        "timers": {
            name: {
                "mean": _mean(values),
                "stdev": _stdev(values),
                "samples": values,
            }
            for name, values in samples.items()
        },
    }


def _vae_qwen_stream_inputs(generator, latents: list[torch.Tensor]) -> list[torch.Tensor]:
    decoder = generator._audio_vae.decoder
    decode_pad: torch.Tensor | None = None
    qwen_inputs: list[torch.Tensor] = []

    with torch.no_grad():
        for lat in latents:
            vae_input = torch.cat([decode_pad, lat], dim=1) if decode_pad is not None else lat
            x = decoder.fc1(vae_input)
            if decoder.patch_size != -1:
                x = decoder.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)
            qwen_inputs.append(x)
            decode_pad = vae_input[:, -generator._vae_decode_pad_frames :, :].detach()

    return qwen_inputs


def run_vae_qwen_hidden_graph_probe(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()

    torch.manual_seed(1234)
    latents_by_request, ar_timers = _run(generator, inputs[:1], args, force_original=False)
    latents = latents_by_request[0]
    vae_decoder = generator._audio_vae.decoder
    original_enabled = vae_decoder._qwen_decode_graph_enabled
    original_attn_impl = getattr(vae_decoder.decoder.config, "_attn_implementation", None)
    effective_attn_impl = args.vae_qwen_attn_impl or original_attn_impl

    rows = []
    try:
        if args.vae_qwen_attn_impl:
            vae_decoder.decoder.config._attn_implementation = args.vae_qwen_attn_impl
        qwen_inputs = _vae_qwen_stream_inputs(generator, latents)
        for idx, x in enumerate(qwen_inputs):
            row = {
                "idx": idx,
                "shape": list(x.shape),
                "stride": list(x.stride()),
                "is_contiguous": bool(x.is_contiguous()),
            }

            for label, candidate in (("original_stride", x), ("contiguous", x.contiguous())):
                vae_decoder._qwen_decode_graph_enabled = False
                vae_decoder._qwen_decode_graphs.clear()
                eager = vae_decoder.decoder(inputs_embeds=candidate, use_cache=False).last_hidden_state

                vae_decoder._qwen_decode_graph_enabled = True
                vae_decoder._qwen_decode_graphs.clear()
                graph = vae_decoder._execute_qwen_decode_graph(candidate)
                graph_again = vae_decoder._execute_qwen_decode_graph(candidate)

                eager_graph_diff = (eager - graph).float().abs()
                graph_self_diff = (graph - graph_again).float().abs()
                row[f"{label}_eager_graph_max_abs"] = float(eager_graph_diff.max().item())
                row[f"{label}_eager_graph_mean_abs"] = float(eager_graph_diff.mean().item())
                row[f"{label}_graph_self_max_abs"] = float(graph_self_diff.max().item())
                row[f"{label}_graph_self_mean_abs"] = float(graph_self_diff.mean().item())
            rows.append(row)
    finally:
        if original_attn_impl is not None:
            vae_decoder.decoder.config._attn_implementation = original_attn_impl
        vae_decoder._qwen_decode_graph_enabled = original_enabled
        vae_decoder._qwen_decode_graphs.clear()

    return {
        "mode": "vae_qwen_hidden_graph_probe",
        "attn_implementation": effective_attn_impl,
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": 1,
            "input_tokens": inputs.shape[1],
            "profile_steps": len(latents),
            "latent_frames": int(sum(lat.shape[1] for lat in latents)),
        },
        "ar_timers": ar_timers,
        "rows": rows,
        "max_original_stride_eager_graph": max(
            (row["original_stride_eager_graph_max_abs"] for row in rows), default=0.0
        ),
        "max_contiguous_eager_graph": max((row["contiguous_eager_graph_max_abs"] for row in rows), default=0.0),
        "max_original_stride_graph_self": max((row["original_stride_graph_self_max_abs"] for row in rows), default=0.0),
        "max_contiguous_graph_self": max((row["contiguous_graph_self_max_abs"] for row in rows), default=0.0),
    }


def run_fused_llm_mlp_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._maybe_pack_qkv_projections()
    generator._cfm.use_preembedded_cfg = True
    generator._cfm.use_precomputed_temb = True
    generator._llm_decode_graph_enabled = True
    generator._sampler_pools.clear()

    original_graph_enabled = getattr(generator, "_llm_decode_graph_enabled", False)
    original_graphs = generator._llm_decode_graphs
    mode_graphs: dict[str, dict] = {"baseline": {}, "fused_llm_mlp": {}}
    mlp_count = _ensure_fused_llm_mlp(generator._model)

    def run_mode(name: str, enabled: bool, seed: int):
        _set_fused_llm_mlp_enabled(generator._model, enabled)
        generator._llm_decode_graphs = mode_graphs[name]
        torch.manual_seed(seed)
        return _run(generator, inputs, args, force_original=False)

    try:
        run_mode("baseline", False, 0)
        baseline_latents, baseline_timers = run_mode("baseline", False, 1234)

        run_mode("fused_llm_mlp", True, 0)
        opt_latents, opt_timers = run_mode("fused_llm_mlp", True, 1234)
        diff = (_cat_all_latents(baseline_latents) - _cat_all_latents(opt_latents)).float().abs()

        samples = {
            "baseline": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
            "fused_llm_mlp": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        }
        order = (
            (("fused_llm_mlp", True), ("baseline", False))
            if args.reverse_order
            else (
                ("baseline", False),
                ("fused_llm_mlp", True),
            )
        )
        for idx in range(args.repeats):
            for name, enabled in order:
                _latents, timers = run_mode(name, enabled, 1000 + idx)
                for key in samples[name]:
                    samples[name][key].append(float(timers[key]))
    finally:
        _set_fused_llm_mlp_enabled(generator._model, False)
        generator._llm_decode_graph_enabled = original_graph_enabled
        generator._llm_decode_graphs = original_graphs
        generator._llm_decode_graphs.clear()
        generator._sampler_pools.clear()

    return {
        "mode": "fused_llm_mlp_compare_reverse_order" if args.reverse_order else "fused_llm_mlp_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "llm_intermediate": llm_config.intermediate_size,
            "fused_mlp_modules": mlp_count,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
        },
        "timers": _summarize_samples(samples),
    }


def run_fused_qkv_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = False
    generator._qkv_packed = False
    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=False)

    samples = {
        "separate_qkv": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        "packed_qkv": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
    }
    for idx in range(args.repeats):
        torch.manual_seed(1000 + idx)
        _latents, timers = _run(generator, inputs, args, force_original=False)
        for key in samples["separate_qkv"]:
            samples["separate_qkv"][key].append(float(timers[key]))

    generator._sampler_pools.clear()
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    opt_latents, opt_timers = _run(generator, inputs, args, force_original=False)
    diff = (_cat_latents(baseline_latents) - _cat_latents(opt_latents)).float().abs()

    for idx in range(args.repeats):
        torch.manual_seed(1000 + idx)
        _latents, timers = _run(generator, inputs, args, force_original=False)
        for key in samples["packed_qkv"]:
            samples["packed_qkv"][key].append(float(timers[key]))

    return {
        "mode": "fused_qkv_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
        },
        "timers": _summarize_samples(samples),
    }


def run_prepared_cfg_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_prepared_cfg = False
    generator._cfm.use_preembedded_cfg = False
    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=False)

    samples = {
        "original_cfg": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        "prepared_cfg": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        "preembedded_cfg": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
    }
    for idx in range(args.repeats):
        torch.manual_seed(1000 + idx)
        _latents, timers = _run(generator, inputs, args, force_original=False)
        for key in samples["original_cfg"]:
            samples["original_cfg"][key].append(float(timers[key]))

    generator._cfm.use_prepared_cfg = True
    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    opt_latents, opt_timers = _run(generator, inputs, args, force_original=False)
    diff = (_cat_latents(baseline_latents) - _cat_latents(opt_latents)).float().abs()

    for idx in range(args.repeats):
        torch.manual_seed(1000 + idx)
        _latents, timers = _run(generator, inputs, args, force_original=False)
        for key in samples["prepared_cfg"]:
            samples["prepared_cfg"][key].append(float(timers[key]))

    generator._cfm.use_preembedded_cfg = True
    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    preembed_latents, preembed_timers = _run(generator, inputs, args, force_original=False)
    preembed_diff = (_cat_latents(baseline_latents) - _cat_latents(preembed_latents)).float().abs()

    for idx in range(args.repeats):
        torch.manual_seed(1000 + idx)
        _latents, timers = _run(generator, inputs, args, force_original=False)
        for key in samples["preembedded_cfg"]:
            samples["preembedded_cfg"][key].append(float(timers[key]))

    return {
        "mode": "prepared_cfg_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "preembedded_max_abs_latent_diff": float(preembed_diff.max().item()),
            "preembedded_mean_abs_latent_diff": float(preembed_diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
            "preembedded_timers": preembed_timers,
        },
        "timers": _summarize_samples(samples),
    }


def run_stop_logit_decision_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._cfm.use_precomputed_temb = True
    original_enabled = generator._stop_logit_decision_enabled

    try:
        generator._stop_logit_decision_enabled = False
        generator._sampler_pools.clear()
        torch.manual_seed(0)
        _run(generator, inputs, args, force_original=False)
        torch.manual_seed(1234)
        baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=False)

        generator._stop_logit_decision_enabled = True
        generator._sampler_pools.clear()
        torch.manual_seed(0)
        _run(generator, inputs, args, force_original=False)
        torch.manual_seed(1234)
        opt_latents, opt_timers = _run(generator, inputs, args, force_original=False)
        diff = (_cat_all_latents(baseline_latents) - _cat_all_latents(opt_latents)).float().abs()

        samples = {
            "stop_softmax": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
            "stop_logit_decision": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        }
        for idx in range(args.repeats):
            for name, enabled in (("stop_softmax", False), ("stop_logit_decision", True)):
                generator._stop_logit_decision_enabled = enabled
                generator._sampler_pools.clear()
                torch.manual_seed(1000 + idx)
                _latents, timers = _run(generator, inputs, args, force_original=False)
                for key in samples[name]:
                    samples[name][key].append(float(timers[key]))
    finally:
        generator._stop_logit_decision_enabled = original_enabled
        generator._sampler_pools.clear()

    return {
        "mode": "stop_logit_decision_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
        },
        "timers": _summarize_samples(samples),
    }


def run_rope_trig_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._cfm.use_precomputed_temb = True
    generator._llm_decode_graph_enabled = True

    generator._cfm.model.use_precomputed_rope_trig = False
    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=False)

    generator._cfm.model.use_precomputed_rope_trig = True
    generator._sampler_pools.clear()
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    opt_latents, opt_timers = _run(generator, inputs, args, force_original=False)
    diff = (_cat_all_latents(baseline_latents) - _cat_all_latents(opt_latents)).float().abs()

    samples = {
        "freqs_rope": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        "cached_trig_rope": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
    }
    for idx in range(args.repeats):
        for name, enabled in (("freqs_rope", False), ("cached_trig_rope", True)):
            generator._cfm.model.use_precomputed_rope_trig = enabled
            generator._sampler_pools.clear()
            torch.manual_seed(1000 + idx)
            _latents, timers = _run(generator, inputs, args, force_original=False)
            for key in samples[name]:
                samples[name][key].append(float(timers[key]))

    return {
        "mode": "rope_trig_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
        },
        "timers": _summarize_samples(samples),
    }


def run_aggregator_rope_trig_compare(generator, inputs, args, llm_config, talker_cfg) -> dict:
    """Compare Aggregator freqs RoPE with cached trig RoPE."""

    generator._pack_qkv_enabled = True
    generator._qkv_packed = False
    generator._cfm.use_preembedded_cfg = True
    generator._cfm.use_precomputed_temb = True
    generator._cfm.model.use_precomputed_rope_trig = True
    generator._llm_decode_graph_enabled = True

    original_enabled = generator._aggregator.use_precomputed_rope_trig
    try:
        generator._aggregator.use_precomputed_rope_trig = False
        generator._sampler_pools.clear()
        torch.manual_seed(0)
        _run(generator, inputs, args, force_original=False)
        torch.manual_seed(1234)
        baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=False)

        generator._aggregator.use_precomputed_rope_trig = True
        generator._sampler_pools.clear()
        torch.manual_seed(0)
        _run(generator, inputs, args, force_original=False)
        torch.manual_seed(1234)
        opt_latents, opt_timers = _run(generator, inputs, args, force_original=False)
        diff = (_cat_all_latents(baseline_latents) - _cat_all_latents(opt_latents)).float().abs()

        samples = {
            "aggregator_freqs_rope": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
            "aggregator_cached_trig_rope": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        }
        for idx in range(args.repeats):
            for name, enabled in (("aggregator_freqs_rope", False), ("aggregator_cached_trig_rope", True)):
                generator._aggregator.use_precomputed_rope_trig = enabled
                generator._sampler_pools.clear()
                torch.manual_seed(1000 + idx)
                _latents, timers = _run(generator, inputs, args, force_original=False)
                for key in samples[name]:
                    samples[name][key].append(float(timers[key]))
    finally:
        generator._aggregator.use_precomputed_rope_trig = original_enabled
        generator._sampler_pools.clear()

    return {
        "mode": "aggregator_rope_trig_compare",
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
        },
        "timers": _summarize_samples(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/admin/workspace/remote_workspace/models/Ming-flash-omni-2.0")
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--compare-fused-llm-mlp", action="store_true")
    parser.add_argument("--compare-fused-qkv", action="store_true")
    parser.add_argument("--compare-prepared-cfg", action="store_true")
    parser.add_argument("--compare-full-history-recompute", action="store_true")
    parser.add_argument("--probe-llm-recompute-semantics", action="store_true")
    parser.add_argument("--compare-llm-decode-graph", action="store_true")
    parser.add_argument("--compare-llm-decode-graph-ar", action="store_true")
    parser.add_argument("--compare-vae-stream-full", action="store_true")
    parser.add_argument("--compare-vae-qwen-graph", action="store_true")
    parser.add_argument("--probe-vae-qwen-hidden-graph", action="store_true")
    parser.add_argument("--vae-qwen-attn-impl", choices=("eager", "sdpa"), default="")
    parser.add_argument("--compare-stop-logit-decision", action="store_true")
    parser.add_argument("--compare-rope-trig", action="store_true")
    parser.add_argument("--compare-aggregator-rope-trig", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16
    talker_cfg, llm_config, model, _audio_vae, generator = load_talker(args.model_path, args.device, dtype)
    inputs = build_inputs(model, args.model_path, args.batch_size, args.device, dtype)

    if args.compare_fused_llm_mlp:
        summary = run_fused_llm_mlp_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_fused_qkv:
        summary = run_fused_qkv_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_prepared_cfg:
        summary = run_prepared_cfg_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_full_history_recompute:
        summary = run_full_history_recompute_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.probe_llm_recompute_semantics:
        summary = run_llm_recompute_semantics_probe(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_llm_decode_graph:
        summary = run_llm_decode_graph_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_llm_decode_graph_ar:
        summary = run_llm_decode_graph_ar_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_vae_stream_full:
        summary = run_vae_stream_full_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_vae_qwen_graph:
        summary = run_vae_qwen_graph_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.probe_vae_qwen_hidden_graph:
        summary = run_vae_qwen_hidden_graph_probe(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_stop_logit_decision:
        summary = run_stop_logit_decision_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_rope_trig:
        summary = run_rope_trig_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    if args.compare_aggregator_rope_trig:
        summary = run_aggregator_rope_trig_compare(generator, inputs, args, llm_config, talker_cfg)
        text = json.dumps(summary, indent=2, sort_keys=True)
        if args.output:
            Path(args.output).write_text(text)
        print(text)
        return

    # Initialize both graph variants before measuring or comparing.
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=True)
    torch.manual_seed(0)
    _run(generator, inputs, args, force_original=False)

    torch.manual_seed(1234)
    baseline_latents, baseline_timers = _run(generator, inputs, args, force_original=True)
    torch.manual_seed(1234)
    opt_latents, opt_timers = _run(generator, inputs, args, force_original=False)
    diff = (_cat_latents(baseline_latents) - _cat_latents(opt_latents)).float().abs()

    samples = {
        "original_sde_graph": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
        "deterministic_sde_graph": {"llm_decode_ms": [], "cfm_step_ms": [], "collect_ms": []},
    }
    for idx in range(args.repeats):
        for name, force_original in (("original_sde_graph", True), ("deterministic_sde_graph", False)):
            torch.manual_seed(1000 + idx)
            _latents, timers = _run(generator, inputs, args, force_original=force_original)
            for key in samples[name]:
                samples[name][key].append(float(timers[key]))

    summary = {
        "model": {
            "llm_layers": llm_config.num_hidden_layers,
            "llm_hidden": llm_config.hidden_size,
            "cfm_steps": talker_cfg["steps"],
            "dit_depth": talker_cfg["flowmodel"]["depth"],
            "aggregator_depth": talker_cfg["aggregator"]["depth"],
        },
        "workload": {
            "batch_size": args.batch_size,
            "input_tokens": inputs.shape[1],
            "profile_steps": args.steps,
            "repeats": args.repeats,
        },
        "equivalence": {
            "max_abs_latent_diff": float(diff.max().item()),
            "mean_abs_latent_diff": float(diff.mean().item()),
            "baseline_timers": baseline_timers,
            "optimized_timers": opt_timers,
        },
        "timers": _summarize_samples(samples),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
