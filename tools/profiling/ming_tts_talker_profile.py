#!/usr/bin/env python3
"""Focused Ming-flash-omni-2.0 Talker profiler for H20 experiments.

This script profiles the Talker path outside the HTTP server so the core
model stages can be measured without scheduler noise:

    text embeddings -> Talker LLM -> CFM/DiT + Aggregator + StopHead -> VAE

It intentionally keeps CFM steps unchanged.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model

from vllm_omni.model_executor.models.ming_flash_omni.audio_vae import AudioVAE, AudioVAEConfig
from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
    CFM,
    Aggregator,
    DiT,
    MingAudioGenerator,
)


def _sync(device: str) -> None:
    if torch.cuda.is_available():
        torch.accelerator.synchronize(device)


def _mean_ms(values: list[float]) -> float:
    return stats.mean(values) * 1000.0 if values else 0.0


def load_talker(model_path: str, device: str, dtype: torch.dtype):
    talker_dir = Path(model_path) / "talker"
    with open(talker_dir / "config.json") as f:
        talker_cfg = json.load(f)

    llm_config = Qwen2Config.from_pretrained(talker_dir / "llm")
    llm_config._attn_implementation = "sdpa"
    flow_cfg = talker_cfg["flowmodel"]
    agg_cfg = talker_cfg["aggregator"]

    model = Qwen2Model(llm_config).to(device=device, dtype=dtype).eval()
    dit = DiT(llm_input_dim=llm_config.hidden_size, **flow_cfg).to(device=device, dtype=dtype).eval()
    cfm = CFM(dit, steps=talker_cfg["steps"])
    aggregator = Aggregator(llm_input_dim=llm_config.hidden_size, **agg_cfg).to(device=device, dtype=dtype).eval()
    stop_head = torch.nn.Linear(llm_config.hidden_size, 2, bias=True).to(device=device, dtype=dtype).eval()

    vae_config = AudioVAEConfig.from_pretrained(talker_dir / "vae")
    audio_vae = AudioVAE(vae_config).to(device=device, dtype=dtype).eval()

    sd = load_file(str(talker_dir / "model.safetensors"), device=device)
    model.load_state_dict({k.removeprefix("model."): sd[k] for k in sd if k.startswith("model.")}, strict=False)
    dit.load_state_dict({k.removeprefix("cfm.model."): sd[k] for k in sd if k.startswith("cfm.model.")}, strict=False)
    aggregator.load_state_dict(
        {k.removeprefix("aggregator."): sd[k] for k in sd if k.startswith("aggregator.")},
        strict=False,
    )
    stop_head.load_state_dict(
        {k.removeprefix("stop_head."): sd[k] for k in sd if k.startswith("stop_head.")},
        strict=False,
    )

    vae_sd = load_file(str(talker_dir / "vae" / "model.safetensors"), device=device)
    audio_vae.load_state_dict(vae_sd, strict=False)
    del sd, vae_sd
    torch.accelerator.empty_cache()

    generator = MingAudioGenerator(
        config=SimpleNamespace(**talker_cfg),
        llm_config=llm_config,
        model=model,
        cfm=cfm,
        aggregator=aggregator,
        stop_head=stop_head,
        audio_vae=audio_vae,
        patch_size=talker_cfg["patch_size"],
        his_patch_size=talker_cfg["history_patch_size"],
        latent_dim=flow_cfg["in_channels"],
        cfg_strength=talker_cfg["cfg_strength"],
        use_cuda_graphs=True,
    )
    return talker_cfg, llm_config, model, audio_vae, generator


def build_inputs(model, model_path: str, batch_size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    talker_dir = Path(model_path) / "talker"
    tokenizer = AutoTokenizer.from_pretrained(talker_dir / "llm", trust_remote_code=True)
    text = "Hello, this is a Ming flash omni text to speech profiling request."
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nPlease synthesize the following text into speech.\n Text input:\n"
        + text
        + "<|im_end|>\n<|im_start|>assistant\n<audio>"
    )
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    embeds = model.get_input_embeddings()(input_ids).to(dtype=dtype)
    if batch_size == 1:
        return embeds
    return embeds.expand(batch_size, -1, -1).contiguous()


def run_ar_profile(
    generator: MingAudioGenerator,
    inputs_embeds: torch.Tensor,
    *,
    max_steps: int,
    use_static_cache: bool,
    device: str,
) -> tuple[list[list[torch.Tensor]], dict[str, float]]:
    cfg = generator.cfg_strength
    dtype = next(generator._model.parameters()).dtype
    batch_size = inputs_embeds.shape[0]
    his_lat = torch.zeros(batch_size, generator.his_patch_size, generator.latent_dim, device=device, dtype=dtype)
    past_key_values, max_cache_len = generator._init_batched_kv_cache(
        batch_size, use_static_cache, torch.device(device), dtype
    )
    current_inputs = inputs_embeds
    latents_by_request: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
    llm_s: list[float] = []
    cfm_s: list[float] = []
    collect_s: list[float] = []

    with torch.no_grad():
        for step in range(min(max_steps, max_cache_len - inputs_embeds.shape[1])):
            _sync(device)
            t0 = time.perf_counter()
            with record_function("ming.llm_step"):
                last_hs = generator.llm_step(
                    current_inputs,
                    step=step,
                    past_key_values=past_key_values,
                    use_static_cache=use_static_cache,
                )
            _sync(device)
            t1 = time.perf_counter()

            with record_function("ming.cfm_graph_or_eager_step"):
                gen_lat, next_inputs, _stop_out = generator.cfm_sample_step(last_hs, his_lat, cfg=cfg)
            _sync(device)
            t2 = time.perf_counter()

            with record_function("ming.collect_latents"):
                for row in range(batch_size):
                    latents_by_request[row].append(gen_lat[row : row + 1])
                his_lat = generator._update_his_lat(his_lat, gen_lat)
                current_inputs = next_inputs
            _sync(device)
            t3 = time.perf_counter()

            if step > 0:
                llm_s.append(t1 - t0)
                cfm_s.append(t2 - t1)
                collect_s.append(t3 - t2)

    return latents_by_request, {
        "llm_decode_ms": _mean_ms(llm_s),
        "cfm_step_ms": _mean_ms(cfm_s),
        "collect_ms": _mean_ms(collect_s),
    }


def decode_first(
    generator: MingAudioGenerator,
    latents: list[list[torch.Tensor]],
    device: str,
    *,
    stream_decode: bool,
) -> tuple[float, float]:
    _sync(device)
    t0 = time.perf_counter()
    with torch.no_grad(), record_function("ming.vae_decode_one_request"):
        wav = generator.decode_to_waveform(latents[0], stream_decode=stream_decode)
        if not stream_decode:
            wav = generator.trim_trailing_silence(wav)
    _sync(device)
    elapsed = time.perf_counter() - t0
    sr = int(generator._audio_vae.config.sample_rate)
    audio_s = float(wav.shape[-1]) / sr if wav is not None else 0.0
    return elapsed * 1000.0, audio_s


def warm_vae_qwen_decode_graph(generator: MingAudioGenerator, steps: int, device: str) -> bool:
    decoder = generator._audio_vae.decoder if generator._audio_vae is not None else None
    if decoder is None or not getattr(decoder, "_qwen_decode_graph_enabled", False):
        return False

    dtype = next(generator._model.parameters()).dtype
    latents = [
        torch.zeros(1, generator.patch_size, generator.latent_dim, device=device, dtype=dtype) for _ in range(steps)
    ]
    with torch.no_grad():
        generator.decode_to_waveform(latents, stream_decode=True)
    _sync(device)
    return True


def static_cache_semantics_probe(
    generator: MingAudioGenerator, inputs_embeds: torch.Tensor, device: str
) -> dict[str, float]:
    """Compare static-cache decode to the current no-cache mode.

    The current no-cache mode feeds only the latest generated embedding after
    prefill, so it is a performance number for a different computation, not a
    quality-preserving replacement.
    """
    single = inputs_embeds[:1]
    _sync(device)
    t0 = time.perf_counter()
    lats_cache, _ = run_ar_profile(generator, single, max_steps=6, use_static_cache=True, device=device)
    t1 = time.perf_counter()
    lats_no_cache, _ = run_ar_profile(generator, single, max_steps=6, use_static_cache=False, device=device)
    t2 = time.perf_counter()
    last_cache = torch.cat(lats_cache[0], dim=1)
    last_no_cache = torch.cat(lats_no_cache[0], dim=1)
    diff = (last_cache - last_no_cache).float().abs()
    return {
        "static_cache_6step_ms": (t1 - t0) * 1000.0,
        "no_cache_6step_ms": (t2 - t1) * 1000.0,
        "max_abs_latent_diff": float(diff.max().item()),
        "mean_abs_latent_diff": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/admin/workspace/remote_workspace/models/Ming-flash-omni-2.0")
    parser.add_argument("--output-dir", default="/home/admin/workspace/remote_workspace/ming_profiles")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vae-decode-mode", choices=("stream", "full"), default="stream")
    parser.add_argument("--disable-cfm-graph", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trace_path = Path(args.output_dir) / f"ming_talker_b{args.batch_size}_s{args.steps}.trace.json"

    print("loading model...")
    talker_cfg, llm_config, model, _audio_vae, generator = load_talker(args.model_path, args.device, dtype)
    if args.disable_cfm_graph:
        generator._use_cuda_graphs = False
    inputs = build_inputs(model, args.model_path, args.batch_size, args.device, dtype)
    print(
        json.dumps(
            {
                "llm_layers": llm_config.num_hidden_layers,
                "llm_hidden": llm_config.hidden_size,
                "cfm_steps": talker_cfg["steps"],
                "dit_depth": talker_cfg["flowmodel"]["depth"],
                "aggregator_depth": talker_cfg["aggregator"]["depth"],
                "input_tokens": inputs.shape[1],
                "batch_size": args.batch_size,
                "profile_steps": args.steps,
            },
            sort_keys=True,
        )
    )

    # Warm graph captures for B=1 and the requested batch, outside profiler.
    run_ar_profile(generator, inputs[:1], max_steps=2, use_static_cache=True, device=args.device)
    run_ar_profile(generator, inputs, max_steps=2, use_static_cache=True, device=args.device)
    vae_qwen_graph_warmed = warm_vae_qwen_decode_graph(generator, args.steps, args.device)
    _sync(args.device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("ming.total_ar_profile"):
            latents, timers = run_ar_profile(
                generator,
                inputs,
                max_steps=args.steps,
                use_static_cache=True,
                device=args.device,
            )
        vae_ms, audio_s = decode_first(
            generator,
            latents,
            args.device,
            stream_decode=args.vae_decode_mode == "stream",
        )

    prof.export_chrome_trace(str(trace_path))
    print("PROFILE_TABLE_START")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print("PROFILE_TABLE_END")

    kv_probe = static_cache_semantics_probe(generator, inputs, args.device)
    summary = {
        "trace_path": str(trace_path),
        "batch_size": args.batch_size,
        "steps": args.steps,
        "vae_decode_mode": args.vae_decode_mode,
        "input_tokens": inputs.shape[1],
        "timers": timers,
        "vae_qwen_graph_warmed": vae_qwen_graph_warmed,
        "vae_decode_one_request_ms": vae_ms,
        "audio_s_first_request": audio_s,
        "kv_probe": kv_probe,
        "gpu_alloc_gb": torch.cuda.memory_allocated(args.device) / 1e9,
        "gpu_reserved_gb": torch.cuda.memory_reserved(args.device) / 1e9,
    }
    print("SUMMARY_JSON_START")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("SUMMARY_JSON_END")


if __name__ == "__main__":
    main()
