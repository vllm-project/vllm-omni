# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matched single-request Chatterbox qualification through both public Python APIs.

Run each backend in a fresh process on an exclusive GPU. Model initialization is
excluded; cloned-voice conditioning, tokenization, synthesis and CPU waveform
assembly are included. Upstream's canonical API also includes Perth watermarking.
"""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

TEXTS = (
    "The weather is nice today, perfect for a walk in the park.",
    "Please leave your name and number after the tone. We will return your call tomorrow morning.",
    "The morning train arrived beside the river just as the town began to wake. "
    "People carried baskets to the market, and a baker opened the windows of a small shop near the station. "
    "Across the square, children gathered around an old map and tried to trace the road into the hills.",
)
SEEDS = (0, 42, 123)


def main():
    """Execute one backend/variant and retain every timed waveform and result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("upstream", "omni", "quality"))
    parser.add_argument("variant", choices=("turbo", "original"))
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", default="0", help="Physical nvidia-smi GPU index or UUID")
    parser.add_argument("--runner", choices=("v1", "v2"), default="v2")
    parser.add_argument("--cuda-graphs", action="store_true", help="Use the existing AR CUDA-graph path at FP32")
    parser.add_argument(
        "--profile", action="store_true", help="Trace an additional warmed request after each voice mode"
    )
    args = parser.parse_args()
    identity = subprocess.check_output(
        ["nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True
    ).strip()
    args.gpu_uuid, args.gpu_name = identity.split(", ", 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.backend == "quality":
        evaluate_quality(args)
        return
    synthesize, close, engine, sample_rate, flow_dtype = make_backend(args)
    rows = []
    try:
        for voice in ("builtin", "clone"):
            reference = args.reference if voice == "clone" else None
            # Warm every text length in each voice mode before timing.
            for text in TEXTS:
                for seed in (997, 998):
                    seed_everything(seed)
                    synthesize(text, reference, seed)
            for text_index, text in enumerate(TEXTS):
                for seed in SEEDS:
                    seed_everything(seed)
                    samples = []
                    finished = threading.Event()
                    monitor = threading.Thread(
                        target=sample_gpu_memory, args=(finished, samples, args.gpu_uuid), daemon=True
                    )
                    monitor.start()
                    torch.accelerator.synchronize()
                    started = time.perf_counter()
                    try:
                        audio = synthesize(text, reference, seed).flatten().float().cpu()
                        torch.accelerator.synchronize()
                        elapsed = time.perf_counter() - started
                    finally:
                        finished.set()
                        monitor.join()
                    if audio.numel() == 0 or not torch.isfinite(audio).all():
                        raise ValueError("Generated empty or nonfinite audio")
                    filename = f"{args.backend}-{args.variant}-{voice}-{text_index}-{seed}.wav"
                    sf.write(args.output / filename, audio.numpy(force=True), sample_rate, subtype="FLOAT")
                    duration = audio.numel() / sample_rate
                    rows.append(
                        dict(
                            voice=voice,
                            text_index=text_index,
                            text=text,
                            seed=seed,
                            audio=filename,
                            audio_samples=audio.numel(),
                            audio_seconds=duration,
                            e2e_seconds=elapsed,
                            rtf=elapsed / duration,
                            peak_gpu_used_mib=max(samples) if samples else None,
                        )
                    )
                    logging.info("%s %s %s: %.3fs, RTF %.3f", args.backend, voice, seed, elapsed, elapsed / duration)
            write_report(args, rows, flow_dtype)
            if args.profile:
                profile_generation(args, synthesize, reference, voice, engine)
    finally:
        close()


def make_backend(args):
    """Use canonical upstream generate or Omni.generate without patched model methods.

    Upstream runs with its own ``generate`` defaults; Omni with the deploy
    file's ``default_sampling_params``, which record them, and each request's seed.

    Returns:
        The synthesis function, its cleanup, the Omni engine (None upstream),
        the output sample rate and the precision the flow decoder runs in.
    """
    if args.backend == "upstream":
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        cls = ChatterboxTurboTTS if args.variant == "turbo" else ChatterboxTTS
        model = cls.from_local(str(args.model), "cuda")
        builtin = model.conds
        for component in (model.t3, model.s3gen, model.ve):
            if {parameter.dtype for parameter in component.parameters()} != {torch.float32}:
                raise ValueError("The canonical reference must run in FP32")

        def synthesize(text, reference, seed):
            model.conds = builtin
            return model.generate(text, audio_prompt_path=str(reference) if reference else None)

        return synthesize, lambda: None, None, model.sr, "float32"

    import yaml

    from vllm_omni import Omni
    from vllm_omni.entrypoints.openai.tts_adapters.chatterbox import CONTROLS
    from vllm_omni.model_executor.models.chatterbox.conditioning import (
        VoiceConditioner,
        VoiceConditioning,
        build_prompt,
        load_text_tokenizer,
        tokenize_text,
    )
    from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

    config = ChatterboxConfig(args.variant)
    tokenizer = load_text_tokenizer(str(args.model), args.variant)
    builtin = VoiceConditioning.from_builtin(str(args.model))
    conditioner = VoiceConditioner(str(args.model), config, torch.device("cuda"))
    deploy = (
        Path(__file__).resolve().parents[2]
        / "vllm_omni"
        / "deploy"
        / ("chatterbox_turbo.yaml" if args.variant == "turbo" else "chatterbox.yaml")
    )
    settings = yaml.safe_load(deploy.read_text())
    # Original's controls come from stage 0's additional_config, as the serving adapter reads them.
    controls = (
        {} if args.variant == "turbo" else {name: settings["stages"][0]["additional_config"][name] for name in CONTROLS}
    )
    settings["async_chunk"] = False
    settings["model_runner"] = args.runner
    for stage in settings["stages"]:
        stage["dtype"] = "float32"
        stage["enforce_eager"] = not (args.cuda_graphs and stage["stage_id"] == 0)
        if args.profile:
            stage["profiler_config"] = {
                "profiler": "torch",
                "torch_profiler_dir": str(args.output / "traces" / f"stage-{stage['stage_id']}"),
                "torch_profiler_with_stack": False,
                "torch_profiler_record_shapes": True,
            }
    profile = args.output / "matched-deploy.yaml"
    profile.write_text(yaml.safe_dump(settings))
    engine = Omni(model=str(args.model), deploy_config=str(profile))

    def synthesize(text, reference, seed):
        voice = builtin
        if reference is not None:
            wav, rate = sf.read(reference, dtype="float32", always_2d=True)
            voice = conditioner.prepare(wav.mean(axis=1), rate)
        ids = tokenize_text(text, tokenizer, args.variant)
        prompt = build_prompt(ids, voice, config, **controls)
        sampling = engine.default_sampling_params_list[0].clone()
        sampling.seed = seed
        params = [sampling, *engine.default_sampling_params_list[1:]]
        output = list(engine.generate([prompt], sampling_params_list=params))[0].outputs[0].multimodal_output
        audio = output["audio"]
        return torch.cat([chunk.flatten() for chunk in audio]) if isinstance(audio, list) else audio

    # Every weight is float32 below; the flow decoder runs in the deploy file's flow_dtype.
    flow_dtype = ChatterboxConfig(**settings["stages"][1].get("hf_overrides", {})).flow_dtype
    return synthesize, engine.close, engine, config.sample_rate, flow_dtype


def profile_generation(args, synthesize, reference, voice, engine):
    """Trace one extra request; profiled latency never enters benchmark rows."""
    seed_everything(42)
    if engine is not None:
        engine.start_profile(profile_prefix=voice)
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profile:
            with torch.profiler.record_function(f"{args.backend}_{voice}_generate"):
                synthesize(TEXTS[-1], reference, 42)
                torch.accelerator.synchronize()
    finally:
        if engine is not None:
            engine.stop_profile()
    profile.export_chrome_trace(str(args.output / f"{args.backend}-{voice}.trace.json"))
    (args.output / f"{args.backend}-{voice}.profile.txt").write_text(
        profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=60)
    )


def seed_everything(seed):
    """Reset host and device generators before each paired stochastic trial."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_gpu_memory(finished, samples, gpu_uuid):
    """Sample whole-device used memory, including Omni's separate workers."""
    while not finished.is_set():
        value = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_uuid}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        samples.append(int(value.strip()))
        finished.wait(0.1)


def write_report(args, rows, flow_dtype):
    """Persist provenance and per-mode variability after each completed mode."""
    summaries = {}
    for voice in ("builtin", "clone"):
        selected = [row for row in rows if row["voice"] == voice]
        if selected:
            summaries[voice] = {
                metric: dict(
                    mean=statistics.mean(values),
                    stdev=statistics.stdev(values),
                    p50=float(np.percentile(values, 50)),
                    p95=float(np.percentile(values, 95)),
                )
                for metric in ("e2e_seconds", "rtf", "audio_seconds", "peak_gpu_used_mib")
                if (values := [row[metric] for row in selected if row[metric] is not None])
            }
    report = dict(
        backend=args.backend,
        gpu=args.gpu_name,
        gpu_uuid=args.gpu_uuid,
        variant=args.variant,
        model=str(args.model),
        checkpoint=args.model.name,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in ("torch", "torchaudio", "transformers", "numpy")},
        precision=f"float32 weights, {flow_dtype} flow decoder; TF32 disabled",
        concurrency=1,
        model_runner=args.runner if args.backend == "omni" else None,
        cuda_graphs=args.backend == "omni" and args.cuda_graphs,
        warmup_per_mode=2 * len(TEXTS),
        rows=rows,
        summaries=summaries,
        scope="warm whole utterance including clone conditioning and CPU waveform assembly; excludes WAV writing",
        differences=(
            "upstream includes Perth watermarking; Omni uses fixed flow noise; equal seeds are not equal token draws"
        ),
    )
    (args.output / f"{args.backend}-{args.variant}.json").write_text(json.dumps(report, indent=2) + "\n")


def evaluate_quality(args):
    """Score every retained output with the shared Seed-TTS evaluator, one group per backend and voice."""
    from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSSampleRequest
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import compute_seed_tts_wer_metrics
    from vllm_omni.benchmarks.patch.patch import MixRequestFuncOutput
    from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

    # The built-in voice ships as embeddings, not a recording, so SIM covers cloned rows only.
    os.environ["SEED_TTS_SIM_EVAL"] = "1"
    sample_rate = ChatterboxConfig(args.variant).sample_rate
    rows, summaries = [], {}
    for backend in ("upstream", "omni"):
        report = json.loads((args.output / f"{backend}-{args.variant}.json").read_text())
        for voice in ("builtin", "clone"):
            selected = [row for row in report["rows"] if row["voice"] == voice]
            requests, outputs = [], []
            for row in selected:
                wav, rate = sf.read(args.output / row["audio"], dtype="float32")
                if rate != sample_rate or wav.ndim != 1 or len(wav) != row["audio_samples"]:
                    raise ValueError("Output waveform does not match its timed report")
                requests.append(
                    SeedTTSSampleRequest(
                        prompt=row["text"],
                        prompt_len=0,
                        seed_tts_utterance_id=Path(row["audio"]).stem,
                        seed_tts_locale="en",
                        seed_tts_ref_wav_path=str(args.reference) if voice == "clone" else "",
                    )
                )
                pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                outputs.append(MixRequestFuncOutput(success=True, tts_output_pcm_bytes=pcm))
            metrics = compute_seed_tts_wer_metrics(requests, outputs, include_per_item=True)
            if "seed_tts_eval_setup_error" in metrics:
                raise RuntimeError(metrics["seed_tts_eval_setup_error"])
            items = metrics.pop("seed_tts_wer_eval_items")
            summaries[f"{backend}-{voice}"] = metrics
            for row, item in zip(selected, items, strict=True):
                if "error" in item:
                    raise ValueError(f"{item['utterance_id']}: {item['error']} {item.get('detail', '')}")
                expected, actual = (
                    "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()
                    for text in (item["reference_raw"], item["asr_raw"])
                )
                sim = item.get("sim")
                rows.append(
                    {
                        **row,
                        "backend": backend,
                        "transcript": item["asr_raw"],
                        "wer": item["wer"],
                        "tail_complete": actual[-3:] == expected[-3:],
                        "speaker_sim": sim,
                    }
                )
                logging.info("%s %s seed %s: WER %.3f, SIM %s", backend, voice, row["seed"], item["wer"], sim)
    (args.output / f"quality-{args.variant}.json").write_text(
        json.dumps(
            {
                "evaluator": "vllm_omni.benchmarks.data_modules.seed_tts_eval: Whisper large-v3 WER; "
                "WavLM cosine SIM to the cloned reference; not human listening scores",
                "summaries": summaries,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
