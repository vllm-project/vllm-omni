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
    parser.add_argument("--runner", choices=("v1", "v2"), default="v2")
    parser.add_argument("--cuda-graphs", action="store_true", help="Use the existing AR CUDA-graph path at FP32")
    parser.add_argument(
        "--profile", action="store_true", help="Trace an additional warmed request after each voice mode"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.backend == "quality":
        evaluate_quality(args)
        return
    synthesize, close, engine = make_backend(args)
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
                    monitor = threading.Thread(target=sample_gpu_memory, args=(finished, samples), daemon=True)
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
                    sf.write(args.output / filename, audio.numpy(force=True), 24000, subtype="FLOAT")
                    duration = audio.numel() / 24000
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
            write_report(args, rows)
            if args.profile:
                profile_generation(args, synthesize, reference, voice, engine)
    finally:
        close()


def make_backend(args):
    """Use canonical upstream generate or Omni.generate without patched model methods."""
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
            controls = dict(temperature=0.8, repetition_penalty=1.2, top_p=0.95 if args.variant == "turbo" else 1.0)
            if args.variant == "turbo":
                controls.update(top_k=1000, min_p=0.0, cfg_weight=0.0, exaggeration=0.0)
            else:
                controls.update(min_p=0.05, cfg_weight=0.5, exaggeration=0.5)
            return model.generate(text, audio_prompt_path=str(reference) if reference else None, **controls)

        return synthesize, lambda: None, None

    import yaml
    from vllm.sampling_params import SamplingParams

    from vllm_omni import Omni
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
        prompt = build_prompt(ids, voice, config)
        sampling = SamplingParams(
            seed=seed,
            max_tokens=1000,
            temperature=0.8,
            repetition_penalty=1.2,
            top_p=0.95 if args.variant == "turbo" else 1.0,
            top_k=1000 if args.variant == "turbo" else -1,
            min_p=0.0 if args.variant == "turbo" else 0.05,
        )
        params = [sampling, *engine.default_sampling_params_list[1:]]
        output = list(engine.generate([prompt], sampling_params_list=params))[0].outputs[0].multimodal_output
        audio = output["audio"]
        return torch.cat([chunk.flatten() for chunk in audio]) if isinstance(audio, list) else audio

    return synthesize, engine.close, engine


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


def sample_gpu_memory(finished, samples):
    """Sample whole-device used memory, including Omni's separate workers."""
    while not finished.is_set():
        value = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        samples.append(int(value.strip()))
        finished.wait(0.1)


def write_report(args, rows):
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
        variant=args.variant,
        model=str(args.model),
        checkpoint=args.model.name,
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        packages={name: importlib.metadata.version(name) for name in ("torch", "torchaudio", "transformers", "numpy")},
        precision="float32; TF32 disabled",
        concurrency=1,
        model_runner=args.runner if args.backend == "omni" else None,
        cuda_graphs=args.backend == "omni" and args.cuda_graphs,
        max_new_tokens=1000,
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
    """Apply one frozen ASR and speaker encoder to every retained output."""
    import jiwer
    import torchaudio
    import whisper
    from safetensors.torch import load_file
    from whisper.normalizers import EnglishTextNormalizer

    from vllm_omni.model_executor.models.chatterbox.conditioning import trim_silence, voice_encoder_mel
    from vllm_omni.model_executor.models.chatterbox.voice_encoder import VoiceEncConfig, VoiceEncoder

    encoder = VoiceEncoder(VoiceEncConfig()).to("cuda").eval()
    weights = load_file(str(args.model / "ve.safetensors"))
    encoder.load_state_dict(weights)
    asr = whisper.load_model("large-v3", device="cuda")
    normalizer = EnglishTextNormalizer()

    @torch.inference_mode()
    def speaker(wav16):
        mel = voice_encoder_mel(trim_silence(wav16.numpy(force=True)), encoder.hp)
        return encoder.inference(mel[None].to("cuda"), mel_lens=[mel.shape[0]], rate=1.3)[0].float().cpu()

    ref, ref_rate = sf.read(args.reference, dtype="float32", always_2d=True)
    reference_embedding = speaker(torchaudio.functional.resample(torch.from_numpy(ref.mean(axis=1)), ref_rate, 16000))
    builtin = torch.load(args.model / "conds.pt", map_location="cpu", weights_only=True)["t3"]["speaker_emb"].flatten()
    rows = []
    reports = {
        backend: json.loads((args.output / f"{backend}-{args.variant}.json").read_text())
        for backend in ("upstream", "omni")
    }
    for backend, report in reports.items():
        for row in report["rows"]:
            wav, rate = sf.read(args.output / row["audio"], dtype="float32")
            if rate != 24000 or wav.ndim != 1 or len(wav) != row["audio_samples"]:
                raise ValueError("Output waveform does not match its timed report")
            wav16 = torchaudio.functional.resample(torch.from_numpy(wav), rate, 16000)
            transcript = asr.transcribe(wav16.numpy(force=True), language="en", temperature=0, fp16=False)["text"]
            expected, actual = normalizer(row["text"]), normalizer(transcript)
            errors = jiwer.process_words(expected, actual)
            target = reference_embedding if row["voice"] == "clone" else builtin
            embedding = speaker(wav16)
            cosine = torch.nn.functional.cosine_similarity(embedding[None], target[None]).item()
            rows.append(
                {
                    **row,
                    "backend": backend,
                    "transcript": transcript,
                    "wer": errors.wer,
                    "substitutions": errors.substitutions,
                    "deletions": errors.deletions,
                    "insertions": errors.insertions,
                    "reference_words": len(expected.split()),
                    "tail_complete": actual.endswith(" ".join(expected.split()[-3:])),
                    "speaker_cosine": cosine,
                }
            )
            logging.info(
                "%s %s seed %s: WER %.3f, speaker %.3f", backend, row["voice"], row["seed"], errors.wer, cosine
            )
            (args.output / f"quality-{args.variant}.json").write_text(
                json.dumps(
                    {
                        "asr": "openai-whisper large-v3; temperature=0; FP32",
                        "normalization": "Whisper EnglishTextNormalizer",
                        "speaker_metric": "checkpoint VoiceEncoder cosine to reference/builtin target; not human MOS",
                        "rows": rows,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
