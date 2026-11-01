# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import time
import uuid
import wave
from pathlib import Path

import torch
from vllm import SamplingParams

from vllm_omni import AsyncOmni, Omni
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_SPEAKER_EMBEDDING,
    SAMPLE_RATE,
    TEXT_EOS_TOKEN_ID,
)


def coerce_audio_tensor(audio, *, async_chunk: bool) -> torch.Tensor:
    if isinstance(audio, list):
        parts = []
        for item in audio:
            tensor = torch.as_tensor(item, dtype=torch.float32).reshape(-1)
            if tensor.numel() > 0:
                parts.append(tensor)
        if not parts:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.cat(parts, dim=0)

    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1)


def resolve_sr(sr) -> int:
    if isinstance(sr, list):
        sr = sr[-1]
    if hasattr(sr, "item"):
        return int(sr.item())
    return int(sr)


def extract_sample_rate(multimodal_output: dict) -> int:
    sr = multimodal_output.get("sr")
    if sr is None:
        raise RuntimeError("Expected multimodal_output['sr']")
    return resolve_sr(sr)


def write_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.clamp(-1.0, 1.0)
    pcm16 = (audio * 32767.0).round().to(torch.int16).cpu().numpy()
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def request_index(request_id: str | None, fallback: int) -> int:
    try:
        return int(request_id)
    except (TypeError, ValueError):
        if isinstance(request_id, str):
            head = request_id.split("_", 1)[0]
            if head.isdigit():
                return int(head)
    return fallback


def audio_summary(audio: torch.Tensor, sample_rate: int) -> dict:
    waveform = audio.detach().cpu().reshape(-1).to(torch.float32)
    return {
        "sample_rate": int(sample_rate),
        "num_samples": int(waveform.numel()),
        "duration_seconds": float(waveform.numel()) / float(sample_rate),
        "max_abs_amplitude": float(waveform.abs().max().item()) if waveform.numel() > 0 else 0.0,
    }


def resolve_output_name(output_name: str | None, case: str, index: int, total: int) -> str:
    if total == 1:
        return output_name or f"ming_{case}.wav"
    base = Path(output_name or f"ming_{case}.wav")
    return f"{base.stem}_{index:05d}{base.suffix or '.wav'}"


def resolve_stats_log_file(args) -> str | None:
    if not args.log_stats:
        return None
    if args.stats_log_file:
        return args.stats_log_file
    base = Path(args.output_name or f"ming_{args.case}.wav").stem
    return str(Path(args.output_dir) / f"{base}_pipeline.log")


def resolve_metadata_json(args) -> str | None:
    if args.metadata_json:
        return args.metadata_json
    if args.log_stats:
        base = Path(args.output_name or f"ming_{args.case}.wav").stem
        return str(Path(args.output_dir) / f"{base}_manifest.json")
    return None


def build_manifest(args, prompt_payload, stats_log_file: str | None, outputs: list[dict]) -> dict:
    additional_information = {}
    if isinstance(prompt_payload, dict):
        additional_information = dict(prompt_payload.get("additional_information", {}))
    return {
        "model": args.model,
        "case": args.case,
        "streaming": bool(args.streaming),
        "deploy_config": args.deploy_config,
        "enforce_eager": bool(args.enforce_eager),
        "num_prompts": int(args.num_prompts),
        "log_stats": bool(args.log_stats),
        "stats_log_file": stats_log_file,
        "prompt_text": additional_information.get("prompt_text"),
        "instruction": additional_information.get("instruction"),
        "speaker_embedding_shape": (
            list(additional_information[KEY_SPEAKER_EMBEDDING].shape)
            if KEY_SPEAKER_EMBEDDING in additional_information
            and hasattr(additional_information[KEY_SPEAKER_EMBEDDING], "shape")
            else None
        ),
        "outputs": outputs,
        "generated_at_unix": time.time(),
    }


def build_engine_kwargs(args, stats_log_file: str | None) -> dict:
    kwargs = {
        "model": args.model,
        "deploy_config": args.deploy_config,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "log_stats": args.log_stats,
        "stage_init_timeout": args.stage_init_timeout,
        "init_timeout": args.init_timeout,
        "batch_timeout": args.batch_timeout,
        "worker_backend": args.worker_backend,
    }
    if stats_log_file is not None:
        kwargs["log_file"] = stats_log_file
    if args.ray_address is not None:
        kwargs["ray_address"] = args.ray_address
    return kwargs


def build_sampling_params(max_decode_steps: int) -> list[SamplingParams]:
    return [
        SamplingParams(
            temperature=0.0,
            max_tokens=max_decode_steps + 1,
            stop_token_ids=[int(TEXT_EOS_TOKEN_ID)],
        ),
        SamplingParams(temperature=0.0, max_tokens=1),
    ]


async def run_streaming(args, prompt_payload, sampling_params_list, output_dir: Path, stats_log_file: str | None):
    engine = AsyncOmni(**build_engine_kwargs(args, stats_log_file))
    try:
        all_audio_chunks = []
        accumulated_samples = 0
        chunk_idx = 0
        start_time = time.time()
        chunk_times = []
        ttfp_seconds = None
        final_stage_output = None
        async for stage_output in engine.generate(
            prompt=prompt_payload,
            request_id=str(uuid.uuid4()),
            sampling_params_list=sampling_params_list,
        ):
            final_stage_output = stage_output
            multimodal_output = stage_output.multimodal_output or {}
            audio = multimodal_output.get("audio")
            if audio is None:
                continue

            finished = stage_output.finished
            if isinstance(audio, torch.Tensor):
                if finished:
                    audio_chunk = audio[accumulated_samples:].float().detach().cpu()
                else:
                    audio_chunk = audio.float().detach().cpu()
            elif isinstance(audio, list):
                audio_chunk = torch.as_tensor(audio[chunk_idx], dtype=torch.float32).reshape(-1).cpu()
            else:
                audio_chunk = torch.as_tensor(audio, dtype=torch.float32).reshape(-1).cpu()

            accumulated_samples += int(audio_chunk.numel())
            chunk_idx += 1
            if audio_chunk.numel() > 0:
                now = time.time()
                if ttfp_seconds is None:
                    ttfp_seconds = now - start_time
                chunk_times.append(now)
                all_audio_chunks.append(audio_chunk)

        if not all_audio_chunks:
            raise RuntimeError("Streaming Ming example produced no audio chunks")

        waveform = torch.cat(all_audio_chunks, dim=0)
        output_name = resolve_output_name(args.output_name, args.case, 0, 1)
        output_path = str(output_dir / output_name)
        write_wav(output_path, waveform, SAMPLE_RATE)
        summary = {
            "request_id": getattr(final_stage_output, "request_id", None),
            "stage_id": getattr(final_stage_output, "stage_id", None),
            "output_path": output_path,
            "stage_durations": getattr(final_stage_output, "stage_durations", {}),
            "peak_memory_mb": getattr(final_stage_output, "peak_memory_mb", 0.0),
            "ttfp_seconds": ttfp_seconds,
            "mean_inter_chunk_seconds": (
                sum(t1 - t0 for t0, t1 in zip(chunk_times, chunk_times[1:])) / (len(chunk_times) - 1)
                if len(chunk_times) > 1
                else None
            ),
        }
        summary.update(audio_summary(waveform, SAMPLE_RATE))
        print(f"Saved streaming output to {output_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return [summary]
    finally:
        engine.shutdown()


def run_non_streaming(args, prompt_payload, sampling_params_list, output_dir: Path, stats_log_file: str | None):
    engine = Omni(**build_engine_kwargs(args, stats_log_file))
    try:
        outputs = engine.generate(
            prompts=[prompt_payload for _ in range(args.num_prompts)],
            sampling_params_list=sampling_params_list,
            py_generator=False,
        )
        summaries = []
        for fallback_index, output in enumerate(outputs):
            if output.final_output_type != "audio":
                continue
            multimodal_output = output.multimodal_output or {}
            waveform = coerce_audio_tensor(multimodal_output.get("audio"), async_chunk=False)
            sample_rate = extract_sample_rate(multimodal_output)
            output_name = resolve_output_name(
                args.output_name,
                args.case,
                request_index(output.request_id, fallback_index),
                args.num_prompts,
            )
            output_path = str(output_dir / output_name)
            write_wav(output_path, waveform, sample_rate)
            summary = {
                "request_id": output.request_id,
                "stage_id": output.stage_id,
                "output_path": output_path,
                "stage_durations": output.stage_durations,
                "peak_memory_mb": output.peak_memory_mb,
            }
            summary.update(audio_summary(waveform, sample_rate))
            summaries.append(summary)
            print(f"Saved output to {output_path}")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summaries:
            raise RuntimeError("Non-streaming Ming example produced no audio outputs")
        return summaries
    finally:
        engine.close()


def run_generation(args, prompt_payload, sampling_params_list, output_dir: Path, stats_log_file: str | None):
    if args.streaming:
        return asyncio.run(run_streaming(args, prompt_payload, sampling_params_list, output_dir, stats_log_file))
    return run_non_streaming(args, prompt_payload, sampling_params_list, output_dir, stats_log_file)
