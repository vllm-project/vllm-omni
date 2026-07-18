#!/usr/bin/env python3
"""Measure MiniCPM-o Whisper hidden-state retention with a real audio input."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from safetensors import safe_open
from transformers import AutoProcessor, WhisperConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from vllm.multimodal.utils import fetch_audio

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
)


class AudioEncoderAdapter:
    """Override hidden-state retention while preserving the encoder interface."""

    def __init__(self, encoder: MiniCPMWhisperEncoder, retain_all: bool) -> None:
        self.encoder = encoder
        self.retain_all = retain_all
        self.conv1 = encoder.conv1

    def __call__(
        self,
        input_features: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> BaseModelOutputWithPast | tuple[object, ...]:
        del output_hidden_states
        return self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_hidden_states=self.retain_all,
        )


@dataclass
class AudioConfig:
    audio_pool_step: int
    audio_chunk_length: float


class AudioModelHarness:
    """Minimum typed model surface needed by ``get_audio_hidden_states``."""

    def __init__(
        self,
        *,
        encoder: MiniCPMWhisperEncoder,
        projection: MultiModalProjector,
        audio_pool_step: int,
        audio_chunk_length: float,
        retain_all: bool,
    ) -> None:
        self.config = AudioConfig(
            audio_pool_step=audio_pool_step,
            audio_chunk_length=audio_chunk_length,
        )
        self.apm = AudioEncoderAdapter(encoder, retain_all)
        self.audio_projection_layer = projection
        self.audio_avg_pooler = torch.nn.AvgPool1d(
            audio_pool_step,
            stride=audio_pool_step,
        )
        self.audio_encoder_layer = -1

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        mask = torch.zeros(size, size, device=device, dtype=torch.bool)
        for index in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max(
                    (index // chunk_size - num_left_chunks) * chunk_size,
                    0,
                )
            end = min(
                (index // chunk_size + 1) * chunk_size + num_lookhead,
                size,
            )
            mask[index, start:end] = True
        return mask

    def _get_feat_extract_output_lengths(
        self,
        input_lengths: torch.LongTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        return input_lengths_after_cnn, input_lengths_after_pooling.to(dtype=torch.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--latency-repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.latency_repetitions < 1:
        parser.error("--latency-repetitions must be positive")
    return args


def load_audio_states(
    model_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    encoder_state: dict[str, torch.Tensor] = {}
    projection_state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.startswith("apm."):
                    encoder_state[key.removeprefix("apm.")] = checkpoint.get_tensor(key)
                elif key.startswith("audio_projection_layer."):
                    projection_state[key.removeprefix("audio_projection_layer.")] = checkpoint.get_tensor(key)
    if not encoder_state or not projection_state:
        raise RuntimeError(f"audio weights not found under {model_path}")
    return encoder_state, projection_state


def prepare_audio_inputs(
    model_path: Path,
    audio_path: Path,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    waveform, sampling_rate = fetch_audio(audio_path.resolve().as_uri())
    if sampling_rate != 16_000:
        waveform = torchaudio.functional.resample(
            torch.from_numpy(waveform),
            int(sampling_rate),
            16_000,
        ).numpy()
        sampling_rate = 16_000
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        pool_step=5,
    )
    audio_features, audio_feature_lens, _ = processor.audio_feature_extract(
        audios=waveform,
        sampling_rate=sampling_rate,
    )
    data = {
        "audio_features": audio_features.to(device=device),
        "audio_feature_lens": [lengths.to(device=device) for lengths in audio_feature_lens],
    }
    report = {
        "audio_path": str(audio_path.resolve()),
        "sampling_rate": sampling_rate,
        "waveform_samples": int(waveform.shape[0]),
        "duration_seconds": waveform.shape[0] / sampling_rate,
        "audio_features_shape": list(audio_features.shape),
        "audio_feature_lens": [[int(value) for value in lengths.tolist()] for lengths in audio_feature_lens],
    }
    return data, report


def make_model(
    *,
    encoder: MiniCPMWhisperEncoder,
    projection: MultiModalProjector,
    audio_pool_step: int,
    audio_chunk_length: float,
    retain_all: bool,
) -> AudioModelHarness:
    return AudioModelHarness(
        encoder=encoder,
        projection=projection,
        audio_pool_step=audio_pool_step,
        audio_chunk_length=audio_chunk_length,
        retain_all=retain_all,
    )


def run_encoder(
    model: AudioModelHarness,
    data: dict[str, object],
) -> tuple[list[torch.Tensor], dict[str, int]]:
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    baseline_allocated = torch.accelerator.memory_allocated()
    baseline_reserved = torch.accelerator.memory_reserved()
    with torch.inference_mode():
        output = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_hidden_states(
            model,
            data,
        )
    torch.accelerator.synchronize()
    peak_allocated = torch.accelerator.max_memory_allocated()
    peak_reserved = torch.accelerator.max_memory_reserved()
    return output, {
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_delta_bytes": peak_allocated - baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_delta_bytes": peak_reserved - baseline_reserved,
    }


def benchmark_latency(
    model: AudioModelHarness,
    data: dict[str, object],
    repetitions: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        warmup = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_hidden_states(
            model,
            data,
        )
    del warmup
    torch.accelerator.synchronize()

    elapsed_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(repetitions):
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            output = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_hidden_states(
                model,
                data,
            )
            ended.record()
            ended.synchronize()
            elapsed_ms.append(started.elapsed_time(ended))
            del output

    return {
        "repetitions": repetitions,
        "mean_ms": statistics.fmean(elapsed_ms),
        "stdev_ms": statistics.stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
        "median_ms": statistics.median(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    }


def flatten_output(output: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in output]).float().cpu()


def main() -> None:
    args = parse_args()
    config_dict = json.loads((args.model / "config.json").read_text())
    audio_config = WhisperConfig(**config_dict["audio_config"])
    audio_config._attn_implementation = "eager"

    encoder = MiniCPMWhisperEncoder(audio_config)
    projection = MultiModalProjector(
        in_dim=audio_config.d_model,
        out_dim=config_dict["hidden_size"],
    )
    encoder_state, projection_state = load_audio_states(args.model)
    encoder.load_state_dict(encoder_state, strict=True)
    projection.load_state_dict(projection_state, strict=True)
    del encoder_state, projection_state

    device = torch.device("cuda")
    dtype = torch.bfloat16
    encoder.to(device=device, dtype=dtype).eval()
    projection.to(device=device, dtype=dtype).eval()
    data, input_report = prepare_audio_inputs(args.model, args.audio, device)

    common = {
        "encoder": encoder,
        "projection": projection,
        "audio_pool_step": int(config_dict.get("audio_pool_step", 5)),
        "audio_chunk_length": float(config_dict.get("audio_chunk_length", 1.0)),
    }
    retaining_model = make_model(**common, retain_all=True)
    final_only_model = make_model(**common, retain_all=False)

    retaining_output, retaining_memory = run_encoder(retaining_model, data)
    retaining_cpu = flatten_output(retaining_output)
    del retaining_output
    final_output, final_memory = run_encoder(final_only_model, data)
    final_cpu = flatten_output(final_output)
    del final_output

    retaining_latency = benchmark_latency(
        retaining_model,
        data,
        args.latency_repetitions,
    )
    final_latency = benchmark_latency(
        final_only_model,
        data,
        args.latency_repetitions,
    )

    difference = (retaining_cpu.double() - final_cpu.double()).abs()
    report = {
        "model": str(args.model.resolve()),
        "input": input_report,
        "encoder": {
            "attention_implementation": audio_config._attn_implementation,
            "layers": audio_config.encoder_layers,
            "hidden_size": audio_config.d_model,
            "dtype": str(dtype),
        },
        "output": {
            "numel": retaining_cpu.numel(),
            "max_absolute_error": difference.max().item(),
            "mean_absolute_error": difference.mean().item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                retaining_cpu.double(),
                final_cpu.double(),
                dim=0,
            ).item(),
        },
        "retaining_all_hidden_states": {
            "memory": retaining_memory,
            "latency": retaining_latency,
        },
        "final_hidden_state_only": {
            "memory": final_memory,
            "latency": final_latency,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
