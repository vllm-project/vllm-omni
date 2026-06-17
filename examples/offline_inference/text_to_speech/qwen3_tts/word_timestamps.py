"""Offline Qwen3-TTS example with word-level timestamps.

This demo runs Qwen3-TTS offline, aligns the synthesized audio in memory with
the same shared forced aligner used by the streaming server path, then writes
the WAV and JSON sidecar:

    python examples/offline_inference/text_to_speech/qwen3_tts/word_timestamps.py \
        --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
        --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B

On machines without a local CUDA toolkit, set ``VLLM_USE_FLASHINFER_SAMPLER=0``
to avoid FlashInfer sampler JIT during warmup.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# The in-process offline aligner consumes token_classify outputs. Keeping V1
# multiprocessing off avoids msgspec serialization issues for token-wise logits.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import soundfile as sf
import torch
from end2end import _estimate_prompt_len
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni
from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.utils.forced_aligner import align, build_forced_aligner_config


def _default_stage_config() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "vllm_omni" / "deploy" / "qwen3_tts.yaml")


def _build_custom_voice_input(args: Any) -> dict[str, Any]:
    additional_information = {
        "task_type": ["CustomVoice"],
        "text": [args.text],
        "language": [args.language],
        "speaker": [args.speaker],
        "instruct": [args.instructions],
        "max_new_tokens": [args.max_new_tokens],
    }
    return {
        "prompt_token_ids": [0] * _estimate_prompt_len(additional_information, args.model),
        "additional_information": additional_information,
    }


def _audio_tensor_and_sample_rate(mm: dict[str, Any]) -> tuple[torch.Tensor, int]:
    audio_data = mm["audio"]
    sr_raw = mm["sr"]
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sample_rate = int(sr_val.item()) if hasattr(sr_val, "item") else int(sr_val)
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    return audio_tensor.float().detach().cpu().flatten(), sample_rate


def _float_audio_to_pcm16_bytes(audio: torch.Tensor) -> bytes:
    samples = audio.numpy()
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes()


async def _run_alignment(args: Any, audio: torch.Tensor, sample_rate: int) -> list[dict[str, Any]] | None:
    aligner_config = build_forced_aligner_config(args)
    if aligner_config is None:
        raise ValueError("--forced-aligner or --forced-aligner-config is required")
    timestamps = await align(
        audio=_float_audio_to_pcm16_bytes(audio),
        text=args.text,
        sample_rate=sample_rate,
        config=aligner_config,
        language=args.language,
    )
    if timestamps is None:
        return None
    return [
        {
            "word": item.word,
            "start_ms": item.start_ms,
            "end_ms": item.end_ms,
        }
        for item in timestamps
    ]


def main(args: Any) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    omni_kwargs = vars(args).copy()
    for key in (
        "forced_aligner",
        "text",
        "language",
        "speaker",
        "instructions",
        "max_new_tokens",
        "output_dir",
    ):
        omni_kwargs.pop(key, None)
    omni_kwargs["stage_configs_path"] = args.stage_configs_path or _default_stage_config()
    omni_kwargs["log_stats"] = args.log_stats
    omni = Omni(**omni_kwargs)

    prompt = _build_custom_voice_input(args)
    final_output = None
    for stage_outputs in omni.generate([prompt]):
        final_output = stage_outputs.request_output
    if final_output is None:
        raise RuntimeError("Qwen3-TTS did not produce an output.")

    mm = final_output.outputs[0].multimodal_output
    audio, sample_rate = _audio_tensor_and_sample_rate(mm)

    wav_path = output_dir / "qwen3_tts_word_timestamps.wav"
    # Align the in-memory tensor, not the saved WAV. The WAV is only a demo
    # artifact so users can listen to the same audio referenced by the sidecar.
    timestamps = asyncio.run(_run_alignment(args, audio, sample_rate))
    sf.write(wav_path, audio.numpy(), samplerate=sample_rate, format="WAV")

    sidecar = {
        "text": args.text,
        "sample_rate": sample_rate,
        "audio_path": str(wav_path),
        "timestamps": timestamps,
    }
    json_path = output_dir / "qwen3_tts_word_timestamps.json"
    json_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved audio: {wav_path}")
    print(f"Saved timestamps: {json_path}")
    print(json.dumps(sidecar["timestamps"], ensure_ascii=False, indent=2))


def parse_args() -> Any:
    parser = FlexibleArgumentParser(description="Offline Qwen3-TTS word timestamp example")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", help="Qwen3-TTS model path/name")
    parser.add_argument(
        "--forced-aligner",
        default=None,
        help="Qwen3 forced aligner model path/name",
    )
    parser.add_argument(
        "--forced-aligner-config",
        default=None,
        help="Optional YAML file for forced aligner settings (incl. gpu_memory_utilization)",
    )
    parser.add_argument("--stage-configs-path", default=None, help="Qwen3-TTS deploy YAML")
    parser.add_argument("--text", default="Hello world.", help="Text to synthesize and align")
    parser.add_argument("--language", default="English", help="Qwen3-TTS language field")
    parser.add_argument("--speaker", default="Vivian", help="CustomVoice speaker name")
    parser.add_argument("--instructions", default="", help="Optional speaking style instruction")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="TTS max_new_tokens")
    parser.add_argument("--output-dir", default="output_audio", help="Directory for WAV and JSON sidecar")
    parser.add_argument("--log-stats", action="store_true", default=False, help="Enable vLLM-Omni stats logging")
    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
