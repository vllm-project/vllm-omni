"""Minimal offline VoxCPM example for vLLM Omni."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYNC_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "voxcpm.yaml"


def _build_prompt(args) -> dict[str, Any]:
    additional_information: dict[str, list[Any]] = {
        "text": [args.text],
        "cfg_value": [args.cfg_value],
        "inference_timesteps": [args.inference_timesteps],
        "min_len": [args.min_len],
        "max_new_tokens": [args.max_new_tokens],
    }
    if args.streaming_prefix_len is not None:
        additional_information["streaming_prefix_len"] = [args.streaming_prefix_len]
    if args.ref_audio is not None:
        additional_information["ref_audio"] = [args.ref_audio]
    if args.ref_text is not None:
        additional_information["ref_text"] = [args.ref_text]
    return {
        "prompt_token_ids": [1],
        "additional_information": additional_information,
    }


def _extract_audio_tensor(mm: dict[str, Any]) -> torch.Tensor:
    audio = mm.get("audio", mm.get("model_outputs"))
    if audio is None:
        raise ValueError("No audio output found in multimodal output.")
    if isinstance(audio, list):
        parts = [torch.as_tensor(item).float().cpu().reshape(-1) for item in audio]
        audio = torch.cat(parts, dim=-1) if parts else torch.zeros(0)
    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    return audio.float().cpu().reshape(-1)


def _extract_sample_rate(mm: dict[str, Any]) -> int:
    sr_raw = mm.get("sr", 24000)
    if isinstance(sr_raw, list) and sr_raw:
        sr_raw = sr_raw[-1]
    if hasattr(sr_raw, "item"):
        return int(sr_raw.item())
    return int(sr_raw)


def _is_streaming_stage_config(stage_config_path: str) -> bool:
    return "async_chunk" in Path(stage_config_path).stem


def _save_audio(audio: torch.Tensor, sample_rate: int, output_dir: Path, request_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"output_{request_id}.wav"
    sf.write(
        output_path,
        audio.float().cpu().clamp(-1.0, 1.0).numpy(),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return output_path


async def _run_streaming(args) -> Path:
    prompt = _build_prompt(args)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path("output_audio_streaming")
    request_id = "streaming_example"
    sample_rate = 24000
    buffered_samples = 0
    chunks: list[torch.Tensor] = []
    started = time.perf_counter()
    omni = AsyncOmni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )
    try:
        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = getattr(stage_output, "multimodal_output", None)
            if not isinstance(mm, dict):
                request_output = getattr(stage_output, "request_output", None)
                if request_output is None:
                    continue
                mm = getattr(request_output, "multimodal_output", None)
                if not isinstance(mm, dict) and getattr(request_output, "outputs", None):
                    mm = getattr(request_output.outputs[0], "multimodal_output", None)
            if not isinstance(mm, dict):
                continue
            audio = _extract_audio_tensor(mm)
            if audio.numel() == 0:
                continue
            sample_rate = _extract_sample_rate(mm)
            if audio.numel() > buffered_samples:
                delta = audio[buffered_samples:]
                buffered_samples = int(audio.numel())
            else:
                delta = audio
                buffered_samples += int(delta.numel())
            if delta.numel() > 0:
                chunks.append(delta)
        if not chunks:
            raise RuntimeError("No streaming audio chunks received from VoxCPM.")
        output_audio = torch.cat(chunks, dim=0)
        output_path = _save_audio(output_audio, sample_rate, output_dir, request_id)
        print(f"Saved streaming audio to: {output_path} ({time.perf_counter() - started:.2f}s)")
        return output_path
    finally:
        omni.shutdown()


def _run_sync(args) -> Path:
    prompt = _build_prompt(args)
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path("output_audio")
    request_id = "sync_example"
    started = time.perf_counter()
    last_mm: dict[str, Any] | None = None
    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )
    for stage_outputs in omni.generate(prompt):
        request_output = getattr(stage_outputs, "request_output", None)
        if request_output is None:
            continue
        outputs = getattr(request_output, "outputs", None)
        if outputs:
            for output in outputs:
                mm = getattr(output, "multimodal_output", None)
                if isinstance(mm, dict):
                    last_mm = mm
        mm = getattr(request_output, "multimodal_output", None)
        if isinstance(mm, dict):
            last_mm = mm
    if last_mm is None:
        raise RuntimeError("No audio output received from VoxCPM.")
    output_path = _save_audio(
        _extract_audio_tensor(last_mm),
        _extract_sample_rate(last_mm),
        output_dir,
        request_id,
    )
    print(f"Saved audio to: {output_path} ({time.perf_counter() - started:.2f}s)")
    return output_path


def parse_args():
    parser = FlexibleArgumentParser(description="Minimal offline VoxCPM example for vLLM Omni.")
    parser.add_argument("--model", type=str, required=True, help="Local VoxCPM model directory.")
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=str(DEFAULT_SYNC_STAGE_CONFIG),
        help=("Stage config path. Use voxcpm.yaml for non-streaming or voxcpm_async_chunk.yaml for streaming."),
    )
    parser.add_argument("--text", type=str, required=True, help="Input text for synthesis.")
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio path for voice cloning.")
    parser.add_argument("--ref-text", type=str, default=None, help="Transcript of the reference audio.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for generated wav files.")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="Guidance value passed to VoxCPM.")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="Number of diffusion timesteps.")
    parser.add_argument("--min-len", type=int, default=2, help="Minimum latent length.")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum latent length.")
    parser.add_argument(
        "--streaming-prefix-len",
        type=int,
        default=3,
        help="Streaming prefix length used by voxcpm_async_chunk.yaml.",
    )
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Stage initialization timeout in seconds.")
    parser.add_argument("--log-stats", action="store_true", help="Enable vLLM Omni stats logging.")
    args = parser.parse_args()
    if (args.ref_audio is None) != (args.ref_text is None):
        raise ValueError("Voice cloning requires --ref-audio and --ref-text together.")
    return args


def main(args) -> None:
    route = "streaming" if _is_streaming_stage_config(args.stage_configs_path) else "sync"
    print(f"Model: {args.model}")
    print(f"Stage config: {args.stage_configs_path}")
    print(f"Route: {route}")
    if route == "streaming":
        asyncio.run(_run_streaming(args))
    else:
        _run_sync(args)


if __name__ == "__main__":
    main(parse_args())
