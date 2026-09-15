# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline inference demo for Audio8 TTS Preview 0.6B via vLLM Omni.

Usage:
    # Text-only synthesis
    python end2end.py --text "Welcome to Audio8 TTS."

    # Zero-shot voice cloning (the transcript must match the reference audio)
    python end2end.py --text "Welcome to Audio8 TTS." \
        --ref-audio reference.wav --ref-text "The exact transcript of the reference recording."

    # Streaming (per-chunk TTFA / inter-chunk timings)
    python end2end.py --text "Welcome to Audio8 TTS." --streaming
"""

import asyncio
import os
import time

import numpy as np
import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_omni import AsyncOmni, Omni
from vllm_omni.model_executor.models.audio8_tts.codec_utils import estimate_reference_code_frames
from vllm_omni.model_executor.models.audio8_tts.prompt_utils import (
    build_text_only_prompt_ids,
    estimate_voice_clone_prompt_len,
    normalize_text,
)
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

DEFAULT_MODEL = "Audio8/Audio8-TTS-Preview-0.6b"


def build_prompt(
    text: str,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Build an engine prompt, using the same protocol as online serving."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if ref_audio_path is None and ref_text is None:
        prompt_ids, normalized_text = build_text_only_prompt_ids(tokenizer, text)
        return {
            "prompt_token_ids": prompt_ids,
            "additional_information": {"text": [normalized_text]},
        }

    if not ref_audio_path or not ref_text:
        raise ValueError("Audio8 TTS voice cloning requires both --ref-audio and --ref-text")

    normalized_text = normalize_text(text)
    normalized_ref_text = normalize_text(ref_text, add_default_speaker=True)
    ref_audio_wav, ref_audio_sr = sf.read(ref_audio_path, dtype="float32", always_2d=False)
    if ref_audio_wav.ndim > 1:
        ref_audio_wav = ref_audio_wav.mean(axis=-1)
    ref_frames = estimate_reference_code_frames(int(ref_audio_wav.shape[0]), int(ref_audio_sr))
    placeholder_len = estimate_voice_clone_prompt_len(tokenizer, normalized_text, normalized_ref_text, ref_frames)

    # The clone prompt embeds the reference audio's own codec codes, so the
    # model builds it in preprocess(); we only reserve the exact length here.
    return {
        "prompt_token_ids": [1] * placeholder_len,
        "additional_information": {
            "text": normalized_text,
            "ref_text": normalized_ref_text,
            "ref_audio_wav": torch.from_numpy(np.asarray(ref_audio_wav, dtype=np.float32)),
            "ref_audio_sr": int(ref_audio_sr),
            "audio8_structured_voice_clone": True,
        },
    }


def extract_audio(multimodal_output: dict) -> tuple[torch.Tensor, int]:
    """Pull the waveform and sample rate out of a multimodal output.

    The output processor concatenates the per-step delta tensors under
    ``model_outputs``; ``audio`` is the older alias. Never use ``a or b`` on
    these values -- truthiness on a tensor raises.
    """
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    if audio is None:
        raise ValueError(f"No audio in multimodal_output: {sorted(multimodal_output)}")
    if isinstance(audio, list):
        audio = torch.cat([torch.as_tensor(chunk).reshape(-1) for chunk in audio], dim=0)
    audio = torch.as_tensor(audio).reshape(-1)

    sr_raw = multimodal_output.get("sr")
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else None
    if sr_raw is None:
        raise ValueError("Missing sample rate in multimodal_output")
    sample_rate = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw)
    return audio, sample_rate


def _save_wav(output_dir: str, request_id: str, audio: torch.Tensor, sample_rate: int) -> None:
    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio.float().cpu().numpy(), samplerate=sample_rate, format="WAV")
    duration = audio.numel() / sample_rate
    # print(), not logging: vLLM reconfigures logging on engine start and
    # disables loggers created before that, which would swallow these lines.
    print(f"Request {request_id}: saved {out_wav} (sr={sample_rate}, {duration:.2f}s)")


def _omni_kwargs(args, model_name: str) -> dict:
    kwargs = {
        "model": model_name,
        "log_stats": args.log_stats,
        "stage_init_timeout": args.stage_init_timeout,
    }
    if args.deploy_config:
        kwargs["deploy_config"] = args.deploy_config
    return kwargs


def main(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model or DEFAULT_MODEL
    prompts = [
        build_prompt(text, args.ref_audio, args.ref_text, model_name) for text in ([args.text] * args.num_prompts)
    ]

    omni = Omni(**_omni_kwargs(args, model_name))
    t_start = time.perf_counter()
    total_audio_seconds = 0.0
    for stage_outputs in omni.generate(prompts):
        request_output = stage_outputs
        if request_output is None or not request_output.outputs:
            continue
        audio, sample_rate = extract_audio(request_output.outputs[0].multimodal_output)
        assert audio.numel() > 0, "Audio8 TTS produced an empty waveform"
        total_audio_seconds += audio.numel() / sample_rate
        _save_wav(args.output_dir, request_output.request_id, audio, sample_rate)
    elapsed = time.perf_counter() - t_start
    rtf = elapsed / max(total_audio_seconds, 1e-6)
    print(f"Total {elapsed * 1000:.1f} ms for {total_audio_seconds:.2f} s of audio (RTF={rtf:.3f})")


async def main_streaming(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model or DEFAULT_MODEL
    prompt = build_prompt(args.text, args.ref_audio, args.ref_text, model_name)

    omni = AsyncOmni(**_omni_kwargs(args, model_name))
    request_id = "0"
    chunks: list[torch.Tensor] = []
    sample_rate = None
    t_start = time.perf_counter()
    t_prev = t_start
    chunk_index = 0

    async for stage_output in omni.generate(prompt, request_id=request_id):
        multimodal_output = stage_output.outputs[0].multimodal_output
        if not multimodal_output:
            continue
        audio = multimodal_output.get("model_outputs")
        if audio is None:
            audio = multimodal_output.get("audio")
        if audio is not None:
            if isinstance(audio, list):
                chunks.extend(torch.as_tensor(chunk).reshape(-1) for chunk in audio)
            else:
                chunks.append(torch.as_tensor(audio).reshape(-1))
        sr_raw = multimodal_output.get("sr")
        if isinstance(sr_raw, list):
            sr_raw = sr_raw[-1] if sr_raw else None
        if sr_raw is not None:
            sample_rate = int(sr_raw.item()) if hasattr(sr_raw, "item") else int(sr_raw)

        if stage_output.finished:
            break
        now = time.perf_counter()
        if chunk_index == 0:
            print(f"chunk 0: TTFA={(now - t_start) * 1000:.1f} ms")
        else:
            print(f"chunk {chunk_index}: inter_chunk={(now - t_prev) * 1000:.1f} ms")
        t_prev = now
        chunk_index += 1

    if not chunks or sample_rate is None:
        raise RuntimeError("Audio8 TTS streaming produced no audio")
    _save_wav(args.output_dir, request_id, torch.cat(chunks, dim=0), sample_rate)
    print(f"Streaming done: total={(time.perf_counter() - t_start) * 1000:.1f} ms chunks={chunk_index}")


def parse_args():
    parser = TrackingArgumentParser(description="Audio8 TTS Preview offline inference with vLLM Omni")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or HuggingFace repo ID.")
    parser.add_argument(
        "--text",
        type=str,
        default="Welcome to Audio8 TTS, a compact multilingual text to speech model.",
        help="Text to synthesize.",
    )
    parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio path for voice cloning.")
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of --ref-audio; required for voice cloning and must match the recording.",
    )
    parser.add_argument("--output-dir", type=str, default="output_audio", help="Directory for output WAV files.")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Duplicate the prompt N times to smoke-test concurrent requests.",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override the deploy config path; defaults to vllm_omni/deploy/audio8_tts.yaml via model_type.",
    )
    parser.add_argument("--streaming", action="store_true", help="Use AsyncOmni and report per-chunk latency.")
    parser.add_argument("--log-stats", action="store_true", help="Enable engine stats logging.")
    parser.add_argument("--stage-init-timeout", type=int, default=600, help="Per-stage init timeout in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.streaming:
        asyncio.run(main_streaming(parsed_args))
    else:
        main(parsed_args)
