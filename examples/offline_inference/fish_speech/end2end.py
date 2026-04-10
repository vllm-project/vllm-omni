"""Offline inference demo for Fish Speech S2 Pro via vLLM Omni.

Generates speech from text using the fishaudio/s2-pro model.
Supports both text-only synthesis and voice cloning with reference audio.

Usage:
    # Text-only synthesis
    python end2end.py --text "Hello, this is a test."

    # Voice cloning with reference audio
    python end2end.py --text "Hello, this is a test." --ref-audio ref.wav --ref-text "Reference text."

    # Streaming mode
    python end2end.py --text "Hello, this is a test." --streaming
"""

import asyncio
import logging
import math
import os
import time

import numpy as np
import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni
from vllm_omni.model_executor.models.fish_speech.dac_utils import DAC_HOP_LENGTH, DAC_SAMPLE_RATE
from vllm_omni.model_executor.models.fish_speech.prompt_utils import (
    build_fish_text_only_prompt_ids,
    estimate_fish_voice_clone_prompt_len_from_normalized,
    normalize_fish_voice_clone_texts,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "fishaudio/s2-pro"
DEFAULT_STAGE_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "vllm_omni",
    "model_executor",
    "stage_configs",
    "fish_speech_s2_pro.yaml",
)


def build_prompt(
    text: str,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Build a prompt for Fish Speech S2 Pro.

    Uses the same text-only / structured-clone protocol as serving.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if ref_audio_path is None and ref_text is None:
        prompt_ids, normalized_text = build_fish_text_only_prompt_ids(tokenizer, text)
        additional_information = {
            "text": [normalized_text],
        }
        return {
            "prompt_token_ids": prompt_ids,
            "additional_information": additional_information,
        }

    if not ref_audio_path or not ref_text:
        raise ValueError("Fish Speech voice cloning requires both --ref-audio and --ref-text")

    normalized_text, normalized_ref_text = normalize_fish_voice_clone_texts(text, ref_text)
    ref_audio_wav, ref_audio_sr = sf.read(ref_audio_path, dtype="float32", always_2d=False)
    semantic_len = _estimate_fish_ref_code_len(ref_audio_wav, ref_audio_sr)
    ph_len = estimate_fish_voice_clone_prompt_len_from_normalized(
        tokenizer,
        normalized_text,
        normalized_ref_text,
        semantic_len,
    )

    additional_information = {
        "text": normalized_text,
        "ref_text": normalized_ref_text,
        "ref_audio_wav": torch.from_numpy(np.asarray(ref_audio_wav, dtype=np.float32)),
        "ref_audio_sr": int(ref_audio_sr),
        "fish_structured_voice_clone": True,
    }

    return {
        "prompt_token_ids": [1] * ph_len,
        "additional_information": additional_information,
    }


def _estimate_fish_ref_code_len(wav: np.ndarray, sample_rate: int) -> int:
    """Estimate Fish semantic token length from local reference audio."""
    n_samples = int(np.asarray(wav).shape[0]) if np.asarray(wav).ndim > 0 else 0
    if sample_rate <= 0 or n_samples <= 0:
        raise ValueError("Reference audio must contain samples and a positive sample rate")
    resampled_len = max(1, math.ceil(n_samples * DAC_SAMPLE_RATE / int(sample_rate)))
    return max(1, math.ceil(resampled_len / DAC_HOP_LENGTH))


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Concatenate audio chunks and write to a wav file."""
    audio_data = mm["audio"]
    sr_raw = mm["sr"]
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
    audio_tensor = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_tensor.float().cpu().numpy().flatten(), samplerate=sr, format="WAV")
    logger.info("Request %s: saved audio to %s (sr=%d)", request_id, out_wav, sr)


def main(args):
    """Run offline inference with Omni."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    stage_configs_path = args.stage_configs_path or DEFAULT_STAGE_CONFIG
    model_name = args.model or DEFAULT_MODEL

    inputs = [
        build_prompt(
            text=args.text,
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            model_name=model_name,
        )
    ]

    omni = Omni(
        model=model_name,
        stage_configs_path=stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    for stage_outputs in omni.generate(inputs):
        request_output = stage_outputs.request_output
        if request_output is None or not request_output.outputs:
            continue
        _save_wav(
            output_dir,
            request_output.request_id,
            request_output.outputs[0].multimodal_output,
        )
    t_end = time.perf_counter()
    logger.info("Total inference time: %.1f ms", (t_end - t_start) * 1000)


async def main_streaming(args):
    """Run offline inference with AsyncOmni for streaming audio chunks."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    stage_configs_path = args.stage_configs_path or DEFAULT_STAGE_CONFIG
    model_name = args.model or DEFAULT_MODEL

    prompt = build_prompt(
        text=args.text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        model_name=model_name,
    )

    omni = AsyncOmni(
        model=model_name,
        stage_configs_path=stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    request_id = "0"
    t_start = time.perf_counter()
    t_prev = t_start
    chunk_idx = 0

    async for stage_output in omni.generate(prompt, request_id=request_id):
        mm = stage_output.request_output.outputs[0].multimodal_output
        if not stage_output.finished:
            t_now = time.perf_counter()
            audio = mm.get("audio")
            n = len(audio) if isinstance(audio, list) else (0 if audio is None else 1)
            dt_ms = (t_now - t_prev) * 1000
            ttfa_ms = (t_now - t_start) * 1000
            if chunk_idx == 0:
                logger.info("Request %s: chunk %d samples=%d TTFA=%.1fms", request_id, chunk_idx, n, ttfa_ms)
            else:
                logger.info("Request %s: chunk %d samples=%d inter_chunk=%.1fms", request_id, chunk_idx, n, dt_ms)
            t_prev = t_now
            chunk_idx += 1
        else:
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000
            logger.info("Request %s: done total=%.1fms chunks=%d", request_id, total_ms, chunk_idx)
            _save_wav(output_dir, request_id, mm)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Fish Speech S2 Pro offline inference with vLLM Omni",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the Fish Speech text to speech system.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Reference text corresponding to the reference audio.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage configs YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated wav files.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Timeout for initializing stages in seconds.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Stream audio chunks as they arrive via AsyncOmni.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.streaming:
        asyncio.run(main_streaming(args))
    else:
        main(args)
