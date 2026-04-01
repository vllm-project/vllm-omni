"""Offline inference demo for TADA TTS via vLLM-Omni.

TADA (Text-Acoustic Dual-Aligned) is a two-stage TTS model by HumeAI:
  Stage 0: Llama-3.2 AR backbone generates 512-dim acoustic features via
           per-token flow-matching diffusion (HumeAI/tada-1b or tada-3b-ml).
  Stage 1: TADA codec decoder converts acoustic features to waveform at
           44 100 Hz (HumeAI/tada-codec, loaded lazily from HuggingFace).

Usage (batch mode, requires tada_tts.yaml):
  python end2end.py \\
    --model HumeAI/tada-1b \\
    --stage-configs-path path/to/vllm_omni/model_executor/stage_configs/tada_tts.yaml

Usage (async-chunk mode, requires tada_tts_async_chunk.yaml):
  python end2end.py \\
    --model HumeAI/tada-1b \\
    --stage-configs-path path/to/vllm_omni/model_executor/stage_configs/tada_tts_async_chunk.yaml \\
    --streaming

Prerequisites:
  pip install hume-tada soundfile
"""

import asyncio
import logging
import os
import time

import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Hello! I'm TADA, a text-to-speech model by Hume AI. "
    "I generate speech with natural prosody by combining language modeling "
    "with continuous acoustic feature synthesis.",
    "The quick brown fox jumps over the lazy dog. "
    "This sentence contains every letter of the English alphabet.",
]


def _tokenize(text: str, model_name: str, _cache: dict = {}) -> list[int]:
    """Tokenize text using the model's tokenizer (Llama-3.2 by default)."""
    if model_name not in _cache:
        from transformers import AutoTokenizer

        # TADA uses the Llama-3.2 tokenizer.
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        _cache[model_name] = tok
    tok = _cache[model_name]
    return tok(text, return_tensors=None)["input_ids"]


def _build_prompt(text: str, model_name: str) -> dict:
    """Build a single vLLM-Omni prompt dict for TADA TTS."""
    token_ids = _tokenize(text, model_name)
    return {"prompt_token_ids": token_ids}


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Write accumulated audio to a WAV file."""
    audio_data = mm.get("model_outputs") or mm.get("audio")
    if audio_data is None:
        logger.warning("No audio in multimodal_output for request %s", request_id)
        return
    sr_raw = mm.get("sr", 44100)
    sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
    sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

    if isinstance(audio_data, list):
        audio_tensor = torch.cat([a.reshape(-1) for a in audio_data], dim=-1)
    else:
        audio_tensor = audio_data.reshape(-1)

    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_tensor.float().cpu().numpy(), samplerate=sr, format="WAV")
    logger.info("Saved %.2f s of audio to %s (sr=%d Hz)", len(audio_tensor) / sr, out_wav, sr)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def main(args):
    """Run batch (non-streaming) offline inference."""
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model
    texts = args.texts or _DEFAULT_PROMPTS[: args.num_prompts]
    prompts = [_build_prompt(t, model_name) for t in texts]

    omni = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    for stage_outputs in omni.generate(prompts):
        output = stage_outputs.request_output
        mm = output.outputs[0].multimodal_output or {}
        _save_wav(args.output_dir, output.request_id, mm)


# ---------------------------------------------------------------------------
# Streaming inference (async-chunk mode)
# ---------------------------------------------------------------------------


async def main_streaming(args):
    """Run streaming offline inference via AsyncOmni."""
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model
    texts = args.texts or _DEFAULT_PROMPTS[: args.num_prompts]

    omni = AsyncOmni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    for i, text in enumerate(texts):
        request_id = str(i)
        prompt = _build_prompt(text, model_name)
        t_start = time.perf_counter()
        chunk_idx = 0

        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output or {}
            if stage_output.finished:
                elapsed = time.perf_counter() - t_start
                logger.info("Request %s done in %.2f s (chunks=%d)", request_id, elapsed, chunk_idx)
                _save_wav(args.output_dir, request_id, mm)
            else:
                chunk_idx += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = FlexibleArgumentParser(description="TADA TTS offline inference via vLLM-Omni")
    parser.add_argument(
        "--model",
        type=str,
        default="HumeAI/tada-1b",
        help="HuggingFace model ID (default: HumeAI/tada-1b).",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage config YAML (tada_tts.yaml or tada_tts_async_chunk.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_audio",
        help="Output directory for generated WAV files (default: output_audio).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of default prompts to use (default: 1).",
    )
    parser.add_argument(
        "--texts",
        nargs="+",
        default=None,
        help="One or more text strings to synthesise.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use AsyncOmni streaming mode (requires tada_tts_async_chunk.yaml).",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable detailed statistics logging.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialisation timeout in seconds (default: 600; codec download may take time).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.streaming:
        asyncio.run(main_streaming(args))
    else:
        main(args)
