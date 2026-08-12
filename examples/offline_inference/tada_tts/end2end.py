"""Offline inference demo for TADA TTS via vLLM-Omni.

TADA (Text-Acoustic Dual-Aligned) is a two-stage TTS model by HumeAI:
  Stage 0: Llama-3.2 AR backbone generates 512-dim acoustic features via
           per-token flow-matching diffusion (HumeAI/tada-1b or tada-3b-ml).
  Stage 1: codec decoder converts acoustic features to a 24 kHz waveform
           (HumeAI/tada-codec).

Usage (batch mode, requires tada_tts.yaml):
  python end2end.py \\
    --model HumeAI/tada-1b \\
    --stage-configs-path path/to/vllm_omni/model_executor/stage_configs/tada_tts.yaml \\
    --ref-audio ref.wav --ref-text "<transcript of ref.wav>"

Usage (async-chunk streaming mode, requires tada_tts_async_chunk.yaml):
  python end2end.py \\
    --model HumeAI/tada-1b \\
    --stage-configs-path path/to/vllm_omni/model_executor/stage_configs/tada_tts_async_chunk.yaml \\
    --streaming --ref-audio ref.wav --ref-text "<transcript of ref.wav>"

Prerequisites:
  pip install soundfile
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
from vllm_omni.model_executor.models.tada_tts import prompt_utils

logger = logging.getLogger(__name__)

# TADA codec output sample rate (50 Hz frames × 480 upsample).
TADA_CODEC_SR = 24000


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Hello! I'm TADA, a text-to-speech model by Hume AI. "
    "I generate speech with natural prosody by combining language modeling "
    "with continuous acoustic feature synthesis.",
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet.",
]


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Write accumulated audio to a WAV file."""
    audio_data = mm.get("model_outputs") or mm.get("audio")
    if audio_data is None:
        logger.warning("No audio in multimodal_output for request %s", request_id)
        return
    sr_raw = mm.get("sr", 24000)
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


def _make_prompt(text: str, args, model_name: str) -> tuple[dict, int]:
    """Build a prompt for ``text``: voice cloning when --ref-audio is given, else zero-shot."""
    if args.ref_audio:
        if not args.ref_text:
            raise ValueError("--ref-audio requires --ref-text (the reference transcript).")
        return prompt_utils.build_voice_clone_prompt(
            text, args.ref_audio, args.ref_text, model_name, num_transition_steps=args.num_transition_steps
        )
    return prompt_utils.build_zeroshot_prompt(text, model_name)


def main(args):
    """Run batch (non-streaming) offline inference."""
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model
    texts = args.texts or _DEFAULT_PROMPTS[: args.num_prompts]

    omni = Omni(
        model=model_name,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    # One generate() call per text so each request gets its own fixed walk length
    # (max_tokens). The engine persists across calls.
    for text in texts:
        prompt, walk_len = _make_prompt(text, args, model_name)
        sampling_params_list = prompt_utils.apply_walk_sampling_params(omni.default_sampling_params_list, walk_len)
        for stage_outputs in omni.generate([prompt], sampling_params_list=sampling_params_list):
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
        prompt, walk_len = _make_prompt(text, args, model_name)
        sampling_params_list = prompt_utils.apply_walk_sampling_params(omni.default_sampling_params_list, walk_len)
        t_start = time.perf_counter()
        chunk_idx = 0
        consumed = 0  # list case: chunks already taken
        audio_chunks: list = []  # list case: collected new chunks
        latest_audio = None  # tensor case: cumulative waveform-so-far
        sr = TADA_CODEC_SR
        first_audio_ts = None

        async for stage_output in omni.generate(
            prompt, request_id=request_id, sampling_params_list=sampling_params_list
        ):
            # Stage-0 emits text-type CompletionOutputs (no audio); audio arrives only on
            # the final audio stage. Gate on final_output_type like other TTS examples.
            if getattr(stage_output, "final_output_type", None) != "audio":
                continue
            output = stage_output.request_output
            comp = output.outputs[0] if output.outputs else None
            mm = getattr(comp, "multimodal_output", None) or {}

            # The output processor remaps model_outputs -> "audio" and CONCAT_LAST-accumulates
            # it. In practice each audio yield delivers the request's CUMULATIVE waveform as a
            # growing Tensor; keep the latest. (Handle a chunk-list delivery too, via a cursor.)
            audio_data = mm.get("audio")
            if audio_data is None:
                audio_data = mm.get("model_outputs")
            got = False
            if isinstance(audio_data, torch.Tensor) and audio_data.numel() > 0:
                latest_audio = audio_data
                got = True
            elif isinstance(audio_data, list):
                new = audio_data[consumed:]
                if new:
                    audio_chunks.extend(new)
                    consumed = len(audio_data)
                    got = True
            if got:
                chunk_idx += 1
                if first_audio_ts is None:
                    first_audio_ts = time.perf_counter() - t_start
                sr_raw = mm.get("sr", sr)
                sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
                sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Request %s done in %.2f s (chunks=%d, first audio @ %.2fs)",
            request_id,
            elapsed,
            chunk_idx,
            first_audio_ts if first_audio_ts is not None else -1.0,
        )
        final = [latest_audio] if latest_audio is not None else audio_chunks
        _save_wav(args.output_dir, request_id, {"model_outputs": final, "sr": sr})


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
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio WAV for voice cloning. Requires --ref-text.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of --ref-audio (used for forced alignment; avoids ASR).",
    )
    parser.add_argument(
        "--num-transition-steps",
        type=int,
        default=5,
        help="Voice cloning: number of transcript-tail tokens walked (and dropped) to smooth "
        "the prompt→synthesis boundary.",
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
