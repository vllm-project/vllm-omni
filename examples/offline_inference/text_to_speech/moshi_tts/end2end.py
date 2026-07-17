"""Offline inference demo for Moshi TTS via vLLM Omni.

Examples
--------
# Single sentence (non-streaming):
python end2end.py --model /path/to/tts-0.75b-en-public-hf

# Stream audio chunks as they arrive (async_chunk mode):
python end2end.py --model /path/to/tts-0.75b-en-public-hf --streaming

# Speaker conditioning via pre-computed embedding (tts-1.6b-en_fr):
python end2end.py --model /path/to/tts-1.6b-en_fr \\
    --text "Bonjour, je suis Moshi." \\
    --speaker-embedding /path/to/speaker.safetensors \\
    --deploy-config vllm_omni/deploy/moshi_tts.yaml
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
from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

logger = logging.getLogger(__name__)

_DEFAULT_TEXT = "Hello, this is Moshi, a real-time spoken dialogue language model."


def _build_prompt(
    text: str,
    voice: str | None = None,
    speaker_embedding: str | None = None,
) -> dict:
    """Build a single Moshi TTS request prompt.

    Args:
        text: Text to synthesise.
        voice: Optional voice prefix — local path, http(s) URL, or base64
               data URI (audio code prefix for tts-0.75b-en-public).
        speaker_embedding: Optional path or http(s) URL to a ``.npy``/``.pt`` / ``.safetensors``
               speaker embedding file for cross-attention conditioning
               (tts-1.6b-en_fr).  Downloaded and decoded server-side.
    """
    info: dict = {"text": [text]}
    if voice:
        info["prefix_wav"] = [voice]
        info["prefix_wav_key"] = [voice]  # stable key for speaker cache
    if speaker_embedding:
        info["speaker_embedding"] = speaker_embedding
    return {
        "prompt_token_ids": [0],
        "additional_information": info,
    }


def _save_wav(output_dir: str, request_id: str, mm: dict) -> None:
    """Concatenate audio chunks and write to a WAV file."""
    audio_data = mm.get("audio")
    if audio_data is None:
        audio_data = mm.get("model_outputs")
    sr_raw = mm.get("sr")
    if sr_raw is None:
        sr = 24000
    else:
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sr = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)

    if isinstance(audio_data, list):
        audio_tensor = torch.cat([a.reshape(-1) for a in audio_data], dim=-1)
    elif isinstance(audio_data, torch.Tensor):
        audio_tensor = audio_data.reshape(-1)
    else:
        logger.warning("No audio data in output for request %s", request_id)
        return

    out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
    sf.write(out_wav, audio_tensor.float().cpu().numpy(), samplerate=sr, format="WAV")
    logger.info("Saved audio to %s (sr=%d, samples=%d)", out_wav, sr, audio_tensor.numel())


def main(args) -> None:
    """Run offline inference with Omni (blocking)."""
    os.makedirs(args.output_dir, exist_ok=True)

    texts = [_DEFAULT_TEXT]
    if args.text:
        texts = [args.text]
    elif args.txt_prompts:
        with open(args.txt_prompts) as fh:
            texts = [line.strip() for line in fh if line.strip()]
        if not texts:
            raise ValueError(f"No valid prompts found in {args.txt_prompts}")

    inputs = [_build_prompt(t, voice=args.voice, speaker_embedding=args.speaker_embedding) for t in texts]

    omni_kwargs = vars(args).copy()
    omni = Omni(**omni_kwargs)

    for stage_outputs in omni.generate(inputs):
        output = stage_outputs.request_output
        _save_wav(args.output_dir, output.request_id, output.outputs[0].multimodal_output)


async def main_streaming(args) -> None:
    """Run offline inference with AsyncOmni, logging each audio chunk."""
    os.makedirs(args.output_dir, exist_ok=True)

    texts = [_DEFAULT_TEXT]
    if args.text:
        texts = [args.text]
    elif args.txt_prompts:
        with open(args.txt_prompts) as fh:
            texts = [line.strip() for line in fh if line.strip()]

    omni_kwargs = vars(args).copy()
    omni = AsyncOmni(**omni_kwargs)

    for i, text in enumerate(texts):
        prompt = _build_prompt(text, voice=args.voice, speaker_embedding=args.speaker_embedding)
        request_id = str(i)
        t_start = time.perf_counter()
        t_prev = t_start
        chunk_idx = 0

        async for stage_output in omni.generate(prompt, request_id=request_id):
            mm = stage_output.request_output.outputs[0].multimodal_output
            t_now = time.perf_counter()
            if not stage_output.finished:
                audio = mm.get("audio") or mm.get("model_outputs")
                n = (
                    sum(a.numel() for a in audio)
                    if isinstance(audio, list)
                    else (audio.numel() if isinstance(audio, torch.Tensor) else 0)
                )
                dt_ms = (t_now - t_prev) * 1000
                ttfa_ms = (t_now - t_start) * 1000
                if chunk_idx == 0:
                    logger.info("Request %s chunk %d samples=%d TTFA=%.1f ms", request_id, chunk_idx, n, ttfa_ms)
                else:
                    logger.info("Request %s chunk %d samples=%d inter=%.1f ms", request_id, chunk_idx, n, dt_ms)
                t_prev = t_now
            else:
                total_ms = (t_now - t_start) * 1000
                logger.info("Request %s done total=%.1f ms chunks=%d", request_id, total_ms, chunk_idx)
            _save_wav(args.output_dir, f"{request_id}_chunk_{chunk_idx:03d}", mm)
            chunk_idx += 1


def parse_args():
    parser = FlexibleArgumentParser(description="Moshi TTS offline inference via vLLM Omni")
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="HF repo id or local path to a Moshi TTS checkpoint"
    )
    parser.add_argument("--text", "-t", type=str, default=None, help="Text to synthesise")
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default=None,
        help=(
            "Voice prefix for audio-conditioned models (e.g. tts-0.75b-en-public). "
            "Accepts a local file path, http(s) URL, or base64 data URI."
        ),
    )
    parser.add_argument(
        "--speaker-embedding",
        type=str,
        default=None,
        help=(
            "Path or http(s) URL to a speaker embedding file for cross-attention "
            "conditioning (tts-1.6b-en_fr). Accepts .safetensors (key 'speaker_wavs', "
            "as produced by moshi/), .npy, or .pt. Loaded server-side."
        ),
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one sentence per line",
    )
    parser.add_argument(
        "--output-dir",
        default="output_moshi_tts",
        help="Directory for generated WAV files (default: output_moshi_tts)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Stream audio chunks via AsyncOmni as they arrive (async_chunk mode)",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Stage initialisation timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Overall init timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default="vllm_omni/deploy/moshi_tts.yaml",
        help="Path to a deploy YAML file (e.g. vllm_omni/deploy/moshi_tts.yaml). "
        "For TTS it should always be set to avoid confusion with moshi speech-to-speech.",
    )
    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.streaming:
        asyncio.run(main_streaming(args))
    else:
        main(args)
