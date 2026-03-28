"""
end2end.py  —  Llama-3.1-8B + Qwen3-TTS composable offline demo
================================================================
Demonstrates RFC Theme 2: any vLLM text LLM can be paired with
any TTS decoder without a built-in talker stage.

Stage 0: meta-llama/Llama-3.1-8B-Instruct  (generates response text)
Stage 1: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice  (synthesizes audio)

The sentence-chunking bridge in text_tts_bridge.py connects them.

Usage
-----
# Basic
python end2end.py --prompt "Explain how neural networks learn."

# Custom voice / language
python end2end.py \\
    --prompt  "Bonjour, comment allez-vous?" \\
    --voice   serena \\
    --language French

# Streaming (requires async_chunk: true in the YAML, default)
python end2end.py --prompt "Tell me a short story." --streaming

# Override both model paths (if not using HF Hub defaults)
python end2end.py \\
    --llm-model  /path/to/Llama-3.1-8B-Instruct \\
    --tts-model  /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \\
    --prompt     "Hello from a local model."
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Resolve project root so the script runs from the repo root or this dir
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]  # examples/offline_inference/llama3_qwen3tts → root
sys.path.insert(0, str(_REPO_ROOT))

from vllm_omni.entrypoints.omni import Omni  # noqa: E402  (after sys.path fix)

# ---------------------------------------------------------------------------
# Default model IDs  (override with CLI args)
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_STAGE_CONFIGS = str(_REPO_ROOT / "vllm_omni/model_executor/stage_configs/text_tts.yaml")


# ---------------------------------------------------------------------------
# Input builder
# ---------------------------------------------------------------------------


def build_input(
    prompt: str,
    voice: str = "vivian",
    language: str = "English",
    instructions: str | None = None,
) -> dict:
    """
    Build the Stage 0 input dict.

    The bridge processor (text_tts_bridge.text2tts) will intercept
    Stage 0's streaming output and inject `voice`, `language`, and
    `instructions` into every Stage 1 chunk via `extra`.
    """
    return {
        "prompt": prompt,
        # These are forwarded through `stage0_output["extra"]` by the
        # async_chunk orchestrator so the bridge can inject them into
        # Qwen3-TTS inputs without Stage 0 knowing about them.
        "extra": {
            "tts_voice": voice,
            "tts_language": language,
            **({"tts_instructions": instructions} if instructions else {}),
        },
    }


# ---------------------------------------------------------------------------
# Audio sink: collect and write WAV
# ---------------------------------------------------------------------------


def save_audio(stage_outputs, output_path: str) -> None:
    """Collect audio tensors from stage outputs and write a WAV file."""
    audio_chunks: list[torch.Tensor] = []
    sample_rate: int = 24000  # Qwen3-TTS default output sample rate

    for stage_out in stage_outputs:
        mm = stage_out.get("multimodal_outputs", {})
        if "audio" not in mm:
            continue
        audio_data = mm["audio"]
        sr_raw = mm.get("sr", sample_rate)
        sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
        sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)

        chunk = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
        audio_chunks.append(chunk)

    if not audio_chunks:
        print("[warn] No audio output received — check bridge config and model logs.")
        return

    audio_tensor = torch.cat(audio_chunks, dim=-1).float().cpu()
    sf.write(output_path, audio_tensor.numpy().flatten(), samplerate=sample_rate, format="WAV")
    print(f"[✓] Audio saved → {output_path}  (sr={sample_rate} Hz)")


# ---------------------------------------------------------------------------
# Streaming variant (uses AsyncOmni internally via --streaming flag)
# ---------------------------------------------------------------------------


async def run_streaming(omni_async, inputs: dict, output_path: str) -> None:
    """Stream audio chunks as they arrive from the async_chunk pipeline."""

    request_id = f"req-{uuid.uuid4().hex[:8]}"
    audio_chunks: list[torch.Tensor] = []
    sample_rate = 24000
    chunk_idx = 0

    async for stage_outputs in omni_async.generate(inputs, request_id=request_id):
        for out in stage_outputs:
            mm = out.get("multimodal_outputs", {})
            if "audio" not in mm:
                continue
            audio_data = mm["audio"]
            sr_raw = mm.get("sr", sample_rate)
            sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
            sample_rate = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
            chunk = torch.cat(audio_data, dim=-1) if isinstance(audio_data, list) else audio_data
            audio_chunks.append(chunk)
            print(f"  [stream] received audio chunk #{chunk_idx}  ({chunk.shape[-1] / sample_rate:.2f}s)")
            chunk_idx += 1

    if audio_chunks:
        full = torch.cat(audio_chunks, dim=-1).float().cpu()
        sf.write(output_path, full.numpy().flatten(), samplerate=sample_rate, format="WAV")
        print(f"[✓] Streamed audio saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Llama-3.1 + Qwen3-TTS composable pipeline offline demo")
    p.add_argument(
        "--prompt",
        type=str,
        default="Tell me a fascinating fact about the universe in two sentences.",
        help="Text prompt sent to the LLM (Stage 0).",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="HF model ID or local path for Stage 0 text LLM.",
    )
    p.add_argument(
        "--tts-model",
        type=str,
        default=DEFAULT_TTS_MODEL,
        help="HF model ID or local path for Stage 1 TTS model.",
    )
    p.add_argument(
        "--stage-configs-path",
        type=str,
        default=DEFAULT_STAGE_CONFIGS,
        help="Path to the stage config YAML file.",
    )
    p.add_argument(
        "--voice",
        type=str,
        default="vivian",
        help="TTS voice/speaker name (injected by the bridge; Stage 0 is unaware).",
    )
    p.add_argument(
        "--language",
        type=str,
        default="English",
        help="TTS language tag.",
    )
    p.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Optional TTS style instruction, e.g. 'speak slowly and warmly'.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output_audio",
        help="Directory to write output WAV files.",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Stream audio chunks progressively via AsyncOmni.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    request_id = uuid.uuid4().hex[:8]
    output_path = os.path.join(args.output_dir, f"output_{request_id}.wav")

    inputs = build_input(
        prompt=args.prompt,
        voice=args.voice,
        language=args.language,
        instructions=args.instructions,
    )

    print(f"[pipeline]  LLM  : {args.llm_model}")
    print(f"[pipeline]  TTS  : {args.tts_model}")
    print(f"[pipeline]  voice: {args.voice}  lang: {args.language}")
    print(f"[prompt]    {args.prompt!r}\n")

    if args.streaming:
        # ---- Async streaming path ----------------------------------------
        import asyncio

        from vllm_omni.entrypoints.async_omni import AsyncOmni

        async def _run():
            async_omni = AsyncOmni(
                model=args.llm_model,
                stage_configs_path=args.stage_configs_path,
            )
            await run_streaming(async_omni, inputs, output_path)
            await async_omni.shutdown()

        asyncio.run(_run())

    else:
        # ---- Synchronous offline path ------------------------------------
        omni = Omni(
            model=args.llm_model,
            stage_configs_path=args.stage_configs_path,
        )

        all_stage_outputs: list[dict] = []
        for stage_outputs in omni.generate([inputs]):
            all_stage_outputs.extend(stage_outputs)

        save_audio(all_stage_outputs, output_path)


if __name__ == "__main__":
    main()
