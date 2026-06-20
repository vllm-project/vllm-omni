# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSM-1B (Sesame) End-to-End Offline Inference Example.

CSM-1B is a two-stage TTS model:
  - Stage 0 (AR): a Llama-3.2-1B backbone samples codebook-0 per 80 ms frame and
    runs a 31-step depth decoder inline to complete the 32-code Mimi frame.
  - Stage 1: the Mimi vocoder decodes the frames to 24 kHz audio.

Plain text to speech with speaker-id conditioning (no reference-audio voice
cloning on this path); the OpenAI-style ``voice`` field maps to a CSM speaker id.

Usage:
    python examples/offline_inference/text_to_speech/csm/end2end.py \
        --model sesame/csm-1b \
        --text "The quick brown fox jumps over the lazy dog." \
        --speaker 0 \
        --output-dir ./output
"""

import logging
import os
import time
from typing import Any

import numpy as np
import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import Omni

logger = logging.getLogger(__name__)

DEFAULT_DEPLOY_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "vllm_omni", "deploy", "csm.yaml"
)
SAMPLE_RATE = 24000
# CSM's inline-depth AR loop has no reliable natural EOS; bound length with the
# same cap the served path uses (64 frames == 5.12 s at 80 ms/frame).
DEFAULT_MAX_FRAMES = 64


def _concat_audio(audio_val: Any) -> np.ndarray:
    """Concatenate audio tensors from the multimodal output."""
    if isinstance(audio_val, list):
        tensors = [torch.as_tensor(t).float().reshape(-1) for t in audio_val if t is not None]
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).cpu().numpy().astype(np.float32, copy=False)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.float().cpu().numpy().reshape(-1)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _extract_sample_rate(audio_mm: dict) -> int:
    sr_raw = audio_mm.get("sr", SAMPLE_RATE)
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else SAMPLE_RATE
    if hasattr(sr_raw, "item"):
        return int(sr_raw.item())
    return int(sr_raw)


def _build_csm_input(model: str, text: str, speaker: str, max_frames: int) -> dict:
    """Build the Stage-0 backbone prompt: ``<|begin_of_text|>[<spk>]<text><|end_of_text|>``.

    The backbone reads the ids back out of ``additional_information`` via ``_pick``
    (batch-of-1 convention), so every field is list-wrapped. Greedy (temperature 0,
    top_k 0) matches the served path and the bit-parity tests.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tagged = f"<|begin_of_text|>[{speaker}]{text}<|end_of_text|>"
    prompt_token_ids = list(tokenizer(tagged, add_special_tokens=False)["input_ids"])
    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": {
            "prompt_token_ids": [prompt_token_ids],
            "temperature": [0.0],
            "top_k": [0],
            "max_new_frames": [max_frames],
        },
    }


def main(args):
    """Run offline CSM-1B inference."""
    os.makedirs(args.output_dir, exist_ok=True)
    deploy_config_path = args.deploy_config or DEFAULT_DEPLOY_CONFIG
    max_frames = args.max_new_tokens or DEFAULT_MAX_FRAMES

    inputs = [_build_csm_input(args.model, args.text, str(args.speaker), max_frames)]

    omni = Omni(
        model=args.model,
        stage_configs_path=deploy_config_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    t_start = time.perf_counter()
    outputs = omni.generate(inputs)
    elapsed = (time.perf_counter() - t_start) * 1000

    assert outputs, "No outputs returned"
    audio_mm = outputs[0].multimodal_output
    assert "audio" in audio_mm, "No audio output found"

    audio = _concat_audio(audio_mm["audio"])
    sr = _extract_sample_rate(audio_mm)
    out_path = os.path.join(args.output_dir, "output.wav")
    sf.write(out_path, audio, samplerate=sr, format="WAV")

    logger.info("Saved %s (%.2fs @ %dHz)", out_path, len(audio) / sr, sr)
    logger.info("Total inference: %.1f ms", elapsed)


def parse_args():
    parser = FlexibleArgumentParser(description="CSM-1B Text-to-Speech Example")
    parser.add_argument("--model", type=str, default="sesame/csm-1b", help="Model path or HF id")
    parser.add_argument("--text", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--speaker", type=str, default="0", help="CSM speaker id (non-negative integer)")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max frames to generate (1 frame == 80 ms)")
    parser.add_argument("--deploy-config", type=str, default=None)
    parser.add_argument("--log-stats", action="store_true")
    parser.add_argument("--stage-init-timeout", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
