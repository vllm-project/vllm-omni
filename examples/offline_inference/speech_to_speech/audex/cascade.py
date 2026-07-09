# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Audex (Nemotron-Labs-Audex-2B) cascaded speech-to-speech example.

The official three-pass cascade over ONE ``nemotron_labs_audex_full``
deployment:

  1. ASR pass  (audio + "Transcribe the input speech." → transcript, text)
  2. chat pass (transcript → answer, text)
  3. TTS pass  (answer → speech, audio via the streaming causal decoder)

Text passes carry ``modalities: ["text"]`` and finish at stage 0; only the
TTS pass (``modalities: ["audio"]``) streams codec frames into code2wav.

Example:

    python examples/offline_inference/speech_to_speech/audex/cascade.py \\
        --audio-file question.wav --output results/answer.wav
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vllm_omni import Omni
from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt, build_null_prompt

SAMPLE_RATE = 16_000
ASR_QUESTION = "Transcribe the input speech."


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Audex cascaded S2S")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument("--audio-file", type=str, required=True, help="Spoken question (WAV).")
    parser.add_argument("--output", type=str, default="results/audex_s2s_answer.wav")
    parser.add_argument("--deploy-config", type=str, default=None)
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="CFG strength for the TTS pass (1.0 disables; official setting 1.5).",
    )
    return parser.parse_args()


def _chat_prompt(user_text: str) -> str:
    return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def _asr_prompt() -> str:
    return f"<|im_start|>user\n<so_embedding>\n{ASR_QUESTION}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def _clean(text: str) -> str:
    # The full checkpoint prefixes transcripts with a language tag sentence
    # and quotes the content; extract the quoted payload when present.
    text = text.strip()
    if "'" in text:
        parts = text.split("'")
        if len(parts) >= 3:
            return parts[1].strip()
    return text


def _text_of(outputs) -> str:
    (req_output,) = outputs
    return (req_output.outputs[0].text or "").strip()


def _tts_params(engine: Omni, cfg_scale: float, cond_prompt: str, tokenizer):
    params = copy.deepcopy(engine.resolve_sampling_params_list(None))
    stage0 = params[0]
    if cfg_scale > 1.0:
        if stage0.extra_args is None:
            stage0.extra_args = {}
        stage0.extra_args.update(
            {
                "cfg_scale": float(cfg_scale),
                "cfg_role": "cond",
                "cfg_pair_id": "s2s-tts-pass",
                "cfg_null_prompt": build_null_prompt(cond_prompt, tokenizer),
            }
        )
    return params


def main():
    args = parse_args()
    engine = Omni(model=args.model, deploy_config=args.deploy_config, trust_remote_code=True)

    tokenizer = None
    if args.cfg_scale > 1.0:
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.audex.checkpoint import ensure_audex_snapshot

        root = ensure_audex_snapshot(args.model, profile="full")
        tokenizer = AutoTokenizer.from_pretrained(str(Path(root) / "checkpoint_folder_full"))

    audio, sr = sf.read(args.audio_file, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Pass 1 — ASR (text-final; never touches the speech decoder).
    transcript = _clean(
        _text_of(
            engine.generate(
                [
                    {
                        "prompt": _asr_prompt(),
                        "multi_modal_data": {"audio": (audio, sr)},
                        "modalities": ["text"],
                    }
                ]
            )
        )
    )
    print(f"[1/3] transcript : {transcript}")
    if not transcript:
        raise SystemExit("ASR pass produced an empty transcript; aborting cascade")

    # Pass 2 — chat answer (text-final).
    answer = _text_of(engine.generate([{"prompt": _chat_prompt(transcript), "modalities": ["text"]}]))
    print(f"[2/3] answer     : {answer}")
    if not answer:
        raise SystemExit("Chat pass produced an empty answer; aborting cascade")

    # Pass 3 — TTS (audio-final; streams through code2wav).
    tts_prompt = build_cond_prompt(answer)
    outputs = engine.generate(
        [{"prompt": tts_prompt, "modalities": ["audio"]}],
        _tts_params(engine, args.cfg_scale, tts_prompt, tokenizer),
    )
    (req_output,) = outputs
    audio_val = req_output.outputs[0].multimodal_output.get("model_outputs")
    if isinstance(audio_val, list):
        pcm = torch.cat([torch.as_tensor(a).float().cpu().reshape(-1) for a in audio_val if a is not None])
    else:
        pcm = torch.as_tensor(audio_val).float().cpu().reshape(-1)
    if pcm.numel() == 0:
        raise SystemExit("TTS pass produced empty audio")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(pcm.numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(str(out_path), arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    print(f"[3/3] speech     : {pcm.numel() / SAMPLE_RATE:.2f}s -> {out_path}")


if __name__ == "__main__":
    main()
