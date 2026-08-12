# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline speech-to-speech with NVIDIA-NemotronLabs-VoiceChat-11B.

Feeds a 16 kHz user utterance (plus a spoken-style system prompt) through the
3-stage nemotron_voicechat pipeline and writes the agent's reply as text and a
22.05 kHz WAV.

Usage:
    python end2end.py \
        --checkpoint /path/to/NVIDIA-NemotronLabs-VoiceChat-11B \
        --wav /path/to/user_question_16k.wav \
        --output-dir results/nemotron_voicechat

The system prompt defaults to the NeMo reference default. The text tokenizer
resolves from the ``nvidia/NVIDIA-Nemotron-Nano-9B-v2`` HF id (or the
``NEMOTRON_VOICECHAT_LLM_PATH`` env var for air-gapped runs).

Sizing the input WAV: the timeline is frame-locked, so the reply budget IS the
input duration — the acoustic channel trails the text channel, and a reply
that has not finished when the frames run out is truncated SILENTLY (no error;
ASR of the WAV will just disagree with the emitted text). Leave generous
trailing silence after the question (a question ending at ~4.5 s truncated in
an 8 s WAV but completed cleanly in a 16 s one). The script warns when the
text channel is still emitting non-PAD tokens near the last frame.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

# Warn if the text channel is still speaking within this many final frames
# (12 frames = ~1 s) — the frame-locked reply likely got cut off.
TRUNCATION_GUARD_FRAMES = 12

# The NeMo reference offline script's default system prompt.
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)


def load_wav_16k_mono(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        from vllm.multimodal.audio import resample_audio_scipy

        wav = resample_audio_scipy(wav, orig_sr=sr, target_sr=16000)
    return wav.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="NVIDIA-NemotronLabs-VoiceChat-11B directory")
    parser.add_argument("--wav", required=True, help="user speech (any sr; resampled to 16 kHz mono)")
    parser.add_argument("--output-dir", default="results/nemotron_voicechat")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--stage-configs-path",
        default=None,
        help=(
            "Deploy yaml override. Pass vllm_omni/deploy/nemotron_labs_voicechat_streaming.yaml "
            "for the async-chunk streaming pipeline (stages overlap; audio starts decoding "
            "before the thinker finishes). Default: the bundled non-streaming yaml (bit-exact "
            "parity path)."
        ),
    )
    args = parser.parse_args()

    if not Path(args.wav).is_file():
        raise FileNotFoundError(f"Input WAV not found: {args.wav}")
    ckpt = Path(args.checkpoint)
    raw_cfg = json.loads((ckpt / "config.json").read_text())
    stt_cfg = raw_cfg["model"]["stt"]["model"]

    from transformers import AutoTokenizer
    from vllm.sampling_params import SamplingParams

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
        compute_acoustic_frame_count,
    )

    tok_ref = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH") or stt_cfg.get(
        "pretrained_llm", "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_ref, trust_remote_code=False)
    bos_id = tokenizer.convert_tokens_to_ids(stt_cfg.get("bos_token", "<s>"))
    eos_id = tokenizer.convert_tokens_to_ids(stt_cfg.get("eos_token", "</s>"))
    pad_id = tokenizer.convert_tokens_to_ids(stt_cfg.get("pad_token", "<SPECIAL_12>"))
    # NeMo encode_system_prompt: [bos] + tokens + [eos].
    prompt_ids = [bos_id] + tokenizer.encode(args.system_prompt, add_special_tokens=False) + [eos_id]

    wav = load_wav_16k_mono(args.wav)
    n_frames = compute_acoustic_frame_count(stt_cfg, int(wav.shape[0]))
    print(f"input: {wav.shape[0]} samples @16kHz -> {n_frames} acoustic frames; prompt {len(prompt_ids)} tokens")

    inputs = {
        # vLLM prompt = system prompt + one placeholder position (acoustic
        # frame 0); the thinker replaces every position with fused embeddings.
        "prompt_token_ids": prompt_ids + [pad_id],
        "additional_information": {
            # Plain float list: raw ndarrays do not survive the engine-core
            # message serialization.
            "nvc_audio": wav.astype(np.float32).tolist(),
            "nvc_sr": 16000,
            "nvc_prompt_token_ids": prompt_ids,
            "nvc_expected_frames": n_frames,
        },
    }
    sampling_params_list = [
        # thinker: greedy, frame-locked generation length. NO min_tokens: the
        # tokenizer's eos_token is <SPECIAL_12> — the frame-locked PAD/silence
        # token — and vLLM's min_tokens masks "EOS" to -inf, which would forbid
        # silence and force the agent to babble through the whole utterance.
        SamplingParams(temperature=0.0, max_tokens=n_frames, ignore_eos=True, seed=0),
        # talker: placeholder loop, stops on the stage's stop token. 16383:
        # the 1-token placeholder prompt + max_tokens must fit the stage's
        # max_model_len (16384).
        SamplingParams(temperature=0.0, max_tokens=16383, stop_token_ids=[1], detokenize=False, seed=0),
        # code2wav: single decode step.
        SamplingParams(temperature=0.0, max_tokens=1, detokenize=False, seed=0),
    ]

    omni_kwargs: dict = {}
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = args.stage_configs_path
    omni = Omni(model=str(ckpt), trust_remote_code=True, **omni_kwargs)
    outputs = omni.generate(inputs, sampling_params_list)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.wav).stem

    agent_text = None
    audio = None
    sr = 22050
    for stage_outputs in outputs:
        # One entry per final stage (text from the thinker, audio from code2wav);
        # mirrors the qwen2_5_omni offline example's collection loop.
        output = getattr(stage_outputs, "request_output", stage_outputs)
        final_type = getattr(stage_outputs, "final_output_type", None)
        completions = getattr(output, "outputs", None) or []
        if not completions:
            continue
        completion = completions[0]
        if final_type in (None, "text"):
            text = getattr(completion, "text", None)
            if text:
                agent_text = text
            token_ids = getattr(completion, "token_ids", None)
            if token_ids:
                (out_dir / f"{stem}_text_tokens.json").write_text(json.dumps(list(map(int, token_ids))))
                # Frame lock: the reply budget IS the input length. If the text
                # channel is still mid-sentence when the frames run out, the
                # spoken answer is silently truncated — warn instead.
                tail = list(map(int, token_ids))[-TRUNCATION_GUARD_FRAMES:]
                if any(t != pad_id for t in tail):
                    print(
                        f"WARNING: the text channel emits non-PAD tokens within the last "
                        f"{TRUNCATION_GUARD_FRAMES} frames ({TRUNCATION_GUARD_FRAMES * 80} ms) — the reply "
                        "likely ran out of acoustic frames and is cut off mid-sentence. "
                        "Pad the input WAV with more trailing silence and rerun."
                    )
            # Best-effort: dump the function-channel debug timeline if the
            # engine surfaced the request info on this output.
            info = getattr(output, "additional_information", None)
            func_timeline = info.get("nvc_function_tokens") if isinstance(info, dict) else None
            if func_timeline is not None:
                values = func_timeline.tolist() if hasattr(func_timeline, "tolist") else list(func_timeline)
                (out_dir / f"{stem}_function_tokens.json").write_text(json.dumps([int(v) for v in values]))
                print(f"function-channel timeline: {len(values)} entries")
        if final_type in (None, "audio"):
            # multimodal_output is a MultimodalPayload (Mapping) — duck-type it.
            mm = getattr(completion, "multimodal_output", None)
            get = getattr(mm, "get", None)
            if not callable(get):
                continue
            candidate = None
            for key in ("audio", "model_outputs", "wav", "waveform"):
                value = get(key)
                if value is not None:
                    candidate = value
                    break
            if isinstance(candidate, list):
                candidate = candidate[0] if candidate else None
            if candidate is not None:
                arr = np.asarray(
                    candidate.float().cpu() if hasattr(candidate, "cpu") else candidate, dtype=np.float32
                ).reshape(-1)
                if arr.size > 0:
                    audio = arr
                sr_val = get("sr")
                if sr_val is not None:
                    sr_item = sr_val[0] if isinstance(sr_val, list) else sr_val
                    sr = int(sr_item.item() if hasattr(sr_item, "item") else sr_item)

    if agent_text is not None:
        # The frame-locked text channel emits PAD tokens on silent frames;
        # strip them (NeMo's post-inference detokenizer does the same).
        pad_token = stt_cfg.get("pad_token", "<SPECIAL_12>")
        agent_text = " ".join(agent_text.replace(pad_token, " ").split())
        text_path = out_dir / f"{stem}_output.txt"
        text_path.write_text(agent_text + "\n")
        print(f"agent text -> {text_path}\n{agent_text}")
    else:
        print("WARNING: no agent text in outputs")

    if audio is not None and audio.size > 0:
        wav_path = out_dir / f"{stem}_output.wav"
        sf.write(wav_path, audio, sr)
        print(f"agent audio -> {wav_path} ({audio.size / sr:.2f}s @ {sr} Hz)")
    else:
        raise RuntimeError("No audio produced by the code2wav stage.")


if __name__ == "__main__":
    main()
