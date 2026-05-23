#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge end-to-end verifier for offline and online speech modes."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import soundfile as sf

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

DEFAULT_MODEL = "FunAudioLLM/Fun-CineForge"
DEFAULT_TEXT = (
    "Every closet on a Carnival cruise ship. To make the numbers work, I needed a lot of cedar, fast and cheap."
)
DEFAULT_REF_TEXT = (
    "A single middle-aged male speaker describes a business or construction "
    "requirement with a practical and matter-of-fact tone. His voice is deep "
    "and slightly gravelly, maintaining a professional and informative demeanor "
    "throughout the segment."
)


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"}


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    if _is_url(path):
        with urlopen(path) as response:
            data = response.read()
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    else:
        wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _to_data_url(path_or_url: str) -> str:
    if _is_url(path_or_url):
        return path_or_url
    suffix = Path(path_or_url).suffix.lower().lstrip(".") or "wav"
    data = Path(path_or_url).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:audio/{suffix};base64,{encoded}"


def _speech_len_from_demo(face_path: str | None) -> int | None:
    if not face_path:
        return None
    path = Path(face_path)
    demo_path = path.parents[1] / "demo.jsonl" if len(path.parents) > 1 else None
    if demo_path is None or not demo_path.exists():
        return None
    utt = path.stem
    with demo_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("utt") == utt and item.get("speech_length") is not None:
                return int(item["speech_length"])
    return None


def _resolve_demo_asset(demo_path: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    base = demo_path.parent.parent if value.startswith("data/") else demo_path.parent
    return str(base / path)


def _load_demo_item(demo_jsonl: str, utt: str) -> dict[str, Any]:
    demo_path = Path(demo_jsonl)
    with demo_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("utt") != utt:
                continue
            roles = {msg["role"]: msg["content"] for msg in item["messages"]}
            return {
                "text": roles["text"],
                "ref_audio": _resolve_demo_asset(demo_path, roles["vocal"]),
                "ref_text": roles.get("clue", DEFAULT_REF_TEXT),
                "face_path": _resolve_demo_asset(demo_path, roles["face"]) if roles.get("face") else None,
                "dialogue": roles.get("dialogue"),
                "speech_type": item.get("type"),
                "speech_len": item.get("speech_length"),
            }
    raise ValueError(f"Could not find utt={utt!r} in {demo_jsonl}")


def _apply_demo_args(args: argparse.Namespace) -> None:
    if not args.demo_jsonl:
        return
    sample = _load_demo_item(args.demo_jsonl, args.utt)
    args.text = sample["text"]
    args.ref_audio = sample["ref_audio"]
    args.ref_text = sample["ref_text"]
    args.face_path = sample["face_path"]
    args.dialogue = sample["dialogue"]
    args.speech_type = sample["speech_type"]
    args.speech_len = int(sample["speech_len"]) if sample["speech_len"] is not None else None


def _build_mm_kwargs(args: argparse.Namespace, ref_sr: int) -> dict[str, Any]:
    mm_processor_kwargs: dict[str, Any] = {
        "prompt_text": args.ref_text,
        "sample_rate": ref_sr,
    }
    if getattr(args, "speech_len", None) is not None:
        mm_processor_kwargs["speech_len"] = args.speech_len
    if getattr(args, "speech_type", None) is not None:
        mm_processor_kwargs["speech_type"] = args.speech_type
    if getattr(args, "dialogue", None) is not None:
        mm_processor_kwargs["dialogue"] = args.dialogue
    if args.face_path:
        from vllm_omni.model_executor.models.funcineforge.config import FunCineForgeConfig
        from vllm_omni.model_executor.models.funcineforge.utils import load_face_embedding

        cfg = FunCineForgeConfig()
        speech_len = args.speech_len or _speech_len_from_demo(args.face_path)
        if speech_len is None:
            speech_len = max(cfg.min_length, min(cfg.max_length, int(len(args.text) * 0.65)))
        mm_processor_kwargs["face_embedding"] = load_face_embedding(
            args.face_path,
            speech_len=speech_len,
            face_size=cfg.face_size,
        )
    return mm_processor_kwargs


def _sampling_params(args: argparse.Namespace) -> list[Any]:
    from vllm import SamplingParams

    talker_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        stop_token_ids=[6562],
        detokenize=False,
    )
    code2wav_params = SamplingParams(max_tokens=args.code2wav_max_tokens, detokenize=False)
    return [talker_params, code2wav_params]


def _concat_audio(audio_val: Any) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        tensors = []
        arrays = []
        for item in audio_val:
            if item is None:
                continue
            if isinstance(item, torch.Tensor):
                tensors.append(item.detach().cpu().float().reshape(-1))
            else:
                arrays.append(np.asarray(item, dtype=np.float32).reshape(-1))
        if tensors:
            return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)
        return np.concatenate(arrays).astype(np.float32, copy=False) if arrays else np.zeros((0,), dtype=np.float32)

    if isinstance(audio_val, torch.Tensor):
        return audio_val.detach().cpu().float().numpy().reshape(-1).astype(np.float32, copy=False)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


def _extract_audio(outputs: list[Any]) -> tuple[np.ndarray, int]:
    from vllm_omni.model_executor.models.funcineforge.config import FunCineForgeConfig

    audio_outputs: list[np.ndarray] = []
    sample_rate = FunCineForgeConfig().sample_rate
    for output in outputs:
        mm = getattr(output, "multimodal_output", None)
        if mm and "audio" in mm:
            sr = mm.get("sr", FunCineForgeConfig().sample_rate)
            if isinstance(sr, list) and sr:
                sr = sr[-1]
            if hasattr(sr, "item"):
                sr = sr.item()
            sample_rate = int(sr)
            audio = _concat_audio(mm["audio"])
            if audio.size:
                audio_outputs.append(audio)
    if audio_outputs:
        cumulative = True
        for prev, cur in zip(audio_outputs, audio_outputs[1:]):
            if cur.size < prev.size or not np.allclose(cur[: prev.size], prev, atol=1e-4):
                cumulative = False
                break
        if cumulative:
            return audio_outputs[-1].astype(np.float32, copy=False), sample_rate
        return np.concatenate(audio_outputs).astype(np.float32, copy=False), sample_rate
    raise RuntimeError("No audio output found in Omni generation result")


def _dump_token_outputs(outputs: list[Any]) -> None:
    for idx, output in enumerate(outputs):
        stage_id = getattr(output, "stage_id", None)
        request_output = getattr(output, "request_output", None)
        completions = getattr(request_output, "outputs", None) if request_output is not None else None
        if not completions:
            continue
        for out_idx, completion in enumerate(completions):
            token_ids = getattr(completion, "cumulative_token_ids", None)
            if token_ids is None:
                token_ids = getattr(completion, "token_ids", None)
            if token_ids is None:
                continue
            ids = list(token_ids)
            print(f"tokens stage={stage_id} output={idx}:{out_idx} len={len(ids)} first={ids[:32]} last={ids[-16:]}")


def _save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sr, format="WAV")
    print(f"saved {path} ({audio.size / sr:.2f}s, sr={sr})")


def _verify_asr(path: str, expected_text: str, threshold: float) -> None:
    try:
        from tests.helpers.media import convert_audio_file_to_text, cosine_similarity_text

        transcript = convert_audio_file_to_text(path).strip()
        score = cosine_similarity_text(transcript.lower(), expected_text.lower())
    except ModuleNotFoundError:
        import re
        from collections import Counter

        import whisper

        def normalize(text: str) -> str:
            text = re.sub(r"[^\w\s]", "", text)
            return re.sub(r"\s+", " ", text).lower().strip()

        def cosine_similarity_text(text1: str, text2: str, n: int = 3) -> float:
            text1 = normalize(text1)
            text2 = normalize(text2)
            grams1 = [text1[i : i + n] for i in range(len(text1) - n + 1)]
            grams2 = [text2[i : i + n] for i in range(len(text2) - n + 1)]
            c1, c2 = Counter(grams1), Counter(grams2)
            keys = set(c1) | set(c2)
            dot = sum(c1[k] * c2[k] for k in keys)
            norm1 = sum(v * v for v in c1.values()) ** 0.5
            norm2 = sum(v * v for v in c2.values()) ** 0.5
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

        transcript = whisper.load_model("small").transcribe(path)["text"].strip()
        score = cosine_similarity_text(transcript, expected_text)
    print(f"ASR transcript: {transcript}")
    print(f"ASR similarity: {score:.3f}")
    if score < threshold:
        raise AssertionError(f"ASR similarity {score:.3f} < threshold {threshold:.3f}")


def _build_video_offline_prompt(args: argparse.Namespace) -> dict[str, Any]:
    """Build offline prompt from video input using build_video_conditions()."""
    from vllm_omni.model_executor.models.funcineforge.video_preprocess import (
        build_video_conditions,
    )

    conditions = build_video_conditions(
        video=args.video,
        start=args.video_start,
        end=args.video_end,
        age=getattr(args, "speaker_age", None),
        gender=getattr(args, "speaker_gender", None),
        speech_type=getattr(args, "speech_type", None),
        work_dir=getattr(args, "preprocess_work_dir", None),
    )
    if args.ref_audio:
        ref_wav, ref_sr = _load_audio(args.ref_audio)
    else:
        ref_wav, ref_sr = conditions.ref_audio

    mm_kwargs: dict[str, Any] = {
        "prompt_text": args.ref_text or "",
        "sample_rate": ref_sr,
        "face_embedding": conditions.face_embedding,
        "speech_len": args.speech_len or conditions.speech_len,
        "speech_type": args.speech_type or conditions.speech_type,
        "dialogue": args.dialogue or conditions.dialogue,
    }
    return {
        "prompt": args.text,
        "multi_modal_data": {"audio": (ref_wav, ref_sr)},
        "modalities": ["audio"],
        "mm_processor_kwargs": mm_kwargs,
    }


def _offline_prompt(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "video", None):
        return _build_video_offline_prompt(args)
    ref_wav, ref_sr = _load_audio(args.ref_audio)
    return {
        "prompt": args.text,
        "multi_modal_data": {"audio": (ref_wav, ref_sr)},
        "modalities": ["audio"],
        "mm_processor_kwargs": _build_mm_kwargs(args, ref_sr),
    }


def run_offline_sync(args: argparse.Namespace) -> tuple[np.ndarray, int]:
    from vllm_omni import Omni

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        trust_remote_code=True,
        seed=args.seed,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        async_chunk=False,
    )
    try:
        outputs = list(omni.generate([_offline_prompt(args)], _sampling_params(args), use_tqdm=False))
        if args.dump_tokens:
            _dump_token_outputs(outputs)
        return _extract_audio(outputs)
    finally:
        omni.close()


async def run_offline_async(args: argparse.Namespace) -> tuple[np.ndarray, int]:
    from vllm_omni import AsyncOmni

    omni = AsyncOmni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        trust_remote_code=True,
        seed=args.seed,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        async_chunk=True,
    )
    outputs = []
    try:
        async for out in omni.generate(
            _offline_prompt(args),
            request_id="funcineforge-example",
            sampling_params_list=_sampling_params(args),
            output_modalities=["audio"],
        ):
            outputs.append(out)
        if args.dump_tokens:
            _dump_token_outputs(outputs)
        return _extract_audio(outputs)
    finally:
        omni.shutdown()


def run_online(args: argparse.Namespace) -> tuple[np.ndarray, int]:
    import httpx

    body: dict[str, Any] = {
        "model": args.model,
        "input": args.text,
        "stream": args.async_chunk,
        "response_format": "wav",
    }

    if getattr(args, "video", None):
        if _is_url(args.video):
            body["video"] = args.video
        else:
            body["video"] = _to_data_url(args.video)
        if getattr(args, "video_start", None) is not None:
            body["video_start"] = args.video_start
        if getattr(args, "video_end", None) is not None:
            body["video_end"] = args.video_end
        if getattr(args, "speaker_age", None):
            body["speaker_age"] = args.speaker_age
        if getattr(args, "speaker_gender", None):
            body["speaker_gender"] = args.speaker_gender
        if getattr(args, "preprocess_work_dir", None):
            body["preprocess_work_dir"] = args.preprocess_work_dir
    if args.ref_audio:
        body["ref_audio"] = _to_data_url(args.ref_audio)
    if args.ref_text:
        body["ref_text"] = args.ref_text
    if args.face_path:
        body["face_path"] = args.face_path
    if getattr(args, "speech_len", None) is not None:
        body["speech_len"] = args.speech_len
    if getattr(args, "speech_type", None) is not None:
        body["speech_type"] = args.speech_type
    if getattr(args, "dialogue", None) is not None:
        body["dialogue"] = args.dialogue
    url = f"{args.api_base.rstrip('/')}/v1/audio/speech"
    timeout = httpx.Timeout(args.http_timeout)
    if args.async_chunk:
        with httpx.stream("POST", url, json=body, timeout=timeout) as response:
            response.raise_for_status()
            audio_bytes = b"".join(response.iter_bytes())
    else:
        response = httpx.post(url, json=body, timeout=timeout)
        response.raise_for_status()
        audio_bytes = response.content

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--async-chunk", action="store_true", help="Use AsyncOmni offline or streaming online request")
    parser.add_argument("--model", "--model-path", default=DEFAULT_MODEL, help="HF model ID or local model directory")
    parser.add_argument("--stage-configs-path", default="vllm_omni/deploy/funcineforge.yaml")
    parser.add_argument("--api-base", default="http://127.0.0.1:8091")
    parser.add_argument("--demo-jsonl", help="Official FunCineForge demo.jsonl to build a consistent request")
    parser.add_argument("--utt", default="ref_en_monologue_1", help="Utterance ID to load from --demo-jsonl")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--video", help="Input video for end-to-end dubbing (path or URL)")
    parser.add_argument("--video-start", type=float, help="Video segment start time in seconds")
    parser.add_argument("--video-end", type=float, help="Video segment end time in seconds")
    parser.add_argument("--speaker-age", help="Speaker age tag for video preprocessing")
    parser.add_argument("--speaker-gender", help="Speaker gender tag for video preprocessing")
    parser.add_argument("--preprocess-work-dir", help="Scratch dir for video preprocessing artifacts")
    parser.add_argument("--face-path")
    parser.add_argument("--speech-len", type=int)
    parser.add_argument("--speech-type")
    parser.add_argument("--output", default="outputs/funcineforge.wav")
    parser.add_argument("--verify-asr", action="store_true")
    parser.add_argument("--asr-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--log-stats", action="store_true")
    parser.add_argument("--dump-tokens", action="store_true", help="Print generated token summaries for offline debug")
    parser.add_argument("--min-tokens", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--code2wav-max-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dialogue = None
    _apply_demo_args(args)
    if not args.ref_audio and not getattr(args, "video", None):
        raise ValueError("--ref-audio or --video is required unless --demo-jsonl supplies it")
    if args.mode == "offline" and args.async_chunk:
        audio, sr = asyncio.run(run_offline_async(args))
    elif args.mode == "offline":
        audio, sr = run_offline_sync(args)
    else:
        audio, sr = run_online(args)

    if audio.size == 0:
        raise RuntimeError("Generated audio is empty")
    _save_wav(args.output, audio, sr)
    if args.verify_asr:
        _verify_asr(args.output, args.text, args.asr_threshold)


if __name__ == "__main__":
    main()
