#!/usr/bin/env python3
"""Benchmark MiniCPM-o 4.5 text/image/video → audio generation.

Measures request latency, generated audio duration, and RTF (real-time factor)
across two modes: non_async and HuggingFace reference.

No Modal dependency — runs on any GPU machine with vllm-omni installed.

Usage:
    python benchmarks/minicpmo4_5/bench_minicpmo4_5.py --model-path /path/to/model --mode all
    python benchmarks/minicpmo4_5/bench_minicpmo4_5.py --model-path /path/to/model --mode non_async --num-repeats 3
    python benchmarks/minicpmo4_5/bench_minicpmo4_5.py --model-path /path/to/model --mode hf --modalities text
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_CONFIG = str(REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "minicpmo.yaml")
Mode = Literal["non_async", "hf", "all"]
Modality = Literal["text", "text+image", "text+video"]

AUDIO_OUTPUT_SYSTEM_PROMPT = (
    "When audio output is requested, reply with speech only and follow any requested length constraints."
)

TEXT_PROMPT = (
    "Please read this single long sentence aloud exactly once without shortening it: "
    "vLLM Omni is running a benchmark for MiniCPM speech generation, and this sentence intentionally "
    "includes enough detail about streaming text to audio generation, multimodal reasoning, "
    "stage connectors, careful benchmarking, and stable speech synthesis behavior to last well "
    "over ten seconds when spoken at a natural pace."
)

IMAGE_PROMPT = (
    "Describe the image in one single detailed spoken sentence of at least sixty words, "
    "mentioning every visible shape, its color, its approximate size, its position "
    "relative to the other shapes, the plain background, and the overall layout, and keep "
    "the answer natural but long enough to last more than ten seconds."
)

VIDEO_PROMPT = (
    "Describe the video in one single detailed spoken sentence of at least sixty words, "
    "covering the moving objects, their colors, their approximate sizes, the direction and "
    "pattern of their motion over time, the dark background, and the overall scene, and "
    "keep the answer natural but long enough to last more than ten seconds."
)

MODALITY_PROMPT: dict[Modality, str] = {
    "text": TEXT_PROMPT,
    "text+image": IMAGE_PROMPT,
    "text+video": VIDEO_PROMPT,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    mode: str
    modality: str
    run_index: int
    success: bool
    latency_ms: float
    audio_duration_s: float
    rtf: float
    num_audio_samples: int
    sample_rate: int
    output_text_tokens: int = 0
    output_text_chars: int = 0
    output_tokens_by_stage: dict[str, int] = field(default_factory=dict)
    error: str = ""
    error_type: str = ""
    error_traceback: str = ""


@dataclass
class BenchmarkResult:
    mode: str
    num_requests: int
    completed: int
    failed: int
    duration_s: float
    mean_latency_ms: float
    median_latency_ms: float
    std_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_rtf: float
    median_rtf: float
    std_rtf: float
    mean_output_text_tokens: float
    per_request: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic media generators (self-contained, no test dependency)
# ---------------------------------------------------------------------------


def _make_image(seed: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    random.seed(seed)
    width, height = 224, 224
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 8)):
        sq = random.randint(min(width, height) // 8, min(width, height) // 4)
        x = random.randint(0, width - sq - 1)
        y = random.randint(0, height - sq - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle([x, y, x + sq, y + sq], fill=color, outline=(0, 0, 0), width=random.randint(1, 5))
    return np.asarray(img, dtype=np.uint8).copy()


def _make_pil_image(seed: int) -> Any:
    from PIL import Image

    return Image.fromarray(_make_image(seed))


def _make_video(seed: int, num_frames: int = 30) -> np.ndarray:
    import math

    import cv2

    random.seed(seed)
    width, height = 64, 64
    balls = []
    for _ in range(random.randint(3, 8)):
        radius = min(width, height) // 8
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)
        speed = random.uniform(3.0, 8.0)
        angle = random.uniform(0, 2 * math.pi)
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        balls.append(
            {
                "x": float(x),
                "y": float(y),
                "vx": speed * math.cos(angle),
                "vy": speed * math.sin(angle),
                "radius": radius,
                "color_bgr": color,
            }
        )

    frames = []
    for _ in range(num_frames):
        frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
        for b in balls:
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            r = int(b["radius"])
            if b["x"] - r <= 0 or b["x"] + r >= width:
                b["vx"] = -b["vx"]
                b["x"] = max(r, min(width - r, b["x"]))
            if b["y"] - r <= 0 or b["y"] + r >= height:
                b["vy"] = -b["vy"]
                b["y"] = max(r, min(height - r, b["y"]))
            cv2.circle(frame_bgr, (int(b["x"]), int(b["y"])), r, b["color_bgr"], -1)
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _normalize_token_ids(tokenized: Any) -> list[int]:
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(t) for t in tokenized]


def _build_tts_prompt(model_path: str, text: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [
        {"role": "system", "content": AUDIO_OUTPUT_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        use_tts_template=True,
        enable_thinking=False,
    )
    return {
        "prompt_token_ids": _normalize_token_ids(tokenized),
        "modalities": ["audio"],
    }


def _build_multimodal_audio_prompt(
    text: str,
    image: np.ndarray | None = None,
    video: np.ndarray | None = None,
) -> dict[str, Any]:
    user_content = ""

    multi_modal_data: dict[str, Any] = {}
    if image is not None:
        user_content += "(<image>./</image>)"
        multi_modal_data["image"] = image
    if video is not None:
        user_content += "(<video>./</video>)"
        multi_modal_data["video"] = video

    user_content += text

    assistant_prefix = "<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>"

    prompt: dict[str, Any] = {
        "prompt": (
            f"<|im_start|>system\n{AUDIO_OUTPUT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"{assistant_prefix}"
        ),
        "modalities": ["audio"],
    }
    if multi_modal_data:
        prompt["multi_modal_data"] = multi_modal_data
    return prompt


def _build_prompt(
    model_path: str,
    modality: Modality,
    *,
    media_seed: int,
) -> dict[str, Any]:
    text = MODALITY_PROMPT[modality]
    if modality == "text":
        return _build_tts_prompt(model_path, text)
    elif modality == "text+image":
        image = _make_image(media_seed)
        return _build_multimodal_audio_prompt(text, image=image)
    elif modality == "text+video":
        video = _make_video(media_seed)
        return _build_multimodal_audio_prompt(text, video=video)
    else:
        raise ValueError(f"Unknown modality: {modality}")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _format_exception(exc: BaseException) -> tuple[str, str, str]:
    exc_type = type(exc).__name__
    exc_repr = repr(exc)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=20))
    return exc_type, exc_repr, tb


def _cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import subprocess

        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpus = []
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", maxsplit=2)]
            if len(parts) == 3:
                gpus.append({"index": parts[0], "name": parts[1], "uuid": parts[2]})
        info["nvidia_smi_available"] = True
        info["device_count"] = len(gpus)
        info["devices"] = gpus
    except Exception as exc:
        info["nvidia_smi_available"] = False
        info["error"] = f"{type(exc).__name__}: {exc!r}"
    return info


def _print_cuda_info() -> None:
    info = _cuda_info()
    devices = info.get("devices") or []
    print(f"CUDA_VISIBLE_DEVICES: {info.get('cuda_visible_devices') or '<unset>'}")
    print(f"CUDA devices: nvidia_smi_available={info.get('nvidia_smi_available')} count={info.get('device_count')}")
    for device in devices:
        print(f"  gpu:{device['index']}: {device['name']} uuid={device['uuid']}")
    if "error" in info:
        print(f"CUDA diagnostics error: {info['error']}")


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def _extract_audio_from_outputs(outputs: list[Any]) -> tuple[np.ndarray, int]:
    import torch

    for stage_output in outputs:
        final_output_type = getattr(stage_output, "final_output_type", None)
        request_output = getattr(stage_output, "request_output", None)
        if request_output is None or final_output_type != "audio":
            continue

        multimodal_output = getattr(request_output, "multimodal_output", None)
        if not multimodal_output and getattr(request_output, "outputs", None):
            multimodal_output = getattr(request_output.outputs[0], "multimodal_output", None)
        if not isinstance(multimodal_output, dict):
            continue

        audio_obj = multimodal_output.get("audio")
        sr_obj = multimodal_output.get("sr", 24000)
        if isinstance(sr_obj, list) and sr_obj:
            sr_obj = sr_obj[-1]
        if hasattr(sr_obj, "item"):
            sr_obj = sr_obj.item()
        sample_rate = int(sr_obj)

        if isinstance(audio_obj, list):
            tensor_parts = [p for p in audio_obj if isinstance(p, torch.Tensor)]
            if tensor_parts:
                audio_tensor = torch.cat(tensor_parts, dim=-1)
            elif audio_obj and isinstance(audio_obj[0], np.ndarray):
                audio_tensor = torch.from_numpy(np.concatenate(audio_obj, axis=-1))
        elif isinstance(audio_obj, torch.Tensor):
            audio_tensor = audio_obj
        elif isinstance(audio_obj, np.ndarray):
            audio_tensor = torch.from_numpy(audio_obj)
        else:
            continue

        audio_np = audio_tensor.detach().cpu().float().numpy().reshape(-1)
        return audio_np, sample_rate

    raise RuntimeError("No audio output found in stage outputs.")


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        tokenized = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokenized = tokenizer.encode(text)
    return len(_normalize_token_ids(tokenized))


def _request_output_token_count(request_output: Any) -> int:
    count = 0
    for output in getattr(request_output, "outputs", []) or []:
        token_ids = getattr(output, "token_ids", None)
        if token_ids is not None:
            count += len(token_ids)
    return count


def _collect_vllm_output_metadata(
    outputs: list[Any],
    tokenizer: Any,
) -> tuple[int, int, dict[str, int]]:
    text_parts: list[str] = []
    text_token_count = 0
    stage_token_counts: dict[str, int] = {}

    for stage_output in outputs:
        request_output = getattr(stage_output, "request_output", None)
        if request_output is None:
            continue
        stage_id = getattr(stage_output, "stage_id", "unknown")
        final_output_type = getattr(stage_output, "final_output_type", "unknown")
        key = f"stage_{stage_id}_{final_output_type}"
        stage_token_counts[key] = _request_output_token_count(request_output)

        if final_output_type != "text":
            continue
        for output in getattr(request_output, "outputs", []) or []:
            text = getattr(output, "text", "")
            if text:
                text_parts.append(str(text))
            token_ids = getattr(output, "token_ids", None)
            if token_ids is not None:
                text_token_count += len(token_ids)

    text_output = "".join(text_parts)
    if text_output and text_token_count == 0:
        text_token_count = _count_text_tokens(tokenizer, text_output)
    return text_token_count, len(text_output), stage_token_counts


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _iter_omni_outputs(omni: Any, prompt: dict[str, Any]) -> Any:
    sampling_params_list = omni.resolve_sampling_params_list(omni.default_sampling_params_list)
    return omni._run_generation(prompt, sampling_params_list, use_tqdm=False)


def run_vllm_omni_bench(
    model_path: str,
    stage_config_path: str,
    *,
    mode_label: str,
    modalities: list[Modality],
    num_repeats: int,
    seed: int,
) -> list[RequestResult]:
    from transformers import AutoTokenizer

    from vllm_omni.entrypoints.omni import Omni

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    log_stats = os.environ.get("VLLM_OMNI_BENCH_LOG_STATS", "0") == "1"
    omni = Omni(
        model=model_path,
        stage_configs_path=stage_config_path,
        trust_remote_code=True,
        log_stats=log_stats,
        stage_init_timeout=20 * 60,
        init_timeout=30 * 60,
    )

    results: list[RequestResult] = []
    media_seed_base = seed

    try:
        for run_idx in range(num_repeats):
            for modality in modalities:
                media_seed = media_seed_base + run_idx
                prompt = _build_prompt(model_path, modality, media_seed=media_seed)

                t0 = time.perf_counter()
                last_ts = t0
                all_outputs: list[Any] = []

                try:
                    for output in _iter_omni_outputs(omni, prompt):
                        ts = time.perf_counter()
                        last_ts = ts
                        all_outputs.append(output)

                    latency = last_ts - t0
                    audio_np, sample_rate = _extract_audio_from_outputs(all_outputs)
                    audio_duration_s = float(audio_np.size / sample_rate)
                    rtf = latency / audio_duration_s if audio_duration_s > 0 else float("inf")
                    output_text_tokens, output_text_chars, output_tokens_by_stage = _collect_vllm_output_metadata(
                        all_outputs, tokenizer
                    )

                    results.append(
                        RequestResult(
                            mode=mode_label,
                            modality=modality,
                            run_index=run_idx,
                            success=True,
                            latency_ms=latency * 1000,
                            audio_duration_s=audio_duration_s,
                            rtf=rtf,
                            num_audio_samples=int(audio_np.size),
                            sample_rate=sample_rate,
                            output_text_tokens=output_text_tokens,
                            output_text_chars=output_text_chars,
                            output_tokens_by_stage=output_tokens_by_stage,
                        )
                    )
                    print(
                        f"  [{mode_label}] {modality} run={run_idx} "
                        f"latency={latency * 1000:.0f}ms rtf={rtf:.2f} "
                        f"audio={audio_duration_s:.1f}s text_tokens={output_text_tokens}"
                    )
                except Exception as e:
                    error_type, error_repr, error_tb = _format_exception(e)
                    results.append(
                        RequestResult(
                            mode=mode_label,
                            modality=modality,
                            run_index=run_idx,
                            success=False,
                            latency_ms=0.0,
                            audio_duration_s=0.0,
                            rtf=0.0,
                            num_audio_samples=0,
                            sample_rate=0,
                            error=error_repr,
                            error_type=error_type,
                            error_traceback=error_tb,
                        )
                    )
                    print(f"  [{mode_label}] {modality} run={run_idx} ERROR: {error_type}: {error_repr}")
    finally:
        omni.close()

    return results


def _call_with_supported_kwargs(func: Any, /, **kwargs: Any) -> Any:
    import inspect

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_var_kwargs:
        return func(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return func(**filtered)


def _coerce_audio_chunk(chunk: Any) -> np.ndarray:
    import torch

    if isinstance(chunk, torch.Tensor):
        audio_np = chunk.detach().cpu().float().numpy()
    else:
        audio_np = np.asarray(chunk, dtype=np.float32)

    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim == 0:
        return audio_np.reshape(1)
    if audio_np.ndim == 1:
        return audio_np
    if audio_np.ndim == 2:
        if audio_np.shape[0] == 1:
            return audio_np[0]
        if audio_np.shape[1] == 1:
            return audio_np[:, 0]
        return audio_np.mean(axis=0)
    raise ValueError(f"Unsupported audio chunk shape: {audio_np.shape}")


def _patch_dynamic_cache_seen_tokens() -> None:
    """Patch DynamicCache to add key_cache/value_cache/seen_tokens attrs.

    Needed for some transformers versions used by MiniCPM-o 4.5 HF inference.
    """
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "key_cache"):

        def _key_cache(self: Any) -> list[Any]:
            return [layer.keys for layer in getattr(self, "layers", []) if getattr(layer, "keys", None) is not None]

        DynamicCache.key_cache = property(_key_cache)  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "value_cache"):

        def _value_cache(self: Any) -> list[Any]:
            return [layer.values for layer in getattr(self, "layers", []) if getattr(layer, "values", None) is not None]

        DynamicCache.value_cache = property(_value_cache)  # type: ignore[attr-defined]

    if not hasattr(DynamicCache, "seen_tokens"):

        def _seen_tokens(self: Any) -> int:
            get_seq_length = getattr(self, "get_seq_length", None)
            if callable(get_seq_length):
                return int(get_seq_length())
            key_cache = getattr(self, "key_cache", None)
            if key_cache:
                return int(key_cache[0].shape[-2])
            return 0

        DynamicCache.seen_tokens = property(_seen_tokens)  # type: ignore[attr-defined]


def run_hf_bench(
    model_path: str,
    *,
    modalities: list[Modality],
    num_repeats: int,
    seed: int,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> list[RequestResult]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _patch_dynamic_cache_seen_tokens()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=False,
        init_tts=True,
    )
    model.eval().cuda()
    model.init_tts()

    torch.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    results: list[RequestResult] = []
    media_seed_base = seed

    try:
        for run_idx in range(num_repeats):
            for modality in modalities:
                media_seed = media_seed_base + run_idx

                if hasattr(model, "reset_session"):
                    model.reset_session()
                if hasattr(model, "init_token2wav_cache"):
                    try:
                        model.init_token2wav_cache(None)
                    except Exception:
                        model.init_token2wav_cache(np.zeros((16000,), dtype=np.float32))

                if hasattr(model, "get_sys_prompt"):
                    sys_msg = model.get_sys_prompt(ref_audio=None, mode="omni", language="en")
                    content = sys_msg.get("content")
                    if isinstance(content, list) and content and isinstance(content[-1], str):
                        content[-1] = f"{content[-1]} {AUDIO_OUTPUT_SYSTEM_PROMPT}".strip()
                else:
                    sys_msg = {
                        "role": "system",
                        "content": [
                            "You are MiniCPM, a helpful multimodal assistant. "
                            "When audio output is requested, reply with speech only."
                        ],
                    }

                text = MODALITY_PROMPT[modality]

                if modality == "text":
                    user_content = [text]
                elif modality == "text+image":
                    user_content = [_make_pil_image(media_seed), text]
                elif modality == "text+video":
                    from PIL import Image

                    video_np = _make_video(media_seed)
                    pil_frames = [Image.fromarray(frame) for frame in video_np]
                    user_content = [*pil_frames, text]
                else:
                    raise ValueError(f"Unknown modality: {modality}")

                user_msg = {"role": "user", "content": user_content}

                try:
                    t0 = time.perf_counter()
                    session_id = uuid.uuid4().hex
                    _call_with_supported_kwargs(
                        model.streaming_prefill,
                        session_id=session_id,
                        msgs=[sys_msg],
                    )
                    _call_with_supported_kwargs(
                        model.streaming_prefill,
                        session_id=session_id,
                        msgs=[user_msg],
                        is_last_chunk=True,
                    )

                    last_ts = time.perf_counter()
                    audio_chunks: list[np.ndarray] = []
                    text_output = ""

                    iter_gen = _call_with_supported_kwargs(
                        model.streaming_generate,
                        session_id=session_id,
                        generate_audio=True,
                        use_tts_template=True,
                        enable_thinking=False,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )

                    for chunk in iter_gen:
                        last_ts = time.perf_counter()
                        if not isinstance(chunk, tuple) or len(chunk) != 2:
                            raise RuntimeError(f"Unexpected chunk shape: {type(chunk).__name__}")
                        wav_chunk_raw, text_chunk_raw = chunk
                        if wav_chunk_raw is not None:
                            audio_chunks.append(_coerce_audio_chunk(wav_chunk_raw))
                        if text_chunk_raw:
                            text_output += str(text_chunk_raw)

                    if not audio_chunks:
                        raise RuntimeError("HF streaming_generate returned no audio chunks.")

                    latency = last_ts - t0
                    audio_np = np.concatenate(audio_chunks, axis=0).astype(np.float32)
                    sample_rate = 24000
                    audio_duration_s = float(audio_np.size / sample_rate)
                    rtf = latency / audio_duration_s if audio_duration_s > 0 else float("inf")
                    output_text_tokens = _count_text_tokens(tokenizer, text_output)

                    results.append(
                        RequestResult(
                            mode="hf",
                            modality=modality,
                            run_index=run_idx,
                            success=True,
                            latency_ms=latency * 1000,
                            audio_duration_s=audio_duration_s,
                            rtf=rtf,
                            num_audio_samples=int(audio_np.size),
                            sample_rate=sample_rate,
                            output_text_tokens=output_text_tokens,
                            output_text_chars=len(text_output),
                        )
                    )
                    print(
                        f"  [hf] {modality} run={run_idx} "
                        f"latency={latency * 1000:.0f}ms rtf={rtf:.2f} "
                        f"audio={audio_duration_s:.1f}s text_tokens={output_text_tokens}"
                    )
                except Exception as e:
                    error_type, error_repr, error_tb = _format_exception(e)
                    results.append(
                        RequestResult(
                            mode="hf",
                            modality=modality,
                            run_index=run_idx,
                            success=False,
                            latency_ms=0.0,
                            audio_duration_s=0.0,
                            rtf=0.0,
                            num_audio_samples=0,
                            sample_rate=0,
                            error=error_repr,
                            error_type=error_type,
                            error_traceback=error_tb,
                        )
                    )
                    print(f"  [hf] {modality} run={run_idx} ERROR: {error_type}: {error_repr}")
    finally:
        del model

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


def aggregate(results: list[RequestResult], mode_label: str) -> BenchmarkResult:
    """Aggregate per-request results into summary statistics."""
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not succeeded:
        return BenchmarkResult(
            mode=mode_label,
            num_requests=len(results),
            completed=0,
            failed=len(failed),
            duration_s=0.0,
            mean_latency_ms=0.0,
            median_latency_ms=0.0,
            std_latency_ms=0.0,
            p90_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            mean_rtf=0.0,
            median_rtf=0.0,
            std_rtf=0.0,
            mean_output_text_tokens=0.0,
            per_request=[_result_to_dict(r) for r in results],
        )

    latencies = [r.latency_ms for r in succeeded]
    rtfs = [r.rtf for r in succeeded]
    output_text_tokens = [r.output_text_tokens for r in succeeded]

    return BenchmarkResult(
        mode=mode_label,
        num_requests=len(results),
        completed=len(succeeded),
        failed=len(failed),
        duration_s=sum(latencies) / 1000.0,
        mean_latency_ms=float(np.mean(latencies)),
        median_latency_ms=float(np.median(latencies)),
        std_latency_ms=float(np.std(latencies)),
        p90_latency_ms=_percentile(latencies, 90),
        p95_latency_ms=_percentile(latencies, 95),
        p99_latency_ms=_percentile(latencies, 99),
        mean_rtf=float(np.mean(rtfs)),
        median_rtf=float(np.median(rtfs)),
        std_rtf=float(np.std(rtfs)),
        mean_output_text_tokens=float(np.mean(output_text_tokens)),
        per_request=[_result_to_dict(r) for r in results],
    )


def _result_to_dict(r: RequestResult) -> dict[str, Any]:
    d = {
        "mode": r.mode,
        "modality": r.modality,
        "run_index": r.run_index,
        "success": r.success,
        "latency_ms": round(r.latency_ms, 1),
        "audio_duration_s": round(r.audio_duration_s, 3),
        "rtf": round(r.rtf, 3),
        "output_text_tokens": r.output_text_tokens,
        "output_text_chars": r.output_text_chars,
    }
    if r.output_tokens_by_stage:
        d["output_tokens_by_stage"] = r.output_tokens_by_stage
    if r.error:
        d["error"] = r.error
    if r.error_type:
        d["error_type"] = r.error_type
    if r.error_traceback:
        d["error_traceback"] = r.error_traceback
    return d


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_separator() -> None:
    print("-" * 78)


def print_summary_table(all_results: list[BenchmarkResult]) -> None:
    _print_separator()
    print(f"{'Mode':<16} {'#OK':>4} {'#FAIL':>5} {'Latency(ms)':>12} {'RTF':>7} {'TxtTok':>8}")
    _print_separator()
    for br in all_results:
        print(
            f"{br.mode:<16} {br.completed:>4} {br.failed:>5} "
            f"{br.mean_latency_ms:>12.0f} {br.mean_rtf:>7.2f} "
            f"{br.mean_output_text_tokens:>8.1f}"
        )
    _print_separator()

    # Per-modality breakdown
    if all_results:
        print("\nPer-modality breakdown:")
        print(f"{'Mode':<16} {'Modality':<14} {'Latency(ms)':>12} {'RTF':>7} {'TxtTok':>8}")
        _print_separator()
        for br in all_results:
            by_modality: dict[str, list[dict[str, Any]]] = {}
            for req in br.per_request:
                if req["success"]:
                    by_modality.setdefault(req["modality"], []).append(req)
            for mod, reqs in sorted(by_modality.items()):
                m_latency = np.mean([r["latency_ms"] for r in reqs])
                m_rtf = np.mean([r["rtf"] for r in reqs])
                m_txt_tok = np.mean([r["output_text_tokens"] for r in reqs])
                print(f"{br.mode:<16} {mod:<14} {m_latency:>12.0f} {m_rtf:>7.2f} {m_txt_tok:>8.1f}")
        _print_separator()


def save_json_report(
    all_results: list[BenchmarkResult],
    output_path: Path,
    *,
    model_path: str,
    seed: int,
    num_repeats: int,
    modalities: list[str],
    modes: list[str],
) -> None:
    report = {
        "model_path": model_path,
        "cuda": _cuda_info(),
        "seed": seed,
        "num_repeats": num_repeats,
        "modalities": modalities,
        "modes": modes,
        "results": [
            {
                "mode": br.mode,
                "num_requests": br.num_requests,
                "completed": br.completed,
                "failed": br.failed,
                "duration_s": round(br.duration_s, 1),
                "mean_latency_ms": round(br.mean_latency_ms, 1),
                "median_latency_ms": round(br.median_latency_ms, 1),
                "std_latency_ms": round(br.std_latency_ms, 1),
                "p90_latency_ms": round(br.p90_latency_ms, 1),
                "p95_latency_ms": round(br.p95_latency_ms, 1),
                "p99_latency_ms": round(br.p99_latency_ms, 1),
                "mean_rtf": round(br.mean_rtf, 3),
                "median_rtf": round(br.median_rtf, 3),
                "std_rtf": round(br.std_rtf, 3),
                "mean_output_text_tokens": round(br.mean_output_text_tokens, 1),
                "per_request": br.per_request,
            }
            for br in all_results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nBenchmark report saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_path, resume_download=True)


def _parse_modalities(raw: str) -> list[Modality]:
    parts = [p.strip() for p in raw.split(",")]
    valid: set[Modality] = {"text", "text+image", "text+video"}
    result: list[Modality] = []
    for p in parts:
        if p not in valid:
            raise ValueError(f"Unknown modality: {p!r}. Valid: {sorted(valid)}")
        result.append(p)  # type: ignore[arg-type]
    return result


def _parse_modes(raw: str) -> list[Mode]:
    if raw == "all":
        return ["non_async", "hf"]
    parts = [p.strip() for p in raw.split(",")]
    valid: set[Mode] = {"non_async", "hf", "all"}
    result: list[Mode] = []
    for p in parts:
        if p not in valid:
            raise ValueError(f"Unknown mode: {p!r}. Valid: {sorted(valid)}")
        result.append(p)  # type: ignore[arg-type]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MiniCPM-o 4.5 text/image/video → audio generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --model-path openbmb/MiniCPM-o-4_5 --mode all
              %(prog)s --model-path openbmb/MiniCPM-o-4_5 --mode non_async --num-repeats 5
              %(prog)s --model-path openbmb/MiniCPM-o-4_5 --mode hf --modalities text
              %(prog)s --model-path /local/path/MiniCPM-o-4_5 --mode all --output-dir results/
        """),
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="HuggingFace model ID or local path to MiniCPM-o 4.5 checkpoint.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        help="Benchmark modes: non_async, hf, all (default: all).",
    )
    parser.add_argument(
        "--modalities",
        default="text",
        help="Modalities to test: text, text+image, text+video (comma-separated, default: text).",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=1,
        help="Number of times to repeat each mode x modality combination (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens for HF mode (default: 2048).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for HF mode (default: 0.7).",
    )
    parser.add_argument(
        "--output-dir",
        default="bench_results",
        help="Directory for JSON output (default: bench_results/).",
    )
    parser.add_argument(
        "--stage-config-path",
        help="Path to non-async stage config YAML (default: auto-detected).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help=(
            "Optional CUDA_VISIBLE_DEVICES value to apply before loading HF/vLLM. "
            "Use the same value for comparable HF and vLLM runs."
        ),
    )
    parser.add_argument(
        "--skip-vllm-omni",
        action="store_true",
        help="Skip vLLM-Omni benchmarks (only run HF).",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HF benchmark (only run vLLM-Omni).",
    )
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    modes = _parse_modes(args.mode)
    modalities = _parse_modalities(args.modalities)

    if args.skip_vllm_omni:
        modes = [m for m in modes if m == "hf"]
    if args.skip_hf:
        modes = [m for m in modes if m != "hf"]

    if not modes:
        print("ERROR: No modes selected after applying --skip-* flags.", file=sys.stderr)
        sys.exit(1)

    model_path = _resolve_model_path(args.model_path)
    print(f"Model path: {model_path}")
    print(f"Modes: {modes}")
    print(f"Modalities: {modalities}")
    print(f"Repeats per combination: {args.num_repeats}")
    print(f"Seed: {args.seed}")
    _print_cuda_info()
    print()

    non_async_config = args.stage_config_path or DEFAULT_STAGE_CONFIG

    all_aggregated: list[BenchmarkResult] = []

    for mode in modes:
        print(f"=== Running {mode} benchmark ===")
        t0 = time.perf_counter()

        if mode == "hf":
            results = run_hf_bench(
                model_path,
                modalities=modalities,
                num_repeats=args.num_repeats,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        elif mode == "non_async":
            results = run_vllm_omni_bench(
                model_path,
                non_async_config,
                mode_label="non_async",
                modalities=modalities,
                num_repeats=args.num_repeats,
                seed=args.seed,
            )
        else:
            print(f"Unknown mode: {mode}, skipping.")
            continue

        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.0f}s\n")

        agg = aggregate(results, mode)
        all_aggregated.append(agg)

    print_summary_table(all_aggregated)

    output_dir = Path(args.output_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"minicpmo45_bench_{ts}.json"
    save_json_report(
        all_aggregated,
        json_path,
        model_path=model_path,
        seed=args.seed,
        num_repeats=args.num_repeats,
        modalities=[str(m) for m in modalities],
        modes=modes,
    )

    # Exit with error if any benchmark failed entirely
    failures = [br for br in all_aggregated if br.completed == 0]
    if failures:
        print(f"\nERROR: {len(failures)} mode(s) had zero successful runs.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
