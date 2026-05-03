#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline MiniCPM-o 4.5 text/image/video to audio example."""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = "openbmb/MiniCPM-o-4_5"
DEFAULT_STAGE_CONFIG = REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "minicpmo.yaml"

QueryType = Literal["text", "use_image", "use_video"]

AUDIO_OUTPUT_SYSTEM_PROMPT = (
    "When audio output is requested, reply with speech only and follow any requested length constraints."
)

DEFAULT_PROMPTS: dict[QueryType, str] = {
    "text": "Please read this sentence aloud: vLLM Omni is testing MiniCPM text to audio generation.",
    "use_image": "Describe the image in one concise spoken sentence.",
    "use_video": "Describe the video in one concise spoken sentence.",
}


def _normalize_token_ids(tokenized: Any) -> list[int]:
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def _parse_modalities(value: str) -> list[str]:
    modalities = [item.strip() for item in value.split(",") if item.strip()]
    if not modalities:
        raise ValueError("--modalities must include at least one modality")
    valid = {"text", "audio"}
    invalid = sorted(set(modalities) - valid)
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}. Valid: {sorted(valid)}")
    return modalities


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


def _build_text_prompt(text: str, modalities: list[str]) -> dict[str, Any]:
    assistant_prefix = "<|im_start|>assistant\n"
    if "audio" in modalities:
        assistant_prefix += "<think>\n\n</think>\n\n<|tts_bos|>"
    return {
        "prompt": (
            f"<|im_start|>system\n{AUDIO_OUTPUT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"{assistant_prefix}"
        ),
        "modalities": modalities,
    }


def _synthetic_image(seed: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    width, height = 224, 224
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for _ in range(5):
        side = rng.randint(28, 56)
        x = rng.randint(0, width - side - 1)
        y = rng.randint(0, height - side - 1)
        color = (rng.randint(20, 240), rng.randint(20, 240), rng.randint(20, 240))
        draw.rectangle([x, y, x + side, y + side], fill=color, outline=(0, 0, 0), width=3)
    return np.asarray(image, dtype=np.uint8).copy()


def _synthetic_video(seed: int, num_frames: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    width, height = 96, 96
    objects = []
    for _ in range(4):
        radius = rng.randint(7, 11)
        angle = rng.uniform(0, 2 * math.pi)
        speed = rng.uniform(2.0, 4.5)
        objects.append(
            {
                "x": float(rng.randint(radius, width - radius)),
                "y": float(rng.randint(radius, height - radius)),
                "vx": speed * math.cos(angle),
                "vy": speed * math.sin(angle),
                "radius": radius,
                "color": (
                    rng.randint(40, 255),
                    rng.randint(40, 255),
                    rng.randint(40, 255),
                ),
            }
        )

    frames = []
    for _ in range(num_frames):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for item in objects:
            item["x"] += item["vx"]
            item["y"] += item["vy"]
            radius = int(item["radius"])
            if item["x"] - radius <= 0 or item["x"] + radius >= width:
                item["vx"] = -item["vx"]
                item["x"] = max(radius, min(width - radius, item["x"]))
            if item["y"] - radius <= 0 or item["y"] + radius >= height:
                item["vy"] = -item["vy"]
                item["y"] = max(radius, min(height - radius, item["y"]))
            x = int(item["x"])
            y = int(item["y"])
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=item["color"],
            )
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames, axis=0)


def _load_image(path: str | None, seed: int) -> np.ndarray:
    if not path:
        return _synthetic_image(seed)
    from PIL import Image

    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8).copy()


def _load_video(path: str | None, seed: int, num_frames: int) -> np.ndarray:
    if not path:
        return _synthetic_video(seed, num_frames)
    from vllm.assets.video import video_to_ndarrays

    video_path = Path(path).expanduser()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return np.asarray(video_to_ndarrays(str(video_path), num_frames=num_frames), dtype=np.uint8)


def _build_multimodal_prompt(
    query_type: QueryType,
    text: str,
    modalities: list[str],
    *,
    image_path: str | None,
    video_path: str | None,
    seed: int,
    num_video_frames: int,
) -> dict[str, Any]:
    prompt = _build_text_prompt(text, modalities)
    user_content = ""
    multi_modal_data: dict[str, Any] = {}

    if query_type == "use_image":
        user_content += "(<image>./</image>)"
        multi_modal_data["image"] = _load_image(image_path, seed)
    elif query_type == "use_video":
        user_content += "(<video>./</video>)"
        multi_modal_data["video"] = _load_video(video_path, seed, num_video_frames)

    user_content += text
    assistant_prefix = "<|im_start|>assistant\n"
    if "audio" in modalities:
        assistant_prefix += "<think>\n\n</think>\n\n<|tts_bos|>"

    prompt["prompt"] = (
        f"<|im_start|>system\n{AUDIO_OUTPUT_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"{assistant_prefix}"
    )
    if multi_modal_data:
        prompt["multi_modal_data"] = multi_modal_data
    return prompt


def _build_prompt(args: argparse.Namespace, modalities: list[str]) -> dict[str, Any]:
    query_type: QueryType = args.query_type
    prompt_text = args.prompt or DEFAULT_PROMPTS[query_type]
    if query_type == "text" and modalities == ["audio"]:
        return _build_tts_prompt(args.model_path, prompt_text)
    if query_type == "text":
        return _build_text_prompt(prompt_text, modalities)
    return _build_multimodal_prompt(
        query_type,
        prompt_text,
        modalities,
        image_path=args.image_path,
        video_path=args.video_path,
        seed=args.seed,
        num_video_frames=args.num_video_frames,
    )


def _extract_text_and_audio(outputs: list[Any]) -> tuple[str, np.ndarray | None, int]:
    import torch

    text_output = ""
    audio_np: np.ndarray | None = None
    sample_rate = 24000

    for stage_output in outputs:
        final_output_type = getattr(stage_output, "final_output_type", None)
        request_output = getattr(stage_output, "request_output", None)
        if request_output is None:
            continue

        if final_output_type == "text" and getattr(request_output, "outputs", None):
            text_output += request_output.outputs[0].text or ""
            continue

        if final_output_type != "audio":
            continue

        multimodal_output = getattr(request_output, "multimodal_output", None)
        if not multimodal_output and getattr(request_output, "outputs", None):
            multimodal_output = getattr(request_output.outputs[0], "multimodal_output", None)
        if not isinstance(multimodal_output, dict):
            continue

        sr_obj = multimodal_output.get("sr", sample_rate)
        if isinstance(sr_obj, list) and sr_obj:
            sr_obj = sr_obj[-1]
        if hasattr(sr_obj, "item"):
            sr_obj = sr_obj.item()
        sample_rate = int(sr_obj)

        audio_obj = multimodal_output.get("audio")
        if isinstance(audio_obj, list):
            tensor_parts = [part for part in audio_obj if isinstance(part, torch.Tensor)]
            array_parts = [part for part in audio_obj if isinstance(part, np.ndarray)]
            if tensor_parts:
                audio_tensor = torch.cat(tensor_parts, dim=-1)
                audio_np = audio_tensor.detach().cpu().float().numpy().reshape(-1)
            elif array_parts:
                audio_np = np.concatenate(array_parts, axis=-1).astype(np.float32).reshape(-1)
        elif isinstance(audio_obj, torch.Tensor):
            audio_np = audio_obj.detach().cpu().float().numpy().reshape(-1)
        elif isinstance(audio_obj, np.ndarray):
            audio_np = audio_obj.astype(np.float32).reshape(-1)

    return text_output, audio_np, sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL)
    parser.add_argument("--stage-configs-path", default=str(DEFAULT_STAGE_CONFIG))
    parser.add_argument("--query-type", choices=["text", "use_image", "use_video"], default="text")
    parser.add_argument("--prompt")
    parser.add_argument("--image-path")
    parser.add_argument("--video-path")
    parser.add_argument("--num-video-frames", type=int, default=30)
    parser.add_argument("--modalities", default="audio", help="Comma-separated output modalities: text,audio")
    parser.add_argument("--output-wav", default="minicpmo45_output.wav")
    parser.add_argument("--output-text", default="minicpmo45_output.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--log-stats", action="store_true")
    parser.add_argument("--stage-init-timeout", type=int, default=20 * 60)
    parser.add_argument("--init-timeout", type=int, default=30 * 60)
    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    modalities = _parse_modalities(args.modalities)
    prompt = _build_prompt(args, modalities)

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(
        model=args.model_path,
        stage_configs_path=args.stage_configs_path,
        trust_remote_code=True,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.init_timeout,
    )
    try:
        start = time.perf_counter()
        outputs = omni.generate(prompt, use_tqdm=False)
        elapsed = time.perf_counter() - start
        text_output, audio_np, sample_rate = _extract_text_and_audio(outputs)
    finally:
        omni.close()

    if text_output.strip():
        text_path = Path(args.output_text)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text_output, encoding="utf-8")
        print(f"Text saved to {text_path}")

    if audio_np is not None and audio_np.size > 0:
        wav_path = Path(args.output_wav)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, audio_np, sample_rate)
        duration = float(audio_np.size / sample_rate)
        print(f"Audio saved to {wav_path}")
        print(f"Audio duration: {duration:.2f}s")

    print(f"E2E latency: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
