#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference for all supported Omni-Diffusion tasks."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from PIL import Image
from transformers import AutoTokenizer
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni

TASKS = ("t2i", "vqa", "asr", "tts", "s2i", "svqa")
TEXT_TASKS = frozenset({"vqa", "asr", "svqa"})

DEFAULT_PROMPTS = {
    "t2i": (
        "A super realistic and hyper-detailed 8K fantasy night scene showing "
        "an amazing beach under the full moon, lit by dramatic lighting."
    ),
    "vqa": "Describe the image in detail.",
    "asr": "Convert the speech to text.",
    "tts": "Get the trust fund to the bank early.",
    "s2i": "Please generate an image based on the input audio.",
    "svqa": "Please respond to the input audio based on the given image.",
}


def _require_file(value: str, option: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{option} does not exist: {path}")
    return path


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = load_audio(str(path), sr=None, mono=False)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def _get_image(image_path: str | None) -> Image.Image:
    if image_path:
        return Image.open(_require_file(image_path, "--image-path")).convert("RGB")
    return ImageAsset("cherry_blossom").pil_image.convert("RGB")


def _get_audio(audio_path: str | None) -> tuple[np.ndarray, int]:
    if audio_path:
        return _load_audio(_require_file(audio_path, "--audio-path"))
    waveform, sample_rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def build_prompt(args: argparse.Namespace) -> dict[str, Any]:
    prompt = args.prompt or DEFAULT_PROMPTS[args.task]

    if args.task == "t2i":
        return {
            "prompt": f"Generate an image based on the provided text description.\n{prompt}",
            "modalities": ["image"],
        }
    if args.task == "tts":
        return {
            "prompt": f"Convert the text to speech.\n{prompt}",
            "modalities": ["audio"],
        }
    if args.task == "s2i":
        audio_path = (
            _require_file(args.audio_path, "--audio-path")
            if args.audio_path
            else AudioAsset("mary_had_lamb").get_local_path()
        )
        return {
            "prompt": prompt,
            "multi_modal_data": {"audio": str(audio_path)},
            "modalities": ["image"],
        }

    multi_modal_data: dict[str, Any] = {}
    if args.task == "svqa":
        multi_modal_data["image"] = _get_image(args.image_path)
        multi_modal_data["audio"] = _get_audio(args.audio_path)
        prompt = (
            f"<|im_start|>system\n{prompt}<|im_end|>\n"
            "<|im_start|>user\n<|audio|>\n<|image|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    elif args.task == "vqa":
        multi_modal_data["image"] = _get_image(args.image_path)
        prompt = f"{prompt}\n<|image|>"
    elif args.task == "asr":
        multi_modal_data["audio"] = _get_audio(args.audio_path)

    return {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
        "modalities": ["text"],
    }


def _render_chat_prompt(model: str, prompt: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"Expected rendered chat prompt to be a string, got {type(rendered)!r}.")
    return rendered


def _completion_output(result: Any) -> Any:
    request_output = getattr(result, "request_output", None)
    outputs = getattr(request_output, "outputs", None) or []
    if not outputs:
        raise RuntimeError("Omni-Diffusion returned no completion output.")
    return outputs[0]


def _multimodal_output(result: Any) -> Mapping[str, Any]:
    completion = _completion_output(result)
    output = getattr(completion, "multimodal_output", None)
    if not isinstance(output, Mapping):
        raise RuntimeError("Omni-Diffusion completion did not contain multimodal output.")
    return output


def _image_from_tensor(value: Any) -> Image.Image:
    tensor = torch.as_tensor(value).detach().float().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected image tensor with shape [C, H, W], got {tuple(tensor.shape)}.")
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor[:3].permute(1, 2, 0)
    array = (tensor.clamp(0, 1) * 255).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="RGB")


def _get_result_image(result: Any) -> Image.Image:
    images = getattr(result, "images", None)
    if not images:
        request_output = getattr(result, "request_output", None)
        images = getattr(request_output, "images", None)
    if images:
        image = images[0]
        return image if isinstance(image, Image.Image) else _image_from_tensor(image)

    multimodal_output = _multimodal_output(result)
    image = multimodal_output.get("image")
    if image is None:
        raise RuntimeError(f"Omni-Diffusion returned no image; available multimodal keys: {list(multimodal_output)}.")
    return image if isinstance(image, Image.Image) else _image_from_tensor(image)


def save_result(task: str, result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if task in TEXT_TASKS:
        text = str(getattr(_completion_output(result), "text", ""))
        output_path.write_text(text, encoding="utf-8")
        print(text)
    elif task == "tts":
        mm_output = _multimodal_output(result)
        waveform = torch.as_tensor(mm_output["audio"]).detach().float().cpu().squeeze().numpy()
        sample_rate = int(torch.as_tensor(mm_output["sr"]).item())
        sf.write(output_path, waveform, sample_rate, format="WAV")
    elif task in {"t2i", "s2i"}:
        _get_result_image(result).save(output_path)

    print(f"Saved {task.upper()} output to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--model", default="lijiang/Omni-Diffusion")
    parser.add_argument("--deploy-config", required=True, help="Resolved task deploy override YAML.")
    parser.add_argument("--prompt")
    parser.add_argument("--image-path")
    parser.add_argument("--audio-path")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_suffix = ".txt" if args.task in TEXT_TASKS else ".wav" if args.task == "tts" else ".png"
    output_path = Path(args.output or f"/tmp/omni_diffusion_offline/{args.task}{default_suffix}")

    prompt = build_prompt(args)
    if args.task in {"t2i", "tts"}:
        prompt["prompt"] = _render_chat_prompt(args.model, prompt["prompt"])

    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        trust_remote_code=True,
    )
    try:
        results = omni.generate(prompt, use_tqdm=True)
    finally:
        omni.close()
    if not results:
        raise RuntimeError("Omni-Diffusion returned no result.")
    save_result(args.task, results[-1], output_path)


if __name__ == "__main__":
    main()
