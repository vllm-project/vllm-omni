#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

TASK_CHOICES = ("t2t", "t2i", "t2s", "i2i", "i2t", "s2t", "v2t")

TASK_DEFAULT_RUNTIME = {
    "t2t": ("mmu", "mmu", 0, "text"),
    "t2i": ("t2i", "t2i_gen", 2, "image"),
    "t2s": ("t2s_mmu_like", "t2s_gen", 1, "audio"),
    "i2i": ("i2i", "i2i", 2, "image"),
    "i2t": ("mmu", "mmu", 0, "text"),
    "s2t": ("s2t", "s2t", 0, "text"),
    "v2t": ("v2t", "v2t", 0, "text"),
}

TASK_RUNTIME_FALLBACKS: dict[str, dict[str, Any]] = {
    "t2t": {
        "output_dir": "/tmp/dynin_end2end_outputs",
        "prompt_max_text_len": 1024,
        "max_new_tokens": 1024,
        "steps": 1024,
        "block_length": 16,
        "temperature": 0.0,
        "cfg_scale": 0.0,
    },
    "t2i": {
        "output_dir": "/tmp/dynin_t2i_outputs",
        "prompt_max_text_len": 128,
        "image_token_count": 1024,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "timesteps": 20,
        "guidance_scale": 3.5,
        "temperature": 1.0,
    },
    "i2i": {
        "output_dir": "/tmp/dynin_i2i_outputs",
        "prompt_max_text_len": 128,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "timesteps": 64,
        "guidance_scale": 3.5,
        "temperature": 1.0,
        "image_resolution": 336,
        "use_train_i2i_prompt": True,
    },
    "i2t": {
        "output_dir": "/tmp/dynin_i2t_outputs",
        "prompt_max_text_len": 128,
        "max_new_tokens": 128,
        "steps": 128,
        "block_length": 2,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "image_resolution": 480,
        "remasking": "low_confidence",
    },
    "s2t": {
        "output_dir": "/tmp/dynin_s2t_outputs",
        "prompt_max_text_len": 1024,
        "max_new_tokens": 128,
        "steps": 128,
        "block_length": 2,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "remasking": "low_confidence",
    },
    "t2s": {
        "output_dir": "/tmp/dynin_t2s_outputs",
        "runtime_task": "t2s_mmu_like",
        "prompting_task": "t2s_gen",
        "prompt_max_text_len": 1024,
        "t2s_token_length": 512,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "audio_codebook_size": 4096,
        "steps": 512,
        "block_length": 128,
        "temperature": 1.0,
        "cfg_scale": 2.5,
        "t2s_condition": "gender-female_emotion-neutral_speed-normal_pitch-normal",
    },
    "v2t": {
        "output_dir": "/tmp/dynin_v2t_outputs",
        "prompt_max_text_len": 1024,
        "max_new_tokens": 128,
        "steps": 128,
        "block_length": 2,
        "temperature": 0.0,
        "cfg_scale": 0.0,
        "mask_token_id": 126336,
        "codebook_size": 8192,
        "image_resolution": 224,
        "num_frames": 5,
        "remasking": "low_confidence",
    },
}

DEFAULT_I2T_QUESTION = "Please describe this image in detail."
DEFAULT_S2T_INSTRUCTION = "Transcribe the given audio."
DEFAULT_V2T_QUESTION = "Please provide a detailed description of the video."
DEFAULT_T2T_PROMPT = "Explain multimodal LLM inference in 3 sentences."
DEFAULT_T2S_INSTRUCTION = "Convert the given text into spoken audio."
DEFAULT_T2S_PROMPT = "Hello. This is a default text-to-speech sample."

DYNIN_SPECIAL_TOKENS = (
    "<|soi|>",
    "<|eoi|>",
    "<|sov|>",
    "<|eov|>",
    "<|t2i|>",
    "<|mmu|>",
    "<|t2v|>",
    "<|v2v|>",
    "<|lvg|>",
    "<|i2i|>",
    "<|ti2ti|>",
    "<|v2t|>",
    "<|v2s|>",
    "<|s2t|>",
    "<|t2s|>",
    "<|s2s|>",
    "<|soa|>",
    "<|eoa|>",
)


def bootstrap_repo_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def ensure_safe_import_for_vllm() -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    try:
        import torchvision  # noqa: F401

        return
    except Exception:
        pass

    import enum

    class _InterpolationMode(enum.Enum):
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3
        LANCZOS = 1
        HAMMING = 4
        BOX = 5

    tv_mod = types.ModuleType("torchvision")
    tv_mod.__dict__["__version__"] = "0.0-stub"
    tv_mod.__spec__ = ModuleSpec(name="torchvision", loader=None)
    transforms_mod = types.ModuleType("torchvision.transforms")
    transforms_mod.__spec__ = ModuleSpec(name="torchvision.transforms", loader=None)
    transforms_mod.InterpolationMode = _InterpolationMode
    tv_mod.transforms = transforms_mod
    sys.modules["torchvision"] = tv_mod
    sys.modules["torchvision.transforms"] = transforms_mod


def sanitize_repo_id(repo_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", repo_id)


def is_hf_repo_id(value: str) -> bool:
    return isinstance(value, str) and value.count("/") == 1 and all(value.split("/", 1))


def ensure_local_model_dir(model: str, cache_dir: Path, localize: bool) -> Path:
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        return model_path.resolve()
    if not localize:
        return Path(model)

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / ".hf_home"))
    local_dir = cache_dir / sanitize_repo_id(model)
    if not local_dir.exists():
        print(f"[end2end] Downloading model into local cache: {local_dir}")
        snapshot_download(
            repo_id=model,
            local_dir=str(local_dir),
            local_dir_use_symlinks=True,
            resume_download=True,
        )
    return local_dir.resolve()


def resolve_local_only(
    override: bool | None,
    source: str,
    default: bool,
) -> bool:
    if override is not None:
        return bool(override)
    return default or Path(source).expanduser().is_dir()


def load_text_tokenizer(tokenizer_source: str, local_files_only: bool):
    from transformers import AutoTokenizer

    kwargs = {
        "trust_remote_code": True,
        "padding_side": "left",
        "local_files_only": bool(local_files_only),
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)
    return tokenizer


def preprocess_image(image: Image.Image, resolution: int) -> torch.Tensor:
    w, h = image.size
    short_side = min(w, h)
    scale = resolution / short_side
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    image = image.crop((left, top, left + resolution, top + resolution))
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - 0.5) / 0.5


def load_vq_image_encoder(source: str, local_files_only: bool, device: torch.device) -> Any:
    from vllm_omni.model_executor.models.dynin_omni.dynin_omni_common import get_dynin_magvit_attr

    MAGVITv2 = get_dynin_magvit_attr("MAGVITv2", source=source, local_files_only=local_files_only)
    vq_model = MAGVITv2.from_pretrained(source, local_files_only=local_files_only).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    return vq_model


def encode_image_tokens(
    image_path: Path,
    vq_model: Any,
    device: torch.device,
    resolution: int,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess_image(image, resolution=resolution).unsqueeze(0).to(device)
    with torch.no_grad():
        token_ids = vq_model.get_code(image_tensor)
    token_ids = torch.as_tensor(token_ids, dtype=torch.long).detach().cpu()
    if token_ids.ndim == 2 and token_ids.shape[0] == 1:
        token_ids = token_ids[0]
    return token_ids.contiguous()


def encode_video_tokens(
    video_path: Path,
    vq_model: Any,
    device: torch.device,
    resolution: int,
    num_frames: int,
) -> torch.Tensor:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"Video has no readable frames: {video_path}")
    if len(frames) < num_frames:
        raise ValueError(f"Video has {len(frames)} frames, requires >= {num_frames}: {video_path}")

    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    token_list: list[torch.Tensor] = []
    for idx in indices:
        pil = Image.fromarray(frames[int(idx)])
        frame_tensor = preprocess_image(pil, resolution=resolution).unsqueeze(0).to(device)
        with torch.no_grad():
            token_list.append(torch.as_tensor(vq_model.get_code(frame_tensor), dtype=torch.long))
    merged = torch.cat(token_list, dim=1).detach().cpu()
    if merged.ndim == 2 and merged.shape[0] == 1:
        merged = merged[0]
    return merged.contiguous()


def load_vq_audio_encoder(source: str, local_files_only: bool, device: torch.device) -> Any:
    from transformers import AutoModel

    kwargs = {
        "trust_remote_code": True,
        "local_files_only": bool(local_files_only),
        "low_cpu_mem_usage": False,
    }
    try:
        model = AutoModel.from_pretrained(source, **kwargs)
    except TypeError:
        kwargs.pop("low_cpu_mem_usage", None)
        try:
            model = AutoModel.from_pretrained(source, **kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            model = AutoModel.from_pretrained(source, **kwargs)
    model.requires_grad_(False)
    model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def encode_audio_tokens(audio_path: Path, vq_audio_model: Any) -> torch.Tensor:
    encoded = vq_audio_model.encode(str(audio_path))
    if isinstance(encoded, dict):
        for key in ("input_ids", "token_ids", "codes", "tokens"):
            if key in encoded:
                encoded = encoded[key]
                break
    encoded = torch.as_tensor(encoded, dtype=torch.long).detach().cpu()
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    elif encoded.ndim > 2:
        encoded = encoded.view(encoded.shape[0], -1)
    return encoded.contiguous()


def build_chat_prompt(content: str) -> str:
    return (
        f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def resolve_task_text(
    *,
    task_name: str,
    text: str,
    instruction: str = "",
    raw_prompt: bool = False,
) -> str:
    text = str(text or "").strip()

    if task_name == "t2t" and not text:
        return DEFAULT_T2T_PROMPT
    if task_name == "i2t" and not text:
        return DEFAULT_I2T_QUESTION
    if task_name == "s2t" and not text:
        return DEFAULT_S2T_INSTRUCTION
    if task_name == "v2t" and not text:
        return DEFAULT_V2T_QUESTION
    if task_name in {"t2i", "i2i"} and not text:
        return "A high quality detailed image."

    if task_name != "t2s":
        return text

    if not text:
        text = DEFAULT_T2S_PROMPT

    if raw_prompt:
        return text

    instruction = str(instruction or "").strip() or DEFAULT_T2S_INSTRUCTION
    return build_chat_prompt(f"{instruction}\n{text}")


def load_universal_prompting(
    *,
    tokenizer: Any,
    tokenizer_source: str,
    max_text_len: int,
    cond_dropout_prob: float,
    local_files_only: bool,
    max_audio_len: int = 512,
    max_audio_len_short: int = 256,
) -> Any:
    from vllm_omni.model_executor.models.dynin_omni.dynin_omni_common import (
        DYNIN_REMOTE_SETTINGS,
        resolve_remote_attr,
    )

    UniversalPrompting = resolve_remote_attr(
        "UniversalPrompting",
        module_name="prompting_utils",
        settings=DYNIN_REMOTE_SETTINGS,
        source=tokenizer_source,
        local_files_only=bool(local_files_only),
        fallback_module_names=("modeling_dynin_omni",),
    )
    init_kwargs: dict[str, Any] = {
        "max_text_len": int(max_text_len),
        "special_tokens": DYNIN_SPECIAL_TOKENS,
        "ignore_id": -100,
        "cond_dropout_prob": float(cond_dropout_prob),
        "use_reserved_token": True,
        "max_audio_len": int(max_audio_len),
        "max_audio_len_short": int(max_audio_len_short),
    }
    try:
        return UniversalPrompting(tokenizer, **init_kwargs)
    except TypeError:
        init_kwargs.pop("max_audio_len", None)
        init_kwargs.pop("max_audio_len_short", None)
        return UniversalPrompting(tokenizer, **init_kwargs)


def _runtime_fallback(task: str, key: str, value: Any) -> Any:
    if isinstance(value, str):
        if value.strip() != "":
            return value
    elif value is not None:
        return value
    return TASK_RUNTIME_FALLBACKS.get(task, {}).get(key)


def _validate_generation_args(*, task: str, max_new_tokens: int, steps: int, block_length: int) -> None:
    # Keep i2t/v2t generation constraints aligned with i2t.py/v2t.py.
    if task not in {"i2t", "v2t"}:
        return
    if max_new_tokens <= 0:
        raise ValueError(f"{task} requires max_new_tokens > 0.")
    if block_length <= 0:
        raise ValueError(f"{task} requires block_length > 0.")
    if steps <= 0:
        raise ValueError(f"{task} requires steps > 0.")
    if max_new_tokens % block_length != 0:
        raise ValueError(f"{task} requires max_new_tokens % block_length == 0, got {max_new_tokens} % {block_length}")
    num_blocks = max_new_tokens // block_length
    if num_blocks <= 0:
        raise ValueError(f"{task} has invalid num_blocks.")
    if steps % num_blocks != 0:
        raise ValueError(
            f"{task} requires steps % (max_new_tokens // block_length) == 0, "
            f"got steps={steps}, max_new_tokens={max_new_tokens}, block_length={block_length}"
        )


def make_prompt_payload(
    *,
    task: str,
    text: str,
    image_tokens: torch.Tensor | None,
    audio_tokens: torch.Tensor | None,
    video_tokens: torch.Tensor | None,
    image_placeholder_tokens: int,
    audio_placeholder_tokens: int,
    image_token_offset: int,
    speech_token_offset: int,
    mask_token_id: int,
    use_train_i2i_prompt: bool,
) -> tuple[Any, str]:
    runtime_task, prompting_task, _, _ = TASK_DEFAULT_RUNTIME[task]
    del runtime_task

    if task == "t2t":
        payload = ([[]], [build_chat_prompt(text)])
        return payload, prompting_task

    if task == "i2t":
        if image_tokens is None:
            raise ValueError("i2t requires image tokens")
        img = image_tokens.view(-1).long() + int(image_token_offset)
        payload = ([[img]], [build_chat_prompt(text)])
        return payload, prompting_task

    if task == "s2t":
        if audio_tokens is None:
            raise ValueError("s2t requires audio tokens")
        aud = audio_tokens.long() + int(speech_token_offset)
        if aud.ndim == 1:
            aud = aud.unsqueeze(0)
        payload = ([aud], [build_chat_prompt(text)])
        return payload, prompting_task

    if task == "v2t":
        if video_tokens is None:
            raise ValueError("v2t requires video tokens")
        vid = video_tokens.view(-1).long() + int(image_token_offset)
        payload = (vid.unsqueeze(0), [build_chat_prompt(text)])
        return payload, prompting_task

    if task == "t2i":
        image_placeholder = torch.full(
            (1, int(image_placeholder_tokens)),
            fill_value=int(mask_token_id),
            dtype=torch.long,
        )
        payload = ([text], image_placeholder)
        return payload, prompting_task

    if task == "i2i":
        if image_tokens is None:
            raise ValueError("i2i requires image tokens")
        src = image_tokens.view(1, -1).long() + int(image_token_offset)
        target_len = int(image_placeholder_tokens) if image_placeholder_tokens > 0 else int(src.shape[1])
        image_placeholder = torch.full(
            (1, target_len),
            fill_value=int(mask_token_id),
            dtype=torch.long,
        )
        if use_train_i2i_prompt:
            labels_placeholder = torch.full(
                (1, target_len),
                fill_value=-100,
                dtype=torch.long,
            )
            payload = ([text], src, image_placeholder, labels_placeholder)
            return payload, "i2i"
        payload = ([text], src, image_placeholder)
        return payload, "i2i_gen"

    if task == "t2s":
        audio_placeholder = torch.full(
            (1, int(audio_placeholder_tokens)),
            fill_value=int(mask_token_id),
            dtype=torch.long,
        )
        payload = ([text], audio_placeholder)
        return payload, prompting_task

    raise ValueError(f"Unsupported task: {task}")


def _to_1d_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.long)
    else:
        tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 0:
        tensor = tensor.view(1)
    elif tensor.ndim >= 2:
        tensor = tensor.view(tensor.shape[0], -1)[0]
    return [int(v) for v in tensor.tolist()]


def _run_uni_prompting(uni_prompting: Any, payload: Any, prompting_task: str) -> tuple[list[int], list[int]]:
    prepared = uni_prompting(payload, prompting_task)
    if isinstance(prepared, tuple):
        prepared_input_ids = prepared[0] if len(prepared) > 0 else None
        prepared_attention_mask = prepared[1] if len(prepared) > 1 else None
    else:
        prepared_input_ids = prepared
        prepared_attention_mask = None

    input_ids = _to_1d_int_list(prepared_input_ids)
    attention_mask = _to_1d_int_list(prepared_attention_mask)
    if not input_ids:
        raise RuntimeError(f"UniversalPrompting returned empty input_ids for task={prompting_task}")
    return input_ids, attention_mask


def _get_special_token_id(uni_prompting: Any, token: str) -> int:
    sptids = getattr(uni_prompting, "sptids_dict", None) or {}
    if token not in sptids:
        raise KeyError(f"Special token not found in UniversalPrompting.sptids_dict: {token}")
    token_ids = _to_1d_int_list(sptids[token])
    if not token_ids:
        raise ValueError(f"Special token id is empty for token: {token}")
    return int(token_ids[0])


def _tokenize_chat_query(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(build_chat_prompt(text), return_tensors="pt").input_ids[0]
    token_ids = _to_1d_int_list(encoded)
    if not token_ids:
        raise RuntimeError("Failed to tokenize chat query text.")
    return token_ids


def _flatten_media_token_ids_with_offset(token_ids: Any, token_offset: int) -> list[int]:
    media_ids = token_ids
    if isinstance(media_ids, torch.Tensor):
        media_ids = media_ids.detach().cpu().reshape(-1).tolist()
    else:
        media_ids = np.asarray(media_ids).reshape(-1).tolist()
    return [int(x) + int(token_offset) for x in media_ids]


def _scalar_token_id(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise ValueError("Empty special-token tensor.")
        return int(value.view(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty special-token list.")
        return int(value[0])
    return int(value)


def build_v2t_input_ids(
    *,
    video_token_ids: Any,
    tokenizer: Any,
    uni_prompting: Any,
    question: str,
    image_token_offset: int,
) -> tuple[list[int], str]:
    media_ids = video_token_ids
    if isinstance(media_ids, torch.Tensor):
        media_ids = media_ids.detach().cpu().reshape(-1).tolist()
    else:
        media_ids = np.asarray(media_ids).reshape(-1).tolist()
    media_ids = [int(x) + int(image_token_offset) for x in media_ids]

    sptids = uni_prompting.sptids_dict
    task_id = _scalar_token_id(sptids["<|v2t|>"])
    soi_id = _scalar_token_id(sptids["<|soi|>"])
    eoi_id = _scalar_token_id(sptids["<|eoi|>"])
    sot_id = _scalar_token_id(sptids["<|sot|>"])

    prompt_text = build_v2t_chat_prompt(question)
    query_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].detach().cpu().tolist()
    input_ids = [task_id, soi_id] + media_ids + [eoi_id, sot_id] + [int(v) for v in query_ids]
    return input_ids, prompt_text


def build_i2t_input_ids(
    *,
    image_token_ids: Any,
    tokenizer: Any,
    uni_prompting: Any,
    question: str,
    image_token_offset: int,
) -> tuple[list[int], str]:
    image_ids = image_token_ids
    if isinstance(image_ids, torch.Tensor):
        image_ids = image_ids.detach().cpu().reshape(-1).tolist()
    else:
        image_ids = np.asarray(image_ids).reshape(-1).tolist()
    image_ids = [int(x) + int(image_token_offset) for x in image_ids]

    sptids = uni_prompting.sptids_dict
    task_id = _scalar_token_id(sptids["<|mmu|>"])
    soi_id = _scalar_token_id(sptids["<|soi|>"])
    eoi_id = _scalar_token_id(sptids["<|eoi|>"])
    sot_id = _scalar_token_id(sptids["<|sot|>"])

    prompt_text = build_i2t_chat_prompt(question)
    query_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].detach().cpu().tolist()
    input_ids = [task_id, soi_id] + image_ids + [eoi_id, sot_id] + [int(v) for v in query_ids]
    return input_ids, prompt_text


def build_v2t_chat_prompt(question: str) -> str:
    return (
        f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_i2t_chat_prompt(question: str) -> str:
    return (
        f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def make_mmu_prompt(
    *,
    task: str,
    text: str,
    tokenizer: Any,
    uni_prompting: Any,
    image_tokens: torch.Tensor | None,
    audio_tokens: torch.Tensor | None,
    video_tokens: torch.Tensor | None,
    image_token_offset: int,
    speech_token_offset: int,
) -> tuple[list[int], list[int]]:
    query_ids = _tokenize_chat_query(tokenizer, text)

    if task == "i2t":
        token_ids, _ = build_i2t_input_ids(
            image_token_ids=image_tokens,
            tokenizer=tokenizer,
            uni_prompting=uni_prompting,
            question=text,
            image_token_offset=int(image_token_offset),
        )
        token_ids = [int(v) for v in token_ids]
        return token_ids, [1] * len(token_ids)

    if task == "v2t":
        token_ids, _ = build_v2t_input_ids(
            video_token_ids=video_tokens,
            tokenizer=tokenizer,
            uni_prompting=uni_prompting,
            question=text,
            image_token_offset=int(image_token_offset),
        )
        token_ids = [int(v) for v in token_ids]
        return token_ids, [1] * len(token_ids)

    if task == "s2t":
        if audio_tokens is None:
            raise ValueError("s2t requires audio tokens")
        audio_ids = _to_1d_int_list(audio_tokens.long() + int(speech_token_offset))
        token_ids = [
            _get_special_token_id(uni_prompting, "<|s2t|>"),
            _get_special_token_id(uni_prompting, "<|soa|>"),
            *audio_ids,
            _get_special_token_id(uni_prompting, "<|eoa|>"),
            *query_ids,
        ]
        return token_ids, [1] * len(token_ids)

    raise ValueError(f"Unsupported task for validation-style MMU prompt: {task}")


def iter_mm_outputs(outputs: list[Any]):
    for omni_out in outputs:
        req_out = getattr(omni_out, "request_output", None)
        req_list = req_out if isinstance(req_out, list) else [req_out]
        for item in req_list:
            if item is None:
                continue
            mm_out = getattr(item, "multimodal_output", None) or {}
            if mm_out:
                yield mm_out
            completions = getattr(item, "outputs", None) or []
            for completion in completions:
                c_mm_out = getattr(completion, "multimodal_output", None) or {}
                if c_mm_out:
                    yield c_mm_out
        omni_mm = getattr(omni_out, "multimodal_output", None) or {}
        if omni_mm:
            yield omni_mm


def _to_token_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "flatten"):
        value = value.flatten().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for token in value:
        if isinstance(token, bool):
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def extract_text_output(outputs: list[Any], tokenizer: Any) -> str:
    for mm_out in iter_mm_outputs(outputs):
        text = mm_out.get("text")
        if isinstance(text, list) and text:
            text = text[-1]
        if isinstance(text, str) and text.strip():
            return text.strip()
        for key in ("text_tokens", "token_ids"):
            token_ids = _to_token_list(mm_out.get(key))
            if not token_ids:
                continue
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            if isinstance(decoded, str) and decoded.strip():
                return decoded.strip()
    return ""


def extract_image_output(outputs: list[Any]) -> torch.Tensor | None:
    for mm_out in iter_mm_outputs(outputs):
        image = mm_out.get("image")
        if isinstance(image, list) and image:
            image = image[-1]
        if isinstance(image, torch.Tensor):
            return image
    return None


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def extract_audio_output(outputs: list[Any]) -> tuple[np.ndarray, int] | None:
    for mm_out in iter_mm_outputs(outputs):
        audio = mm_out.get("audio")
        if audio is None:
            audio = mm_out.get("speech")
        if audio is None:
            continue

        def _to_wav_array(value: Any) -> np.ndarray:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy().reshape(-1).astype(np.float32)
            return np.asarray(value).reshape(-1).astype(np.float32)

        if isinstance(audio, list):
            chunks = [_to_wav_array(chunk) for chunk in audio]
            wav = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
        else:
            wav = _to_wav_array(audio)
        sr = mm_out.get("sr", 24000)
        if hasattr(sr, "item"):
            try:
                sr = int(sr.item())
            except Exception:
                sr = 24000
        elif isinstance(sr, list):
            sr = int(sr[0]) if sr else 24000
        else:
            sr = int(sr)
        return wav, sr
    return None


def save_audio_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    try:
        import soundfile as sf

        sf.write(str(path), wav, int(sr), format="WAV")
    except Exception:
        from scipy.io import wavfile

        wav_i16 = np.clip(wav, -1.0, 1.0)
        wav_i16 = (wav_i16 * 32767.0).astype(np.int16)
        wavfile.write(str(path), int(sr), wav_i16)


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynin-Omni unified offline end2end example.")
    parser.add_argument("--task", type=str, required=True, choices=TASK_CHOICES)
    parser.add_argument("--model", type=str, required=True, help="HF repo id or local model directory.")
    parser.add_argument(
        "--stage-config-path",
        type=str,
        default=str(repo_root / "vllm_omni/model_executor/stage_configs/dynin_omni.yaml"),
        help="Path to stage config yaml.",
    )
    parser.add_argument(
        "--dynin-config-path",
        type=str,
        default="",
        help="Path to DYNIN config yaml (passed through additional_information).",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/tmp/dynin_localized_models",
        help="Cache directory used when --model is HF repo id.",
    )
    parser.add_argument(
        "--localize-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true and --model is HF repo id, snapshot it under --model-cache-dir.",
    )
    parser.add_argument("--text", type=str, default="", help="Prompt/edit/question text.")
    parser.add_argument("--instruction", type=str, default="", help="Optional extra instruction.")
    parser.add_argument("--raw-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image", type=str, default="", help="Input image path for i2i/i2t.")
    parser.add_argument("--audio", type=str, default="", help="Input audio path for s2t.")
    parser.add_argument("--video", type=str, default="", help="Input video path for v2t.")
    parser.add_argument("--image-resolution", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for generated outputs.",
    )
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-tokens-per-stage", type=int, default=1)

    parser.add_argument("--runtime-task", type=str, default="", help="Override runtime task key.")
    parser.add_argument("--prompting-task", type=str, default="", help="Override prompting task key.")
    parser.add_argument("--detok-id", type=int, default=None, help="Override detok id.")

    parser.add_argument("--prompt-max-text-len", type=int, default=None)
    parser.add_argument("--cond-dropout-prob", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--remasking", type=str, default="low_confidence")

    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--noise-type", type=str, default="mask")
    parser.add_argument("--noise-schedule-name", type=str, default="cosine")
    parser.add_argument("--noise-schedule-params", type=str, default="{}")

    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--codebook-size", type=int, default=None)
    parser.add_argument("--audio-codebook-size", type=int, default=None)
    parser.add_argument("--image-token-count", type=int, default=None)
    parser.add_argument("--t2s-token-length", type=int, default=None)
    parser.add_argument(
        "--t2s-condition",
        type=str,
        default="",
    )
    parser.add_argument(
        "--use-train-i2i-prompt",
        action="store_true",
        help="Use i2i training prompt template (default behavior of i2i.py).",
    )
    parser.add_argument(
        "--no-use-train-i2i-prompt",
        dest="use_train_i2i_prompt",
        action="store_false",
        help="Use i2i_gen prompt template.",
    )
    parser.set_defaults(use_train_i2i_prompt=None)

    parser.add_argument("--tokenizer-path", type=str, default="")
    parser.add_argument("--model-local-files-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tokenizer-local-files-only", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--vq-model-image-path", type=str, default="")
    parser.add_argument("--vq-model-image-local-files-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vq-model-audio-path", type=str, default="")
    parser.add_argument("--vq-model-audio-local-files-only", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--disable-hf-xet", action=argparse.BooleanOptionalAction, default=True)
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def main() -> None:
    repo_root = bootstrap_repo_path()
    ensure_safe_import_for_vllm()
    from vllm_omni.model_executor.models.dynin_omni.dynin_omni_common import (
        DYNIN_PROMPT_SOURCE_KEY,
        DYNIN_PROMPT_SOURCE_OFFLINE_PREBUILT,
    )

    args = parse_args(repo_root)

    if args.disable_hf_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = ensure_local_model_dir(
        model=args.model,
        cache_dir=Path(args.model_cache_dir).expanduser(),
        localize=bool(args.localize_model),
    )
    model_source = str(model_dir)

    task_name = str(args.task)
    dynin_config_path = str(Path(args.dynin_config_path).expanduser())
    os.environ["DYNIN_CONFIG_PATH"] = dynin_config_path
    default_runtime_task, default_prompting_task, default_detok_id, final_modality = TASK_DEFAULT_RUNTIME[task_name]
    runtime_task = args.runtime_task.strip() or str(
        _runtime_fallback(task_name, "runtime_task", None) or default_runtime_task
    )
    prompting_task = args.prompting_task.strip() or str(
        _runtime_fallback(task_name, "prompting_task", None) or default_prompting_task
    )
    detok_id_default = _runtime_fallback(task_name, "detok_id", None)
    if detok_id_default is None:
        detok_id_default = default_detok_id
    detok_id = int(detok_id_default if args.detok_id is None else args.detok_id)

    output_dir_default = _runtime_fallback(task_name, "output_dir", args.output_dir)
    resolved_output_dir = str(output_dir_default or "/tmp/dynin_end2end_outputs")

    image_resolution_value = _runtime_fallback(
        task_name,
        "image_resolution",
        args.image_resolution,
    )
    if image_resolution_value is None:
        image_resolution_value = 336
    image_resolution = int(image_resolution_value)

    num_frames_value = _runtime_fallback(
        task_name,
        "num_frames",
        args.num_frames,
    )
    if num_frames_value is None:
        num_frames_value = 8
    num_frames = int(num_frames_value)

    prompt_max_text_len_value = _runtime_fallback(
        task_name,
        "prompt_max_text_len",
        args.prompt_max_text_len,
    )
    if prompt_max_text_len_value is None:
        prompt_max_text_len_value = 1024
    prompt_max_text_len = int(prompt_max_text_len_value)

    max_new_tokens_value = _runtime_fallback(
        task_name,
        "max_new_tokens",
        args.max_new_tokens,
    )
    if max_new_tokens_value is None:
        max_new_tokens_value = 256
    max_new_tokens = int(max_new_tokens_value)

    steps_value = _runtime_fallback(
        task_name,
        "steps",
        args.steps,
    )
    if steps_value is None:
        steps_value = 256
    steps = int(steps_value)

    block_length_value = _runtime_fallback(
        task_name,
        "block_length",
        args.block_length,
    )
    if block_length_value is None:
        block_length_value = 2
    block_length = int(block_length_value)

    temperature_value = _runtime_fallback(
        task_name,
        "temperature",
        args.temperature,
    )
    if temperature_value is None:
        temperature_value = 0.0
    temperature = float(temperature_value)

    cfg_scale_value = _runtime_fallback(
        task_name,
        "cfg_scale",
        args.cfg_scale,
    )
    if cfg_scale_value is None:
        cfg_scale_value = 0.0
    cfg_scale = float(cfg_scale_value)

    remasking = str(_runtime_fallback(task_name, "remasking", args.remasking) or "low_confidence")

    timesteps_value = _runtime_fallback(
        task_name,
        "timesteps",
        args.timesteps,
    )
    if timesteps_value is None:
        timesteps_value = 20
    timesteps = int(timesteps_value)

    guidance_scale_value = _runtime_fallback(
        task_name,
        "guidance_scale",
        args.guidance_scale,
    )
    if guidance_scale_value is None:
        guidance_scale_value = 0.0
    guidance_scale = float(guidance_scale_value)

    mask_token_id_value = _runtime_fallback(
        task_name,
        "mask_token_id",
        args.mask_token_id,
    )
    if mask_token_id_value is None:
        mask_token_id_value = 126336
    mask_token_id = int(mask_token_id_value)

    codebook_size_value = _runtime_fallback(
        task_name,
        "codebook_size",
        args.codebook_size,
    )
    if codebook_size_value is None:
        codebook_size_value = 8192
    codebook_size = int(codebook_size_value)

    audio_codebook_size_value = _runtime_fallback(
        task_name,
        "audio_codebook_size",
        args.audio_codebook_size,
    )
    if audio_codebook_size_value is None:
        audio_codebook_size_value = 4096
    audio_codebook_size = int(audio_codebook_size_value)

    image_token_count_value = _runtime_fallback(
        task_name,
        "image_token_count",
        args.image_token_count,
    )
    image_token_count = int(image_token_count_value) if image_token_count_value is not None else 0

    t2s_token_length_value = _runtime_fallback(
        task_name,
        "t2s_token_length",
        args.t2s_token_length,
    )
    if t2s_token_length_value is None:
        t2s_token_length_value = 383
    t2s_token_length = int(t2s_token_length_value)

    t2s_condition = str(
        _runtime_fallback(task_name, "t2s_condition", args.t2s_condition)
        or "gender-female_emotion-neutral_speed-normal_pitch-normal"
    )

    _validate_generation_args(
        task=task_name,
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_length=block_length,
    )

    use_train_i2i_prompt = _runtime_fallback(task_name, "use_train_i2i_prompt", args.use_train_i2i_prompt)
    if use_train_i2i_prompt is None:
        use_train_i2i_prompt = bool(task_name == "i2i")
    use_train_i2i_prompt = bool(use_train_i2i_prompt)

    if task_name in {"i2i", "i2t"} and not args.image:
        raise ValueError(f"--task {task_name} requires --image")
    if task_name == "s2t" and not args.audio:
        raise ValueError("--task s2t requires --audio")
    if task_name == "v2t" and not args.video:
        raise ValueError("--task v2t requires --video")

    text = resolve_task_text(
        task_name=task_name,
        text=args.text,
        instruction=args.instruction,
        raw_prompt=bool(args.raw_prompt),
    )

    tokenizer_source = args.tokenizer_path.strip() or model_source
    model_local_only = resolve_local_only(
        args.model_local_files_only, model_source, default=Path(model_source).is_dir()
    )
    tokenizer_local_only = resolve_local_only(
        args.tokenizer_local_files_only,
        tokenizer_source,
        default=model_local_only,
    )
    tokenizer = load_text_tokenizer(tokenizer_source, local_files_only=tokenizer_local_only)
    text_vocab_size = int(len(tokenizer))

    image_tokens: torch.Tensor | None = None
    audio_tokens: torch.Tensor | None = None
    video_tokens: torch.Tensor | None = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vq_image_source = args.vq_model_image_path.strip() or "snu-aidas/magvitv2"
    vq_audio_source = args.vq_model_audio_path.strip() or "snu-aidas/emova_speech_tokenizer_vllm"
    vq_image_local_only = resolve_local_only(args.vq_model_image_local_files_only, vq_image_source, default=False)
    vq_audio_local_only = resolve_local_only(args.vq_model_audio_local_files_only, vq_audio_source, default=False)

    if task_name in {"i2i", "i2t", "v2t"}:
        vq_image = load_vq_image_encoder(vq_image_source, vq_image_local_only, device)
        if task_name in {"i2i", "i2t"}:
            image_tokens = encode_image_tokens(
                Path(args.image).expanduser().resolve(),
                vq_model=vq_image,
                device=device,
                resolution=int(image_resolution),
            )
        if task_name == "v2t":
            video_tokens = encode_video_tokens(
                Path(args.video).expanduser().resolve(),
                vq_model=vq_image,
                device=device,
                resolution=int(image_resolution),
                num_frames=int(num_frames),
            )
        if hasattr(vq_image, "cpu"):
            vq_image = vq_image.cpu()

    if task_name == "s2t":
        vq_audio = load_vq_audio_encoder(vq_audio_source, vq_audio_local_only, device)
        audio_tokens = encode_audio_tokens(Path(args.audio).expanduser().resolve(), vq_audio)
        if hasattr(vq_audio, "cpu"):
            vq_audio = vq_audio.cpu()

    noise_schedule_params: dict[str, Any] = {}
    try:
        parsed = json.loads(args.noise_schedule_params)
        if isinstance(parsed, dict):
            noise_schedule_params = {str(k): v for k, v in parsed.items()}
    except Exception:
        noise_schedule_params = {}

    image_token_count = int(image_token_count)
    if image_token_count <= 0:
        if image_tokens is not None:
            image_token_count = int(image_tokens.numel())
        else:
            base_res = int(image_resolution)
            image_token_count = max(1, (base_res // 16) ** 2)

    uncond_input_ids: list[int] | None = None
    uncond_attention_mask: list[int] | None = None
    if task_name == "t2t":
        messages = [{"role": "user", "content": text}]
        if getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        else:
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        prompt_token_ids = _to_1d_int_list(encoded["input_ids"])
        prompt_attention_mask = _to_1d_int_list(encoded.get("attention_mask"))
        if not prompt_attention_mask:
            prompt_attention_mask = [1] * len(prompt_token_ids)
    else:
        max_audio_len_for_prompt = int(max(t2s_token_length, 512))
        if audio_tokens is not None:
            max_audio_len_for_prompt = max(max_audio_len_for_prompt, int(audio_tokens.numel()))
        max_audio_len_short_for_prompt = max(256, max_audio_len_for_prompt // 2)

        uni_prompting = load_universal_prompting(
            tokenizer=tokenizer,
            tokenizer_source=tokenizer_source,
            max_text_len=int(prompt_max_text_len),
            cond_dropout_prob=float(args.cond_dropout_prob),
            local_files_only=bool(tokenizer_local_only),
            max_audio_len=int(max_audio_len_for_prompt),
            max_audio_len_short=int(max_audio_len_short_for_prompt),
        )
        prompting_text_vocab_size = int(len(uni_prompting.text_tokenizer))

        is_mmu_task = task_name in {"i2t", "s2t", "v2t"} and not args.prompting_task.strip()
        if is_mmu_task:
            prompt_token_ids, prompt_attention_mask = make_mmu_prompt(
                task=task_name,
                text=text,
                tokenizer=uni_prompting.text_tokenizer,
                uni_prompting=uni_prompting,
                image_tokens=image_tokens,
                audio_tokens=audio_tokens,
                video_tokens=video_tokens,
                image_token_offset=prompting_text_vocab_size,
                speech_token_offset=prompting_text_vocab_size + int(codebook_size),
            )
        else:
            prompt_payload, prompting_task = make_prompt_payload(
                task=task_name,
                text=text,
                image_tokens=image_tokens,
                audio_tokens=audio_tokens,
                video_tokens=video_tokens,
                image_placeholder_tokens=image_token_count,
                audio_placeholder_tokens=int(t2s_token_length),
                image_token_offset=text_vocab_size,
                speech_token_offset=text_vocab_size + int(codebook_size),
                mask_token_id=int(mask_token_id),
                use_train_i2i_prompt=use_train_i2i_prompt,
            )
            if args.prompting_task.strip():
                prompting_task = args.prompting_task.strip()

            prompt_token_ids, prompt_attention_mask = _run_uni_prompting(
                uni_prompting,
                prompt_payload,
                prompting_task,
            )

        if task_name in {"i2t", "s2t", "v2t"}:
            prompt_attention_mask = [1] * len(prompt_token_ids)
        if not prompt_attention_mask:
            prompt_attention_mask = [1] * len(prompt_token_ids)

        if task_name in {"t2i", "i2i"} and guidance_scale > 0:
            uncond_payload, uncond_prompting_task = make_prompt_payload(
                task=task_name,
                text="",
                image_tokens=image_tokens,
                audio_tokens=audio_tokens,
                video_tokens=video_tokens,
                image_placeholder_tokens=image_token_count,
                audio_placeholder_tokens=int(t2s_token_length),
                image_token_offset=text_vocab_size,
                speech_token_offset=text_vocab_size + int(codebook_size),
                mask_token_id=int(mask_token_id),
                use_train_i2i_prompt=use_train_i2i_prompt,
            )
            uncond_input_ids, uncond_attention_mask = _run_uni_prompting(
                uni_prompting,
                uncond_payload,
                args.prompting_task.strip() or uncond_prompting_task,
            )
            if not uncond_attention_mask:
                uncond_attention_mask = [1] * len(uncond_input_ids)

    runtime_info: dict[str, Any] = {
        "task": [runtime_task],
        "detok_id": [int(detok_id)],
        DYNIN_PROMPT_SOURCE_KEY: [DYNIN_PROMPT_SOURCE_OFFLINE_PREBUILT],
        "dynin_config_path": [str(dynin_config_path)],
        "attention_mask": [prompt_attention_mask],
        "prompt_max_text_len": [int(prompt_max_text_len)],
        "prompting_max_text_len": [int(prompt_max_text_len)],
        "cond_dropout_prob": [float(args.cond_dropout_prob)],
        "prompting_cond_dropout_prob": [float(args.cond_dropout_prob)],
        "tokenizer_path": [str(tokenizer_source)],
        "text_vocab_size": [int(text_vocab_size)],
        "model_local_files_only": [bool(model_local_only)],
        "max_new_tokens": [int(max_new_tokens)],
        "steps": [int(steps)],
        "block_length": [int(block_length)],
        "temperature": [float(temperature)],
        "cfg_scale": [float(cfg_scale)],
        "remasking": [str(remasking)],
        "mask_id": [int(mask_token_id)],
        "mask_token_id": [int(mask_token_id)],
        "codebook_size": [int(codebook_size)],
        "audio_codebook_size": [int(audio_codebook_size)],
        "timesteps": [int(timesteps)],
        "guidance_scale": [float(guidance_scale)],
        "noise_type": [str(args.noise_type)],
        "noise_schedule_name": [str(args.noise_schedule_name)],
        "noise_schedule_params": [noise_schedule_params],
        "seq_len": [int(image_token_count)],
        "condition": [str(t2s_condition)],
        "vq_model_image_path": [str(vq_image_source)],
        "vq_model_image_local_files_only": [bool(vq_image_local_only)],
        "vq_model_audio_path": [str(vq_audio_source)],
        "vq_model_audio_local_files_only": [bool(vq_audio_local_only)],
    }

    if task_name in {"t2t", "i2t", "s2t", "v2t"}:
        runtime_info["prompt_length"] = [int(len(prompt_token_ids))]
    if uncond_input_ids is not None:
        runtime_info["uncond_input_ids"] = [uncond_input_ids]
    if uncond_attention_mask is not None:
        runtime_info["uncond_attention_mask"] = [uncond_attention_mask]

    if task_name == "t2s":
        runtime_info["max_new_tokens"] = [int(t2s_token_length)]

    prompt = {
        "prompt_token_ids": [int(v) for v in prompt_token_ids],
        "additional_information": runtime_info,
        "modalities": [final_modality],
    }

    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    stage_config_path = str(Path(args.stage_config_path).expanduser())
    omni = Omni(model=model_source, stage_configs_path=stage_config_path, dtype=args.dtype)
    sampling_params_list = [
        SamplingParams(max_tokens=int(args.max_tokens_per_stage), temperature=0.0, top_p=1.0, detokenize=False)
        for _ in range(omni.num_stages)
    ]

    try:
        outputs = list(omni.generate(prompt, sampling_params_list))
    finally:
        omni.close()

    out_dir = Path(resolved_output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix.strip() or f"{task_name}_{stamp}"

    if final_modality == "text":
        text_out = extract_text_output(outputs, tokenizer=tokenizer)
        if not text_out:
            raise RuntimeError("No text output found.")
        out_path = out_dir / f"{prefix}.txt"
        out_path.write_text(text_out + "\n", encoding="utf-8")
        print(f"[end2end] text saved: {out_path}")
        print(text_out)
        return

    if final_modality == "image":
        image_out = extract_image_output(outputs)
        if image_out is None:
            raise RuntimeError("No image output found.")
        pil = tensor_to_pil_image(image_out)
        out_path = out_dir / f"{prefix}.png"
        pil.save(out_path)
        print(f"[end2end] image saved: {out_path}")
        return

    if final_modality == "audio":
        audio_out = extract_audio_output(outputs)
        if audio_out is None:
            raise RuntimeError("No audio output found.")
        wav, sr = audio_out
        out_path = out_dir / f"{prefix}.wav"
        save_audio_wav(out_path, wav, sr)
        print(f"[end2end] audio saved: {out_path} (sr={sr}, samples={wav.shape[0]})")
        return

    raise RuntimeError(f"Unsupported final modality: {final_modality}")


if __name__ == "__main__":
    main()
