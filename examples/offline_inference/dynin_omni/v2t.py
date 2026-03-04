#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
import unicodedata
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_V2T_QUESTIONS = [
    "Please describe this video in detail.",
    "What is happening in this video?",
    "Summarize the events shown in the video.",
    "What do you observe from this video?",
]


_WS_RE = re.compile(r"\s+")
_TOKENIZER_MARKER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
)
_VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


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


def sanitize_file_stem(value: str, fallback: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    text = text.strip("._")
    return text or fallback


def ensure_local_model_dir(model: str, cache_dir: Path) -> Path:
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        return model_path.resolve()

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / ".hf_home"))

    local_dir = cache_dir / sanitize_repo_id(model)
    if not local_dir.exists():
        print(f"[v2t] Downloading model to local dir: {local_dir}")
        snapshot_download(
            repo_id=model,
            local_dir=str(local_dir),
            local_dir_use_symlinks=True,
            resume_download=True,
        )
    return local_dir.resolve()


def has_local_tokenizer_assets(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / name).exists() for name in _TOKENIZER_MARKER_FILES)


def resolve_tokenizer_source_with_local_priority(
    *,
    model_dir: Path,
    configured_tokenizer_path: str | None,
) -> str:
    if has_local_tokenizer_assets(model_dir):
        print(f"[v2t] Using tokenizer from local model dir: {model_dir}")
        return str(model_dir)

    if configured_tokenizer_path:
        configured_path = Path(configured_tokenizer_path).expanduser()
        if configured_path.is_dir():
            resolved = configured_path.resolve()
            print(f"[v2t] Using tokenizer from configured local path: {resolved}")
            return str(resolved)

        print(
            "[v2t] Local tokenizer files were not found in model dir; "
            f"falling back to configured tokenizer source: {configured_tokenizer_path}"
        )
        return str(configured_tokenizer_path)

    print(
        "[v2t] Local tokenizer markers were not found; "
        f"falling back to model dir anyway: {model_dir}"
    )
    return str(model_dir)


def _resolve_path_value(value: str, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.resolve()


def _get_video_field(video_entry: Any, key: str) -> Any:
    if video_entry is None:
        return None
    if isinstance(video_entry, dict):
        return video_entry.get(key)
    try:
        return video_entry[key]
    except Exception:
        return getattr(video_entry, key, None)


def resolve_video_path_from_row(*, row: dict[str, Any], base_dir: Path) -> Path:
    candidates = [
        row.get("video_path"),
        row.get("video_file"),
        row.get("path"),
        row.get("file"),
    ]
    video_entry = row.get("video")
    if isinstance(video_entry, str):
        candidates.append(video_entry)
    else:
        candidates.extend(
            [
                _get_video_field(video_entry, "path"),
                _get_video_field(video_entry, "video_path"),
                _get_video_field(video_entry, "file"),
            ]
        )

    missing_paths: list[Path] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        resolved = _resolve_path_value(candidate.strip(), base_dir=base_dir)
        if resolved.exists():
            return resolved
        missing_paths.append(resolved)

    if missing_paths:
        first = missing_paths[0]
        raise FileNotFoundError(
            f"video file not found: {first} (checked {len(missing_paths)} candidate path(s))"
        )
    raise ValueError(
        "metadata row has no usable video source. "
        "Expected one of: video_path/video_file/path/file/video.path"
    )


def parse_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("id")
            if sample_id is None:
                sample_id = row.get("sample_id")
            if sample_id is None:
                sample_id = f"sample-{line_idx:05d}"
            sample_id = str(sample_id)

            video_path = resolve_video_path_from_row(row=row, base_dir=metadata_path.parent)

            question = row.get("question")
            if question is None:
                question = row.get("prompt")
            if question is None:
                question = row.get("instruction")
            if question is not None:
                question = str(question)

            gt_text = row.get("text")
            if gt_text is None:
                gt_text = row.get("gt_text")
            if gt_text is None:
                gt_text = row.get("answer")
            if gt_text is not None:
                gt_text = str(gt_text)

            rows.append(
                {
                    "id": sample_id,
                    "video_path": str(video_path),
                    "question": question,
                    "gt_text": gt_text,
                }
            )
    if not rows:
        raise ValueError(f"metadata is empty: {metadata_path}")
    return rows


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("<|endoftext|>", "")
    text = text.lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
    text = "".join(out)
    text = _WS_RE.sub(" ", text).strip()
    return text


def resolve_v2t_runtime_defaults(
    *,
    dynin_config_path: str,
    prompt_max_text_len: int | None,
    v2t_max_new_tokens: int | None,
    v2t_steps: int | None,
    v2t_block_length: int | None,
    v2t_temperature: float | None,
    v2t_cfg_scale: float | None,
    mask_token_id: int | None,
    codebook_size: int | None,
    image_resolution: int | None,
    num_frames: int | None,
    sample_method: str | None,
) -> dict[str, Any]:
    def _pick_int(
        cli_value: int | None,
        cfg: Any,
        paths: tuple[str, ...],
        default: int,
    ) -> int:
        if cli_value is not None:
            return int(cli_value)
        for path in paths:
            value = cfg(path)
            if value is not None:
                return int(value)
        return int(default)

    def _pick_float(
        cli_value: float | None,
        cfg: Any,
        paths: tuple[str, ...],
        default: float,
    ) -> float:
        if cli_value is not None:
            return float(cli_value)
        for path in paths:
            value = cfg(path)
            if value is not None:
                return float(value)
        return float(default)

    cfg_select = lambda _path: None
    try:
        from omegaconf import OmegaConf

        dynin_cfg = OmegaConf.load(dynin_config_path)
        cfg_select = lambda _path: OmegaConf.select(dynin_cfg, _path)
    except Exception:
        pass

    resolved_sample_method = sample_method
    if not resolved_sample_method:
        resolved_sample_method = cfg_select("mmu_v2t_sample_method")
    if not resolved_sample_method:
        resolved_sample_method = cfg_select("dataset.params.video_caption_dataset.sample_method")
    if not resolved_sample_method:
        resolved_sample_method = "uniform_sequential"

    return {
        "prompt_max_text_len": _pick_int(
            prompt_max_text_len,
            cfg_select,
            ("dataset.preprocessing.max_seq_length_text", "dataset.preprocessing.max_seq_length"),
            1024,
        ),
        "v2t_max_new_tokens": _pick_int(
            v2t_max_new_tokens,
            cfg_select,
            ("mmu_v2t_max_new_tokens",),
            128,
        ),
        "v2t_steps": _pick_int(
            v2t_steps,
            cfg_select,
            ("mmu_v2t_steps",),
            128,
        ),
        "v2t_block_length": _pick_int(
            v2t_block_length,
            cfg_select,
            ("mmu_v2t_block_length",),
            128,
        ),
        "v2t_temperature": _pick_float(
            v2t_temperature,
            cfg_select,
            ("training.generation_temperature",),
            0.0,
        ),
        "v2t_cfg_scale": _pick_float(
            v2t_cfg_scale,
            cfg_select,
            ("training.guidance_scale",),
            0.0,
        ),
        "mask_token_id": _pick_int(
            mask_token_id,
            cfg_select,
            ("model.omada.mask_token_id",),
            126336,
        ),
        "codebook_size": _pick_int(
            codebook_size,
            cfg_select,
            ("model.omada.codebook_size",),
            8192,
        ),
        "image_resolution": _pick_int(
            image_resolution,
            cfg_select,
            ("dataset.params.resolution",),
            336,
        ),
        "num_frames": _pick_int(
            num_frames,
            cfg_select,
            ("mmu_v2t_num_frames", "dataset.params.video_caption_dataset.num_frames"),
            8,
        ),
        "sample_method": str(resolved_sample_method),
    }


def resolve_prompting_defaults(
    dynin_config_path: str,
) -> tuple[float, str | None, bool]:
    cond_dropout_prob = 0.0
    tokenizer_path: str | None = None
    model_local_files_only = False
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(dynin_config_path)
        cond_dropout_value = OmegaConf.select(cfg, "training.cond_dropout_prob")
        if cond_dropout_value is not None:
            cond_dropout_prob = float(cond_dropout_value)
        tokenizer_path_value = OmegaConf.select(cfg, "model.omada.tokenizer_path")
        if tokenizer_path_value is not None:
            tokenizer_path = str(tokenizer_path_value)
        model_local_only_value = OmegaConf.select(cfg, "model.omada.local_files_only")
        if model_local_only_value is not None:
            model_local_files_only = bool(model_local_only_value)
    except Exception:
        pass
    return cond_dropout_prob, tokenizer_path, model_local_files_only


def resolve_vq_model_path(
    *,
    cli_vq_path: str | None,
    cli_local_files_only: bool | None,
    dynin_config_path: str,
    default_model_local_files_only: bool,
) -> tuple[str, bool]:
    source = "showlab/magvitv2"
    local_files_only = bool(default_model_local_files_only)
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(dynin_config_path)
        repo_id = OmegaConf.select(cfg, "model.vq_model_image.repo_id")
        if repo_id is not None:
            source = str(repo_id)
        local_only_value = OmegaConf.select(cfg, "model.vq_model_image.local_files_only")
        if local_only_value is not None:
            local_files_only = bool(local_only_value)
    except Exception:
        pass

    if cli_vq_path:
        source = str(cli_vq_path)
    if cli_local_files_only is not None:
        local_files_only = bool(cli_local_files_only)
    elif Path(source).expanduser().is_dir():
        local_files_only = True
    return source, local_files_only


def load_uni_prompting(
    *,
    tokenizer_source: str,
    max_text_len: int,
    cond_dropout_prob: float,
    local_files_only: bool,
) -> Any:
    from transformers import AutoTokenizer
    from vllm_omni.model_executor.models.dynin_omni.models.runtime.prompting_utils import (
        UniversalPrompting,
    )

    load_kwargs = {
        "padding_side": "left",
        "trust_remote_code": True,
        "local_files_only": bool(local_files_only),
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **load_kwargs)
    except TypeError:
        load_kwargs.pop("local_files_only", None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **load_kwargs)

    return UniversalPrompting(
        tokenizer,
        max_text_len=int(max_text_len),
        special_tokens=(
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
        ),
        ignore_id=-100,
        cond_dropout_prob=float(cond_dropout_prob),
        use_reserved_token=True,
    )


def load_vq_encoder(vq_path: str, device: torch.device, local_files_only: bool = False) -> Any:
    from vllm_omni.model_executor.models.dynin_omni.models import MAGVITv2

    vq_model = MAGVITv2.from_pretrained(vq_path, local_files_only=local_files_only).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    return vq_model


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
    tensor = (tensor - 0.5) / 0.5
    return tensor


def _load_video_frames(video_path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        for frame in iio.imiter(str(video_path)):
            if frame is None:
                continue
            arr = np.asarray(frame)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            frames.append(Image.fromarray(arr.astype(np.uint8)))
    except Exception as exc:
        raise RuntimeError(f"Failed to decode video {video_path}: {exc}") from exc
    return frames


def load_video_token_ids(
    *,
    video_path: Path,
    vq_model: Any,
    device: torch.device,
    resolution: int,
    num_frames: int,
    sample_method: str,
) -> torch.Tensor:
    frames = _load_video_frames(video_path)
    total_frames = len(frames)
    if total_frames == 0:
        raise ValueError(f"Video has no decodable frames: {video_path}")
    if total_frames < num_frames:
        raise ValueError(
            f"Video {video_path} has {total_frames} frame(s), required >= {num_frames}."
        )

    if sample_method not in {"uniform", "uniform_sequential"}:
        raise ValueError(f"Sampling method {sample_method} not supported.")
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    token_list: list[torch.Tensor] = []
    for idx in indices:
        frame = preprocess_image(frames[int(idx)], resolution=resolution).to(device)
        frame = frame.unsqueeze(0)
        with torch.no_grad():
            token_list.append(vq_model.get_code(frame))
    return torch.cat(token_list, dim=1)


def build_v2t_chat_prompt(question: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


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


def build_v2t_prompt(
    *,
    prompt_token_ids: list[int],
    dynin_config_path: str,
    task: str,
    detok_id: int,
    prompt_max_text_len: int,
    cond_dropout_prob: float,
    max_new_tokens: int,
    steps: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_token_id: int,
    tokenizer_path: str | None,
    model_local_files_only: bool,
) -> dict[str, Any]:
    additional_information: dict[str, Any] = {
        "task": [str(task)],
        "detok_id": [int(detok_id)],
        "prompt_length": [int(len(prompt_token_ids))],
        "dynin_config_path": [str(dynin_config_path)],
        "max_new_tokens": [int(max_new_tokens)],
        "steps": [int(steps)],
        "block_length": [int(block_length)],
        "temperature": [float(temperature)],
        "cfg_scale": [float(cfg_scale)],
        "remasking": [str(remasking)],
        "mask_id": [int(mask_token_id)],
        "prompt_max_text_len": [int(prompt_max_text_len)],
        "prompting_max_text_len": [int(prompt_max_text_len)],
        "cond_dropout_prob": [float(cond_dropout_prob)],
        "prompting_cond_dropout_prob": [float(cond_dropout_prob)],
        "model_local_files_only": [bool(model_local_files_only)],
        "attention_mask": [[1] * len(prompt_token_ids)],
    }
    if tokenizer_path:
        additional_information["tokenizer_path"] = [str(tokenizer_path)]

    return {
        "prompt_token_ids": [int(v) for v in prompt_token_ids],
        "additional_information": additional_information,
        "modalities": ["text"],
    }


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
    for omni_out in outputs:
        if getattr(omni_out, "final_output_type", None) != "text":
            continue
        req_out = getattr(omni_out, "request_output", None)
        req_out_list = req_out if isinstance(req_out, list) else [req_out]
        for item in req_out_list:
            if item is None:
                continue
            mm_out = getattr(item, "multimodal_output", None) or {}
            completion = None
            if not mm_out and getattr(item, "outputs", None):
                completion = item.outputs[0]
                mm_out = getattr(completion, "multimodal_output", None) or {}
            if not mm_out:
                mm_out = getattr(omni_out, "multimodal_output", None) or {}

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

            if completion is None and getattr(item, "outputs", None):
                completion = item.outputs[0]
            fallback = getattr(completion, "text", None) if completion is not None else None
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
    return ""


def validate_generation_args(
    *,
    max_new_tokens: int,
    steps: int,
    block_length: int,
) -> None:
    if max_new_tokens <= 0:
        raise ValueError("--v2t-max-new-tokens must be > 0.")
    if block_length <= 0:
        raise ValueError("--v2t-block-length must be > 0.")
    if steps <= 0:
        raise ValueError("--v2t-steps must be > 0.")
    if max_new_tokens % block_length != 0:
        raise ValueError(
            f"v2t requires max_new_tokens % block_length == 0, got {max_new_tokens} % {block_length}"
        )
    num_blocks = max_new_tokens // block_length
    if num_blocks <= 0:
        raise ValueError("Invalid number of generation blocks.")
    if steps % num_blocks != 0:
        raise ValueError(
            "v2t requires steps % (max_new_tokens // block_length) == 0, "
            f"got steps={steps}, max_new_tokens={max_new_tokens}, block_length={block_length}"
        )


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DYNIN video-to-text example using vllm_omni.")
    parser.add_argument("--model", required=True, help="HF repo id or local model dir.")
    parser.add_argument(
        "--stage-config-path",
        type=str,
        default=str(repo_root / "vllm_omni/model_executor/stage_configs/dynin_omni.yaml"),
        help="Path to stage config yaml.",
    )
    parser.add_argument(
        "--dynin-config-path",
        type=str,
        default=str(repo_root / "vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml"),
        help="Path to DYNIN config yaml.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="",
        help="Optional jsonl path containing video rows.",
    )
    parser.add_argument(
        "--video-path",
        action="append",
        default=[],
        help="Video path to query. Repeat --video-path for multiple samples.",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="",
        help="Optional directory of videos. All supported video files are added.",
    )
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Question for the video. Repeatable; if multiple, rotates by sample index.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/dynin_v2t_outputs",
        help="Output directory to save transcripts.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/tmp/dynin_localized_models",
        help="Cache dir used when --model is HF repo id.",
    )
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all inputs.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic question selection.")

    parser.add_argument(
        "--task",
        type=str,
        default="v2t",
        choices=["v2t"],
        help="DYNIN token2text generation task used for v2t.",
    )
    parser.add_argument("--detok-id", type=int, default=0, help="Detokenizer id (text=0).")
    parser.add_argument("--prompt-max-text-len", type=int, default=None)
    parser.add_argument("--v2t-max-new-tokens", type=int, default=256, help="Number of generated text tokens.")
    parser.add_argument("--v2t-steps", type=int, default=256, help="Number of v2t generation steps.")
    parser.add_argument("--v2t-block-length", type=int, default=2, help="Block length for v2t generation.")
    parser.add_argument("--v2t-temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--v2t-cfg-scale", type=float, default=0.0, help="Classifier-free guidance scale.")
    parser.add_argument("--v2t-remasking", type=str, default="low_confidence", help="Remasking strategy.")
    parser.add_argument("--mask-token-id", type=int, default=None, help="Mask token id.")
    parser.add_argument("--codebook-size", type=int, default=None, help="Image codebook size in DYNIN config.")
    parser.add_argument("--image-resolution", type=int, default=224, help="Frame encode resolution.")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to sample.")
    parser.add_argument(
        "--sample-method",
        type=str,
        default=None,
        choices=["uniform", "uniform_sequential"],
        help="Frame sampling method.",
    )
    parser.add_argument("--max-tokens-per-stage", type=int, default=1)

    parser.add_argument(
        "--vq-model-image-path",
        type=str,
        default="",
        help="MAGVITv2 model path or repo id.",
    )
    parser.add_argument(
        "--vq-model-image-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override VQ image local_files_only.",
    )
    parser.add_argument(
        "--disable-hf-xet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set HF_HUB_DISABLE_XET=1 to avoid CAS/xet download path.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Optional vLLM dtype override (e.g., float16, bfloat16, float32, auto).",
    )
    return parser.parse_args()


def main() -> None:
    repo_root = bootstrap_repo_path()
    ensure_safe_import_for_vllm()
    args = parse_args(repo_root)

    dynin_config_path = str(Path(args.dynin_config_path).expanduser())
    runtime_defaults = resolve_v2t_runtime_defaults(
        dynin_config_path=dynin_config_path,
        prompt_max_text_len=args.prompt_max_text_len,
        v2t_max_new_tokens=args.v2t_max_new_tokens,
        v2t_steps=args.v2t_steps,
        v2t_block_length=args.v2t_block_length,
        v2t_temperature=args.v2t_temperature,
        v2t_cfg_scale=args.v2t_cfg_scale,
        mask_token_id=args.mask_token_id,
        codebook_size=args.codebook_size,
        image_resolution=args.image_resolution,
        num_frames=args.num_frames,
        sample_method=args.sample_method,
    )

    prompt_max_text_len = int(runtime_defaults["prompt_max_text_len"])
    v2t_max_new_tokens = int(runtime_defaults["v2t_max_new_tokens"])
    v2t_steps = int(runtime_defaults["v2t_steps"])
    v2t_block_length = int(runtime_defaults["v2t_block_length"])
    v2t_temperature = float(runtime_defaults["v2t_temperature"])
    v2t_cfg_scale = float(runtime_defaults["v2t_cfg_scale"])
    mask_token_id = int(runtime_defaults["mask_token_id"])
    codebook_size = int(runtime_defaults["codebook_size"])
    image_resolution = int(runtime_defaults["image_resolution"])
    num_frames = int(runtime_defaults["num_frames"])
    sample_method = str(runtime_defaults["sample_method"])

    validate_generation_args(
        max_new_tokens=v2t_max_new_tokens,
        steps=v2t_steps,
        block_length=v2t_block_length,
    )

    cond_dropout_prob, tokenizer_path, model_local_files_only = resolve_prompting_defaults(
        dynin_config_path
    )

    os.environ["DYNIN_CONFIG_PATH"] = dynin_config_path
    if args.disable_hf_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    records: list[dict[str, Any]] = []
    if args.metadata_path:
        metadata_path = Path(args.metadata_path).expanduser()
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata path not found: {metadata_path}")
        records.extend(parse_metadata(metadata_path))

    for idx, video_path_raw in enumerate(args.video_path):
        resolved_video_path = _resolve_path_value(video_path_raw, base_dir=Path.cwd())
        if not resolved_video_path.exists():
            raise FileNotFoundError(f"video path not found: {resolved_video_path}")
        records.append(
            {
                "id": f"cli-{idx:05d}",
                "video_path": str(resolved_video_path),
                "question": None,
                "gt_text": None,
            }
        )

    if args.video_root:
        video_root = _resolve_path_value(args.video_root, base_dir=Path.cwd())
        if not video_root.is_dir():
            raise NotADirectoryError(f"video root is not a directory: {video_root}")
        for file_name in sorted(os.listdir(video_root)):
            if not file_name.lower().endswith(_VIDEO_SUFFIXES):
                continue
            full_path = (video_root / file_name).resolve()
            records.append(
                {
                    "id": f"root-{file_name}",
                    "video_path": str(full_path),
                    "question": None,
                    "gt_text": None,
                }
            )

    if not records:
        raise ValueError("No inputs provided. Use --metadata-path or --video-path or --video-root.")
    if args.max_prompts > 0:
        records = records[: args.max_prompts]

    model_dir = ensure_local_model_dir(args.model, cache_dir=Path(args.model_cache_dir))
    tokenizer_source = resolve_tokenizer_source_with_local_priority(
        model_dir=model_dir,
        configured_tokenizer_path=tokenizer_path,
    )
    effective_model_local_files_only = bool(
        model_local_files_only or Path(tokenizer_source).expanduser().is_dir()
    )
    if effective_model_local_files_only and not model_local_files_only:
        print("[v2t] Enabling local_files_only because tokenizer source is local.")

    uni_prompting = load_uni_prompting(
        tokenizer_source=tokenizer_source,
        max_text_len=prompt_max_text_len,
        cond_dropout_prob=cond_dropout_prob,
        local_files_only=effective_model_local_files_only,
    )
    tokenizer = uni_prompting.text_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_source, vq_local_only = resolve_vq_model_path(
        cli_vq_path=args.vq_model_image_path.strip() or None,
        cli_local_files_only=args.vq_model_image_local_files_only,
        dynin_config_path=dynin_config_path,
        default_model_local_files_only=effective_model_local_files_only,
    )
    print(
        f"[v2t] Loading image tokenizer from: {vq_source} "
        f"(local_files_only={vq_local_only})"
    )
    vq_model = load_vq_encoder(vq_source, device=device, local_files_only=vq_local_only)

    image_token_offset = len(uni_prompting.text_tokenizer)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=str(model_dir), stage_configs_path=args.stage_config_path, dtype=args.dtype)
    sampling_params_list = [
        SamplingParams(max_tokens=args.max_tokens_per_stage, temperature=0.0, top_p=1.0, detokenize=False)
        for _ in range(len(omni.stage_list))
    ]

    fixed_questions = [str(q).strip() for q in args.question if str(q).strip()]
    result_rows: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(records):
            sample_id_raw = str(row.get("id", f"sample-{idx:05d}"))
            sample_id = sanitize_file_stem(sample_id_raw, fallback=f"sample-{idx:05d}")
            video_path = Path(str(row["video_path"])).expanduser().resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"video file not found: {video_path}")

            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                if fixed_questions:
                    question = fixed_questions[idx % len(fixed_questions)]
                else:
                    question = DEFAULT_V2T_QUESTIONS[(int(args.seed) + idx) % len(DEFAULT_V2T_QUESTIONS)]

            video_token_ids = load_video_token_ids(
                video_path=video_path,
                vq_model=vq_model,
                device=device,
                resolution=image_resolution,
                num_frames=num_frames,
                sample_method=sample_method,
            )

            prompt_token_ids, prompt_text = build_v2t_input_ids(
                video_token_ids=video_token_ids,
                tokenizer=tokenizer,
                uni_prompting=uni_prompting,
                question=question,
                image_token_offset=image_token_offset,
            )

            prompt = build_v2t_prompt(
                prompt_token_ids=prompt_token_ids,
                dynin_config_path=dynin_config_path,
                task=args.task,
                detok_id=args.detok_id,
                prompt_max_text_len=prompt_max_text_len,
                cond_dropout_prob=cond_dropout_prob,
                max_new_tokens=v2t_max_new_tokens,
                steps=v2t_steps,
                block_length=v2t_block_length,
                temperature=v2t_temperature,
                cfg_scale=v2t_cfg_scale,
                remasking=args.v2t_remasking,
                mask_token_id=mask_token_id,
                tokenizer_path=tokenizer_source,
                model_local_files_only=effective_model_local_files_only,
            )

            outputs = list(omni.generate(prompt, sampling_params_list))
            decoded_text = extract_text_output(outputs, tokenizer=tokenizer)
            decoded_text_norm = normalize_text(decoded_text)

            row_out = {
                "id": sample_id,
                "video_path": str(video_path),
                "question": question,
                "prompt": prompt_text,
                "decoded_text": decoded_text,
                "decoded_text_norm": decoded_text_norm,
                "num_frames": num_frames,
                "sample_method": sample_method,
            }
            gt_text = row.get("gt_text")
            if isinstance(gt_text, str):
                row_out["gt_text"] = gt_text
            result_rows.append(row_out)
            print(f"[{idx + 1}/{len(records)}] {sample_id}: {decoded_text_norm}")
    finally:
        omni.close()

    manifest_path = output_dir / "results.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

"""
example usage:
python /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/v2t.py \
  --model snu-aidas/Dynin-Omni_vllm \
  --dynin-config-path /home/kdg6245/vllm-omni/vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml \
  --video-path /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/data/video/baseball.mp4 \
  --question "Describe this video in detail." \
  --output-dir /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/results/v2t_from_vllm
"""
