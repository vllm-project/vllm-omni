#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Bootstrap & environment
# ---------------------------------------------------------------------------

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


def ensure_local_model_dir(model: str, cache_dir: Path) -> Path:
    model_path = Path(model).expanduser()
    if model_path.is_dir():
        return model_path.resolve()

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir / ".hf_home"))

    local_dir = cache_dir / sanitize_repo_id(model)
    if not local_dir.exists():
        print(f"[i2i] Downloading model to local dir: {local_dir}")
        snapshot_download(
            repo_id=model,
            local_dir=str(local_dir),
            local_dir_use_symlinks=True,
            resume_download=True,
        )
    return local_dir.resolve()


# ---------------------------------------------------------------------------
# Config resolution helpers
# ---------------------------------------------------------------------------

def resolve_prompting_defaults(dynin_config_path: str) -> tuple[str, float, str | None, bool]:
    """Resolve noise_type, cond_dropout_prob, tokenizer_path, local_files_only from DYNIN config."""
    noise_type = "mask"
    cond_dropout_prob = 0.0
    tokenizer_path: str | None = None
    model_local_files_only = False
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(dynin_config_path)
        noise_type_value = OmegaConf.select(cfg, "training.noise_type")
        if noise_type_value is not None:
            noise_type = str(noise_type_value)
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
    return noise_type, cond_dropout_prob, tokenizer_path, model_local_files_only


def resolve_i2i_runtime_defaults(
    *,
    dynin_config_path: str,
    prompt_max_text_len: int | None,
    mask_token_id: int | None,
    codebook_size: int | None,
    timesteps: int | None,
    guidance_scale: float | None,
    temperature: float | None,
    source_resolution: int | None,
    target_resolution: int | None,
) -> dict[str, Any]:
    """Resolve i2i generation defaults from CLI overrides or DYNIN config."""

    def _pick_int(cli_val: int | None, cfg_fn: Any, paths: tuple[str, ...], default: int) -> int:
        if cli_val is not None:
            return int(cli_val)
        for p in paths:
            v = cfg_fn(p)
            if v is not None:
                return int(v)
        return int(default)

    def _pick_float(cli_val: float | None, cfg_fn: Any, paths: tuple[str, ...], default: float) -> float:
        if cli_val is not None:
            return float(cli_val)
        for p in paths:
            v = cfg_fn(p)
            if v is not None:
                return float(v)
        return float(default)

    dynin_cfg = None
    cfg_select = lambda _path: None
    try:
        from omegaconf import OmegaConf
        dynin_cfg = OmegaConf.load(dynin_config_path)
        cfg_select = lambda _path: OmegaConf.select(dynin_cfg, _path)
    except Exception:
        pass

    base_resolution = _pick_int(None, cfg_select, ("dataset.params.resolution",), 336)
    resolved_src_res = _pick_int(
        source_resolution, cfg_select, ("dataset.params.i2i_source_resolution",), base_resolution,
    )
    resolved_tgt_res = _pick_int(
        target_resolution, cfg_select, ("dataset.params.i2i_target_resolution",), base_resolution,
    )
    num_vq_tokens = (resolved_tgt_res // 16) ** 2

    resolved_noise_schedule_name = "cosine"
    resolved_noise_schedule_params: dict[str, Any] = {}
    if dynin_cfg is not None:
        try:
            from omegaconf import OmegaConf
            explicit_schedule = OmegaConf.select(dynin_cfg, "mask_schedule.schedule")
            if explicit_schedule is not None:
                resolved_noise_schedule_name = str(explicit_schedule)
            else:
                legacy = OmegaConf.select(dynin_cfg, "training.mask_schedule")
                if legacy is not None:
                    resolved_noise_schedule_name = str(legacy)
            explicit_params = OmegaConf.select(dynin_cfg, "mask_schedule.params")
            if explicit_params is not None:
                if OmegaConf.is_config(explicit_params):
                    explicit_params = OmegaConf.to_container(explicit_params, resolve=True)
                if isinstance(explicit_params, dict):
                    resolved_noise_schedule_params = {str(k): v for k, v in explicit_params.items()}
        except Exception:
            pass

    return {
        "prompt_max_text_len": _pick_int(
            prompt_max_text_len, cfg_select, ("dataset.preprocessing.max_seq_length",), 128,
        ),
        "mask_token_id": _pick_int(
            mask_token_id, cfg_select, ("model.omada.mask_token_id",), 126336,
        ),
        "codebook_size": _pick_int(
            codebook_size, cfg_select, ("model.omada.codebook_size",), 8192,
        ),
        "timesteps": _pick_int(
            timesteps, cfg_select, ("training.generation_timesteps",), 18,
        ),
        "guidance_scale": _pick_float(
            guidance_scale, cfg_select, ("training.guidance_scale",), 0.0,
        ),
        "temperature": _pick_float(
            temperature, cfg_select, ("training.generation_temperature",), 1.0,
        ),
        "source_resolution": resolved_src_res,
        "target_resolution": resolved_tgt_res,
        "num_vq_tokens": num_vq_tokens,
        "noise_schedule_name": resolved_noise_schedule_name,
        "noise_schedule_params": resolved_noise_schedule_params,
    }


# ---------------------------------------------------------------------------
# VQ encoder
# ---------------------------------------------------------------------------

def resolve_vq_model_path(
    *,
    cli_vq_path: str | None,
    dynin_config_path: str,
) -> tuple[str, bool]:
    """Return (vq_model_path, local_files_only) from CLI or config."""
    if cli_vq_path:
        return cli_vq_path, True

    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(dynin_config_path)
        repo_id = OmegaConf.select(cfg, "model.vq_model_image.repo_id")
        if repo_id is not None:
            local_only = bool(OmegaConf.select(cfg, "model.omada.local_files_only") or False)
            return str(repo_id), local_only
    except Exception:
        pass
    return "showlab/magvitv2", False


def load_vq_encoder(vq_path: str, device: torch.device, local_files_only: bool = False) -> Any:
    """Load MAGVITv2 for source image VQ encoding."""
    from vllm_omni.model_executor.models.dynin_omni.models import MAGVITv2

    vq_model = MAGVITv2.from_pretrained(vq_path, local_files_only=local_files_only).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    return vq_model


def preprocess_source_image(image: Image.Image, resolution: int) -> torch.Tensor:
    """Resize (shortest-side), center-crop, normalize to [-1, 1]. Matches training image_transform."""
    w, h = image.size
    short_side = min(w, h)
    scale = resolution / short_side
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    image = image.crop((left, top, left + resolution, top + resolution))
    arr = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    tensor = (tensor - 0.5) / 0.5
    return tensor


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_uni_prompting(
    *,
    tokenizer_source: str,
    max_text_len: int,
    cond_dropout_prob: float,
    local_files_only: bool,
) -> Any:
    """Initialise UniversalPrompting with the DYNIN tokenizer."""
    from transformers import AutoTokenizer
    from vllm_omni.model_executor.models.dynin_omni.models.runtime.prompting_utils import UniversalPrompting

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
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=float(cond_dropout_prob),
        use_reserved_token=True,
    )


def build_i2i_model_inputs(
    *,
    prompt_text: str,
    input_image_tokens: torch.Tensor,
    uni_prompting: Any,
    num_vq_tokens: int,
    mask_token_id: int,
    guidance_scale: float,
    use_train_i2i_prompt: bool,
) -> dict[str, list[int] | None]:
    """Build i2i input_ids and attention_mask via UniversalPrompting.

    When use_train_i2i_prompt is True the training template is used:
        <|i2i|> <|soi|> [source] <|eoi|> text <|soi|> [masked target] <|eoi|>
    Otherwise the i2i_gen template is used:
        <|t2i|> <|soi|> [source] <|eoi|> <|sot|> text <|eot|> <|soi|> [masked target] <|eoi|>
    """
    device = input_image_tokens.device
    output_placeholder = torch.full(
        (1, num_vq_tokens), fill_value=mask_token_id, dtype=torch.long, device=device,
    )

    src_tokens = input_image_tokens
    if src_tokens.ndim == 1:
        src_tokens = src_tokens.unsqueeze(0)

    if use_train_i2i_prompt:
        labels_placeholder = torch.full(
            (1, num_vq_tokens), fill_value=-100, dtype=torch.long, device=device,
        )
        input_ids, attention_mask, _ = uni_prompting(
            ([str(prompt_text)], src_tokens, output_placeholder, labels_placeholder), "i2i",
        )
        attention_mask = attention_mask.long()

        uncond_input_ids = None
        uncond_attention_mask = None
        if float(guidance_scale) > 0:
            uncond_input_ids, uncond_attention_mask, _ = uni_prompting(
                ([""], src_tokens, output_placeholder, labels_placeholder), "i2i",
            )
            uncond_attention_mask = uncond_attention_mask.long()
    else:
        input_ids, attention_mask = uni_prompting(
            ([str(prompt_text)], src_tokens, output_placeholder), "i2i_gen",
        )
        uncond_input_ids = None
        uncond_attention_mask = None
        if float(guidance_scale) > 0:
            uncond_input_ids, uncond_attention_mask = uni_prompting(
                ([""], src_tokens, output_placeholder), "i2i_gen",
            )

    def _to_1d_list(tensor: torch.Tensor | None) -> list[int] | None:
        if tensor is None:
            return None
        t = tensor.detach().to(device="cpu", dtype=torch.long)
        if t.ndim == 2:
            t = t[0]
        return [int(v) for v in t.tolist()]

    return {
        "input_ids": _to_1d_list(input_ids),
        "attention_mask": _to_1d_list(attention_mask),
        "uncond_input_ids": _to_1d_list(uncond_input_ids),
        "uncond_attention_mask": _to_1d_list(uncond_attention_mask),
    }


def build_i2i_prompt(
    *,
    input_ids: list[int],
    attention_mask: list[int],
    uncond_input_ids: list[int] | None,
    uncond_attention_mask: list[int] | None,
    prompt_max_text_len: int,
    num_vq_tokens: int,
    mask_token_id: int,
    dynin_config_path: str,
    timesteps: int,
    guidance_scale: float,
    temperature: float,
    codebook_size: int,
    noise_schedule_name: str,
    noise_schedule_params: dict[str, Any] | None,
    noise_type: str,
    cond_dropout_prob: float,
    tokenizer_path: str | None,
    model_local_files_only: bool,
    vq_model_image_path: str | None,
    vq_model_image_local_files_only: bool | None,
    resolution: int,
) -> dict[str, Any]:
    """Package i2i inputs into the vllm_omni Omni.generate() prompt format."""
    prompt_token_ids = [int(v) for v in input_ids]
    additional_information: dict[str, Any] = {
        "task": ["i2i"],
        "detok_id": [2],
        "attention_mask": [[int(v) for v in attention_mask]],
        "dynin_config_path": [dynin_config_path],
        "timesteps": [int(timesteps)],
        "guidance_scale": [float(guidance_scale)],
        "temperature": [float(temperature)],
        "noise_schedule_name": [str(noise_schedule_name)],
        "noise_type": [str(noise_type)],
        "prompt_max_text_len": [int(prompt_max_text_len)],
        "prompting_max_text_len": [int(prompt_max_text_len)],
        "cond_dropout_prob": [float(cond_dropout_prob)],
        "prompting_cond_dropout_prob": [float(cond_dropout_prob)],
        "seq_len": [int(num_vq_tokens)],
        "mask_token_id": [int(mask_token_id)],
        "codebook_size": [int(codebook_size)],
        "resolution": [int(resolution)],
        "noise_schedule_params": [dict(noise_schedule_params or {})],
    }
    if tokenizer_path:
        additional_information["tokenizer_path"] = [str(tokenizer_path)]
    additional_information["model_local_files_only"] = [bool(model_local_files_only)]
    if uncond_input_ids is not None:
        additional_information["uncond_input_ids"] = [[int(v) for v in uncond_input_ids]]
    if uncond_attention_mask is not None:
        additional_information["uncond_attention_mask"] = [[int(v) for v in uncond_attention_mask]]
    if vq_model_image_path:
        additional_information["vq_model_image_path"] = [str(vq_model_image_path)]
    if vq_model_image_local_files_only is not None:
        additional_information["vq_model_image_local_files_only"] = [bool(vq_model_image_local_files_only)]

    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": additional_information,
        "modalities": ["image"],
    }


# ---------------------------------------------------------------------------
# Image tensor extraction & conversion (reused from t2i.py)
# ---------------------------------------------------------------------------

def extract_image_tensor(outputs: list[Any]) -> torch.Tensor | None:
    """Walk Omni stage outputs and return the first image tensor found."""
    for omni_out in outputs:
        if getattr(omni_out, "final_output_type", None) != "image":
            continue
        req_out = getattr(omni_out, "request_output", None)
        req_out_list = req_out if isinstance(req_out, list) else [req_out]
        for item in req_out_list:
            if item is None:
                continue
            mm_out = getattr(item, "multimodal_output", None) or {}
            if not mm_out and getattr(item, "outputs", None):
                completion = item.outputs[0]
                mm_out = getattr(completion, "multimodal_output", None) or {}
            if not mm_out:
                mm_out = getattr(omni_out, "multimodal_output", None) or {}
            image = mm_out.get("image", None)
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


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def parse_i2i_metadata(metadata_path: str) -> list[tuple[str, dict[str, str]]]:
    """Parse i2i edits JSON. Returns sorted (key, {id, prompt}) pairs."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        edit_infos = json.load(f)

    def _sort_key(item: tuple[str, Any]) -> int | str:
        try:
            return int(item[0])
        except ValueError:
            return item[0]

    return sorted(edit_infos.items(), key=_sort_key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DYNIN image-to-image editing using vllm_omni.")
    parser.add_argument("--model", required=True, help="HF repo id or local model dir.")
    parser.add_argument(
        "--stage-config-path", type=str,
        default=str(repo_root / "vllm_omni/model_executor/stage_configs/dynin_omni.yaml"),
        help="Path to stage config yaml.",
    )
    parser.add_argument(
        "--dynin-config-path", type=str,
        default=str(repo_root / "vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml"),
        help="Path to DYNIN config yaml.",
    )
    parser.add_argument(
        "--edit-json", type=str,
        default=str(repo_root / "examples/offline_inference/dynin_omni/data/text/i2i_edits.json"),
        help="JSON file with i2i edit entries ({key: {id, prompt}}).",
    )
    parser.add_argument(
        "--origin-img-root", type=str,
        default=str(repo_root / "examples/offline_inference/dynin_omni/data/image"),
        help="Root folder for source images.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="/tmp/dynin_i2i_outputs",
        help="Output directory to save edited images.",
    )
    parser.add_argument(
        "--model-cache-dir", type=str, default="/tmp/dynin_localized_models",
        help="Cache dir used when --model is HF repo id.",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=0,
        help="0 means all entries from metadata.",
    )
    parser.add_argument(
        "--source-resolution", type=int, default=None,
        help="Source image resolution. Default: dataset.params.i2i_source_resolution from config.",
    )
    parser.add_argument(
        "--target-resolution", type=int, default=None,
        help="Target image resolution. Default: dataset.params.i2i_target_resolution from config.",
    )
    parser.add_argument(
        "--prompt-max-text-len", type=int, default=None,
        help="UniversalPrompting max_text_len. Default: from config.",
    )
    parser.add_argument(
        "--mask-token-id", type=int, default=None,
        help="Mask token id. Default: model.omada.mask_token_id (or 126336).",
    )
    parser.add_argument(
        "--codebook-size", type=int, default=None,
        help="Image codebook size. Default: model.omada.codebook_size from config.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=128,
        help="Generation timesteps. Default: training.generation_timesteps from config.",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=3.5,
        help="CFG guidance scale. Default: training.guidance_scale from config.",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Generation temperature. Default: training.generation_temperature from config.",
    )
    parser.add_argument(
        "--use-train-i2i-prompt", action="store_true",
        help="Use training i2i prompt template (<|i2i|> ...). This is the default.",
    )
    parser.add_argument(
        "--no-use-train-i2i-prompt", dest="use_train_i2i_prompt", action="store_false",
        help="Use i2i_gen prompt template (<|t2i|> ...).",
    )
    parser.set_defaults(use_train_i2i_prompt=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-tokens-per-stage", type=int, default=1)
    parser.add_argument(
        "--vq-model-image-path", type=str, default="",
        help="Local directory path for MAGVITv2 weights.",
    )
    parser.add_argument(
        "--vq-model-image-local-files-only",
        action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--disable-hf-xet", action=argparse.BooleanOptionalAction, default=True,
        help="Set HF_HUB_DISABLE_XET=1 to avoid CAS/xet download path.",
    )
    parser.add_argument(
        "--dtype", type=str, default=None,
        help="Optional vLLM dtype override (e.g., float16, bfloat16, float32, auto).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    repo_root = bootstrap_repo_path()
    ensure_safe_import_for_vllm()
    args = parse_args(repo_root)

    dynin_config_path = str(Path(args.dynin_config_path).expanduser())
    runtime = resolve_i2i_runtime_defaults(
        dynin_config_path=dynin_config_path,
        prompt_max_text_len=args.prompt_max_text_len,
        mask_token_id=args.mask_token_id,
        codebook_size=args.codebook_size,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        source_resolution=args.source_resolution,
        target_resolution=args.target_resolution,
    )
    prompt_max_text_len: int = int(runtime["prompt_max_text_len"])
    mask_token_id: int = int(runtime["mask_token_id"])
    codebook_size: int = int(runtime["codebook_size"])
    timesteps: int = int(runtime["timesteps"])
    guidance_scale: float = float(runtime["guidance_scale"])
    temperature: float = float(runtime["temperature"])
    src_resolution: int = int(runtime["source_resolution"])
    tgt_resolution: int = int(runtime["target_resolution"])
    num_vq_tokens: int = int(runtime["num_vq_tokens"])
    noise_schedule_name: str = str(runtime["noise_schedule_name"])
    noise_schedule_params: dict[str, Any] = runtime.get("noise_schedule_params", {})
    if not isinstance(noise_schedule_params, dict):
        noise_schedule_params = {}

    noise_type, cond_dropout_prob, tokenizer_path, model_local_files_only = resolve_prompting_defaults(
        dynin_config_path
    )
    # Disable prompt dropout for deterministic inference with training template.
    if args.use_train_i2i_prompt:
        cond_dropout_prob = 0.0

    os.environ["DYNIN_CONFIG_PATH"] = dynin_config_path
    if args.disable_hf_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # -- metadata -----------------------------------------------------------
    items = parse_i2i_metadata(args.edit_json)
    if args.max_prompts > 0:
        items = items[: args.max_prompts]

    # -- model / tokenizer --------------------------------------------------
    model_dir = ensure_local_model_dir(args.model, cache_dir=Path(args.model_cache_dir))
    tokenizer_source = str(tokenizer_path) if tokenizer_path else str(model_dir)
    uni_prompting = load_uni_prompting(
        tokenizer_source=tokenizer_source,
        max_text_len=prompt_max_text_len,
        cond_dropout_prob=cond_dropout_prob,
        local_files_only=model_local_files_only,
    )

    # -- VQ encoder for source images ---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_path, vq_local_only = resolve_vq_model_path(
        cli_vq_path=args.vq_model_image_path.strip() or None,
        dynin_config_path=dynin_config_path,
    )
    print(f"[i2i] Loading VQ encoder from: {vq_path}")
    vq_model = load_vq_encoder(vq_path, device, local_files_only=vq_local_only)

    # -- output dir ---------------------------------------------------------
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- vllm_omni engine ---------------------------------------------------
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=str(model_dir), stage_configs_path=args.stage_config_path, dtype=args.dtype)
    sampling_params_list = [
        SamplingParams(max_tokens=args.max_tokens_per_stage, temperature=0.0, top_p=1.0, detokenize=False)
        for _ in range(len(omni.stage_list))
    ]

    # -- generation loop ----------------------------------------------------
    result_rows: list[dict[str, Any]] = []
    try:
        for idx, (key, item) in enumerate(items):
            out_path = output_dir / f"{key}.png"
            if args.skip_existing and out_path.is_file():
                print(f"[{idx + 1}/{len(items)}] skipping existing: {out_path}")
                continue

            origin_path = os.path.join(args.origin_img_root, item["id"])
            if not os.path.isfile(origin_path):
                print(f"[{idx + 1}/{len(items)}] missing source image: {origin_path}")
                continue
            try:
                src_img = Image.open(origin_path).convert("RGB")
            except Exception as exc:
                print(f"[{idx + 1}/{len(items)}] failed to open {origin_path}: {exc}")
                continue

            prompt_text = str(item["prompt"])

            # VQ-encode source image
            src_tensor = preprocess_source_image(src_img, resolution=src_resolution)
            src_tensor = src_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                src_image_tokens = vq_model.get_code(src_tensor) + len(uni_prompting.text_tokenizer)

            # Build i2i inputs via UniversalPrompting
            model_inputs = build_i2i_model_inputs(
                prompt_text=prompt_text,
                input_image_tokens=src_image_tokens,
                uni_prompting=uni_prompting,
                num_vq_tokens=num_vq_tokens,
                mask_token_id=mask_token_id,
                guidance_scale=guidance_scale,
                use_train_i2i_prompt=args.use_train_i2i_prompt,
            )
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            if input_ids is None or attention_mask is None:
                raise RuntimeError(f"Failed to build i2i model inputs for entry '{key}'")

            # Package for vllm_omni (resolution=512 matches reference i2i_generate default)
            prompt = build_i2i_prompt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                uncond_input_ids=model_inputs["uncond_input_ids"],
                uncond_attention_mask=model_inputs["uncond_attention_mask"],
                prompt_max_text_len=prompt_max_text_len,
                num_vq_tokens=num_vq_tokens,
                mask_token_id=mask_token_id,
                dynin_config_path=dynin_config_path,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                temperature=temperature,
                codebook_size=codebook_size,
                noise_schedule_name=noise_schedule_name,
                noise_schedule_params=noise_schedule_params,
                noise_type=noise_type,
                cond_dropout_prob=cond_dropout_prob,
                tokenizer_path=tokenizer_source,
                model_local_files_only=model_local_files_only,
                vq_model_image_path=args.vq_model_image_path.strip() or None,
                vq_model_image_local_files_only=args.vq_model_image_local_files_only,
                resolution=512,
            )

            outputs = list(omni.generate(prompt, sampling_params_list))
            image_tensor = extract_image_tensor(outputs)
            if image_tensor is None:
                raise RuntimeError(f"No image output found for entry '{key}'")

            image = tensor_to_pil_image(image_tensor)
            image.save(out_path)

            result_rows.append({
                "id": key,
                "source": item["id"],
                "prompt": prompt_text,
                "image_path": str(out_path),
            })
            print(f"[{idx + 1}/{len(items)}] saved: {out_path}")
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
python /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/i2i.py \
  --model snu-aidas/Dynin-Omni_vllm \
  --dynin-config-path /home/kdg6245/vllm-omni/vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml \
  --edit-json /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/data/text/i2i_edits.json \
  --origin-img-root /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/data/image \
  --output-dir /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/results/i2i_from_vllm
"""
