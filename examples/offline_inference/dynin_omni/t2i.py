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
        print(f"[t2i] Downloading model to local dir: {local_dir}")
        snapshot_download(
            repo_id=model,
            local_dir=str(local_dir),
            local_dir_use_symlinks=True,
            resume_download=True,
        )
    return local_dir.resolve()


def parse_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"metadata line {line_idx} has no valid 'prompt'")
            if "id" not in row:
                row["id"] = f"sample-{line_idx:05d}"
            rows.append(row)
    if not rows:
        raise ValueError(f"metadata is empty: {metadata_path}")
    return rows


def resolve_prompt_max_text_len(prompt_max_text_len: int | None, dynin_config_path: str) -> int:
    if prompt_max_text_len is not None and int(prompt_max_text_len) > 0:
        return int(prompt_max_text_len)
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(dynin_config_path)
        value = OmegaConf.select(cfg, "dataset.preprocessing.max_seq_length")
        if value is not None:
            return int(value)
    except Exception:
        pass
    return 128


def resolve_prompting_defaults(dynin_config_path: str) -> tuple[str, float, str | None, bool]:
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


def resolve_t2i_runtime_defaults(
    *,
    dynin_config_path: str,
    prompt_max_text_len: int | None,
    image_token_count: int | None,
    mask_token_id: int | None,
    codebook_size: int | None,
    timesteps: int | None,
    guidance_scale: float | None,
    temperature: float | None,
) -> dict[str, Any]:
    def _pick_int(cli_value: int | None, cfg: Any, paths: tuple[str, ...], default: int) -> int:
        if cli_value is not None:
            return int(cli_value)
        for path in paths:
            value = cfg(path)
            if value is not None:
                return int(value)
        return int(default)

    def _pick_float(cli_value: float | None, cfg: Any, paths: tuple[str, ...], default: float) -> float:
        if cli_value is not None:
            return float(cli_value)
        for path in paths:
            value = cfg(path)
            if value is not None:
                return float(value)
        return float(default)

    dynin_cfg: Any | None = None

    def cfg_select(_path: str) -> Any:
        return None

    try:
        from omegaconf import OmegaConf

        dynin_cfg = OmegaConf.load(dynin_config_path)

        def cfg_select(_path: str) -> Any:
            return OmegaConf.select(dynin_cfg, _path)

    except Exception:
        pass

    resolved_prompt_max_text_len = _pick_int(
        prompt_max_text_len,
        cfg_select,
        ("dataset.preprocessing.max_seq_length",),
        128,
    )
    resolved_image_token_count = _pick_int(
        image_token_count,
        cfg_select,
        ("model.omada.num_vq_tokens",),
        1024,
    )
    resolved_mask_token_id = _pick_int(
        mask_token_id,
        cfg_select,
        ("model.omada.mask_token_id",),
        126336,
    )
    resolved_codebook_size = _pick_int(
        codebook_size,
        cfg_select,
        ("model.omada.codebook_size",),
        8192,
    )
    resolved_timesteps = _pick_int(
        timesteps,
        cfg_select,
        ("training.generation_timesteps",),
        18,
    )
    resolved_guidance_scale = _pick_float(
        guidance_scale,
        cfg_select,
        ("training.guidance_scale",),
        0.0,
    )
    resolved_temperature = _pick_float(
        temperature,
        cfg_select,
        ("training.generation_temperature",),
        1.0,
    )

    resolved_noise_schedule_name = "cosine"
    resolved_noise_schedule_params: dict[str, Any] = {}
    if dynin_cfg is not None:
        try:
            from omegaconf import OmegaConf

            explicit_schedule = OmegaConf.select(dynin_cfg, "mask_schedule.schedule")
            if explicit_schedule is not None:
                resolved_noise_schedule_name = str(explicit_schedule)
            else:
                legacy_schedule = OmegaConf.select(dynin_cfg, "training.mask_schedule")
                if legacy_schedule is not None:
                    resolved_noise_schedule_name = str(legacy_schedule)

            explicit_params = OmegaConf.select(dynin_cfg, "mask_schedule.params")
            if explicit_params is not None:
                if OmegaConf.is_config(explicit_params):
                    explicit_params = OmegaConf.to_container(explicit_params, resolve=True)
                if isinstance(explicit_params, dict):
                    resolved_noise_schedule_params = {str(k): v for k, v in explicit_params.items()}
        except Exception:
            pass

    return {
        "prompt_max_text_len": resolved_prompt_max_text_len,
        "image_token_count": resolved_image_token_count,
        "mask_token_id": resolved_mask_token_id,
        "codebook_size": resolved_codebook_size,
        "timesteps": resolved_timesteps,
        "guidance_scale": resolved_guidance_scale,
        "temperature": resolved_temperature,
        "noise_schedule_name": resolved_noise_schedule_name,
        "noise_schedule_params": resolved_noise_schedule_params,
    }


def build_t2i_prompt(
    *,
    input_ids: list[int],
    attention_mask: list[int],
    uncond_input_ids: list[int] | None,
    uncond_attention_mask: list[int] | None,
    prompt_max_text_len: int,
    image_token_count: int,
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
) -> dict[str, Any]:
    prompt_token_ids = [int(v) for v in input_ids]
    additional_information: dict[str, Any] = {
        "task": ["t2i"],
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
        "seq_len": [int(image_token_count)],
        "mask_token_id": [int(mask_token_id)],
        "codebook_size": [int(codebook_size)],
    }
    additional_information["noise_schedule_params"] = [dict(noise_schedule_params or {})]
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


def load_uni_prompting_for_t2i(
    *,
    tokenizer_source: str,
    max_text_len: int,
    cond_dropout_prob: float,
    local_files_only: bool,
) -> Any:
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.dynin_omni.prompting_utils import UniversalPrompting

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
        ),
        ignore_id=-100,
        cond_dropout_prob=float(cond_dropout_prob),
        use_reserved_token=True,
    )


def build_model_inputs_like_validation(
    *,
    prompt_text: str,
    uni_prompting: Any,
    image_token_count: int,
    mask_token_id: int,
    guidance_scale: float,
) -> dict[str, list[int] | None]:
    image_tokens = torch.full(
        (1, int(image_token_count)),
        fill_value=int(mask_token_id),
        dtype=torch.long,
    )
    input_ids, attention_mask = uni_prompting(([str(prompt_text)], image_tokens), "t2i_gen")
    uncond_input_ids = None
    uncond_attention_mask = None
    if float(guidance_scale) > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(([""], image_tokens), "t2i_gen")

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


def extract_image_tensor(outputs: list[Any]) -> torch.Tensor | None:
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


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DYNIN text-to-image example using vllm_omni.")
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
        default=str(repo_root / "vllm_omni/model_executor/models/dynin_omni/configs/dynin_omni.yaml"),
        help="Path to DYNIN config yaml.",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="/home/kdg6245/omada/validation/data/text/t2i_metadata.jsonl",
        help="jsonl metadata path containing {'id','prompt'} rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/dynin_t2i_outputs",
        help="Output directory to save generated images.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/tmp/dynin_localized_models",
        help="Cache dir used when --model is HF repo id.",
    )
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all prompts from metadata.")
    parser.add_argument(
        "--prompt-max-text-len",
        type=int,
        default=None,
        help="UniversalPrompting max_text_len for t2i prompt building. Default: dataset.preprocessing.max_seq_length from --dynin-config-path.",
    )
    parser.add_argument(
        "--image-token-count",
        type=int,
        default=None,
        help="Number of image tokens. Default: model.omada.num_vq_tokens from --dynin-config-path.",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Mask token id for image placeholders. Default: model.omada.mask_token_id (or 126336).",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=None,
        help="Image codebook size. Default: model.omada.codebook_size from --dynin-config-path.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Generation timesteps. Default: training.generation_timesteps from --dynin-config-path.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG guidance scale. Default: training.guidance_scale from --dynin-config-path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature. Default: training.generation_temperature from --dynin-config-path.",
    )
    parser.add_argument("--max-tokens-per-stage", type=int, default=1)
    parser.add_argument(
        "--vq-model-image-path",
        type=str,
        default="",
        help="Local directory path for MAGVITv2 weights. If set, Stage-1 will use this path.",
    )
    parser.add_argument(
        "--vq-model-image-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Stage-1 image VQ local_files_only. Example: --vq-model-image-local-files-only",
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
        default="auto",
        help="Optional vLLM dtype override (e.g., float16, bfloat16, float32, auto).",
    )
    return parser.parse_args()


def main() -> None:
    # import torch
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_math_sdp(True)
    repo_root = bootstrap_repo_path()
    ensure_safe_import_for_vllm()
    args = parse_args(repo_root)

    dynin_config_path = str(Path(args.dynin_config_path).expanduser())
    runtime_defaults = resolve_t2i_runtime_defaults(
        dynin_config_path=dynin_config_path,
        prompt_max_text_len=args.prompt_max_text_len,
        image_token_count=args.image_token_count,
        mask_token_id=args.mask_token_id,
        codebook_size=args.codebook_size,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
    )
    prompt_max_text_len = int(runtime_defaults["prompt_max_text_len"])
    image_token_count = int(runtime_defaults["image_token_count"])
    mask_token_id = int(runtime_defaults["mask_token_id"])
    codebook_size = int(runtime_defaults["codebook_size"])
    timesteps = int(runtime_defaults["timesteps"])
    guidance_scale = float(runtime_defaults["guidance_scale"])
    temperature = float(runtime_defaults["temperature"])
    noise_schedule_name = str(runtime_defaults["noise_schedule_name"])
    noise_schedule_params = runtime_defaults.get("noise_schedule_params", {})
    if not isinstance(noise_schedule_params, dict):
        noise_schedule_params = {}
    noise_type, cond_dropout_prob, tokenizer_path, model_local_files_only = resolve_prompting_defaults(
        dynin_config_path
    )
    os.environ["DYNIN_CONFIG_PATH"] = dynin_config_path
    if args.disable_hf_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    metadata_path = Path(args.metadata_path).expanduser()
    records = parse_metadata(metadata_path)
    if args.max_prompts > 0:
        records = records[: args.max_prompts]

    model_dir = ensure_local_model_dir(args.model, cache_dir=Path(args.model_cache_dir))
    tokenizer_source = str(tokenizer_path) if tokenizer_path else str(model_dir)
    uni_prompting = load_uni_prompting_for_t2i(
        tokenizer_source=tokenizer_source,
        max_text_len=prompt_max_text_len,
        cond_dropout_prob=cond_dropout_prob,
        local_files_only=model_local_files_only,
    )
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model=str(model_dir), stage_configs_path=args.stage_config_path, dtype=args.dtype)

    sampling_params_list = [
        SamplingParams(max_tokens=args.max_tokens_per_stage, temperature=0.0, top_p=1.0, detokenize=False)
        for _ in range(len(omni.stage_list))
    ]

    result_rows: list[dict[str, Any]] = []
    try:
        for idx, row in enumerate(records):
            sample_id = str(row.get("id", f"sample-{idx:05d}"))
            prompt_text = str(row["prompt"])
            model_inputs = build_model_inputs_like_validation(
                prompt_text=prompt_text,
                uni_prompting=uni_prompting,
                image_token_count=image_token_count,
                mask_token_id=mask_token_id,
                guidance_scale=guidance_scale,
            )
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            if input_ids is None or attention_mask is None:
                raise RuntimeError(f"Failed to build t2i model inputs for sample '{sample_id}'")

            prompt = build_t2i_prompt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                uncond_input_ids=model_inputs["uncond_input_ids"],
                uncond_attention_mask=model_inputs["uncond_attention_mask"],
                prompt_max_text_len=prompt_max_text_len,
                image_token_count=image_token_count,
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
            )
            outputs = list(omni.generate(prompt, sampling_params_list))
            image_tensor = extract_image_tensor(outputs)
            if image_tensor is None:
                raise RuntimeError(f"No image output found for sample '{sample_id}'")

            image = tensor_to_pil_image(image_tensor)
            save_path = output_dir / f"{sample_id}.png"
            image.save(save_path)

            result_rows.append(
                {
                    "id": sample_id,
                    "prompt": prompt_text,
                    "image_path": str(save_path),
                }
            )
            print(f"[{idx + 1}/{len(records)}] saved: {save_path}")
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
python <REPO_ROOT>/examples/offline_inference/dynin_omni/t2i.py \
  --model snu-aidas/Dynin-Omni \
  --dynin-config-path <REPO_ROOT>/vllm_omni/model_executor/models/dynin_omni/configs/dynin_omni.yaml \
  --metadata-path <REPO_ROOT>/examples/offline_inference/dynin_omni/data/text/t2i_metadata.jsonl \
  --output-dir <REPO_ROOT>/examples/offline_inference/dynin_omni/results/t2i_from_vllm \
  --max-prompts 4
"""
