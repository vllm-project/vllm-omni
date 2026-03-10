#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_T2S_INSTRUCTIONS = [
    "Generate speech for the given text.",
    "Read the given sentence aloud.",
    "Say the given words.",
    "Convert the given text into spoken audio.",
    "Speak the given text.",
    "Synthesize the text into speech.",
]

DEFAULT_T2S_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call me back when you arrive at the station.",
    "Today is a good day to run a text to speech experiment.",
]


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


def looks_like_hf_repo_id(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    if value.count("/") != 1:
        return False
    org, name = value.split("/", 1)
    return bool(org) and bool(name)


def _get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _resolve_hf_modules_cache_root() -> Path:
    env_path = os.environ.get("HF_MODULES_CACHE")
    if env_path:
        return Path(env_path).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "modules").resolve()
    try:
        from transformers.utils.hub import HF_MODULES_CACHE

        return Path(HF_MODULES_CACHE).expanduser().resolve()
    except Exception:
        pass
    return (Path.home() / ".cache" / "huggingface" / "modules").resolve()


def _infer_module_cache_dirs_for_source(source: str) -> list[Path]:
    root = _resolve_hf_modules_cache_root() / "transformers_modules"
    candidates: list[Path] = []

    repo_name = source
    if looks_like_hf_repo_id(source):
        org, repo = source.split("/", 1)
        repo_name = repo
        org_token = org.replace("-", "_hyphen_")
        repo_token = repo.replace("-", "_hyphen_")
        org_repo_root = root / org_token / repo_token
        candidates.append(org_repo_root)
        if org_repo_root.is_dir():
            for p in org_repo_root.iterdir():
                if not p.is_dir():
                    continue
                if p.name.startswith("__") or p.name in {"runtime_assets", "__pycache__"}:
                    continue
                candidates.append(p)
    else:
        repo_name = Path(source).name

    candidates.append(root / repo_name)

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique_dirs.append(cand)
    return unique_dirs


def _inject_runtime_assets_into_module_cache(*, source: str, runtime_assets_root: Path) -> None:
    runtime_assets_root = runtime_assets_root.expanduser().resolve()
    if not runtime_assets_root.is_dir():
        return

    for module_dir in _infer_module_cache_dirs_for_source(source):
        try:
            module_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue

        target = module_dir / "runtime_assets"
        try:
            if target.is_symlink() and target.resolve() == runtime_assets_root:
                continue
        except Exception:
            pass

        if not target.exists():
            try:
                os.symlink(str(runtime_assets_root), str(target), target_is_directory=True)
                continue
            except Exception:
                pass

        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        for name in ("u2s_config.json", "condition2style_centroid.txt"):
            src_file = runtime_assets_root / name
            dst_file = target / name
            if src_file.is_file():
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception:
                    pass


def ensure_local_vq_audio_repo(
    *,
    source: str,
    local_repo_dir: Path,
    revision: str | None,
    local_files_only: bool,
    force_download: bool,
) -> Path:
    """Ensure local EMOVA tokenizer repo exists; download it if needed."""
    local_repo_dir = local_repo_dir.expanduser().resolve()
    marker_files = (
        "config.json",
        "modeling_emova_speech_tokenizer.py",
        "emova_decode_runtime.py",
    )
    if local_repo_dir.is_dir() and all((local_repo_dir / name).exists() for name in marker_files):
        return local_repo_dir

    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return source_path.resolve()

    if local_files_only:
        raise FileNotFoundError(
            f"Local EMOVA tokenizer repo not found at {local_repo_dir}, "
            "and local_files_only=True prevents snapshot download."
        )

    if not looks_like_hf_repo_id(source):
        raise ValueError(
            "Cannot snapshot_download non-HF source. "
            f"Expected repo id like 'org/repo', got: {source}"
        )

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub unavailable; cannot download {source}: {e}") from e

    local_repo_dir.mkdir(parents=True, exist_ok=True)
    token = _get_hf_token()
    kwargs: dict[str, Any] = {
        "repo_id": source,
        "repo_type": "model",
        "revision": revision,
        "local_dir": str(local_repo_dir),
        "token": token,
        "force_download": bool(force_download),
    }
    try:
        snapshot_dir = snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("token", None)
        kwargs.pop("force_download", None)
        snapshot_dir = snapshot_download(**kwargs)

    resolved = Path(snapshot_dir).resolve()
    print(f"[t2s] Local EMOVA tokenizer repo ready at: {resolved}")
    return resolved


def resolve_vq_audio_source(
    *,
    cli_vq_model_audio_path: str | None,
    cli_vq_model_audio_local_files_only: bool | None,
    dynin_config_path: str,
    default_model_local_files_only: bool,
) -> tuple[str, bool]:
    source = "snu-aidas/emova_speech_tokenizer_vllm"
    local_files_only = bool(default_model_local_files_only)
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(dynin_config_path)
        vq_source_value = OmegaConf.select(cfg, "model.vq_model_audio.repo_id")
        if vq_source_value is not None:
            source = str(vq_source_value)
        vq_local_only_value = OmegaConf.select(cfg, "model.vq_model_audio.local_files_only")
        if vq_local_only_value is not None:
            local_files_only = bool(vq_local_only_value)
    except Exception:
        pass

    if cli_vq_model_audio_path:
        source = str(cli_vq_model_audio_path)
    if cli_vq_model_audio_local_files_only is not None:
        local_files_only = bool(cli_vq_model_audio_local_files_only)
    elif Path(source).expanduser().is_dir():
        local_files_only = True
    return source, local_files_only


def prepare_local_vq_audio_runtime(
    *,
    source: str,
    local_repo_dir: Path,
    revision: str | None,
    local_files_only: bool,
    force_download: bool,
) -> tuple[str, bool]:
    local_repo = ensure_local_vq_audio_repo(
        source=source,
        local_repo_dir=local_repo_dir,
        revision=revision,
        local_files_only=bool(local_files_only),
        force_download=bool(force_download),
    )

    local_s2u_vendor = (local_repo / "s2u_vendor").resolve()
    if local_s2u_vendor.is_dir():
        os.environ["DYNIN_S2U_VENDOR_ROOT"] = str(local_s2u_vendor)

    local_runtime_assets = (local_repo / "runtime_assets").resolve()
    if local_runtime_assets.is_dir():
        os.environ["DYNIN_RUNTIME_ASSETS_ROOT"] = str(local_runtime_assets)
        local_u2s_cfg = local_runtime_assets / "u2s_config.json"
        if local_u2s_cfg.is_file():
            os.environ["DYNIN_U2S_CONFIG_PATH"] = str(local_u2s_cfg)
        _inject_runtime_assets_into_module_cache(
            source=str(local_repo),
            runtime_assets_root=local_runtime_assets,
        )

    return str(local_repo), True


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
        print(f"[t2s] Downloading model to local dir: {local_dir}")
        snapshot_download(
            repo_id=model,
            local_dir=str(local_dir),
            local_dir_use_symlinks=True,
            resume_download=True,
        )
    return local_dir.resolve()


_TOKENIZER_MARKER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
)


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
        print(f"[t2s] Using tokenizer from local model dir: {model_dir}")
        return str(model_dir)

    if configured_tokenizer_path:
        configured_path = Path(configured_tokenizer_path).expanduser()
        if configured_path.is_dir():
            resolved = configured_path.resolve()
            print(f"[t2s] Using tokenizer from configured local path: {resolved}")
            return str(resolved)

        print(
            "[t2s] Local tokenizer files were not found in model dir; "
            f"falling back to configured tokenizer source: {configured_tokenizer_path}"
        )
        return str(configured_tokenizer_path)

    print(f"[t2s] Local tokenizer markers were not found; falling back to model dir anyway: {model_dir}")
    return str(model_dir)


def parse_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if text is None:
                text = row.get("prompt")
            if text is None:
                text = row.get("gt_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"metadata line {line_idx} has no valid 'text'/'prompt'")
            sample_id = row.get("id")
            if sample_id is None:
                sample_id = row.get("sample_id")
            if sample_id is None:
                sample_id = f"sample-{line_idx:05d}"
            rows.append(
                {
                    "id": str(sample_id),
                    "text": text.strip(),
                    "condition": row.get("condition"),
                }
            )
    if not rows:
        raise ValueError(f"metadata is empty: {metadata_path}")
    return rows


def resolve_t2s_runtime_defaults(
    *,
    dynin_config_path: str,
    prompt_max_text_len: int | None,
    t2s_token_length: int | None,
    mask_token_id: int | None,
    codebook_size: int | None,
    audio_codebook_size: int | None,
    t2s_steps: int | None,
    t2s_block_length: int | None,
    t2s_temperature: float | None,
    t2s_cfg_scale: float | None,
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

    cfg_select = lambda _path: None
    try:
        from omegaconf import OmegaConf

        dynin_cfg = OmegaConf.load(dynin_config_path)
        cfg_select = lambda _path: OmegaConf.select(dynin_cfg, _path)
    except Exception:
        pass

    return {
        "prompt_max_text_len": _pick_int(
            prompt_max_text_len,
            cfg_select,
            ("dataset.preprocessing.max_seq_length_text", "dataset.preprocessing.max_seq_length"),
            1024,
        ),
        "t2s_token_length": _pick_int(
            t2s_token_length,
            cfg_select,
            ("speech.t2s_token_length", "training.t2s_token_length"),
            383,
        ),
        "mask_token_id": _pick_int(mask_token_id, cfg_select, ("model.omada.mask_token_id",), 126336),
        "codebook_size": _pick_int(codebook_size, cfg_select, ("model.omada.codebook_size",), 8192),
        "audio_codebook_size": _pick_int(
            audio_codebook_size,
            cfg_select,
            ("model.vq_model_audio.codebook_size",),
            4096,
        ),
        "t2s_steps": _pick_int(
            t2s_steps,
            cfg_select,
            ("speech.t2s_steps", "training.t2s_steps"),
            383,
        ),
        "t2s_block_length": _pick_int(
            t2s_block_length,
            cfg_select,
            ("speech.t2s_block_length", "training.t2s_block_length"),
            128,
        ),
        "t2s_temperature": _pick_float(
            t2s_temperature,
            cfg_select,
            ("speech.t2s_temperature", "training.t2s_temperature"),
            1.0,
        ),
        "t2s_cfg_scale": _pick_float(
            t2s_cfg_scale,
            cfg_select,
            ("speech.t2s_cfg_scale", "training.t2s_cfg_scale"),
            2.5,
        ),
    }


def resolve_prompting_defaults(
    dynin_config_path: str,
) -> tuple[float, str | None, bool, int, int]:
    cond_dropout_prob = 0.0
    tokenizer_path: str | None = None
    model_local_files_only = False
    max_audio_len = 512
    max_audio_len_short = 256
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
        max_audio_len_value = OmegaConf.select(cfg, "dataset.preprocessing.max_aud_length")
        if max_audio_len_value is not None:
            max_audio_len = int(max_audio_len_value)
        max_audio_len_short_value = OmegaConf.select(cfg, "dataset.preprocessing.max_aud_length_short")
        if max_audio_len_short_value is not None:
            max_audio_len_short = int(max_audio_len_short_value)
    except Exception:
        pass
    return cond_dropout_prob, tokenizer_path, model_local_files_only, max_audio_len, max_audio_len_short


def load_uni_prompting_for_t2s(
    *,
    tokenizer_source: str,
    max_text_len: int,
    cond_dropout_prob: float,
    max_audio_len: int,
    max_audio_len_short: int,
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
        max_audio_len=int(max_audio_len),
        max_audio_len_short=int(max_audio_len_short),
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


def build_t2s_chat_prompt(text: str, instruction: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}\n{text}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_model_inputs_like_validation(
    *,
    prompt_text: str,
    uni_prompting: Any,
    t2s_token_length: int,
    mask_token_id: int,
) -> dict[str, list[int] | None]:
    audio_tokens = torch.full(
        (1, int(t2s_token_length)),
        fill_value=int(mask_token_id),
        dtype=torch.long,
    )
    input_ids, attention_mask = uni_prompting(([str(prompt_text)], audio_tokens), "t2s_gen")

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
    }


def build_t2s_prompt(
    *,
    input_ids: list[int],
    attention_mask: list[int],
    dynin_config_path: str,
    task: str,
    detok_id: int,
    prompt_max_text_len: int,
    cond_dropout_prob: float,
    t2s_token_length: int,
    t2s_steps: int,
    t2s_block_length: int,
    t2s_temperature: float,
    t2s_cfg_scale: float,
    mask_token_id: int,
    codebook_size: int,
    audio_codebook_size: int,
    tokenizer_path: str | None,
    model_local_files_only: bool,
    t2s_condition: str | None,
    vq_model_audio_path: str | None,
    vq_model_audio_local_files_only: bool | None,
) -> dict[str, Any]:
    prompt_token_ids = [int(v) for v in input_ids]
    additional_information: dict[str, Any] = {
        "task": [str(task)],
        "detok_id": [int(detok_id)],
        "attention_mask": [[int(v) for v in attention_mask]],
        "dynin_config_path": [str(dynin_config_path)],
        "max_new_tokens": [int(t2s_token_length)],
        "steps": [int(t2s_steps)],
        "block_length": [int(t2s_block_length)],
        "temperature": [float(t2s_temperature)],
        "cfg_scale": [float(t2s_cfg_scale)],
        "mask_token_id": [int(mask_token_id)],
        "codebook_size": [int(codebook_size)],
        "audio_codebook_size": [int(audio_codebook_size)],
        "prompt_max_text_len": [int(prompt_max_text_len)],
        "prompting_max_text_len": [int(prompt_max_text_len)],
        "cond_dropout_prob": [float(cond_dropout_prob)],
        "prompting_cond_dropout_prob": [float(cond_dropout_prob)],
        "model_local_files_only": [bool(model_local_files_only)],
    }
    if tokenizer_path:
        additional_information["tokenizer_path"] = [str(tokenizer_path)]
    if t2s_condition:
        additional_information["condition"] = [str(t2s_condition)]
    if vq_model_audio_path:
        additional_information["vq_model_audio_path"] = [str(vq_model_audio_path)]
    if vq_model_audio_local_files_only is not None:
        additional_information["vq_model_audio_local_files_only"] = [bool(vq_model_audio_local_files_only)]

    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": additional_information,
        "modalities": ["audio"],
    }


def to_scalar_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, list):
        if not value:
            return default
        return int(value[0])
    if hasattr(value, "item"):
        try:
            return int(value.item())
        except Exception:
            return default
    return int(value)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def extract_audio_output(outputs: list[Any]) -> tuple[np.ndarray, int] | None:
    for omni_out in outputs:
        if getattr(omni_out, "final_output_type", None) != "audio":
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

            audio = mm_out.get("audio")
            if audio is None:
                audio = mm_out.get("speech")
            if audio is None:
                continue

            if isinstance(audio, list):
                chunks = [tensor_to_numpy(chunk).reshape(-1).astype(np.float32) for chunk in audio]
                wav = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
            else:
                wav = tensor_to_numpy(audio).reshape(-1).astype(np.float32)

            sr = to_scalar_int(mm_out.get("sr"), default=24000)
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
    parser = argparse.ArgumentParser(description="DYNIN text-to-speech example using vllm_omni.")
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
        default="",
        help="Optional jsonl path containing {'id','text'} or {'id','prompt'} rows.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text to synthesize. Repeat --text for multiple samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/dynin_t2s_outputs",
        help="Output directory to save generated wav files.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/tmp/dynin_localized_models",
        help="Cache dir used when --model is HF repo id.",
    )
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all inputs.")
    parser.add_argument(
        "--prompt-max-text-len",
        type=int,
        default=None,
        help="UniversalPrompting max_text_len for t2s prompt building.",
    )
    parser.add_argument("--t2s-token-length", type=int, default=None, help="Speech token length.")
    parser.add_argument("--mask-token-id", type=int, default=None, help="Mask token id.")
    parser.add_argument("--codebook-size", type=int, default=None, help="Image codebook size in DYNIN config.")
    parser.add_argument("--audio-codebook-size", type=int, default=None, help="Audio codebook size.")
    parser.add_argument("--t2s-steps", type=int, default=None, help="Number of t2s generation steps.")
    parser.add_argument("--t2s-block-length", type=int, default=None, help="Block length for MMU-like t2s.")
    parser.add_argument("--t2s-temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--t2s-cfg-scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--t2s-condition",
        type=str,
        default="gender-female_emotion-neutral_speed-normal_pitch-normal",
        help="Condition string passed to speech decoder.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2s_mmu_like",
        choices=["t2s", "t2s_mmu_like", "t2s_fixed"],
        help="DYNIN token2text generation task.",
    )
    parser.add_argument("--detok-id", type=int, default=1, help="Detokenizer id (audio=1).")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Optional fixed instruction prefix. If empty, one is selected from defaults.",
    )
    parser.add_argument(
        "--raw-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, use input text directly without chat/instruction wrapping.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic instruction selection.")
    parser.add_argument("--max-tokens-per-stage", type=int, default=1)
    parser.add_argument(
        "--vq-model-audio-path",
        type=str,
        default="",
        help="EMOVA speech tokenizer model path or repo id.",
    )
    parser.add_argument(
        "--vq-model-audio-local-repo",
        type=str,
        default="~/hf_models/emova_speech_tokenizer_vllm",
        help="Local directory for EMOVA tokenizer repo. Download target if missing.",
    )
    parser.add_argument(
        "--vq-model-audio-revision",
        type=str,
        default="main",
        help="Revision used when snapshot-downloading EMOVA tokenizer repo.",
    )
    parser.add_argument(
        "--vq-model-audio-force-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-download EMOVA tokenizer repo into --vq-model-audio-local-repo.",
    )
    parser.add_argument(
        "--vq-model-audio-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override Stage-2 audio VQ local_files_only.",
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
    repo_root = bootstrap_repo_path()
    ensure_safe_import_for_vllm()
    args = parse_args(repo_root)

    dynin_config_path = str(Path(args.dynin_config_path).expanduser())
    runtime_defaults = resolve_t2s_runtime_defaults(
        dynin_config_path=dynin_config_path,
        prompt_max_text_len=args.prompt_max_text_len,
        t2s_token_length=args.t2s_token_length,
        mask_token_id=args.mask_token_id,
        codebook_size=args.codebook_size,
        audio_codebook_size=args.audio_codebook_size,
        t2s_steps=args.t2s_steps,
        t2s_block_length=args.t2s_block_length,
        t2s_temperature=args.t2s_temperature,
        t2s_cfg_scale=args.t2s_cfg_scale,
    )
    prompt_max_text_len = int(runtime_defaults["prompt_max_text_len"])
    t2s_token_length = int(runtime_defaults["t2s_token_length"])
    mask_token_id = int(runtime_defaults["mask_token_id"])
    codebook_size = int(runtime_defaults["codebook_size"])
    audio_codebook_size = int(runtime_defaults["audio_codebook_size"])
    t2s_steps = int(runtime_defaults["t2s_steps"])
    t2s_block_length = int(runtime_defaults["t2s_block_length"])
    t2s_temperature = float(runtime_defaults["t2s_temperature"])
    t2s_cfg_scale = float(runtime_defaults["t2s_cfg_scale"])

    cond_dropout_prob, tokenizer_path, model_local_files_only, max_audio_len, max_audio_len_short = (
        resolve_prompting_defaults(dynin_config_path)
    )

    os.environ["DYNIN_CONFIG_PATH"] = dynin_config_path
    if args.disable_hf_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Ensure HF_HOME/module cache path is stabilized before injecting EMOVA runtime_assets.
    model_dir = ensure_local_model_dir(args.model, cache_dir=Path(args.model_cache_dir))

    vq_audio_source, vq_audio_local_files_only = resolve_vq_audio_source(
        cli_vq_model_audio_path=args.vq_model_audio_path.strip() or None,
        cli_vq_model_audio_local_files_only=args.vq_model_audio_local_files_only,
        dynin_config_path=dynin_config_path,
        default_model_local_files_only=False,
    )
    print(f"[t2s] Loading speech tokenizer from: {vq_audio_source} (local_files_only={vq_audio_local_files_only})")
    effective_vq_audio_path, effective_vq_audio_local_files_only = prepare_local_vq_audio_runtime(
        source=vq_audio_source,
        local_repo_dir=Path(args.vq_model_audio_local_repo),
        revision=args.vq_model_audio_revision,
        local_files_only=vq_audio_local_files_only,
        force_download=bool(args.vq_model_audio_force_download),
    )
    print(
        "[t2s] Using local speech tokenizer for Stage-2: "
        f"{effective_vq_audio_path} (local_files_only={effective_vq_audio_local_files_only})"
    )

    records: list[dict[str, Any]] = []
    if args.metadata_path:
        metadata_path = Path(args.metadata_path).expanduser()
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata path not found: {metadata_path}")
        records.extend(parse_metadata(metadata_path))

    for idx, text in enumerate(args.text):
        records.append(
            {
                "id": f"cli-{idx:05d}",
                "text": str(text),
                "condition": None,
            }
        )

    if not records:
        records = [
            {
                "id": f"default-{idx:05d}",
                "text": text,
                "condition": None,
            }
            for idx, text in enumerate(DEFAULT_T2S_TEXTS)
        ]

    if args.max_prompts > 0:
        records = records[: args.max_prompts]

    tokenizer_source = resolve_tokenizer_source_with_local_priority(
        model_dir=model_dir,
        configured_tokenizer_path=tokenizer_path,
    )
    effective_model_local_files_only = bool(model_local_files_only or Path(tokenizer_source).expanduser().is_dir())
    if effective_model_local_files_only and not model_local_files_only:
        print("[t2s] Enabling local_files_only because tokenizer source is local.")
    uni_prompting = load_uni_prompting_for_t2s(
        tokenizer_source=tokenizer_source,
        max_text_len=prompt_max_text_len,
        cond_dropout_prob=cond_dropout_prob,
        max_audio_len=max_audio_len,
        max_audio_len_short=max_audio_len_short,
        local_files_only=effective_model_local_files_only,
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
            sample_id_raw = str(row.get("id", f"sample-{idx:05d}"))
            sample_id = sanitize_file_stem(sample_id_raw, fallback=f"sample-{idx:05d}")
            source_text = str(row["text"]).strip()
            if not source_text:
                continue

            if args.raw_prompt:
                prompt_text = source_text
                instruction = ""
            else:
                instruction = args.instruction.strip()
                if not instruction:
                    instruction = DEFAULT_T2S_INSTRUCTIONS[(int(args.seed) + idx) % len(DEFAULT_T2S_INSTRUCTIONS)]
                prompt_text = build_t2s_chat_prompt(source_text, instruction)

            model_inputs = build_model_inputs_like_validation(
                prompt_text=prompt_text,
                uni_prompting=uni_prompting,
                t2s_token_length=t2s_token_length,
                mask_token_id=mask_token_id,
            )
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs["attention_mask"]
            if input_ids is None or attention_mask is None:
                raise RuntimeError(f"Failed to build t2s model inputs for sample '{sample_id}'")

            condition = row.get("condition")
            if not isinstance(condition, str) or not condition.strip():
                condition = args.t2s_condition

            prompt = build_t2s_prompt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dynin_config_path=dynin_config_path,
                task=args.task,
                detok_id=args.detok_id,
                prompt_max_text_len=prompt_max_text_len,
                cond_dropout_prob=cond_dropout_prob,
                t2s_token_length=t2s_token_length,
                t2s_steps=t2s_steps,
                t2s_block_length=t2s_block_length,
                t2s_temperature=t2s_temperature,
                t2s_cfg_scale=t2s_cfg_scale,
                mask_token_id=mask_token_id,
                codebook_size=codebook_size,
                audio_codebook_size=audio_codebook_size,
                tokenizer_path=tokenizer_source,
                model_local_files_only=effective_model_local_files_only,
                t2s_condition=condition,
                vq_model_audio_path=effective_vq_audio_path,
                vq_model_audio_local_files_only=effective_vq_audio_local_files_only,
            )

            outputs = list(omni.generate(prompt, sampling_params_list))
            audio_out = extract_audio_output(outputs)
            if audio_out is None:
                raise RuntimeError(f"No audio output found for sample '{sample_id}'")
            wav, sr = audio_out

            save_path = output_dir / f"{sample_id}.wav"
            save_audio_wav(save_path, wav, sr)

            result_rows.append(
                {
                    "id": sample_id,
                    "text": source_text,
                    "prompt": prompt_text,
                    "audio_path": str(save_path),
                    "sampling_rate": int(sr),
                    "num_samples": int(wav.shape[0]),
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
p <REPO_ROOT>/examples/offline_inference/dynin_omni/t2s.py \
  --model snu-aidas/Dynin-Omni \
  --dynin-config-path <REPO_ROOT>/vllm_omni/model_executor/models/dynin_omni/configs/dynin_omni.yaml \
  --text "Hello, this is a text to speech test." \
  --output-dir <REPO_ROOT>/examples/offline_inference/dynin_omni/results/t2s_from_vllm
"""


"""
python <REPO_ROOT>/examples/offline_inference/dynin_omni/t2s.py \
  --model snu-aidas/Dynin-Omni \
  --dynin-config-path <REPO_ROOT>/vllm_omni/model_executor/models/dynin_omni/configs/dynin_omni.yaml \
  --metadata-path <REPO_ROOT>/examples/offline_inference/dynin_omni/data/text/t2i_metadata.jsonl \
  --max-prompts 4 \
  --output-dir <REPO_ROOT>/examples/offline_inference/dynin_omni/results/t2s_from_vllm
"""
