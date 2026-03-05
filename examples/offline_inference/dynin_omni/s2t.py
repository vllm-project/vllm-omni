#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import inspect
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


DEFAULT_S2T_INSTRUCTIONS = [
    "Transcribe the given audio.",
    "Write down what you hear in the audio.",
    "Provide a transcript for the given speech.",
    "What does the speaker in the audio say?",
    "Convert the speech in the audio to text.",
    "Listen to the audio and write out the text.",
]


_ASR_WS_RE = re.compile(r"\s+")


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
        print(f"[s2t] Downloading model to local dir: {local_dir}")
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
        print(f"[s2t] Using tokenizer from local model dir: {model_dir}")
        return str(model_dir)

    if configured_tokenizer_path:
        configured_path = Path(configured_tokenizer_path).expanduser()
        if configured_path.is_dir():
            resolved = configured_path.resolve()
            print(f"[s2t] Using tokenizer from configured local path: {resolved}")
            return str(resolved)

        print(
            "[s2t] Local tokenizer files were not found in model dir; "
            f"falling back to configured tokenizer source: {configured_tokenizer_path}"
        )
        return str(configured_tokenizer_path)

    print(
        "[s2t] Local tokenizer markers were not found; "
        f"falling back to model dir anyway: {model_dir}"
    )
    return str(model_dir)


def _get_audio_field(audio_entry: Any, key: str) -> Any:
    if audio_entry is None:
        return None
    if isinstance(audio_entry, dict):
        return audio_entry.get(key)
    try:
        return audio_entry[key]
    except Exception:
        return getattr(audio_entry, key, None)


def _save_audio_array(path: Path, audio_array: Any, sampling_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = np.asarray(audio_array, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    try:
        import soundfile as sf

        sf.write(str(path), wav, int(sampling_rate), format="WAV")
    except Exception:
        from scipy.io import wavfile

        wav_i16 = np.clip(wav, -1.0, 1.0)
        wav_i16 = (wav_i16 * 32767.0).astype(np.int16)
        wavfile.write(str(path), int(sampling_rate), wav_i16)


def _resolve_audio_path_value(value: str, *, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path.resolve()


def resolve_audio_path_from_row(
    *,
    row: dict[str, Any],
    base_dir: Path,
    audio_cache_dir: Path,
    sample_id: str,
) -> Path:
    audio_candidates = [
        row.get("audio_path"),
        row.get("audio_file"),
        row.get("path"),
        row.get("file"),
    ]
    audio_entry = row.get("audio")
    if isinstance(audio_entry, str):
        audio_candidates.append(audio_entry)
    else:
        audio_candidates.extend(
            [
                _get_audio_field(audio_entry, "path"),
                _get_audio_field(audio_entry, "audio_path"),
                _get_audio_field(audio_entry, "file"),
            ]
        )

    missing_paths: list[Path] = []
    for candidate in audio_candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        resolved = _resolve_audio_path_value(candidate.strip(), base_dir=base_dir)
        if resolved.exists():
            return resolved
        missing_paths.append(resolved)

    audio_array = _get_audio_field(audio_entry, "array")
    if audio_array is not None:
        sampling_rate = _get_audio_field(audio_entry, "sampling_rate")
        sampling_rate = int(sampling_rate or 16000)
        cached_name = sanitize_file_stem(sample_id, fallback="sample")
        cached_path = audio_cache_dir / f"{cached_name}.wav"
        if not cached_path.exists():
            _save_audio_array(cached_path, audio_array, sampling_rate)
        return cached_path.resolve()

    if missing_paths:
        first = missing_paths[0]
        raise FileNotFoundError(
            f"audio file not found: {first} (checked {len(missing_paths)} candidate path(s))"
        )

    raise ValueError(
        "metadata row has no usable audio source. "
        "Expected one of: audio_path/audio_file/path/file/audio.path/audio.array"
    )


def parse_metadata(metadata_path: Path, audio_cache_dir: Path) -> list[dict[str, Any]]:
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

            audio_path = resolve_audio_path_from_row(
                row=row,
                base_dir=metadata_path.parent,
                audio_cache_dir=audio_cache_dir,
                sample_id=sample_id,
            )

            gt_text = row.get("text")
            if gt_text is None:
                gt_text = row.get("gt_text")
            if gt_text is not None:
                gt_text = str(gt_text)

            instruction = row.get("instruction")
            if instruction is not None:
                instruction = str(instruction)

            rows.append(
                {
                    "id": sample_id,
                    "audio_path": str(audio_path),
                    "gt_text": gt_text,
                    "instruction": instruction,
                }
            )
    if not rows:
        raise ValueError(f"metadata is empty: {metadata_path}")
    return rows


def normalize_asr_text(text: str) -> str:
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
    text = _ASR_WS_RE.sub(" ", text).strip()
    return text


def resolve_s2t_runtime_defaults(
    *,
    dynin_config_path: str,
    prompt_max_text_len: int | None,
    s2t_new_tokens: int | None,
    s2t_steps: int | None,
    s2t_block_length: int | None,
    s2t_temperature: float | None,
    s2t_cfg_scale: float | None,
    mask_token_id: int | None,
    codebook_size: int | None,
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

    return {
        "prompt_max_text_len": _pick_int(
            prompt_max_text_len,
            cfg_select,
            ("dataset.preprocessing.max_seq_length_text", "dataset.preprocessing.max_seq_length"),
            1024,
        ),
        "s2t_new_tokens": _pick_int(
            s2t_new_tokens,
            cfg_select,
            ("dataset.preprocessing.max_seq_length_s2t", "training.max_seq_length_s2t"),
            128,
        ),
        "s2t_steps": _pick_int(
            s2t_steps,
            cfg_select,
            ("speech.s2t_steps", "training.s2t_steps"),
            256,
        ),
        "s2t_block_length": _pick_int(
            s2t_block_length,
            cfg_select,
            ("speech.s2t_block_length", "training.s2t_block_length"),
            2,
        ),
        "s2t_temperature": _pick_float(
            s2t_temperature,
            cfg_select,
            ("speech.s2t_temperature", "training.s2t_temperature"),
            0.0,
        ),
        "s2t_cfg_scale": _pick_float(
            s2t_cfg_scale,
            cfg_select,
            ("speech.s2t_cfg_scale", "training.s2t_cfg_scale"),
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


def resolve_vq_audio_source(
    *,
    cli_vq_model_audio_path: str | None,
    cli_vq_model_audio_local_files_only: bool | None,
    dynin_config_path: str,
    default_model_local_files_only: bool,
) -> tuple[str, bool]:
    source = "Emova-ollm/emova_speech_tokenizer_hf"
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


def load_uni_prompting_for_s2t(
    *,
    tokenizer_source: str,
    max_text_len: int,
    cond_dropout_prob: float,
    max_audio_len: int,
    max_audio_len_short: int,
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


def load_vq_model_audio(
    *,
    source: str,
    local_files_only: bool,
) -> Any:
    from transformers.modeling_utils import PreTrainedModel
    from vllm_omni.model_executor.models.dynin_omni.models.speech.modeling_emova_speech_tokenizer import (
        EMOVASpeechTokenizer,
    )

    try:
        load_state_dict_src = inspect.getsource(EMOVASpeechTokenizer.load_state_dict)
    except Exception:
        load_state_dict_src = ""

    if "pdb.set_trace" in load_state_dict_src:

        def _safe_load_state_dict(self: Any, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
            try:
                return PreTrainedModel.load_state_dict(self, state_dict, strict=strict, assign=assign)
            except TypeError:
                return PreTrainedModel.load_state_dict(self, state_dict, strict=strict)

        EMOVASpeechTokenizer.load_state_dict = _safe_load_state_dict  # type: ignore[method-assign]

    load_kwargs = {"local_files_only": bool(local_files_only)}
    try:
        vq_model_audio = EMOVASpeechTokenizer.from_pretrained(source, **load_kwargs)
    except TypeError:
        load_kwargs.pop("local_files_only", None)
        vq_model_audio = EMOVASpeechTokenizer.from_pretrained(source, **load_kwargs)
    vq_model_audio.requires_grad_(False)
    vq_model_audio.eval()
    return vq_model_audio


def build_s2t_chat_prompt(instruction: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
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


def build_s2t_input_ids(
    *,
    audio_token_ids: Any,
    tokenizer: Any,
    uni_prompting: Any,
    instruction: str,
    speech_token_offset: int,
) -> tuple[list[int], str]:
    audio_ids = audio_token_ids
    if isinstance(audio_ids, torch.Tensor):
        audio_ids = audio_ids.detach().cpu().reshape(-1).tolist()
    else:
        audio_ids = np.asarray(audio_ids).reshape(-1).tolist()
    audio_ids = [int(x) + int(speech_token_offset) for x in audio_ids]

    sptids = uni_prompting.sptids_dict
    task_id = _scalar_token_id(sptids["<|s2t|>"])
    soa_id = _scalar_token_id(sptids["<|soa|>"])
    eoa_id = _scalar_token_id(sptids["<|eoa|>"])

    prompt_text = build_s2t_chat_prompt(instruction)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0].detach().cpu().tolist()

    input_ids = [task_id, soa_id] + audio_ids + [eoa_id] + [int(v) for v in prompt_ids]
    return input_ids, prompt_text


def build_s2t_prompt(
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
        raise ValueError("--s2t-new-tokens must be > 0.")
    if block_length <= 0:
        raise ValueError("--s2t-block-length must be > 0.")
    if steps <= 0:
        raise ValueError("--s2t-steps must be > 0.")
    if max_new_tokens % block_length != 0:
        raise ValueError(
            f"s2t requires max_new_tokens % block_length == 0, got {max_new_tokens} % {block_length}"
        )
    num_blocks = max_new_tokens // block_length
    if num_blocks <= 0:
        raise ValueError("Invalid number of generation blocks.")
    if steps % num_blocks != 0:
        raise ValueError(
            "s2t requires steps % (max_new_tokens // block_length) == 0, "
            f"got steps={steps}, max_new_tokens={max_new_tokens}, block_length={block_length}"
        )


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DYNIN speech-to-text example using vllm_omni.")
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
        help="Optional jsonl path containing audio rows.",
    )
    parser.add_argument(
        "--audio-path",
        action="append",
        default=[],
        help="Audio path to transcribe. Repeat --audio-path for multiple samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/dynin_s2t_outputs",
        help="Output directory to save transcripts.",
    )
    parser.add_argument(
        "--audio-cache-dir",
        type=str,
        default="/tmp/dynin_s2t_audio_cache",
        help="Cache directory for metadata rows that embed audio arrays.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/tmp/dynin_localized_models",
        help="Cache dir used when --model is HF repo id.",
    )
    parser.add_argument("--max-prompts", type=int, default=0, help="0 means all inputs.")
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Optional fixed instruction. If empty, one is selected from defaults.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic instruction selection.")

    parser.add_argument(
        "--task",
        type=str,
        default="s2t",
        choices=["s2t"],
        help="DYNIN token2text generation task.",
    )
    parser.add_argument("--detok-id", type=int, default=0, help="Detokenizer id (text=0).")
    parser.add_argument("--prompt-max-text-len", type=int, default=None)
    parser.add_argument("--s2t-new-tokens", type=int, default=None, help="Number of generated text tokens.")
    parser.add_argument("--s2t-steps", type=int, default=None, help="Number of s2t generation steps.")
    parser.add_argument("--s2t-block-length", type=int, default=None, help="Block length for s2t generation.")
    parser.add_argument("--s2t-temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--s2t-cfg-scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--s2t-remasking", type=str, default="low_confidence", help="Remasking strategy.")
    parser.add_argument("--mask-token-id", type=int, default=None, help="Mask token id.")
    parser.add_argument("--codebook-size", type=int, default=None, help="Image codebook size in DYNIN config.")
    parser.add_argument("--max-tokens-per-stage", type=int, default=1)

    parser.add_argument(
        "--vq-model-audio-path",
        type=str,
        default="",
        help="EMOVA speech tokenizer model path or repo id.",
    )
    parser.add_argument(
        "--vq-model-audio-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override VQ audio local_files_only.",
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
    runtime_defaults = resolve_s2t_runtime_defaults(
        dynin_config_path=dynin_config_path,
        prompt_max_text_len=args.prompt_max_text_len,
        s2t_new_tokens=args.s2t_new_tokens,
        s2t_steps=args.s2t_steps,
        s2t_block_length=args.s2t_block_length,
        s2t_temperature=args.s2t_temperature,
        s2t_cfg_scale=args.s2t_cfg_scale,
        mask_token_id=args.mask_token_id,
        codebook_size=args.codebook_size,
    )

    prompt_max_text_len = int(runtime_defaults["prompt_max_text_len"])
    s2t_new_tokens = int(runtime_defaults["s2t_new_tokens"])
    s2t_steps = int(runtime_defaults["s2t_steps"])
    s2t_block_length = int(runtime_defaults["s2t_block_length"])
    s2t_temperature = float(runtime_defaults["s2t_temperature"])
    s2t_cfg_scale = float(runtime_defaults["s2t_cfg_scale"])
    mask_token_id = int(runtime_defaults["mask_token_id"])
    codebook_size = int(runtime_defaults["codebook_size"])

    validate_generation_args(
        max_new_tokens=s2t_new_tokens,
        steps=s2t_steps,
        block_length=s2t_block_length,
    )

    cond_dropout_prob, tokenizer_path, model_local_files_only, max_audio_len, max_audio_len_short = (
        resolve_prompting_defaults(dynin_config_path)
    )

    os.environ["DYNIN_CONFIG_PATH"] = dynin_config_path
    if args.disable_hf_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    audio_cache_dir = Path(args.audio_cache_dir).expanduser()
    audio_cache_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    if args.metadata_path:
        metadata_path = Path(args.metadata_path).expanduser()
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata path not found: {metadata_path}")
        records.extend(parse_metadata(metadata_path, audio_cache_dir))

    for idx, audio_path_raw in enumerate(args.audio_path):
        resolved_audio_path = _resolve_audio_path_value(audio_path_raw, base_dir=Path.cwd())
        if not resolved_audio_path.exists():
            raise FileNotFoundError(f"audio path not found: {resolved_audio_path}")
        records.append(
            {
                "id": f"cli-{idx:05d}",
                "audio_path": str(resolved_audio_path),
                "gt_text": None,
                "instruction": None,
            }
        )

    if not records:
        raise ValueError("No inputs provided. Use --metadata-path or --audio-path.")

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
        print("[s2t] Enabling local_files_only because tokenizer source is local.")

    uni_prompting = load_uni_prompting_for_s2t(
        tokenizer_source=tokenizer_source,
        max_text_len=prompt_max_text_len,
        cond_dropout_prob=cond_dropout_prob,
        max_audio_len=max_audio_len,
        max_audio_len_short=max_audio_len_short,
        local_files_only=effective_model_local_files_only,
    )
    tokenizer = uni_prompting.text_tokenizer

    vq_audio_source, vq_audio_local_files_only = resolve_vq_audio_source(
        cli_vq_model_audio_path=args.vq_model_audio_path.strip() or None,
        cli_vq_model_audio_local_files_only=args.vq_model_audio_local_files_only,
        dynin_config_path=dynin_config_path,
        default_model_local_files_only=False,
    )
    print(
        f"[s2t] Loading speech tokenizer from: {vq_audio_source} "
        f"(local_files_only={vq_audio_local_files_only})"
    )
    vq_model_audio = load_vq_model_audio(
        source=vq_audio_source,
        local_files_only=vq_audio_local_files_only,
    )

    speech_token_offset = len(uni_prompting.text_tokenizer) + int(codebook_size)
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
            audio_path = Path(str(row["audio_path"])).expanduser().resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"audio file not found: {audio_path}")

            instruction = row.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                instruction = args.instruction.strip()
            if not instruction:
                instruction = DEFAULT_S2T_INSTRUCTIONS[(int(args.seed) + idx) % len(DEFAULT_S2T_INSTRUCTIONS)]

            audio_token_ids = vq_model_audio.encode(str(audio_path))
            prompt_token_ids, prompt_text = build_s2t_input_ids(
                audio_token_ids=audio_token_ids,
                tokenizer=tokenizer,
                uni_prompting=uni_prompting,
                instruction=instruction,
                speech_token_offset=speech_token_offset,
            )

            prompt = build_s2t_prompt(
                prompt_token_ids=prompt_token_ids,
                dynin_config_path=dynin_config_path,
                task=args.task,
                detok_id=args.detok_id,
                prompt_max_text_len=prompt_max_text_len,
                cond_dropout_prob=cond_dropout_prob,
                max_new_tokens=s2t_new_tokens,
                steps=s2t_steps,
                block_length=s2t_block_length,
                temperature=s2t_temperature,
                cfg_scale=s2t_cfg_scale,
                remasking=args.s2t_remasking,
                mask_token_id=mask_token_id,
                tokenizer_path=tokenizer_source,
                model_local_files_only=effective_model_local_files_only,
            )

            outputs = list(omni.generate(prompt, sampling_params_list))
            decoded_text = extract_text_output(outputs, tokenizer=tokenizer)
            decoded_text_norm = normalize_asr_text(decoded_text)

            row_out = {
                "id": sample_id,
                "audio_path": str(audio_path),
                "instruction": instruction,
                "prompt": prompt_text,
                "decoded_text": decoded_text,
                "decoded_text_norm": decoded_text_norm,
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
python /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/s2t.py \
  --model snu-aidas/Dynin-Omni \
  --dynin-config-path /home/kdg6245/vllm-omni/vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml \
  --audio-path /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/results/t2s_from_vllm/cli-00000.wav \
  --output-dir /home/kdg6245/vllm-omni/examples/offline_inference/dynin_omni/results/s2t_from_vllm
"""
