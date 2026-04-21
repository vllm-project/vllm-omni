#!/usr/bin/env python3
"""Run a one-shot MiniCPM-o 4.5 text-to-audio smoke test on Modal.

Usage:
    modal run scripts/modal_minicpmo45_tts_smoke.py --preload-only
    modal run scripts/modal_minicpmo45_tts_smoke.py
    modal run scripts/modal_minicpmo45_tts_smoke.py \
        --prompt "Please read this sentence aloud."
    modal run scripts/modal_minicpmo45_tts_smoke.py \
        --ref-audio-path /abs/path/to/reference.wav
    modal run scripts/modal_minicpmo45_tts_smoke.py \
        --code2wav-fixture-output minicpmo45_code2wav_fixture.json

This script:
1. downloads MiniCPM-o 4.5 into a shared HF cache volume
2. installs vLLM-Omni runtime deps plus ``minicpmo-utils[all]`` for Token2wav
3. runs the local 3-stage MiniCPM pipeline from ``minicpmo.yaml``
4. saves the generated audio to a local WAV file via the local entrypoint

Unlike the existing text-only MiniCPM smoke, this script requests 2 GPUs so the
stage config can place thinker on GPU 0 and talker/code2wav on GPU 1.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Final

import modal
import numpy as np

APP_NAME: Final = "minicpmo45-tts-smoke"
MODEL_ID: Final = "openbmb/MiniCPM-o-4_5"
VLLM_VERSION: Final = "0.19.0"
GPU_REQUEST: Final = "A100-80GB:2"
MAX_MODEL_LEN: Final = 4096
HF_CACHE_DIR: Final = "/root/.cache/huggingface"
VLLM_CACHE_DIR: Final = "/root/.cache/vllm"
REMOTE_OUTPUT_DIR: Final = Path("/root/minicpmo45-smoke-output")
REMOTE_REPO_ROOT: Final = Path("/root/vllm-omni-local")
REMOTE_REQUIREMENTS_ROOT: Final = Path("/root/vllm-omni-requirements")
REMOTE_STAGE_CONFIG: Final = REMOTE_REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "minicpmo.yaml"
REMOTE_ASYNC_STAGE_CONFIG: Final = (
    REMOTE_REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "minicpmo_async_chunk.yaml"
)
MINUTES: Final = 60

LOCAL_REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_IGNORE = [
    ".git",
    ".cursor",
    ".venv",
    "build",
    "dist",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/*.pyc",
]


app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(f"{APP_NAME}-hf", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(f"{APP_NAME}-vllm", create_if_missing=True)
output_volume = modal.Volume.from_name(f"{APP_NAME}-output", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]==0.34.4",
    )
    .add_local_dir(
        str(LOCAL_REPO_ROOT / "requirements"),
        remote_path=str(REMOTE_REQUIREMENTS_ROOT),
        copy=True,
    )
    .run_commands(f"VLLM_OMNI_TARGET_DEVICE=cuda python -m pip install -r {REMOTE_REQUIREMENTS_ROOT / 'cuda.txt'}")
    .run_commands("python -m pip install --no-build-isolation 'minicpmo-utils[all]'")
    .run_commands("python -m pip install 'torchcodec'")
    .run_commands("python -m pip install --force-reinstall 'transformers==4.57.5'")
    .run_commands("python -m pip install --force-reinstall 'numpy==2.2.6' 'numba==0.61.2'")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONPATH": str(REMOTE_REPO_ROOT),
            "VLLM_CACHE_ROOT": VLLM_CACHE_DIR,
            "VLLM_OMNI_TARGET_DEVICE": "cuda",
        }
    )
    .add_local_dir(
        str(LOCAL_REPO_ROOT),
        remote_path=str(REMOTE_REPO_ROOT),
        copy=False,
        ignore=LOCAL_IGNORE,
    )
)


def _preview_text(text: str | None, limit: int = 240) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_audio_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    wav = np.asarray(payload["wav"], dtype=np.float32).reshape(-1)
    sr = int(payload["sr"])
    return {
        "num_samples": int(wav.shape[0]),
        "sample_rate": sr,
        "duration_sec": float(wav.shape[0] / max(sr, 1)),
    }


def _summarize_prompt(omni_prompt: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "keys": sorted(omni_prompt.keys()),
        "modalities": omni_prompt.get("modalities"),
    }
    if "prompt" in omni_prompt:
        prompt = omni_prompt["prompt"]
        summary["prompt_length"] = len(prompt)
        summary["prompt_preview"] = _preview_text(prompt)
    if "prompt_token_ids" in omni_prompt:
        token_ids = omni_prompt["prompt_token_ids"]
        summary["prompt_token_count"] = len(token_ids)
        summary["prompt_token_head"] = token_ids[:20]
        summary["prompt_token_tail"] = token_ids[-20:]
    additional_information = omni_prompt.get("additional_information")
    if isinstance(additional_information, dict):
        summary["additional_information"] = {
            "keys": sorted(additional_information.keys()),
            "ref_audio": _summarize_audio_payload(additional_information.get("ref_audio")),
        }
    multi_modal_data = omni_prompt.get("multi_modal_data")
    if isinstance(multi_modal_data, dict):
        mm_summary: dict[str, Any] = {"keys": sorted(multi_modal_data.keys())}
        audio = multi_modal_data.get("audio")
        if isinstance(audio, tuple) and len(audio) == 2:
            audio_np = np.asarray(audio[0], dtype=np.float32).reshape(-1)
            sr = int(audio[1])
            mm_summary["audio"] = {
                "num_samples": int(audio_np.shape[0]),
                "sample_rate": sr,
                "duration_sec": float(audio_np.shape[0] / max(sr, 1)),
            }
        elif isinstance(audio, list):
            mm_summary["audio_items"] = len(audio)
            if audio and isinstance(audio[0], tuple) and len(audio[0]) == 2:
                audio_np = np.asarray(audio[0][0], dtype=np.float32).reshape(-1)
                sr = int(audio[0][1])
                mm_summary["audio_first_item"] = {
                    "num_samples": int(audio_np.shape[0]),
                    "sample_rate": sr,
                    "duration_sec": float(audio_np.shape[0] / max(sr, 1)),
                }
        elif audio is not None:
            mm_summary["audio_type"] = type(audio).__name__
        summary["multi_modal_data"] = mm_summary
    return summary


def _summarize_sampling_params_list(sampling_params_list: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for params in sampling_params_list:
        summaries.append(
            {
                "temperature": getattr(params, "temperature", None),
                "top_p": getattr(params, "top_p", None),
                "top_k": getattr(params, "top_k", None),
                "max_tokens": getattr(params, "max_tokens", None),
                "detokenize": getattr(params, "detokenize", None),
                "repetition_penalty": getattr(params, "repetition_penalty", None),
                "stop_token_ids": getattr(params, "stop_token_ids", None),
            }
        )
    return summaries


def _summarize_audio_object(audio_obj: Any) -> dict[str, Any] | None:
    import torch

    if isinstance(audio_obj, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(audio_obj.shape),
            "dtype": str(audio_obj.dtype),
            "device": str(audio_obj.device),
        }
    if isinstance(audio_obj, list):
        tensor_parts = [part for part in audio_obj if isinstance(part, torch.Tensor)]
        return {
            "kind": "tensor_list",
            "parts": len(tensor_parts),
            "shapes": [list(part.shape) for part in tensor_parts[:8]],
        }
    return None


def _summarize_stage_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, stage_output in enumerate(outputs):
        final_output_type = getattr(stage_output, "final_output_type", None)
        request_output = getattr(stage_output, "request_output", None)
        summary: dict[str, Any] = {
            "index": index,
            "final_output_type": final_output_type,
            "has_request_output": request_output is not None,
        }
        if request_output is not None and getattr(request_output, "outputs", None):
            first_output = request_output.outputs[0]
            text = getattr(first_output, "text", None)
            if text is not None:
                summary["text_preview"] = _preview_text(text)
                summary["text_length"] = len(text)
            mm = getattr(first_output, "multimodal_output", None)
        else:
            mm = getattr(request_output, "multimodal_output", None) if request_output is not None else None
        if isinstance(mm, dict):
            summary["multimodal_keys"] = sorted(mm.keys())
            summary["audio"] = _summarize_audio_object(mm.get("audio"))
            sr = mm.get("sr")
            if hasattr(sr, "item"):
                sr = sr.item()
            if isinstance(sr, list) and sr:
                sr = sr[-1]
            if sr is not None:
                summary["sample_rate"] = int(sr)
        summaries.append(summary)
    return summaries


def _canonicalize_ref_audio_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None

    wav = np.asarray(payload["wav"], dtype=np.float32)
    if wav.ndim == 0:
        raise ValueError("Reference audio payload is empty.")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = wav.reshape(-1)
    if wav.size == 0:
        raise ValueError("Reference audio payload is empty.")

    return {
        "wav": wav.tolist(),
        "sr": int(payload["sr"]),
    }


def _load_code2wav_fixture_config(model_path: str) -> dict[str, Any]:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tts_config = getattr(hf_config, "tts_config", None)
    config = tts_config if tts_config is not None else hf_config
    num_audio_tokens = getattr(config, "num_audio_tokens", getattr(hf_config, "num_audio_tokens", None))
    if num_audio_tokens is None:
        raise RuntimeError("MiniCPM config is missing num_audio_tokens for the code2wav fixture.")

    return {
        "audio_eos_token_id": int(num_audio_tokens) - 1,
        "audio_prompt_sample_rate": int(getattr(config, "audio_tokenizer_sample_rate", 16000)),
        "output_sample_rate": 24000,
        "s3_stream_n_timesteps": int(getattr(config, "s3_stream_n_timesteps", 10) or 10),
    }


def _extract_code2wav_fixture(
    omni: Any,
    omni_prompt: dict[str, Any],
    *,
    model_path: str,
    prompt_text: str,
    ref_audio_route: str,
) -> dict[str, Any]:
    stage_list = getattr(getattr(omni, "engine", None), "stage_clients", None)
    if not isinstance(stage_list, list) or len(stage_list) < 3:
        raise RuntimeError(
            "MiniCPM smoke could not access the 3-stage engine state needed for code2wav fixture export."
        )

    code2wav_stage = stage_list[2]
    stage_prompts = code2wav_stage.process_engine_inputs(stage_list, prompt=omni_prompt)
    if not stage_prompts:
        raise RuntimeError("MiniCPM smoke produced no Stage 1 -> Stage 2 prompts to export.")

    requests: list[dict[str, Any]] = []
    for stage_prompt in stage_prompts:
        ref_audio = None
        additional_information = stage_prompt.get("additional_information")
        if isinstance(additional_information, dict):
            ref_audio = _canonicalize_ref_audio_payload(additional_information.get("ref_audio"))

        requests.append(
            {
                "prompt_token_ids": [int(token_id) for token_id in stage_prompt["prompt_token_ids"]],
                "additional_information": {"ref_audio": ref_audio} if ref_audio is not None else None,
            }
        )

    return {
        "version": 1,
        "model_id": MODEL_ID,
        "prompt_text": prompt_text,
        "ref_audio_route": ref_audio_route,
        "token2wav_assets_subdir": "assets/token2wav",
        "config": _load_code2wav_fixture_config(model_path),
        "requests": requests,
    }


def _summarize_code2wav_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    requests = fixture.get("requests", [])
    return {
        "num_requests": len(requests),
        "token_counts": [len(request.get("prompt_token_ids", [])) for request in requests],
        "has_ref_audio": [
            isinstance(request.get("additional_information"), dict)
            and request["additional_information"].get("ref_audio") is not None
            for request in requests
        ],
        "config": fixture.get("config"),
    }


def _record_debug(
    trace: list[dict[str, Any]] | None,
    *,
    stage: str,
    payload: dict[str, Any],
) -> None:
    event = {
        "stage": stage,
        "payload": payload,
    }
    print(json.dumps({"debug": event}, ensure_ascii=False, sort_keys=True))
    if trace is not None:
        trace.append(event)


def _list_relative_files(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(REMOTE_OUTPUT_DIR)) for path in root.rglob("*") if path.is_file())


def _download_model() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        resume_download=True,
    )


def _normalize_token_ids(tokenized_output: Any) -> list[int]:
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict) and "input_ids" in tokenized_output:
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"Expected token ids as list-like output, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids: list[int] = []
    for token_id in token_ids:
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        normalized_ids.append(int(token_id))
    return normalized_ids


def _build_ref_audio_system_prompt(
    system_profile: str | None = None,
    *,
    ref_audio_route: str = "multimodal",
) -> str:
    profile = (system_profile or "").strip()
    if profile:
        suffix = f"Please chat with the user in a highly human-like and oral style. {profile}"
    else:
        # Match HF get_sys_prompt(..., mode="omni") as closely as possible.
        suffix = "As an assistant, you will speak using this voice style."

    if ref_audio_route == "multimodal":
        return "\n".join(
            [
                "Clone the voice in the provided audio prompt.",
                "(<audio>./</audio>)",
                suffix,
            ]
        )
    if ref_audio_route == "stage2_only":
        return "\n".join(
            [
                "Respond to the user's request naturally in speech.",
                suffix,
            ]
        )
    raise ValueError(f"Unsupported ref_audio_route: {ref_audio_route}")


def _build_tts_prompt(
    model_path: str,
    text: str,
    ref_audio_asset: str | None = None,
    ref_audio_payload: dict[str, Any] | None = None,
    system_profile: str | None = None,
    ref_audio_route: str = "multimodal",
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm.assets.audio import AudioAsset

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if ref_audio_payload is not None or ref_audio_asset:
        if ref_audio_payload is not None:
            ref_audio = np.asarray(ref_audio_payload["wav"], dtype=np.float32).reshape(-1)
            ref_audio_sr = int(ref_audio_payload["sr"])
        else:
            ref_audio, ref_audio_sr = AudioAsset(ref_audio_asset).audio_and_sample_rate
            ref_audio = np.asarray(ref_audio, dtype=np.float32).reshape(-1)
        messages = [
            {
                "role": "system",
                "content": _build_ref_audio_system_prompt(
                    system_profile=system_profile,
                    ref_audio_route=ref_audio_route,
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]
        additional_information = {
            "ref_audio": {
                "wav": ref_audio.tolist(),
                "sr": int(ref_audio_sr),
            },
        }

        if ref_audio_route == "multimodal":
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    use_tts_template=True,
                    enable_thinking=False,
                )
            except TypeError as e:
                raise RuntimeError(
                    "MiniCPM tokenizer.apply_chat_template does not accept "
                    "the TTS template kwargs needed for speech mode."
                ) from e

            return {
                "prompt": prompt,
                "multi_modal_data": {
                    "audio": (ref_audio, int(ref_audio_sr)),
                },
                "additional_information": additional_information,
                "modalities": ["audio"],
            }

        if ref_audio_route == "stage2_only":
            try:
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    use_tts_template=True,
                    enable_thinking=False,
                )
            except TypeError as e:
                raise RuntimeError(
                    "MiniCPM tokenizer.apply_chat_template does not accept "
                    "the TTS template kwargs needed for speech mode."
                ) from e

            return {
                "prompt_token_ids": _normalize_token_ids(tokenized),
                "additional_information": additional_information,
                "modalities": ["audio"],
            }

        raise ValueError(f"Unsupported ref_audio_route: {ref_audio_route}")

    messages = [
        {
            "role": "system",
            "content": (
                "Respond to the user's request naturally in speech. "
                "Reply with speech only and do not add extra meta commentary."
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]
    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            use_tts_template=True,
            enable_thinking=False,
        )
    except TypeError as e:
        raise RuntimeError(
            "MiniCPM tokenizer.apply_chat_template does not accept the TTS template kwargs needed for speech mode."
        ) from e

    return {
        "prompt_token_ids": _normalize_token_ids(tokenized),
        "modalities": ["audio"],
    }


def _load_local_ref_audio(ref_audio_path: str) -> dict[str, Any]:
    import soundfile as sf

    path = Path(ref_audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference audio file not found: {path}")

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav_np = np.asarray(wav, dtype=np.float32)
    if wav_np.ndim == 0:
        raise ValueError(f"Reference audio at {path} is empty.")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=-1)

    wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
    if wav_np.size == 0:
        raise ValueError(f"Reference audio at {path} is empty.")

    return {
        "wav": wav_np.tolist(),
        "sr": int(sr),
    }


def _load_local_code2wav_fixture(fixture_path: str) -> dict[str, Any]:
    path = Path(fixture_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Code2wav fixture file not found: {path}")
    return json.loads(path.read_text())


def _get_code2wav_request_fixture(fixture: dict[str, Any], request_index: int) -> dict[str, Any]:
    requests = fixture.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Fixture is missing a non-empty 'requests' list.")
    if request_index < 0 or request_index >= len(requests):
        raise IndexError(f"Invalid request index {request_index}; fixture has {len(requests)} request(s).")
    request = requests[request_index]
    if "prompt_token_ids" not in request:
        raise ValueError("Fixture request is missing 'prompt_token_ids'.")
    return request


def _resolve_code2wav_config(fixture: dict[str, Any]) -> dict[str, int]:
    fixture_config = fixture.get("config") if isinstance(fixture.get("config"), dict) else {}
    return {
        "audio_eos_token_id": int(fixture_config.get("audio_eos_token_id", 6561)),
        "audio_prompt_sample_rate": int(fixture_config.get("audio_prompt_sample_rate", 16000)),
        "output_sample_rate": int(fixture_config.get("output_sample_rate", 24000)),
        "s3_stream_n_timesteps": int(fixture_config.get("s3_stream_n_timesteps", 10) or 10),
    }


def _patch_minicpm_audio_io(audio_prompt_sample_rate: int) -> None:
    import io

    import soundfile as sf
    import torch

    patch_targets: list[object] = []
    torchaudio_targets: list[object] = []
    direct_audio_io_targets: list[object] = []

    try:
        import s3tokenizer as s3_pkg

        patch_targets.append(s3_pkg)
    except ImportError:
        pass

    try:
        import s3tokenizer.utils as s3_utils

        patch_targets.append(s3_utils)
    except ImportError:
        pass

    try:
        import stepaudio2.token2wav as token2wav_mod

        if hasattr(token2wav_mod, "s3tokenizer"):
            patch_targets.append(token2wav_mod.s3tokenizer)
        patch_targets.append(token2wav_mod)
        direct_audio_io_targets.append(token2wav_mod)
        if hasattr(token2wav_mod, "torchaudio"):
            torchaudio_targets.append(token2wav_mod.torchaudio)
    except ImportError:
        token2wav_mod = None

    try:
        import torchaudio as torchaudio_mod

        torchaudio_targets.append(torchaudio_mod)
    except ImportError:
        torchaudio_mod = None

    if token2wav_mod is None and torchaudio_mod is None and not patch_targets:
        return

    def _load_audio(file: str | None, sr: int = 16000) -> torch.Tensor:
        if file is None:
            return torch.zeros((0,), dtype=torch.float32)

        audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        audio_np = audio_np.reshape(-1)
        if int(sample_rate) != int(sr):
            import librosa

            audio_np = librosa.resample(y=audio_np, orig_sr=int(sample_rate), target_sr=int(sr))
        return torch.from_numpy(np.asarray(audio_np, dtype=np.float32))

    def _torchaudio_load(file: str | None, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, int]:
        if file is None:
            audio_np = np.zeros((audio_prompt_sample_rate,), dtype=np.float32)
            sample_rate = audio_prompt_sample_rate
        else:
            audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
            audio_np = np.asarray(audio, dtype=np.float32)

        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        elif audio_np.ndim > 1:
            audio_np = np.asarray(audio_np, dtype=np.float32).T

        return torch.from_numpy(np.ascontiguousarray(audio_np, dtype=np.float32)), int(sample_rate)

    def _torchaudio_save(file: str | io.BytesIO, src: Any, sample_rate: int, *args: Any, **kwargs: Any) -> None:
        audio_np = np.asarray(src.detach().cpu().numpy(), dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        elif audio_np.ndim != 1:
            raise ValueError(f"Expected 1-D or 2-D audio tensor, got shape {tuple(audio_np.shape)}")
        sf.write(file, audio_np, int(sample_rate), format=kwargs.pop("format", None))

    seen: set[int] = set()
    for target in patch_targets:
        if id(target) in seen:
            continue
        seen.add(id(target))
        setattr(target, "load_audio", _load_audio)

    seen.clear()
    for target in torchaudio_targets:
        if id(target) in seen:
            continue
        seen.add(id(target))
        setattr(target, "load", _torchaudio_load)
        setattr(target, "save", _torchaudio_save)

    seen.clear()
    for target in direct_audio_io_targets:
        if id(target) in seen:
            continue
        seen.add(id(target))
        setattr(target, "load", _torchaudio_load)
        setattr(target, "save", _torchaudio_save)


def _write_code2wav_prompt_wav(ref_audio: dict[str, Any] | None, target_sr: int) -> str:
    import tempfile

    import soundfile as sf

    if ref_audio is None:
        wav_np = np.zeros((target_sr,), dtype=np.float32)
        sr = target_sr
    else:
        canonical = _canonicalize_ref_audio_payload(ref_audio)
        assert canonical is not None
        wav_np = np.asarray(canonical["wav"], dtype=np.float32).reshape(-1)
        sr = int(canonical["sr"])
        if sr != target_sr:
            import librosa

            wav_np = librosa.resample(y=wav_np, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    with tempfile.NamedTemporaryFile(prefix="minicpm_ref_", suffix=".wav", delete=False) as f:
        prompt_wav_path = f.name
    sf.write(prompt_wav_path, wav_np, sr)
    return prompt_wav_path


def _decode_code2wav_one(
    token2wav: Any,
    token_ids: list[int],
    *,
    ref_audio: dict[str, Any] | None,
    audio_eos_token_id: int,
    audio_prompt_sample_rate: int,
    output_sample_rate: int,
) -> tuple[np.ndarray, int]:
    import io

    import soundfile as sf

    trimmed_token_ids = list(token_ids)
    while trimmed_token_ids and trimmed_token_ids[-1] == audio_eos_token_id:
        trimmed_token_ids.pop()

    if not trimmed_token_ids:
        return np.zeros((0,), dtype=np.float32), output_sample_rate

    prompt_wav_path = _write_code2wav_prompt_wav(ref_audio, audio_prompt_sample_rate)
    try:
        # stepaudio2.Token2wav caches prompt conditioning on the instance.
        # Reset it so each decode reflects the prompt wav we just wrote.
        token2wav.cache = None
        wav_bytes = token2wav(trimmed_token_ids, prompt_wav_path)
    finally:
        Path(prompt_wav_path).unlink(missing_ok=True)

    waveform, sample_rate = sf.read(io.BytesIO(wav_bytes))
    waveform_np = np.asarray(waveform, dtype=np.float32)
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=-1)
    return waveform_np.reshape(-1), int(sample_rate)


def _save_code2wav_output(path: Path, waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate, format="WAV", subtype="PCM_16")
    return {
        "remote_path": path.name,
        "num_samples": int(waveform.shape[0]),
        "sample_rate": int(sample_rate),
        "duration_sec": float(waveform.shape[0] / max(sample_rate, 1)),
    }


def _extract_text_and_audio(outputs: list[Any]) -> tuple[str | None, Any, int]:
    import torch

    text_output: str | None = None
    audio_tensor = None
    sample_rate = 24000

    for stage_output in outputs:
        final_output_type = getattr(stage_output, "final_output_type", None)
        request_output = getattr(stage_output, "request_output", None)
        if request_output is None:
            continue

        if final_output_type == "text" and getattr(request_output, "outputs", None):
            text_output = request_output.outputs[0].text
            continue

        if final_output_type != "audio":
            continue

        mm = getattr(request_output, "multimodal_output", None)
        if not mm and getattr(request_output, "outputs", None):
            mm = getattr(request_output.outputs[0], "multimodal_output", None)
        if not isinstance(mm, dict):
            continue

        audio_obj = mm.get("audio")
        sr_obj = mm.get("sr", sample_rate)

        if isinstance(sr_obj, list) and sr_obj:
            sr_obj = sr_obj[-1]
        if hasattr(sr_obj, "item"):
            sr_obj = sr_obj.item()
        sample_rate = int(sr_obj)

        if isinstance(audio_obj, list):
            audio_parts = [part for part in audio_obj if isinstance(part, torch.Tensor)]
            if audio_parts:
                audio_tensor = torch.cat(audio_parts, dim=-1)
        elif isinstance(audio_obj, torch.Tensor):
            audio_tensor = audio_obj

    if audio_tensor is None:
        raise RuntimeError("MiniCPMO TTS smoke returned no final audio output.")

    return text_output, audio_tensor, sample_rate


def _best_effort_close_omni(omni: Any, timeout_s: float = 15.0) -> None:
    close_error: list[BaseException] = []

    def _run_close() -> None:
        try:
            omni.close()
        except BaseException as e:
            close_error.append(e)

    close_thread = threading.Thread(
        target=_run_close,
        name="minicpmo45-smoke-close",
        daemon=True,
    )
    close_thread.start()
    close_thread.join(timeout_s)

    if close_thread.is_alive():
        finalizer = getattr(omni, "_weak_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            finalizer.detach()
        print(f"MiniCPM smoke: omni.close() exceeded {timeout_s:.1f}s; returning without waiting for full teardown.")
        return

    if close_error:
        raise close_error[0]


@app.function(
    image=image,
    timeout=30 * MINUTES,
    volumes={HF_CACHE_DIR: hf_cache_volume},
)
def preload_model() -> str:
    snapshot_path = _download_model()
    hf_cache_volume.commit()
    return snapshot_path


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    timeout=30 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        VLLM_CACHE_DIR: vllm_cache_volume,
        str(REMOTE_OUTPUT_DIR): output_volume,
    },
)
def run_code2wav_from_fixture(
    fixture: dict[str, Any],
    request_index: int = 0,
    ref_audio_payload: dict[str, Any] | None = None,
    remote_output_name: str | None = None,
    remote_with_ref_output_name: str | None = None,
    remote_without_ref_output_name: str | None = None,
) -> dict[str, Any]:
    from stepaudio2 import Token2wav

    snapshot_path = _download_model()
    hf_cache_volume.commit()

    request = _get_code2wav_request_fixture(fixture, request_index)
    config = _resolve_code2wav_config(fixture)
    token_ids = [int(token_id) for token_id in request["prompt_token_ids"]]
    fixture_ref_audio = None
    request_info = request.get("additional_information")
    if isinstance(request_info, dict):
        fixture_ref_audio = _canonicalize_ref_audio_payload(request_info.get("ref_audio"))
    override_ref_audio = _canonicalize_ref_audio_payload(ref_audio_payload)
    effective_ref_audio = override_ref_audio if override_ref_audio is not None else fixture_ref_audio

    _patch_minicpm_audio_io(config["audio_prompt_sample_rate"])
    token2wav_assets_dir = Path(snapshot_path) / fixture.get("token2wav_assets_subdir", "assets/token2wav")
    token2wav = Token2wav(
        str(token2wav_assets_dir),
        float16=False,
        n_timesteps=config["s3_stream_n_timesteps"],
    )

    outputs: list[dict[str, Any]] = []
    if not any([remote_output_name, remote_with_ref_output_name, remote_without_ref_output_name]):
        remote_output_name = f"{uuid.uuid4().hex}.wav"

    if remote_output_name is not None:
        waveform, sample_rate = _decode_code2wav_one(
            token2wav,
            token_ids,
            ref_audio=effective_ref_audio,
            audio_eos_token_id=config["audio_eos_token_id"],
            audio_prompt_sample_rate=config["audio_prompt_sample_rate"],
            output_sample_rate=config["output_sample_rate"],
        )
        output_path = REMOTE_OUTPUT_DIR / remote_output_name
        result = _save_code2wav_output(output_path, waveform, sample_rate)
        result["mode"] = "fixture_ref" if effective_ref_audio is not None else "no_ref"
        outputs.append(result)

    if remote_with_ref_output_name is not None:
        if effective_ref_audio is None:
            raise ValueError(
                "No reference audio available for with-ref decode. "
                "Provide --ref-audio-path or use a fixture containing ref_audio."
            )
        waveform, sample_rate = _decode_code2wav_one(
            token2wav,
            token_ids,
            ref_audio=effective_ref_audio,
            audio_eos_token_id=config["audio_eos_token_id"],
            audio_prompt_sample_rate=config["audio_prompt_sample_rate"],
            output_sample_rate=config["output_sample_rate"],
        )
        output_path = REMOTE_OUTPUT_DIR / remote_with_ref_output_name
        result = _save_code2wav_output(output_path, waveform, sample_rate)
        result["mode"] = "with_ref"
        outputs.append(result)

    if remote_without_ref_output_name is not None:
        waveform, sample_rate = _decode_code2wav_one(
            token2wav,
            token_ids,
            ref_audio=None,
            audio_eos_token_id=config["audio_eos_token_id"],
            audio_prompt_sample_rate=config["audio_prompt_sample_rate"],
            output_sample_rate=config["output_sample_rate"],
        )
        output_path = REMOTE_OUTPUT_DIR / remote_without_ref_output_name
        result = _save_code2wav_output(output_path, waveform, sample_rate)
        result["mode"] = "without_ref"
        outputs.append(result)

    output_volume.commit()
    return {
        "snapshot_path": snapshot_path,
        "request_index": request_index,
        "token_count": len(token_ids),
        "assets_dir": str(token2wav_assets_dir),
        "config": config,
        "has_fixture_ref_audio": fixture_ref_audio is not None,
        "used_override_ref_audio": override_ref_audio is not None,
        "outputs": outputs,
    }


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    timeout=45 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        VLLM_CACHE_DIR: vllm_cache_volume,
        str(REMOTE_OUTPUT_DIR): output_volume,
    },
)
def run_tts_smoke(
    prompt: str = "Please read this sentence aloud: vLLM Omni is testing MiniCPM text to audio generation.",
    ref_audio_asset: str | None = None,
    ref_audio_payload: dict[str, Any] | None = None,
    ref_audio_label: str | None = None,
    system_profile: str | None = None,
    ref_audio_route: str = "multimodal",
    remote_output_name: str | None = None,
    remote_code2wav_fixture_name: str | None = None,
    artifact_dir_name: str | None = None,
    async_chunk: bool = False,
    debug: bool = False,
    debug_ref_audio: bool = False,
) -> dict[str, Any]:
    import os

    import soundfile as sf
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    snapshot_path = _download_model()
    hf_cache_volume.commit()
    debug_trace = [] if debug else None
    dump_code2wav_fixture = remote_code2wav_fixture_name is not None
    if debug_ref_audio:
        os.environ["VLLM_OMNI_MINICPMO45_DEBUG_REF_AUDIO"] = "1"
    else:
        os.environ.pop("VLLM_OMNI_MINICPMO45_DEBUG_REF_AUDIO", None)
    artifact_dir = REMOTE_OUTPUT_DIR / artifact_dir_name if artifact_dir_name else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MINICPMO45_E2E_OUTPUT_DIR"] = str(artifact_dir)
    else:
        os.environ.pop("MINICPMO45_E2E_OUTPUT_DIR", None)

    omni_prompt = _build_tts_prompt(
        snapshot_path,
        prompt,
        ref_audio_asset=ref_audio_asset,
        ref_audio_payload=ref_audio_payload,
        system_profile=system_profile,
        ref_audio_route=ref_audio_route,
    )

    if debug:
        _record_debug(
            debug_trace,
            stage="prompt",
            payload={
                "prompt_text_preview": _preview_text(prompt),
                "ref_audio_asset": ref_audio_asset,
                "ref_audio_label": ref_audio_label,
                "ref_audio_payload": _summarize_audio_payload(ref_audio_payload),
                "omni_prompt": _summarize_prompt(omni_prompt),
            },
        )

    sampling_params_list = [
        SamplingParams(
            temperature=0.7,
            top_p=1.0,
            top_k=50,
            max_tokens=4096,
            detokenize=True,
        ),
        SamplingParams(
            temperature=0.9,
            top_k=50,
            max_tokens=4096,
            detokenize=False,
            repetition_penalty=1.05,
            stop_token_ids=[6561],
        ),
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=65536,
            detokenize=True,
        ),
    ]

    if debug:
        _record_debug(
            debug_trace,
            stage="sampling_params",
            payload={
                "sampling_params_list": _summarize_sampling_params_list(sampling_params_list),
            },
        )

    omni = Omni(
        model=snapshot_path,
        stage_configs_path=str(REMOTE_ASYNC_STAGE_CONFIG if async_chunk else REMOTE_STAGE_CONFIG),
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        stage_init_timeout=20 * MINUTES,
        init_timeout=30 * MINUTES,
        output_modalities=["audio"],
        log_stats=True,
    )

    try:
        outputs = list(omni.generate(omni_prompt, sampling_params_list=sampling_params_list, use_tqdm=False))
        if not outputs:
            raise RuntimeError("MiniCPMO TTS smoke returned no stage outputs.")

        if debug:
            _record_debug(
                debug_trace,
                stage="stage_outputs",
                payload={
                    "outputs": _summarize_stage_outputs(outputs),
                },
            )

        code2wav_fixture = None
        if dump_code2wav_fixture or debug:
            try:
                code2wav_fixture = _extract_code2wav_fixture(
                    omni,
                    omni_prompt,
                    model_path=snapshot_path,
                    prompt_text=prompt,
                    ref_audio_route=ref_audio_route,
                )
            except Exception as e:
                if dump_code2wav_fixture:
                    raise
                if debug:
                    _record_debug(
                        debug_trace,
                        stage="code2wav_fixture_error",
                        payload={"error": str(e)},
                    )
            else:
                if debug:
                    _record_debug(
                        debug_trace,
                        stage="code2wav_fixture",
                        payload=_summarize_code2wav_fixture(code2wav_fixture),
                    )

        text_output, audio_tensor, sample_rate = _extract_text_and_audio(outputs)
        audio_np = audio_tensor.detach().cpu().float().numpy().reshape(-1)
        duration_sec = float(audio_np.shape[0] / max(sample_rate, 1))

        output_name = remote_output_name or f"{uuid.uuid4().hex}.wav"
        remote_output_path = REMOTE_OUTPUT_DIR / output_name
        remote_output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(remote_output_path), audio_np, sample_rate, format="WAV", subtype="PCM_16")
        remote_code2wav_fixture_path = None
        if dump_code2wav_fixture and code2wav_fixture is not None:
            remote_code2wav_fixture_path = REMOTE_OUTPUT_DIR / remote_code2wav_fixture_name
            remote_code2wav_fixture_path.parent.mkdir(parents=True, exist_ok=True)
            remote_code2wav_fixture_path.write_text(
                json.dumps(code2wav_fixture, indent=2, ensure_ascii=False, sort_keys=True)
            )
        output_volume.commit()

        result = {
            "snapshot_path": snapshot_path,
            "text_output": text_output,
            "sample_rate": sample_rate,
            "num_samples": int(audio_np.shape[0]),
            "duration_sec": duration_sec,
            "prompt_token_count": len(omni_prompt["prompt_token_ids"]) if "prompt_token_ids" in omni_prompt else None,
            "ref_audio_asset": ref_audio_asset,
            "ref_audio_label": ref_audio_label,
            "system_profile": system_profile,
            "ref_audio_route": ref_audio_route,
            "remote_output_path": output_name,
            "artifact_dir_name": artifact_dir_name,
            "artifact_paths": _list_relative_files(artifact_dir) if artifact_dir is not None else [],
            "async_chunk": bool(async_chunk),
        }
        if remote_code2wav_fixture_path is not None:
            result["remote_code2wav_fixture_path"] = remote_code2wav_fixture_name
        if debug:
            _record_debug(
                debug_trace,
                stage="final_output",
                payload={
                    "text_output_preview": _preview_text(text_output),
                    "text_output_length": len(text_output) if text_output is not None else None,
                    "audio": {
                        "num_samples": int(audio_np.shape[0]),
                        "sample_rate": int(sample_rate),
                        "duration_sec": duration_sec,
                    },
                    "output_path": str(remote_output_path),
                },
            )
        if debug_trace is not None:
            result["debug_trace"] = debug_trace
        return result
    finally:
        _best_effort_close_omni(omni)


@app.local_entrypoint()
def main(
    prompt: str = "Please read this sentence aloud: vLLM Omni is testing MiniCPM text to audio generation.",
    output: str = "minicpmo45_tts_smoke.wav",
    artifact_output_dir: str = "",
    code2wav_fixture_output: str = "",
    code2wav_fixture_input: str = "",
    code2wav_request_index: int = 0,
    code2wav_output: str = "",
    with_ref_output: str = "",
    without_ref_output: str = "",
    ref_audio_asset: str = "",
    ref_audio_path: str = "",
    system_profile: str = "",
    ref_audio_route: str = "multimodal",
    async_chunk: bool = False,
    preload_only: bool = False,
    debug: bool = False,
    debug_ref_audio: bool = False,
) -> None:
    snapshot_path = preload_model.remote()
    print(f"Model cached at: {snapshot_path}")

    if preload_only:
        return

    if ref_audio_asset and ref_audio_path:
        raise ValueError("Use only one of --ref-audio-asset or --ref-audio-path.")

    ref_audio_payload: dict[str, Any] | None = None
    ref_audio_label: str | None = None
    if ref_audio_path:
        ref_audio_payload = _load_local_ref_audio(ref_audio_path)
        ref_audio_label = str(Path(ref_audio_path).expanduser().resolve())
    elif ref_audio_asset:
        ref_audio_label = ref_audio_asset

    if code2wav_fixture_input:
        if preload_only:
            return
        fixture = _load_local_code2wav_fixture(code2wav_fixture_input)
        remote_output_name = f"{uuid.uuid4().hex}_{Path(code2wav_output).name}" if code2wav_output else None
        remote_with_ref_output_name = f"{uuid.uuid4().hex}_{Path(with_ref_output).name}" if with_ref_output else None
        remote_without_ref_output_name = (
            f"{uuid.uuid4().hex}_{Path(without_ref_output).name}" if without_ref_output else None
        )
        result = run_code2wav_from_fixture.remote(
            fixture=fixture,
            request_index=code2wav_request_index,
            ref_audio_payload=ref_audio_payload,
            remote_output_name=remote_output_name,
            remote_with_ref_output_name=remote_with_ref_output_name,
            remote_without_ref_output_name=remote_without_ref_output_name,
        )
        for output_info in result["outputs"]:
            remote_name = output_info["remote_path"]
            mode = output_info["mode"]
            if mode in {"fixture_ref", "no_ref"}:
                local_path_str = code2wav_output or "minicpmo45_code2wav.wav"
            elif mode == "with_ref":
                local_path_str = with_ref_output
            elif mode == "without_ref":
                local_path_str = without_ref_output
            else:
                raise RuntimeError(f"Unexpected code2wav output mode: {mode}")

            local_path = Path(local_path_str)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            wav_bytes = b"".join(output_volume.read_file(remote_name))
            local_path.write_bytes(wav_bytes)
            output_info["local_path"] = str(local_path.resolve())

        print(json.dumps(result, indent=2, sort_keys=True))
        return

    remote_output_name = f"{uuid.uuid4().hex}_{Path(output).name}"
    remote_code2wav_fixture_name = (
        f"{uuid.uuid4().hex}_{Path(code2wav_fixture_output).name}" if code2wav_fixture_output else None
    )
    artifact_dir_name = f"{uuid.uuid4().hex}_artifacts" if artifact_output_dir else None
    result = run_tts_smoke.remote(
        prompt=prompt,
        ref_audio_asset=ref_audio_asset or None,
        ref_audio_payload=ref_audio_payload,
        ref_audio_label=ref_audio_label,
        system_profile=system_profile or None,
        ref_audio_route=ref_audio_route,
        remote_output_name=remote_output_name,
        remote_code2wav_fixture_name=remote_code2wav_fixture_name,
        artifact_dir_name=artifact_dir_name,
        async_chunk=async_chunk,
        debug=debug,
        debug_ref_audio=debug_ref_audio,
    )
    remote_output_path = result.pop("remote_output_path")
    debug_trace = result.pop("debug_trace", None)
    remote_code2wav_fixture_path = result.pop("remote_code2wav_fixture_path", None)
    remote_artifact_dir_name = result.pop("artifact_dir_name", None)
    artifact_paths = result.pop("artifact_paths", [])
    wav_bytes = b"".join(output_volume.read_file(remote_output_path))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)

    if code2wav_fixture_output:
        if remote_code2wav_fixture_path is None:
            raise RuntimeError("MiniCPM smoke did not return a code2wav fixture.")
        fixture_output_path = Path(code2wav_fixture_output)
        fixture_output_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_bytes = b"".join(output_volume.read_file(remote_code2wav_fixture_path))
        fixture_output_path.write_bytes(fixture_bytes)
        print(f"Saved code2wav fixture to: {fixture_output_path}")

    if artifact_output_dir and remote_artifact_dir_name:
        local_artifact_root = Path(artifact_output_dir)
        local_artifact_root.mkdir(parents=True, exist_ok=True)
        for remote_path in artifact_paths:
            rel_path = Path(remote_path).relative_to(remote_artifact_dir_name)
            local_path = local_artifact_root / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"".join(output_volume.read_file(remote_path)))
        print(f"Saved async debug artifacts to: {local_artifact_root}")

    print(json.dumps(result, indent=2, sort_keys=True))
    if debug_trace is not None:
        print(json.dumps({"debug_trace": debug_trace}, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Saved generated audio to: {output_path}")
