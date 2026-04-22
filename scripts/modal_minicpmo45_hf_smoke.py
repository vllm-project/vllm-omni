#!/usr/bin/env python3
"""Run the official HuggingFace MiniCPM-o 4.5 path on Modal.

Usage:
    modal run scripts/modal_minicpmo45_hf_smoke.py --preload-only
    modal run scripts/modal_minicpmo45_hf_smoke.py \
        --ref-audio-path scripts/ref.mp3 \
        --output minicpmo45_hf_ref.wav \
        --prompt "Please answer this question naturally in the reference voice."

This script is meant for debugging and side-by-side comparison against the
native vLLM-Omni MiniCPM pipeline. It uses the official
``transformers.AutoModel(...).chat(...)`` path from the model card.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, Final

import modal
import numpy as np
import soundfile as sf

APP_NAME: Final = "minicpmo45-hf-smoke"
MODEL_ID: Final = "openbmb/MiniCPM-o-4_5"
GPU_REQUEST: Final = "A100-80GB:1"
HF_CACHE_DIR: Final = "/root/.cache/huggingface"
REMOTE_OUTPUT_DIR: Final = Path("/root/minicpmo45-hf-output")
PYTORCH_CUDA_INDEX: Final = "https://download.pytorch.org/whl/cu128"
MINUTES: Final = 60


app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(f"{APP_NAME}-hf", create_if_missing=True)
output_volume = modal.Volume.from_name(f"{APP_NAME}-output", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .run_commands(
        f"python -m pip install --force-reinstall --index-url {PYTORCH_CUDA_INDEX} torch torchvision torchaudio"
    )
    .pip_install(
        "huggingface_hub[hf_transfer]==0.34.4",
        "soxr",
    )
    .run_commands("python -m pip install --no-build-isolation 'minicpmo-utils[all]'")
    .run_commands("python -m pip install 'torchcodec'")
    .run_commands("python -m pip install --force-reinstall 'numpy==2.2.6' 'numba==0.61.2'")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_CACHE": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
)


def _preview_text(text: str | None, limit: int = 240) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_audio_array(audio: np.ndarray | None, *, sample_rate: int | None = None) -> dict[str, Any] | None:
    if audio is None:
        return None
    num_samples = int(audio.shape[0])
    summary: dict[str, Any] = {
        "num_samples": num_samples,
        "dtype": str(audio.dtype),
    }
    if sample_rate is not None:
        summary["sample_rate"] = int(sample_rate)
        summary["duration_sec"] = float(num_samples / max(int(sample_rate), 1))
    return summary


def _summarize_message_content_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {
            "type": "text",
            "length": len(item),
            "preview": _preview_text(item, limit=160),
        }
    if isinstance(item, np.ndarray):
        return {
            "type": "audio_array",
            **(_summarize_audio_array(item) or {}),
        }
    return {
        "type": type(item).__name__,
    }


def _summarize_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, list):
        content_summary = [_summarize_message_content_item(item) for item in content]
    else:
        content_summary = [_summarize_message_content_item(content)]
    return {
        "role": message.get("role"),
        "content": content_summary,
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _summarize_tensor(tensor: Any) -> dict[str, Any]:
    import torch

    tensor_cpu = torch.as_tensor(tensor).detach().cpu().contiguous()
    tensor_buffer = io.BytesIO()
    torch.save(tensor_cpu, tensor_buffer)
    summary: dict[str, Any] = {
        "shape": list(tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "numel": int(tensor_cpu.numel()),
        "sha256": hashlib.sha256(tensor_buffer.getvalue()).hexdigest(),
    }
    if tensor_cpu.numel() == 0:
        return summary

    if tensor_cpu.is_floating_point():
        values = tensor_cpu.to(torch.float32)
        summary.update(
            {
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "l2_norm": float(torch.linalg.vector_norm(values).item()),
            }
        )
    else:
        summary.update(
            {
                "min": int(tensor_cpu.min().item()),
                "max": int(tensor_cpu.max().item()),
            }
        )

    return summary


def _write_tensor_dump(path: Path, tensor: Any) -> dict[str, Any]:
    import torch

    tensor_cpu = torch.as_tensor(tensor).detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_cpu, path)
    summary = _summarize_tensor(tensor_cpu)
    _write_json(path.with_suffix(".summary.json"), summary)
    return summary


def _relative_to_root(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _call_with_supported_kwargs(func: Any, /, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_var_kwargs:
        return func(**kwargs)

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return func(**filtered_kwargs)


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
    raise ValueError(f"Unsupported streamed audio chunk shape: {tuple(audio_np.shape)}")


def _coerce_token_chunk(chunk: Any) -> list[int]:
    import torch

    if chunk is None:
        return []
    if isinstance(chunk, torch.Tensor):
        token_np = chunk.detach().cpu().to(dtype=torch.long).numpy()
    else:
        token_np = np.asarray(chunk)
    token_np = np.asarray(token_np).reshape(-1)
    return [int(token) for token in token_np.tolist()]


def _load_local_blob(path_str: str) -> bytes:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    return path.read_bytes()


def _topk_logits_summary(logits: Any, probs: Any, k: int = 8) -> list[dict[str, Any]]:
    import torch

    logits_tensor = torch.as_tensor(logits)
    probs_tensor = torch.as_tensor(probs)
    if logits_tensor.ndim != 2 or probs_tensor.ndim != 2 or logits_tensor.shape != probs_tensor.shape:
        return []
    vocab_size = int(logits_tensor.shape[-1])
    if vocab_size <= 0:
        return []
    k = max(1, min(int(k), vocab_size))
    top_vals, top_ids = torch.topk(logits_tensor, k=k, dim=-1)
    top_probs = probs_tensor.gather(-1, top_ids)
    rows: list[dict[str, Any]] = []
    for token_id, logit_val, prob_val in zip(top_ids[0], top_vals[0], top_probs[0], strict=False):
        rows.append(
            {
                "token_id": int(token_id.item()),
                "logit": float(logit_val.item()),
                "prob": float(prob_val.item()),
            }
        )
    return rows


def _best_effort_prompt_token_ids(snapshot_path: str, msgs: list[dict[str, Any]]) -> list[int] | None:
    from transformers import AutoTokenizer

    text_only_msgs: list[dict[str, Any]] = []
    for msg in msgs:
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        if not all(isinstance(item, str) for item in content):
            return None
        text_only_msgs.append(
            {
                "role": msg.get("role"),
                "content": "".join(content),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)
    try:
        tokenized = tokenizer.apply_chat_template(
            text_only_msgs,
            tokenize=True,
            add_generation_prompt=True,
            use_tts_template=True,
            enable_thinking=False,
        )
    except Exception:
        return None

    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, list) and tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    if not isinstance(tokenized, list):
        return None
    return [int(token_id) for token_id in tokenized]


def _resolve_streaming_trace_class(
    owner_globals: dict[str, Any],
    *,
    preferred_name: str,
    required_attr: str,
) -> tuple[type[Any] | None, dict[str, Any]]:
    preferred = owner_globals.get(preferred_name)
    if inspect.isclass(preferred) and hasattr(preferred, required_attr):
        return preferred, {
            "resolution": "preferred_name",
            "global_name": preferred_name,
            "class_name": preferred.__name__,
            "required_attr": required_attr,
        }

    name_matches: list[tuple[str, type[Any]]] = []
    attr_matches: list[tuple[str, type[Any]]] = []
    preferred_name_lower = preferred_name.lower()
    for global_name, value in owner_globals.items():
        if not inspect.isclass(value) or not hasattr(value, required_attr):
            continue
        attr_matches.append((global_name, value))
        if value.__name__ == preferred_name or preferred_name_lower in value.__name__.lower():
            name_matches.append((global_name, value))

    candidates = name_matches or attr_matches
    if candidates:
        global_name, value = candidates[0]
        return value, {
            "resolution": "fallback_search",
            "global_name": global_name,
            "class_name": value.__name__,
            "required_attr": required_attr,
            "candidate_names": [name for name, _ in candidates[:8]],
        }

    return None, {
        "resolution": "not_found",
        "preferred_name": preferred_name,
        "required_attr": required_attr,
        "available_candidates": [
            name for name, value in owner_globals.items() if inspect.isclass(value) and hasattr(value, required_attr)
        ][:16],
    }


def _inspect_streaming_trace_targets(model: Any) -> tuple[type[Any] | None, type[Any] | None, dict[str, Any]]:
    streaming_method = getattr(model, "streaming_generate", None)
    if streaming_method is None:
        return (
            None,
            None,
            {
                "has_streaming_generate": False,
            },
        )

    streaming_func = inspect.unwrap(getattr(streaming_method, "__func__", streaming_method))
    owner_globals = getattr(streaming_func, "__globals__", {})
    try:
        source_text = inspect.getsource(streaming_func)
    except (OSError, TypeError):
        source_text = None
    source_focus = None
    if source_text is not None:
        source_focus = [
            line.rstrip()
            for line in source_text.splitlines()
            if any(
                marker in line
                for marker in (
                    "ChunkPrefillChunkGenerate",
                    "TTSStreamingGenerator",
                    "yield_chunk_token_ids",
                    "generate_with_buffer",
                    "import",
                    "tts_generator",
                )
            )
        ][:24]
    chunk_cls, chunk_meta = _resolve_streaming_trace_class(
        owner_globals,
        preferred_name="ChunkPrefillChunkGenerate",
        required_attr="chunk_generate",
    )
    tts_cls, tts_meta = _resolve_streaming_trace_class(
        owner_globals,
        preferred_name="TTSStreamingGenerator",
        required_attr="generate_with_buffer",
    )
    return (
        chunk_cls,
        tts_cls,
        {
            "has_streaming_generate": True,
            "streaming_generate_module": getattr(streaming_func, "__module__", None),
            "streaming_generate_qualname": getattr(streaming_func, "__qualname__", None),
            "streaming_generate_source_file": inspect.getsourcefile(streaming_func),
            "streaming_generate_source_focus": source_focus,
            "chunk_target": chunk_meta,
            "tts_target": tts_meta,
        },
    )


def _install_streaming_trace_hooks(
    model: Any,
    *,
    artifact_dir: Path | None,
    tokenizer: Any | None,
    thinker_rows: list[dict[str, Any]],
    talker_rows: list[dict[str, Any]],
) -> tuple[tuple[type[Any], str, Any], tuple[type[Any], str, Any]] | None:
    chunk_cls, tts_cls, _ = _inspect_streaming_trace_targets(model)
    if chunk_cls is None or tts_cls is None:
        return None

    original_chunk_generate = chunk_cls.chunk_generate
    original_generate_with_buffer = tts_cls.generate_with_buffer
    chunk_state: dict[str, Any] = {
        "generated_ids": [],
        "chunk_index": 0,
    }
    talker_state: dict[str, int] = {
        "chunk_index": 0,
        "condition_index": 0,
    }
    thinker_tensor_root = artifact_dir / "condition_dumps" / "hf_thinker" if artifact_dir is not None else None
    talker_tensor_root = artifact_dir / "condition_dumps" / "hf_talker" if artifact_dir is not None else None

    def traced_chunk_generate(this: Any, *args: Any, **kwargs: Any) -> Any:
        output = original_chunk_generate(this, *args, **kwargs)
        raw_chunk_ids = _coerce_token_chunk(getattr(output, "chunk_token_ids", None))
        is_first_generate_chunk = bool(kwargs.get("is_first_generate_chunk", False))
        finished = bool(getattr(output, "finished", False))

        yield_chunk_ids: list[int] = []
        if raw_chunk_ids:
            prev_last = chunk_state["generated_ids"][-1:] if chunk_state["generated_ids"] else []
            if is_first_generate_chunk:
                yield_chunk_ids = raw_chunk_ids if finished else raw_chunk_ids[:-1]
            elif finished:
                yield_chunk_ids = prev_last + raw_chunk_ids
            else:
                yield_chunk_ids = prev_last + raw_chunk_ids[:-1]
            chunk_state["generated_ids"].extend(raw_chunk_ids)

        hidden_states = getattr(output, "last_hidden_states", None)
        current_inputs_embeds = getattr(output, "current_inputs_embeds", None)
        dump_dir_rel: str | None = None
        tensor_summaries: dict[str, Any] | None = None
        if thinker_tensor_root is not None and yield_chunk_ids and hasattr(hidden_states, "shape"):
            import torch
            import torch.nn.functional as F

            hidden_states_tensor = torch.as_tensor(hidden_states)
            if hidden_states_tensor.ndim == 3 and int(hidden_states_tensor.shape[1]) == len(yield_chunk_ids):
                yield_chunk_token_ids = torch.tensor(
                    yield_chunk_ids,
                    dtype=torch.long,
                    device=hidden_states_tensor.device,
                ).reshape(1, -1)
                llm_embeds = model.tts.emb_text(yield_chunk_token_ids)
                hidden_embeds = model.tts.projector_semantic(hidden_states_tensor)
                if getattr(model.tts.config, "normalize_projected_hidden", False):
                    hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
                tts_embeds = llm_embeds + hidden_embeds

                dump_dir = thinker_tensor_root / f"chunk_{int(chunk_state['chunk_index']):04d}"
                tensor_summaries = {
                    "yield_chunk_token_ids": _write_tensor_dump(
                        dump_dir / "yield_chunk_token_ids.pt", yield_chunk_token_ids
                    ),
                    "last_hidden_states": _write_tensor_dump(dump_dir / "last_hidden_states.pt", hidden_states_tensor),
                    "llm_embeds": _write_tensor_dump(dump_dir / "llm_embeds.pt", llm_embeds),
                    "hidden_embeds": _write_tensor_dump(dump_dir / "hidden_embeds.pt", hidden_embeds),
                    "tts_embeds": _write_tensor_dump(dump_dir / "tts_embeds.pt", tts_embeds),
                }
                dump_dir_rel = _relative_to_root(dump_dir, artifact_dir)

        thinker_rows.append(
            {
                "chunk_index": int(chunk_state["chunk_index"]),
                "is_first_generate_chunk": is_first_generate_chunk,
                "finished": finished,
                "raw_chunk_token_ids": raw_chunk_ids,
                "raw_chunk_token_count": len(raw_chunk_ids),
                "yield_chunk_token_ids": yield_chunk_ids,
                "yield_chunk_token_count": len(yield_chunk_ids),
                "yield_chunk_text_preview": _preview_text(
                    tokenizer.decode(yield_chunk_ids) if tokenizer is not None and yield_chunk_ids else "",
                    limit=160,
                ),
                "last_hidden_states_shape": list(hidden_states.shape) if hasattr(hidden_states, "shape") else None,
                "current_inputs_embeds_shape": (
                    list(current_inputs_embeds.shape) if hasattr(current_inputs_embeds, "shape") else None
                ),
                "tensor_dump_dir": dump_dir_rel,
                "tensor_summaries": tensor_summaries,
            }
        )
        chunk_state["chunk_index"] += 1
        return output

    def traced_generate_with_buffer(this: Any, *args: Any, **kwargs: Any) -> Any:
        condition = kwargs.get("condition")
        if condition is None and args:
            condition = args[0]
        text_finished = kwargs.get("text_finished")
        if text_finished is None and len(args) >= 2:
            text_finished = args[1]

        condition_index = int(talker_state["condition_index"])
        talker_state["condition_index"] += 1
        condition_dump_dir_rel: str | None = None
        condition_summary: dict[str, Any] | None = None
        if talker_tensor_root is not None and hasattr(condition, "shape"):
            dump_dir = talker_tensor_root / f"condition_{condition_index:04d}"
            condition_summary = _write_tensor_dump(dump_dir / "condition.pt", condition)
            _write_json(
                dump_dir / "metadata.json",
                {
                    "condition_index": condition_index,
                    "condition_shape": list(condition.shape) if hasattr(condition, "shape") else None,
                    "text_finished": bool(text_finished),
                },
            )
            condition_dump_dir_rel = _relative_to_root(dump_dir, artifact_dir)
        generator = original_generate_with_buffer(this, *args, **kwargs)
        for audio_token_chunk, is_last_audio_chunk in generator:
            token_ids = _coerce_token_chunk(audio_token_chunk)
            talker_rows.append(
                {
                    "chunk_index": int(talker_state["chunk_index"]),
                    "condition_index": condition_index,
                    "text_finished": bool(text_finished),
                    "is_last_audio_chunk": bool(is_last_audio_chunk),
                    "condition_shape": list(condition.shape) if hasattr(condition, "shape") else None,
                    "audio_token_ids": token_ids,
                    "audio_token_count": len(token_ids),
                    "condition_dump_dir": condition_dump_dir_rel,
                    "condition_summary": condition_summary,
                }
            )
            talker_state["chunk_index"] += 1
            yield audio_token_chunk, is_last_audio_chunk

    chunk_cls.chunk_generate = traced_chunk_generate
    tts_cls.generate_with_buffer = traced_generate_with_buffer
    return (
        (chunk_cls, "chunk_generate", original_chunk_generate),
        (tts_cls, "generate_with_buffer", original_generate_with_buffer),
    )


def _restore_trace_hooks(patches: tuple[tuple[type[Any], str, Any], ...] | None) -> None:
    if patches is None:
        return
    for owner, attr_name, original in patches:
        setattr(owner, attr_name, original)


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


def _load_local_talker_codec_jsonl(codec_jsonl_path: str) -> dict[str, Any]:
    path = Path(codec_jsonl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Talker codec jsonl not found: {path}")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Talker codec jsonl is empty: {path}")

    rows = sorted(rows, key=lambda row: int(row.get("chunk_index", 0)))
    token_ids: list[int] = []
    for row in rows:
        audio_token_ids = row.get("audio_token_ids", [])
        if not isinstance(audio_token_ids, list):
            raise ValueError(f"Expected 'audio_token_ids' list in {path}, got {type(audio_token_ids)}")
        token_ids.extend(int(token_id) for token_id in audio_token_ids)

    return {
        "path": str(path),
        "rows": rows,
        "token_ids": token_ids,
    }


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


def _write_code2wav_prompt_wav(ref_audio: dict[str, Any] | None, target_sr: int) -> str:
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
    trimmed_token_ids = list(token_ids)
    while trimmed_token_ids and trimmed_token_ids[-1] == int(audio_eos_token_id):
        trimmed_token_ids.pop()

    if not trimmed_token_ids:
        return np.zeros((0,), dtype=np.float32), int(output_sample_rate)

    prompt_wav_path = _write_code2wav_prompt_wav(ref_audio, int(audio_prompt_sample_rate))
    try:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return {
        "remote_path": str(path.relative_to(REMOTE_OUTPUT_DIR)),
        "num_samples": int(waveform.shape[0]),
        "sample_rate": int(sample_rate),
        "duration_sec": float(waveform.shape[0] / max(int(sample_rate), 1)),
    }


def _build_system_message(
    ref_audio: np.ndarray | None,
    *,
    language: str,
    system_mode: str,
    system_profile: str | None,
) -> dict[str, Any]:
    if ref_audio is None:
        return {
            "role": "system",
            "content": ["You are a helpful assistant. You can accept audio and text input and output voice and text."],
        }

    if system_profile:
        if language == "zh":
            suffix = "你是一个具有以上声音风格的AI助手。请用高拟人度、口语化的方式和用户聊天。" + system_profile
        else:
            suffix = "Please chat with the user in a highly human-like and oral style." + system_profile
        return {
            "role": "system",
            "content": [
                "模仿音频样本的音色并生成新的内容。"
                if language == "zh"
                else "Clone the voice in the provided audio prompt.",
                ref_audio,
                suffix,
            ],
        }

    if system_mode == "omni":
        return {
            "role": "system",
            "content": [
                "模仿音频样本的音色并生成新的内容。"
                if language == "zh"
                else "Clone the voice in the provided audio prompt.",
                ref_audio,
                "请用这种声音风格来为用户提供帮助。"
                if language == "zh"
                else "As an assistant, you will speak using this voice style.",
            ],
        }

    if system_mode == "audio_assistant":
        return {
            "role": "system",
            "content": [
                "模仿音频样本的音色并生成新的内容。"
                if language == "zh"
                else "Clone the voice in the provided audio prompt.",
                ref_audio,
                (
                    "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。"
                    if language == "zh"
                    else (
                        "Please assist users while maintaining this voice style. "
                        "Please answer the user's questions seriously and in a "
                        "high quality. Please chat with the user in a highly "
                        "human-like and oral style."
                    )
                ),
            ],
        }

    if system_mode == "voice_cloning":
        return {
            "role": "system",
            "content": [
                "模仿输入音频中的声音特征。" if language == "zh" else "Clone the voice in the provided audio prompt.",
                ref_audio,
            ],
        }

    raise ValueError(f"Unsupported system_mode: {system_mode}")


def _patch_minicpm_audio_io() -> None:
    import librosa
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
        pass

    if not patch_targets and not torchaudio_targets:
        return

    def _soundfile_load_audio(file: str | None, sr: int = 16000) -> torch.Tensor:
        if file is None:
            return torch.zeros((0,), dtype=torch.float32)

        audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
        audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=-1)
        audio_np = audio_np.reshape(-1)
        if int(sample_rate) != int(sr):
            audio_np = librosa.resample(y=audio_np, orig_sr=int(sample_rate), target_sr=int(sr))
        return torch.from_numpy(np.asarray(audio_np, dtype=np.float32))

    def _torchaudio_load(file: str | None, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, int]:
        if file is None:
            sample_rate = 16000
            audio_np = np.zeros((sample_rate,), dtype=np.float32)
        else:
            audio, sample_rate = sf.read(file, dtype="float32", always_2d=False)
            audio_np = np.asarray(audio, dtype=np.float32)

        if audio_np.ndim == 1:
            audio_np = audio_np[None, :]
        elif audio_np.ndim > 1:
            audio_np = np.asarray(audio_np, dtype=np.float32).T

        return torch.from_numpy(np.ascontiguousarray(audio_np, dtype=np.float32)), int(sample_rate)

    def _torchaudio_save(
        file: str | io.BytesIO,
        src: torch.Tensor,
        sample_rate: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        save_format = kwargs.pop("format", None)
        kwargs.pop("backend", None)

        audio_np = src.detach().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = np.asarray(audio_np, dtype=np.float32).T
        elif audio_np.ndim == 1:
            audio_np = np.asarray(audio_np, dtype=np.float32)
        else:
            raise ValueError(f"Expected 1-D or 2-D audio tensor, got shape {tuple(audio_np.shape)}")

        sf.write(file, audio_np, int(sample_rate), format=save_format)

    patched_target_ids: set[int] = set()
    for target in patch_targets:
        if id(target) in patched_target_ids:
            continue
        patched_target_ids.add(id(target))
        setattr(target, "load_audio", _soundfile_load_audio)

    patched_torchaudio_ids: set[int] = set()
    for target in torchaudio_targets:
        if id(target) in patched_torchaudio_ids:
            continue
        patched_torchaudio_ids.add(id(target))
        setattr(target, "load", _torchaudio_load)
        setattr(target, "save", _torchaudio_save)

    patched_direct_audio_io_ids: set[int] = set()
    for target in direct_audio_io_targets:
        if id(target) in patched_direct_audio_io_ids:
            continue
        patched_direct_audio_io_ids.add(id(target))
        setattr(target, "load", _torchaudio_load)
        setattr(target, "save", _torchaudio_save)


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
    timeout=45 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        str(REMOTE_OUTPUT_DIR): output_volume,
    },
)
def run_hf_smoke(
    prompt: str,
    ref_audio_payload: dict[str, Any] | None = None,
    ref_audio_label: str | None = None,
    output_name: str | None = None,
    artifact_dir_name: str | None = None,
    system_mode: str = "omni",
    system_profile: str | None = None,
    language: str = "en",
    do_sample: bool = True,
    temperature: float = 0.7,
    seed: int = 42,
    max_new_tokens: int = 4096,
    init_audio: bool = True,
    streaming: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoTokenizer

    snapshot_path = _download_model()
    hf_cache_volume.commit()
    _patch_minicpm_audio_io()

    model = AutoModel.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=init_audio,
        init_tts=True,
    )
    model.eval().cuda()
    model.init_tts()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    debug_trace = [] if debug else None

    ref_audio_np: np.ndarray | None = None
    if ref_audio_payload is not None:
        ref_audio_np = np.asarray(ref_audio_payload["wav"], dtype=np.float32).reshape(-1)
        ref_sr = int(ref_audio_payload["sr"])
        if ref_sr != 16000:
            import librosa

            ref_audio_np = librosa.resample(ref_audio_np, orig_sr=ref_sr, target_sr=16000)
    else:
        ref_sr = None

    if debug:
        _record_debug(
            debug_trace,
            stage="ref_audio",
            payload={
                "label": ref_audio_label,
                "summary": _summarize_audio_array(
                    ref_audio_np, sample_rate=ref_sr if ref_audio_np is not None else None
                ),
            },
        )

    if hasattr(model, "get_sys_prompt"):
        sys_msg = model.get_sys_prompt(
            ref_audio=ref_audio_np,
            mode=system_mode,
            language=language,
        )
        if system_profile:
            content = sys_msg.get("content")
            if isinstance(content, list) and content and isinstance(content[-1], str):
                content[-1] = f"{content[-1]} {system_profile}".strip()
    else:
        sys_msg = _build_system_message(
            ref_audio_np,
            language=language,
            system_mode=system_mode,
            system_profile=system_profile,
        )
    user_msg = {"role": "user", "content": [prompt]}
    msgs = [sys_msg, user_msg]

    if debug:
        _record_debug(
            debug_trace,
            stage="messages",
            payload={
                "system_mode": system_mode,
                "language": language,
                "do_sample": bool(do_sample),
                "temperature": temperature,
                "seed": int(seed),
                "max_new_tokens": max_new_tokens,
                "messages": [_summarize_message(message) for message in msgs],
            },
        )

    remote_output_path = REMOTE_OUTPUT_DIR / (output_name or f"{uuid.uuid4().hex}.wav")
    remote_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = REMOTE_OUTPUT_DIR / artifact_dir_name if artifact_dir_name else None
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            artifact_dir / "prompt.json",
            {
                "prompt": prompt,
                "ref_audio_label": ref_audio_label,
                "system_mode": system_mode,
                "system_profile": system_profile,
                "language": language,
                "do_sample": bool(do_sample),
                "temperature": temperature,
                "seed": int(seed),
                "max_new_tokens": max_new_tokens,
                "streaming": streaming,
                "messages": [_summarize_message(message) for message in msgs],
            },
        )
        prompt_token_ids = _best_effort_prompt_token_ids(snapshot_path, msgs)
        if prompt_token_ids is not None:
            _write_json(artifact_dir / "prompt_token_ids.json", prompt_token_ids)

    try:
        if streaming:
            thinker_rows: list[dict[str, Any]] = []
            talker_rows: list[dict[str, Any]] = []
            tokenizer = None
            patches = None

            if hasattr(model, "reset_session"):
                model.reset_session()
            if hasattr(model, "init_token2wav_cache"):
                token2wav_cache_mode = "direct"
                try:
                    model.init_token2wav_cache(ref_audio_np)
                except Exception:
                    if ref_audio_np is not None:
                        raise
                    silence_ref_audio = np.zeros((16000,), dtype=np.float32)
                    model.init_token2wav_cache(silence_ref_audio)
                    token2wav_cache_mode = "silence_fallback"

            session_id = uuid.uuid4().hex
            if debug:
                _record_debug(
                    debug_trace,
                    stage="streaming_runtime",
                    payload={
                        "session_id": session_id,
                        "has_streaming_prefill": hasattr(model, "streaming_prefill"),
                        "has_streaming_generate": hasattr(model, "streaming_generate"),
                        "token2wav_cache_mode": token2wav_cache_mode
                        if hasattr(model, "init_token2wav_cache")
                        else None,
                    },
                )

            if not hasattr(model, "streaming_prefill") or not hasattr(model, "streaming_generate"):
                raise RuntimeError("MiniCPM HF model does not expose streaming_prefill/streaming_generate.")

            _, _, trace_target_info = _inspect_streaming_trace_targets(model)
            if debug:
                _record_debug(
                    debug_trace,
                    stage="streaming_trace_targets",
                    payload=trace_target_info,
                )

            if artifact_dir is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True)
                except Exception as e:
                    tokenizer = None
                    if debug:
                        _record_debug(
                            debug_trace,
                            stage="streaming_trace_tokenizer_load_failed",
                            payload={
                                "error_type": type(e).__name__,
                                "error": str(e),
                            },
                        )
                patches = _install_streaming_trace_hooks(
                    model,
                    artifact_dir=artifact_dir,
                    tokenizer=tokenizer,
                    thinker_rows=thinker_rows,
                    talker_rows=talker_rows,
                )
                if debug:
                    _record_debug(
                        debug_trace,
                        stage="streaming_trace_install",
                        payload={
                            "installed": patches is not None,
                        },
                    )

            audio_chunks: list[np.ndarray] = []
            audio_rows: list[dict[str, Any]] = []
            text_rows: list[dict[str, Any]] = []
            text_output = ""
            cumulative_samples = 0

            try:
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

                iter_gen = _call_with_supported_kwargs(
                    model.streaming_generate,
                    session_id=session_id,
                    generate_audio=True,
                    use_tts_template=True,
                    enable_thinking=False,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )

                for chunk_index, chunk in enumerate(iter_gen):
                    if not isinstance(chunk, tuple) or len(chunk) != 2:
                        raise RuntimeError(
                            "MiniCPM HF streaming_generate returned an unexpected chunk shape: "
                            f"{type(chunk).__name__} value={chunk!r}"
                        )
                    wav_chunk_raw, text_chunk_raw = chunk
                    audio_chunk = _coerce_audio_chunk(wav_chunk_raw)
                    text_chunk = str(text_chunk_raw or "")
                    audio_chunks.append(audio_chunk)
                    text_output += text_chunk
                    cumulative_samples += int(audio_chunk.shape[0])

                    audio_row = {
                        "chunk_index": chunk_index,
                        "num_samples": int(audio_chunk.shape[0]),
                        "duration_sec": float(audio_chunk.shape[0] / 24000.0),
                        "cumulative_num_samples": cumulative_samples,
                        "cumulative_duration_sec": float(cumulative_samples / 24000.0),
                        "text_chunk_length": len(text_chunk),
                        "text_chunk_preview": _preview_text(text_chunk, limit=120),
                    }
                    text_row = {
                        "chunk_index": chunk_index,
                        "text_chunk": text_chunk,
                        "text_chunk_length": len(text_chunk),
                        "cumulative_text_length": len(text_output),
                        "cumulative_text_preview": _preview_text(text_output, limit=240),
                    }
                    audio_rows.append(audio_row)
                    text_rows.append(text_row)

                    if artifact_dir is not None:
                        chunk_path = artifact_dir / "audio_chunks" / f"{chunk_index:04d}.wav"
                        chunk_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(chunk_path), audio_chunk, 24000, format="WAV", subtype="PCM_16")
            finally:
                _restore_trace_hooks(patches)

            if not audio_chunks:
                raise RuntimeError("MiniCPM HF streaming_generate returned no audio chunks.")

            final_audio_np = np.concatenate(audio_chunks, axis=0).astype(np.float32, copy=False)
            sample_rate = 24000
            sf.write(str(remote_output_path), final_audio_np, sample_rate, format="WAV", subtype="PCM_16")

            if artifact_dir is not None:
                _write_jsonl(artifact_dir / "audio_chunks.jsonl", audio_rows)
                _write_jsonl(artifact_dir / "text_chunks.jsonl", text_rows)
                _write_jsonl(artifact_dir / "hf_thinker_chunks.jsonl", thinker_rows)
                _write_jsonl(artifact_dir / "hf_talker_chunks.jsonl", talker_rows)
                (artifact_dir / "final.txt").write_text(text_output, encoding="utf-8")
                first_10s = final_audio_np[: sample_rate * 10]
                if first_10s.size > 0:
                    sf.write(
                        str(artifact_dir / "first_10s.wav"),
                        first_10s,
                        sample_rate,
                        format="WAV",
                        subtype="PCM_16",
                    )
                if debug:
                    _record_debug(
                        debug_trace,
                        stage="streaming_trace_rows",
                        payload={
                            "thinker_row_count": len(thinker_rows),
                            "talker_row_count": len(talker_rows),
                        },
                    )
        else:
            text_output = model.chat(
                msgs=msgs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                use_tts_template=True,
                generate_audio=True,
                temperature=temperature,
                output_audio_path=str(remote_output_path),
                enable_thinking=False,
            )
    except Exception as e:
        if debug:
            _record_debug(
                debug_trace,
                stage="chat_exception",
                payload={
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
        raise

    if not remote_output_path.exists():
        if debug:
            _record_debug(
                debug_trace,
                stage="missing_audio_output",
                payload={
                    "text_output_preview": _preview_text(text_output),
                    "text_output_length": len(text_output) if isinstance(text_output, str) else None,
                    "output_path": str(remote_output_path),
                },
            )
        raise RuntimeError(
            "MiniCPM HF chat returned without writing audio output. "
            f"text_output={text_output!r}, max_new_tokens={max_new_tokens}"
        )

    audio_np, sample_rate = sf.read(str(remote_output_path), dtype="float32", always_2d=False)
    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if artifact_dir is not None and streaming and not (artifact_dir / "final.wav").exists():
        sf.write(str(artifact_dir / "final.wav"), audio_np, int(sample_rate), format="WAV", subtype="PCM_16")
    output_volume.commit()

    if debug:
        _record_debug(
            debug_trace,
            stage="final_output",
            payload={
                "text_output_preview": _preview_text(text_output),
                "text_output_length": len(text_output) if isinstance(text_output, str) else None,
                "audio": _summarize_audio_array(audio_np, sample_rate=int(sample_rate)),
                "output_path": str(remote_output_path),
            },
        )

    result = {
        "snapshot_path": snapshot_path,
        "text_output": text_output,
        "sample_rate": int(sample_rate),
        "num_samples": int(audio_np.shape[0]),
        "duration_sec": float(audio_np.shape[0] / max(int(sample_rate), 1)),
        "ref_audio_label": ref_audio_label,
        "system_mode": system_mode,
        "system_profile": system_profile,
        "language": language,
        "output_name": remote_output_path.name,
        "artifact_dir_name": artifact_dir_name,
        "artifact_paths": _list_relative_files(artifact_dir) if artifact_dir is not None else [],
        "streaming": streaming,
    }
    if debug_trace is not None:
        result["debug_trace"] = debug_trace
    return result


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    timeout=30 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        str(REMOTE_OUTPUT_DIR): output_volume,
    },
)
def replay_hf_code2wav_from_token_ids(
    token_ids: list[int],
    ref_audio_payload: dict[str, Any] | None = None,
    output_name: str | None = None,
    audio_eos_token_id: int = 6561,
    audio_prompt_sample_rate: int = 16000,
    output_sample_rate: int = 24000,
    n_timesteps: int = 10,
) -> dict[str, Any]:
    from stepaudio2 import Token2wav

    snapshot_path = _download_model()
    hf_cache_volume.commit()
    _patch_minicpm_audio_io()

    token2wav_assets_dir = Path(snapshot_path) / "assets" / "token2wav"
    token2wav = Token2wav(
        str(token2wav_assets_dir),
        float16=False,
        n_timesteps=int(n_timesteps),
    )

    remote_output_name = output_name or f"{uuid.uuid4().hex}.wav"
    waveform, sample_rate = _decode_code2wav_one(
        token2wav,
        [int(token_id) for token_id in token_ids],
        ref_audio=_canonicalize_ref_audio_payload(ref_audio_payload),
        audio_eos_token_id=int(audio_eos_token_id),
        audio_prompt_sample_rate=int(audio_prompt_sample_rate),
        output_sample_rate=int(output_sample_rate),
    )
    output_path = REMOTE_OUTPUT_DIR / remote_output_name
    output_info = _save_code2wav_output(output_path, waveform, sample_rate)
    output_volume.commit()
    return {
        "snapshot_path": snapshot_path,
        "assets_dir": str(token2wav_assets_dir),
        "token_count": len(token_ids),
        "audio_eos_token_id": int(audio_eos_token_id),
        "audio_prompt_sample_rate": int(audio_prompt_sample_rate),
        "output_sample_rate": int(output_sample_rate),
        "n_timesteps": int(n_timesteps),
        "used_ref_audio": ref_audio_payload is not None,
        "output": output_info,
    }


@app.function(
    image=image,
    gpu=GPU_REQUEST,
    timeout=20 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
    },
)
def replay_hf_talker_decode_step(
    inputs_embeds_blob: bytes,
    position_ids_blob: bytes | None = None,
    temperature: float = 0.9,
    topk: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel

    snapshot_path = _download_model()
    hf_cache_volume.commit()
    _patch_minicpm_audio_io()

    model = AutoModel.from_pretrained(
        snapshot_path,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=False,
        init_tts=True,
    )
    model.eval().cuda()
    model.init_tts()

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    inputs_embeds = torch.load(io.BytesIO(inputs_embeds_blob), map_location="cpu")
    inputs_embeds = torch.as_tensor(inputs_embeds)
    if inputs_embeds.ndim == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0)
    if inputs_embeds.ndim != 3:
        raise ValueError(f"Expected inputs_embeds to have 2 or 3 dims, got {tuple(inputs_embeds.shape)}")
    inputs_embeds = inputs_embeds.cuda().to(dtype=torch.bfloat16)

    if position_ids_blob is not None:
        position_ids = torch.load(io.BytesIO(position_ids_blob), map_location="cpu")
        position_ids = torch.as_tensor(position_ids, dtype=torch.long)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
    else:
        position_ids = torch.arange(inputs_embeds.shape[1], dtype=torch.long).unsqueeze(0)
    if position_ids.ndim != 2:
        raise ValueError(f"Expected position_ids to have 1 or 2 dims, got {tuple(position_ids.shape)}")
    position_ids = position_ids.cuda()

    outputs = model.tts.model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state

    logits = torch.empty(
        hidden_states.size(0),
        hidden_states.size(1),
        model.tts.num_audio_tokens,
        model.tts.num_vq,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for num_vq_iter in range(model.tts.num_vq):
        logits[..., num_vq_iter] = model.tts.head_code[num_vq_iter](hidden_states)

    logits = logits[:, -1].float()
    logits = logits.permute(0, 2, 1)
    logits = logits.reshape(-1, logits.size(2))

    safe_temperature = max(float(temperature), 1e-5)
    sampling_logits = logits / safe_temperature
    raw_probs = F.softmax(logits, dim=-1)
    sampling_probs = F.softmax(sampling_logits, dim=-1)
    greedy_token = torch.argmax(sampling_logits, dim=-1, keepdim=True)

    return {
        "snapshot_path": snapshot_path,
        "seed": int(seed),
        "temperature": float(temperature),
        "inputs_embeds_summary": _summarize_tensor(inputs_embeds),
        "position_ids_summary": _summarize_tensor(position_ids),
        "hidden_states_summary": _summarize_tensor(hidden_states),
        "raw_logits_summary": _summarize_tensor(logits),
        "sampling_logits_summary": _summarize_tensor(sampling_logits),
        "raw_top_tokens": _topk_logits_summary(logits, raw_probs, topk),
        "sampling_top_tokens": _topk_logits_summary(sampling_logits, sampling_probs, topk),
        "greedy_token_id": int(greedy_token.reshape(-1)[0].item()),
        "position_ids": [int(x) for x in position_ids.detach().cpu().reshape(-1).tolist()],
    }


@app.local_entrypoint()
def main(
    prompt: str = (
        "Please answer this question naturally in the reference voice: "
        "How do you view today's NBA compared to your era?"
    ),
    output: str = "minicpmo45_hf_smoke.wav",
    artifact_output_dir: str = "",
    ref_audio_path: str = "",
    preload_only: bool = False,
    system_mode: str = "omni",
    system_profile: str = "",
    language: str = "en",
    do_sample: bool = True,
    temperature: float = 0.7,
    seed: int = 42,
    max_new_tokens: int = 4096,
    init_audio: bool = True,
    streaming: bool = False,
    debug: bool = False,
    replay_inputs_embeds_path: str = "",
    replay_position_ids_path: str = "",
    replay_output_json: str = "",
    replay_codec_jsonl_path: str = "",
    replay_codec_audio_eos_token_id: int = 6561,
    replay_codec_audio_prompt_sample_rate: int = 16000,
    replay_codec_output_sample_rate: int = 24000,
    replay_codec_n_timesteps: int = 10,
) -> None:
    snapshot_path = preload_model.remote()
    print(f"Model cached at: {snapshot_path}")

    if preload_only:
        return

    if replay_inputs_embeds_path:
        replay_result = replay_hf_talker_decode_step.remote(
            inputs_embeds_blob=_load_local_blob(replay_inputs_embeds_path),
            position_ids_blob=_load_local_blob(replay_position_ids_path) if replay_position_ids_path else None,
            temperature=temperature,
            seed=seed,
        )
        replay_json = json.dumps(replay_result, indent=2, ensure_ascii=False, sort_keys=True)
        if replay_output_json:
            output_json_path = Path(replay_output_json)
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            output_json_path.write_text(replay_json, encoding="utf-8")
            print(f"Saved replay JSON to: {output_json_path}")
        print(replay_json)
        return

    if replay_codec_jsonl_path:
        codec_payload = _load_local_talker_codec_jsonl(replay_codec_jsonl_path)
        remote_output_name = f"{uuid.uuid4().hex}_{Path(output).name}"
        result = replay_hf_code2wav_from_token_ids.remote(
            token_ids=codec_payload["token_ids"],
            ref_audio_payload=_load_local_ref_audio(ref_audio_path) if ref_audio_path else None,
            output_name=remote_output_name,
            audio_eos_token_id=int(replay_codec_audio_eos_token_id),
            audio_prompt_sample_rate=int(replay_codec_audio_prompt_sample_rate),
            output_sample_rate=int(replay_codec_output_sample_rate),
            n_timesteps=int(replay_codec_n_timesteps),
        )
        wav_bytes = b"".join(output_volume.read_file(result["output"]["remote_path"]))
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        result["codec_jsonl_path"] = codec_payload["path"]
        result["codec_chunk_count"] = len(codec_payload["rows"])
        result["local_output_path"] = str(output_path.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"Saved code2wav replay audio to: {output_path}")
        return

    ref_audio_payload: dict[str, Any] | None = None
    ref_audio_label: str | None = None
    if ref_audio_path:
        ref_audio_payload = _load_local_ref_audio(ref_audio_path)
        ref_audio_label = str(Path(ref_audio_path).expanduser().resolve())

    output_name = f"{uuid.uuid4().hex}_{Path(output).name}"
    artifact_dir_name = f"{uuid.uuid4().hex}_artifacts" if artifact_output_dir else None
    result = run_hf_smoke.remote(
        prompt=prompt,
        ref_audio_payload=ref_audio_payload,
        ref_audio_label=ref_audio_label,
        output_name=output_name,
        artifact_dir_name=artifact_dir_name,
        system_mode=system_mode,
        system_profile=system_profile or None,
        language=language,
        do_sample=do_sample,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
        init_audio=init_audio,
        streaming=streaming,
        debug=debug,
    )

    remote_output_name = result.pop("output_name")
    remote_artifact_dir_name = result.pop("artifact_dir_name", None)
    artifact_paths = result.pop("artifact_paths", [])
    debug_trace = result.pop("debug_trace", None)
    wav_bytes = b"".join(output_volume.read_file(remote_output_name))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(wav_bytes)

    if artifact_output_dir and remote_artifact_dir_name:
        local_artifact_root = Path(artifact_output_dir)
        local_artifact_root.mkdir(parents=True, exist_ok=True)
        for remote_path in artifact_paths:
            rel_path = Path(remote_path).relative_to(remote_artifact_dir_name)
            local_path = local_artifact_root / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"".join(output_volume.read_file(remote_path)))
        print(f"Saved HF streaming artifacts to: {local_artifact_root}")

    print(json.dumps(result, indent=2, sort_keys=True))
    if debug_trace is not None:
        print(json.dumps({"debug_trace": debug_trace}, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"Saved generated audio to: {output_path}")
