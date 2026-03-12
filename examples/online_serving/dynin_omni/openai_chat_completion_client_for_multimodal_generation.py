#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import mimetypes
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "snu-aidas/Dynin-Omni"
DEFAULT_OUTPUT_DIR = "/tmp/dynin_online_outputs"

QUERY_CHOICES = ("t2t", "t2i", "t2s", "i2i")
DEFAULT_PROMPT_BY_QUERY = {
    "t2t": "Explain multimodal LLM inference in 3 sentences.",
    "t2i": "A high quality detailed living room interior photo.",
    "t2s": "Please read this sentence naturally: Hello from Dynin-Omni online serving.",
    "i2i": "Transform this image into a realistic indoor living room while preserving layout.",
}
DEFAULT_MODALITIES_BY_QUERY = {
    "t2t": ["text"],
    "t2i": ["image"],
    "t2s": ["audio"],
    "i2i": ["image"],
}
OFFLINE_PARITY_STAGE_COUNT = 3
OFFLINE_PARITY_STAGE_SAMPLING = {
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "detokenize": False,
}
END2END_PARITY_TASKS = {"t2i", "t2s", "i2i"}


def _infer_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def _encode_file_as_data_url(path: Path) -> str:
    mime_type = _infer_mime_type(path)
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _to_image_url(path_or_url: str) -> str:
    value = str(path_or_url)
    if value.startswith(("http://", "https://", "data:image/")):
        return value
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    return _encode_file_as_data_url(path)


def _build_user_content(query_type: str, prompt: str, image_path: str | None) -> list[dict[str, Any]]:
    if query_type == "t2t":
        return [{"type": "text", "text": prompt}]

    if query_type == "t2i":
        return [{"type": "text", "text": f"<|t2i|> {prompt}"}]

    if query_type == "t2s":
        return [{"type": "text", "text": f"<|t2s|> {prompt}"}]

    if query_type == "i2i":
        if not image_path:
            raise ValueError("--image-path is required for query type i2i")
        return [
            {"type": "text", "text": f"<|i2i|> {prompt}"},
            {"type": "image_url", "image_url": {"url": _to_image_url(image_path)}},
        ]

    raise ValueError(f"Unsupported query_type: {query_type}")


@lru_cache(maxsize=1)
def _load_offline_end2end_module() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "offline_inference"
        / "dynin_omni"
        / "end2end.py"
    )
    spec = importlib.util.spec_from_file_location("dynin_end2end_offline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load end2end module spec from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_jsonable(value: Any) -> Any:
    # Convert tensors/ndarrays in prompting payloads to JSON-serializable forms.
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_jsonable(tolist())
        except Exception:
            pass
    return value


def _build_end2end_style_inputs(
    *,
    query_type: str,
    prompt: str,
    image_path: str | None,
    model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if query_type not in END2END_PARITY_TASKS:
        return _build_user_content(query_type=query_type, prompt=prompt, image_path=image_path), {}

    offline = _load_offline_end2end_module()

    task_name = query_type
    default_runtime_task, default_prompting_task, default_detok_id, _ = offline.TASK_DEFAULT_RUNTIME[task_name]
    runtime_task = str(offline._runtime_fallback(task_name, "runtime_task", None) or default_runtime_task)
    prompting_task = str(offline._runtime_fallback(task_name, "prompting_task", None) or default_prompting_task)
    detok_id_default = offline._runtime_fallback(task_name, "detok_id", None)
    if detok_id_default is None:
        detok_id_default = default_detok_id
    detok_id = int(detok_id_default)

    image_resolution = int(offline._runtime_fallback(task_name, "image_resolution", None) or 336)
    prompt_max_text_len = int(offline._runtime_fallback(task_name, "prompt_max_text_len", None) or 1024)
    max_new_tokens = int(offline._runtime_fallback(task_name, "max_new_tokens", None) or 256)
    steps = int(offline._runtime_fallback(task_name, "steps", None) or 256)
    block_length = int(offline._runtime_fallback(task_name, "block_length", None) or 2)
    temperature = float(offline._runtime_fallback(task_name, "temperature", None) or 0.0)
    cfg_scale = float(offline._runtime_fallback(task_name, "cfg_scale", None) or 0.0)
    remasking = str(offline._runtime_fallback(task_name, "remasking", None) or "low_confidence")
    timesteps = int(offline._runtime_fallback(task_name, "timesteps", None) or 20)
    guidance_scale = float(offline._runtime_fallback(task_name, "guidance_scale", None) or 0.0)
    mask_token_id = int(offline._runtime_fallback(task_name, "mask_token_id", None) or 126336)
    codebook_size = int(offline._runtime_fallback(task_name, "codebook_size", None) or 8192)
    audio_codebook_size = int(offline._runtime_fallback(task_name, "audio_codebook_size", None) or 4096)

    image_token_count_value = offline._runtime_fallback(task_name, "image_token_count", None)
    image_token_count = int(image_token_count_value) if image_token_count_value is not None else 0
    t2s_token_length = int(offline._runtime_fallback(task_name, "t2s_token_length", None) or 383)
    t2s_condition = str(
        offline._runtime_fallback(task_name, "t2s_condition", None)
        or "gender-female_emotion-neutral_speed-normal_pitch-normal"
    )
    use_train_i2i_prompt = offline._runtime_fallback(task_name, "use_train_i2i_prompt", None)
    if use_train_i2i_prompt is None:
        use_train_i2i_prompt = bool(task_name == "i2i")
    use_train_i2i_prompt = bool(use_train_i2i_prompt)

    text = prompt.strip()
    if task_name == "t2s":
        if not text:
            text = "Hello. This is a default text-to-speech sample."
        instruction = str(offline._runtime_fallback(task_name, "instruction", None) or "").strip()
        if not instruction:
            instruction = str(offline.DEFAULT_T2S_INSTRUCTION)
        text = offline.build_chat_prompt(f"{instruction}\n{text}")
    elif task_name in {"t2i", "i2i"} and not text:
        text = "A high quality detailed image."

    model_source = str(model).strip()
    tokenizer_source = model_source
    model_local_only = offline.resolve_local_only(None, model_source, default=Path(model_source).expanduser().is_dir())
    tokenizer_local_only = offline.resolve_local_only(
        None,
        tokenizer_source,
        default=bool(model_local_only),
    )
    tokenizer = offline.load_text_tokenizer(tokenizer_source, local_files_only=bool(tokenizer_local_only))
    text_vocab_size = int(len(tokenizer))

    image_tokens = None
    vq_image_source = "snu-aidas/magvitv2"
    vq_audio_source = "snu-aidas/emova_speech_tokenizer_vllm"
    vq_image_local_only = offline.resolve_local_only(None, vq_image_source, default=False)
    vq_audio_local_only = offline.resolve_local_only(None, vq_audio_source, default=False)

    if task_name == "i2i":
        if not image_path:
            raise ValueError("--image-path is required for query type i2i")
        torch_mod = offline.torch
        device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        vq_image = offline.load_vq_image_encoder(vq_image_source, bool(vq_image_local_only), device)
        image_tokens = offline.encode_image_tokens(
            Path(image_path).expanduser().resolve(),
            vq_model=vq_image,
            device=device,
            resolution=int(image_resolution),
        )
        if hasattr(vq_image, "cpu"):
            vq_image = vq_image.cpu()

    if image_token_count <= 0:
        if image_tokens is not None:
            image_token_count = int(image_tokens.numel())
        else:
            image_token_count = max(1, (int(image_resolution) // 16) ** 2)

    max_audio_len_for_prompt = int(max(t2s_token_length, 512))
    max_audio_len_short_for_prompt = max(256, max_audio_len_for_prompt // 2)
    cond_dropout_prob = 0.0
    uni_prompting = offline.load_universal_prompting(
        tokenizer=tokenizer,
        tokenizer_source=tokenizer_source,
        max_text_len=int(prompt_max_text_len),
        cond_dropout_prob=float(cond_dropout_prob),
        local_files_only=bool(tokenizer_local_only),
        max_audio_len=int(max_audio_len_for_prompt),
        max_audio_len_short=int(max_audio_len_short_for_prompt),
    )

    prompt_payload, prompting_task = offline.make_prompt_payload(
        task=task_name,
        text=text,
        image_tokens=image_tokens,
        audio_tokens=None,
        video_tokens=None,
        image_placeholder_tokens=int(image_token_count),
        audio_placeholder_tokens=int(t2s_token_length),
        image_token_offset=int(text_vocab_size),
        speech_token_offset=int(text_vocab_size) + int(codebook_size),
        mask_token_id=int(mask_token_id),
        use_train_i2i_prompt=bool(use_train_i2i_prompt),
    )
    _prompt_token_ids, prompt_attention_mask = offline._run_uni_prompting(uni_prompting, prompt_payload, prompting_task)
    if not prompt_attention_mask:
        prompt_attention_mask = [1] * len(_prompt_token_ids)

    uncond_payload = None
    if task_name in {"t2i", "i2i"} and guidance_scale > 0:
        uncond_payload, uncond_prompting_task = offline.make_prompt_payload(
            task=task_name,
            text="",
            image_tokens=image_tokens,
            audio_tokens=None,
            video_tokens=None,
            image_placeholder_tokens=int(image_token_count),
            audio_placeholder_tokens=int(t2s_token_length),
            image_token_offset=int(text_vocab_size),
            speech_token_offset=int(text_vocab_size) + int(codebook_size),
            mask_token_id=int(mask_token_id),
            use_train_i2i_prompt=bool(use_train_i2i_prompt),
        )

    dynin_config_default = (
        Path(__file__).resolve().parents[3]
        / "vllm_omni"
        / "model_executor"
        / "models"
        / "dynin_omni"
        / "configs"
        / "dynin_omni.yaml"
    )
    dynin_config_path = os.environ.get("DYNIN_CONFIG_PATH", str(dynin_config_default))

    runtime_info: dict[str, Any] = {
        "task": [runtime_task],
        "prompting_task": [prompting_task],
        "prompting_input": [_to_jsonable(prompt_payload)],
        "detok_id": [int(detok_id)],
        "dynin_config_path": [str(dynin_config_path)],
        "attention_mask": [prompt_attention_mask],
        "prompt_max_text_len": [int(prompt_max_text_len)],
        "prompting_max_text_len": [int(prompt_max_text_len)],
        "cond_dropout_prob": [float(cond_dropout_prob)],
        "prompting_cond_dropout_prob": [float(cond_dropout_prob)],
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
        "noise_type": ["mask"],
        "noise_schedule_name": ["cosine"],
        "noise_schedule_params": [{}],
        "seq_len": [int(image_token_count)],
        "condition": [str(t2s_condition)],
        "vq_model_image_path": [str(vq_image_source)],
        "vq_model_image_local_files_only": [bool(vq_image_local_only)],
        "vq_model_audio_path": [str(vq_audio_source)],
        "vq_model_audio_local_files_only": [bool(vq_audio_local_only)],
    }
    if uncond_payload is not None:
        runtime_info["uncond_prompting_input"] = [_to_jsonable(uncond_payload)]
    if task_name == "t2s":
        runtime_info["max_new_tokens"] = [int(t2s_token_length)]

    user_content = _build_user_content(
        query_type=query_type,
        prompt=text,
        image_path=image_path,
    )
    return user_content, runtime_info


def _collect_text_from_content(content: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            texts.append(stripped)
        return texts

    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            text_value = content.get(key)
            if isinstance(text_value, str) and text_value.strip():
                texts.append(text_value.strip())
        return texts

    if isinstance(content, list):
        for item in content:
            texts.extend(_collect_text_from_content(item))
        return texts

    content_text = getattr(content, "text", None)
    if isinstance(content_text, str) and content_text.strip():
        texts.append(content_text.strip())
    content_value = getattr(content, "content", None)
    if isinstance(content_value, str) and content_value.strip():
        texts.append(content_value.strip())
    output_text = getattr(content, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        texts.append(output_text.strip())
    return texts


def _extract_text_outputs(chat_completion: Any) -> list[str]:
    texts: list[str] = []
    for choice in getattr(chat_completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        content = getattr(message, "content", None)
        texts.extend(_collect_text_from_content(content))
        reasoning_content = getattr(message, "reasoning_content", None)
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            texts.append(reasoning_content.strip())
        choice_text = getattr(choice, "text", None)
        if isinstance(choice_text, str) and choice_text.strip():
            texts.append(choice_text.strip())
    top_level_output_text = getattr(chat_completion, "output_text", None)
    if isinstance(top_level_output_text, str) and top_level_output_text.strip():
        texts.append(top_level_output_text.strip())
    return texts


def _extract_image_data_urls(chat_completion: Any) -> list[str]:
    urls: list[str] = []
    for choice in getattr(chat_completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image_url":
                continue
            image_url = (item.get("image_url") or {}).get("url")
            if isinstance(image_url, str) and image_url.startswith("data:image"):
                urls.append(image_url)
    return urls


def _extract_audio_payloads(chat_completion: Any) -> list[bytes]:
    payloads: list[bytes] = []
    for choice in getattr(chat_completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        message_audio = getattr(message, "audio", None)
        if message_audio is None:
            continue
        data_b64 = getattr(message_audio, "data", None)
        if isinstance(data_b64, str) and data_b64:
            try:
                payloads.append(base64.b64decode(data_b64))
            except Exception:
                continue
    return payloads


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    header, data = data_url.split(",", 1)
    mime_type = "image/png"
    if ";" in header and ":" in header:
        mime_type = header.split(":", 1)[1].split(";", 1)[0]
    return base64.b64decode(data), mime_type


def _image_extension_from_mime(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type == "image/gif":
        return ".gif"
    return ".png"


def _save_outputs(
    *,
    query_type: str,
    chat_completion: Any,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    text_outputs = _extract_text_outputs(chat_completion)
    image_data_urls = _extract_image_data_urls(chat_completion)
    audio_payloads = _extract_audio_payloads(chat_completion)

    if text_outputs:
        text_path = output_dir / f"{query_type}_{stamp}.txt"
        text_path.write_text("\n\n".join(text_outputs) + "\n", encoding="utf-8")
        print(f"[dynin-online] text saved: {text_path}")
        print(text_outputs[0])

    for idx, image_url in enumerate(image_data_urls):
        image_bytes, mime_type = _decode_data_url(image_url)
        ext = _image_extension_from_mime(mime_type)
        image_path = output_dir / f"{query_type}_{stamp}_{idx}{ext}"
        image_path.write_bytes(image_bytes)
        print(f"[dynin-online] image saved: {image_path}")

    for idx, audio_bytes in enumerate(audio_payloads):
        audio_path = output_dir / f"{query_type}_{stamp}_{idx}.wav"
        audio_path.write_bytes(audio_bytes)
        print(f"[dynin-online] audio saved: {audio_path}")

    if not text_outputs and not image_data_urls and not audio_payloads:
        print("[dynin-online] no output extracted from response")
        raw_path = output_dir / f"{query_type}_{stamp}_raw_response.json"
        try:
            if hasattr(chat_completion, "model_dump_json"):
                serialized = chat_completion.model_dump_json(indent=2)
            else:
                if hasattr(chat_completion, "model_dump"):
                    raw_payload: Any = chat_completion.model_dump(mode="json")
                else:
                    raw_payload = chat_completion
                try:
                    serialized = json.dumps(raw_payload, ensure_ascii=False, indent=2)
                except Exception:
                    serialized = json.dumps({"repr": repr(raw_payload)}, ensure_ascii=False, indent=2)
            raw_path.write_text(serialized + "\n", encoding="utf-8")
            print(f"[dynin-online] raw response saved: {raw_path}")
        except Exception:
            pass


def _build_offline_parity_sampling_params_list() -> list[dict[str, Any]]:
    return [dict(OFFLINE_PARITY_STAGE_SAMPLING) for _ in range(OFFLINE_PARITY_STAGE_COUNT)]


def run_request(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{args.host}:{args.port}/v1",
    )

    prompt = args.prompt.strip() if args.prompt else DEFAULT_PROMPT_BY_QUERY[args.query_type]
    try:
        user_content, runtime_info = _build_end2end_style_inputs(
            query_type=args.query_type,
            prompt=prompt,
            image_path=args.image_path,
            model=args.model,
        )
    except Exception as exc:
        print(f"[dynin-online] warning: end2end-style input build failed, fallback to simple mode: {exc}")
        user_content = _build_user_content(
            query_type=args.query_type,
            prompt=prompt,
            image_path=args.image_path,
        )
        runtime_info = {}

    if args.modalities:
        modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    else:
        modalities = DEFAULT_MODALITIES_BY_QUERY[args.query_type]

    extra_body = {
        "sampling_params_list": _build_offline_parity_sampling_params_list(),
    }
    if runtime_info:
        # Carry offline-style runtime information in OpenAI request body.
        # Server-side path must forward this into engine prompt as additional_information.
        extra_body["additional_information"] = runtime_info

    chat_completion = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": user_content}],
        modalities=modalities,
        extra_body=extra_body,
    )

    _save_outputs(
        query_type=args.query_type,
        chat_completion=chat_completion,
        output_dir=Path(args.output_dir).expanduser(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynin-Omni online chat completion client")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="t2i",
        choices=QUERY_CHOICES,
        help="Dynin query type",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name/path",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host/IP of the vLLM Omni API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="Port of the vLLM Omni API server",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="",
        help="Custom prompt text",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Image path/URL for i2i",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="",
        help="Comma-separated output modalities override (e.g., text,image,audio)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    run_request(args)


if __name__ == "__main__":
    main()
