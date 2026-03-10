#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import re
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any


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
        print(f"[easy_example] Downloading model to local dir: {local_dir}")
        snapshot_download(
            repo_id=model,
            local_dir=str(local_dir),
            local_dir_use_symlinks=True,
            resume_download=True,
        )
    return local_dir.resolve()


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


def extract_text(outputs: list[Any], tokenizer: Any) -> str:
    for omni_out in outputs:
        if getattr(omni_out, "final_output_type", None) != "text":
            continue
        req_out = getattr(omni_out, "request_output", None)
        req_out_list = req_out if isinstance(req_out, list) else [req_out]
        for item in req_out_list:
            if item is None or not getattr(item, "outputs", None):
                continue
            completion = item.outputs[0]
            mm_out = (
                getattr(completion, "multimodal_output", None)
                or getattr(item, "multimodal_output", None)
                or getattr(omni_out, "multimodal_output", None)
                or {}
            )
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
            fallback = getattr(completion, "text", None)
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
    return ""


def build_prompt(
    tokenizer: Any, question: str, dynin_config_path: str, max_new_tokens: int, steps: int, block_length: int
) -> dict[str, Any]:
    # Match mmu_generate.py style: build a chat prompt then tokenize to ids.
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    else:
        encoded = tokenizer(question, return_tensors="pt", add_special_tokens=True)

    prompt_token_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded.get("attention_mask")

    additional_information: dict[str, Any] = {
        "task": ["mmu"],
        "prompt_length": [len(prompt_token_ids)],
        "dynin_config_path": [dynin_config_path],
        "max_new_tokens": [int(max_new_tokens)],
        "steps": [int(steps)],
        "block_length": [int(block_length)],
        "temperature": [0.0],
    }
    if attention_mask is not None:
        additional_information["attention_mask"] = [attention_mask[0].tolist()]

    return {
        "prompt_token_ids": prompt_token_ids,
        "additional_information": additional_information,
        "modalities": ["text"],
    }


def parse_args(repo_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DYNIN Omni text-to-text example using vllm_omni.")
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
        help="Path to DYNIN model config yaml.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="/tmp/dynin_localized_models",
        help="Cache dir used when --model is HF repo id.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is 2 + 2? Answer in one short sentence.",
        help="Question to ask.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    repo_root = bootstrap_repo_path()
    ensure_safe_import_for_vllm()
    args = parse_args(repo_root)

    dynin_config_path = str(Path(args.dynin_config_path).expanduser())
    os.environ["DYNIN_CONFIG_PATH"] = dynin_config_path

    model_dir = ensure_local_model_dir(args.model, cache_dir=Path(args.model_cache_dir))

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    prompt = build_prompt(
        tokenizer=tokenizer,
        question=args.question,
        dynin_config_path=dynin_config_path,
        max_new_tokens=args.max_new_tokens,
        steps=args.steps,
        block_length=args.block_length,
    )

    omni = Omni(model=str(model_dir), stage_configs_path=args.stage_config_path)
    try:
        sampling_params_list = [
            SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, detokenize=False)
            for _ in range(len(omni.stage_list))
        ]
        outputs = list(omni.generate(prompt, sampling_params_list))
    finally:
        omni.close()
    answer = extract_text(outputs, tokenizer=tokenizer)
    if not answer:
        raise RuntimeError("No text answer found in output.")

    print("Model   :", model_dir)
    print("Question:", args.question)
    print("Answer  :", answer)


if __name__ == "__main__":
    main()

"""
example commands:
python <REPO_ROOT>/examples/offline_inference/dynin_omni/t2t.py\
    --model snu-aidas/Dynin-Omni\
    --dynin-config-path <REPO_ROOT>/vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml\
    --question "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
"""
