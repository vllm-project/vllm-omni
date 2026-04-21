#!/usr/bin/env python3
"""Run a one-shot MiniCPM-o 4.5 smoke test on Modal via local vLLM-Omni.

Usage:
    modal run scripts/modal_minicpmo45_omni_smoke.py --preload-only
    modal run scripts/modal_minicpmo45_omni_smoke.py
    modal run scripts/modal_minicpmo45_omni_smoke.py --prompt "Hello from Modal"

This script installs:
1. pip `vllm`
2. vLLM-Omni runtime dependencies
3. the local `vllm-omni` source tree from this workspace via Modal sync
3. MiniCPM-o 4.5 weights in a shared HF cache volume

It then runs one tiny text-only Omni generation as a smoke test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import modal

APP_NAME: Final = "minicpmo45-omni-smoke"
MODEL_ID: Final = "openbmb/MiniCPM-o-4_5"
VLLM_VERSION: Final = "0.19.0"
GPU_TYPE: Final = "A100-80GB"
MAX_MODEL_LEN: Final = 4096
HF_CACHE_DIR: Final = "/root/.cache/huggingface"
VLLM_CACHE_DIR: Final = "/root/.cache/vllm"
REMOTE_REPO_ROOT: Final = Path("/root/vllm-omni-local")
REMOTE_REQUIREMENTS_ROOT: Final = Path("/root/vllm-omni-requirements")
REMOTE_STAGE_CONFIG: Final = REMOTE_REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "minicpmo.yaml"
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


def _download_model() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        resume_download=True,
    )


def _build_text_prompt(question: str) -> tuple[str, list[int]]:
    from transformers import AutoTokenizer
    from vllm.entrypoints.chat_utils import load_chat_template
    from vllm.transformers_utils.chat_templates import get_chat_template_fallback_path

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    chat_template = None
    if hasattr(tokenizer, "get_chat_template"):
        try:
            chat_template = tokenizer.get_chat_template(chat_template=None)
        except Exception:
            chat_template = None

    if chat_template is None:
        fallback_path = get_chat_template_fallback_path("minicpmv", tokenizer.name_or_path)
        if fallback_path is not None:
            chat_template = load_chat_template(fallback_path)

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
    )
    stop_tokens = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in stop_tokens]

    return formatted_prompt, stop_token_ids


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
    gpu=GPU_TYPE,
    timeout=45 * MINUTES,
    volumes={
        HF_CACHE_DIR: hf_cache_volume,
        VLLM_CACHE_DIR: vllm_cache_volume,
    },
)
def run_smoke(
    prompt: str = "Please answer briefly: what is two plus two?",
    max_tokens: int = 4096,
) -> dict[str, str]:
    from vllm.sampling_params import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    snapshot_path = _download_model()
    hf_cache_volume.commit()
    formatted_prompt, stop_token_ids = _build_text_prompt(prompt)

    omni = Omni(
        model=MODEL_ID,
        stage_configs_path=str(REMOTE_STAGE_CONFIG),
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        stage_init_timeout=20 * MINUTES,
        init_timeout=30 * MINUTES,
        enforce_eager=True,
    )

    try:
        outputs = omni.generate(
            formatted_prompt,
            sampling_params_list=SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
            ),
            use_tqdm=False,
        )
        if not outputs:
            raise RuntimeError("MiniCPMO smoke test returned no outputs.")

        request_output = outputs[0].request_output
        if request_output is None or not request_output.outputs:
            raise RuntimeError("MiniCPMO smoke test returned no text completions.")

        text = request_output.outputs[0].text
        return {
            "formatted_prompt": formatted_prompt,
            "snapshot_path": snapshot_path,
            "text": text,
        }
    finally:
        omni.close()


@app.local_entrypoint()
def main(
    prompt: str = "Please answer briefly: what is two plus two?",
    max_tokens: int = 4096,
    preload_only: bool = False,
) -> None:
    snapshot_path = preload_model.remote()
    print(f"Model cached at: {snapshot_path}")

    if preload_only:
        return

    result = run_smoke.remote(prompt=prompt, max_tokens=max_tokens)
    print(json.dumps(result, indent=2, sort_keys=True))
