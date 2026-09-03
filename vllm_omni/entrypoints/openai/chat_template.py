# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chat-template loading helpers for OpenAI server bootstrap."""

import json
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)


def _load_model_chat_template_json(model: str) -> str | None:
    """Load a model-level chat_template.json from a local path or HF cache.

    Some multimodal HF repos, including Qwen3-Omni, ship the chat template as a
    separate file instead of embedding it in tokenizer_config.json. Transformers
    4.44+ no longer supplies a default template, so serving must pass that model
    template explicitly when the user did not provide --chat-template.
    """
    candidate = Path(model) / "chat_template.json"
    template_path: str | None = str(candidate) if candidate.is_file() else None

    if template_path is None:
        try:
            from huggingface_hub import hf_hub_download

            template_path = hf_hub_download(
                repo_id=model,
                filename="chat_template.json",
                local_files_only=True,
            )
        except Exception:
            return None

    try:
        with open(template_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load chat template from %s: %s", template_path, exc)
        return None

    if isinstance(payload, dict):
        template = payload.get("chat_template")
    elif isinstance(payload, str):
        template = payload
    else:
        template = None

    if not isinstance(template, str) or not template.strip():
        logger.warning("Ignoring malformed chat template payload in %s", template_path)
        return None

    logger.info("Loaded chat template from %s", template_path)
    return template
