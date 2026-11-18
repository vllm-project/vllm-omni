# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 reasoner -> generator handoff: forwards the prompt plus the UND K/V."""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.models.cosmos3_pipeline_config import COSMOS3_UND_KV_KEY as KV_KEY

logger = init_logger(__name__)


def _find_und_kv(source_outputs: list[Any]) -> Any:
    # The payload sits on the request or its completion output, depending on the transport.
    for output in source_outputs:
        for holder in (output, *(getattr(output, "outputs", None) or [])):
            trajectory = (getattr(holder, "multimodal_output", None) or {}).get("trajectory")
            if isinstance(trajectory, dict) and KV_KEY in trajectory:
                return trajectory[KV_KEY]
    return None


def reasoner2generator(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> dict[str, Any] | None:
    """Build the generator-stage prompt from the reasoner stage's output."""
    del requires_multimodal_data
    kv = _find_und_kv(source_outputs or [])
    if kv is None:
        logger.warning("[reasoner2generator] reasoner output carried no %s; dropping request", KV_KEY)
        return None

    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else None
    original = prompt if isinstance(prompt, dict) else {"prompt": prompt or ""}

    generator_input: dict[str, Any] = {
        "prompt": original.get("prompt", ""),
        "modalities": ["image"],
        "extra": {KV_KEY: kv},
    }
    # The generator re-tokenizes it to find the unconditional branch.
    if original.get("negative_prompt") is not None:
        generator_input["negative_prompt"] = original["negative_prompt"]
    return generator_input
