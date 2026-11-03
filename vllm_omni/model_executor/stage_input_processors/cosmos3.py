# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-input processor for the Cosmos3 reasoner -> generator handoff.

Like ``glm_image.ar2diffusion``, but the payload is the UND tower's per-layer K/V,
carried in ``extra`` for the generator's replay stub. It forwards only the prompt
text, the negative prompt, the ``modalities`` marker, and the K/V plus its
metadata. Generation knobs, geometry, and tokenization settings reach every stage
through ``sampling_params_list``, so they are not copied here.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_KV_KEY as KV_KEY,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_META_KEY as META_KEY,
)

logger = init_logger(__name__)


def _as_dict(prompt: Any) -> dict[str, Any]:
    """Normalize the several prompt shapes the stage plumbing may hand us."""
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    if prompt is None:
        return {}
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _find_und_payload(source_outputs: list[Any]) -> dict[str, Any] | None:
    """Return ``multimodal_output["trajectory"]`` holding the K/V.

    It is on either the request or its completion output, depending on the transport.
    """
    for output in source_outputs:
        for holder in (output, *(getattr(output, "outputs", None) or [])):
            mm = getattr(holder, "multimodal_output", None)
            if not isinstance(mm, dict):
                continue
            trajectory = mm.get("trajectory")
            if isinstance(trajectory, dict) and KV_KEY in trajectory:
                return trajectory
    return None


def reasoner2generator(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> dict[str, Any] | None:
    """Build the generator-stage prompt from the reasoner stage's output."""
    del requires_multimodal_data  # T2I: no image input to forward.

    if not source_outputs:
        logger.warning("[reasoner2generator] no reasoner output; dropping request")
        return None

    payload = _find_und_payload(source_outputs)
    if payload is None:
        logger.warning("[reasoner2generator] reasoner output carried no %s; dropping request", KV_KEY)
        return None

    original = _as_dict(prompt)
    meta = payload.get(META_KEY) or {}

    generator_input: dict[str, Any] = {
        "prompt": original.get("prompt", ""),
        # The generator's _is_t2i_request keys off this.
        "modalities": ["image"],
        # The output formatter strips ``extra`` before echoing the prompt.
        "extra": {
            KV_KEY: payload[KV_KEY],
            META_KEY: meta,
        },
    }

    # The generator re-tokenizes it to look up the unconditional branch.
    if original.get("negative_prompt") is not None:
        generator_input["negative_prompt"] = original["negative_prompt"]

    logger.info(
        "[reasoner2generator] forwarding %d K/V branch(es) (%s MiB) target=%sx%s",
        len(payload[KV_KEY]),
        meta.get("payload_mib", "?"),
        meta.get("height", "?"),
        meta.get("width", "?"),
    )
    return generator_input
