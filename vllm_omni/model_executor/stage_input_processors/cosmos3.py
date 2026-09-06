# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-input processor for the Cosmos3 reasoner -> generator handoff.

The Cosmos3 analogue of ``glm_image.ar2diffusion``: it turns stage 0's output
into stage 1's prompt. GLM-Image passes ``prior_token_ids``; Cosmos3 passes the
UND tower's per-layer K/V, which travels in ``extra`` and is installed on the
generator's replay stub.

The signature follows the ``(source_outputs, prompt, requires_multimodal_data)``
convention the orchestrator dispatches on for a ``diffusion`` next stage, the same
one ``glm_image.ar2diffusion`` uses.

The payload keys are imported from ``diffusion.models.cosmos3_pipeline_config``,
not from the tower pipelines: this module runs in the orchestrator process, which
must not pay for a diffusion-pipeline import just to read two strings.

WHAT THIS DOES *NOT* FORWARD
----------------------------
Generation knobs -- seed, ``num_inference_steps``, ``guidance_scale``,
``flow_shift`` -- are deliberately absent. Those ride to every stage in
``sampling_params_list``, threaded through unchanged from the original request;
re-copying them into the prompt dict would create a second, competing source of
truth.

The reasoner's geometry and tokenization settings are not forwarded either, for
the same reason: the generator resolves them from those same sampling params,
through the very helpers the reasoner used (``_resolve_t2i_geometry`` /
``_resolve_text_encode_params``), so a prompt-dict copy would be a second source
of truth for values the generator does not read from the prompt dict anyway. They
travel in ``META_KEY`` instead, purely as diagnostics -- the generator compares
the reasoner's declared K/V layout against its own and reports a stage-config
mismatch in those terms rather than as a bare shape error.

So what this forwards is exactly three things: the prompt text, the
``modalities`` marker the generator's own T2I check keys off, and the K/V.
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
    """Locate the reasoner payload in whatever envelope stage 0 produced.

    The reasoner's postprocessor parks the K/V under the ``trajectory`` payload
    key, and ``output_formatter._build_multimodal_output`` copies ``trajectory``
    verbatim into ``multimodal_output`` -- so the canonical location is
    ``multimodal_output["trajectory"][KV_KEY]``. The un-nested form is accepted
    too, for driving this bridge directly from a hand-built payload in a test.

    Both the ``RequestOutput`` and the inner ``CompletionOutput`` are probed
    because ``multimodal_output`` is a dynamic attribute on both: the connector
    serde restores it wherever it was set (``_decode_request_output`` /
    ``_decode_completion_output`` in
    ``vllm_omni/distributed/omni_connectors/utils/serialization.py``), so which
    holder carries it depends on the transport.
    """
    for output in source_outputs:
        for holder in (output, *(getattr(output, "outputs", None) or [])):
            mm = getattr(holder, "multimodal_output", None)
            if not isinstance(mm, dict):
                continue
            if KV_KEY in mm:
                return mm
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
        # The generator stage re-runs _is_t2i_request, which keys purely off this
        # list, so it has to be restated here.
        "modalities": ["image"],
        "extra": {
            KV_KEY: payload[KV_KEY],
            META_KEY: meta,
        },
    }

    # The negative prompt must match too: with guidance active the reasoner
    # encoded an unconditional branch from it, and the generator re-tokenizes
    # that same text to look the branch up.
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
