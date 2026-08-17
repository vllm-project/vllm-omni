# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic cross-stage handoff for disaggregated diffusion pipelines.

A producing diffusion stage (e.g. an ``encode`` stage running only the text
encoder, or a ``denoise`` stage emitting latents) surfaces its payload on
``DiffusionOutput.custom_output`` (exposed on the stage request output as
``_custom_output``). This processor threads that payload — plus the connector
transfer handle, if any — into the consuming stage's prompt dict so the
downstream stage skips the work the upstream stage already did.

This processor is model-agnostic: it forwards *whatever* keys the upstream
stage published on
``custom_output`` (prompt embeddings, negative embeddings, latents, conditioning
tensors, ...) together with the diffusion sampling/control fields, so any DiT
model can be disaggregated by declaring stage roles in config rather than
writing a bespoke processor.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Diffusion sampling/control fields worth forwarding verbatim to the next stage.
_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "negative_prompt",
    "height",
    "width",
    "num_frames",
    "num_inference_steps",
    "guidance_scale",
    "guidance_scale_2",
    "boundary_ratio",
    "fps",
    "seed",
    "modalities",
)

# Suffix marking connector transfer-handle entries on ``custom_output`` (e.g.
# ``_encode_embed_transfer`` / ``_latents_transfer``). These are forwarded so
# the consuming stage can pull the real tensor payload worker-to-worker.
_TRANSFER_HANDLE_SUFFIX = "_transfer"


def _as_dict(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, dict):
        return prompt
    if hasattr(prompt, "_asdict"):
        return prompt._asdict()
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    return {}


def _extract_custom_output(source_output: Any) -> dict[str, Any]:
    """Pull the producing stage's emitted payload from a stage output object."""
    for attr in ("_custom_output", "custom_output"):
        value = getattr(source_output, attr, None)
        if isinstance(value, dict) and value:
            return value
    return {}


def diffusion_stage_handoff(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[dict[str, Any]]:
    """Build the next diffusion stage's prompt dicts from upstream outputs.

    Accepts the stage-pool transition interface
    ``processor(source_outputs, prompt, requires_multimodal_data)``. Forwards
    every non-internal ``custom_output`` key published by the upstream stage,
    along with any connector transfer handle(s) and the diffusion sampling
    fields, into the downstream stage's prompt dict.
    """
    del requires_multimodal_data, streaming_context

    if not isinstance(prompt, list):
        prompts = [prompt] if prompt is not None else [{}]
    else:
        prompts = prompt

    diffusion_inputs: list[dict[str, Any]] = []
    for i, source_output in enumerate(source_outputs):
        original_prompt = _as_dict(prompts[i] if i < len(prompts) else {})
        custom_output = _extract_custom_output(source_output)

        next_prompt: dict[str, Any] = {}
        # Keep the raw text for logging/metadata; the downstream pipeline drops
        # it when payload tensors (embeddings/latents) are present.
        if original_prompt.get("prompt") is not None:
            next_prompt["prompt"] = original_prompt["prompt"]
        for key in _PASSTHROUGH_KEYS:
            if original_prompt.get(key) is not None:
                next_prompt[key] = original_prompt[key]

        payload_keys: list[str] = []
        handle_keys: list[str] = []
        for key, value in custom_output.items():
            if value is None:
                continue
            if key.startswith("_"):
                # Internal entries: forward connector transfer handles only.
                if key.endswith(_TRANSFER_HANDLE_SUFFIX):
                    next_prompt[key] = value
                    handle_keys.append(key)
                continue
            next_prompt[key] = value
            payload_keys.append(key)

        if not payload_keys and not handle_keys:
            logger.warning(
                "[diffusion_stage_handoff] request %d: upstream custom_output "
                "carried no payload (keys=%s); downstream stage will fall back "
                "to running the upstream work itself.",
                i,
                list(custom_output.keys()),
            )

        diffusion_inputs.append(next_prompt)

    return diffusion_inputs
