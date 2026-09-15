# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route diffusion stage outputs without owning transport or session state.

Inline custom_output payloads merge into additional_information, matching
connector reception. Transfer handles stay at the prompt top level for the
runner. A failed send retains inline data and follows the same route.
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

# Recognize per-edge transfer handles in addition to the runner's canonical key.
_TRANSFER_HANDLE_SUFFIX = "_transfer"


def _stage_payload_handle_key() -> str:
    """Load the runner's canonical handle key lazily to avoid worker imports at stage init."""
    try:
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        return str(DiffusionModelRunner._STAGE_PAYLOAD_HANDLE_KEY)
    except (ImportError, AttributeError):  # pragma: no cover - orchestrator without worker deps
        # The suffix rule below still forwards the handle; this only affects the
        # exact-match fast path.
        return "_stage_payload_transfer"


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
    """Build downstream prompts through the orchestrator transition interface.

    Merge payloads into additional_information and place transfer handles
    at the top level for the runner.
    """
    del requires_multimodal_data, streaming_context

    if not isinstance(prompt, list):
        prompts = [prompt] if prompt is not None else [{}]
    else:
        prompts = prompt

    handle_key = _stage_payload_handle_key()

    diffusion_inputs: list[dict[str, Any]] = []
    for i, source_output in enumerate(source_outputs):
        original_prompt = _as_dict(prompts[i] if i < len(prompts) else {})
        custom_output = _extract_custom_output(source_output)

        next_prompt: dict[str, Any] = {}
        # Preserve text for logging; payload tensors supply downstream conditioning.
        if original_prompt.get("prompt") is not None:
            next_prompt["prompt"] = original_prompt["prompt"]
        for key in _PASSTHROUGH_KEYS:
            if original_prompt.get(key) is not None:
                next_prompt[key] = original_prompt[key]
        # Merge payload fields without replacing existing additional_information.
        additional: dict[str, Any] = dict(original_prompt.get("additional_information") or {})

        payload_keys: list[str] = []
        handle_keys: list[str] = []
        for key, value in custom_output.items():
            if value is None:
                continue
            if key == handle_key or (key.startswith("_") and key.endswith(_TRANSFER_HANDLE_SUFFIX)):
                # Transfer handles are runner-facing, not pipeline-facing.
                next_prompt[key] = value
                handle_keys.append(key)
                continue
            if key.startswith("_"):
                # Other internal entries are not part of the payload contract.
                continue
            additional[key] = value
            payload_keys.append(key)

        if additional:
            next_prompt["additional_information"] = additional

        if not payload_keys and not handle_keys:
            logger.warning(
                "[diffusion_stage_handoff] request %d: upstream custom_output "
                "carried no payload (keys=%s); the downstream stage will have to "
                "fall back to running the upstream work itself.",
                i,
                list(custom_output.keys()),
            )

        diffusion_inputs.append(next_prompt)

    return diffusion_inputs
