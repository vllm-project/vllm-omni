# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic cross-stage handoff for disaggregated diffusion pipelines.

A producing diffusion stage (e.g. an ``encode`` stage running only the
encoders, or a ``denoise`` stage emitting latents) surfaces its payload on
``DiffusionOutput.custom_output`` (exposed on the stage request output as
``_custom_output``). This processor threads that payload -- plus the connector
transfer handle, if any -- into the consuming stage's prompt dict so the
downstream stage skips the work the upstream stage already did.

This processor is model-agnostic: it forwards *whatever* keys the upstream stage
published on ``custom_output`` (prompt embeddings, latents, conditioning
tensors, opaque model payload dicts, ...) together with the diffusion
sampling/control fields, so any DiT model can be disaggregated by declaring
stage roles in config rather than writing a bespoke processor.

It is also transport-neutral and stateless. It never selects a connector, never
calls ``put``/``get``, and owns no TP broadcast, KV or session lifecycle. It only
routes, and it handles all three states the send path can leave behind:

1. **inline** -- the payload keys are still on ``custom_output`` (no connector
   edge configured), so they are carried inline to the next stage;
2. **connector** -- the send succeeded, so the payload keys are gone and only the
   transfer handle is forwarded;
3. **send failed** -- no handle was produced and the inline payload is untouched,
   so this degrades to case 1.

Payload entries land in ``prompt["additional_information"]``, which is exactly
where the worker-side connector receive path writes them. A pipeline therefore
reads the same location under either transport and needs no transport branch.
The transfer handle goes to the top level of the prompt instead, because that is
where the runner pops it from.
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

# Suffix marking connector transfer-handle entries on ``custom_output``. The
# canonical handle key is owned by the generic runner (see
# ``_stage_payload_handle_key``); this suffix additionally catches any further
# per-edge handles a producer may publish, without this module having to know
# their names.
_TRANSFER_HANDLE_SUFFIX = "_transfer"


def _stage_payload_handle_key() -> str:
    """Return the runner's stage-payload transfer handle key.

    Read from the generic runner rather than restated here, so the producer, this
    router and the consumer cannot drift apart. Imported lazily: this module is
    loaded in the orchestrator process at stage-init time, which has no reason to
    pull in the worker's import graph until a handoff actually happens.
    """
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
    """Build the next diffusion stage's prompt dicts from upstream outputs.

    Accepts the orchestrator's transition interface
    ``processor(source_outputs, prompt, requires_multimodal_data)``. Payload keys
    published by the upstream stage go into
    ``next_prompt["additional_information"]`` -- the same place the worker-side
    connector receive path writes them -- while connector transfer handles go to
    the top level of the prompt, where the runner pops them from.
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
        # Keep the raw text for logging/metadata; the downstream pipeline drops
        # it when payload tensors (embeddings/latents) are present.
        if original_prompt.get("prompt") is not None:
            next_prompt["prompt"] = original_prompt["prompt"]
        for key in _PASSTHROUGH_KEYS:
            if original_prompt.get(key) is not None:
                next_prompt[key] = original_prompt[key]
        # Preserve anything the upstream prompt already carried here so the
        # payload merge below adds to it rather than replacing it.
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
