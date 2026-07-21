# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generic diffusion -> diffusion stage transition processor.

One model-agnostic adapter moves a :class:`StagePayload` from an upstream
diffusion stage's output into the downstream diffusion stage's request, for every
disaggregated diffusion model. It does *not* decode model-specific tensor keys;
the payload round-trips opaquely through the owning pipeline's
``pack_stage_state`` / ``unpack_stage_state`` hooks (the DiffusionV2Atoms
contract).

Wiring: set a stage's ``custom_process_input_func`` to
``vllm_omni.model_executor.stage_input_processors.diffusion.diffusion_stage_transition``.
The orchestrator calls it as
``fn(source_outputs, prompt, requires_multimodal_data, sampling_params=...)`` and
uses the returned prompt dict as the downstream ``OmniDiffusionRequest.prompt``.

The payload rides in the upstream ``DiffusionOutput.custom_output`` under
:data:`STAGE_PAYLOAD_OUTPUT_KEY`, and is re-emitted into the downstream prompt's
``extra`` sub-dict under :data:`STAGE_PAYLOAD_PROMPT_KEY` (the channel the
diffusion pipeline reads for cross-stage data). Both the upstream
``DiffusionOutput.custom_output`` and the prompt ``extra`` dict are already
part of the msgpack transport contract; msgpack does not preserve dataclass
identity across the out-of-process (multiproc) stage client, so
``_unwrap_stage_payload`` rehydrates the payload via ``StagePayload.from_dict``
when it arrives as a plain dict.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Canonical dotted path to :func:`diffusion_stage_transition`, co-located with
#: the function it names so a pipeline wiring the generic processor references one
#: source of truth instead of hand-copying the string. Declared as a literal (not
#: derived from ``__name__``) because the torch-free foundation tests load this
#: module by file path under a synthetic name (see conftest.py), which would make
#: any ``__name__``-derived path wrong.
GENERIC_DIFFUSION_PROCESSOR = (
    "vllm_omni.model_executor.stage_input_processors.diffusion.diffusion_stage_transition"
)

# STAGE_PAYLOAD_*_KEY: canonical definitions live in
# ``vllm_omni.diffusion.stage_payload`` (the torch-free transport leaf). They are
# re-declared here as literals — not imported — on purpose: this processor is
# loaded by file path in the torch-free foundation tests (see
# tests/diffusion/disaggregated/conftest.py), and a top-level ``from vllm_omni...``
# import would trigger the package ``__init__`` (and torch). ``test_payload_key_sync``
# asserts these literals stay equal to the canonical constants so the copies cannot drift.

#: Key under which a stage's ``DiffusionOutput.custom_output`` carries the
#: outgoing :class:`StagePayload`.
STAGE_PAYLOAD_OUTPUT_KEY = "__diffusion_stage_payload__"

#: Key under which the downstream request prompt's ``extra`` dict carries the
#: incoming :class:`StagePayload`.
STAGE_PAYLOAD_PROMPT_KEY = "diffusion_stage_payload"


def _is_stage_payload(value: Any) -> bool:
    """Duck-typed StagePayload check (no hard import of the class).

    Recognizes a payload by its distinctive attribute surface so the processor
    stays importable even in minimal environments; falls back to an ``isinstance``
    check against the real class when it is importable.
    """
    if all(
        hasattr(value, attr)
        for attr in ("request_id", "boundary", "tensor_fields", "scalar_fields", "payload_version")
    ):
        return True
    try:
        from vllm_omni.diffusion.models.interface import StagePayload

        return isinstance(value, StagePayload)
    except Exception:  # pragma: no cover - import-environment dependent
        return False


def _rehydrate_stage_payload(payload_dict: dict[str, Any]) -> Any:
    """Rebuild a :class:`StagePayload` from a msgpack-flattened plain dict."""
    from vllm_omni.diffusion.models.interface import StagePayload

    return StagePayload.from_dict(payload_dict)


def _unwrap_stage_payload(source_output: Any) -> Any | None:
    """Extract a :class:`StagePayload` from an upstream stage output.

    Accepts either a ``DiffusionOutput``-like object exposing ``custom_output``,
    a mapping, or an object that already *is* a payload. Returns ``None`` when
    no payload is present so the caller can route a terminal error.

    The out-of-process (multiproc) stage client round-trips ``custom_output``
    through msgpack, which does not preserve dataclass identity: the payload
    arrives here as a plain ``dict`` with the same keys as
    :class:`StagePayload`'s fields rather than the dataclass itself. Rehydrate it
    via :meth:`StagePayload.from_dict` before returning so :meth:`validate`
    (called by the caller) works on both paths.
    """
    if _is_stage_payload(source_output):
        return source_output

    custom_output = getattr(source_output, "custom_output", None)
    if custom_output is None and isinstance(source_output, dict):
        custom_output = source_output.get("custom_output", source_output)
    if isinstance(custom_output, dict):
        payload = custom_output.get(STAGE_PAYLOAD_OUTPUT_KEY)
        if payload is not None:
            if _is_stage_payload(payload):
                return payload
            if isinstance(payload, dict):
                return _rehydrate_stage_payload(payload)
            return payload
    return None


def _passthrough_prompt_fields(prompt: Any, target: dict[str, Any]) -> None:
    """Copy request-level knobs from the original prompt onto ``target``.

    Preserves sampling-relevant fields the downstream stage may still need
    (seed, steps, guidance, negative prompt, session id, resolution). Model
    semantics are NOT interpreted here — these are the same generic passthrough
    keys the existing AR->diffusion processors forward.
    """
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else {}
    if hasattr(prompt, "_asdict"):
        prompt = prompt._asdict()
    elif not isinstance(prompt, dict) and hasattr(prompt, "__dict__"):
        prompt = vars(prompt)
    if not isinstance(prompt, dict):
        return
    for key in (
        "seed",
        "num_inference_steps",
        "guidance_scale",
        "negative_prompt",
        "session_id",
        "height",
        "width",
    ):
        if key in prompt and key not in target:
            target[key] = prompt[key]


def diffusion_stage_transition(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    *,
    sampling_params: Any = None,
    **_ignored: Any,
) -> dict[str, Any] | None:
    """Move the upstream diffusion payload into the downstream request prompt.

    Args:
        source_outputs: Upstream stage outputs; ``source_outputs[0]`` carries
            the :class:`StagePayload` in its ``custom_output``.
        prompt: The original request prompt (for request-level passthrough).
        requires_multimodal_data: Honored generically — unused here because the
            payload already carries every conditioning tensor the model packed.
        sampling_params: The downstream stage's sampling params (accepted so the
            orchestrator's signature probe passes it; also mirrored so the
            pipeline's import hook can recover a session id from ``extra_args``).

    Returns:
        A diffusion prompt dict carrying the payload in ``extra``, or ``None``
        when no payload is present (the orchestrator routes a terminal error).
    """
    del requires_multimodal_data  # payload is self-contained; kept for contract parity

    if not source_outputs:
        logger.warning("[diffusion_stage_transition] no source outputs; routing terminal error")
        return None

    payload = _unwrap_stage_payload(source_outputs[0])
    if payload is None:
        logger.warning(
            "[diffusion_stage_transition] upstream output carries no %s; routing terminal error",
            STAGE_PAYLOAD_OUTPUT_KEY,
        )
        return None

    # Validate the envelope at the transition boundary. The processor stays
    # model-agnostic: it checks identity/transition/versioning only, never the
    # model-private tensor keys or metadata.
    try:
        payload.validate()
    except Exception as exc:  # StagePayloadError and subclasses
        logger.error("[diffusion_stage_transition] invalid stage payload: %s", exc)
        return None

    diffusion_prompt: dict[str, Any] = {
        "prompt": "",
        "extra": {STAGE_PAYLOAD_PROMPT_KEY: payload},
    }

    # Mirror the session id into extra_args so the AR-Diffusion denoise runner
    # (which keys session state by extra_args["session_id"]) attaches the right
    # live session — this is generic session-id plumbing, not model semantics.
    # session_id rides in the PUBLIC ``scalar_fields`` so the processor never
    # decodes model-private schema.
    session_id = payload.scalar_fields.get("session_id")
    if session_id is not None:
        diffusion_prompt["session_id"] = session_id

    _passthrough_prompt_fields(prompt, diffusion_prompt)

    logger.debug(
        "[diffusion_stage_transition] req=%s boundary=%s tensors=%d private_tensors=%d",
        payload.request_id,
        getattr(payload.boundary, "value", payload.boundary),
        len(payload.tensor_fields),
        len(payload.private_tensor_fields),
    )
    return diffusion_prompt
