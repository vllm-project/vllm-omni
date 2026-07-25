# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""InternVL-U CFG expansion and AR-to-diffusion input bridge.

The released checkpoint fixes the token IDs below.  Stage 0 still resolves
special-token IDs from its tokenizer when constructing embeddings; these
constants are used only to validate and slice that checkpoint's serialized
token stream at the stage boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

PARTIAL_SUFFIX = "__internvlu_partial"
UNCONDITIONAL_SUFFIX = "__internvlu_unconditional"

# Do not reorder CFG_ROLES: Stage 1 builds its CFG batch in this order
# and unpacks the denoiser output positionally against it.
CONDITIONAL_ROLE = "conditional"
PARTIAL_ROLE = "partial"
UNCONDITIONAL_ROLE = "unconditional"
CFG_ROLES = (CONDITIONAL_ROLE, PARTIAL_ROLE, UNCONDITIONAL_ROLE)

IM_START_TOKEN_ID = 151644
EOS_TOKEN_ID = 151645
IMG_START_TOKEN_ID = 151669
IMG_END_TOKEN_ID = 151670
IMG_CONTEXT_TOKEN_ID = 151671
IMG_CONTEXT_TOKENS_PER_IMAGE = 256

PARTIAL_EDIT_TEXT = "Generate an image based on reference images."
PURE_UNCONDITIONAL_TEXT = "Here is a random image <img_uncond>:"

_GENERATION_MODE_KEY = "internvlu_generation_mode"
_IMAGE_ALIASES = ("image", "images", "img2img")


@dataclass(frozen=True)
class ExpandedPrompt:
    """An InternVL-U CFG companion consumed by ``AsyncOmniEngine``."""

    prompt: dict[str, Any]
    role: str
    request_id_suffix: str
    sampling_params_override: dict[str, Any] | None = None

    def apply_overrides(
        self,
        base_params: Any,
        base_spl: list[Any],
    ) -> tuple[Any, list[Any]]:
        if not self.sampling_params_override:
            return base_params, base_spl

        patched = base_params.clone()
        for key, value in self.sampling_params_override.items():
            setattr(patched, key, value)

        sampling_params_list = list(base_spl)
        if sampling_params_list:
            sampling_params_list[0] = patched
        return patched, sampling_params_list


def _clone_prompt(prompt: Mapping[str, Any]) -> dict[str, Any]:
    cloned = dict(prompt)
    for key in ("multi_modal_data", "mm_processor_kwargs"):
        value = cloned.get(key)
        if isinstance(value, Mapping):
            cloned[key] = dict(value)
    modalities = cloned.get("modalities")
    if isinstance(modalities, list):
        cloned["modalities"] = list(modalities)
    return cloned


def _as_prompt_dict(prompt: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(prompt, dict):
        return _clone_prompt(prompt)
    if isinstance(prompt, str):
        return {"prompt": prompt}
    raise TypeError(f"InternVL-U prompt must be a string or dict, got {type(prompt).__name__}")


def _nonempty_alias_values(mm_data: Any) -> list[tuple[str, Any]]:
    if mm_data is None:
        return []
    if not isinstance(mm_data, Mapping):
        raise TypeError("InternVL-U multi_modal_data must be a mapping")

    values: list[tuple[str, Any]] = []
    for key in _IMAGE_ALIASES:
        value = mm_data.get(key)
        if value is not None and not (isinstance(value, (list, tuple)) and not value):
            values.append((key, value))
    return values


def _flatten_images(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_images(item))
        return flattened
    return [value]


def _same_image_sequence(left: list[Any], right: list[Any]) -> bool:
    return len(left) == len(right) and all(a is b for a, b in zip(left, right))


def _extract_reference_images(prompt: Mapping[str, Any]) -> list[Any]:
    mm_values = _nonempty_alias_values(prompt.get("multi_modal_data"))
    top_values = [(key, prompt[key]) for key in _IMAGE_ALIASES if prompt.get(key) is not None]
    values = mm_values or top_values
    if not values:
        return []

    canonical = _flatten_images(values[0][1])
    for key, value in values[1:]:
        candidate = _flatten_images(value)
        if not _same_image_sequence(canonical, candidate):
            aliases = ", ".join(name for name, _ in values)
            raise ValueError(f"Conflicting InternVL-U image aliases ({aliases}); cannot determine reference order")
    if not canonical:
        raise ValueError("InternVL-U reference image list is empty")
    return canonical


def _set_generation_marker(prompt: dict[str, Any], *, editing: bool) -> None:
    modalities = prompt.get("modalities")
    if not modalities:
        prompt["modalities"] = ["image"]
    mm_kwargs = prompt.get("mm_processor_kwargs")
    if mm_kwargs is None:
        mm_kwargs = {}
    elif not isinstance(mm_kwargs, Mapping):
        raise TypeError("InternVL-U mm_processor_kwargs must be a mapping")
    else:
        mm_kwargs = dict(mm_kwargs)
    mm_kwargs[_GENERATION_MODE_KEY] = "editing" if editing else "image"
    prompt["mm_processor_kwargs"] = mm_kwargs


def _canonicalize_prompt_images(prompt: dict[str, Any]) -> None:
    refs = _extract_reference_images(prompt)
    mm_data = prompt.get("multi_modal_data")
    canonical_mm = dict(mm_data) if isinstance(mm_data, Mapping) else {}
    for key in _IMAGE_ALIASES:
        canonical_mm.pop(key, None)
        prompt.pop(key, None)
    if refs:
        canonical_mm["image"] = refs
    if canonical_mm:
        prompt["multi_modal_data"] = canonical_mm
    else:
        prompt.pop("multi_modal_data", None)

    mm_uuids = prompt.get("multi_modal_uuids")
    if isinstance(mm_uuids, Mapping):
        canonical_uuids = dict(mm_uuids)
        image_uuids = None
        for key in _IMAGE_ALIASES:
            value = canonical_uuids.pop(key, None)
            if value is not None:
                if image_uuids is not None and value != image_uuids:
                    raise ValueError("Conflicting InternVL-U image UUID aliases")
                image_uuids = value
        if refs and image_uuids is not None:
            canonical_uuids["image"] = image_uuids
        if canonical_uuids:
            prompt["multi_modal_uuids"] = canonical_uuids
        else:
            prompt.pop("multi_modal_uuids", None)


def _drop_reference_images(prompt: dict[str, Any]) -> None:
    mm_data = prompt.get("multi_modal_data")
    if isinstance(mm_data, Mapping):
        mm_data = dict(mm_data)
        for key in _IMAGE_ALIASES:
            mm_data.pop(key, None)
        if mm_data:
            prompt["multi_modal_data"] = mm_data
        else:
            prompt.pop("multi_modal_data", None)
    for key in _IMAGE_ALIASES:
        prompt.pop(key, None)

    mm_uuids = prompt.get("multi_modal_uuids")
    if isinstance(mm_uuids, Mapping):
        mm_uuids = dict(mm_uuids)
        for key in _IMAGE_ALIASES:
            mm_uuids.pop(key, None)
        if mm_uuids:
            prompt["multi_modal_uuids"] = mm_uuids
        else:
            prompt.pop("multi_modal_uuids", None)


def expand_cfg_prompts(
    prompt: dict[str, Any] | str,
    sampling_params: Any,
) -> list[ExpandedPrompt]:
    """Create InternVL-U's partial and unconditional CFG companions.

    This hook runs after the parent has been tokenized.  Therefore the native
    processor, not this function, appends the final ``<img>`` after the
    assistant generation prefix for both the parent and marked companions.
    """

    del sampling_params
    if not isinstance(prompt, dict):
        return []

    raw_modalities = prompt.get("modalities", [])
    if isinstance(raw_modalities, str):
        raw_modalities = [raw_modalities]
    elif not isinstance(raw_modalities, (list, tuple, set, frozenset)):
        return []
    modalities = {modality.strip().lower() for modality in raw_modalities if isinstance(modality, str)}
    if not modalities.intersection(("image", "img2img")):
        return []

    base = _as_prompt_dict(prompt)
    refs = _extract_reference_images(base)

    partial = _clone_prompt(base)
    partial["prompt"] = PARTIAL_EDIT_TEXT if refs else PURE_UNCONDITIONAL_TEXT
    _set_generation_marker(partial, editing=bool(refs))
    if refs:
        _canonicalize_prompt_images(partial)
    else:
        _drop_reference_images(partial)

    unconditional = _clone_prompt(base)
    unconditional["prompt"] = PURE_UNCONDITIONAL_TEXT
    _set_generation_marker(unconditional, editing=bool(refs))
    _drop_reference_images(unconditional)

    companion_sampling = {"max_tokens": 1}
    return [
        ExpandedPrompt(
            prompt=partial,
            role=PARTIAL_ROLE,
            request_id_suffix=PARTIAL_SUFFIX,
            sampling_params_override=companion_sampling,
        ),
        ExpandedPrompt(
            prompt=unconditional,
            role=UNCONDITIONAL_ROLE,
            request_id_suffix=UNCONDITIONAL_SUFFIX,
            sampling_params_override=companion_sampling,
        ),
    ]


def _request_id(source_output: Any) -> str:
    request_id = getattr(source_output, "request_id", None)
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("Every InternVL-U source output must have a non-empty request_id")
    return request_id


def _bind_roles(source_outputs: list[Any]) -> dict[str, Any]:
    if len(source_outputs) != 3:
        raise ValueError(f"InternVL-U requires exactly three CFG outputs, got {len(source_outputs)}")

    bound: dict[str, Any] = {}
    for source_output in source_outputs:
        request_id = _request_id(source_output)
        if request_id.endswith(PARTIAL_SUFFIX):
            role = PARTIAL_ROLE
        elif request_id.endswith(UNCONDITIONAL_SUFFIX):
            role = UNCONDITIONAL_ROLE
        elif "__internvlu_" in request_id:
            raise ValueError(f"Unknown InternVL-U CFG request suffix: {request_id}")
        else:
            role = CONDITIONAL_ROLE

        if role in bound:
            raise ValueError(f"Duplicate InternVL-U CFG role {role!r}")
        bound[role] = source_output

    missing = [role for role in CFG_ROLES if role not in bound]
    if missing:
        raise ValueError(f"Missing InternVL-U CFG role(s): {', '.join(missing)}")

    parent_id = _request_id(bound[CONDITIONAL_ROLE])
    expected_ids = {
        PARTIAL_ROLE: parent_id + PARTIAL_SUFFIX,
        UNCONDITIONAL_ROLE: parent_id + UNCONDITIONAL_SUFFIX,
    }
    for role, expected_id in expected_ids.items():
        actual_id = _request_id(bound[role])
        if actual_id != expected_id:
            raise ValueError(
                f"InternVL-U {role} output belongs to a different parent: expected {expected_id!r}, got {actual_id!r}"
            )
    return bound


def _completion(source_output: Any) -> Any:
    outputs = getattr(source_output, "outputs", None)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ValueError("Each InternVL-U CFG branch must contain exactly one completion output")
    return outputs[0]


def _token_list(value: Any, *, name: str, role: str) -> list[int]:
    if value is None:
        raise ValueError(f"InternVL-U {role} branch is missing {name}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"InternVL-U {role} {name} must be a token sequence") from exc


def _mapping_get(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    getter = getattr(mapping, "get", None)
    return getter(key) if callable(getter) else None


def _conditioning_tensor(source_output: Any, completion: Any, role: str) -> torch.Tensor:
    payload = getattr(source_output, "multimodal_output", None)
    if not payload:
        payload = getattr(completion, "multimodal_output", None)

    hidden_states = _mapping_get(payload, "hidden_states")
    conditioning = _mapping_get(hidden_states, "output")
    if conditioning is None:
        conditioning = _mapping_get(payload, "hidden_states.output")
    if not isinstance(conditioning, torch.Tensor):
        raise ValueError(f"InternVL-U {role} branch is missing hidden_states.output")
    # The conditioning width is checkpoint configuration owned by Stage 1
    # (generation_decoder input_hidden_size); the bridge validates only the
    # row structure its own slicing depends on.
    if conditioning.ndim != 2:
        raise ValueError(
            f"InternVL-U {role} conditioning must be a [T, hidden] matrix, got {tuple(conditioning.shape)}"
        )
    if not conditioning.is_floating_point():
        raise TypeError(f"InternVL-U {role} conditioning must be floating point")
    return conditioning.detach().cpu().contiguous()


def _reference_spans(token_ids: list[int], role: str) -> list[tuple[int, int]]:
    image_starts = [idx for idx, token_id in enumerate(token_ids) if token_id == IMG_START_TOKEN_ID]
    if not image_starts or image_starts[-1] != len(token_ids) - 1:
        raise ValueError(f"InternVL-U {role} conditioning must end at the generation <img> token")

    spans: list[tuple[int, int]] = []
    for ref_idx, start in enumerate(image_starts[:-1]):
        next_start = image_starts[ref_idx + 1]
        ends = [idx for idx in range(start + 1, next_start) if token_ids[idx] == IMG_END_TOKEN_ID]
        if len(ends) != 1:
            raise ValueError(f"InternVL-U {role} reference span {ref_idx} must contain exactly one </img> token")
        end = ends[0]
        context_ids = token_ids[start + 1 : end]
        if len(context_ids) != IMG_CONTEXT_TOKENS_PER_IMAGE or any(
            token_id != IMG_CONTEXT_TOKEN_ID for token_id in context_ids
        ):
            raise ValueError(
                f"InternVL-U {role} reference span {ref_idx} must contain exactly "
                f"{IMG_CONTEXT_TOKENS_PER_IMAGE} contiguous <IMG_CONTEXT> tokens"
            )
        spans.append((start, end + 1))

    stray_ends = [idx for idx, token_id in enumerate(token_ids) if token_id == IMG_END_TOKEN_ID]
    if len(stray_ends) != len(spans):
        raise ValueError(f"InternVL-U {role} conditioning contains an unmatched </img> token")
    return spans


def _build_branch(
    source_output: Any,
    role: str,
    expected_reference_count: int,
) -> dict[str, Any]:
    completion = _completion(source_output)
    prompt_ids = _token_list(
        getattr(source_output, "prompt_token_ids", None),
        name="prompt_token_ids",
        role=role,
    )
    output_ids = _token_list(
        getattr(completion, "cumulative_token_ids", None),
        name="cumulative_token_ids",
        role=role,
    )
    conditioning = _conditioning_tensor(source_output, completion, role)

    if not output_ids or output_ids[-1] != EOS_TOKEN_ID:
        actual = output_ids[-1] if output_ids else None
        raise ValueError(
            f"InternVL-U {role} branch must terminate with forced EOS {EOS_TOKEN_ID}, got {actual}. "
            "For text-then-image requests this usually means max_tokens ran out before the CoT "
            "reached <img>, or the deployment is missing the static think configuration "
            "(see vllm_omni/deploy/internvlu_chat_think.yaml)."
        )

    all_token_ids = prompt_ids + output_ids
    if len(all_token_ids) != conditioning.shape[0] + 1:
        raise ValueError(
            f"InternVL-U {role} token/hidden mismatch: tokens={len(all_token_ids)}, "
            f"hidden={conditioning.shape[0]}; expected tokens == hidden + 1"
        )

    aligned_ids = all_token_ids[:-1]
    im_start_positions = [idx for idx, token_id in enumerate(aligned_ids) if token_id == IM_START_TOKEN_ID]
    if len(im_start_positions) < 2:
        raise ValueError(f"InternVL-U {role} prompt has fewer than two <|im_start|> tokens")

    slice_start = im_start_positions[1]
    if aligned_ids[-1] != IMG_START_TOKEN_ID:
        raise ValueError(f"InternVL-U {role} aligned sequence does not end with <img>")
    sliced_ids = aligned_ids[slice_start:]
    sliced_hidden = conditioning[slice_start:]

    spans = _reference_spans(sliced_ids, role)
    if len(spans) != expected_reference_count:
        raise ValueError(
            f"InternVL-U {role} branch has {len(spans)} reference span(s), expected {expected_reference_count}"
        )

    reference_count = len(spans)
    image_fhw_cond = torch.tensor(
        [[1, 16, 16]] * reference_count,
        dtype=torch.long,
    ).reshape(reference_count, 3)

    return {
        "encoder_hidden_states": sliced_hidden,
        "encoder_image_token_mask": torch.tensor(
            [token_id == IMG_CONTEXT_TOKEN_ID for token_id in sliced_ids],
            dtype=torch.bool,
        ),
        "image_fhw_cond": image_fhw_cond,
        "reference_image_indices": torch.arange(reference_count, dtype=torch.long),
    }


def _prompt_mapping(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, list):
        if len(prompt) != 1:
            raise ValueError(f"InternVL-U bridge accepts one original prompt, got {len(prompt)}")
        prompt = prompt[0]
    if prompt is None:
        return {}
    if isinstance(prompt, dict):
        return prompt
    if isinstance(prompt, str):
        return {"prompt": prompt}
    if hasattr(prompt, "_asdict"):
        return dict(prompt._asdict())
    if hasattr(prompt, "__dict__"):
        return vars(prompt)
    raise TypeError(f"Unsupported InternVL-U original prompt type: {type(prompt).__name__}")


def _positive_int(value: Any) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _requested_size(prompt: Mapping[str, Any], sampling_params: Any) -> tuple[int, int]:
    mm_kwargs = prompt.get("mm_processor_kwargs")
    mm_kwargs = mm_kwargs if isinstance(mm_kwargs, Mapping) else {}
    resolution = _positive_int(getattr(sampling_params, "resolution", None))

    requested_height = (
        _positive_int(getattr(sampling_params, "height", None))
        or _positive_int(prompt.get("height"))
        or _positive_int(mm_kwargs.get("height"))
        or _positive_int(mm_kwargs.get("target_h"))
        or resolution
        or 1024
    )
    requested_width = (
        _positive_int(getattr(sampling_params, "width", None))
        or _positive_int(prompt.get("width"))
        or _positive_int(mm_kwargs.get("width"))
        or _positive_int(mm_kwargs.get("target_w"))
        or resolution
        or 1024
    )
    return requested_height, requested_width


def _align_down(value: int, stride: int = 16) -> int:
    return max(stride, value // stride * stride)


def ar2diffusion(
    source_outputs: list[Any],
    prompt: Any | None = None,
    requires_multimodal_data: bool = False,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    """Validate and package the three Stage-0 branches for diffusion."""

    # This custom bridge explicitly copies reference images into the Stage-1
    # prompt below.  Do not depend on generic diffusion-client metadata, which
    # is not propagated to custom diffusion input processors.
    del requires_multimodal_data
    role_outputs = _bind_roles(source_outputs)
    original_prompt = _prompt_mapping(prompt)
    reference_images = _extract_reference_images(original_prompt)
    reference_count = len(reference_images)

    expected_reference_counts = {
        CONDITIONAL_ROLE: reference_count,
        PARTIAL_ROLE: reference_count,
        UNCONDITIONAL_ROLE: 0,
    }
    branches = {role: _build_branch(role_outputs[role], role, expected_reference_counts[role]) for role in CFG_ROLES}

    # Text-then-image: the conditional branch generated CoT text before the
    # terminal <img>; direct image requests emit only the forced EOS.
    conditional_completion = _completion(role_outputs[CONDITIONAL_ROLE])
    conditional_output_ids = _token_list(
        getattr(conditional_completion, "cumulative_token_ids", None),
        name="cumulative_token_ids",
        role=CONDITIONAL_ROLE,
    )
    is_cot = len(conditional_output_ids) > 1
    cot_text = getattr(conditional_completion, "cumulative_text", None)

    requested_height, requested_width = _requested_size(original_prompt, sampling_params)
    if sampling_params is not None and getattr(sampling_params, "resolution", None) is None:
        # The shared output formatter requires a scalar resolution metric even
        # when callers specify rectangular height/width instead.
        sampling_params.resolution = max(requested_height, requested_width)
    height = _align_down(requested_height)
    width = _align_down(requested_width)

    internvlu_payload: dict[str, Any] = {
        "branches": branches,
        "image_grid_thw_gen": torch.tensor([[1, height // 8, width // 8]], dtype=torch.long),
        "requested_height": requested_height,
        "requested_width": requested_width,
        "is_cot": is_cot,
    }
    extra: dict[str, Any] = {"internvlu": internvlu_payload}
    if isinstance(cot_text, str) and cot_text.strip():
        # Shared serving convention: surfaced as the response's CoT text.
        extra["ar_generated_text"] = cot_text
    diffusion_prompt: dict[str, Any] = {
        "prompt": "",
        "height": height,
        "width": width,
        "extra": extra,
    }
    if reference_images:
        diffusion_prompt["multi_modal_data"] = {"image": reference_images}
    return diffusion_prompt


__all__ = [
    "CFG_ROLES",
    "CONDITIONAL_ROLE",
    "ExpandedPrompt",
    "PARTIAL_ROLE",
    "PARTIAL_SUFFIX",
    "UNCONDITIONAL_ROLE",
    "UNCONDITIONAL_SUFFIX",
    "ar2diffusion",
    "expand_cfg_prompts",
]
