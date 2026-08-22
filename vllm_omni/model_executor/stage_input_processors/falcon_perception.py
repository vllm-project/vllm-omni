# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception stage transition: thinker (stage 0) -> segmentation (stage 1).

Stage 0 runs with ``engine_output_type="latent"``, so its ``RequestOutput``
carries the whole last-layer hidden-state trajectory
(``multimodal_output["latent"]``, one row per prompt + generated token). The
orchestrator-side hook below slices that tensor into the two things the mask
head needs and hands them over in ``additional_information`` — no worker
connector involved.

From the trajectory we take:

* **image patch features** — prompt rows sitting at ``<|image|>`` positions,
  which fold back into the ``(h, w)`` patch grid;
* **segmentation features** — for each generated ``<|seg|>`` token, the row
  that *produced* it;
* **boxes** — the per-row ``(xy, hw)`` the thinker emits, sampled at the
  ``<|coord|>`` / ``<|size|>`` rows.

Deliberately, no segmentation weights run here: ``proj_segm`` / ``conv_segm`` /
``itok_upsampler`` all belong to stage 1, which applies them to these raw rows.
That keeps the handoff a plain tensor ship and avoids inventing any new
runner<->model protocol.

Row indexing, which is easy to get subtly wrong
-----------------------------------------------
``hidden[i]`` is the state that *produced* the token at position ``i + 1``.
The prompt occupies ``[0, len(prompt))`` and its last row produces
``output_token_ids[0]``. So the row behind ``output_token_ids[i]`` is::

    hidden[len(prompt) - 1 + i]

This mirrors the reference, where ``get_segm_tokens(h_last, next_token)`` is
called with the token that was just sampled *from* ``h_last``.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)

# Placeholder token for the stage-1 prompt. The segmentation stage is not
# autoregressive and never embeds these ids; the scheduler only needs a slot.
_PLACEHOLDER_TOKEN_ID = 0

# Defaults match the published tiiuae/Falcon-Perception config, used only when
# the caller cannot supply a config object.
_DEFAULT_IMG_ID = 227
_DEFAULT_SEG_ID = 262
_DEFAULT_COORD_ID = 240
_DEFAULT_SIZE_ID = 241


def _ship(tensor: torch.Tensor) -> torch.Tensor:
    """Make a tensor safe to serialise across the stage process boundary."""
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.flatten().tolist()]
    return [int(x) for x in value]


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _payload_get(payload: Any, *keys: str) -> Any:
    """Read a key from a payload that may be flat-dotted, nested, or a Mapping."""
    for key in keys:
        if payload is None:
            return None
        try:
            value = payload.get(key)
        except AttributeError:
            return None
        if value is not None:
            return value
    return None


def _extract_latent(source_output: Any) -> torch.Tensor | None:
    """The stage-0 hidden-state trajectory, ``(prompt + generated, hidden)``."""
    outputs = getattr(source_output, "outputs", None) or []
    for completion in outputs:
        mm = getattr(completion, "multimodal_output", None)
        if mm is None:
            continue
        latent = _payload_get(mm, "latent", "hidden", "hidden_states")
        if isinstance(latent, torch.Tensor) and latent.numel() > 0:
            return latent
    return None


def _extract_geometry(source_output: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    outputs = getattr(source_output, "outputs", None) or []
    for completion in outputs:
        mm = getattr(completion, "multimodal_output", None)
        if mm is None:
            continue
        xy = _payload_get(mm, "hidden_states.box_xy", "geometry.xy")
        hw = _payload_get(mm, "hidden_states.box_hw", "geometry.hw")
        if xy is None or hw is None:
            nested = _payload_get(mm, "hidden_states", "geometry")
            if isinstance(nested, dict):
                # Explicit None checks, not ``a or b``: truth-testing a tensor
                # raises "Boolean value of Tensor with more than one value is
                # ambiguous".
                if xy is None:
                    xy = _first_not_none(nested.get("box_xy"), nested.get("xy"))
                if hw is None:
                    hw = _first_not_none(nested.get("box_hw"), nested.get("hw"))
        if isinstance(xy, torch.Tensor) and isinstance(hw, torch.Tensor):
            return xy, hw
    return None, None


def _extract_selected_coords(source_output: Any) -> torch.Tensor | None:
    """Coordinates accepted by the request-scoped decode feedback loop."""
    outputs = getattr(source_output, "outputs", None) or []
    for completion in outputs:
        mm = getattr(completion, "multimodal_output", None)
        if mm is None:
            continue
        selected = _payload_get(mm, "hidden_states.selected_box_xy")
        if selected is None:
            nested = _payload_get(mm, "hidden_states")
            if isinstance(nested, dict):
                selected = nested.get("selected_box_xy")
        if isinstance(selected, torch.Tensor):
            return selected.reshape(-1, 2)
    return None


def _token_ids(source_output: Any) -> tuple[list[int], list[int]]:
    prompt_ids = _as_int_list(getattr(source_output, "prompt_token_ids", None))
    output_ids: list[int] = []
    outputs = getattr(source_output, "outputs", None) or []
    if outputs:
        completion = outputs[0]
        output_ids = _as_int_list(
            getattr(completion, "cumulative_token_ids", None) or getattr(completion, "token_ids", None)
        )
    return prompt_ids, output_ids


def build_segmentation_payload(
    hidden: torch.Tensor,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    *,
    xy: torch.Tensor | None = None,
    hw: torch.Tensor | None = None,
    selected_xy: torch.Tensor | None = None,
    img_id: int = _DEFAULT_IMG_ID,
    seg_id: int = _DEFAULT_SEG_ID,
    coord_id: int = _DEFAULT_COORD_ID,
    size_id: int = _DEFAULT_SIZE_ID,
) -> dict[str, Any] | None:
    """Slice the trajectory into the mask head's inputs. ``None`` if unusable."""
    n_prompt = len(prompt_token_ids)
    if hidden.ndim != 2 or n_prompt == 0 or hidden.shape[0] < n_prompt:
        logger.warning(
            "falcon_perception: unusable stage-0 trajectory (hidden=%s, prompt_len=%d)",
            tuple(hidden.shape),
            n_prompt,
        )
        return None

    img_positions = [i for i, t in enumerate(prompt_token_ids) if t == img_id]
    if not img_positions:
        logger.warning("falcon_perception: no <|image|> tokens in the stage-0 prompt")
        return None

    def rows_for(token_id: int, limit: int) -> list[int]:
        rows = [n_prompt - 1 + i for i, t in enumerate(output_token_ids) if t == token_id]
        return [r for r in rows if 0 <= r < limit]

    n_rows = hidden.shape[0]
    image_features = hidden[torch.tensor(img_positions, device=hidden.device)]
    seg_rows = rows_for(seg_id, n_rows)
    seg_features = (
        hidden[torch.tensor(seg_rows, device=hidden.device)] if seg_rows else hidden.new_zeros((0, hidden.shape[-1]))
    )

    # float32, not the model dtype: this payload is serialised to the stage-1
    # engine process, and bfloat16 has no numpy equivalent
    # ("Got unsupported ScalarType BFloat16"). Stage 1 casts back on arrival.
    payload: dict[str, Any] = {
        "hidden_states": {
            "image_features": _ship(image_features),
            "seg_features": _ship(seg_features),
        },
        "meta": {
            "finished": torch.tensor(True, dtype=torch.bool),
            "num_image_tokens": torch.tensor(len(img_positions), dtype=torch.long),
            "num_seg_tokens": torch.tensor(len(seg_rows), dtype=torch.long),
        },
    }

    # Boxes ride along so the client gets geometry and masks from one place.
    if isinstance(xy, torch.Tensor) and isinstance(hw, torch.Tensor):
        coord_rows = rows_for(coord_id, xy.shape[0])
        size_rows = rows_for(size_id, hw.shape[0])
        if coord_rows:
            # Same reason as the pixels below: only the schema's sections
            # survive transit, so the boxes ride inside ``hidden_states``.
            if isinstance(selected_xy, torch.Tensor) and selected_xy.numel() > 0:
                # Accepted coordinates are emitted one-per-<coord> token, in
                # generation order. They are the exact values re-encoded into
                # the next model step after duplicate rejection.
                selected_xy = selected_xy.reshape(-1, 2)
                payload["hidden_states"]["box_xy"] = _ship(selected_xy[: len(coord_rows)])
            else:
                payload["hidden_states"]["box_xy"] = _ship(xy[torch.tensor(coord_rows, device=xy.device)])
            payload["hidden_states"]["box_hw"] = (
                _ship(hw[torch.tensor(size_rows, device=hw.device)])
                if size_rows
                else torch.zeros((0, 2), dtype=torch.float32)
            )

    return payload


def _original_hw(multi_modal_data: Any) -> torch.Tensor | None:
    """Original (H, W) of the request's image, before any resizing.

    The reference thresholds mask logits at the original image resolution
    (``finalize_masks(original_image_size=...)``), not at the patch-aligned
    processed size, so the mask head needs the true size to target.
    """
    if not isinstance(multi_modal_data, dict):
        return None
    size = getattr(_resolve_image(multi_modal_data), "size", None)
    if not size:
        return None
    w, h = size
    return torch.tensor([int(h), int(w)], dtype=torch.float32)


def _resolve_image(multi_modal_data: Any) -> Any:
    """The request's single image, or ``None``."""
    image = None
    if isinstance(multi_modal_data, dict):
        image = multi_modal_data.get("image") or multi_modal_data.get("images")
    if isinstance(image, (list, tuple)):
        image = image[0] if image else None
    return image


# ------------------------------------------------------------- per-image memo
#
# The content hash and the processed pixels are both pure functions of the
# request's image, and both sit on the stage-0 -> stage-1 critical path, which
# is dead GPU time: the hook below runs synchronously on the orchestrator's
# asyncio loop, after stage 0's last decode step and before stage 1 has
# anything to do. Measured on a 1024x681 image, the hash costs ~4 ms and the
# processor ~53 ms per call. A workload that asks several queries about one
# image -- exactly the workload stage 1's AnyUp cache exists to serve -- paid
# both on every request for a result it had already computed.
#
# Keyed on object identity, not content: hashing the image to find out whether
# we already hashed it is circular. The entry holds a strong reference to the
# image, so ``id()`` cannot be recycled onto a different object while the entry
# is live. Two equal-but-distinct image objects therefore miss and recompute,
# which is merely slow, never wrong. A caller that mutates an image in place
# after submitting it would read a stale entry, but mutating a request's
# payload mid-flight is not a supported pattern.
_IMAGE_MEMO_MAX = 4


class _ImageMemo:
    """Lazily-filled cache of the two pure functions of one image."""

    __slots__ = ("image", "key", "pixels")

    def __init__(self, image: Any) -> None:
        self.image = image  # strong ref: pins id(image) for this entry's life
        self.key: torch.Tensor | None = None
        self.pixels: torch.Tensor | None = None


_IMAGE_MEMO: OrderedDict[tuple, _ImageMemo] = OrderedDict()


def _memo_for(image: Any) -> _ImageMemo | None:
    if image is None:
        return None
    ident = (id(image), getattr(image, "size", None), getattr(image, "mode", None))
    entry = _IMAGE_MEMO.get(ident)
    if entry is not None and entry.image is image:
        _IMAGE_MEMO.move_to_end(ident)
        return entry
    entry = _ImageMemo(image)
    _IMAGE_MEMO[ident] = entry
    while len(_IMAGE_MEMO) > _IMAGE_MEMO_MAX:
        _IMAGE_MEMO.popitem(last=False)
    return entry


def _image_key(multi_modal_data: Any) -> torch.Tensor | None:
    """Stable content identity for the image, so stage 1 can reuse its AnyUp output.

    AnyUp is by far the most expensive step of a request and depends only on
    the image: the image block sits at position 0 of the prompt and attention is
    causal (bidirectional only *within* the span), so image tokens never see the
    query. Measured over different text queries on the same image, both the
    thinker's image hidden states and the AnyUp output come back **bit-identical**,
    so reusing them is exactly equivalent to recomputing them.

    Hashing the source image (~2.5 MB of uint8) costs ~4 ms against the ~560 ms
    it lets stage 1 skip. The processed float tensor would be 4x larger to hash
    for no extra safety. Memoised per image, so repeat queries about the same
    image pay it once.
    """
    image = _resolve_image(multi_modal_data)
    if image is None or not hasattr(image, "tobytes"):
        return None

    memo = _memo_for(image)
    if memo is not None and memo.key is not None:
        return memo.key

    # Narrow, and non-fatal on purpose: the only consequence of no key is that
    # stage 1 recomputes AnyUp, which is correct, just slower.
    try:
        digest = hashlib.blake2b(digest_size=8)
        digest.update(f"{getattr(image, 'mode', '?')}:{getattr(image, 'size', '?')}|".encode())
        digest.update(image.tobytes())
    except (OSError, ValueError, TypeError):
        logger.warning("falcon_perception: could not hash the image; stage 1 will recompute AnyUp")
        return None
    # Positive int64: the payload crosses a process boundary as a tensor.
    key = torch.tensor([int.from_bytes(digest.digest(), "big") >> 1], dtype=torch.long)
    if memo is not None:
        memo.key = key
    return key


def _processed_pixels(multi_modal_data: Any) -> torch.Tensor | None:
    """Re-run the image pipeline to get the pixels AnyUp needs.

    The generation runner has no multimodal plumbing — ``requires_multimodal_data``
    forwards the request's data to *this* hook but never reaches the stage-1
    model's forward kwargs. So the pixels travel in the payload instead.

    Re-running the processor is a pure function of the image
    (resize -> smart_resize -> rescale -> normalize), so it reproduces
    stage 0's tensor exactly, including the patch grid the features align to.

    It is not cheap, though: ~53 ms for a 1024x681 image, all of it on the
    handoff's critical path with the GPU idle. Memoised per image so a run of
    queries about one image pays it once; see the memo above for why identity
    rather than content is the key.
    """
    image = _resolve_image(multi_modal_data)
    if image is None:
        logger.warning(
            "falcon_perception: no image found for the mask head (multi_modal_data=%s); stage 1 will emit no masks",
            sorted(multi_modal_data) if isinstance(multi_modal_data, dict) else type(multi_modal_data).__name__,
        )
        return None

    memo = _memo_for(image)
    if memo is not None and memo.pixels is not None:
        return memo.pixels

    from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
        FalconPerceptionConfig,
    )
    from vllm_omni.model_executor.models.falcon_perception.processing_falcon_perception import (
        FalconPerceptionImageProcessor,
    )

    # Deliberately not wrapped: AnyUp cannot run without these pixels, so a
    # processing failure has no meaningful fallback. Swallowing it here would
    # return a payload with no image and the mask head would report "nothing
    # detected" for a broken request.
    processor = FalconPerceptionImageProcessor(FalconPerceptionConfig())
    pixels = _ship(processor(image))
    if memo is not None:
        # Shared with every later request for this image. ``_ship`` already
        # detached it and the payload path only reads it, so handing out the
        # same tensor is safe; stage 1 gets its own copy across the process
        # boundary regardless.
        memo.pixels = pixels
    return pixels


def thinker2segmentation_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build one stage-1 input per finished request.

    Carries the sliced hidden states, the boxes and the processed pixels in
    ``additional_information``; stage 1 reads everything from there.
    """
    multi_modal_data = None
    if isinstance(prompt, dict):
        multi_modal_data = prompt.get("multi_modal_data") or prompt.get("multi_modal_inputs")
    elif prompt is not None:
        multi_modal_data = getattr(prompt, "multi_modal_data", None)
    if multi_modal_data is None:
        logger.warning(
            "falcon_perception: stage-1 hook got no multimodal data (prompt type=%s, keys=%s)",
            type(prompt).__name__,
            sorted(prompt) if isinstance(prompt, dict) else "n/a",
        )
    pixel_values = _processed_pixels(multi_modal_data)
    original_hw = _original_hw(multi_modal_data)
    image_key = _image_key(multi_modal_data)

    inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        if not getattr(source_output, "finished", False):
            continue

        latent = _extract_latent(source_output)
        prompt_ids, output_ids = _token_ids(source_output)
        payload: dict[str, Any] | None = None
        if latent is None:
            logger.warning(
                "falcon_perception: stage-0 output carries no latent hidden states; "
                "stage 1 will emit no masks. Is engine_output_type='latent' set on stage 0?"
            )
        else:
            xy, hw = _extract_geometry(source_output)
            selected_xy = _extract_selected_coords(source_output)
            payload = build_segmentation_payload(
                latent,
                prompt_ids,
                output_ids,
                xy=xy,
                hw=hw,
                selected_xy=selected_xy,
            )

        if payload is not None and pixel_values is not None:
            # Inside ``hidden_states`` on purpose: the stage payload only
            # transports a fixed set of sections (see data_entry_keys.py —
            # hidden_states / embed / codes / ids / meta). A custom top-level
            # "image" section is silently dropped in transit.
            payload["hidden_states"]["pixel_values"] = pixel_values
        if payload is not None and original_hw is not None:
            payload["hidden_states"]["original_hw"] = original_hw
        if payload is not None and image_key is not None:
            # ``meta`` rather than ``hidden_states``: this is an identifier, not
            # a feature, and meta already carries non-float tensors safely.
            payload["meta"]["image_key"] = image_key

        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[_PLACEHOLDER_TOKEN_ID],
                multi_modal_data=multi_modal_data,
                additional_information=payload or {},
            )
        )
    return inputs
