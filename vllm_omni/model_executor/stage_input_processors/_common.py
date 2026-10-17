# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Canonical shared helpers; semantics golden-locked in ``test_common_helpers_golden.py``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from vllm_omni.inputs.data import OmniTokensPrompt

# ===========================================================================
# ensure_list
# ===========================================================================


def _unwrap_constant_list(x: Any) -> Any:
    """Unwrap a ``ConstantList``-like object exposing ``_x``."""
    if hasattr(x, "_x"):
        return x._x
    return x


def ensure_list(x: Any) -> list[Any]:
    """Canonical ``ensure_list`` (union of legacy variants).

    ``ConstantList`` unwrapped; tensor ``.tolist()``; ``None`` -> ``[]``;
    other iterable via ``list(x)``; non-iterable scalar wrapped as ``[x]``.
    """
    x = _unwrap_constant_list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return [x]


def ensure_list_unchanged(x: Any) -> Any:
    """Legacy ``qwen3_omni._ensure_list``: non-``list`` (incl. ``None``) unchanged."""
    x = _unwrap_constant_list(x)
    if not isinstance(x, list):
        return x
    return list(x)


def ensure_list_strict(x: Any) -> list[Any]:
    """Strict variant (``ming_flash_omni``): no ``None`` case; non-iterable raises."""
    x = _unwrap_constant_list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x)


def ensure_list_flatten(x: Any) -> list[Any]:
    """Flattening variant (``audex``): tensors via ``reshape(-1).tolist()``."""
    x = _unwrap_constant_list(x)
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x)


def ensure_list_preserve_none(x: Any) -> list[Any]:
    """Preserve-``None`` variant (``step_audio2``): non-iterable wrapped as ``[x]``."""
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x) if hasattr(x, "__iter__") else [x]


def ensure_list_wrap_only(x: Any) -> list[Any]:
    """Wrap-only variant (``diffusion.output_formatter``): non-list wrapped whole.

    Unlike :func:`ensure_list`: no tensor flattening, no iterable iteration.
    """
    if isinstance(x, list):
        return x
    return [x] if x is not None else []


# ===========================================================================
# to_cpu_tensor
# ===========================================================================


def to_cpu_tensor(value: Any) -> torch.Tensor | None:
    """CPU tensor when possible; else ``None`` (``glm_tts``)."""
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return None


# ===========================================================================
# to_token_id_list
# ===========================================================================


def to_token_id_list(value: Any, *, recursive: bool = False) -> list[int]:
    """Flatten a token-ish value to ``list[int]``.

    ``recursive=False`` matches ``dynin_omni`` (first row only);
    ``recursive=True`` matches ``cosyvoice3`` (per-item flatten).
    """
    if recursive:
        return _to_token_id_list_recursive(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu")
        if value.ndim == 0:
            return [int(value.item())]
        if value.ndim > 1:
            value = value[0]
        return [int(x) for x in value.tolist()]
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], list):
            return [int(x) for x in value[0]]
        return [int(x) for x in value]
    if value is None:
        return []
    return [int(value)]


def _to_token_id_list_recursive(value: Any) -> list[int]:
    """Recursive token-id flattening: tensors reshaped, scalars ``int()``."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu").reshape(-1).tolist()
    out: list[int] = []
    for item in ensure_list(value):
        if isinstance(item, torch.Tensor):
            out.extend(_to_token_id_list_recursive(item))
            continue
        if isinstance(item, (list, tuple)):
            out.extend(_to_token_id_list_recursive(item))
            continue
        out.append(int(item))
    return out


# ===========================================================================
# revert_delay_pattern  (higgs_audio_v2 lenient vs higgs_audio_v3 strict)
# ===========================================================================


def revert_delay_pattern(
    audio_codes_qt: torch.Tensor,
    *,
    expected_codebooks: int | None = None,
    allow_short: bool = False,
) -> torch.Tensor:
    """Reverse the MusicGen-style delay pattern ``[Q, T]`` -> ``[Q, seq_len]``.

    ``allow_short`` (higgs v2) vs strict raise (v3); ``expected_codebooks``
    row-count check (v3 only).
    """
    if audio_codes_qt.ndim != 2:
        raise ValueError(f"_revert_delay_pattern expects [Q, T] input; got {tuple(audio_codes_qt.shape)}")
    q, t = audio_codes_qt.shape
    if expected_codebooks is not None and q != expected_codebooks:
        raise ValueError(f"Expected exactly {expected_codebooks} codebook rows, got {q}. Input shape: [{q}, {t}]")
    if t < q:
        if allow_short:
            return audio_codes_qt
        raise ValueError(f"Not enough frames to revert delay pattern: T={t} < Q={q}")
    seq_len = t - q + 1
    out_l = [audio_codes_qt[i : i + 1, i : seq_len + i] for i in range(q)]
    return torch.cat(out_l, dim=0)


# ===========================================================================
# filter_real_code_frames  (higgs v2 [frames,Q] vs v3 [Q,frames])
# ===========================================================================


def filter_real_code_frames(
    audio_codes: torch.Tensor,
    *,
    num_real_codes: int,
    layout: str = "frames_first",
) -> torch.Tensor:
    """Keep frames with all codebook values in ``[0, num_real_codes)``.

    ``frames_first`` (v2) vs ``codebooks_first`` (v3, transposed) layouts.
    """
    if audio_codes.numel() == 0:
        return audio_codes
    if layout == "frames_first":
        if audio_codes.ndim != 2:
            raise ValueError(f"expected [num_frames, num_codebooks] audio_codes; got shape {tuple(audio_codes.shape)}")
        valid = (audio_codes >= 0).all(dim=1) & (audio_codes < num_real_codes).all(dim=1)
        return audio_codes[valid]
    if layout == "codebooks_first":
        frames = audio_codes.t()
        valid = (frames >= 0).all(dim=1) & (frames < num_real_codes).all(dim=1)
        return frames[valid].t().contiguous()
    raise ValueError(f"unknown layout: {layout!r}")


# ===========================================================================
# extract_last_codec_frame
# ===========================================================================


def extract_last_codec_frame(
    payload: Any,
    *,
    key_path: tuple[str, ...] = ("codes", "audio"),
    validate: str | None = None,
    to_cpu: bool = False,
    to_long: bool = True,
) -> torch.Tensor | None:
    """Extract the last inter-stage codec frame (4 legacy variants merged).

    ``key_path`` codes/audio vs audio_codes; ``validate`` None/"any"/"valid_mask";
    ``to_cpu``/``to_long`` (fish_speech).  Validity gate applies only to 2-D
    frames; 1-D frames return directly; result is always 1-D.  3-D per-step
    frames ``[B, 1, C]`` (Voxtral ``compute_mm_logits``) are normalized to
    ``[B, C]`` first; any other 3-D layout still raises ``ValueError``.
    """
    if not isinstance(payload, dict):
        return None
    audio_codes: Any = payload
    for key in key_path:
        audio_codes = audio_codes.get(key) if isinstance(audio_codes, dict) else None
        if audio_codes is None:
            return None
    if isinstance(audio_codes, list):
        if not audio_codes:
            return None
        audio_codes = audio_codes[-1]
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 3 and audio_codes.shape[1] == 1:
        # Voxtral TTS: compute_mm_logits() emits per-step [B, 1, C] codes
        # (payload list elements of shape (1, 1, C)); the legacy voxtral
        # helper flattened them, so 3-D per-step input is a supported
        # layout, not a contract violation.  Targeted squeeze(1) reduces
        # it to [B, C] so the shared 2-D last-row semantics (including
        # any validity gate) apply unchanged.  Non-singleton frame dims
        # (e.g. (2, 2, 2)) intentionally fall through to the ValueError
        # below, preserving the golden-locked fish/qwen3/higgs contracts.
        audio_codes = audio_codes.squeeze(1)
    if audio_codes.ndim == 2:
        frame: torch.Tensor = audio_codes[-1]
        if frame.numel() == 0:
            return None
        if validate in ("any", "valid_mask"):
            if validate == "valid_mask":
                valid = payload.get("audio_code_valid")
                if isinstance(valid, torch.Tensor) and valid.numel() > 0:
                    is_valid = bool(valid.reshape(-1)[-1].item())
                elif valid is not None:
                    is_valid = bool(valid)
                else:
                    is_valid = bool(frame.any().item())
            else:
                is_valid = bool(frame.any().item())
            if not is_valid:
                return None
    elif audio_codes.ndim == 1:
        # Legacy semantics: a flat (1-D) codec frame is returned directly
        # without the validity/zero gate (Fish Speech / Qwen3-TTS).
        frame = audio_codes
    else:
        raise ValueError(f"unexpected audio_codes shape: {tuple(audio_codes.shape)}")
    frame = frame.flatten()
    if to_cpu:
        frame = frame.detach().cpu()
    if to_long:
        frame = frame.to(torch.long)
    return frame


# ===========================================================================
# Placeholder prompt length + packing
# ===========================================================================

# Qwen chat-template sentinel ids (single source within this module; the
# codec token ids themselves are centralized separately in ``_constants``).
_IM_START_TOKEN_ID = 151644
_SYSTEM_TOKEN_ID = 8948
_USER_TOKEN_ID = 872
_ASSISTANT_TOKEN_ID = 77091
_ASSISTANT_TAIL_LEN = 9  # 3 + 4 + 1 + 1


def _chat_template_prompt_len(
    all_ids: Sequence[int],
    prompt_ids: Sequence[int],
    device: torch.device | str,
) -> int:
    """Qwen chat-template scan: ``sum(user segments) + 9`` (assistant tail).

    Shared by both modes so sync forward and async prewarm agree.
    """
    thinker_sequences = torch.tensor(all_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == _IM_START_TOKEN_ID).squeeze(1),
            torch.tensor(
                [thinker_sequences.shape[-1]],
                device=input_ids.device,
                dtype=input_ids.dtype,
            ),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == _SYSTEM_TOKEN_ID:
            continue
        if role == _USER_TOKEN_ID:
            sum_user_len += e - s
        elif role == _ASSISTANT_TOKEN_ID and i == len(im_start_indexes) - 2:
            assistant_len += _ASSISTANT_TAIL_LEN
    return sum_user_len + assistant_len


def compute_placeholder_prompt_len(
    *,
    ids_or_prompt: Any,
    mode: str = "full",
    device: torch.device | str = "cpu",
) -> int:
    """Compute the downstream placeholder prompt length.

    ``mode="full"``: ``ids.all``/``ids.prompt`` scan (``sum(user) + 9``).
    ``mode="stage0_only"``: prewarm estimate feeding stage-0 ids to both roles,
    matching ``adapter.compute_talker_prompt_ids_length`` so builder and inline
    fallback agree.
    """
    if mode == "stage0_only":
        prompt = list(ids_or_prompt)
        return _chat_template_prompt_len(prompt, prompt, device)
    if mode != "full":
        raise ValueError(f"unknown mode: {mode!r}")

    ids = (ids_or_prompt or {}).get("ids", {})
    return _chat_template_prompt_len(ids["all"], ids["prompt"], device)


def pack_placeholder_prompt(
    *,
    prompt_len: int,
    voice_metadata: dict[str, Any] | None = None,
) -> OmniTokensPrompt:
    """Pack a KV-slot placeholder prompt: ``[0] * prompt_len`` + voice meta."""
    return OmniTokensPrompt(
        prompt_token_ids=[0] * max(1, int(prompt_len)),
        additional_information=(voice_metadata if voice_metadata else None),  # type: ignore[typeddict-item]
    )
