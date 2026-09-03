# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Canonical, shared helpers for ``stage_input_processors``.

Related to RFC #4872 (https://github.com/vllm-project/vllm-omni/issues/4872):
these are consolidated implementations of helpers that were previously
duplicated across model modules with subtle behavioral differences (see
``tests/model_executor/stage_input_processors/test_common_helpers_golden.py``,
which locks the observed per-module behaviour).

**Consolidation rule:** where legacy variants disagreed, the default
implementation follows the most-complete semantics; divergent legacy behaviour
is preserved through explicit named variants (e.g. ``ensure_list_unchanged``)
or parameters (e.g. ``filter_real_code_frames(..., layout=...)``).  A module
may switch to ``_common`` only after its legacy behaviour is golden-locked and
the matching variant is used.
"""

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
    """Canonical ``ensure_list``.

    Semantics (the union of legacy variants):
    - ``ConstantList``-like objects (``_x``) are unwrapped.
    - ``list`` / ``tuple`` are converted to a fresh ``list``.
    - ``torch.Tensor`` is converted via ``.tolist()`` (all dims flattened).
    - ``None`` becomes ``[]``.
    - Any other iterable is converted via ``list(x)``; a non-iterable scalar
      becomes ``[x]``.
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


def ensure_list_or_empty(x: Any) -> list[Any]:
    """Explicit alias: always returns a list; ``None`` -> ``[]``.

    Matches ``audex`` / ``cosyvoice3`` ``_ensure_list`` semantics.
    """
    return ensure_list(x)


def ensure_list_unchanged(x: Any) -> Any:
    """Legacy ``qwen3_omni._ensure_list`` semantics.

    ``ConstantList`` is unwrapped; a non-``list`` value (including ``None``,
    a ``tuple``, or a ``torch.Tensor``) is returned **unchanged**.
    """
    x = _unwrap_constant_list(x)
    if not isinstance(x, list):
        return x
    return list(x)


def ensure_list_strict(x: Any) -> list[Any]:
    """Strict ``ensure_list``: no ``None`` special-case, non-iterables raise.

    Matches ``ming_flash_omni._ensure_list``: ``None`` and non-iterable
    scalars both fall through to ``list(x)`` and raise ``TypeError``; a
    ``torch.Tensor`` is converted via ``.tolist()``.
    """
    x = _unwrap_constant_list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x)


def ensure_list_flatten(x: Any) -> list[Any]:
    """``ensure_list`` that **flattens** tensors with ``reshape(-1).tolist()``.

    Matches ``audex._ensure_list``: ``None`` -> ``[]``; a ``torch.Tensor`` is
    flattened to a 1-D list; a non-iterable scalar raises ``TypeError``.
    """
    x = _unwrap_constant_list(x)
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x)


def ensure_list_preserve_none(x: Any) -> list[Any]:
    """``ensure_list`` that **preserves** ``None`` as ``[None]``.

    Matches ``step_audio2._ensure_list``:
    - ``ConstantList``-like objects (``_x``) are unwrapped.
    - ``torch.Tensor`` is converted via ``.tolist()`` (dimensions preserved).
    - ``list`` / ``tuple`` are converted to a fresh ``list``.
    - Any other **iterable** is converted via ``list(x)``.
    - A non-iterable value (including ``None`` and a scalar like ``5``) is
      wrapped as ``[x]`` (so ``None`` -> ``[None]``, ``5`` -> ``[5]``).
    """
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x) if hasattr(x, "__iter__") else [x]


def ensure_list_wrap_only(x: Any) -> list[Any]:
    """Wrap-only ``ensure_list``: list passthrough, non-list -> ``[x]``, ``None`` -> ``[]``.

    Preserves the legacy ``diffusion.output_formatter._ensure_list`` semantics:
    a non-list primary payload is wrapped as ``[x]``.  Unlike the canonical
    :func:`ensure_list`, it performs **no** tensor flattening, no dict-key
    iteration, and no row-wise iteration of an iterable (e.g. a PIL ``Image``) —
    a non-``list`` value is wrapped whole, verbatim.
    """
    if isinstance(x, list):
        return x
    return [x] if x is not None else []


# ===========================================================================
# to_cpu_tensor
# ===========================================================================


def to_cpu_tensor(value: Any) -> torch.Tensor | None:
    """Convert a value to a CPU tensor when possible; else ``None``.

    Matches ``glm_tts._to_cpu_tensor``:
    - a list is unwrapped to its first element first;
    - empty list -> ``None``;
    - only ``torch.Tensor`` inputs yield a result (``detach().cpu()``).
    """
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
    """Convert a token-ish value into a flat ``list[int]``.

    ``recursive=False`` (default) matches ``dynin_omni._to_token_id_list``:
    for a 2-D tensor or a nested list only the **first** row / element is
    used.  ``recursive=True`` matches ``cosyvoice3._to_token_id_list``:
    per-item recursive normalization — nested ``list``/``tuple`` and
    non-scalar tensors anywhere in the tree are flattened, so
    ``[torch.tensor([[4, 5]])]`` -> ``[4, 5]`` and ``(4, 5)`` -> ``[4, 5]``.
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
    """cosyvoice3 recursive token-id flattening (per-item normalization).

    Mirrors the pre-consolidation ``cosyvoice3._to_token_id_list``: ``None`` ->
    ``[]``; a tensor is reshaped flat; every item that is a tensor or a
    ``list``/``tuple`` is recursed into, scalars are converted with ``int()``.
    """
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
    """Reverse the MusicGen-style delay pattern.

    Input ``[Q, T]`` (``T = seq_len + Q - 1``) -> output ``[Q, seq_len]``.

    ``allow_short=True`` matches ``higgs_audio_v2``: when ``T < Q`` the input
    is returned **unchanged**.  ``allow_short=False`` (default, matches
    ``higgs_audio_v3``) raises on ``T < Q``.  When ``expected_codebooks`` is
    given, a row-count mismatch raises ``ValueError`` (v3 checks ``Q ==
    _NUM_CODEBOOKS``; v2 does not).
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
    """Keep only frames whose codebook values are all in ``[0, num_real_codes)``.

    ``layout="frames_first"`` matches ``higgs_audio_v2``: input is
    ``[num_frames, num_codebooks]`` and the same layout is returned.
    ``layout="codebooks_first"`` matches ``higgs_audio_v3``: input is
    ``[num_codebooks, num_frames]``; frames are filtered and the layout is
    restored (``[num_codebooks, kept_frames]``).
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
    """Extract the last inter-stage codec frame from a payload.

    Parameterized canonical implementation covering the four legacy
    ``_extract_last_frame`` variants:

    - ``key_path``: ``("codes", "audio")`` (higgs_v2 / qwen3_tts / voxtral)
      vs ``("audio_codes",)`` (fish_speech top-level).
    - ``validate``:
      - ``None``: only empty checks (higgs_v2 / voxtral);
      - ``"any"``: drop a frame whose values are all zero (qwen3_tts);
      - ``"valid_mask"``: consult ``audio_code_valid`` then fall back to
        ``any`` (fish_speech).
    - ``to_cpu`` / ``to_long``: fish_speech moves to CPU and casts to long;
      higgs_v2 / qwen3_tts cast to long without moving to CPU; voxtral does
      neither (plain ``flatten``).

    The returned tensor is always 1-D.
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
    if audio_codes.ndim == 2:
        frame: torch.Tensor = audio_codes[-1]
    elif audio_codes.ndim == 1:
        frame = audio_codes
    else:
        raise ValueError(f"unexpected audio_codes shape: {tuple(audio_codes.shape)}")
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

    Shared by ``mode="full"`` (prompt + generated ids) and ``mode="stage0_only"``
    (stage-0 ids only, fed to both roles) so the sync forward placeholder and
    the async-chunk prewarm estimate return the same length.
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

    ``mode="full"`` replicates the legacy Qwen chat-template scan: the input is
    an ``OmniPayload``-like dict with ``ids: {"all": [...], "prompt": [...]}``,
    and the length is ``sum(user segments) + 9`` (the assistant tail).

    ``mode="stage0_only"`` is the async-chunk **prewarm** estimate.  There is no
    upstream output yet, so ``ids_or_prompt`` is the stage-0 input token-id list,
    which is fed to **both** chat-template roles (``all`` / ``prompt``) — exactly
    what ``adapter.compute_talker_prompt_ids_length`` does.  This makes the
    prewarm builder and the inline fallback return the same number (15 for the
    golden prompt; 6 == 6 for a single user segment), removing the
    ``len()`` vs scan split.
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
    """Pack a KV-slot placeholder prompt: ``[0] * prompt_len`` + voice meta.

    Shared by both the sync forward path (``*_token_only``) and the
    async-chunk prewarm path.  Bulk conditioning arrives separately via the
    connector, so only the length matters here.
    """
    return OmniTokensPrompt(
        prompt_token_ids=[0] * max(1, int(prompt_len)),
        additional_information=(voice_metadata if voice_metadata else None),  # type: ignore[typeddict-item]
    )
