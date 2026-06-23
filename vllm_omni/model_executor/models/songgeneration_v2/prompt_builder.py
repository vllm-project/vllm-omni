# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prompt builder for SongGeneration v2 LeLM Stage 0.

Converts raw user inputs (lyric, descriptions, prompt_audio_codes) into
CFG-paired condition tensors. Single entry point: build_condition_tensors().
"""

from __future__ import annotations

import torch

from .conditioners import (
    AudioCondition,
    ClassifierFreeGuidanceDropoutInference,
    ConditionerProvider,
    ConditioningAttributes,
)

# Default token IDs for SongGeneration v2-large
# (upstream conf/vocab.yaml). The real values used at
# runtime are derived from the prompt_audio conditioner's ``vocab_size``
# below, so unit tests can use smaller dims; these constants are only the
# fallback for callers that pass no provider.
_DEFAULT_CODE_SIZE = 16384
_DEFAULT_EOS_TOKEN_ID = _DEFAULT_CODE_SIZE  # 16384
_DEFAULT_SPECIAL_TOKEN_ID = _DEFAULT_CODE_SIZE + 1  # 16385


def _resolve_token_ids(provider: ConditionerProvider) -> tuple[int, int]:
    """Read ``(eos_id, special_id)`` from the prompt_audio conditioner.

    Falls back to the v2-large defaults if no prompt_audio conditioner is
    registered (e.g. text-only conditioning).
    """
    cond = getattr(provider, "conditioners", None)
    if cond is not None and "prompt_audio" in cond:
        code_size = int(getattr(cond["prompt_audio"], "vocab_size", _DEFAULT_CODE_SIZE))
        return code_size, code_size + 1
    return _DEFAULT_EOS_TOKEN_ID, _DEFAULT_SPECIAL_TOKEN_ID


def _preprocess_description(
    descriptions: str | None,
    *,
    gen_type: str = "mixed",
    version: str = "v2",
) -> str | None:
    """Match upstream ``generate.py:319-326`` description preprocessing."""
    if version == "v1":
        return descriptions.lower() if descriptions else None

    if gen_type == "bgm":
        if descriptions:
            return "[Musicality-very-high], [Pure-Music], " + descriptions.lower()
        return "."
    # mixed / vocal
    base = descriptions.lower() if descriptions else "."
    return "[Musicality-very-high], " + base


def _preprocess_lyric(lyric: str | None, *, gen_type: str = "mixed") -> str | None:
    """Match upstream ``generate.py:260``: force ``lyrics='.'`` for bgm so the
    AR stage generates a pure-music sequence instead of a masked-out melody."""
    if gen_type == "bgm":
        return "."
    return lyric


def _mask_prompt_tokens(
    audio_qt_emb: torch.Tensor,
    *,
    eos_id: int,
    special_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend an EOS row and re-apply the absent-stream mask.

    Mirrors ``LmModel.prepare_condition_tensors`` lyrically: if the first
    frame of any stream is ``special_id`` (the "no prompt audio" sentinel),
    the whole sequence for that stream gets re-stamped to ``special_id``
    after the EOS prepend, so the conditioner sees a clean null prompt.

    Args:
        audio_qt_emb: ``[B, code_depth, T_prompt]`` LongTensor.
        eos_id: token id prepended at position 0 (= ``code_size``).
        special_id: token id used for masked-out streams (= ``code_size + 1``).

    Returns:
        ``(audio_qt_seq, length)`` -- ``audio_qt_seq`` shape
        ``[B, code_depth, T_prompt + 1]``; ``length`` is the per-sample
        ``[B]`` sequence length used by the conditioner mask.
    """
    if audio_qt_emb.dim() != 3:
        raise ValueError(f"audio_qt_emb must be [B, code_depth, T_prompt]; got {tuple(audio_qt_emb.shape)}")

    audio_qt_emb = audio_qt_emb.long()
    B, K, T = audio_qt_emb.shape

    # Mask = streams whose first frame is the "no prompt" sentinel.
    mask = (audio_qt_emb[:, :, 0] == special_id).bool()

    # Prepend an EOS frame on every sample so the conditioner has a fixed
    # leading row to anchor on.
    eos_frame = torch.full((B, K, 1), eos_id, dtype=audio_qt_emb.dtype, device=audio_qt_emb.device)
    audio_qt_seq = torch.cat([eos_frame, audio_qt_emb], dim=-1)

    # Re-stamp absent streams with the special id so the conditioner's
    # padding/eos rows aren't read as real codes.
    expanded = mask.unsqueeze(-1).expand(B, K, audio_qt_seq.shape[-1])
    audio_qt_seq = torch.where(expanded, torch.full_like(audio_qt_seq, special_id), audio_qt_seq)

    length = torch.full((B,), audio_qt_seq.shape[-1], dtype=torch.long)
    return audio_qt_seq, length


def build_condition_tensors(
    provider: ConditionerProvider,
    *,
    lyrics: list[str | None],
    descriptions: list[str | None] | None = None,
    prompt_audio_codes: torch.Tensor | None = None,
    sample_rate: int = 48000,
    gen_type: str = "mixed",
    version: str = "v2",
    cfg_pair: bool = True,
    device: torch.device | str | None = None,
) -> tuple[dict, int]:
    """Build a CFG-paired ``condition_tensors`` dict from raw user inputs.

    Args:
        provider: The Stage-0 module's ConditionerProvider (carries the
            embedding weights for description / type_info / prompt_audio).
        lyrics: Per-sample lyric strings (or ``None``). One entry per real
            sample; the null half of the CFG batch is added by this function.
        descriptions: Per-sample style strings. Same length as ``lyrics``
            or ``None`` (treated as missing for all samples).
        prompt_audio_codes: Optional ``[B, code_depth=3, T_prompt]`` LongTensor
            of prompt-audio codec indices. Streams not in use should be filled
            with ``16385``. ``None`` is equivalent to all streams being all-16385.
        sample_rate: Forwarded to the ``AudioCondition`` metadata (the
            conditioner doesn't use it but the upstream signature includes it).
        gen_type: ``"mixed" | "vocal" | "bgm"``; controls the v2 description
            preprocessing (``[Pure-Music]`` tag for ``bgm``).
        version: ``"v2"`` (default) applies the Musicality-tag prepend.
        cfg_pair: If True (default), the returned condition_tensors has batch
            size ``2 * B`` -- first half is the real conditioning, second half
            is the null/CFG-dropped version. Stage-0 ``forward`` will split
            this back along the batch dim.
        device: Where to place the output tensors. Defaults to the device of
            the provider's first parameter.

    Returns:
        ``(condition_tensors, B)`` where ``condition_tensors`` is the dict
        ``{name: (e1, e2, mask)}`` and ``B`` is the number of *real* samples
        (so the batch dim of each entry is ``B`` if ``cfg_pair=False``,
        ``2*B`` otherwise).
    """
    if device is None:
        try:
            device = next(provider.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    eos_id, special_id = _resolve_token_ids(provider)

    B = len(lyrics)
    if descriptions is None:
        descriptions = [None] * B
    elif len(descriptions) != B:
        raise ValueError(f"descriptions length ({len(descriptions)}) must match lyrics length ({B})")

    if prompt_audio_codes is not None:
        if prompt_audio_codes.dim() != 3:
            raise ValueError(f"prompt_audio_codes must be [B, code_depth, T]; got {tuple(prompt_audio_codes.shape)}")
        if prompt_audio_codes.shape[0] != B:
            raise ValueError(f"prompt_audio_codes batch ({prompt_audio_codes.shape[0]}) != lyrics batch ({B})")

    # ---- Build per-sample ConditioningAttributes ---------------------
    available = set(provider.conditioners.keys())
    samples: list[ConditioningAttributes] = []
    for i in range(B):
        attr = ConditioningAttributes()
        if "description" in available:
            attr.text["description"] = _preprocess_lyric(lyrics[i], gen_type=gen_type)
        if "type_info" in available:
            attr.text["type_info"] = _preprocess_description(descriptions[i], gen_type=gen_type, version=version)
        if "prompt_audio" in available:
            # Per-sample prompt tokens. ``prompt_audio_codes`` is either
            # explicit (shape ``[B,3,T]``) or ``None`` -> single-frame
            # ``[special, special, special]`` placeholder (no-prompt path).
            if prompt_audio_codes is None:
                wav_i = torch.full((1, 3, 1), special_id, dtype=torch.long)
            else:
                wav_i = prompt_audio_codes[i : i + 1].long()
            attr.audio["prompt_audio"] = AudioCondition(
                wav=wav_i.to(device),
                length=torch.tensor([wav_i.shape[-1] + 1], dtype=torch.long),
                sample_rate=[sample_rate],
                path=[None],
                seek_time=[None],
            )
        samples.append(attr)

    # ---- Apply the prompt-token EOS-prepend + mask re-stamp ----------
    if "prompt_audio" in available and prompt_audio_codes is not None:
        for i, s in enumerate(samples):
            ac = s.audio["prompt_audio"]
            qt_seq, length = _mask_prompt_tokens(ac.wav, eos_id=eos_id, special_id=special_id)
            s.audio["prompt_audio"] = AudioCondition(
                wav=qt_seq,
                length=length.to(device),
                sample_rate=ac.sample_rate,
                path=ac.path,
                seek_time=ac.seek_time,
            )

    # ---- CFG pair: append null-conditioned samples -------------------
    if cfg_pair:
        cfg = ClassifierFreeGuidanceDropoutInference()
        null_samples = cfg(samples, condition_types=["audio", "text"], code_size=eos_id)
        samples = samples + null_samples

    # ---- Tokenize + forward through provider -------------------------
    tokenized = provider.tokenize(samples)
    condition_tensors = provider(tokenized)
    return condition_tensors, B


__all__ = ["build_condition_tensors"]
