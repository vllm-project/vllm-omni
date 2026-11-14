# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame-v5 sequence geometry and causal visibility shared by streaming paths."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

KINDS = frozenset({"reference", "shared", "clean", "text", "noisy", "padding"})
PERSISTENT = frozenset({"reference", "shared", "clean"})


@dataclass(frozen=True)
class RealizationGroup:
    sample: int
    block: int
    kind: str
    start: int
    end: int

    def __post_init__(self):
        if self.kind not in KINDS or self.start < 0 or self.end <= self.start:
            raise ValueError("Invalid Realization semantic group")


def validate_groups(groups: tuple[RealizationGroup, ...], length: int) -> None:
    cursor = 0
    for group in groups:
        if group.start != cursor:
            raise ValueError("Realization semantic groups must partition rows without holes/overlap")
        cursor = group.end
    if cursor != length:
        raise ValueError("Realization semantic group length mismatch")


def visible(q: RealizationGroup, k: RealizationGroup, *, refiner: bool = False) -> bool:
    """Torchtitan frame-v5 causal visibility; never a token triangular mask."""
    if q.sample != k.sample:
        return False
    if q.kind == "padding" or k.kind == "padding":
        return q.kind == k.kind == "padding"
    if refiner:
        return k.kind == "shared" or (q.kind == k.kind == "text" and q.block == k.block)
    if q.kind in ("reference", "shared"):
        return k.kind == q.kind
    if k.kind in ("reference", "shared"):
        return True
    if k.kind == "text":
        return k.block == q.block
    if k.kind == "noisy":
        return q.kind == "noisy" and k.block == q.block
    if k.kind == "clean":
        return k.block < q.block or (q.kind == "clean" and k.block == q.block)
    return False


def dense_mask(query: tuple[RealizationGroup, ...], keys: tuple[RealizationGroup, ...], device, *, refiner=False):
    qlen = query[-1].end if query else 0
    klen = keys[-1].end if keys else 0
    validate_groups(query, qlen)
    validate_groups(keys, klen)
    mask = torch.zeros(qlen, klen, dtype=torch.bool, device=device)
    for q in query:
        allowed = [k for k in keys if visible(q, k, refiner=refiner)]
        if not allowed:
            raise ValueError(f"All-masked Realization query: {q}")
        for k in allowed:
            mask[q.start : q.end, k.start : k.end] = True
    return mask


@dataclass(frozen=True)
class RealizationSequence:
    video: torch.Tensor
    audio: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    positions: torch.Tensor
    tags: torch.Tensor
    roles: torch.Tensor
    time_ids: torch.Tensor
    groups: tuple[RealizationGroup, ...]
    refiner_text: torch.Tensor
    refiner_groups: tuple[RealizationGroup, ...]
    text_selection: torch.Tensor
    text_valid_rows: torch.Tensor | None = None

    @property
    def length(self):
        return self.tags.numel()

    @property
    def device(self):
        return self.video.device

    def validate(self):
        n = self.length
        validate_groups(self.groups, n)
        if self.refiner_text.ndim != 3 or self.refiner_text.shape[0] != 1:
            raise ValueError("Realization refiner input must be [1,rows,text_dim]")
        validate_groups(self.refiner_groups, self.refiner_text.shape[1])
        if self.positions.shape != (n, 3):
            raise ValueError("Realization positions must be [rows,3]")
        for value in (self.tags, self.roles, self.time_ids):
            if value.shape != (n,) or value.dtype != torch.long:
                raise ValueError("Realization row metadata must be long [rows]")
        if (self.tags < 0).any() or (self.tags > 2).any() or (self.time_ids < 0).any():
            raise ValueError("Invalid Realization modality/time metadata")
        if self.text_valid_rows is not None and (
            self.text_valid_rows.shape != (1,)
            or self.text_valid_rows.dtype != torch.int32
            or self.text_valid_rows.device != self.device
        ):
            raise ValueError("Realization bucket length must be a device int32 scalar buffer")
        all_ids = []
        for name in ("video", "audio", "text"):
            ids = getattr(self, name + "_indices")
            if ids.ndim != 1 or ids.dtype != torch.long or (ids < 0).any() or (ids >= n).any():
                raise ValueError("Invalid Realization modality row indices")
            all_ids.append(ids)
            if name != "text":
                value = getattr(self, name)
                if value.ndim != 3 or value.shape[:2] != (1, ids.numel()):
                    raise ValueError("Realization media must be [1,modality_rows,width]")
        ids = torch.cat(all_ids)
        if ids.unique().numel() != ids.numel():
            raise ValueError("Realization modality row ownership overlaps")
        owned = torch.zeros(n, dtype=torch.bool, device=self.device)
        owned[ids] = True
        real = torch.zeros_like(owned)
        for group in self.groups:
            if group.kind != "padding":
                real[group.start : group.end] = True
        if not torch.equal(owned, real):
            raise ValueError("Each nonpadding Realization row must have exactly one modality owner")
        for tag, indices in enumerate((self.video_indices, self.text_indices, self.audio_indices)):
            if torch.any(self.tags[indices] != tag):
                raise ValueError("Realization modality tags disagree with row ownership")
        if torch.any(self.roles < -1) or torch.any(self.roles > 1):
            raise ValueError("Invalid Realization role metadata")
        if self.text_selection.shape != self.text_indices.shape or self.text_selection.dtype != torch.long:
            raise ValueError("Realization text selection geometry mismatch")
        if (self.text_selection < 0).any() or (self.text_selection >= self.refiner_text.shape[1]).any():
            raise ValueError("Realization text selection out of range")
        tensors = [
            self.video,
            self.audio,
            self.positions,
            self.refiner_text,
            self.text_selection,
            self.tags,
            self.roles,
            self.time_ids,
            *all_ids,
        ]
        if any(t.device != self.device for t in tensors):
            raise ValueError("Realization sequence tensors must reside on one device")
        return self

    def to(self, device):
        """Explicit worker ownership transfer; retain semantic metadata."""
        return replace(
            self,
            **{
                name: value.to(device=device)
                for name in self.__dataclass_fields__
                if isinstance(value := getattr(self, name), torch.Tensor)
            },
        )

    def with_media(self, video, audio):
        if video.shape != self.video.shape or audio.shape != self.audio.shape:
            raise ValueError("Realization replacement media geometry mismatch")
        return replace(self, video=video, audio=audio)

    def as_clean(self):
        """Keep current text for commit attention, persist only media KV."""
        if any(g.kind not in ("text", "noisy", "padding") for g in self.groups):
            raise ValueError("Clean commit expects a current-block sequence")
        return replace(
            self,
            groups=tuple(replace(g, kind="clean") if g.kind == "noisy" else g for g in self.groups),
            time_ids=torch.zeros_like(self.time_ids),
        )


def compact_trailing_padding(sequence: RealizationSequence) -> RealizationSequence:
    """Remove ONLY a trailing isolated padding domain, preserving real indices.

    Row-local Realization projections/modulation and domain visibility are independent
    of the padding rows. Different GEMM shapes can still affect BF16 rounding;
    this optimization is explicitly opt-in, with a separate precision gate.
    """
    sequence.validate()
    padding = [i for i, group in enumerate(sequence.groups) if group.kind == "padding"]
    if not padding:
        return sequence
    if padding != [len(sequence.groups) - 1]:
        raise ValueError("Realization compaction requires one trailing padding domain")
    length = sequence.groups[-1].start
    if length == 0:
        raise ValueError("Realization cannot compact an all-padding sequence")
    return replace(
        sequence,
        groups=sequence.groups[:-1],
        positions=sequence.positions[:length].contiguous(),
        tags=sequence.tags[:length].contiguous(),
        roles=sequence.roles[:length].contiguous(),
        time_ids=sequence.time_ids[:length].contiguous(),
    ).validate()
