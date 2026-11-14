# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native frame-v5 inference layout, matched to Torchtitan compact/select.

Geometry is explicit: legacy streaming uses one code / four frames;
duo_5_17 checkpoints generate five codes / seventeen frames per model block.
No training/repository imports, no user media inserted into Realization context.
"""

import math
from dataclasses import replace

import torch

from .contracts import RealizationGroup, RealizationSequence
from .pipeline import RealizationBlockCondition, validate_condition


def assemble_condition(
    ticket,
    *,
    reference,
    shared,
    body,
    shared_lengths,
    seed,
    final=False,
    codec_mode="stream_4frame",
    rope="old",
    reset_media_cache=False,
):
    """Build padded prefix/current views from committed, selected H50 snapshots.

    Shared rows partition description + speech cue. All Realization positions are the
    codec clock/spatial coordinates, NOT the Intent MRoPE positions.
    """
    if rope not in ("old", "new"):
        raise ValueError("Unknown Realization RoPE policy")
    if codec_mode == "duo_5_17":
        frames, video_offsets, audio_steps = 17, (0, 1, 5, 9, 13), 29
    elif codec_mode in ("stream_4frame", "old"):
        frames, video_offsets, audio_steps = 4, (0,), 7
    else:
        raise ValueError("Unknown Realization codec geometry")
    from .condition import RealizationConditionTicket

    if not isinstance(ticket, RealizationConditionTicket):
        raise ValueError("Expected fenced Realization condition ticket")
    if reference.ndim != 4 or tuple(reference.shape[:2]) != (24, 1) or any(d < 2 or d % 2 for d in reference.shape[2:]):
        raise ValueError("Expected single normalized reference latent")
    if (
        shared.ndim != 2
        or body.ndim != 2
        or shared.shape[1] != 5120
        or body.shape[1] != 5120
        or not len(body)
        or len(shared_lengths) != 2
        or any(type(n) is not int or n < 0 for n in shared_lengths)
        or sum(shared_lengths) != len(shared)
        or not len(shared)
    ):
        raise ValueError("Expected partitioned shared H50 and nonempty committed body")
    if any(not z.is_floating_point() or not torch.isfinite(z).all() for z in (reference, shared, body)):
        raise ValueError("Nonfinite/nonfloating conditioning")
    # Transport snapshots are owned by this block, not mutable producer views.
    reference, shared, body = (z.detach().cpu().clone() for z in (reference, shared, body))
    h, w = reference.shape[-2:]
    rows = h * w // 4
    k = ticket.block
    area = math.sqrt(h * w)
    hg, wg = [((1 - d / area) / 2 + torch.arange(d // 2, dtype=torch.float64) * 2 / area) * 32 for d in (h, w)]
    spatial = torch.cartesian_prod(hg, wg)
    start_ms = k * frames * 1000 / 24
    packed = reference.reshape(24, 1, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(1, rows, 96).float()
    ref_groups = []
    cursor = 0
    for n in shared_lengths:
        if n:
            ref_groups.append(RealizationGroup(0, -1, "shared", cursor, cursor + n))
        cursor += n
    ref_groups.append(RealizationGroup(0, k, "text", cursor, cursor + len(body)))
    all_text = torch.cat((shared, body))[None]

    def build(prefix):
        live = rows + len(shared) if prefix else len(body) + rows * len(video_offsets) + 2 * audio_steps
        length = max(128, math.ceil(live / 128) * 128)
        positions = torch.zeros(length, 3, dtype=torch.float64)
        tags = torch.zeros(length, dtype=torch.long)
        roles = torch.full_like(tags, -1)
        times = torch.zeros_like(tags)
        if prefix:
            groups = [RealizationGroup(0, -1, "reference", 0, rows)]
            vi = torch.arange(rows)
            ai = torch.empty(0, dtype=torch.long)
            ti = torch.arange(rows, live)
            video = packed
            audio = torch.empty(1, 0, 32)
            positions[:rows, 1:] = spatial
            roles[:rows] = 1
            positions[ti, 0] = -1 if rope == "old" else torch.arange(-len(shared), 0, dtype=torch.float64)
            tags[ti] = 1
            cursor = rows
            for n in shared_lengths:
                if n:
                    groups.append(RealizationGroup(0, -1, "shared", cursor, cursor + n))
                cursor += n
            text = shared[None]
            rg = tuple(g for g in ref_groups if g.kind == "shared")
            selection = torch.arange(len(shared))
        else:
            n = len(body)
            vi = torch.arange(n, n + rows * len(video_offsets))
            ai = torch.arange(n + rows * len(video_offsets), live)
            ti = torch.arange(n)
            groups = [
                RealizationGroup(0, k, "text", 0, n),
                RealizationGroup(0, k, "noisy", n, n + rows * len(video_offsets)),
                RealizationGroup(0, k, "noisy", n + rows * len(video_offsets), live),
            ]
            video = torch.zeros(1, rows * len(video_offsets), 96)
            audio = torch.zeros(1, 2 * audio_steps, 32)
            positions[ti, 0] = start_ms * 0.04
            if rope == "new":
                positions[ti, 0] += torch.arange(n, dtype=torch.float64)
            positions[vi, 0] = (
                torch.tensor([(start_ms + o * 1000 / 24) * 0.04 for o in video_offsets])
                .repeat_interleave(rows)
                .double()
            )
            positions[vi, 1:] = spatial.repeat(len(video_offsets), 1)
            positions[ai, 0] = (start_ms * 0.04 + torch.arange(audio_steps)).repeat(2).double()
            positions[ai[:audio_steps], 2] = wg[0]
            positions[ai[audio_steps:], 2] = wg[-1]
            tags[ti] = 1
            tags[ai] = 2
            roles[:live] = 1
            times[n:live] = 1
            text = all_text
            rg = tuple(ref_groups)
            selection = torch.arange(len(shared), len(shared) + n)
        if live < length:
            groups.append(RealizationGroup(-1, -1, "padding", live, length))
        return RealizationSequence(
            video, audio, vi, ai, ti, positions, tags, roles, times, tuple(groups), text, rg, selection
        ).validate()

    c = RealizationBlockCondition(
        ticket.session_id,
        ticket.fence,
        k,
        build(False),
        seed,
        prefix=build(True) if k == 0 or reset_media_cache else None,
        final=final,
        tenant=ticket.tenant,
        reset_media_cache=reset_media_cache,
    )
    validate_condition(c)
    return c


def bucket_text_sequence(sequence, boundaries):
    """Cold admission: fixed capacity with a device-side valid text length."""
    n = sequence.text_indices.numel()
    capacity = next(bound for bound in boundaries if bound >= n)
    extra = capacity - n
    live = max(g.end for g in sequence.groups if g.kind != "padding")
    length = math.ceil((live + extra) / 128) * 128
    fields = {}
    for name in ("positions", "tags", "roles", "time_ids"):
        old = getattr(sequence, name)
        value = old.new_zeros((length, *old.shape[1:]))
        if name == "roles":
            value.fill_(-1)
        value[:n] = old[:n]
        value[capacity : live + extra] = old[n:live]
        if name == "tags":
            value[n:capacity] = 1
        fields[name] = value
    groups = [
        replace(g, end=capacity) if g.kind == "text" else replace(g, start=g.start + extra, end=g.end + extra)
        for g in sequence.groups
        if g.kind != "padding"
    ]
    if live + extra < length:
        groups.append(RealizationGroup(-1, -1, "padding", live + extra, length))
    text = torch.cat(
        (sequence.refiner_text, sequence.refiner_text.new_zeros((1, extra, sequence.refiner_text.shape[-1]))), dim=1
    )
    refiner_groups = tuple(replace(g, end=g.end + extra) if g.kind == "text" else g for g in sequence.refiner_groups)
    start = sequence.refiner_text.shape[1] - n
    return replace(
        sequence,
        **fields,
        groups=tuple(groups),
        video_indices=sequence.video_indices + extra,
        audio_indices=sequence.audio_indices + extra,
        text_indices=torch.arange(capacity, device=sequence.device),
        text_selection=torch.arange(start, start + capacity, device=sequence.device),
        refiner_text=text,
        refiner_groups=refiner_groups,
        text_valid_rows=torch.tensor([n], device=sequence.device, dtype=torch.int32),
    ).validate()
