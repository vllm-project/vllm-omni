# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Training-free, causal evidence selection for the MiniCPM-o duplex window."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any

# Separate host storage of reusable input embeddings. It carries no KV blocks.
STREAMOV_ARCHIVE_TOKENS = 16000
# Peak mean-square energy of a 20 ms window at -50 dBFS.
_AUDIBLE_ENERGY = 1e-5
# Match one CLAP neighborhood on either side of the selected anchor.
_SPEECH_CONTEXT_UNITS = 4
_MAX_AUDIO_RETRIEVAL_TOKENS = 200


def evidence_score(visual: float, audio: float, query_visual: float = 0.0, query_audio: float = 0.0) -> float:
    # Approximate coarse co-burst by salience in the same append, as in v0.
    co_burst = min(visual, audio)
    query_audio *= max(audio, co_burst)
    visual_evidence = max(visual, query_visual)
    audio_evidence = max(audio, query_audio)
    aligned = max(min(query_visual, query_audio), min(visual_evidence, audio_evidence, co_burst))
    return max(visual_evidence - aligned, audio_evidence - aligned, aligned)


def _ranks(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values.values())
    if len(ordered) < 2:
        return dict.fromkeys(values, 0.0)
    return {
        key: (bisect_left(ordered, value) + bisect_right(ordered, value) - 1) / (2 * (len(ordered) - 1))
        for key, value in values.items()
    }


def _pack_by_score(units: list[dict[str, Any]], scores: dict[int, float], budget: int) -> list[dict[str, Any]]:
    chosen = []
    for unit in sorted(units, key=lambda item: (-scores[item["append_seq"]], -item["append_seq"])):
        if unit["length"] <= budget:
            chosen.append(unit)
            budget -= unit["length"]
    return sorted(chosen, key=lambda unit: unit["append_seq"])


def routed_evidence_scores(
    visual: float, audio: float, onset: float, query_visual: float, query_audio: float
) -> dict[str, float]:
    # The acoustic attack near the sampled frame contributes independently
    # from peak energy elsewhere in the observation.
    co_burst = min(visual, audio, onset)
    effective_audio = query_audio * max(audio, co_burst)
    visual_evidence = max(visual, query_visual)
    audio_evidence = max(audio, effective_audio)
    aligned = max(min(query_visual, effective_audio), min(visual_evidence, audio_evidence, co_burst))
    return {"vision": visual_evidence - aligned, "audio": audio_evidence - aligned, "av": aligned}


def routed_unit(unit: dict[str, Any], route: str) -> dict[str, Any]:
    """Project a whole completed append to one modality and its framing tokens."""
    video_cost, audio_cost = unit.get("modality_costs", (0, 0))
    if not (video_cost and audio_cost) or route == "av" or unit.get("route", "av") != "av":
        return {**unit, "historical": True}
    removed = video_cost if route == "audio" else audio_cost
    if removed >= unit["length"]:
        return {**unit, "historical": True}
    return {**unit, "historical": True, "route": route, "length": unit["length"] - removed}


def reserve_routed_history(active: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Keep retrieval capacity when salience selected a single dense suffix."""
    if any(unit.get("historical") for unit in active):
        return active
    count = 0
    for unit in active[:-1]:
        if unit["length"] > budget:
            break
        budget -= unit["length"]
        count += 1
    return [{**unit, "historical": True} for unit in active[:count]] + active[count:]


def pack_opening_event(sources: dict[int, dict[str, Any]], budget: int) -> dict[int, dict[str, Any]]:
    """Restore whole consecutive audio/video units from the session start."""
    kept = {}
    seq = 1
    while (unit := sources.get(seq)) is not None:
        if unit.get("route", "av") != "av" or not all(unit.get("modality_costs", (0, 0))) or unit["length"] > budget:
            break
        kept[seq] = routed_unit(unit, "av")
        budget -= unit["length"]
        seq += 1
    return kept if len(kept) >= 2 else {}


def rerank_routed_history(
    active: list[dict[str, Any]],
    archive: dict[int, dict[str, Any]],
    relevance: list[tuple[int, float | None, float | None]],
    *,
    keep_audio_spans: bool = False,
    restore_opening: bool = False,
) -> list[dict[str, Any]]:
    old = {unit["append_seq"]: unit for unit in active if unit.get("historical")}
    if not old:
        return active
    recent = [unit for unit in active if not unit.get("historical")]
    recent_ids = {unit["append_seq"] for unit in recent}
    sources = {seq: unit for seq, unit in {**old, **archive}.items() if seq not in recent_ids}
    reports = {seq: (v, a) for seq, v, a in relevance if seq in sources}
    if not reports:
        return active
    if restore_opening:
        selected = pack_opening_event(sources, sum(unit["length"] for unit in old.values()))
        if selected:
            return sorted([*recent, *selected.values()], key=lambda unit: unit["append_seq"])

    visual = _ranks({seq: v for seq, (v, _) in reports.items() if v is not None})
    audio = _ranks({seq: a for seq, (_, a) in reports.items() if a is not None})
    routes = {}
    for seq in reports.keys() | old.keys():
        unit = sources[seq]
        visual_salience, audio_salience = unit["salience"]
        query_audio = audio.get(seq, 0.0)
        if keep_audio_spans and unit.get("audio_peak", 0.0) < _AUDIBLE_ENERGY:
            audio_salience = query_audio = 0.0
        scores = routed_evidence_scores(
            visual_salience, audio_salience, unit.get("boundary_onset", 0.0), visual.get(seq, 0.0), query_audio
        )
        route = max(scores, key=scores.__getitem__)
        routes[seq] = (route, scores[route])

    candidates = sorted(routes, key=lambda seq: (-routes[seq][1], -seq))
    remaining = sum(unit["length"] for unit in old.values())
    kept: dict[int, dict[str, Any]] = {}
    used_visual: list[int] = []
    # Let a complete speech cue compete before visually similar neighboring
    # frames have exhausted the budget. Return unused audio capacity to
    # the common pool. The final pass can fill space without diversification.
    passes = [(True, False), (False, False)]
    if keep_audio_spans:
        passes.insert(0, (False, True))
    audio_budget = min(_MAX_AUDIO_RETRIEVAL_TOKENS, remaining // 3)
    for spread, audio_first in passes:
        for seq in candidates:
            route, score = routes[seq]
            if audio_first and (route == "vision" or seq not in audio):
                continue
            previous = kept.get(seq)
            include_vision = keep_audio_spans and previous and previous.get("route") == "audio" and route != "audio"
            if (previous is not None and not include_vision) or score <= 0:
                continue
            if spread and route != "audio" and any(abs(seq - prior) < 3 for prior in used_visual):
                continue
            group = [seq]
            if keep_audio_spans and route in {"audio", "av"}:
                # Walk the actual audio around the anchor. Append IDs may
                # start at any offset after earlier dialogue, clock or video
                # events and do not identify CLAP's fixed-size chunks.
                before, after = seq, seq
                for _ in range(_SPEECH_CONTEXT_UNITS):
                    if sources.get(before - 1, {}).get("audio_peak", 0.0) < _AUDIBLE_ENERGY:
                        break
                    before -= 1
                for _ in range(_SPEECH_CONTEXT_UNITS):
                    if sources.get(after + 1, {}).get("audio_peak", 0.0) < _AUDIBLE_ENERGY:
                        break
                    after += 1
                group = range(before, after + 1)
            elif route == "audio" and spread:
                start = seq - (seq - 1) % 4
                group = [index for index in range(start, start + 4) if index in reports and index not in kept]
            selection = []
            for index in group:
                kind = "audio" if keep_audio_spans and index != seq else route
                previous = kept.get(index)
                if previous is not None and previous.get("route", "av") != kind:
                    kind = "av"
                selection.append(routed_unit(sources[index], kind))
            cost = sum(
                unit["length"] - kept.get(index, {}).get("length", 0)
                for index, unit in zip(group, selection, strict=True)
            )
            if cost <= remaining and (not audio_first or cost <= audio_budget):
                kept.update(zip(group, selection, strict=True))
                remaining -= cost
                if audio_first:
                    audio_budget -= cost
                if route != "audio":
                    used_visual.append(seq)
    return sorted([*recent, *kept.values()], key=lambda unit: unit["append_seq"])


def update_routed_archive(
    completed: list[dict[str, Any]], active: list[dict[str, Any]], archive: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    full_active = {unit["append_seq"] for unit in active if not unit.get("route")}
    full_sources = {**{unit["append_seq"]: unit for unit in completed}, **archive}
    candidates = [unit for seq, unit in full_sources.items() if seq not in full_active and not unit.get("route")]
    scores = {unit["append_seq"]: unit["evidence_score"] for unit in candidates}
    return {unit["append_seq"]: unit for unit in _pack_by_score(candidates, scores, STREAMOV_ARCHIVE_TOKENS)}


def select_streamov_units(
    lengths: list[int], scores: list[float], content_tokens: int, history_tokens: int
) -> tuple[list[int], list[int]]:
    """Keep a recent contiguous suffix and add whole older units by evidence score.

    Return chronological keep indices and the older evidence indices within those
    indices. Spend any unused history budget by growing the recent suffix.
    A history budget of zero gives the matched full-prefill FIFO control.
    """
    history_tokens = min(content_tokens, history_tokens)
    remaining = content_tokens - history_tokens
    recent_start = len(lengths)
    while recent_start > 0 and lengths[recent_start - 1] <= remaining:
        recent_start -= 1
        remaining -= lengths[recent_start]

    candidate_indices = (
        sorted(range(recent_start), key=lambda index: (-scores[index], -index)) if history_tokens else []
    )
    evidence = set()
    for index in candidate_indices:
        cost = lengths[index]
        if scores[index] > 0 and cost <= history_tokens:
            evidence.add(index)
            history_tokens -= cost

    remaining += history_tokens
    while recent_start > 0:
        index = recent_start - 1
        if index in evidence:
            evidence.remove(index)
        elif lengths[index] <= remaining:
            remaining -= lengths[index]
        else:
            break
        recent_start = index
    return sorted(evidence) + list(range(recent_start, len(lengths))), sorted(evidence)


def plan_retrieval(
    units: list[dict[str, Any]],
    archive: dict[int, dict[str, Any]],
    *,
    projected_len: int,
    high: int,
    low: int,
    history_budget: int,
    relevance: list[tuple[int, float | None, float | None]],
    restore_opening: bool,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, object] | None]:
    """Plan a new active window and host cache from the same source snapshot."""
    lengths = [unit["length"] for unit in units]
    original_tokens = sum(lengths)
    content_tokens = max(0, low - (projected_len - original_tokens))
    retained = units
    if projected_len > high:
        keep, historical = select_streamov_units(
            lengths, [unit["evidence_score"] for unit in units], content_tokens, history_budget
        )
        historical = set(historical)
        retained = [{**units[index], "historical": index in historical} for index in keep]
    if relevance:
        recent, _ = select_streamov_units(lengths, [], max(0, content_tokens - history_budget), 0)
        candidates = update_routed_archive(units, [units[index] for index in recent], archive)
        if candidates:
            retained = rerank_routed_history(
                reserve_routed_history(retained, history_budget),
                candidates,
                relevance,
                keep_audio_spans=True,
                restore_opening=restore_opening,
            )

    def shape(rows):
        return [(row["append_seq"], row["length"], row.get("route", "av")) for row in rows]

    if shape(retained) == shape(units):
        return retained, archive, None
    positions = {unit["append_seq"]: index for index, unit in enumerate(units)}
    next_archive = update_routed_archive(units, retained, archive)
    layout = []
    for unit in retained:
        seq = unit["append_seq"]
        index = positions.get(seq)
        route = unit.get("route", "av")
        if index is not None and units[index].get("route", "av") == route:
            layout.append((index, 0, 0, unit["length"]))
            continue
        source = -seq if seq in archive else positions[seq]
        vision, audio = unit.get("modality_costs", (0, 0))
        start, count = {"vision": (1 + vision, audio), "audio": (1, vision), "av": (0, 0)}[route]
        layout.append((source, start, count, unit["length"]))
    return (
        retained,
        next_archive,
        {
            "retrieval_keep": layout,
            "retrieval_archive": [(seq, -seq if seq in archive else positions[seq]) for seq in next_archive],
            "retrieval_query": bool(relevance),
            "retrieval_history": [
                (unit["append_seq"], unit.get("route", "av"), unit["length"])
                for unit in retained
                if unit.get("historical")
            ],
        },
    )
