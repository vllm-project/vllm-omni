# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA v1 and AURA_v2 control-token ids.

One process loads one thinker. The two families must not be unioned into one
stop list: v2's vocabulary (248320) still contains v1's silent id ``151669``
as an ordinary piece, so stopping on both truncates v2 speech. v1's head
(vocab 151936) cannot emit the v2 ids. The stage's own stop ids select the
family; an empty list keeps v1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AuraControlTokens:
    name: str
    silent_id: int
    im_end_id: int
    im_start_id: int
    assistant_id: int
    respond_id: int | None


# Qwen3-VL AURA. ``<|respond|>`` is five ordinary pieces, not one id.
AURA_V1 = AuraControlTokens(
    name="v1",
    silent_id=151669,
    im_end_id=151645,
    im_start_id=151644,
    assistant_id=77091,
    respond_id=None,
)
# qwen3_5 checkpoint-1318. ``<|respond|>`` is a single id and is not a stop.
AURA_V2 = AuraControlTokens(
    name="v2",
    silent_id=248070,
    im_end_id=248046,
    im_start_id=248045,
    assistant_id=74455,
    respond_id=248077,
)

QWEN_NEWLINE_ID = 198


def profile_for_stop_ids(stop_ids: Sequence[int] | None) -> AuraControlTokens:
    """Pick the thinker family already declared on this stage.

    A listed v2 silent id with no v1 silent id selects v2, even if a copied
    v1 ``<|im_end|>`` is also present. Both silent ids together stay on v1 so
    an accidental union does not drop the v1 stops.
    """
    ids = {int(token_id) for token_id in (stop_ids or ())}
    v2_silent = AURA_V2.silent_id in ids
    v1_silent = AURA_V1.silent_id in ids
    if v2_silent and not v1_silent:
        return AURA_V2
    if v1_silent and not v2_silent:
        return AURA_V1
    v2_end = AURA_V2.im_end_id in ids
    v1_end = AURA_V1.im_end_id in ids
    if v2_end and not v1_end and not v1_silent:
        return AURA_V2
    return AURA_V1


def stop_ids_for_profile(stop_ids: Sequence[int] | None, profile: AuraControlTokens) -> list[int]:
    """Keep one family's silent / im_end pair and drop the other, plus respond."""
    inactive = AURA_V1 if profile is AURA_V2 else AURA_V2
    drop = {inactive.silent_id, inactive.im_end_id}
    if AURA_V2.respond_id is not None:
        drop.add(AURA_V2.respond_id)
    kept = [int(token_id) for token_id in (stop_ids or ()) if int(token_id) not in drop]
    for stop_id in (profile.silent_id, profile.im_end_id):
        if stop_id not in kept:
            kept.append(stop_id)
    return kept


def chatml_prefix(profile: AuraControlTokens) -> list[int]:
    return [profile.im_start_id, profile.assistant_id, QWEN_NEWLINE_ID]


def chatml_suffix(profile: AuraControlTokens) -> list[int]:
    return [
        profile.im_end_id,
        QWEN_NEWLINE_ID,
        profile.im_start_id,
        profile.assistant_id,
        QWEN_NEWLINE_ID,
    ]


def profile_for_token_ids(token_ids: Sequence[int]) -> AuraControlTokens:
    """Match a completion or chatml-wrapped id sequence to one thinker."""
    ids = list(token_ids)
    for profile in (AURA_V2, AURA_V1):
        prefix = chatml_prefix(profile)
        if ids[: len(prefix)] == prefix:
            return profile
    markers = {
        AURA_V2.silent_id,
        AURA_V2.im_end_id,
        AURA_V2.im_start_id,
    }
    if AURA_V2.respond_id is not None:
        markers.add(AURA_V2.respond_id)
    if any(token_id in markers for token_id in ids):
        return AURA_V2
    return AURA_V1
