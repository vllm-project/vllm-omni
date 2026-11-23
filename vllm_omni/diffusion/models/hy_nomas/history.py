# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import SegmentHandle


@dataclass(frozen=True)
class HyNomasHistoryFrame:
    handle: SegmentHandle
    position: int
    segment: int
    text_tokens: int = 0
    action_start: int = -1


def recent_history(frames: list[HyNomasHistoryFrame], budget: int) -> tuple[list[HyNomasHistoryFrame], int]:
    tokens = sum(frame.handle.num_tokens for frame in frames)
    removed = 0
    count = 0
    while tokens - removed > budget:
        removed += frames[count].handle.num_tokens
        count += 1
    return frames[count:], removed


def action_tails(frames: list[HyNomasHistoryFrame]) -> list[tuple[int, int, int]]:
    tails = []
    cursor = 0
    for frame in frames:
        end = cursor + frame.handle.num_tokens
        if 0 <= frame.action_start < frame.text_tokens:
            tails.append((frame.segment, end - frame.text_tokens + frame.action_start, end))
        cursor = end
    return tails
