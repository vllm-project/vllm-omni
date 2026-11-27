# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate retention policy; all KV tensors and transactions belong to the engine."""

from __future__ import annotations

from collections.abc import Callable

from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState, SegmentHandle

AUDIO_TOKEN_TAG = 2
TEXT_TOKEN_TAG = 1
VIDEO_TOKEN_TAG = 0


class TaoMateKVHistory:
    """Select first-phase video and the two latest phases, in original row order."""

    def __init__(self, cache: Callable[[], ARDiffusionSegmentedKVState]) -> None:
        self._cache = cache
        self._sink: SegmentHandle | None = None
        self._recent: list[tuple[SegmentHandle | None, SegmentHandle]] = []
        self._pending: tuple[SegmentHandle, SegmentHandle] | None = None
        self.committed_phases = 0

    @property
    def cache(self) -> ARDiffusionSegmentedKVState:
        return self._cache()

    def begin_commit(self, *, audio_tokens: int, video_tokens: int) -> None:
        if self._pending is not None:
            raise RuntimeError("TaoMate clean commit is already active")
        # The packed document orders text, audio, video. Only its media is
        # persisted, with audio/video separate so either can be released.
        audio, video = self.cache.begin_commit((audio_tokens, video_tokens))
        self._pending = (audio, video)

    def _retained(
        self, recent: list[tuple[SegmentHandle | None, SegmentHandle]], sink: SegmentHandle
    ) -> tuple[SegmentHandle, ...]:
        handles = [handle for phase in recent for handle in phase if handle is not None]
        if sink not in handles:
            handles.insert(0, sink)
        return tuple(handles)

    def commit(self) -> None:
        if self._pending is None:
            raise RuntimeError("TaoMate clean commit is not active")
        sink = self._sink or self._pending[1]
        recent = (self._recent + [self._pending])[-2:]
        # Publish the ledger only after every layer's engine transaction succeeds.
        self.cache.commit(retain=self._retained(recent, sink))
        self._sink, self._recent = sink, recent
        self._pending = None
        self.committed_phases += 1

    def rollback(self) -> None:
        self.cache.rollback()
        self._pending = None

    def drop_audio_history(self) -> int:
        if self._pending is not None:
            raise RuntimeError("Cannot reset audio during a clean commit")
        if self._sink is None:
            return 0
        dropped = sum(audio.num_tokens for audio, _ in self._recent if audio is not None)
        recent = [(None, video) for _, video in self._recent]
        self.cache.retain(self._retained(recent, self._sink))
        self._recent = recent
        return dropped

    def clear(self) -> None:
        # Session reset/close releases the physical allocations in ARDiffusionKVState.
        self._pending = None
        self._recent.clear()
        self._sink = None
        self.committed_phases = 0
