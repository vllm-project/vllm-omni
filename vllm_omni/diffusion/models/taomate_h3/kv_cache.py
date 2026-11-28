# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate retention policy; all KV tensors and transactions belong to the engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState, SegmentHandle

AUDIO_TOKEN_TAG = 2
TEXT_TOKEN_TAG = 1
VIDEO_TOKEN_TAG = 0

# The Kiwi/segment bound the pipeline declares in ``ar_diffusion_kv_cache_spec``
# (``ARDiffusionSegmentedKVSpec.max_retained_segments``). A policy may not
# select more segments than the admission budget reserved for it, so the check
# belongs here, at construction, not at the first commit.
MAX_RETAINED_SEGMENTS = 5


@dataclass(frozen=True)
class TaoMateRetention:
    """Which committed phases survive past the current one.

    ``keep_sink`` retains the first phase's video segment as an attention sink
    and ``recent_phases`` retains the newest phases' audio and video segments.
    Segments are selected, never rewritten: the engine owns the storage, so a
    policy can only choose an ordered subset of handles it already committed.
    """

    keep_sink: bool = True
    recent_phases: int = 2

    def __post_init__(self) -> None:
        if self.recent_phases < 0:
            raise ValueError(f"recent_phases must be non-negative, got {self.recent_phases}")
        # Each retained phase owns an audio and a video segment.
        segments = self.recent_phases * 2 + int(self.keep_sink)
        if segments > MAX_RETAINED_SEGMENTS:
            raise ValueError(
                f"TaoMate retention selects {segments} segments, above the declared budget of "
                f"{MAX_RETAINED_SEGMENTS} (keep_sink={self.keep_sink}, recent_phases={self.recent_phases})"
            )


# Selectable presets. ``sink_recent`` is the original hard-coded behaviour: the
# sink video segment plus the two newest phases, which fills the five-segment
# budget exactly. ``recent_only`` drops the sink and serves as the experimental
# control that shows the window without it.
RETENTION_POLICIES: dict[str, TaoMateRetention] = {
    "sink_recent": TaoMateRetention(keep_sink=True, recent_phases=2),
    "recent_only": TaoMateRetention(keep_sink=False, recent_phases=2),
}
DEFAULT_RETENTION = "sink_recent"


def resolve_retention(name: str | TaoMateRetention | None) -> TaoMateRetention:
    """Resolve a deployment policy name (or policy value) to a policy.

    ``None`` selects :data:`DEFAULT_RETENTION`. Unknown names are rejected
    rather than silently falling back: a retention policy decides which KV
    the model can still attend to, so a typo must not change the video a
    session produces.
    """
    if isinstance(name, TaoMateRetention):
        return name
    key = DEFAULT_RETENTION if name is None else str(name)
    try:
        return RETENTION_POLICIES[key]
    except KeyError:
        raise ValueError(
            f"Unknown TaoMate retention policy {key!r}; expected one of {sorted(RETENTION_POLICIES)}"
        ) from None


class TaoMateKVHistory:
    """Select the policy's retained segments, in original row order.

    The engine publishes or rolls back each phase's transaction; this ledger
    only moves after that transaction succeeded, so a failed commit never
    advances model history.
    """

    def __init__(
        self,
        cache: Callable[[], ARDiffusionSegmentedKVState],
        retention: TaoMateRetention | None = None,
    ) -> None:
        self._cache = cache
        self.retention = TaoMateRetention() if retention is None else retention
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
        self, recent: list[tuple[SegmentHandle | None, SegmentHandle]], sink: SegmentHandle | None
    ) -> tuple[SegmentHandle, ...]:
        handles = [handle for phase in recent for handle in phase if handle is not None]
        if self.retention.keep_sink and sink is not None and sink not in handles:
            handles.insert(0, sink)
        return tuple(handles)

    def _recent_window(
        self, recent: list[tuple[SegmentHandle | None, SegmentHandle]]
    ) -> list[tuple[SegmentHandle | None, SegmentHandle]]:
        if self.retention.recent_phases <= 0:
            return []
        return recent[-self.retention.recent_phases :]

    def commit(self) -> None:
        if self._pending is None:
            raise RuntimeError("TaoMate clean commit is not active")
        sink = self._sink or self._pending[1]
        recent = self._recent_window(self._recent + [self._pending])
        # Publish the ledger only after every layer's engine transaction
        # succeeds: ``cache.commit`` raises on an unstageable layer and leaves
        # ``_pending``/``_sink``/``_recent`` untouched, so the caller's
        # rollback still releases exactly the unpublished segments.
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
        # Same ordering rule as ``commit``: the engine's retention change lands
        # first, so a failed retain leaves the ledger describing what is stored.
        self.cache.retain(self._retained(recent, self._sink))
        self._recent = recent
        return dropped

    def clear(self) -> None:
        # Session reset/close releases the physical allocations in ARDiffusionKVState.
        self._pending = None
        self._recent.clear()
        self._sink = None
        self.committed_phases = 0
