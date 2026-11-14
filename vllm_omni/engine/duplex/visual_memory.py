# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared bounded visual evidence, intermediate summaries and long-term memory."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from time import monotonic
from typing import Literal

from PIL import Image

VisualClosureReason = Literal["count", "context", "image_limit", "capacity"]


@dataclass(frozen=True)
class EncodedVisualImage:
    """Lossless retained pixels; encode/decode in prompt-preparation workers."""

    data: bytes
    width: int
    height: int

    @classmethod
    def from_image(cls, image: Image.Image) -> "EncodedVisualImage":
        if image.mode != "RGB":
            raise ValueError("Visual memory requires normalized RGB images")
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return cls(buffer.getvalue(), image.width, image.height)

    def decode(self) -> Image.Image:
        with Image.open(BytesIO(self.data)) as image:
            return image.convert("RGB")


@dataclass(frozen=True)
class VisualFrame:
    id: int
    source_id: str
    time_s: float
    encoded_image: EncodedVisualImage
    token_count: int

    @property
    def image(self) -> Image.Image:
        return self.encoded_image.decode()


@dataclass(frozen=True)
class VisualSummary:
    """A retained visual span and its summary cost.

    Standalone stores keep summary text here. A ledger-backed adapter may keep
    metadata-only entries and resolve text from its authority before rendering.
    """

    first_id: int
    last_id: int
    start_s: float
    end_s: float
    text: str
    token_count: int
    chunk_index: int | None = None
    closure_reason: VisualClosureReason = "count"
    first_source_id: str = ""
    last_source_id: str = ""


@dataclass(frozen=True)
class VisualMemorySource:
    kind: Literal["frames", "long"]
    generation: int
    covered_ids: tuple[int, ...]
    chunk_index: int | None = None
    closure_reason: VisualClosureReason = "count"


class VisualMemoryCapacityError(ValueError):
    """The next observation cannot fit without losing accepted evidence."""


@dataclass
class VisualMemory:
    """Model-owned visual evidence; only successful summaries retire sources.

    Frames retain lossless encoded pixels, decoded only for model requests.
    The encoded-byte and frame limits bound retained memory per Session.
    Mutation belongs to the Session mailbox, after observation acceptance.
    """

    pending: list[VisualFrame] = field(default_factory=list)
    mid_term: list[VisualSummary] = field(default_factory=list)
    long_term: list[VisualSummary] = field(default_factory=list)
    generation: int = 0
    next_id: int = 0
    next_chunk_index: int = 1
    start_time: float = field(default_factory=monotonic)
    chunk_frames: int = 100
    max_pending: int = 200
    max_pixel_bytes: int = 200 * 262144 * 3
    max_encoded_bytes: int = 32 * 1024 * 1024
    mid_group: int = 5
    max_long: int = 15
    mid_max_tokens: int = 4000
    long_max_tokens: int = 4000

    @property
    def pixel_bytes(self) -> int:
        return sum(frame.encoded_image.width * frame.encoded_image.height * 3 for frame in self.pending)

    @property
    def encoded_bytes(self) -> int:
        return sum(len(frame.encoded_image.data) for frame in self.pending)

    def check_capacity(self, images: Sequence[EncodedVisualImage | Image.Image]) -> None:
        encoded = [
            EncodedVisualImage.from_image(image) if isinstance(image, Image.Image) else image for image in images
        ]
        added_bytes = sum(image.width * image.height * 3 for image in images)
        if (
            len(self.pending) + len(images) > self.max_pending
            or self.pixel_bytes + added_bytes > self.max_pixel_bytes
            or self.encoded_bytes + sum(len(image.data) for image in encoded) > self.max_encoded_bytes
        ):
            raise VisualMemoryCapacityError("Visual memory is full; retry after pending observations are summarized")

    def append(
        self, *, source_id: str, image: EncodedVisualImage | Image.Image, token_count: int, time_s: float | None = None
    ) -> VisualFrame:
        if isinstance(image, Image.Image):
            image = EncodedVisualImage.from_image(image)
        self.check_capacity([image])
        frame = VisualFrame(
            self.next_id, source_id, monotonic() - self.start_time if time_s is None else time_s, image, token_count
        )
        self.pending.append(frame)
        self.next_id += 1
        return frame

    def frame_source(
        self, count: int | None = None, *, closure_reason: VisualClosureReason = "count"
    ) -> VisualMemorySource | None:
        frames = self.pending[: self.chunk_frames if count is None else max(0, count)]
        return (
            VisualMemorySource(
                "frames", self.generation, tuple(frame.id for frame in frames), self.next_chunk_index, closure_reason
            )
            if frames
            else None
        )

    def long_source(
        self, count: int | None = None, *, closure_reason: VisualClosureReason = "count"
    ) -> VisualMemorySource | None:
        summaries = self.mid_term[: self.mid_group if count is None else max(0, count)]
        return (
            VisualMemorySource(
                "long", self.generation, tuple(summary.last_id for summary in summaries), closure_reason=closure_reason
            )
            if summaries
            else None
        )

    def matches(self, source: VisualMemorySource) -> bool:
        if source.generation != self.generation or not source.covered_ids:
            return False
        count = len(source.covered_ids)
        if source.kind == "frames":
            return source.chunk_index == self.next_chunk_index and (
                tuple(frame.id for frame in self.pending[:count]) == source.covered_ids
            )
        if source.kind == "long":
            return tuple(summary.last_id for summary in self.mid_term[:count]) == source.covered_ids
        return False

    def apply(self, source: VisualMemorySource, *, text: str, token_count: int) -> bool:
        text = text.strip()
        max_tokens = self.mid_max_tokens if source.kind == "frames" else self.long_max_tokens
        if not text or not 0 < token_count <= max_tokens or not self.matches(source):
            return False
        count = len(source.covered_ids)
        if source.kind == "frames":
            if len(self.mid_term) >= self.mid_group:
                return False
            first, last = self.pending[0], self.pending[count - 1]
            summary = VisualSummary(
                first.id,
                last.id,
                first.time_s,
                last.time_s,
                text,
                token_count,
                source.chunk_index,
                source.closure_reason,
                first.source_id,
                last.source_id,
            )
            self.mid_term.append(summary)
            del self.pending[:count]
            self.next_chunk_index += 1
        else:
            first_summary, last_summary = self.mid_term[0], self.mid_term[count - 1]
            summary = VisualSummary(
                first_summary.first_id,
                last_summary.last_id,
                first_summary.start_s,
                last_summary.end_s,
                text,
                token_count,
                closure_reason=source.closure_reason,
                first_source_id=first_summary.first_source_id,
                last_source_id=last_summary.last_source_id,
            )
            self.long_term.append(summary)
            del self.mid_term[:count]
            del self.long_term[: max(0, len(self.long_term) - self.max_long)]
        return True

    def clear(self) -> None:
        self.pending.clear()
        self.mid_term.clear()
        self.long_term.clear()
        self.generation += 1
