# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen source bindings for the shared visual memory store.

Session retains image URLs. Only bounded batches are decoded in preparation
workers. Published visual memory refers to source message positions, not an
invented video clock; input_image may represent a photo or an extracted frame.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO

import pybase64
from PIL import Image
from vllm.multimodal.image import open_image

from vllm_omni.engine.duplex.visual_memory import VisualMemory, VisualMemorySource


@dataclass(frozen=True)
class VisualInput:
    message_index: int
    part_index: int
    url: str


@dataclass(frozen=True)
class VisualWork:
    """Either load the next source batch or compress already admitted evidence."""

    generation: int
    inputs: tuple[VisualInput, ...]
    source: VisualMemorySource | None = None
    start: int = 0
    end: int = 0


def decode_image(url: str) -> Image.Image:
    """Match the existing Qwen input_image normalization, without network I/O."""
    header, separator, encoded = url.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ValueError("Qwen visual history requires base64 image data URLs")
    with open_image(BytesIO(pybase64.b64decode(encoded, validate=True))) as image:
        image.thumbnail((448, 448))
        return image.convert("RGB").copy()


@dataclass
class QwenVisualHistory:
    memory: VisualMemory = field(default_factory=VisualMemory)
    inputs: tuple[VisualInput, ...] = ()
    admitted: int = 0
    covered: int = 0

    def reconcile(self, history: Sequence[Mapping[str, object]]) -> None:
        inputs = tuple(
            VisualInput(index, part_index, part["image_url"]["url"])
            for index, message in enumerate(history)
            if isinstance(message.get("content"), list)
            for part_index, part in enumerate(message["content"])
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        if inputs[: len(self.inputs)] != self.inputs:
            self.memory.clear()
            self.admitted = self.covered = 0
        self.inputs = inputs

    def covered_image(self, message_index: int, part_index: int) -> bool:
        return any(
            item.message_index == message_index and item.part_index == part_index
            for item in self.inputs[: self.covered]
        )

    def plan(self, *, required_end: int, completed_end: int, max_images: int) -> VisualWork | None:
        memory = self.memory
        source = None
        if len(memory.mid_term) >= memory.mid_group:
            source = memory.long_source()
        elif memory.pending:
            source = memory.frame_source(min(max_images, memory.chunk_frames))
        if source is not None:
            return VisualWork(memory.generation, self.inputs[: self.admitted], source)
        eligible = self.admitted
        pressure = len(self.inputs) - self.covered >= max_images
        for index in range(self.admitted, len(self.inputs)):
            item = self.inputs[index]
            if item.message_index >= completed_end or (item.message_index >= required_end and not pressure):
                break
            eligible = index + 1
            if eligible - self.admitted >= min(max_images, memory.chunk_frames, memory.max_pending):
                break
        if eligible > self.admitted:
            return VisualWork(memory.generation, self.inputs[:eligible], start=self.admitted, end=eligible)
        return None

    def matches(self, work: VisualWork) -> bool:
        return (
            work.generation == self.memory.generation
            and self.inputs[: len(work.inputs)] == work.inputs
            and (self.memory.matches(work.source) if work.source is not None else self.admitted == work.start)
        )

    def summaries(self) -> tuple[str, ...]:
        return tuple(
            f"Visual observations from source messages "
            f"{self.inputs[int(item.first_source_id)].message_index + 1}–"
            f"{self.inputs[int(item.last_source_id)].message_index + 1}:\n{item.text}"
            for item in [*self.memory.long_term, *self.memory.mid_term]
        )

    def apply(self, work: VisualWork, *, text: str, token_count: int) -> bool:
        if work.source is None or not self.matches(work):
            return False
        covered = self.covered
        if work.source.kind == "frames":
            covered = int(self.memory.pending[len(work.source.covered_ids) - 1].source_id) + 1
        if not self.memory.apply(work.source, text=text, token_count=token_count):
            return False
        self.covered = covered
        return True
