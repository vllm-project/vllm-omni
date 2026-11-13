# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen source bindings for the shared visual memory store.

Session retains image URLs. Only bounded batches are decoded in preparation
workers. Published visual memory refers to source message positions, not an
invented video clock; input_image may represent a photo or an extracted frame.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO

import pybase64
from PIL import Image
from vllm.multimodal.image import open_image

from vllm_omni.engine.duplex.session.context_ledger import (
    DuplexContextLedger,
    DuplexContextNodeRejectedError,
    DuplexSourceRef,
    source_refs,
)
from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot
from vllm_omni.engine.duplex.visual_memory import VisualMemory, VisualMemorySource, VisualSummary


@dataclass(frozen=True)
class VisualInput:
    message_index: int
    part_index: int
    url: str
    source: DuplexSourceRef


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
    omitted: set[int] = field(default_factory=set)
    # This Qwen instance uses VisualMemory for frame/batch metadata. Text is read
    # exclusively from these Session ledger nodes, keyed by last frame ID.
    nodes: dict[int, int] = field(default_factory=dict)
    snapshot: DuplexHistorySnapshot | None = None
    ledger: DuplexContextLedger | None = None

    def reconcile(
        self, history: Sequence[Mapping[str, object]], snapshot: DuplexHistorySnapshot, ledger: DuplexContextLedger
    ) -> None:
        self.snapshot, self.ledger = snapshot, ledger
        refs = source_refs(snapshot, *snapshot.source_ids)
        inputs = tuple(
            VisualInput(index, part_index, part["image_url"]["url"], refs[index])
            for index, message in enumerate(history)
            if isinstance(message.get("content"), list)
            for part_index, part in enumerate(message["content"])
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        # A reply placeholder is invisible until generation/playback materializes
        # it. That can move unchanged camera sources to later history positions.
        # Published memory depends on source identity/revision, not list offsets.
        sources_changed = len(inputs) < len(self.inputs) or any(
            old.source != new.source or old.part_index != new.part_index or old.url != new.url
            for old, new in zip(self.inputs, inputs)
        )
        if sources_changed or any(not ledger.valid(node) for node in self.nodes.values()):
            ledger.remove(*self.nodes.values())
            self.nodes.clear()
            self.omitted.clear()
            self.memory.clear()
            self.admitted = self.covered = 0
        self.inputs = inputs

    def omitted_image(self, message_index: int, part_index: int) -> bool:
        return any(
            self.inputs[index].message_index == message_index and self.inputs[index].part_index == part_index
            for index in self.omitted
        )

    def covered_image(self, message_index: int, part_index: int) -> bool:
        return not self.omitted_image(message_index, part_index) and any(
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
            self.ledger is not None
            and all(self.ledger.valid(node) for node in self.nodes.values())
            and work.generation == self.memory.generation
            and self.inputs[: len(work.inputs)] == work.inputs
            and (self.memory.matches(work.source) if work.source is not None else self.admitted == work.start)
        )

    def resolved(self, items: Sequence[VisualSummary]) -> tuple[VisualSummary, ...]:
        """Detach ledger text for one worker; Qwen's store holds metadata."""
        assert self.ledger is not None
        result = []
        for item in items:
            node = self.ledger.get(self.nodes[item.last_id])
            if node is None:
                raise ValueError("Visual summary was invalidated before rendering")
            result.append(replace(item, text=str(node.message["content"])))
        return tuple(result)

    def summaries(self) -> tuple[str, ...]:
        if not self.nodes:
            return ()
        return tuple(
            f"Visual observations from source messages "
            f"{self.inputs[int(item.first_source_id)].message_index + 1}–"
            f"{self.inputs[int(item.last_source_id)].message_index + 1}:\n{item.text}"
            for item in self.resolved([*self.memory.long_term, *self.memory.mid_term])
        )

    def apply(self, work: VisualWork, *, text: str, token_count: int) -> bool:
        if work.source is None or not self.matches(work):
            return False
        assert self.ledger is not None and self.snapshot is not None
        count = len(work.source.covered_ids)
        covered = self.covered
        sources: tuple[DuplexSourceRef, ...] = ()
        replaces: tuple[int, ...] = ()
        if work.source.kind == "frames":
            frames = self.memory.pending[:count]
            sources = tuple(dict.fromkeys(self.inputs[int(frame.source_id)].source for frame in frames))
            covered = int(frames[-1].source_id) + 1
        else:
            replaces = tuple(self.nodes[item.last_id] for item in self.memory.mid_term[:count])
        # Stage only list/cursor changes; immutable frames keep the same encoded
        # payloads. Either owner's validation can reject without retiring evidence.
        candidate = replace(
            self.memory,
            pending=list(self.memory.pending),
            mid_term=list(self.memory.mid_term),
            long_term=list(self.memory.long_term),
        )
        if not candidate.apply(work.source, text=text, token_count=token_count):
            return False
        try:
            node = self.ledger.replace_summary(
                self.snapshot, sources=sources, replaces=replaces, message={"role": "system", "content": text}
            )
        except DuplexContextNodeRejectedError:
            return False
        # No yield between successful ledger publication and adopting the staged
        # visual state. The original store remains intact on either rejection.
        self.memory = candidate
        target = self.memory.mid_term if work.source.kind == "frames" else self.memory.long_term
        published = target[-1]
        target[-1] = replace(published, text="")
        # Long-memory caps and merges retire only this model's old nodes.
        live = {item.last_id for item in [*self.memory.mid_term, *self.memory.long_term]}
        retired = [node_id for last_id, node_id in self.nodes.items() if last_id not in live or node_id in replaces]
        self.ledger.remove(*retired)
        self.nodes = {last_id: node_id for last_id, node_id in self.nodes.items() if node_id not in retired}
        self.nodes[published.last_id] = node.node_id
        self.covered = covered
        return True
