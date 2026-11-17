# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Derived context nodes over authoritative duplex history sources."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

from vllm_omni.engine.duplex.session.history_snapshot import DuplexHistorySnapshot, copy_history_message


class DuplexContextLevel(IntEnum):
    """Derived node kinds: MEDIA_DROPPED is the L1 view; SUMMARY stores summaries."""

    MEDIA_DROPPED = 1
    SUMMARY = 2


@dataclass(frozen=True)
class DuplexSourceRef:
    """A source ID and revision scoped to one live session."""

    source_id: int
    revision: int


@dataclass(frozen=True)
class DuplexContextNode:
    """A rendered message derived from history sources and earlier nodes.

    Parents always have smaller node IDs, so ascending node ID order is a
    topological order of the dependency graph. Message containers are owned
    by the ledger; readers must treat them and opaque payloads as read-only.
    """

    node_id: int
    level: DuplexContextLevel
    sources: tuple[DuplexSourceRef, ...]
    parents: tuple[int, ...]
    message: Mapping[str, object]


class DuplexContextNodeRejectedError(ValueError):
    """The ledger refused a node that is not valid against the given snapshot."""


def source_refs(snapshot: DuplexHistorySnapshot, *source_ids: int) -> tuple[DuplexSourceRef, ...]:
    revisions = _current_revisions(snapshot)
    return tuple(DuplexSourceRef(source_id, revisions[source_id]) for source_id in source_ids)


def _current_revisions(snapshot: DuplexHistorySnapshot) -> dict[int, int]:
    return dict(zip(snapshot.source_ids, snapshot.source_revisions, strict=True))


def _sources_current(sources: tuple[DuplexSourceRef, ...], current: Mapping[int, int]) -> bool:
    return all(current.get(ref.source_id) == ref.revision for ref in sources)


class DuplexContextLedger:
    """Session-owned derived nodes that never write to history.

    A node is valid while every recorded source has the recorded revision in
    the snapshot and every parent node is valid. ``refresh`` prunes nodes that
    are no longer valid. Node IDs are never reused.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, DuplexContextNode] = {}
        self._media_dropped_by_source: dict[int, int] = {}
        self._next_node_id = 1

    def add_node(
        self,
        snapshot: DuplexHistorySnapshot,
        *,
        level: DuplexContextLevel,
        sources: tuple[DuplexSourceRef, ...],
        parents: tuple[int, ...] = (),
        message: Mapping[str, object],
    ) -> DuplexContextNode:
        """Admit a node only if it is valid against ``snapshot``.

        Validation does not prune: a snapshot older than the last refresh must
        not discard nodes built from newer revisions or displace their L1 lookup.
        At most one L1 node is admitted per source revision between refreshes.
        """
        return self._add_node(
            _current_revisions(snapshot),
            level=level,
            sources=sources,
            parents=parents,
            message=copy_history_message(dict(message)),
        )

    def add_l1_nodes(
        self,
        snapshot: DuplexHistorySnapshot,
        *,
        messages: Mapping[int, Mapping[str, object]],
    ) -> tuple[DuplexContextNode, ...]:
        """Admit L1 messages in mapping iteration order.

        Unknown sources or existing valid L1 nodes reject the entire batch.
        """
        if not messages:
            return ()
        current = _current_revisions(snapshot)
        if any(source_id not in current for source_id in messages):
            raise DuplexContextNodeRejectedError("unknown_source_id")
        for source_id in messages:
            if self._has_l1_revision(source_id, current[source_id]):
                raise DuplexContextNodeRejectedError("source_already_has_media_dropped_node")
        # Copy the whole batch before writing any node or consuming an ID.
        prepared = tuple((source_id, copy_history_message(dict(message))) for source_id, message in messages.items())
        return tuple(
            self._add_node(
                current,
                level=DuplexContextLevel.MEDIA_DROPPED,
                sources=(DuplexSourceRef(source_id, current[source_id]),),
                message=message,
            )
            for source_id, message in prepared
        )

    def _add_node(
        self,
        current: Mapping[int, int],
        *,
        level: DuplexContextLevel,
        sources: tuple[DuplexSourceRef, ...],
        parents: tuple[int, ...] = (),
        message: Mapping[str, object],
    ) -> DuplexContextNode:
        """Validate and store a message already detached by a public admission API."""
        if not sources and not parents:
            raise DuplexContextNodeRejectedError("node_has_no_dependencies")
        if not _sources_current(sources, current):
            raise DuplexContextNodeRejectedError("stale_source_revision")
        if level is DuplexContextLevel.MEDIA_DROPPED:
            if len(sources) != 1 or parents:
                raise DuplexContextNodeRejectedError("media_dropped_node_needs_exactly_one_source")
            if self._has_l1_revision(sources[0].source_id, sources[0].revision):
                raise DuplexContextNodeRejectedError("source_already_has_media_dropped_node")
        memo: dict[int, bool] = {}
        if not all(self._valid_against(parent, current, memo) for parent in parents):
            raise DuplexContextNodeRejectedError("parent_not_valid")
        node = DuplexContextNode(
            node_id=self._next_node_id,
            level=level,
            sources=tuple(sources),
            parents=tuple(parents),
            message=message,
        )
        self._next_node_id += 1
        self._nodes[node.node_id] = node
        if level is DuplexContextLevel.MEDIA_DROPPED:
            existing = self.l1_node_for(sources[0].source_id)
            if existing is None or existing.sources[0].revision < sources[0].revision:
                self._media_dropped_by_source[sources[0].source_id] = node.node_id
        return node

    def _has_l1_revision(self, source_id: int, revision: int) -> bool:
        existing = self.l1_node_for(source_id)
        if existing is None or existing.sources[0].revision < revision:
            return False
        if existing.sources[0].revision == revision:
            return True
        # Only out-of-order historical admission needs a scan. The synchronous
        # current-snapshot renderer uses the indexed newest revision above.
        return any(
            node.level is DuplexContextLevel.MEDIA_DROPPED and node.sources[0] == DuplexSourceRef(source_id, revision)
            for node in self._nodes.values()
        )

    def refresh(self, snapshot: DuplexHistorySnapshot) -> tuple[int, ...]:
        """Prune against a current authoritative snapshot; return IDs in order."""
        current = _current_revisions(snapshot)
        removed: list[int] = []
        # Insertion order is ascending node ID, so parents are decided first.
        for node in tuple(self._nodes.values()):
            if _sources_current(node.sources, current) and all(parent in self._nodes for parent in node.parents):
                continue
            del self._nodes[node.node_id]
            removed.append(node.node_id)
            if node.level is DuplexContextLevel.MEDIA_DROPPED:
                source_id = node.sources[0].source_id
                if self._media_dropped_by_source.get(source_id) == node.node_id:
                    del self._media_dropped_by_source[source_id]
        return tuple(removed)

    def get(self, node_id: int) -> DuplexContextNode | None:
        """Read a node after the owner has refreshed against current history."""
        return self._nodes.get(node_id)

    def remove(self, *node_ids: int) -> tuple[int, ...]:
        """Retire nodes and their descendants; unrelated nodes remain valid."""
        removed = set(node_ids)
        result = []
        reindex: set[int] = set()
        for node in tuple(self._nodes.values()):
            if node.node_id not in removed and not any(parent in removed for parent in node.parents):
                continue
            removed.add(node.node_id)
            result.append(node.node_id)
            del self._nodes[node.node_id]
            if node.level is DuplexContextLevel.MEDIA_DROPPED:
                source_id = node.sources[0].source_id
                if self._media_dropped_by_source.get(source_id) == node.node_id:
                    del self._media_dropped_by_source[source_id]
                    reindex.add(source_id)
        # Historical admission can leave multiple L1 revisions. Keep lookup
        # and duplicate-admission checks correct after explicitly retiring one.
        for source_id in reindex:
            candidates = [
                node
                for node in self._nodes.values()
                if node.level is DuplexContextLevel.MEDIA_DROPPED and node.sources[0].source_id == source_id
            ]
            if candidates:
                self._media_dropped_by_source[source_id] = max(
                    candidates, key=lambda node: node.sources[0].revision
                ).node_id
        return tuple(result)

    def replace_summary(
        self,
        snapshot: DuplexHistorySnapshot,
        *,
        sources: tuple[DuplexSourceRef, ...] = (),
        replaces: tuple[int, ...] = (),
        message: Mapping[str, object],
    ) -> DuplexContextNode:
        """Publish a rolling summary and retire its previous summary nodes.

        Inherit the full source closure before removing old nodes. The new
        node has no parent edges, so rolling updates retain source validity
        without retaining a chain of obsolete summaries. Validation and
        message copying finish before any mutation. Call on the owner loop
        with its current snapshot; admission never refreshes implicitly.
        """
        current = _current_revisions(snapshot)
        memo: dict[int, bool] = {}
        for node_id in replaces:
            node = self._nodes.get(node_id)
            if node is None or node.level is not DuplexContextLevel.SUMMARY:
                raise DuplexContextNodeRejectedError("replacement_is_not_a_summary")
            if not self._valid_against(node_id, current, memo):
                raise DuplexContextNodeRejectedError("replacement_not_valid")
        dependencies = dict.fromkeys(sources)
        pending = list(replaces)
        visited: set[int] = set()
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self._nodes[node_id]
            dependencies.update(dict.fromkeys(node.sources))
            pending.extend(node.parents)
        node = self.add_node(
            snapshot,
            level=DuplexContextLevel.SUMMARY,
            sources=tuple(dependencies),
            message=message,
        )
        self.remove(*replaces)
        return node

    def valid(self, node_id: int) -> bool:
        """Whether the node survived the latest refresh."""
        return node_id in self._nodes

    def nodes(self) -> tuple[DuplexContextNode, ...]:
        return tuple(self._nodes.values())

    def l1_node_for(self, source_id: int) -> DuplexContextNode | None:
        node_id = self._media_dropped_by_source.get(source_id)
        return None if node_id is None else self._nodes.get(node_id)

    def _valid_against(self, node_id: int, current: Mapping[int, int], memo: dict[int, bool]) -> bool:
        if node_id not in memo:
            node = self._nodes.get(node_id)
            memo[node_id] = (
                node is not None
                and _sources_current(node.sources, current)
                and all(self._valid_against(parent, current, memo) for parent in node.parents)
            )
        return memo[node_id]
