# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validated static stage transitions, independent of data dependencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StageRouting:
    """One ordered route starting at stage 0; omitted stages stay inactive.

    Stage IDs remain indices into engine configuration and sampling arrays.
    Branching, joins and loops require a separate execution contract.
    """

    stage_order: tuple[int, ...]

    @classmethod
    def from_transitions(
        cls, num_stages: int, transitions: tuple[tuple[int, int], ...] | None = None
    ) -> "StageRouting":
        if transitions is None:
            return cls(tuple(range(num_stages)))
        if num_stages < 1:
            raise ValueError("Explicit stage transitions require stage 0")
        if not isinstance(transitions, tuple) or any(
            not isinstance(edge, tuple) or len(edge) != 2 for edge in transitions
        ):
            raise ValueError("Stage transitions must be an immutable tuple of (source, target) tuples")
        successors: dict[int, int] = {}
        predecessors: set[int] = set()
        for source, target in transitions:
            if any(type(stage_id) is not int or not 0 <= stage_id < num_stages for stage_id in (source, target)):
                raise ValueError(f"Transition {source!r} -> {target!r} references an invalid stage ID")
            if source == target:
                raise ValueError(f"Self transition at stage {source}")
            if source in successors:
                raise ValueError(f"Stage {source} has duplicate or branching transitions")
            if target in predecessors:
                raise ValueError(f"Stage {target} has multiple incoming transitions")
            successors[source] = target
            predecessors.add(target)
        order = [0]
        while order[-1] in successors:
            target = successors[order[-1]]
            if target in order:
                raise ValueError("Stage transitions contain a cycle")
            order.append(target)
        if len(order) - 1 != len(successors):
            raise ValueError("Stage transitions contain edges unreachable from stage 0")
        return cls(tuple(order))

    def path_to(self, final_stage_id: int) -> tuple[int, ...]:
        """Resolve a request's route prefix, rejecting unreachable endpoints."""
        if type(final_stage_id) is not int or final_stage_id not in self.stage_order:
            raise ValueError(f"Final stage {final_stage_id!r} is not reachable from stage 0")
        return self.stage_order[: self.stage_order.index(final_stage_id) + 1]

    def next_stage(self, stage_id: int, final_stage_id: int) -> int | None:
        path = self.path_to(final_stage_id)
        if stage_id not in path or stage_id == final_stage_id:
            return None
        return path[path.index(stage_id) + 1]
