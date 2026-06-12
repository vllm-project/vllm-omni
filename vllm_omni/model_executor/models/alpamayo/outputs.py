# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo trajectory output type.

The Alpamayo Stage-1 pipeline emits a predicted ego trajectory (64 waypoints of
(x, y, z) + rotation), optionally with the Stage-0 chain-of-thought reasoning
text. :class:`OmniTrajectoryOutput` is the serializable carrier; it is packed
into ``DiffusionOutput.custom_output["trajectory"]`` so it flows through to
``OmniRequestOutput.custom_output`` on the client side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def _to_list(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().tolist()
    return x


@dataclass
class OmniTrajectoryOutput:
    """Predicted trajectory for one request (possibly multiple samples).

    ``traj_xyz``: nested list shaped ``(n_samples, n_waypoints, 3)``.
    ``traj_rot``: nested list shaped ``(n_samples, n_waypoints, 3, 3)`` or None.
    ``reasoning``: Stage-0 chain-of-thought text, if available.
    """

    traj_xyz: list
    traj_rot: list | None = None
    reasoning: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tensors(
        cls,
        traj_xyz: torch.Tensor,
        traj_rot: torch.Tensor | None = None,
        reasoning: str | None = None,
        **meta: Any,
    ) -> OmniTrajectoryOutput:
        xyz = traj_xyz
        if isinstance(xyz, torch.Tensor) and xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)  # (n_waypoints,3) -> (1,n_waypoints,3)
        rot = traj_rot
        if isinstance(rot, torch.Tensor) and rot.dim() == 3:
            rot = rot.unsqueeze(0)
        n_samples = xyz.shape[0] if isinstance(xyz, torch.Tensor) else len(xyz)
        n_waypoints = xyz.shape[1] if isinstance(xyz, torch.Tensor) else len(xyz[0])
        md = {"n_samples": int(n_samples), "n_waypoints": int(n_waypoints)}
        md.update(meta)
        return cls(
            traj_xyz=_to_list(xyz),
            traj_rot=_to_list(rot) if rot is not None else None,
            reasoning=reasoning,
            meta=md,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "traj_xyz": self.traj_xyz,
            "traj_rot": self.traj_rot,
            "reasoning": self.reasoning,
            "meta": self.meta,
        }


__all__ = ["OmniTrajectoryOutput"]
