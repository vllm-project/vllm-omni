# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Dynamic candidate decision heads for the frozen DRVEXO Thinker."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

TASKS = ("choice", "noul", "score")
TASK_TO_ID = {task: index for index, task in enumerate(TASKS)}


@dataclass(frozen=True)
class DecisionGroup:
    """One decision with a shared prefix and dynamic candidate suffixes."""

    task: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.task not in TASK_TO_ID:
            raise ValueError(f"Unsupported task: {self.task}")
        if not 2 <= len(self.candidate_ids) <= 151:
            raise ValueError("A decision must contain 2 through 151 candidates")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Candidate IDs must be unique within a decision")


def group_layout(
    groups: Sequence[DecisionGroup],
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Return candidate owners and per-decision task IDs."""

    if not groups:
        raise ValueError("At least one decision group is required")
    owners = torch.repeat_interleave(
        torch.arange(len(groups), device=device, dtype=torch.long),
        torch.tensor(
            [len(group.candidate_ids) for group in groups],
            device=device,
            dtype=torch.long,
        ),
    )
    task_ids = torch.tensor(
        [TASK_TO_ID[group.task] for group in groups],
        device=device,
        dtype=torch.long,
    )
    return owners, task_ids


class GeneralDecisionHead(nn.Module):
    """Task-specific scalar heads over candidate-conditioned hidden states.

    Each candidate is evaluated after the explicit instruction/state prefix.
    This experiment recomputes full prefix+candidate prefills with prefix caching
    disabled. candidate_hidden contains the final normalized prompt token state.
    """

    format_version = 1

    def __init__(self, hidden_size: int = 4096, projection_size: int = 0) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if projection_size < 0:
            raise ValueError("projection_size cannot be negative")
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.input_norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            dtype=torch.float32,
        )
        self.projection = (
            nn.Sequential(
                nn.Linear(hidden_size, projection_size, dtype=torch.float32),
                nn.GELU(),
            )
            if projection_size
            else nn.Identity()
        )
        head_size = projection_size or hidden_size
        self.heads = nn.ModuleDict({task: nn.Linear(head_size, 1, dtype=torch.float32) for task in TASKS})

    def forward(
        self,
        candidate_hidden: Tensor,
        owners: Tensor,
        task_ids: Tensor,
    ) -> Tensor:
        if candidate_hidden.ndim != 2 or candidate_hidden.shape[1] != self.hidden_size:
            raise ValueError(f"candidate_hidden must have shape [candidates, {self.hidden_size}]")
        if owners.ndim != 1 or len(owners) != len(candidate_hidden):
            raise ValueError("owners must align with candidate_hidden")
        if task_ids.ndim != 1 or not len(task_ids):
            raise ValueError("task_ids must contain one task for each decision")
        if owners.dtype != torch.long or task_ids.dtype != torch.long:
            raise TypeError("owners and task_ids must use torch.long")
        if owners.device != candidate_hidden.device or task_ids.device != candidate_hidden.device:
            raise ValueError("All inputs must be on the same device")
        if int(owners.min()) < 0 or int(owners.max()) >= len(task_ids):
            raise ValueError("owners contains an invalid decision index")
        if int(task_ids.min()) < 0 or int(task_ids.max()) >= len(TASKS):
            raise ValueError("task_ids contains an invalid task")

        hidden = self.projection(self.input_norm(candidate_hidden.float()))
        candidate_tasks = task_ids.index_select(0, owners)
        logits = hidden.new_empty(len(hidden))
        for task_id, task in enumerate(TASKS):
            indices = (candidate_tasks == task_id).nonzero(as_tuple=False).flatten()
            if len(indices):
                values = self.heads[task](hidden.index_select(0, indices)).flatten()
                logits.index_copy_(0, indices, values)
        return logits

    def probabilities(
        self,
        candidate_hidden: Tensor,
        groups: Sequence[DecisionGroup],
    ) -> list[dict[str, float]]:
        owners, task_ids = group_layout(groups, candidate_hidden.device)
        if len(owners) != len(candidate_hidden):
            raise ValueError("Group candidate counts do not match candidate_hidden")
        logits = self(candidate_hidden, owners, task_ids)
        output: list[dict[str, float]] = []
        offset = 0
        for group in groups:
            count = len(group.candidate_ids)
            values = logits[offset : offset + count].softmax(0).double().cpu().tolist()
            output.append(dict(zip(group.candidate_ids, values, strict=True)))
            offset += count
        return output

    def loss(
        self,
        candidate_hidden: Tensor,
        groups: Sequence[DecisionGroup],
        targets: Sequence[Sequence[float]],
    ) -> Tensor:
        if len(groups) != len(targets):
            raise ValueError("targets must align with groups")
        owners, task_ids = group_layout(groups, candidate_hidden.device)
        if len(owners) != len(candidate_hidden):
            raise ValueError("Group candidate counts do not match candidate_hidden")
        logits = self(candidate_hidden, owners, task_ids)
        losses = []
        offset = 0
        for group, target_values in zip(groups, targets, strict=True):
            count = len(group.candidate_ids)
            target = torch.as_tensor(
                target_values,
                device=logits.device,
                dtype=torch.float32,
            )
            if target.shape != (count,) or not torch.isfinite(target).all():
                raise ValueError("Each target must be a finite candidate distribution")
            if bool((target < 0).any()) or not torch.isclose(target.sum(), target.new_tensor(1.0), atol=1e-5):
                raise ValueError("Each target must be nonnegative and sum to one")
            losses.append(-(target * logits[offset : offset + count].log_softmax(0)).sum())
            offset += count
        return torch.stack(losses).mean()

    def save_pretrained(
        self,
        output_directory: str | Path,
        *,
        backbone: str,
        backbone_config_sha256: str,
    ) -> None:
        path = Path(output_directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "format": "qwen3_general_decision_head",
            "format_version": self.format_version,
            "hidden_size": self.hidden_size,
            "projection_size": self.projection_size,
            "tasks": list(TASKS),
            "candidate_count": {"minimum": 2, "maximum": 151},
            "backbone": backbone,
            "backbone_config_sha256": backbone_config_sha256,
            "weights_dtype": "float32",
        }
        (path / "decision_head_config.json").write_text(json.dumps(config, indent=2) + "\n")
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()},
            path / "decision_head.safetensors",
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> GeneralDecisionHead:
        path = Path(directory)
        config = json.loads((path / "decision_head_config.json").read_text())
        if config.get("format") != "qwen3_general_decision_head":
            raise ValueError("Unsupported decision head format")
        if config.get("format_version") != cls.format_version:
            raise ValueError("Unsupported decision head format version")
        if tuple(config.get("tasks", ())) != TASKS:
            raise ValueError("Decision head task contract does not match")
        model = cls(
            hidden_size=int(config["hidden_size"]),
            projection_size=int(config.get("projection_size", 0)),
        )
        model.load_state_dict(
            load_file(path / "decision_head.safetensors"),
            strict=True,
        )
        return model


def selected_ids(probabilities: Iterable[dict[str, float]]) -> list[str]:
    """Choose the maximum-probability candidate while preserving input IDs."""

    output = []
    for decision in probabilities:
        if len(decision) < 2:
            raise ValueError("Each decision must contain at least two probabilities")
        output.append(max(decision, key=decision.__getitem__))
    return output
