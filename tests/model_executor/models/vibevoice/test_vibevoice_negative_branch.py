# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract for the VibeVoice negative-Qwen executor wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.vibevoice.negative_branch import (
    VibeVoiceNegativeBranch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStore:
    def __init__(self, name: str = "negative") -> None:
        self.name = name
        self.reset_ids: list[str] = []
        self.free_ids: list[str] = []
        self.steps: dict[str, int] = {}
        self.batch_contexts: int = 0

    def reset(self, request_id: str) -> None:
        self.reset_ids.append(request_id)
        self.steps[request_id] = 0

    @contextmanager
    def append_and_enter_batch(self, request_ids: list[str]):
        self.batch_contexts += 1
        position = torch.tensor([self.steps.get(request_id, 0) for request_id in request_ids])
        for request_id in request_ids:
            self.steps[request_id] = self.steps.get(request_id, 0) + 1
        yield SimpleNamespace(
            position=position,
            sequence_length=max(self.steps[request_id] for request_id in request_ids),
        )

    def free(self, request_id: str) -> None:
        self.free_ids.append(request_id)


class _FailingQwen(nn.Module):
    def forward(self, **_: Any) -> torch.Tensor:
        raise RuntimeError("injected negative Qwen failure")


class _ReusingQwen(nn.Module):
    """Simulates a Qwen whose output storage is overwritten by later forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("output", torch.empty(2, 4))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids, intermediate_tensors
        assert inputs_embeds is not None
        batch = inputs_embeds.shape[0]
        output = self.output[:batch]
        output.copy_(inputs_embeds + positions.reshape(-1, 1).to(inputs_embeds))
        return output


class _FakeQwen(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids, intermediate_tensors
        assert inputs_embeds is not None
        return inputs_embeds + positions.reshape(-1, 1).to(inputs_embeds)


def test_negative_executor_advances_store_and_returns_one_step_hidden() -> None:
    store = _FakeStore()
    branch = VibeVoiceNegativeBranch(
        store=store,  # type: ignore[arg-type]
        language_model=_FakeQwen(),  # type: ignore[arg-type]
        hidden_size=4,
    )
    branch.reset_audio_segment("request-a")
    first = branch.forward_step(
        ["request-a"],
        [torch.ones(1, 4)],
    )
    second = branch.forward_step(
        ["request-a"],
        [torch.ones(1, 4)],
    )
    assert store.reset_ids == ["request-a"]
    assert torch.equal(first[0], torch.ones(1, 4))
    assert torch.equal(second[0], torch.full((1, 4), 2.0))

    branch.free("request-a")
    assert store.free_ids == ["request-a"]


def test_negative_executor_frees_branch_after_model_exception() -> None:
    store = _FakeStore()
    branch = VibeVoiceNegativeBranch(
        store=store,  # type: ignore[arg-type]
        language_model=_FailingQwen(),  # type: ignore[arg-type]
        hidden_size=4,
    )
    branch.reset_audio_segment("request-a")

    with pytest.raises(RuntimeError, match="injected negative Qwen failure"):
        branch.forward_step(["request-a"], [torch.zeros(1, 4)])

    assert store.free_ids == ["request-a"]


def test_negative_executor_advances_multiple_requests_in_one_batched_forward() -> None:
    store = _FakeStore()
    branch = VibeVoiceNegativeBranch(
        store=store,  # type: ignore[arg-type]
        language_model=_FakeQwen(),  # type: ignore[arg-type]
        hidden_size=4,
    )
    branch.reset_audio_segment("request-a")
    branch.reset_audio_segment("request-b")

    first = branch.forward_step(
        ["request-a", "request-b"],
        [torch.ones(1, 4), torch.full((1, 4), 2.0)],
    )
    second = branch.forward_step(
        ["request-b", "request-a"],
        [torch.full((1, 4), 2.0), torch.ones(1, 4)],
    )

    # One shared attention context per logical batch, not one per request.
    assert store.batch_contexts == 2
    assert torch.equal(first[0], torch.ones(1, 4))
    assert torch.equal(first[1], torch.full((1, 4), 2.0))
    assert torch.equal(second[0], torch.full((1, 4), 3.0))
    assert torch.equal(second[1], torch.full((1, 4), 2.0))


def test_negative_executor_owns_the_batch_across_later_forwards() -> None:
    store = _FakeStore()
    branch = VibeVoiceNegativeBranch(
        store=store,  # type: ignore[arg-type]
        language_model=_ReusingQwen(),  # type: ignore[arg-type]
        hidden_size=4,
    )
    branch.reset_audio_segment("request-a")
    branch.reset_audio_segment("request-b")

    conditions = branch.forward_step(
        ["request-a", "request-b"],
        [torch.ones(1, 4), torch.full((1, 4), 2.0)],
    )
    # A later forward overwrites the model's reusable output buffer; the
    # returned conditions must still hold the first batch's values.
    branch.forward_step(
        ["request-a"],
        [torch.full((1, 4), 9.0)],
    )

    assert torch.equal(conditions[0], torch.ones(1, 4))
    assert torch.equal(conditions[1], torch.full((1, 4), 2.0))
    assert conditions[0].data_ptr() != conditions[1].data_ptr()


def test_negative_executor_rejects_unaligned_duplicate_or_bad_embeddings() -> None:
    branch = VibeVoiceNegativeBranch(
        store=_FakeStore(),  # type: ignore[arg-type]
        language_model=_FakeQwen(),  # type: ignore[arg-type]
        hidden_size=4,
    )
    with pytest.raises(ValueError, match="non-empty and aligned"):
        branch.forward_step(["a", "b"], [torch.zeros(1, 4)])
    with pytest.raises(ValueError, match="duplicate request IDs"):
        branch.forward_step(
            ["a", "a"],
            [torch.zeros(1, 4), torch.zeros(1, 4)],
        )
    with pytest.raises(ValueError, match=r"shape \(1, 4\)"):
        branch.forward_step(["a"], [torch.zeros(2, 4)])
    with pytest.raises(TypeError, match="floating-point"):
        branch.forward_step(["a"], [torch.zeros(1, 4, dtype=torch.long)])


def test_negative_executor_requires_canonical_store_name() -> None:
    with pytest.raises(ValueError, match="called 'negative'"):
        VibeVoiceNegativeBranch(
            store=_FakeStore("other"),  # type: ignore[arg-type]
            language_model=_FakeQwen(),  # type: ignore[arg-type]
            hidden_size=4,
        )
