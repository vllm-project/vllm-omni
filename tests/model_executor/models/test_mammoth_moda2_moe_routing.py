from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    _build_moe_token_routing,
    moe_forward,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _understanding_expert(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states + 10


def _generation_expert(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states * 2


@pytest.mark.parametrize("shape", [(6, 4), (2, 3, 4)])
def test_moe_forward_precomputed_routing_matches_tokenwise_reference(shape):
    hidden_states = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    mask = torch.tensor([False, True, False, True, True, False]).reshape(shape[:-1])
    routing = _build_moe_token_routing(mask)

    output = moe_forward(
        hidden_states,
        _understanding_expert,
        _generation_expert,
        token_routing=routing,
    )

    expected = torch.where(
        mask.unsqueeze(-1),
        _generation_expert(hidden_states),
        _understanding_expert(hidden_states),
    )
    torch.testing.assert_close(output, expected)
    assert output.is_contiguous()


@pytest.mark.parametrize(
    ("mask_value", "expected_expert"),
    [(False, "understanding"), (True, "generation")],
)
def test_moe_forward_precomputed_routing_skips_unused_expert(mask_value, expected_expert):
    calls = {"understanding": 0, "generation": 0}

    def understanding(hidden_states):
        calls["understanding"] += 1
        return hidden_states + 1

    def generation(hidden_states):
        calls["generation"] += 1
        return hidden_states + 2

    hidden_states = torch.ones(8, 4)
    routing = _build_moe_token_routing(torch.full((8,), mask_value))

    output = moe_forward(
        hidden_states,
        understanding,
        generation,
        token_routing=routing,
    )

    assert calls[expected_expert] == 1
    assert calls["generation" if expected_expert == "understanding" else "understanding"] == 0
    expected_offset = 1 if expected_expert == "understanding" else 2
    torch.testing.assert_close(output, hidden_states + expected_offset)


def test_moe_forward_mask_fallback_preserves_existing_api():
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    mask = torch.tensor([False, True, False, True, True, False])

    output = moe_forward(
        hidden_states,
        _understanding_expert,
        _generation_expert,
        mask,
    )

    expected = torch.where(
        mask.unsqueeze(-1),
        _generation_expert(hidden_states),
        _understanding_expert(hidden_states),
    )
    torch.testing.assert_close(output, expected)


def test_moe_forward_rejects_mask_shape_mismatch():
    with pytest.raises(ValueError, match="gen_token_mask shape mismatch"):
        moe_forward(
            torch.ones(4, 8),
            _understanding_expert,
            _generation_expert,
            torch.tensor([True, False]),
        )


def test_moe_forward_rejects_routing_size_mismatch():
    with pytest.raises(ValueError, match="token_routing size mismatch"):
        moe_forward(
            torch.ones(4, 8),
            _understanding_expert,
            _generation_expert,
            token_routing=(torch.tensor([0]), torch.tensor([1])),
        )
