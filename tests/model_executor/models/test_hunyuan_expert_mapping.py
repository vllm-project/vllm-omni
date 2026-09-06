# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for HunyuanImage-3 ``get_expert_mapping`` contract.

vLLM consumers such as ``get_moe_expert_mapping`` / ``process_packed_modules_mapping``
unpack every mapping entry as a flat 4-tuple
``(param_name, weight_name, expert_id, shard_id)``:

.. code-block:: python

    for _, weight_name, _, _ in get_moe_expert_mapping(model):
        ...

Historically HunyuanImage-3 returned a 2-tuple ``(mapping, remapping)`` which
made the unpack fail with ``ValueError: not enough values to unpack
(expected 4, got 2)`` as soon as a MoE LoRA adapter was loaded. These tests
pin the flat 4-tuple contract for both the AR and the diffusion model.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (  # noqa: E402
    HunyuanImage3Model,
)
from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (  # noqa: E402
    HunyuanModel,
)


def _moe_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_size=64,
        head_dim=16,
    )


def _non_moe_config() -> SimpleNamespace:
    return SimpleNamespace(num_experts=1)


class _FakeHunyuanModel(nn.Module):
    """Minimal stand-in exposing just the members ``get_expert_mapping`` reads."""

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        # e.g. a packed gate/up expert parameter so the mapping is non-empty.
        self.routed_experts = nn.Parameter(torch.zeros(8, 64, 64))
        self.num_redundant_experts = 0


@pytest.mark.parametrize(
    ("model_cls", "module_name"),
    [
        (HunyuanModel, "vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3"),
        (
            HunyuanImage3Model,
            "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer",
        ),
    ],
)
def test_get_expert_mapping_flat_4tuple_contract(
    model_cls: type,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MoE mapping entries must be flat 4-tuples, unpackable by vLLM."""
    module = __import__(module_name, fromlist=["_is_moe"])
    monkeypatch.setattr(module, "_is_moe", lambda config: config.num_experts > 1)

    model = _FakeHunyuanModel(_moe_config())
    mapping = model_cls.get_expert_mapping(model)

    assert isinstance(mapping, list)
    assert mapping, "MoE model should produce a non-empty expert mapping"
    for entry in mapping:
        assert len(entry) == 4, f"expected (param, weight, expert_id, shard_id), got {entry!r}"
        param_name, weight_name, expert_id, shard_id = entry
        assert isinstance(param_name, str)
        assert isinstance(weight_name, str)
        assert isinstance(expert_id, int)
        assert isinstance(shard_id, str)
    # The exact unpack vLLM consumers perform must not raise.
    for _, weight_name, _, _ in mapping:
        assert weight_name


@pytest.mark.parametrize(
    ("model_cls", "module_name"),
    [
        (HunyuanModel, "vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3"),
        (
            HunyuanImage3Model,
            "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer",
        ),
    ],
)
def test_get_expert_mapping_non_moe_returns_empty_list(
    model_cls: type,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-MoE models must return an empty list (not a 2-tuple)."""
    module = __import__(module_name, fromlist=["_is_moe"])
    monkeypatch.setattr(module, "_is_moe", lambda config: config.num_experts > 1)

    model = _FakeHunyuanModel(_non_moe_config())
    mapping = model_cls.get_expert_mapping(model)

    assert mapping == []


def test_mapping_entries_survive_vllm_get_moe_expert_mapping_unpack() -> None:
    """The exact unpack vLLM performs must not raise for the diffusion model."""
    from vllm.model_executor.utils import get_moe_expert_mapping

    module = __import__(
        "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer",
        fromlist=["_is_moe", "HunyuanImage3Model"],
    )
    orig_is_moe = module._is_moe
    module._is_moe = lambda config: config.num_experts > 1
    try:
        model = _FakeHunyuanModel(_moe_config())
        # Bind the real implementation so get_moe_expert_mapping can find it.
        model.get_expert_mapping = lambda: module.HunyuanImage3Model.get_expert_mapping(model)
        for _, weight_name, _, _ in get_moe_expert_mapping(model):
            assert weight_name
    finally:
        module._is_moe = orig_is_moe
