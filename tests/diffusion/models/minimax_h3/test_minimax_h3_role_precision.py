# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-role precision for the H3 wide DiT linears.

The roles (attention qkv/out, MLP fc1/fc2) repeat in every block, so their
precision is expressed through the existing quantization config with a pattern
key rather than an H3-only policy plane. The three-way result is the contract:

- a named role takes the configured format;
- a named role set to ``null`` stays bf16, i.e. its linear is left alone;
- an *unnamed* role returns ``None`` so the caller keeps the arm's own default,
  which is what makes the change inert for every existing configuration.
"""

from __future__ import annotations

import pytest

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import resolve_role_precision
from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeConfig:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def get_min_capability(self) -> int:
        return 80

    def apply_vllm_mapper(self, mapper: object) -> None:
        del mapper


def _per_role() -> ComponentQuantizationConfig:
    return ComponentQuantizationConfig(
        {
            "transformer.*.mlp": _FakeConfig("nvfp4"),
            "transformer.*.attn": _FakeConfig("mxfp8"),
            "transformer.*.mlp.fc2": None,
        },
        default_config=_FakeConfig("mxfp8"),
    )


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("blocks.0.mlp.fc1", "nvfp4"),
        ("blocks.5.mlp.fc1", "nvfp4"),
        ("blocks.0.attn.qkv_proj", "mxfp8"),
        ("blocks.0.attn.out_proj", "mxfp8"),
    ],
)
def test_named_roles_take_the_configured_precision(prefix: str, expected: str) -> None:
    assert resolve_role_precision(_per_role(), prefix) == expected


def test_a_role_named_null_stays_bf16() -> None:
    assert resolve_role_precision(_per_role(), "blocks.0.mlp.fc2") == "bf16"


def test_unnamed_roles_return_none_so_the_caller_keeps_its_default() -> None:
    assert resolve_role_precision(_per_role(), "blocks.0.norm") is None


def test_no_config_leaves_every_role_on_the_default_path() -> None:
    assert resolve_role_precision(None, "blocks.0.mlp.fc1") is None
    assert resolve_role_precision(None, "blocks.0.attn.qkv_proj") is None


def test_a_default_entry_is_not_a_statement_about_a_role() -> None:
    config = ComponentQuantizationConfig({}, default_config=_FakeConfig("mxfp8"))
    assert resolve_role_precision(config, "blocks.0.mlp.fc1") is None


def test_a_component_wide_key_is_explicit() -> None:
    config = ComponentQuantizationConfig({"transformer": _FakeConfig("mxfp8")}, default_config=None)
    assert resolve_role_precision(config, "blocks.0.mlp.fc1") == "mxfp8"


def test_a_plain_global_config_applies() -> None:
    assert resolve_role_precision(_FakeConfig("mxfp8"), "blocks.0.mlp.fc1") == "mxfp8"


def test_an_unexpressible_format_does_not_guess() -> None:
    config = ComponentQuantizationConfig({"transformer.*.mlp": _FakeConfig("awq")}, default_config=None)
    assert resolve_role_precision(config, "blocks.0.mlp.fc1") == "bf16"
