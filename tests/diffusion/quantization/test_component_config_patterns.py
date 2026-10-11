# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer-role patterns in the per-component quantization config.

A per-component config keys on module prefixes, which cannot express a role that
repeats at every depth (``blocks.0.mlp.fc1``, ``blocks.1.mlp.fc1``, ...). Pattern
keys extend the same abstraction instead of adding a second configuration plane:

- a key may be a shell-style pattern, matched against the layer prefix and each of
  its dotted ancestors, so ``transformer.*.mlp`` covers every layer under an MLP;
- the deepest matching ancestor wins, then the longest pattern, so a narrow role
  beats a broad one;
- patterns are consulted before plain prefixes, because a pattern refines a
  component;
- ``matches()`` reports whether a key named the layer at all, so a *default* entry
  is never mistaken for a statement about a role.
"""

from __future__ import annotations

import pytest

from vllm_omni.quantization.component_config import ComponentQuantizationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeConfig:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_name(self) -> str:
        return self._name

    def get_min_capability(self) -> int:
        return 80

    def apply_vllm_mapper(self, mapper: object) -> None:
        del mapper


def _name(config: object) -> str | None:
    if config is None:
        return None
    return getattr(config, "get_name")()


def test_pattern_key_covers_every_depth() -> None:
    config = ComponentQuantizationConfig(
        {"transformer": _FakeConfig("mxfp8"), "transformer.*.mlp": _FakeConfig("nvfp4")},
        default_config=_FakeConfig("mxfp8"),
    )
    assert _name(config.resolve("transformer.blocks.0.mlp.fc1")) == "nvfp4"
    assert _name(config.resolve("transformer.blocks.7.mlp.fc2")) == "nvfp4"
    # A role nobody named keeps falling to the plain component prefix.
    assert _name(config.resolve("transformer.blocks.0.attn.qkv_proj")) == "mxfp8"


def test_the_deepest_matching_ancestor_wins() -> None:
    config = ComponentQuantizationConfig(
        {"*.mlp": _FakeConfig("mxfp8"), "*.mlp.fc1": _FakeConfig("nvfp4")},
        default_config=None,
    )
    # fc1 matches a pattern at the prefix itself; fc2 only matches one level up.
    assert _name(config.resolve("blocks.0.mlp.fc1")) == "nvfp4"
    assert _name(config.resolve("blocks.0.mlp.fc2")) == "mxfp8"


def test_a_plain_prefix_does_not_override_a_matching_role_pattern() -> None:
    # Patterns are consulted as a set before plain prefixes. That is what lets a role
    # pattern refine a broad component key ("transformer" covers everything below it, so
    # a depth-first rule would make per-role keys unreachable). Pinned deliberately.
    config = ComponentQuantizationConfig(
        {"transformer": _FakeConfig("mxfp8"), "transformer.*.mlp": _FakeConfig("nvfp4")},
        default_config=None,
    )
    assert _name(config.resolve("transformer.blocks.0.mlp.fc1")) == "nvfp4"
    assert _name(config.resolve("transformer.blocks.0.attn.qkv_proj")) == "mxfp8"


def test_plain_prefixes_still_apply_when_no_pattern_matches() -> None:
    config = ComponentQuantizationConfig(
        {"transformer.blocks.0.mlp.fc1": _FakeConfig("nvfp4"), "transformer": _FakeConfig("mxfp8")},
        default_config=None,
    )
    assert _name(config.resolve("transformer.blocks.0.mlp.fc1")) == "nvfp4"
    assert _name(config.resolve("transformer.blocks.1.mlp.fc1")) == "mxfp8"


def test_a_wildcard_spans_dots() -> None:
    # fnmatch semantics: `transformer.*` covers everything below transformer.
    config = ComponentQuantizationConfig({"transformer.*": _FakeConfig("mxfp8")}, default_config=None)
    assert _name(config.resolve("transformer.blocks.9.mlp.fc1")) == "mxfp8"


def test_longest_pattern_wins_at_equal_depth() -> None:
    config = ComponentQuantizationConfig(
        {"*.mlp": _FakeConfig("mxfp8"), "*.mlp.fc1": _FakeConfig("nvfp4")},
        default_config=None,
    )
    assert _name(config.resolve("blocks.0.mlp.fc1")) == "nvfp4"
    assert _name(config.resolve("blocks.0.mlp.fc2")) == "mxfp8"


def test_pattern_takes_precedence_over_plain_prefix() -> None:
    config = ComponentQuantizationConfig(
        {"transformer": _FakeConfig("mxfp8"), "*.mlp": _FakeConfig("nvfp4")},
        default_config=None,
    )
    assert _name(config.resolve("transformer.blocks.0.mlp.fc1")) == "nvfp4"
    assert _name(config.resolve("transformer.blocks.0.attn.qkv_proj")) == "mxfp8"


def test_an_explicit_null_stays_null() -> None:
    config = ComponentQuantizationConfig({"*.mlp": None}, default_config=_FakeConfig("mxfp8"))
    assert config.resolve("blocks.0.mlp.fc1") is None
    assert config.matches("blocks.0.mlp.fc1") is True


def test_matches_separates_a_named_role_from_the_default() -> None:
    config = ComponentQuantizationConfig({"*.mlp": _FakeConfig("nvfp4")}, default_config=_FakeConfig("mxfp8"))
    # Named by a pattern.
    assert config.matches("blocks.0.mlp.fc1") is True
    # Resolves to the same value the default would give, but is not named.
    assert config.matches("blocks.0.attn.qkv_proj") is False
    assert _name(config.resolve("blocks.0.attn.qkv_proj")) == "mxfp8"


def test_a_key_that_is_not_a_valid_pattern_does_not_raise() -> None:
    config = ComponentQuantizationConfig({"weird[": _FakeConfig("mxfp8")}, default_config=None)
    assert _name(config.resolve("weird[.layer")) == "mxfp8"


def test_plain_prefix_config_behaviour_is_unchanged() -> None:
    config = ComponentQuantizationConfig(
        {"transformer": _FakeConfig("fp8"), "vae": None},
        default_config=None,
    )
    assert _name(config.resolve("transformer.blocks.0.attn.to_q")) == "fp8"
    assert config.resolve("vae.encoder.conv_in") is None
    assert config.resolve("audio.encoder.conv_in") is None
