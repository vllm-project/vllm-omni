# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the quantization dispatch check.

The tool exists because a mismatch between a checkpoint's exclusion list and a
model's prefixes does not raise at construction. These tests drive each way the
two can disagree, since a checker that reports agreement no matter what would
be worse than no checker at all.
"""

from __future__ import annotations

import json

import pytest

from vllm_omni.quantization.tools.check_quantized_dispatch import (
    checkpoint_linear_modules,
    checkpoint_quantized_modules,
    compare,
    format_report,
    load_index,
    summarise_checkpoint,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _index(quantized=("a.q", "a.k"), plain=("a.norm", "b.ffn")):
    """A checkpoint where *quantized* modules carry scales and *plain* do not."""
    names: dict[str, str] = {}
    for module in quantized:
        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale", "bias"):
            names[f"{module}.{suffix}"] = "shard-0"
    for module in plain:
        for suffix in ("weight", "bias"):
            names[f"{module}.{suffix}"] = "shard-0"
    return names


def test_a_module_is_quantized_exactly_when_it_carries_scales():
    index = _index()
    assert checkpoint_quantized_modules(index) == {"a.q", "a.k"}
    assert checkpoint_linear_modules(index) == {"a.q", "a.k", "a.norm", "b.ffn"}


def test_a_bias_or_weight_alone_is_not_mistaken_for_a_scale():
    assert checkpoint_quantized_modules({"a.q.weight": "s", "a.q.bias": "s"}) == set()


def test_agreement_is_reported_when_the_two_match():
    index = _index()
    report = compare(
        checkpoint_index=index,
        model_quantized={"a.q", "a.k"},
        model_unquantized={"a.norm", "b.ffn"},
    )
    assert report.agrees
    assert report.quantized_in_both == {"a.q", "a.k"}
    assert "VERDICT: dispatch matches the checkpoint" in format_report(report)


def test_an_exclusion_that_matched_too_little_is_caught():
    """The model quantized a layer the checkpoint left alone.

    This is the shape of a prefix-rooting bug: every exclusion pattern missed,
    so layers meant to stay full-width were built quantized.
    """
    report = compare(
        checkpoint_index=_index(),
        model_quantized={"a.q", "a.k", "a.norm"},
        model_unquantized={"b.ffn"},
    )
    assert report.quantized_without_scales == {"a.norm"}
    assert not report.agrees
    text = format_report(report)
    assert "QUANTIZED BUT THE CHECKPOINT HAS NO SCALES" in text
    assert "a.norm" in text


def test_an_exclusion_that_matched_too_much_is_caught():
    """The checkpoint quantized a layer the model built full-width.

    Its scale tensors then have nowhere to load, which surfaces far from the
    cause unless something says so here.
    """
    report = compare(
        checkpoint_index=_index(),
        model_quantized={"a.q"},
        model_unquantized={"a.k", "a.norm", "b.ffn"},
    )
    assert report.scales_without_a_quantized_layer == {"a.k"}
    assert "CHECKPOINT SCALES WITH NO QUANTIZED LAYER" in format_report(report)


def test_a_fused_projection_shows_up_on_both_sides_until_it_is_ignored():
    """Fusion is a legitimate mismatch, so it must be declared, not assumed.

    A fused qkv has no checkpoint counterpart and the checkpoint's q/k/v have
    no model counterpart. Silently tolerating that would also hide a genuinely
    missing layer, so the caller names it.
    """
    index = _index(quantized=("a.q", "a.k", "a.v"), plain=())
    noisy = compare(checkpoint_index=index, model_quantized={"a.qkv"}, model_unquantized=set())
    assert noisy.missing_from_model == {"a.q", "a.k", "a.v"}
    assert noisy.missing_from_checkpoint == {"a.qkv"}

    quiet = compare(
        checkpoint_index=index,
        model_quantized={"a.qkv"},
        model_unquantized=set(),
        ignore=("a.q", "a.k", "a.v", "a.qkv"),
    )
    assert quiet.agrees


def test_ignore_patterns_accept_wildcards():
    index = _index(quantized=("blocks.0.q", "blocks.1.q"), plain=())
    report = compare(
        checkpoint_index=index,
        model_quantized=set(),
        model_unquantized=set(),
        ignore=("blocks.*",),
    )
    assert report.agrees


def test_the_report_leads_with_disagreement_not_with_totals():
    report = compare(
        checkpoint_index=_index(),
        model_quantized={"a.q", "a.k", "a.norm"},
        model_unquantized={"b.ffn"},
    )
    text = format_report(report)
    assert text.index("QUANTIZED BUT") < text.index("agreed:")
    assert "VERDICT: DISAGREEMENT" in text


def test_an_index_is_read_from_disk(tmp_path):
    index = _index()
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}), encoding="utf-8")
    assert load_index(tmp_path) == index


def test_a_directory_without_a_checkpoint_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="neither a safetensors index nor any shard"):
        load_index(tmp_path)


def test_summarising_a_checkpoint_counts_both_kinds():
    text = summarise_checkpoint(_index())
    assert "quantized:            2" in text
    assert "unquantized:          2" in text
