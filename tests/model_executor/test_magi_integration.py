# SPDX-License-Identifier: Apache-2.0
"""Tests for optional MagiCompiler hooks on Qwen2 decoder layers."""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest
import torch.nn as nn

from vllm_omni.model_executor.magi_integration import (
    apply_magi_to_bigvgan_amp_blocks,
    apply_magi_to_dit_decoder_layers,
    apply_magi_to_qwen_decoder_layers,
    is_magi_compile_requested,
)


class _FakeDecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual=None):
        return hidden_states, residual


class _FakeInner(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer() for _ in range(n)])


class _FakeLanguageModel(nn.Module):
    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.model = _FakeInner(n_layers)


class _FakeDiTBlock(nn.Module):
    def forward(self, hidden_states, timestep, position_embeddings=None, block_diff=None):
        return hidden_states


class _FakeDiT(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_FakeDiTBlock() for _ in range(n)])


class _FakeAmp(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _FakeBigVGAN(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.resblocks = nn.ModuleList([_FakeAmp() for _ in range(n)])


def test_is_magi_compile_requested_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MAGI_COMPILE", raising=False)
    assert is_magi_compile_requested() is False


@pytest.mark.parametrize("val", ("1", "true", "yes", "on", "TRUE"))
def test_is_magi_compile_requested_on(monkeypatch, val):
    monkeypatch.setenv("VLLM_OMNI_MAGI_COMPILE", val)
    assert is_magi_compile_requested() is True


def test_apply_magi_noop_when_env_not_set(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MAGI_COMPILE", raising=False)
    lm = _FakeLanguageModel(3)
    assert apply_magi_to_qwen_decoder_layers(lm) == 0


@patch("magi_compiler.magi_compile")
def test_apply_magi_wraps_each_layer_when_env_set(mock_magi, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MAGI_COMPILE", "1")
    mock_magi.side_effect = lambda layer, **kwargs: layer

    lm = _FakeLanguageModel(3)
    n = apply_magi_to_qwen_decoder_layers(lm, model_tag_prefix="test_prefix")

    assert n == 3
    assert mock_magi.call_count == 3
    assert mock_magi.call_args_list[0].kwargs["model_tag"] == "test_prefix_layer0"
    assert mock_magi.call_args_list[0].kwargs["dynamic_arg_dims"]["positions"] == -1


def test_apply_magi_returns_zero_when_magi_not_installed(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MAGI_COMPILE", "1")
    lm = _FakeLanguageModel(1)

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "magi_compiler":
            raise ImportError("magi_compiler not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert apply_magi_to_qwen_decoder_layers(lm) == 0


def test_apply_magi_dit_noop_when_env_not_set(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MAGI_COMPILE", raising=False)
    dit = _FakeDiT(2)
    assert apply_magi_to_dit_decoder_layers(dit) == 0


@patch("magi_compiler.magi_compile")
def test_apply_magi_dit_wraps_blocks(mock_magi, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MAGI_COMPILE", "1")
    mock_magi.side_effect = lambda block, **kwargs: block

    dit = _FakeDiT(2)
    n = apply_magi_to_dit_decoder_layers(dit, model_tag_prefix="dit_prefix")
    assert n == 2
    assert mock_magi.call_count == 2
    d = mock_magi.call_args_list[0].kwargs["dynamic_arg_dims"]
    assert d["hidden_states"] == 1
    assert d["block_diff"] == [2, 3]


def test_apply_magi_bigvgan_noop_when_env_not_set(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_MAGI_COMPILE", raising=False)
    bg = _FakeBigVGAN(3)
    assert apply_magi_to_bigvgan_amp_blocks(bg) == 0


@patch("magi_compiler.magi_compile")
def test_apply_magi_bigvgan_wraps_resblocks(mock_magi, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_MAGI_COMPILE", "1")
    mock_magi.side_effect = lambda block, **kwargs: block

    bg = _FakeBigVGAN(3)
    n = apply_magi_to_bigvgan_amp_blocks(bg, model_tag_prefix="bg_prefix")
    assert n == 3
    assert mock_magi.call_count == 3
    assert mock_magi.call_args_list[0].kwargs["dynamic_arg_dims"]["hidden_states"] == 2
