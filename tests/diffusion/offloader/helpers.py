# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared model fixtures for diffusion offloader tests."""

from contextlib import contextmanager

import torch
from torch import nn


class DummyStream:
    def wait_stream(self, _stream) -> None:
        return None

    def wait_event(self, _event) -> None:
        return None


class DummyEvent:
    def record(self, _stream) -> None:
        return None

    def synchronize(self) -> None:
        return None


@contextmanager
def dummy_stream(_stream):
    yield None


def patch_offload_runtime(monkeypatch, platform, *, synchronize: bool = False) -> None:
    monkeypatch.setattr(platform, "Stream", DummyStream)
    monkeypatch.setattr(platform, "Event", DummyEvent)
    monkeypatch.setattr(platform, "current_stream", lambda: DummyStream())
    monkeypatch.setattr(platform, "stream", dummy_stream)
    if synchronize:
        monkeypatch.setattr(platform, "synchronize", lambda: None)
        monkeypatch.setattr(platform, "empty_cache", lambda: None)


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))


class _SingleBlockModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(self, num_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(num_blocks)])


class _StagedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Module()
        self.vision.blocks = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.text_model = nn.Module()
        self.text_model.layers = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.load_calls = 0
        self.offload_calls = 0
        self.to_calls = 0

    def load_to_device(self):
        self.load_calls += 1

    def offload_to_cpu(self):
        self.offload_calls += 1
        for hook in getattr(self, "_omni_layerwise_hooks", []):
            hook.offload_layer()

    def to(self, *args, **kwargs):
        self.to_calls += 1
        return super().to(*args, **kwargs)


class _StagedVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.offload_calls = 0
        self.to_calls = 0

    def load_to_device(self):
        return None

    def offload_to_cpu(self):
        self.offload_calls += 1
        return self.to("cpu")

    def to(self, *args, **kwargs):
        self.to_calls += 1
        return super().to(*args, **kwargs)


class _PlainEncoder(nn.Module):
    """Standard encoder with no offload-specific lifecycle methods."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([_DummyBlock(), _DummyBlock()])
        self.final_norm = nn.Linear(2, 2)
