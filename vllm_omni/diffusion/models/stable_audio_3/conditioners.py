# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Conditioners for Stable Audio 3.

PORT_FROM: Stability-AI/stable-audio-3 stable_audio_3/models/conditioners.py
           + the small ExpoFourierFeatures helper from models/blocks.py

Three conditioner classes route text/duration through MultiConditioner
into the 6 conditioning slots consumed by ConditionedDiffusionModelWrapper.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Fourier features (PORT_FROM: models/blocks.py:64-84 ExpoFourierFeatures)
# ---------------------------------------------------------------------------


class ExpoFourierFeatures(nn.Module):
    """Exponentially-spaced Fourier features for scalar conditioning."""

    def __init__(self, dim: int, min_freq: float = 0.5, max_freq: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    def forward(self, t: Tensor) -> Tensor:
        # PORT_FROM: models/blocks.py:71-83
        raise NotImplementedError


class NumberEmbedder(nn.Module):
    """Scalar → Fourier features → linear projection."""

    def __init__(self, features: int, dim: int = 256, fourier_features_type: str = "learned") -> None:
        super().__init__()
        self.features = features
        # PORT_FROM: models/conditioners.py:95-119 NumberEmbedder
        raise NotImplementedError

    def forward(self, x: list[float]) -> Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Padding modes (PORT_FROM: models/conditioners.py:17-21)
# ---------------------------------------------------------------------------


class PaddingMode(str, Enum):
    NONE = "none"
    ZERO = "zero"
    LEARNED = "learned"


# ---------------------------------------------------------------------------
# Conditioner base class (PORT_FROM: models/conditioners.py:23-92)
# ---------------------------------------------------------------------------


class Conditioner(nn.Module):
    """Base: encodes input → (embeddings, mask) with optional projection."""

    def __init__(
        self,
        dim: int,
        output_dim: int,
        project_out: bool = False,
        padding_mode: str = "zero",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.padding_mode = padding_mode

        if project_out or dim != output_dim:
            self.proj_out = nn.Linear(dim, output_dim, bias=False)
        else:
            self.proj_out = nn.Identity()

        if padding_mode == "learned":
            self.padding_embedding = nn.Parameter(torch.zeros(output_dim))


# ---------------------------------------------------------------------------
# NumberConditioner — duration / numeric conditioning
# PORT_FROM: models/conditioners.py:121-155
# ---------------------------------------------------------------------------


class NumberConditioner(Conditioner):
    """Encode a list of floats (e.g. seconds_total) into (embed, mask)."""

    def __init__(
        self,
        output_dim: int,
        min_val: float = 0,
        max_val: float = 1,
        fourier_features_type: str = "learned",
    ) -> None:
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim, fourier_features_type=fourier_features_type)

    def forward(self, floats: list[float], device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        # PORT_FROM: models/conditioners.py:138-155
        raise NotImplementedError


# ---------------------------------------------------------------------------
# T5GemmaConditioner — text encoding via google/t5gemma-b-b-ul2
# PORT_FROM: models/conditioners.py:157-318
# ---------------------------------------------------------------------------


class T5GemmaConditioner(Conditioner):
    """T5Gemma text encoder. Requires transformers >= 5.8.0."""

    T5GEMMA_MODELS = ["google/t5gemma-b-b-ul2"]
    T5GEMMA_MODEL_DIMS = {"google/t5gemma-b-b-ul2": 768}

    def __init__(
        self,
        output_dim: int,
        model_name: str = "google/t5gemma-b-b-ul2",
        max_length: int = 128,
        enable_grad: bool = False,
        project_out: bool = False,
        padding_mode: str = "zero",
        model_path: str | None = None,
        repo_id: str | None = None,
        subfolder: str | None = None,
    ) -> None:
        assert model_name in self.T5GEMMA_MODELS, f"Unknown T5Gemma model: {model_name}"
        super().__init__(
            self.T5GEMMA_MODEL_DIMS[model_name],
            output_dim,
            project_out=project_out,
            padding_mode=padding_mode,
        )
        self.max_length = max_length
        self.enable_grad = enable_grad
        self.model_name = model_name

        # TODO(stable-audio-3): load tokenizer + encoder from HF.
        # PORT_FROM: models/conditioners.py:165-260
        # vllm-omni note: instantiate via transformers.AutoTokenizer +
        # AutoModelForSeq2SeqLM, keep encoder only. Match upstream's
        # max_length=128, padding_mode=zero, enable_grad=False.
        self.tokenizer = None
        self.encoder = None

    @torch.no_grad()
    def forward(self, texts: list[str], device: torch.device | None = None) -> tuple[Tensor, Tensor]:
        # PORT_FROM: models/conditioners.py:262-318
        # Returns (embeddings [B, S, dim], attention_mask [B, S])
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MultiConditioner — routes multiple conditioners by id
# PORT_FROM: factory.py:115-156 + assemble runtime logic
# ---------------------------------------------------------------------------


class MultiConditioner(nn.Module):
    """Run a set of conditioners and return a dict keyed by id.

    Each value is a (embed, mask) pair. The downstream
    ConditionedDiffusionModelWrapper picks values out by id and routes them
    to cross_attn / global / input_concat / etc. slots.
    """

    def __init__(
        self,
        conditioners: dict[str, Conditioner],
        default_keys: dict[str, Any] | None = None,
        pre_encoded_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys or {}
        self.pre_encoded_keys = pre_encoded_keys or []

    def forward(
        self,
        batch_metadata: list[dict[str, Any]],
        device: torch.device | None = None,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        """Iterate batch, run each conditioner on its column, return id → (embed, mask)."""
        # PORT_FROM: factory.py + runtime: see how upstream's
        # generate_cond() in interface/diffusion_cond.py marshals batch dicts
        # into per-conditioner lists.
        raise NotImplementedError
