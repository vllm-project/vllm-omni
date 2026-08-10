# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Conditioners for Stable Audio 3.

PORT_FROM: Stability-AI/stable-audio-3
  - models/conditioners.py (318 lines, full file)
  - models/blocks.py ExpoFourierFeatures (lines 52-84)

Three conditioner classes route text/duration through MultiConditioner
into the 6 conditioning slots consumed by ConditionedDiffusionModelWrapper.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from enum import Enum
from math import pi
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Padding modes (PORT_FROM: conditioners.py:17-21)
# ---------------------------------------------------------------------------


class PaddingMode(str, Enum):
    """Padding handling mode for text conditioner embeddings."""

    NONE = "none"
    ZERO = "zero"
    LEARNED = "learned"


# ---------------------------------------------------------------------------
# Conditioner base class (PORT_FROM: conditioners.py:23-71)
# ---------------------------------------------------------------------------


class Conditioner(nn.Module):
    """Base class: encodes input → (embeddings, mask) with optional projection."""

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

        # Project only if dimensions differ OR project_out is forced on (upstream behavior).
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

        # Learned padding embedding (only created if used).
        if padding_mode == "learned" or padding_mode == PaddingMode.LEARNED:
            self.padding_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def apply_padding(self, embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        """Apply padding handling based on padding_mode.

        PORT_FROM: conditioners.py:41-68 (verbatim).
        """
        mode = self.padding_mode
        if isinstance(mode, str):
            mode = PaddingMode(mode)

        if mode == PaddingMode.NONE:
            return embeddings
        if mode == PaddingMode.ZERO:
            return embeddings * attention_mask.unsqueeze(-1).float()
        if mode == PaddingMode.LEARNED:
            mask_expanded = attention_mask.unsqueeze(-1).bool()
            return torch.where(
                mask_expanded,
                embeddings,
                self.padding_embedding.unsqueeze(0).unsqueeze(0).expand_as(embeddings),
            )
        raise ValueError(f"Unknown padding mode: {mode}")

    def forward(self, x: Any) -> Any:
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------


class ExpoFourierFeatures(nn.Module):
    """Exponentially-spaced Fourier features for scalar conditioning.

    PORT_FROM: blocks.py:52-84 (verbatim).
    """

    def __init__(self, dim: int, min_freq: float = 0.5, max_freq: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.max_freq = max_freq

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, t: Tensor) -> Tensor:
        """t: [B] tensor → [B, dim] Fourier embedding."""
        in_dtype = t.dtype
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2

        # Frequencies in FP32 for stability
        ramp = torch.linspace(0, 1, half_dim, device=t.device, dtype=torch.float32)
        log_min = math.log(self.min_freq)
        log_max = math.log(self.max_freq)
        freqs = torch.exp(ramp * (log_max - log_min) + log_min)

        args = t * freqs * 2 * math.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding.to(in_dtype)


class LearnedPositionalEmbedding(nn.Module):
    """Learned-freq Fourier features (used by TimePositionalEmbedding).

    PORT_FROM: conditioners.py:73-85 (verbatim).
    """

    def __init__(self, dim: int, std: float = 16.0) -> None:
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim) * std)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    """Factory: LearnedPositionalEmbedding + Linear.

    PORT_FROM: conditioners.py:88-92 (verbatim).
    """
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


# ---------------------------------------------------------------------------
# NumberEmbedder (PORT_FROM: conditioners.py:95-119)
# ---------------------------------------------------------------------------


class NumberEmbedder(nn.Module):
    """Scalar → Fourier features → linear projection."""

    def __init__(
        self,
        features: int,
        dim: int = 256,
        fourier_features_type: str = "learned",
    ) -> None:
        super().__init__()
        self.features = features
        if fourier_features_type == "expo":
            self.embedding = nn.Sequential(
                ExpoFourierFeatures(dim=dim),
                nn.Linear(in_features=dim, out_features=features),
            )
        else:
            self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: list[float] | Tensor) -> Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        return embedding.view(*shape, self.features)


# ---------------------------------------------------------------------------
# NumberConditioner (PORT_FROM: conditioners.py:121-155)
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
        self.embedder = NumberEmbedder(
            features=output_dim,
            fourier_features_type=fourier_features_type,
        )

    def forward(self, floats: list[float], device: torch.device | None = None) -> list[Tensor]:
        """PORT_FROM: conditioners.py:138-155 (verbatim).

        Returns [embed, mask] with embed shape [B, 1, D] and mask [B, 1].
        """
        self.embedder.to(device)
        floats = [float(x) for x in floats]
        floats = torch.tensor(floats).to(device)
        floats = floats.clamp(self.min_val, self.max_val)
        normalized = (floats - self.min_val) / (self.max_val - self.min_val)

        # Cast to embedder dtype
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized = normalized.to(embedder_dtype)

        float_embeds = self.embedder(normalized).unsqueeze(1)
        return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]


# ---------------------------------------------------------------------------
# T5GemmaConditioner (PORT_FROM: conditioners.py:157-272)
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

        load_from = model_path or repo_id or model_name
        self.max_length = max_length
        self.enable_grad = enable_grad
        self.model_name = model_name

        # Silence HF download progress + transformers logging while loading.
        # PORT_FROM: conditioners.py:185-220 (verbatim).
        prev_hf_hub = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        prev_transformers = os.environ.get("TRANSFORMERS_VERBOSITY")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from transformers import (
                    AutoConfig,
                    AutoTokenizer,
                    T5GemmaEncoderModel,
                )

                hf_kwargs = {"subfolder": subfolder} if subfolder else {}
                self.tokenizer = AutoTokenizer.from_pretrained(load_from, **hf_kwargs)
                config = AutoConfig.from_pretrained(load_from, **hf_kwargs)
                config.is_encoder_decoder = False
                model = (
                    T5GemmaEncoderModel.from_pretrained(load_from, config=config, **hf_kwargs)
                    .train(enable_grad)
                    .requires_grad_(enable_grad)
                )
            finally:
                logging.disable(previous_level)
                if prev_hf_hub is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_hf_hub
                if prev_transformers is None:
                    os.environ.pop("TRANSFORMERS_VERBOSITY", None)
                else:
                    os.environ["TRANSFORMERS_VERBOSITY"] = prev_transformers

        # Hide model from nn.Module's parameter registration when frozen
        # (upstream trick to avoid optimizer touching it).
        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

        self._device_initialized = False

    def forward(
        self,
        inputs: list[str] | list[dict[str, Tensor]],
        device: torch.device | str,
    ) -> tuple[Tensor, Tensor]:
        """Encode strings or pre-tokenized dicts → (embeddings, attention_mask).

        PORT_FROM: conditioners.py:244-272 (verbatim).
        """
        if not self._device_initialized:
            self.model.to(device)
            self.proj_out.to(device)
            self.model.eval()
            self._device_initialized = True

        # Accept pre-tokenized inputs (from DataLoader) or raw strings.
        if isinstance(inputs[0], dict):
            input_ids = torch.stack([x["input_ids"] for x in inputs]).to(device, non_blocking=True)
            attention_mask = (
                torch.stack([x["attention_mask"] for x in inputs]).to(device, non_blocking=True).to(torch.bool)
            )
        else:
            encoded = self.tokenizer(
                inputs,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            embeddings = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"]

        # Cast embeddings to proj_out dtype if a real projection is present.
        if not isinstance(self.proj_out, nn.Identity):
            proj_out_dtype = next(self.proj_out.parameters()).dtype
            embeddings = embeddings.to(proj_out_dtype)

        embeddings = self.proj_out(embeddings)
        embeddings = self.apply_padding(embeddings, attention_mask)

        return embeddings, attention_mask


# ---------------------------------------------------------------------------
# MultiConditioner (PORT_FROM: conditioners.py:274-318)
# ---------------------------------------------------------------------------


class MultiConditioner(nn.Module):
    """Run a set of conditioners on batch metadata and return a dict keyed by id."""

    def __init__(
        self,
        conditioners: dict[str, Conditioner],
        default_keys: dict[str, str] | None = None,
        pre_encoded_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys or {}
        self.pre_encoded_keys = pre_encoded_keys or []

    def forward(
        self,
        batch_metadata: list[dict[str, Any]],
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """PORT_FROM: conditioners.py:289-318 (verbatim)."""
        output: dict[str, Any] = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key
            conditioner_inputs = []

            for x in batch_metadata:
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(
                            f"Conditioner key {condition_key} not found in batch metadata",
                        )

                # Unwrap single-element list/tuple from collate functions
                value = x[condition_key]
                if isinstance(value, list) or (isinstance(value, tuple) and len(value) == 1):
                    conditioner_input = value[0]
                else:
                    conditioner_input = value

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output
