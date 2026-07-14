# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared component construction helpers for the LTX model family."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from .ltx2_transformer import LTX2VideoTransformer3DModel


@dataclass(frozen=True)
class LTXComponentProfile:
    """Component discovery contract for one LTX checkpoint family."""

    name: str
    dit_modules: tuple[str, ...]
    encoder_modules: tuple[str, ...]
    vae_modules: tuple[str, ...]
    resident_modules: tuple[str, ...] = ()


LTX2_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder",),
    vae_modules=("vae", "audio_vae"),
)

LTX23_COMPONENT_PROFILE = LTXComponentProfile(
    name="ltx2_3",
    dit_modules=("transformer",),
    encoder_modules=("text_encoder", "connectors"),
    vae_modules=("vae", "audio_vae"),
    resident_modules=("vocoder",),
)


def load_transformer_config(
    model_path: str,
    subfolder: str = "transformer",
    local_files_only: bool = True,
) -> dict:
    """Load an LTX transformer config from a local model or the HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as config_file:
                return json.load(config_file)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as config_file:
                return json.load(config_file)
        except Exception:
            pass
    return {}


def create_transformer_from_config(
    config: dict,
    quant_config: QuantizationConfig | None = None,
) -> LTX2VideoTransformer3DModel:
    """Construct the shared LTX transformer from a Diffusers config."""
    if not config and quant_config is None:
        return LTX2VideoTransformer3DModel()

    signature = inspect.signature(LTX2VideoTransformer3DModel.__init__)
    allowed_keys = set(signature.parameters)
    kwargs = {key: value for key, value in config.items() if key in allowed_keys}
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    return LTX2VideoTransformer3DModel(**kwargs)
