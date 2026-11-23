# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/config.py and
# haru_vllm/haru_vllm/configuration.py @ d3612b0.
"""Haru-4B configuration.

``HaruConfig`` is the Hugging Face config vLLM reads from ``config.json``; its
``text_config`` is the Qwen3.5 backbone. ``HaruConfig.spec`` returns the frozen
``HaruSpec`` that the audio modules (codec, MeanFlow head, prompt) read.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig, Qwen3_5TextConfig


def _pick(cls, params: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in params.items() if k in known})


@dataclass(frozen=True)
class HaruCodecConfig:
    """VibeVoice acoustic codec: 24 kHz waveform <-> continuous latents."""

    latent_dim: int = 64
    base_channels: int = 32
    ratios: tuple[int, ...] = (8, 5, 5, 4, 2, 2)  # upsampling order
    encoder_depths: tuple[int, ...] = (3, 3, 3, 3, 3, 3, 8)
    decoder_depths: tuple[int, ...] = (8, 3, 3, 3, 3, 3, 3)
    norm_eps: float = 1e-5
    latent_std: float = 0.5

    @property
    def frame_samples(self) -> int:
        samples = 1
        for ratio in self.ratios:
            samples *= ratio
        return samples


@dataclass(frozen=True)
class HaruHeadConfig:
    """MeanFlow head: LM hidden state -> next audio latent."""

    hidden_size: int = 2560
    latent_dim: int = 64
    num_layers: int = 4
    ffn_ratio: float = 3.0
    norm_eps: float = 1e-5
    interval_embed: float = 7.0
    num_steps: int = 1
    guidance_scale: float = 1.5
    guidance_range: tuple[float, float] = (1.0, 2.0)


@dataclass(frozen=True)
class HaruSpec:
    """The checkpoint configuration as the audio modules read it."""

    sample_rate: int = 24_000
    frame_samples: int = 3200
    latent_dim: int = 64
    pad_token_id: int = 248076
    eop_token_id: int = 248071
    dtype: str = "bfloat16"
    lm: dict[str, Any] = field(default_factory=dict)
    codec: HaruCodecConfig = field(default_factory=HaruCodecConfig)
    head: HaruHeadConfig = field(default_factory=HaruHeadConfig)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.frame_samples

    @property
    def hidden_size(self) -> int:
        return int(self.lm["hidden_size"])

    @property
    def eos_token_id(self) -> int | None:
        eos = self.lm.get("eos_token_id")
        return None if eos is None else int(eos)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> HaruSpec:
        if params.get("model_type", "haru") != "haru":
            raise ValueError(f"not a Haru checkpoint (model_type={params.get('model_type')!r})")
        params = dict(params)
        codec = dict(params.pop("codec", None) or {})
        head = dict(params.pop("head", None) or {})
        for key in ("ratios", "encoder_depths", "decoder_depths"):
            if key in codec:
                codec[key] = tuple(codec[key])
        if "guidance_range" in head:
            head["guidance_range"] = tuple(head["guidance_range"])
        spec = _pick(cls, {**params, "codec": _pick(HaruCodecConfig, codec), "head": _pick(HaruHeadConfig, head)})
        if spec.codec.frame_samples != spec.frame_samples:
            raise ValueError("codec ratios do not match frame_samples")
        return spec

    @classmethod
    def from_file(cls, path: str | Path) -> HaruSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))


class HaruConfig(PretrainedConfig):
    """Hugging Face config of a Haru checkpoint (``model_type="haru"``)."""

    model_type = "haru"
    sub_configs = {"text_config": Qwen3_5TextConfig}

    def __init__(self, lm: dict[str, Any] | None = None, text_config=None, **kwargs):
        if isinstance(text_config, Qwen3_5TextConfig):
            self.text_config = text_config
        else:
            params = dict(text_config or lm or {})
            accepted = inspect.signature(Qwen3_5TextConfig.__init__).parameters
            self.text_config = Qwen3_5TextConfig(**{k: v for k, v in params.items() if k in accepted})
        self.lm = dict(lm or {})
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs) -> Qwen3_5TextConfig:
        return self.text_config

    @property
    def spec(self) -> HaruSpec:
        params = {k: v for k, v in self.to_dict().items() if k not in ("text_config", "model_type")}
        return HaruSpec.from_dict(params)


__all__ = ["HaruCodecConfig", "HaruConfig", "HaruHeadConfig", "HaruSpec"]
