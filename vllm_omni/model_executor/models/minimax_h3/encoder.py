# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.minimax_h3.vae import (
    MiniMaxH3AudioVAE,
    MiniMaxH3VideoVAE,
)
from vllm_omni.model_executor.models.minimax_h3.conditioning import (
    MINIMAX_H3_ENCODER_LAYOUT_KEY,
    MINIMAX_H3_ENCODER_REQUEST_KEY,
    MiniMaxH3EncoderMediaConditioning,
    MiniMaxH3EncoderMediaInput,
)
from vllm_omni.model_executor.models.minimax_h3.encoder_processing import encode_media
from vllm_omni.model_executor.models.minimax_h3.text_encoder import (
    MiniMaxH3TextEncoderBackbone,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

_MEDIA_CONDITIONING_CACHE_KEY = "_minimax_h3_encoder_media_conditioning"
_MEDIA_CACHE_KEY = "_minimax_h3_encoder_media"
_COMPONENT_CONFIG_KEY = "minimax_h3_encoder_components"
_COMPONENT_ROLES = frozenset({"text_encoder", "video_vae", "audio_vae"})


def _partition_root(model_path: str) -> Path:
    path = Path(model_path)
    if path.name in {"text_encoder", "video_vae", "audio_vae"}:
        return path.parent
    return path


@dataclass(frozen=True)
class MiniMaxH3EncoderComponentConfig:
    text_parallel_mode: str
    video_parallel_mode: str
    audio_parallel_mode: str

    def __post_init__(self) -> None:
        if self.text_parallel_mode != "tp":
            raise ValueError("MiniMax H3 text encoder currently supports parallel_mode='tp' only")
        if self.video_parallel_mode not in {"leader", "patch"}:
            raise ValueError("MiniMax H3 video VAE parallel_mode must be 'leader' or 'patch'")
        if self.audio_parallel_mode != "leader":
            raise ValueError("MiniMax H3 audio VAE currently supports parallel_mode='leader' only")

    @staticmethod
    def _parallel_mode(raw: Mapping[str, Any], role: str) -> str:
        value = raw[role]
        if not isinstance(value, Mapping):
            raise TypeError(f"MiniMax H3 encoder role {role!r} must be a mapping")
        if set(value) != {"parallel_mode"}:
            raise ValueError(f"MiniMax H3 encoder role {role!r} requires only parallel_mode")
        return str(value["parallel_mode"])

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> MiniMaxH3EncoderComponentConfig:
        hf_config = vllm_config.model_config.hf_config
        raw = getattr(hf_config, _COMPONENT_CONFIG_KEY, None)
        if not isinstance(raw, Mapping):
            raise ValueError(f"MiniMax H3 Encoder Stage requires explicit {_COMPONENT_CONFIG_KEY}")
        roles = set(raw)
        if roles != _COMPONENT_ROLES:
            missing = sorted(_COMPONENT_ROLES - roles)
            unknown = sorted(roles - _COMPONENT_ROLES)
            raise ValueError(
                f"{_COMPONENT_CONFIG_KEY} requires exactly {sorted(_COMPONENT_ROLES)}; "
                f"missing={missing}, unknown={unknown}"
            )
        return cls(
            text_parallel_mode=cls._parallel_mode(raw, "text_encoder"),
            video_parallel_mode=cls._parallel_mode(raw, "video_vae"),
            audio_parallel_mode=cls._parallel_mode(raw, "audio_vae"),
        )


class MiniMaxH3Encoder(MiniMaxH3TextEncoderBackbone):
    requires_request_sample_eligibility = True
    enable_update_additional_information = True
    omni_payload_at_request_end = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        try:
            tp_group = get_tp_group()
            tp_rank = int(tp_group.rank_in_group)
        except (AssertionError, RuntimeError):
            tp_group = None
            tp_rank = 0
        self._component_leader = tp_rank == 0
        self.component_config = MiniMaxH3EncoderComponentConfig.from_vllm_config(vllm_config)
        root = _partition_root(vllm_config.model_config.model)
        missing = [name for name in ("text_encoder", "video_vae", "audio_vae") if not (root / name).is_dir()]
        if missing:
            raise RuntimeError(f"MiniMax H3 Encoder Stage requires all three encoders; missing {missing}")
        device = get_local_device()
        self.video_vae: MiniMaxH3VideoVAE | None = None
        self.audio_vae: MiniMaxH3AudioVAE | None = None
        if self._component_leader or self.component_config.video_parallel_mode == "patch":
            self.video_vae = MiniMaxH3VideoVAE(
                str(root / "video_vae"),
                device=device,
                encode_only=True,
            )
            if self.component_config.video_parallel_mode == "patch":
                if tp_group is None:
                    raise RuntimeError("MiniMax H3 video VAE patch parallelism requires an initialized TP group")
                self.video_vae.set_parallel_size(
                    tp_group.world_size,
                    process_group=tp_group.device_group,
                )
        if self._component_leader:
            self.audio_vae = MiniMaxH3AudioVAE(
                str(root / "audio_vae"),
                device=device,
                encode_only=True,
            )

    @staticmethod
    def _media_input(info: dict[str, Any]) -> MiniMaxH3EncoderMediaInput:
        cached = info.get(_MEDIA_CACHE_KEY)
        if isinstance(cached, MiniMaxH3EncoderMediaInput):
            return cached
        meta = info.get("meta")
        metadata = meta.get(MINIMAX_H3_ENCODER_REQUEST_KEY) if isinstance(meta, Mapping) else None
        if not isinstance(metadata, Mapping):
            raise RuntimeError("MiniMax H3 unified encoder requires request metadata")

        hidden_states = info.get("hidden_states")
        raw_media = hidden_states.get("layers") if isinstance(hidden_states, Mapping) else None
        if raw_media is None:
            media_tensors: list[torch.Tensor] = []
        elif isinstance(raw_media, Mapping):
            try:
                ordered_items = sorted(raw_media.items(), key=lambda item: int(item[0]))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("MiniMax H3 encoder media keys must be integer indices") from exc
            if [int(key) for key, _ in ordered_items] != list(range(len(ordered_items))):
                raise RuntimeError("MiniMax H3 encoder media indices must be contiguous from zero")
            media_tensors = [value for _, value in ordered_items]
        else:
            raise RuntimeError("MiniMax H3 encoder media must be a namespaced tensor mapping")
        if any(not isinstance(value, torch.Tensor) for value in media_tensors):
            raise RuntimeError("MiniMax H3 encoder media must contain only tensors")
        media = MiniMaxH3EncoderMediaInput.from_mm_tensors(media_tensors, metadata)
        if media.to_metadata() != dict(metadata):
            raise RuntimeError("MiniMax H3 encoder media does not match request metadata")
        info[_MEDIA_CACHE_KEY] = media
        return media

    def _encode_media(self, media: MiniMaxH3EncoderMediaInput) -> MiniMaxH3EncoderMediaConditioning | None:
        if not self._component_leader and self.component_config.video_parallel_mode == "leader":
            return None
        return encode_media(
            media,
            video_vae=self.video_vae,
            audio_vae=self.audio_vae,
            emit_conditioning=self._component_leader,
        )

    @staticmethod
    def _has_encoder_input(info: object) -> bool:
        if not isinstance(info, Mapping):
            return False
        meta = info.get("meta")
        return isinstance(meta, Mapping) and isinstance(meta.get(MINIMAX_H3_ENCODER_REQUEST_KEY), Mapping)

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        output = super().make_omni_output(model_outputs, **kwargs)
        infos = kwargs.get("model_intermediate_buffer") or []
        if not isinstance(infos, Sequence) or not infos:
            return output
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniMax H3 unified encoder received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        per_request: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = []
        release_workspace = False
        for eligible, info in zip(sample_eligible, infos, strict=True):
            if not eligible:
                per_request.append(None)
                continue
            if not isinstance(info, dict) or not self._has_encoder_input(info):
                raise RuntimeError("MiniMax H3 unified encoder request metadata is missing from the runner")
            release_workspace = True
            if _MEDIA_CONDITIONING_CACHE_KEY not in info:
                info[_MEDIA_CONDITIONING_CACHE_KEY] = self._encode_media(self._media_input(info))
            if not self._component_leader:
                continue
            cached = info.get(_MEDIA_CONDITIONING_CACHE_KEY)
            if not isinstance(cached, MiniMaxH3EncoderMediaConditioning):
                raise RuntimeError("MiniMax H3 encoder components did not produce conditioning")
            per_request.append(cached.to_omni_components())

        if release_workspace:
            torch.accelerator.empty_cache()
        if not self._component_leader:
            return output

        if not any(item is not None for item in per_request):
            return output
        empty = torch.empty((0,), dtype=torch.float32)
        multimodal_outputs = dict(output.multimodal_outputs or {})
        dotted_key = f"kv_metadata.{MINIMAX_H3_ENCODER_LAYOUT_KEY}"
        multimodal_outputs[dotted_key] = [
            item[2] if item is not None else torch.empty((0,), dtype=torch.int64) for item in per_request
        ]

        embed = dict(multimodal_outputs.get("embed") or {})
        embed["embedding"] = [item[0] if item is not None else empty for item in per_request]
        embed["speech_feat"] = [item[1] if item is not None else empty for item in per_request]
        multimodal_outputs["embed"] = embed
        return OmniOutput(
            text_hidden_states=output.text_hidden_states,
            multimodal_outputs=multimodal_outputs,
            intermediate_tensors=output.intermediate_tensors,
            next_token_id=output.next_token_id,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loaded = super().load_weights(weights)
        for component_name in ("video_vae", "audio_vae"):
            component = getattr(self, component_name)
            if component is not None:
                loaded.update(f"{component_name}.{name}" for name, _ in component.named_parameters())
        return loaded


__all__ = ["MiniMaxH3Encoder"]
