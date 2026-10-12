# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from nvidia/Nemotron-Labs-Audex-2B (Apache-2.0):
#   inference_scripts_vllm/audioqa_scripts/audex_2b_vllm/modeling_audex_vllm.py
#   inference_scripts_vllm/audioqa_scripts/audex_2b_vllm/audio_encoder.py
"""Audex audio-understanding model (``checkpoint_folder_full``): S2T / audio QA.

Mirrors the Qwen2-Audio pattern with the 2B dense backbone:
  * ``language_model``  = the vllm-omni ``NemotronDenseForCausalLM`` port
    (wrapped via ``init_vllm_registered_model``; resolved through the omni
    model registry).
  * ``audio_tower``     = transformers ``Qwen2AudioEncoder`` ("NV-Whisper");
    a parameter-free avg pool maps 1500 → 750 frames per padded 30 s clip.
  * ``audio_projector`` = RMSNorm → fc1 → relu² → fc2 (no bias), matching the
    release ``NemotronHAudexProjector``.

``embed_multimodal(...)`` runs encoder+projector and returns one embedding
tensor per audio item; vLLM's ``SupportsMultiModal.embed_input_ids`` merges
them into the ``<so_embedding>`` placeholder positions. The encoder runs in
microbatches over clips so peak memory is bounded regardless of audio length
(up to the 900 s / 30-clip cap enforced by the processor).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm_omni.model_executor.models.audex.audex_omni_processing import (
    AUDIO_ENCODER_MICRO_BATCH,
    SOUND_TOKEN,
    AudexDummyInputsBuilder,
    AudexMultiModalProcessor,
    AudexProcessingInfo,
)


class AudexProjector(nn.Module):
    """Megatron sound_projection equivalent (matches HF NemotronHAudexProjector)."""

    def __init__(
        self,
        audio_encoder_hidden_size: int,
        intermediate_size: int,
        text_hidden_size: int,
        activation: str = "relu2",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm_weight = nn.Parameter(torch.ones(audio_encoder_hidden_size))
        self.norm_eps = norm_eps
        self.fc1 = nn.Linear(audio_encoder_hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, text_hidden_size, bias=False)
        self.activation = activation

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.norm_eps)
        return (self.norm_weight.float() * x).to(dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self._rms_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.activation == "relu2":
            hidden_states = F.relu(hidden_states).pow(2)
        elif self.activation == "gelu":
            hidden_states = F.gelu(hidden_states)
        else:
            raise ValueError(f"Unsupported audio projector activation: {self.activation}")
        return self.fc2(hidden_states)


def _build_qwen2_audio_encoder(audio_config: dict) -> nn.Module:
    from transformers.models.qwen2_audio.configuration_qwen2_audio import (
        Qwen2AudioEncoderConfig,
    )
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

    cfg = Qwen2AudioEncoderConfig(**dict(audio_config))
    return Qwen2AudioEncoder(cfg)


class AudexAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - c: total number of 30s clips (flattened across audio items)
        - b: number of audio items
    """

    type: Literal["audio_features"] = "audio_features"
    input_features: Annotated[torch.Tensor, TensorShape("c", 128, 3000)]
    audio_num_clips: Annotated[list[int], None] = None


@MULTIMODAL_REGISTRY.register_processor(
    AudexMultiModalProcessor,
    info=AudexProcessingInfo,
    dummy_inputs=AudexDummyInputsBuilder,
)
class NemotronDenseAudexForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    _LM_ARCHITECTURE = "NemotronDenseForCausalLM"

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return SOUND_TOKEN
        raise ValueError(f"Only audio modality is supported, got {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=[self._LM_ARCHITECTURE],
            )
        llm_dtype = self.language_model.config.dtype
        assert isinstance(llm_dtype, torch.dtype)
        self.llm_dtype = llm_dtype

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = _build_qwen2_audio_encoder(config.audio_config).to(llm_dtype)
            self.audio_projector = AudexProjector(
                audio_encoder_hidden_size=config.audio_encoder_hidden_size,
                intermediate_size=config.audio_projector_intermediate_size,
                text_hidden_size=config.hidden_size,
                activation=config.audio_projector_activation,
                norm_eps=config.audio_projector_norm_eps,
            ).to(llm_dtype)

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _parse_and_validate_audio_input(self, **kwargs: object):
        input_features = kwargs.pop("input_features", None)
        audio_num_clips = kwargs.pop("audio_num_clips", None)
        if input_features is None:
            return None
        return AudexAudioFeatureInputs(
            type="audio_features",
            input_features=input_features,
            audio_num_clips=audio_num_clips,
            validate=False,
        )

    @torch.inference_mode()
    def _encode_clips(self, input_features: torch.Tensor) -> torch.Tensor:
        """(C, 128, 3000) -> (C, 750, hidden) via microbatched encoder+projector."""
        param = next(self.audio_tower.parameters())
        input_features = input_features.to(device=param.device, dtype=param.dtype)
        out_chunks = []
        for start in range(0, input_features.shape[0], AUDIO_ENCODER_MICRO_BATCH):
            chunk = input_features[start : start + AUDIO_ENCODER_MICRO_BATCH]
            hidden = self.audio_tower(input_features=chunk, return_dict=True).last_hidden_state
            out_chunks.append(self.audio_projector(hidden))
        return torch.cat(out_chunks, dim=0)

    def _process_audio_input(self, audio_input) -> tuple[torch.Tensor, ...]:
        input_features = audio_input["input_features"]
        audio_num_clips = audio_input["audio_num_clips"]
        if input_features.ndim == 4:
            input_features = input_features.flatten(0, 1)

        projected = self._encode_clips(input_features)
        per_clip = projected.reshape(projected.shape[0], -1, projected.shape[-1])

        if audio_num_clips is None:
            num_clips_list = [projected.shape[0]]
        elif isinstance(audio_num_clips, torch.Tensor):
            num_clips_list = audio_num_clips.flatten().tolist()
        else:
            num_clips_list = [int(x) for x in audio_num_clips]

        grouped: list[torch.Tensor] = []
        offset = 0
        for n in num_clips_list:
            clips = per_clip[offset : offset + n]
            grouped.append(clips.reshape(-1, clips.shape[-1]))
            offset += n
        return tuple(grouped)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata=None) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Split the combined weight stream into LLM / audio_tower / projector.

        vLLM streams *all* ``.safetensors`` in the model dir; route
        ``audio_encoder.*``/``audio_projector.*`` to the audio modules and
        leave the remaining ``model.*``/``lm_head.*`` names untouched so the
        wrapped NemotronDenseForCausalLM loader applies its own qkv fusion
        and remapping.
        """
        loaded: set[str] = set()
        lm_weights: list[tuple[str, torch.Tensor]] = []
        enc_sd: dict[str, torch.Tensor] = {}
        proj_map: dict[str, torch.Tensor] = {}

        for name, w in weights:
            if name.startswith("audio_encoder."):
                enc_sd[name[len("audio_encoder.") :]] = w
            elif name == "audio_projector.norm.weight":
                proj_map["norm_weight"] = w
            elif name.startswith("audio_projector."):
                proj_map[name[len("audio_projector.") :]] = w
            else:
                lm_weights.append((name, w))

        lm_loaded = self.language_model.load_weights(lm_weights)
        loaded |= {f"language_model.{n}" for n in lm_loaded}

        if not enc_sd:
            raise ValueError("No audio_encoder.* weights found in checkpoint stream.")
        missing, unexpected = self.audio_tower.load_state_dict(enc_sd, strict=False)
        if missing:
            raise ValueError(f"Missing audio_tower params: {list(missing)[:8]}")
        if unexpected:
            raise ValueError(f"Unexpected audio_tower keys: {list(unexpected)[:8]}")
        loaded |= {f"audio_tower.{k}" for k in enc_sd}

        if set(proj_map) != {"norm_weight", "fc1.weight", "fc2.weight"}:
            raise ValueError(f"Unexpected audio_projector keys: {sorted(proj_map)}")
        self.audio_projector.load_state_dict(proj_map, strict=True)
        loaded |= {f"audio_projector.{k}" for k in proj_map}

        return loaded


@MULTIMODAL_REGISTRY.register_processor(
    AudexMultiModalProcessor,
    info=AudexProcessingInfo,
    dummy_inputs=AudexDummyInputsBuilder,
)
class NemotronHAudexForConditionalGeneration(
    NemotronDenseAudexForConditionalGeneration,
    IsHybrid,
    HasInnerState,
    SupportsMambaPrefixCaching,
):
    """Audex 30B-A3B audio-understanding model (``checkpoint_folder_full``).

    Mirrors nvidia/Nemotron-Labs-Audex-30B-A3B's
    ``inference_scripts_vllm/audioqa_scripts/audex_30b_a3b_vllm/modeling_audex_vllm.py``:
    the same encoder/projector/processing as the dense 2B wrapper around
    vLLM's native hybrid Mamba+MoE ``NemotronHForCausalLM``, whose state
    hooks are delegated below.
    """

    _LM_ARCHITECTURE = "NemotronHForCausalLM"

    is_hybrid = True
    has_inner_state = True
    supports_mamba_prefix_caching = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

        return NemotronHForCausalLM.get_mamba_state_copy_func()
