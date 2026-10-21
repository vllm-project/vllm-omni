# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-input identities owned by the actual Thinker and Talker stages."""

from typing import Any

from vllm import __version__ as vllm_version
from vllm.config.utils import normalize_value
from vllm.lora.request import LoRARequest

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.models.qwen3_omni.talker_conditioning import (
    complete_talker_conditioning_digest,
    conditioning_metadata_digest,
)


def stage_conditioning_namespace(vllm_config: Any) -> str:
    """Snapshot resolved config once per immutable stage, not per decode.

    The graph hash alone is insufficient: loader settings do not enter vLLM's
    compilation hash. Include the resolved HF config and weight/loading source
    explicitly. This uses the engine's immutable model/revision contract; it is
    not an attestation of mutable files or live weight-reload support.
    """
    model = vllm_config.model_config
    load = vllm_config.load_config
    return conditioning_metadata_digest(
        normalize_value(
            {
                "schema": "qwen3-omni.stage-conditioning.v1",
                "vllm": vllm_version,
                "graph": vllm_config.compute_hash(),
                "model": model.model,
                "model_weights": model.model_weights,
                "revision": model.revision,
                "code_revision": model.code_revision,
                "hf_config": model.hf_config.to_dict(),
                "dtype": model.dtype,
                "stage": model.model_stage,
                "architecture": model.model_arch,
                "load_format": load.load_format,
                "loader_options": load.model_loader_extra_config,
                # Dummy loading uses this seed to initialize actual weights.
                "initialization_seed": model.seed if load.load_format == "dummy" else None,
            }
        )
    )


def adapter_conditioning_namespace(request: Any) -> str | None:
    """Use vLLM's immutable adapter-ID contract, never a caller extras flag."""
    adapter = getattr(request, "lora_request", None)
    if adapter is None:
        return None
    if not isinstance(adapter, LoRARequest) or adapter.load_inplace:
        raise ValueError("conditioning identity requires an immutable loaded LoRA adapter")
    return conditioning_metadata_digest(
        normalize_value(
            {
                "schema": "qwen3-omni.adapter-conditioning.v1",
                "id": adapter.lora_int_id,
                "name": adapter.lora_name,
                "path": adapter.lora_path,
                "base_model": adapter.base_model_name,
                "tensorizer": adapter.tensorizer_config_dict,
                "is_3d": adapter.is_3d_lora_weight,
            }
        )
    )


def talker_speaker_map(config: Any) -> dict[str, int]:
    speakers = getattr(config, "speaker_id", None)
    if speakers:
        return {key.lower(): value for key, value in speakers.items()}
    return dict.fromkeys(("default", "ethan", "prefix_caching"), config.audio_start_token_id)


DEFAULT_SPEAKER = "ethan"


def resolve_default_speaker(speaker_token_ids: dict[str, int]) -> str:
    """Match the reference Ethan default rather than checkpoint key order."""
    return DEFAULT_SPEAKER if DEFAULT_SPEAKER in speaker_token_ids else next(iter(speaker_token_ids))


def resolve_talker_speaker(voice: object, speakers: dict[str, int], default: str) -> int:
    """The same normalization/default selection as Talker prefill execution."""
    if isinstance(voice, (list, tuple)) and voice:
        voice = voice[0]
    if not isinstance(voice, str) or not voice.strip():
        voice = default
    else:
        voice = voice.lower().strip()
    return speakers.get(voice, speakers[default])


def _request_speaker(request: Any) -> object:
    info = getattr(request, "additional_information", None)
    if info is not None and not isinstance(info, dict):
        info = deserialize_additional_information(info)
    # GPU admission installs the direct buffer first; a nonempty serialized
    # additional-information dictionary then replaces that buffer.
    if not info:
        info = getattr(request, "model_intermediate_buffer", None)
    return info.get("speaker") if isinstance(info, dict) else None


class TalkerInputIdentity:
    """Consumer-local completion of the full-payload producer's source digest."""

    def __init__(self, vllm_config: Any) -> None:
        self.model_namespace = stage_conditioning_namespace(vllm_config)
        self.projection_namespace = conditioning_metadata_digest(
            ["qwen3-omni.projection.v1", self.model_namespace, "talker.text_projection", "talker.hidden_projection"]
        )
        config = vllm_config.model_config.hf_config
        self.text_controls = (config.tts_pad_token_id, config.tts_bos_token_id, config.tts_eos_token_id)
        talker = config.talker_config
        self.codec_controls = (
            talker.codec_nothink_id,
            talker.codec_think_bos_id,
            talker.codec_think_eos_id,
            talker.codec_pad_id,
            talker.codec_bos_id,
        )
        self.speakers = talker_speaker_map(talker)
        self.default_speaker = resolve_default_speaker(self.speakers)
        self.required = bool(vllm_config.cache_config.enable_prefix_caching)

    def finalize(self, request: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        source_digest = metadata.get("next_stage_source_digest")
        if source_digest is None:
            # A late length/terminal notice may acknowledge an already fixed
            # input. It cannot mint a new identity or replace finalized IDs.
            finalized_notice = (
                getattr(request, "_omni_input_finalized", False)
                and getattr(request, "_omni_conditioning_digest", None) is not None
                and metadata.keys() <= {"next_stage_prompt_len", "input_terminal"}
            )
            if self.required and not finalized_notice:
                raise ValueError("Talker prefix caching requires verified complete source conditioning")
            return metadata
        if metadata.get("next_stage_source_identity_error") is not None:
            raise ValueError("source conditioning cannot be both verified and unavailable")
        speaker_id = resolve_talker_speaker(
            metadata.get("next_stage_speaker", _request_speaker(request)), self.speakers, self.default_speaker
        )
        nothink, think_bos, think_eos, pad, bos = self.codec_controls
        digest = complete_talker_conditioning_digest(
            source_digest=source_digest,
            talker_model=self.model_namespace,
            projection=self.projection_namespace,
            talker_adapter=adapter_conditioning_namespace(request),
            text_controls=self.text_controls,
            codec_controls=(nothink, think_bos, think_eos, speaker_id, pad, bos),
        )
        advertised = metadata.get("next_stage_conditioning_digest")
        if advertised is not None and advertised != digest:
            raise ValueError("producer conditioning conflicts with the actual Talker namespace")
        return {**metadata, "next_stage_conditioning_digest": digest}


def is_qwen3_full_payload_talker(model_config: Any) -> bool:
    architecture = getattr(model_config, "model_arch", None)
    architectures = (
        (architecture,)
        if architecture
        else getattr(model_config, "architectures", None)
        or getattr(getattr(model_config, "hf_config", None), "architectures", ())
        or ()
    )
    return (
        getattr(model_config, "model_stage", None) == "talker"
        and "Qwen3OmniMoeForConditionalGeneration" in architectures
        and not getattr(model_config, "async_chunk", False)
    )
