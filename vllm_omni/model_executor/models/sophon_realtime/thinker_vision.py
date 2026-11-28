# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DuplexJev thinker with vision: speech and an image in one request.

vLLM's Qwen3-VL (vision encoder + language model) plus the DuplexJev audio path
(Qwen3-ASR encoder + projector, see ``thinker.py``). The audio config lives in the
checkpoint's ``audio/`` subdirectory. Audio soft tokens sit on ``<|audio|>`` and
take ordinary text positions in Qwen3-VL's 3-D RoPE, as in training.

Prompt layout (chat API): put ``<|audio|>`` in the text where the speech goes
and an image part where the picture goes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import torch
import vllm.model_executor.models.qwen3_vl as qwen3_vl
from transformers import BatchFeature
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig

from vllm_omni.model_executor.models.sophon_realtime.thinker import (
    AUDIO_KEYS,
    AUDIO_TOKEN,
    CHUNK_S,
    MAX_AUDIO_CHUNKS,
    SAMPLE_RATE,
    DuplexJevAudioMixin,
    audio_fields_config,
    audio_prompt_update,
    audio_tokens_per_chunk,
    load_audio_processor,
    pad_audio_embeddings_for_deepstack,
    process_audio,
)
from vllm_omni.model_executor.models.sophon_realtime.thinker_processing import DuplexJevProcessor

AUDIO_WEIGHT_PREFIXES = ("audio_tower.", "multi_modal_projector.")


def _audio_file(model: str, revision: str | None, name: str) -> str:
    if os.path.isdir(model):
        return os.path.join(model, "audio", name)
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return hf_api().hf_hub_download(model, f"audio/{name}", revision=revision)


@lru_cache(maxsize=4)
def load_audio_config(model: str, revision: str | None):
    """The audio path's config (``audio/config.json``, Ultravox format)."""
    from vllm.transformers_utils.configs.ultravox import UltravoxConfig

    import vllm_omni.transformers_utils.configs.sophon_realtime  # noqa: F401  registers the encoder config

    with open(_audio_file(model, revision, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("architectures", None)
    cfg.update(text_model_id=None, audio_model_id=None)
    return UltravoxConfig(**cfg)


class DuplexJevVisionProcessingInfo(qwen3_vl.Qwen3VLProcessingInfo):
    def get_audio_config(self):
        mc = self.ctx.model_config
        return load_audio_config(mc.model, mc.revision)

    def get_audio_processor(self) -> DuplexJevProcessor:
        mc = self.ctx.model_config
        extractor_dir = os.path.dirname(_audio_file(mc.model, mc.revision, "preprocessor_config.json"))
        return load_audio_processor(extractor_dir, mc.model, None, int(self.get_audio_config().stack_factor), False)

    def get_audio_token_id(self) -> int:
        return int(self.get_tokenizer().convert_tokens_to_ids(AUDIO_TOKEN))

    def get_data_parser(self):
        return qwen3_vl.Qwen3VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
            target_sr=SAMPLE_RATE,
            target_channels=1,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None, "audio": None}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        tokens = dict(super().get_mm_max_tokens_per_item(seq_len, mm_counts))
        tokens["audio"] = MAX_AUDIO_CHUNKS * audio_tokens_per_chunk(int(self.get_audio_config().stack_factor))
        return tokens


class DuplexJevVisionDummyInputsBuilder(qwen3_vl.Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_TOKEN * mm_counts.get("audio", 0) + super().get_dummy_text(mm_counts)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        vision_counts = {k: v for k, v in mm_counts.items() if k != "audio"}
        data = dict(super().get_dummy_mm_data(seq_len, vision_counts, mm_options))
        n = mm_counts.get("audio", 0)
        if n:
            data["audio"] = self._get_dummy_audios(
                length=SAMPLE_RATE * CHUNK_S * MAX_AUDIO_CHUNKS,
                num_audios=n,
                overrides=mm_options.get("audio") if mm_options else None,
            )
        return data


class DuplexJevVisionMultiModalProcessor(qwen3_vl.Qwen3VLMultiModalProcessor):
    def _apply_hf_processor_main(self, mm_items, hf_kwargs) -> BatchFeature:
        counts = mm_items.get_all_counts()
        vision = {k for k, c in counts.items() if k != "audio" and c > 0}
        out = super()._apply_hf_processor_main(mm_items.select(vision), hf_kwargs) if vision else BatchFeature({})
        if counts.get("audio", 0) > 0:
            hf_data, _, passthrough = self._get_hf_mm_inputs(mm_items.select({"audio"}), hf_kwargs)
            audios = hf_data.get("audio") or hf_data.get("audios") or []
            if not isinstance(audios, (list, tuple)):
                audios = [audios]
            if audios:
                out.update(process_audio(self.info.get_audio_processor(), audios))
            out.update(passthrough)
        return out

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]:
        fields = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        fields.update(audio_fields_config(hf_inputs))
        return fields

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[Any]:
        updates = list(super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs))
        updates.append(audio_prompt_update(self.info.get_audio_token_id(), out_mm_kwargs))
        return updates


@MULTIMODAL_REGISTRY.register_processor(
    DuplexJevVisionMultiModalProcessor,
    info=DuplexJevVisionProcessingInfo,
    dummy_inputs=DuplexJevVisionDummyInputsBuilder,
)
class DuplexJevVisionForConditionalGeneration(DuplexJevAudioMixin, qwen3_vl.Qwen3VLForConditionalGeneration):
    """DuplexJev on Qwen3-VL: Qwen3-VL's vision path plus the DuplexJev audio path."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_TOKEN
        return super().get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        mc = vllm_config.model_config
        audio_cfg = load_audio_config(mc.model, mc.revision)
        with self._mark_tower_model(vllm_config, "audio"):
            self._init_audio(audio_cfg, prefix=prefix)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # audio soft tokens are ordinary text positions
        return super().get_mrope_input_positions(input_tokens, [f for f in mm_features if f.modality != "audio"])

    def embed_multimodal(self, **kwargs: object):
        audio_kwargs = {k: kwargs.pop(k) for k in AUDIO_KEYS if k in kwargs}
        out = list(super().embed_multimodal(**kwargs) or ()) if kwargs else []
        embeddings = self._embed_audio(**audio_kwargs) if audio_kwargs else []
        if embeddings:
            if self.use_deepstack:
                embeddings = pad_audio_embeddings_for_deepstack(embeddings, self.multiscale_dim)
            out.extend(embeddings)
        return tuple(out)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        audio: list[tuple[str, torch.Tensor]] = []

        def split(items):
            for name, tensor in items:
                if name.startswith(AUDIO_WEIGHT_PREFIXES):
                    audio.append((name, tensor))
                else:
                    yield name, tensor

        loaded = set(super().load_weights(split(weights)))
        if not audio:  # older checkpoints keep the audio path only in audio/
            from safetensors.torch import load_file

            mc = self.model_config
            audio = list(load_file(_audio_file(mc.model, mc.revision, "model.safetensors")).items())
        return loaded | self._load_audio_weights(audio)
