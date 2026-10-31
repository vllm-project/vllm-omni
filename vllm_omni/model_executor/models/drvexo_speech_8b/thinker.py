# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""vLLM Thinker stage for DRVEXO-speech-8b.

The stage keeps Qwen3's native decoder and loads a separate input projector.
Raw audio uses the Qwen3-ASR encoder; precomputed ``audio_embeds`` contain
2048-dimensional ASR states. Connector training status belongs to the artifact,
not this architecture registration.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_asr import _get_feat_extract_output_lengths
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder
from vllm.model_executor.models.utils import AutoWeightsLoader, _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRAudioEncoderConfig

from .artifacts import resolve_artifact_path, resolve_audio_encoder_quantization, resolve_projector_quantization
from .checkpoint import load_checkpoint
from .processing import DRVEXODummyInputsBuilder, DRVEXOMultiModalProcessor, DRVEXOProcessingInfo


class _InputProjector(nn.Module):
    """State-dict-compatible copy of QwenOmniInputProjector."""

    def __init__(self, audio_dim: int, language_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.net[0].normalized_shape[0]:
            raise ValueError(
                f"DRVEXO audio features must end in {self.net[0].normalized_shape[0]}, got {features.shape[-1]}"
            )
        return self.net(features)


@MULTIMODAL_REGISTRY.register_processor(
    DRVEXOMultiModalProcessor,
    info=DRVEXOProcessingInfo,
    dummy_inputs=DRVEXODummyInputsBuilder,
)
class DRVEXOSpeech8BForConditionalGeneration(Qwen3ForCausalLM, SupportsMultiModal):
    """Dense Qwen3 stage with a vLLM multimodal embedding boundary."""

    supports_multimodal = True
    supports_multimodal_embeddings = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        raise ValueError("DRVEXO-speech-8b only supports audio and text inputs")

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.general_prefill = None
        specification = getattr(config, "drvexo_general_decision", None)
        if specification is not None:
            from .general_prefill import GeneralClassifier, GeneralPrefill

            classifier = GeneralClassifier(vllm_config.model_config.model, specification)
            classifier.validate_scheduler(vllm_config)
            self.general_prefill = GeneralPrefill(classifier)
            self.have_multimodal_outputs = True
            self.requires_request_computed_tokens = True
            self.omni_client_multimodal_output_keys = (
                "drvexo_general_scores",
                "drvexo_general_status",
            )
        self.audio_feature_dim = int(getattr(config, "audio_feature_dim", 2048))
        model_directory = vllm_config.model_config.model
        checkpoint = getattr(config, "connector_checkpoint", None)
        self.connector_checkpoint = resolve_artifact_path(model_directory, checkpoint) if checkpoint else None
        self.projector_quantization = resolve_projector_quantization(
            model_directory,
            getattr(config, "input_projector_quantization", None),
            self.connector_checkpoint,
        )
        if self.projector_quantization is None:
            self.input_projector = _InputProjector(self.audio_feature_dim, config.hidden_size)
        else:
            from .projector import QuantizedInputProjector

            self.input_projector = QuantizedInputProjector(
                self.audio_feature_dim,
                config.hidden_size,
                self.projector_quantization[0],
            )
        self.audio_encoder_path = resolve_artifact_path(model_directory, config.audio_encoder_path)
        self.audio_encoder_quantization = resolve_audio_encoder_quantization(
            model_directory,
            getattr(config, "audio_encoder_quantization", None),
        )
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                Qwen3ASRAudioEncoderConfig(**config.audio_encoder_config),
                prefix="audio_tower",
            )

    def _load_connector_checkpoint(self, checkpoint: str | None) -> set[str]:
        """Load only the input projector from the separate connector artifact."""
        if self.projector_quantization is not None:
            loaded = self.input_projector.load_packed_weights(self.projector_quantization[1])
            return {f"input_projector.{name}" for name in loaded}
        if not checkpoint:
            raise ValueError("DRVEXO requires a connector_checkpoint")
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = load_checkpoint(path)
        state = payload.get("state_dict", payload)
        projector_state = {
            key.removeprefix("input_projector."): value
            for key, value in state.items()
            if key.startswith("input_projector.")
        }
        self.input_projector.load_state_dict(projector_state, strict=True)
        return {f"input_projector.{name}" for name in projector_state}

    def _parse_and_validate_audio_input(self, **kwargs: object) -> torch.Tensor | None:
        features = kwargs.pop("input_audio_features", None)
        if features is None:
            features = kwargs.pop("audio_embeds", None)
        if features is None:
            return None
        if not isinstance(features, torch.Tensor):
            raise TypeError("DRVEXO audio input must be a torch.Tensor")
        if features.ndim not in (2, 3):
            raise ValueError("DRVEXO audio input must have shape [frames,2048] or [batch,frames,2048]")
        if features.shape[-1] != self.audio_feature_dim:
            raise ValueError(
                f"DRVEXO audio input has dimension {features.shape[-1]}, expected {self.audio_feature_dim}"
            )
        return features

    def embed_multimodal(self, **kwargs: object):
        if kwargs.get("audio_mels") is not None:
            mels = kwargs["audio_mels"]
            lengths = kwargs["audio_mel_lengths"]
            output_lengths = _get_feat_extract_output_lengths(lengths)
            features = self.audio_tower(
                mels.to(self.audio_tower.dtype),
                feature_lens=lengths,
                aftercnn_lens=output_lengths,
            )
            return tuple(self.input_projector(features).split(output_lengths.tolist()))
        embeds = kwargs.get("audio_embeds")
        if isinstance(embeds, (list, tuple)):
            return tuple(self.embed_multimodal(audio_embeds=item)[0] for item in embeds)
        features = self._parse_and_validate_audio_input(**kwargs)
        if features is None:
            return []
        projected = self.input_projector(features.to(next(self.input_projector.parameters()).dtype))
        if projected.ndim == 2:
            return (projected,)
        return tuple(projected.unbind(dim=0))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        # Omni passes scheduler metadata alongside the model inputs. Qwen's
        # native forward accepts only decoder arguments.
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    def make_omni_output(
        self,
        model_outputs,
        *,
        model_intermediate_buffer=None,
        request_token_spans=None,
        request_computed_tokens=None,
        **kwargs,
    ):
        if self.general_prefill is None:
            return model_outputs
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        buffers = model_intermediate_buffer or []
        payload = {}
        if any("drvexo_general_decision" in info for info in buffers):
            if request_token_spans is None or request_computed_tokens is None:
                raise ValueError("General decisions require request spans and computed-token offsets")
            payload = self.general_prefill.process(model_outputs, buffers, request_token_spans, request_computed_tokens)
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=payload)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("DRVEXO multimodal input requires is_multimodal placeholder mask")
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Qwen shards use the native loader; connector weights are a separate
        # artifact and are not mixed into the base.
        language_weights = (
            (name, value) for name, value in weights if not name.startswith(("input_projector.", "audio_tower."))
        )
        loaded = AutoWeightsLoader(self).load_weights(language_weights)
        # The outer vLLM loader checks every named parameter, including the
        # separately stored projector. Only report it after a strict load.
        loaded |= self._load_connector_checkpoint(self.connector_checkpoint)

        def audio_weights():
            from .checkpoint import iter_view

            for key, value in iter_view(Path(self.audio_encoder_path) / "weights.json"):
                if key.startswith("thinker.audio_tower."):
                    yield key.removeprefix("thinker.audio_tower."), value

        audio_loaded = self.audio_tower.load_weights(audio_weights())
        expected = {key for key, _ in self.audio_tower.named_parameters()}
        if expected - audio_loaded:
            raise ValueError(f"Missing ASR encoder weights: {sorted(expected - audio_loaded)}")
        loaded |= {f"audio_tower.{name}" for name in audio_loaded}
        if self.audio_encoder_quantization:
            from .audio_encoder import install_packed_audio_encoder

            precision, packed, reference = self.audio_encoder_quantization
            layers = install_packed_audio_encoder(self.audio_tower, precision, packed, reference)
            # Packed tensors are strictly copied above; the outer loader handles
            # kernel finalization once after the complete model is loaded.
            loaded |= {f"audio_tower.{name}" for name, _ in self.audio_tower.named_parameters()}
            logging.getLogger(__name__).info(
                "DRVEXO audio encoder: precision=%s linears=%d packed=%s",
                precision,
                len(layers),
                packed,
            )
        return loaded


__all__ = ["DRVEXOSpeech8BForConditionalGeneration"]
