# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native Qwen3.5 hybrid/vision stage with the trained temporal audio input."""

from collections.abc import Iterable
from pathlib import Path

import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_asr import _get_feat_extract_output_lengths
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.configs.qwen3_asr import Qwen3ASRAudioEncoderConfig

from .checkpoint import UnifiedBF16Checkpoint
from .decision import UnifiedClassifier, UnifiedPrefill
from .processing import UnifiedDummyInputs, UnifiedMultiModalProcessor, UnifiedProcessingInfo
from .temporal_projector import TemporalInputProjector, project_utterances


@MULTIMODAL_REGISTRY.register_processor(
    UnifiedMultiModalProcessor, info=UnifiedProcessingInfo, dummy_inputs=UnifiedDummyInputs
)
class DRVEXOSpeechQwen35ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # The bundle keeps its vision processor under assets/qwen35. The native
    # device-normalisation resolver reads the root processor config, so letting
    # it disable HF rescaling would feed raw pixels through an identity norm.
    supports_mm_device_do_normalize = False
    requires_request_computed_tokens = True
    have_multimodal_outputs = True
    omni_client_multimodal_output_keys = ("drvexo_general_scores", "drvexo_general_status")

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        return super().get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.bundle_root = Path(vllm_config.model_config.model)
        if config.input_projector_type != "temporal_residual_v1" or config.text_config.hidden_size != 2560:
            raise ValueError("Expected the trained 4B temporal connector and native hybrid backbone")
        self.input_projector = TemporalInputProjector(
            config.audio_feature_dim, config.text_config.hidden_size, config.temporal_projector_width
        )
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniMoeAudioEncoder(
                Qwen3ASRAudioEncoderConfig(**config.audio_encoder_config), prefix="audio_tower"
            )
        self.general_prefill = None
        UnifiedClassifier.validate_scheduler(vllm_config)

    def embed_multimodal(self, **kwargs: object):
        result, visited = [], set()
        for key in kwargs:
            if key in ("pixel_values", "image_embeds") and "image" not in visited:
                visited.add("image")
                image = {k: v for k, v in kwargs.items() if k in ("pixel_values", "image_embeds", "image_grid_thw")}
                result.extend(super().embed_multimodal(**image))
            elif key in ("audio_mels", "audio_embeds") and "audio" not in visited:
                visited.add("audio")
                if key == "audio_mels":
                    lengths = kwargs["audio_mel_lengths"]
                    output_lengths = _get_feat_extract_output_lengths(lengths)
                    features = self.audio_tower(
                        kwargs[key].to(self.audio_tower.dtype), feature_lens=lengths, aftercnn_lens=output_lengths
                    )
                    result.extend(project_utterances(self.input_projector, features, output_lengths))
                else:
                    values = kwargs[key] if isinstance(kwargs[key], (list, tuple)) else [kwargs[key]]
                    for value in values:
                        result.extend(project_utterances(self.input_projector, value))
        return tuple(result)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Audio keeps ordinary sequential positions; images retain native mRoPE.
        return super().get_mrope_input_positions(input_tokens, [f for f in mm_features if f.modality == "image"])

    def make_omni_output(
        self,
        model_outputs,
        *,
        model_intermediate_buffer=None,
        request_token_spans=None,
        request_computed_tokens=None,
        **kwargs,
    ):
        buffers = model_intermediate_buffer or []
        if not any(info.get("drvexo_general_decision") is not None for info in buffers):
            return model_outputs
        if self.general_prefill is None or request_token_spans is None or request_computed_tokens is None:
            raise ValueError("Native decision head or scheduler metadata is missing")
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs=self.general_prefill.process(
                model_outputs, buffers, request_token_spans, request_computed_tokens
            ),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        checkpoint = UnifiedBF16Checkpoint(self.bundle_root)
        state = {
            key.removeprefix("input_projector."): value
            for key, value in checkpoint.iter_component("connectors")
            if key.startswith("input_projector.")
        }
        self.input_projector.load_state_dict(state, strict=True)
        names = self.audio_tower.load_weights(
            (key.removeprefix("thinker.audio_tower."), value)
            for key, value in checkpoint.iter_component("audio_encoder")
            if key.startswith("thinker.audio_tower.")
        )
        missing = set(dict(self.audio_tower.named_parameters())) - names
        if missing:
            raise ValueError(f"Missing audio encoder weights: {sorted(missing)}")
        classifier = UnifiedClassifier(checkpoint)
        self.general_prefill = UnifiedPrefill(classifier)
        return loaded | {"input_projector." + key for key in state} | {"audio_tower." + key for key in names}
