"""VibeVoice TTS – top-level routing wrapper.

Routes to either the audio-generation sub-model (Stage 0) or the
audio-tokenizer decoder (Stage 1) based on ``model_stage``.

Follows the same pattern as VoxtralTTSForConditionalGeneration.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cached_property

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.vibevoice_tts.vibevoice_tts_audio_generation import (
    VibeVoiceTTSDummyInputsBuilder,
    VibeVoiceTTSMultiModalProcessor,
    VibeVoiceTTSProcessingInfo,
)

logger = init_logger(__name__)


def parse_batched_latent_input(
    input_ids: torch.Tensor,
    latent_dim: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """Parse batched float-encoded latent input for the tokenizer stage.

    Each request is packed as:
        [ctx_frames(int), context_length(int), <flat float latents>]
    where flat latents have length (ctx_frames + context_length) * latent_dim,
    encoded as float32 values stored in a 1-D tensor.

    Returns:
        latent_list: list of (num_frames, latent_dim) tensors.
        ctx_frames_list: list of ctx_frames values.
    """
    latent_list: list[torch.Tensor] = []
    ctx_frames_list: list[int] = []
    offset = 0
    # input_ids is a flat float tensor
    data = input_ids.float()
    while offset < data.numel():
        ctx_frames = int(data[offset].item())
        context_length = int(data[offset + 1].item())
        offset += 2
        total_frames = ctx_frames + context_length
        n_values = total_frames * latent_dim
        chunk = data[offset: offset + n_values]
        offset += n_values
        latents = chunk.view(total_frames, latent_dim)
        latent_list.append(latents)
        ctx_frames_list.append(ctx_frames)
    return latent_list, ctx_frames_list


def apply_ctx_frames_cutting(
    waveforms: list[torch.Tensor],
    ctx_frames_list: list[int],
    downsample_factor: int,
) -> list[torch.Tensor]:
    """Remove leading context samples from decoded waveforms."""
    result: list[torch.Tensor] = []
    for wav, ctx_frames in zip(waveforms, ctx_frames_list):
        if ctx_frames > 0:
            cut = downsample_factor * ctx_frames
            wav = wav[cut:]
        result.append(wav)
    return result


@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceTTSMultiModalProcessor,
    info=VibeVoiceTTSProcessingInfo,
    dummy_inputs=VibeVoiceTTSDummyInputsBuilder,
)
class VibeVoiceTTSForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    CustomProcessMixin,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.config = config
        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "audio_generation":
            self.has_postprocess = True
            self.requires_raw_input_tokens = True
            self.set_custom_postprocess(self.tts_postprocess)
            self.audio_generation = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "audio_generation"),
                architectures=["VibeVoiceTTSAudioGeneration"],
            )
            self.model = self.audio_generation
            self.audio_tokenizer = None

        elif self.model_stage == "audio_tokenizer":
            self.requires_raw_input_tokens = True
            self.audio_generation = None
            self.audio_tokenizer = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "audio_tokenizer"),
                architectures=["VibeVoiceTTSAudioTokenizer"],
            )
            self.model = self.audio_tokenizer
        else:
            raise ValueError(f"Invalid model_stage: {self.model_stage}")

    # -----------------------------------------------------------------------
    # Postprocess
    # -----------------------------------------------------------------------

    def tts_postprocess(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: object,
        **info_dict: object | None,
    ) -> dict:
        update_dict = {}
        if isinstance(multimodal_outputs, dict) and "audio" in multimodal_outputs:
            idx = getattr(self, "_post_process_idx", 0)
            assert idx < len(multimodal_outputs["audio"])
            update_dict["audio"] = multimodal_outputs["audio"][idx]
            self._post_process_idx = idx + 1
        return update_dict

    # -----------------------------------------------------------------------
    # Sampler
    # -----------------------------------------------------------------------

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "audio_tokenizer":
            hidden_size = self.config.decoder_config.hidden_size
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, hidden_size)
        return self.model.embed_input_ids(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self.model_stage == "audio_generation":
            if inputs_embeds is not None:
                input_ids = None
            hidden_states = self.model(
                input_ids, positions, intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
            return hidden_states

        if self.model_stage == "audio_tokenizer":
            latent_dim = self.config.acoustic_vae_dim
            downsample_factor = self.audio_tokenizer.downsample_factor

            # Dummy run detection
            if (input_ids == 0).all():
                logger.info("audio_tokenizer: sample run with dummy input")
                dummy_latents = torch.randn(1, 10, latent_dim,
                                            device=input_ids.device, dtype=torch.float32)
                with torch.no_grad():
                    wav = self.audio_tokenizer.decode(dummy_latents)
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"audio": [wav.squeeze(0).squeeze(0)]},
                )

            latent_list, ctx_frames_list = parse_batched_latent_input(
                input_ids, latent_dim=latent_dim,
            )

            with torch.no_grad():
                waveforms = self.audio_tokenizer.decode_batch(latent_list)

            waveforms = apply_ctx_frames_cutting(
                waveforms, ctx_frames_list, downsample_factor,
            )

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": waveforms},
            )

        raise ValueError(f"Unexpected model_stage: {self.model_stage}")

    # -----------------------------------------------------------------------
    # Omni output
    # -----------------------------------------------------------------------

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput | tuple,
        logits_index: int | None = None,
        **kwargs,
    ) -> OmniOutput:
        if isinstance(model_outputs, torch.Tensor):
            if self.model_stage == "audio_generation":
                hidden_states = model_outputs
                assert logits_index is not None
                input_hidden_states = hidden_states[logits_index]
                fake_eos, multimodal_outputs = self.model.compute_mm_logits(
                    input_hidden_states,
                )
                hidden_states[logits_index, 0] = fake_eos
                return OmniOutput(
                    text_hidden_states=hidden_states,
                    multimodal_outputs=multimodal_outputs,
                )
            raise ValueError(f"Unsupported model_stage={self.model_stage} for tensor output")

        if isinstance(model_outputs, OmniOutput):
            if self.model_stage == "audio_tokenizer":
                return model_outputs
            raise ValueError(f"Unsupported model_stage={self.model_stage} for OmniOutput")

        raise ValueError(f"Unsupported model_outputs type: {type(model_outputs)}")

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "audio_generation":
            return self.model.fake_logits_for_audio_tokens(fake_eos=hidden_states)
        raise ValueError(f"compute_logits not supported for model_stage={self.model_stage}")

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if hasattr(self.model, "sample"):
            return self.model.sample(logits, sampling_metadata)
        return self.sampler(logits, sampling_metadata)

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights: set[str] = set()

        # Split weights between audio_tokenizer and audio_generation.
        # Semantic tokenizer weights are training-only and skipped at inference.
        tokenizer_prefix = "model.acoustic_tokenizer."
        decoder_prefix = "acoustic_tokenizer."
        skip_prefixes = ("model.semantic_tokenizer.", "semantic_tokenizer.")

        remapping_rules = [
            (r"^model\.acoustic_tokenizer\.decoder\.(.*)$", r"decoder.\1"),
            (r"^acoustic_tokenizer\.decoder\.(.*)$", r"decoder.\1"),
        ]

        def gen_weights():
            nonlocal loaded_weights
            for name, w in weights:
                if any(name.startswith(p) for p in skip_prefixes):
                    continue

                is_tokenizer = (name.startswith(tokenizer_prefix)
                                or name.startswith(decoder_prefix)
                                or name.startswith("model.acoustic_tokenizer."))

                if is_tokenizer and self.audio_tokenizer is not None:
                    remapped = name
                    for pattern, repl in remapping_rules:
                        if re.fullmatch(pattern, remapped):
                            remapped = re.sub(pattern, repl, remapped)
                            break
                    loaded_name = self.audio_tokenizer.load_weight((remapped, w))
                    loaded_weights.add(f"audio_tokenizer.{loaded_name}")
                    continue

                yield (name, w)

        remaining = list(gen_weights())

        if self.audio_generation is not None:
            for name in self.audio_generation.load_weights(remaining):
                loaded_weights.add(f"audio_generation.{name}")

        return loaded_weights
