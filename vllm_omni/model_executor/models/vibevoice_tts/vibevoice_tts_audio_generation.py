"""VibeVoice TTS Stage 0: Qwen2.5 LLM + Diffusion Head.

The LLM autoregressively predicts hidden states. At each decode step the
diffusion head denoises a Gaussian sample into a continuous acoustic latent
(dim=64) conditioned on the LLM hidden state.  The sampler is driven by
fake logits that force the next token to be either EOS or an audio
placeholder, while the real audio payload is carried in the multimodal
side-channel (OmniOutput).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
)
from vllm.multimodal.processing.processor import (
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.vibevoice_tts.vibevoice_tts_diffusion_head import (
    DPMSolverScheduler,
    VibeVoiceDiffusionHead,
)

logger = init_logger(__name__)

SAMPLING_RATE = 24000
FRAME_RATE = 7.5
DOWNSAMPLE_FACTOR = 3200  # 24000 / 7.5


# ---------------------------------------------------------------------------
# SpeechConnector: projects tokenizer latents into LLM hidden space
# ---------------------------------------------------------------------------

class SpeechConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = nn.RMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(features)))


# ---------------------------------------------------------------------------
# Multimodal processor stubs (required by vLLM multimodal registry)
# ---------------------------------------------------------------------------

class VibeVoiceTTSProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return None

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 0}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"audio": 0}


class VibeVoiceTTSDummyInputsBuilder(BaseDummyInputsBuilder[VibeVoiceTTSProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        return {}

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> ProcessorInputs:
        return ProcessorInputs(prompt=[1], mm_data_items=self.info.parse_mm_data({}, validate=False))


class VibeVoiceTTSMultiModalProcessor(BaseMultiModalProcessor[VibeVoiceTTSProcessingInfo]):
    def _get_mm_fields_config(
        self, hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {}

    def _get_prompt_updates(
        self, mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []


# ---------------------------------------------------------------------------
# Stage 0 model
# ---------------------------------------------------------------------------

@MULTIMODAL_REGISTRY.register_processor(
    VibeVoiceTTSMultiModalProcessor,
    info=VibeVoiceTTSProcessingInfo,
    dummy_inputs=VibeVoiceTTSDummyInputsBuilder,
)
class VibeVoiceTTSAudioGenerationForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        # Qwen2.5 backbone
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.decoder_config,
            prefix=maybe_prefix(prefix, "model.language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        hidden_size = config.decoder_config.hidden_size
        vocab_size = config.decoder_config.vocab_size

        # Diffusion head
        self.diffusion_head = VibeVoiceDiffusionHead(config.diffusion_head_config)

        # Noise scheduler for inference
        self.noise_scheduler = DPMSolverScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
        )
        self._num_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Speech connectors
        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, hidden_size)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, hidden_size)

        # Scaling/bias for latent normalization (loaded from checkpoint)
        self.register_buffer("speech_scaling_factor", torch.tensor(1.0))
        self.register_buffer("speech_bias_factor", torch.tensor(0.0))

        # Token IDs
        self.vocab_size = vocab_size
        self.eos_tok_id = getattr(config.decoder_config, "eos_token_id", 151643)
        # Audio placeholder token – use a reserved token from Qwen2.5 vocabulary.
        # This will be configured at runtime from the tokenizer if available.
        self._audio_tok_id = vocab_size - 1

        self._latent_size = config.diffusion_head_config.latent_size

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds,
        )
        return hidden_states

    # -----------------------------------------------------------------------
    # Diffusion inference
    # -----------------------------------------------------------------------

    def _run_diffusion_inference(self, condition: torch.Tensor) -> torch.Tensor:
        """Run reverse diffusion to generate acoustic latent from LLM hidden."""
        B = condition.shape[0]
        device = condition.device
        dtype = condition.dtype

        sample = torch.randn(B, self._latent_size, device=device, dtype=dtype)

        self.noise_scheduler.set_timesteps(self._num_inference_steps, device=device)

        for t in self.noise_scheduler.timesteps:
            timesteps = t.expand(B).to(device)
            model_output = self.diffusion_head(sample, timesteps.float(), condition)
            sample = self.noise_scheduler.step(model_output, t, sample)

        # Un-normalize using checkpoint-loaded scaling/bias
        sf = self.speech_scaling_factor.to(dtype)
        bf = self.speech_bias_factor.to(dtype)
        if sf.abs() > 1e-8:
            sample = sample / sf - bf

        return sample

    # -----------------------------------------------------------------------
    # MM logits (called from wrapper's make_omni_output)
    # -----------------------------------------------------------------------

    def compute_mm_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]] | None]:
        """Generate acoustic latent from LLM hidden and produce fake EOS flag.

        Args:
            hidden_states: (B, D) last hidden state of the LLM.

        Returns:
            fake_eos: (B, 1) float tensor, 1.0 for EOS, 0.0 for continue.
            mm_tokens: dict with "audio" key containing list of (1, latent_size) tensors.
        """
        # Check if the LLM wants to emit EOS by looking at text logits
        embed_weight = self.language_model.model.embed_tokens.weight
        text_logits = F.linear(hidden_states, embed_weight).float()
        predicted_token = text_logits.argmax(dim=-1)
        is_eos = (predicted_token == self.eos_tok_id)

        acoustic_latent = self._run_diffusion_inference(hidden_states)
        # acoustic_latent: (B, latent_size)

        fake_eos = torch.where(
            is_eos,
            torch.tensor(1.0, dtype=torch.bfloat16, device=hidden_states.device),
            torch.tensor(0.0, dtype=torch.bfloat16, device=hidden_states.device),
        )

        # Pack as list of (1, latent_size) for per-step multimodal output
        audio_list = list(acoustic_latent.unsqueeze(1).split(1, dim=0))
        mm_tokens = {"audio": audio_list}

        return fake_eos, mm_tokens

    def fake_logits_for_audio_tokens(
        self,
        fake_eos: torch.Tensor,
    ) -> torch.Tensor:
        """Produce fake logits to force the sampler to pick audio-tok or EOS."""
        shape = (fake_eos.shape[0], self.vocab_size)
        fake_logits = torch.full(shape, float("-inf"), device=fake_eos.device)
        is_eos = fake_eos[:, 0].bool()
        fake_logits[is_eos, self.eos_tok_id] = 1.0
        fake_logits[~is_eos, self._audio_tok_id] = 1.0
        return fake_logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        embeds = self.language_model.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            embeds[is_multimodal] = multimodal_embeddings[: is_multimodal.sum()].to(embeds.dtype)
        return embeds

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        remapping_rules = [
            (r"^model\.prediction_head\.(.*)$", r"diffusion_head.\1"),
            (r"^model\.acoustic_connector\.(.*)$", r"acoustic_connector.\1"),
            (r"^model\.semantic_connector\.(.*)$", r"semantic_connector.\1"),
            (r"^model\.language_model\.(.*)$", r"language_model.\1"),
            (r"^model\.speech_scaling_factor$", r"speech_scaling_factor"),
            (r"^model\.speech_bias_factor$", r"speech_bias_factor"),
        ]

        loaded_weights: set[str] = set()
        llm_weights: list[tuple[str, torch.Tensor]] = []

        for name, w in weights:
            # Apply remapping
            remapped = name
            for pattern, repl in remapping_rules:
                if re.fullmatch(pattern, remapped):
                    remapped = re.sub(pattern, repl, remapped)
                    break

            if remapped.startswith("diffusion_head."):
                sub_name = remapped[len("diffusion_head."):]
                loaded_name = self.diffusion_head.load_weight((sub_name, w))
                loaded_weights.add(f"diffusion_head.{loaded_name}")
            elif remapped.startswith("acoustic_connector."):
                params = dict(self.acoustic_connector.named_parameters())
                sub_name = remapped[len("acoustic_connector."):]
                if sub_name in params:
                    default_weight_loader(params[sub_name], w)
                    loaded_weights.add(remapped)
            elif remapped.startswith("semantic_connector."):
                params = dict(self.semantic_connector.named_parameters())
                sub_name = remapped[len("semantic_connector."):]
                if sub_name in params:
                    default_weight_loader(params[sub_name], w)
                    loaded_weights.add(remapped)
            elif remapped in ("speech_scaling_factor", "speech_bias_factor"):
                buf = getattr(self, remapped)
                buf.copy_(w)
                loaded_weights.add(remapped)
            elif remapped.startswith("language_model."):
                sub_name = remapped[len("language_model."):]
                llm_weights.append((f"model.{sub_name}", w))
            elif remapped == "lm_head.weight":
                pass
            else:
                logger.debug("Skipping unrecognized weight: %s", name)

        for loaded_name in self.language_model.load_weights(llm_weights):
            loaded_weights.add(f"language_model.{loaded_name}")

        return loaded_weights
