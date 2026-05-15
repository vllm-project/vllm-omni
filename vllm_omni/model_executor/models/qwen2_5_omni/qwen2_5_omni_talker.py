from collections.abc import Iterable
from functools import cached_property

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTalkerConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


class Qwen2_5OmniTalkerForConditionalGeneration(
    nn.Module,
    SupportsPP,
):
    logger = init_logger(__name__)
    # Align to thinker-style static mapper for clarity
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # text LM head/body in talker
            "talker.codec_head.": "language_model.lm_head.",
            "talker.model.": "language_model.model.",
            # projection weights
            "talker.thinker_to_talker_proj.": "thinker_to_talker_proj.",
            # fallback root
            "talker.": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: Qwen2_5OmniTalkerConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.quant_config = quant_config

        if hasattr(config, "talker_config"):
            self.config = config.talker_config
            vllm_config.model_config.hf_text_config = vllm_config.model_config.hf_config.talker_config
        else:
            self.config = config

        self.thinker_to_talker_proj = nn.Linear(
            self.config.embedding_size,
            self.config.hidden_size,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            hf_config=getattr(self.config, "text_config", self.config),
            architectures=["Qwen2ForCausalLM_old"],
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # suppress start id
        self.suppress_start_id = None

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return Sampler()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed codec input IDs."""
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        positions: torch.Tensor = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        assert input_ids is not None or inputs_embeds is not None, "input_ids or inputs_embeds must be provided"

        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        input_ids = None

        # projection
        inputs_embeds = self.thinker_to_talker_proj(inputs_embeds)

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def bad_word_processor(self, logits: torch.Tensor) -> torch.Tensor:
        # suppress token IDs unsupported by token2wav
        if self.suppress_start_id and self.suppress_start_id < logits.size(-1):
            # skip the end token id.
            if hasattr(self.config, "tts_codec_end_token_id"):
                end_id = int(getattr(self.config, "tts_codec_end_token_id"))
                if self.suppress_start_id == end_id:
                    logits[..., end_id + 1 : logits.size(-1)] = -1e9
                elif self.suppress_start_id < end_id:
                    logits[..., self.suppress_start_id : end_id] = -1e9
                    logits[..., end_id + 1 : logits.size(-1)] = -1e9
                else:
                    logits[..., self.suppress_start_id : logits.size(-1)] = -1e9
            else:
                raise ValueError("config must have tts_codec_end_token_id attribute")

        if hasattr(self.config, "tts_codec_start_token_id"):
            bos_id = int(getattr(self.config, "tts_codec_start_token_id"))
            logits[..., bos_id] = -1e9
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.language_model.compute_logits(hidden_states)
        logits = self.bad_word_processor(logits)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "token2wav."],
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            self.logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            pass
        return loaded

    def set_suppress_start_id(self, start_id: int):
        self.suppress_start_id = start_id
        self.logger.debug(f"Set suppress start id to {self.suppress_start_id}")
