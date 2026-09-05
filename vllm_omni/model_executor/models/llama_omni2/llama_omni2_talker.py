# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LLaMA-Omni 2 Talker model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

from vllm_omni.model_executor.models.output_templates import OmniOutput

TALKER_WEIGHTS_MAPPER = Qwen2Model.hf_to_vllm_mapper | WeightsMapper(
    orig_to_new_prefix={
        "speech_generator.input_proj.": "input_proj.",
        "speech_generator.gate.": "gate.",
        "speech_generator.model.": "language_model.",
        "model.": None,
        "lm_head.": None,
    }
)

TALKER_EOS_TOKEN_ID = 151643
TALKER_CODEC_TOKEN_OFFSET = 151666
TALKER_CODEC_VOCAB_SIZE = 6561


class LlamaOmni2TalkerForConditionalGeneration(nn.Module, SupportsPP):
    """Projected/gated Thinker features consumed by native vLLM Qwen2."""

    have_multimodal_outputs = True
    prefer_model_sampler = True
    has_preprocess = True
    has_postprocess = True
    preprocess_once_buffer_keys = {
        ("ids", "output"),
        ("embed", "decode"),
        ("hidden_states", "output"),
    }
    cumulative_postprocess_output_buffer_keys = {
        ("codes", "audio"),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        thinker_hidden_size = self.config.thinker_config.hidden_size
        talker_hidden_size = self.config.talker_config.hidden_size
        self._init_fusion_layers(
            thinker_hidden_size,
            talker_hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=self.config.talker_config,
            architectures=["Qwen2ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _init_fusion_layers(
        self,
        thinker_hidden_size: int,
        talker_hidden_size: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        self.input_proj = nn.Sequential(
            ColumnParallelLinear(
                thinker_hidden_size,
                thinker_hidden_size * 2,
                bias=True,
                gather_output=False,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "input_proj.0"),
                return_bias=False,
            ),
            nn.ReLU(),
            RowParallelLinear(
                thinker_hidden_size * 2,
                talker_hidden_size,
                bias=True,
                input_is_parallel=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "input_proj.2"),
                return_bias=False,
            ),
        )
        self.gate = nn.Sequential(
            ColumnParallelLinear(
                talker_hidden_size * 2,
                talker_hidden_size,
                bias=True,
                gather_output=True,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "gate.0"),
                return_bias=False,
            ),
            nn.Sigmoid(),
        )

    def _project_thinker_hidden_states(
        self,
        thinker_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.input_proj[0](thinker_hidden_states)
        hidden_states = self.input_proj[1](hidden_states)
        return self.input_proj[2](hidden_states)

    def fusion(
        self,
        representation: torch.Tensor,
        token_embedding: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.gate[0](torch.cat([representation, token_embedding], dim=-1))
        gate = self.gate[1](gate)
        return representation * gate + token_embedding * (1 - gate)

    def _validate_embedding_token_ids(
        self,
        token_ids: torch.Tensor,
    ) -> None:
        vocab_size = int(self.language_model.config.vocab_size)
        invalid_mask = (token_ids < 0) | (token_ids >= vocab_size)
        if bool(invalid_mask.any().item()):
            invalid_ids = token_ids[invalid_mask][:8].detach().cpu().tolist()
            raise ValueError(
                "LLaMA-Omni 2 token IDs must be within the Talker embedding "
                f"vocabulary range [0, {vocab_size}); got {invalid_ids}"
            )

    def prepare_talker_embeddings(
        self,
        thinker_hidden_states: torch.Tensor,
        thinker_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if thinker_hidden_states.shape[0] != thinker_token_ids.shape[0]:
            raise ValueError("thinker_hidden_states and thinker_token_ids must have the same number of rows")
        self._validate_embedding_token_ids(thinker_token_ids)
        representation = self._project_thinker_hidden_states(thinker_hidden_states)
        token_embedding = self.language_model.embed_input_ids(thinker_token_ids)
        return self.fusion(representation, token_embedding)

    @staticmethod
    def _payload_tensor(
        payload: dict[str, object],
        category: str,
        key: str,
    ) -> torch.Tensor | None:
        nested = payload.get(category)
        value = nested.get(key) if isinstance(nested, dict) else None
        if value is None:
            value = payload.get(f"{category}.{key}")
        if value is None:
            return None
        return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **payload: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if not bool(payload.get("_omni_is_prefill", input_ids.shape[0] > 1)):
            return (
                input_ids,
                self.language_model.embed_input_ids(input_ids),
                {},
            )

        del input_embeds
        ids = payload.get("ids")
        output_ids = ids.get("output") if isinstance(ids, dict) else None
        if output_ids is None:
            output_ids = payload.get("ids.output")
        if output_ids is None:
            raise ValueError("LLaMA-Omni 2 Talker requires ids.output")

        token_ids = torch.as_tensor(
            output_ids,
            dtype=torch.long,
            device=input_ids.device,
        ).reshape(-1)
        hidden_states = self._payload_tensor(
            payload,
            "hidden_states",
            "output",
        )
        if hidden_states is None:
            raise ValueError("LLaMA-Omni 2 Talker requires hidden_states.output")
        hidden_states = hidden_states.to(input_ids.device)

        hidden_count = hidden_states.shape[0]
        if token_ids.shape[0] not in (hidden_count, hidden_count + 1):
            raise ValueError(
                "LLaMA-Omni 2 Talker ids.output must have one token per "
                "Thinker hidden row, plus at most one terminal separator"
            )
        fused = self.prepare_talker_embeddings(
            hidden_states,
            token_ids[:hidden_count],
        )
        if token_ids.shape[0] == hidden_count + 1:
            self._validate_embedding_token_ids(token_ids[-1:])
            separator_embedding = self.language_model.embed_input_ids(token_ids[-1:])
            fused = torch.cat([fused, separator_embedding], dim=0)
        if fused.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "LLaMA-Omni 2 Talker scheduled prompt length does not match "
                f"the inter-stage payload: {input_ids.shape[0]} vs {fused.shape[0]}"
            )
        return token_ids.to(dtype=input_ids.dtype), fused, {}

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del multimodal_embeddings, is_multimodal
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
        thinker_hidden_states: torch.Tensor | None = None,
        thinker_token_ids: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | object:
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None and thinker_hidden_states is not None and thinker_token_ids is not None:
            inputs_embeds = self.prepare_talker_embeddings(
                thinker_hidden_states,
                thinker_token_ids,
            )
            input_ids = None

        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        return self.language_model.compute_logits(hidden_states)

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **_: object,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs={},
        )

    def sample(self, logits: torch.Tensor, sampling_metadata: object):
        codec_end = TALKER_CODEC_TOKEN_OFFSET + TALKER_CODEC_VOCAB_SIZE
        if logits.shape[-1] < codec_end:
            raise ValueError(
                "LLaMA-Omni 2 Talker logits must cover the EOS and codec "
                f"token range through {codec_end - 1}, got {logits.shape[-1]}"
            )
        logits[..., :TALKER_EOS_TOKEN_ID] = float("-inf")
        logits[..., TALKER_EOS_TOKEN_ID + 1 : TALKER_CODEC_TOKEN_OFFSET] = float("-inf")
        logits[..., codec_end:] = float("-inf")

        sampler = getattr(self, "_sampler", None)
        if sampler is None:
            from vllm.v1.sample.sampler import Sampler

            sampler = Sampler()
            self._sampler = sampler
        output = sampler(logits=logits, sampling_metadata=sampling_metadata)
        self._last_sampled_token_ids = output.sampled_token_ids
        self._postprocess_cursor = 0
        return output

    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: object = None,
        **_: object,
    ) -> dict[str, dict[str, torch.Tensor]]:
        del hidden_states_slice, multimodal_outputs
        sampled = getattr(self, "_last_sampled_token_ids", None)
        if not isinstance(sampled, torch.Tensor) or sampled.numel() == 0:
            return {}
        cursor = int(getattr(self, "_postprocess_cursor", 0))
        sampled = sampled.reshape(-1)
        if cursor >= sampled.shape[0]:
            return {}
        self._postprocess_cursor = cursor + 1
        return {
            "codes": {
                "audio": sampled[cursor : cursor + 1].to(dtype=torch.long).detach(),
            }
        }

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=TALKER_WEIGHTS_MAPPER)
