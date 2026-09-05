# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze-TTS-2: T5Gemma2 prompt encoding, paged Qwen3, and RVQ decoding."""

from collections.abc import Iterable
from typing import TypedDict, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.breeze_tts.configuration_breeze import BreezeConfig
from vllm_omni.model_executor.models.breeze_tts.depth_decoder import BreezeDepthDecoder, sample_logits
from vllm_omni.model_executor.models.output_templates import OmniOutput


class BreezeSampling(TypedDict):
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float


class BreezeState(TypedDict):
    generator: torch.Generator
    history: torch.Tensor
    current: torch.Tensor


class BreezeForConditionalGeneration(nn.Module):
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = cast(BreezeConfig, vllm_config.model_config.hf_config)
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Breeze-TTS-2 currently supports TP=1 and PP=1")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Breeze prompt placeholders require enable_prefix_caching=False")
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("Breeze text encoding requires enable_chunked_prefill=False")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("The initial Breeze integration requires enforce_eager=True")
        if not cast(OmniModelConfig, vllm_config.model_config).async_chunk:
            raise ValueError("Breeze requires async_chunk=True for its stateful codec stage")
        if vllm_config.model_config.quantization is not None:
            raise ValueError("The initial Breeze integration does not support quantization")
        if self.config.backbone_model_type != "qwen3" or self.config.text_encoder_proj_type != "linear":
            raise ValueError("This integration supports the Breeze-TTS-2 Qwen3/linear checkpoint")
        if self.config.num_codebooks != 16 or self.config.audio_vocab_size != 2051:
            raise ValueError("Breeze-TTS-2 requires 16 codebooks with 2048 audio codes and 3 reserved IDs")
        self.num_codebooks = self.config.num_codebooks
        self.codebook_size = self.config.audio_vocab_size - 3
        self.hidden_size = self.config.backbone_config.hidden_size
        backbone_config = vllm_config.with_hf_config(self.config.backbone_config, architectures=["Qwen3ForCausalLM"])
        backbone_config.model_config.hf_text_config = backbone_config.model_config.hf_config
        self.model = Qwen3Model(vllm_config=backbone_config, prefix=f"{prefix}.model".strip("."))
        # Breeze replaces the backbone token embedding with summed RVQ embeddings.
        self.model.embed_tokens = nn.Identity()
        text_config = self.config.text_encoder_config
        text_config._attn_implementation = "eager"
        self.text_encoder = T5Gemma2TextEncoder(text_config)
        self.text_encoder_proj = nn.Linear(text_config.hidden_size, self.hidden_size, bias=False)
        self.depth_decoder = BreezeDepthDecoder(self.config.depth_decoder_config)
        self.lm_head = nn.Linear(self.hidden_size, self.config.vocab_size, bias=False, dtype=torch.float32)
        self.register_buffer(
            "offsets", torch.arange(self.num_codebooks) * self.config.audio_vocab_size, persistent=False
        )
        self.gpu_resident_buffer_keys = {("breeze_state", "current"), ("breeze_state", "history")}
        self._next_tokens: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        # vLLM's memory profiling does not have a request payload.
        return self.lm_head.weight.new_zeros((input_ids.numel(), self.hidden_size), dtype=self.model.norm.weight.dtype)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        *,
        ids: dict[str, list[int]],
        breeze_sampling: BreezeSampling,
        breeze_state: BreezeState | None = None,
        _omni_is_prefill: bool,
        _omni_seed: int | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if _omni_is_prefill:
            prompt = torch.tensor(ids["prompt"], device=input_ids.device, dtype=torch.long).unsqueeze(0)
            if prompt.shape[1] != input_ids.numel():
                raise ValueError("Breeze prompt payload and scheduled token count differ")
            encoded = self.text_encoder(input_ids=prompt, attention_mask=torch.ones_like(prompt)).last_hidden_state
            embeds = self.text_encoder_proj(encoded).squeeze(0)
            generator = torch.Generator(device=input_ids.device)
            if _omni_seed is None:
                generator.seed()
            else:
                generator.manual_seed(_omni_seed)
            state: BreezeState = {
                "generator": generator,
                "history": input_ids.new_empty((1, 0)),
                "current": input_ids.new_empty((1, self.num_codebooks)),
            }
            return input_ids, embeds, {"breeze_state": state}
        if breeze_state is None or input_ids.numel() != 1:
            raise ValueError("Breeze decode requires initialized per-request state and one token")
        embeds = self.depth_decoder.embed_tokens(breeze_state["current"] + self.offsets).sum(1)
        return input_ids, embeds, {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(None, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata | None = None
    ) -> torch.Tensor:
        logits = hidden_states.new_full((hidden_states.shape[0], self.config.vocab_size), -torch.inf)
        if self._next_tokens is None:
            logits[:, 0] = 0
        else:
            if self._next_tokens.numel() != hidden_states.shape[0]:
                raise RuntimeError("Breeze sampled rows do not match the current request batch")
            logits.scatter_(1, self._next_tokens[:, None], 0)
        return logits

    def make_omni_output(
        self,
        model_outputs: torch.Tensor,
        *,
        model_intermediate_buffer: list[dict[str, object]] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        **_: object,
    ) -> OmniOutput:
        if not model_intermediate_buffer:
            self._next_tokens = None
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})
        if request_token_spans is None or len(request_token_spans) != len(model_intermediate_buffer):
            raise RuntimeError("Breeze requires one hidden-state span per request")
        self._next_tokens = torch.zeros(len(model_intermediate_buffer), device=model_outputs.device, dtype=torch.long)
        output_codes: list[torch.Tensor] = []
        for index, (info, (start, end)) in enumerate(zip(model_intermediate_buffer, request_token_spans, strict=True)):
            if not 0 <= start < end <= model_outputs.shape[0]:
                raise RuntimeError("Invalid Breeze request span")
            hidden = model_outputs[end - 1 : end]
            state = cast(BreezeState, info["breeze_state"])
            sampling = cast(BreezeSampling, info["breeze_sampling"])
            logits = F.linear(hidden.float(), self.lm_head.weight)
            logits[:, self.codebook_size : self.config.eos_token_id] = -torch.inf
            history = state["history"]
            if history.numel():
                selected = logits.gather(1, history)
                penalty = sampling["repetition_penalty"]
                logits.scatter_(1, history, torch.where(selected < 0, selected * penalty, selected / penalty))
            first = sample_logits(
                logits, sampling["temperature"], sampling["top_k"], sampling["top_p"], state["generator"]
            )
            if first.item() == self.config.eos_token_id:
                self._next_tokens[index] = self.config.eos_token_id
                output_codes.append(first.new_empty((0, self.num_codebooks)))
                continue
            frame = self.depth_decoder.generate_frame(
                hidden,
                first,
                temperature=sampling["temperature"],
                top_k=sampling["top_k"],
                top_p=sampling["top_p"],
                generator=state["generator"],
            )
            state["current"] = frame
            state["history"] = torch.cat((history, first[:, None]), dim=1)
            output_codes.append(frame)
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={"codes": {"audio": output_codes}})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone: list[tuple[str, torch.Tensor]] = []
        depth: dict[str, torch.Tensor] = {}
        loaded: set[str] = set()
        params = dict(self.named_parameters())
        for name, value in weights:
            if name.startswith(("backbone_model.layers.", "backbone_model.norm.")):
                backbone.append((name.removeprefix("backbone_model."), value))
            elif name.startswith("depth_decoder."):
                depth[name] = value
            elif name.startswith(("text_encoder.", "text_encoder_proj.", "lm_head.")):
                default_weight_loader(params[name], value)
                loaded.add(name)
            elif not name.startswith(("codec_model.", "backbone_model.embed_tokens.", "embed_text_tokens.")):
                raise ValueError(f"Unexpected Breeze checkpoint parameter: {name}")
        loaded.update("model." + name for name in self.model.load_weights(backbone))
        for target, parameter in self.depth_decoder.named_parameters():
            if target == "codebooks_head":
                value = depth["depth_decoder.codebooks_head.weight"]
            elif target.endswith("qkv.weight"):
                layer = target.split(".")[1]
                value = torch.cat(
                    [depth[f"depth_decoder.model.layers.{layer}.self_attn.{p}_proj.weight"] for p in ("q", "k", "v")]
                )
            elif target.endswith("gate_up.weight"):
                layer = target.split(".")[1]
                value = torch.cat(
                    [depth[f"depth_decoder.model.layers.{layer}.mlp.{p}_proj.weight"] for p in ("gate", "up")]
                )
            else:
                source = target.replace(".o_proj.", ".self_attn.o_proj.").replace(".down_proj.", ".mlp.down_proj.")
                value = depth["depth_decoder.model." + source]
            default_weight_loader(parameter, value)
            loaded.add("depth_decoder." + target)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"Uninitialized Breeze parameters: {sorted(missing)}")
        return loaded
