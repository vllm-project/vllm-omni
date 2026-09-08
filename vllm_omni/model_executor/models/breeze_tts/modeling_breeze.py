# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze-TTS-2: T5Gemma2 prompt encoding, paged Qwen3, and RVQ decoding."""

from collections.abc import Iterable
from typing import Any, TypedDict, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import record_function_or_nullcontext

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.breeze_tts.configuration_breeze import BreezeConfig
from vllm_omni.model_executor.models.breeze_tts.depth_decoder import BreezeDepthDecoder, sample_logits
from vllm_omni.model_executor.models.breeze_tts.prompt import CFG_UNCOND_SUFFIX
from vllm_omni.model_executor.models.breeze_tts.reference_encoder import BreezeReferenceEncoder
from vllm_omni.model_executor.models.breeze_tts.text_encoder_graph import (
    BreezeTextEncoderCompiled,
    BreezeTextEncoderGraph,
)
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
    requires_full_prefix_cached_hidden_states = False
    omni_pooler_payload_include_hidden = False
    use_async_omni_output = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = cast(BreezeConfig, vllm_config.model_config.hf_config)
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Breeze-TTS-2 currently supports TP=1 and PP=1")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Breeze prompt placeholders require enable_prefix_caching=False")
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("Breeze text encoding requires enable_chunked_prefill=False")
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
        text_config._attn_implementation = "sdpa"
        self.text_encoder = T5Gemma2TextEncoder(text_config)
        self.text_encoder_proj = nn.Linear(text_config.hidden_size, self.hidden_size, bias=False)
        self._long_text_encoder = BreezeTextEncoderCompiled(self.text_encoder, self.text_encoder_proj)
        self.reference_encoder = BreezeReferenceEncoder(vllm_config)
        self.depth_decoder = BreezeDepthDecoder(self.config.depth_decoder_config)
        self.lm_head = nn.Linear(self.hidden_size, self.config.vocab_size, bias=False, dtype=torch.float32)
        self.register_buffer(
            "offsets", torch.arange(self.num_codebooks) * self.config.audio_vocab_size, persistent=False
        )
        self.gpu_resident_buffer_keys = {
            ("breeze_state", "current"),
            ("breeze_state", "history"),
            ("breeze_prepared", "embeds"),
        }
        self._next_tokens: torch.Tensor | None = None
        self._text_graphs: dict[tuple[int, int], BreezeTextEncoderGraph] = {}

    @torch.inference_mode()
    def preprocess_batch(
        self, *, req_ids: list[str], model_intermediate_buffer: dict[str, dict[str, Any]], device: torch.device
    ) -> None:
        pending = [
            model_intermediate_buffer[rid] for rid in req_ids if "breeze_state" not in model_intermediate_buffer[rid]
        ]
        if not pending:
            return
        segment_ids: dict[tuple[int, ...], None] = {}
        references: dict[str, dict[str, Any]] = {}
        reference_keys: dict[str, str] = {}
        for info in pending:
            conditioning = info["breeze_prompt"]
            segment_ids[tuple(conditioning["target_ids"])] = None
            if "reference_waveform" in info:
                segment_ids[tuple(conditioning["reference_ids"])] = None
                request_id = info["global_request_id"][0]
                key = request_id.removesuffix(CFG_UNCOND_SUFFIX) if conditioning["role"] == "uncond" else request_id
                reference_keys[request_id] = key
                references.setdefault(key, info)
        if not self._text_graphs:
            # The serving warmup pays these captures before accepting speech
            # requests. Cover the default two-row deployment's distinct text
            # segments, including two independent reference recordings.
            for batch in range(1, min(4, 2 * self.vllm_config.scheduler_config.max_num_seqs) + 1):
                for size in (32, 64, 128):
                    self._text_graphs[(batch, size)] = BreezeTextEncoderGraph(
                        self.text_encoder, self.text_encoder_proj, size, batch_size=batch
                    )
            self.reference_encoder.warmup(min(2, self.vllm_config.scheduler_config.max_num_seqs))
            self._long_text_encoder.warmup(min(4, 2 * self.vllm_config.scheduler_config.max_num_seqs))
        length = max(len(ids) for ids in segment_ids)
        bucket = max(32, 1 << (length - 1).bit_length())
        with record_function_or_nullcontext("breeze:text_encoder"):
            prompts = [torch.tensor(ids, device="cpu", dtype=torch.long) for ids in segment_ids]
            if bucket > 128:
                encoded = self._long_text_encoder.run_batch(prompts)
                # Large prefill activations must not raise the decoder graph
                # pool's permanent high-water mark. Return free prefill
                # workspace to the device before starting streamed decode.
                torch.accelerator.empty_cache()
            else:
                graph_key = (len(segment_ids), bucket)
                if graph_key not in self._text_graphs:
                    self._text_graphs[graph_key] = BreezeTextEncoderGraph(
                        self.text_encoder, self.text_encoder_proj, bucket, batch_size=len(segment_ids)
                    )
                encoded = self._text_graphs[graph_key].run_batch(prompts)
        segments = dict(zip(segment_ids, encoded, strict=True))
        reference_embeds: dict[str, torch.Tensor] = {}
        if references:
            with record_function_or_nullcontext("breeze:reference_encoder"):
                codes = self.reference_encoder.encode_batch(
                    [info["reference_waveform"] for info in references.values()],
                    [info["breeze_prompt"]["reference_frames"] for info in references.values()],
                )
            for key, code in zip(references, codes, strict=True):
                # Audio EOS is a zero RVQ frame, distinct from generation EOS.
                eos = code.new_full((1, self.num_codebooks), self.config.codebook_eos_token_id)
                audio = self.depth_decoder.embed_tokens(torch.cat((code, eos)) + self.offsets).sum(1)
                reference_ids = tuple(references[key]["breeze_prompt"]["reference_ids"])
                reference_embeds[key] = torch.cat((segments[reference_ids], audio), dim=0)
        for info in pending:
            embeds = segments[tuple(info["breeze_prompt"]["target_ids"])]
            request_id = info["global_request_id"][0]
            if request_id in reference_keys:
                embeds = torch.cat((reference_embeds[reference_keys[request_id]], embeds), dim=0)
            info["breeze_prepared"] = {"embeds": embeds}

    def embed_input_ids(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        # vLLM's memory profiling does not have a request payload.
        return self.lm_head.weight.new_zeros((input_ids.numel(), self.hidden_size), dtype=self.model.norm.weight.dtype)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        *,
        breeze_prompt: dict[str, Any],
        breeze_sampling: BreezeSampling,
        global_request_id: list[str],
        breeze_prepared: dict[str, torch.Tensor] | None = None,
        breeze_state: BreezeState | None = None,
        _omni_is_prefill: bool,
        _omni_seed: int | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if _omni_is_prefill:
            if breeze_prepared is None:
                raise RuntimeError("Breeze prefill requires the runner's batch preprocessing hook")
            embeds = breeze_prepared.pop("embeds")
            if embeds.shape[0] != input_ids.numel():
                raise ValueError("Breeze prompt payload and scheduled token count differ")
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
        output_codes = [
            torch.empty((0, self.num_codebooks), device=model_outputs.device, dtype=torch.long)
            for _ in model_intermediate_buffer
        ]
        infos = model_intermediate_buffer
        request_rows = {cast(list[str], info["global_request_id"])[0]: index for index, info in enumerate(infos)}
        groups: dict[tuple[float, int, float, float], list[tuple[list[int], torch.Tensor, torch.Tensor]]] = {}
        for index, (info, (start, end)) in enumerate(zip(infos, request_token_spans, strict=True)):
            if not 0 <= start < end <= model_outputs.shape[0]:
                raise RuntimeError("Invalid Breeze request span")
            conditioning = cast(dict[str, Any], info["breeze_prompt"])
            if conditioning["role"] == "uncond":
                parent = cast(list[str], info["global_request_id"])[0].removesuffix(CFG_UNCOND_SUFFIX)
                if parent not in request_rows:
                    raise RuntimeError("Breeze CFG companion was scheduled without its conditioned branch")
                continue
            hidden = model_outputs[end - 1 : end]
            state = cast(BreezeState, info["breeze_state"])
            sampling = cast(BreezeSampling, info["breeze_sampling"])
            logits = F.linear(hidden.float(), self.lm_head.weight)
            indices = [index]
            guidance_scale = float(conditioning["guidance_scale"])
            if guidance_scale != 1.0:
                request_id = cast(list[str], info["global_request_id"])[0]
                companion = request_rows.get(request_id + CFG_UNCOND_SUFFIX)
                if companion is None:
                    raise RuntimeError("Breeze CFG request was scheduled without its unconditional branch")
                if infos[companion]["breeze_sampling"] != sampling:
                    raise RuntimeError("Breeze CFG branches have different sampling parameters")
                _, companion_end = request_token_spans[companion]
                uncond = model_outputs[companion_end - 1 : companion_end]
                uncond_logits = F.linear(uncond.float(), self.lm_head.weight)
                logits = uncond_logits + guidance_scale * (logits - uncond_logits)
                hidden = torch.cat((hidden, uncond))
                indices.append(companion)
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
                self._next_tokens[indices] = self.config.eos_token_id
                continue
            state["history"] = torch.cat((history, first[:, None]), dim=1)
            parameters = (sampling["temperature"], sampling["top_k"], sampling["top_p"], guidance_scale)
            groups.setdefault(parameters, []).append((indices, hidden, first))
        for (temperature, top_k, top_p, guidance_scale), rows in groups.items():
            # Depth batches place every conditioned row before every
            # unconditional row. Both branches consume the same sampled codes.
            hidden = torch.stack([row[1] for row in rows]).transpose(0, 1).reshape(-1, self.hidden_size)
            with record_function_or_nullcontext("breeze:depth_decoder"):
                frames = self.depth_decoder.generate_frames(
                    hidden,
                    torch.cat([row[2] for row in rows]),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    generators=[cast(BreezeState, infos[row[0][0]]["breeze_state"])["generator"] for row in rows],
                    guidance_scale=guidance_scale,
                )
            for offset, (indices, _, _) in enumerate(rows):
                frame = frames[offset : offset + 1]
                for index in indices:
                    cast(BreezeState, infos[index]["breeze_state"])["current"] = frame
                output_codes[indices[0]] = frame
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
        loaded.update("reference_encoder." + name for name in self.reference_encoder.load_weights(self.vllm_config))
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"Uninitialized Breeze parameters: {sorted(missing)}")
        return loaded
