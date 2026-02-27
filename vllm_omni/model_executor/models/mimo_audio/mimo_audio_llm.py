# Copyright 2025 Xiaomi Corporation.
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature, DynamicCache, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Model as TransformerQwen2Model,
)
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioFeatureInputs,
    Qwen2AudioMultiModalDataParser,
    Qwen2AudioProcessingInfo,
    _get_feat_extract_output_lengths,
    _qwen2audio_field_config,
)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioInputs as MimoAudioInputs,
)
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    is_pp_missing_parameter,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema

from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import MiMoAudioConfig
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

# CUDA Graph buckets for MiMo local decoding / input_local_transformer.
# We keep the list small to balance warmup time and runtime coverage.
MIMO_CUDAGRAPH_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 6, 8, 16, 32, 64, 128)


@dataclass
class MiMoSampler:
    do_sample: bool | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    def process(self, scores: torch.Tensor):
        if self.temperature is not None:
            scores = scores / self.temperature

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, scores.shape[-1])
            indices_to_remove = scores < torch.topk(scores, top_k)[0][:, -1]
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        if self.top_p is not None and 0.0 < self.top_p <= 1.0:
            top_p = self.top_p if 0.0 < self.top_p <= 1.0 else 1.0
            sorted_logits, sorted_indices = torch.sort(scores)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[:, -1] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        return scores

    def sample(self, scores: torch.Tensor, removed_tokens: list[int] | None = None):
        scores = self.process(scores)
        for t in removed_tokens or []:
            scores[:, t] = float("-inf")

        if self.do_sample:
            probs = scores.softmax(dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        return torch.argmax(scores, dim=-1)


# CUDA Graph implementation for local_forward, adapted from sglang's mimo_audio
# Based on work by yanyihan@xiaomi.com
@dataclass
class MiMoLocalSamplerTensor:
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor

    def process(self, scores: torch.Tensor) -> torch.Tensor:
        if self.temperature is not None:
            scores = scores / self.temperature[:, None]

        if self.top_k is not None:
            _, sorted_indices = torch.sort(scores, descending=True)
            ranks = torch.arange(scores.size(-1), device=scores.device)
            ranks = ranks[None, :].expand_as(sorted_indices)
            sorted_indices_to_remove = (self.top_k[:, None] > 0) & (ranks >= self.top_k[:, None])
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        if self.top_p is not None:
            sorted_logits, sorted_indices = torch.sort(scores)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= torch.sub(1, self.top_p[:, None])
            sorted_indices_to_remove[:, -1] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores = scores.masked_fill(indices_to_remove, float("-inf"))

        return scores

    def sample(self, scores: torch.Tensor, removed_tokens: list[int] | None = None) -> torch.Tensor:
        if removed_tokens is not None and len(removed_tokens) > 0:
            scores = scores.clone()
            for t in removed_tokens:
                scores[:, t] = float("-inf")

        scores = self.process(scores)
        return torch.argmax(scores, dim=-1)


class MiMoLocalDecodeBuffer:
    def __init__(self, model: "MiMoAudioLLMForConditionalGeneration", max_batch_size: int):
        self.max_batch_size = max_batch_size

        device = next(model.hidden_states_downcast.parameters()).device
        dtype = next(model.hidden_states_downcast.parameters()).dtype
        hidden_size = model.local_config.hidden_size

        self.pool = torch.cuda.graph_pool_handle()
        self.input_tensor = torch.zeros((max_batch_size, 1, hidden_size), dtype=dtype, device=device)
        self.sampler = MiMoLocalSamplerTensor(
            temperature=torch.ones(max_batch_size, dtype=torch.float32, device=device),
            top_k=torch.full((max_batch_size,), -1, dtype=torch.int64, device=device),
            top_p=torch.ones(max_batch_size, dtype=torch.float32, device=device),
        )
        self.lock = threading.Lock()

    def inputs(self, batch_size: int = 1):
        sampler = MiMoLocalSamplerTensor(
            temperature=self.sampler.temperature[:batch_size],
            top_k=self.sampler.top_k[:batch_size],
            top_p=self.sampler.top_p[:batch_size],
        )
        return self.input_tensor[:batch_size], sampler

    def prepare(self, input_tensor: torch.Tensor, sampler: MiMoSampler | MiMoLocalSamplerTensor):
        b = input_tensor.shape[0]
        assert b <= self.max_batch_size, f"Expected batch size <= {self.max_batch_size}, got {b}"

        # Be tolerant to shape mismatches (e.g. caller passes [b, hs] / [b, 1, hs]).
        if input_tensor.ndim != self.input_tensor.ndim:
            input_tensor = input_tensor.reshape(self.input_tensor[:b].shape)
        self.input_tensor[:b].copy_(input_tensor)

        # When replaying a fixed-batch CUDA graph with b < max_batch_size, we must
        # sanitize the tail to avoid stale values affecting computation.
        if b < self.max_batch_size:
            self.input_tensor[b : self.max_batch_size].zero_()

        if isinstance(sampler, MiMoSampler):
            temperature = 1.0 if sampler.temperature is None else float(sampler.temperature)
            top_k = -1 if sampler.top_k is None else int(sampler.top_k)
            top_p = 1.0 if sampler.top_p is None else float(sampler.top_p)
            self.sampler.temperature[:b].fill_(temperature)
            self.sampler.top_k[:b].fill_(top_k)
            self.sampler.top_p[:b].fill_(top_p)

            if b < self.max_batch_size:
                self.sampler.temperature[b : self.max_batch_size].fill_(1.0)
                self.sampler.top_k[b : self.max_batch_size].fill_(-1)
                self.sampler.top_p[b : self.max_batch_size].fill_(1.0)

        else:
            self.sampler.temperature[:b].copy_(sampler.temperature)
            self.sampler.top_k[:b].copy_(sampler.top_k)
            self.sampler.top_p[:b].copy_(sampler.top_p)

            if b < self.max_batch_size:
                self.sampler.temperature[b : self.max_batch_size].fill_(1.0)
                self.sampler.top_k[b : self.max_batch_size].fill_(-1)
                self.sampler.top_p[b : self.max_batch_size].fill_(1.0)


class MiMoLocalDecodeCudaGraph:
    def __init__(
        self,
        cuda_graph: torch.cuda.CUDAGraph,
        buffer: MiMoLocalDecodeBuffer,
        output_tensor: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.cuda_graph = cuda_graph
        self.buffer = buffer
        self.output_tensor = output_tensor
        self.batch_size = batch_size

    @classmethod
    def capture(
        cls,
        model: "MiMoAudioLLMForConditionalGeneration",
        buffer: MiMoLocalDecodeBuffer,
        batch_size: int = 1,
        eager_run_first: bool = True,
    ) -> "MiMoLocalDecodeCudaGraph":
        input_tensor, sampler = buffer.inputs(batch_size)

        cuda_graph = torch.cuda.CUDAGraph()
        if eager_run_first:
            model.base_local_forward(input_tensor, local_sampler=sampler)
        with torch.cuda.graph(cuda_graph, buffer.pool):
            output_tensor = model.base_local_forward(input_tensor, local_sampler=sampler)

        return cls(
            cuda_graph=cuda_graph,
            buffer=buffer,
            output_tensor=output_tensor,
            batch_size=batch_size,
        )

    def forward(self, local_embeds: torch.Tensor, local_sampler: MiMoSampler | MiMoLocalSamplerTensor) -> torch.Tensor:
        b = local_embeds.shape[0]
        assert b <= self.batch_size, f"Expected batch size <= {self.batch_size}, got {b}"
        with self.buffer.lock:
            self.buffer.prepare(local_embeds, local_sampler)

            self.cuda_graph.replay()

            if self.output_tensor.dim() == 2:
                return self.output_tensor.clone()
            return self.output_tensor[:b].clone()


class MiMoInputLocalTransformerBuffer:
    def __init__(self, model: "MiMoAudioLLMForConditionalGeneration", max_batch_size: int) -> None:
        self.max_batch_size = max_batch_size

        device = next(model.input_local_transformer.parameters()).device
        dtype = next(model.input_local_transformer.parameters()).dtype
        hidden_size = model.input_local_config.hidden_size
        group_size = model.group_size

        self.pool = torch.cuda.graph_pool_handle()
        self.input_tensor = torch.zeros((max_batch_size, group_size, hidden_size), dtype=dtype, device=device)
        self.lock = threading.Lock()

    def inputs(self, batch_size: int) -> torch.Tensor:
        return self.input_tensor[:batch_size]

    def prepare(self, input_tensor: torch.Tensor) -> None:
        b = int(input_tensor.shape[0])
        assert b <= self.max_batch_size, f"Expected batch size <= {self.max_batch_size}, got {b}"

        if input_tensor.shape != self.input_tensor[:b].shape:
            input_tensor = input_tensor.reshape(self.input_tensor[:b].shape)
        self.input_tensor[:b].copy_(input_tensor)

        # Sanitize tail for fixed-bucket replay.
        if b < self.max_batch_size:
            self.input_tensor[b : self.max_batch_size].zero_()


class MiMoInputLocalTransformerCudaGraph:
    def __init__(
        self,
        cuda_graph: torch.cuda.CUDAGraph,
        buffer: MiMoInputLocalTransformerBuffer,
        output_tensor: torch.Tensor,
        batch_size: int,
    ) -> None:
        self.cuda_graph = cuda_graph
        self.buffer = buffer
        self.output_tensor = output_tensor
        self.batch_size = batch_size

    @classmethod
    def capture(
        cls,
        model: "MiMoAudioLLMForConditionalGeneration",
        buffer: MiMoInputLocalTransformerBuffer,
        batch_size: int,
        eager_run_first: bool = True,
    ) -> "MiMoInputLocalTransformerCudaGraph":
        input_tensor = buffer.inputs(batch_size)

        cuda_graph = torch.cuda.CUDAGraph()
        if eager_run_first:
            out = model.input_local_transformer(inputs_embeds=input_tensor, return_dict=True, is_causal=False)
            _ = out.last_hidden_state

        with torch.cuda.graph(cuda_graph, buffer.pool):
            out = model.input_local_transformer(inputs_embeds=input_tensor, return_dict=True, is_causal=False)
            output_tensor = out.last_hidden_state

        return cls(cuda_graph=cuda_graph, buffer=buffer, output_tensor=output_tensor, batch_size=batch_size)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        b = int(input_embeds.shape[0])
        assert b <= self.batch_size, f"Expected batch size <= {self.batch_size}, got {b}"
        with self.buffer.lock:
            self.buffer.prepare(input_embeds)
            self.cuda_graph.replay()
            if b == self.batch_size:
                return self.output_tensor.clone()
            return self.output_tensor[:b].clone()


class MiMoAudioQwen2Model(TransformerQwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)

    def embed_input_ids(self, input_ids: torch.Tensor):
        return super().get_input_embeddings()(input_ids)


class MimoAudioEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size
        - naf: Number of audio features
        - hs: Hidden size (must match the hidden size of language model
          backbone)
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: list[torch.Tensor]


class MimoAudioProcessingInfo(Qwen2AudioProcessingInfo):
    def build_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return MimoAudioMultiModalDataParser(target_sr=feature_extractor.sampling_rate)


class MimoAudioMultiModalDataParser(Qwen2AudioMultiModalDataParser):
    pass


class MimoAudioFeatureInputs(Qwen2AudioFeatureInputs):
    pass


class MimoAudioDummyInputsBuilder(BaseDummyInputsBuilder[MimoAudioProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        hf_processor = self.info.get_hf_processor()
        audio_token = hf_processor.audio_token

        return audio_token * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios)}


class MimoAudioMultiModalProcessor(BaseMultiModalProcessor[MimoAudioProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios

        # Text-only input not supported in composite processor
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        return super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _qwen2audio_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|AUDIO|>")
        audio_bos_token = getattr(processor, "audio_bos_token", "<|audio_bos|>")
        audio_eos_token = getattr(processor, "audio_eos_token", "<|audio_eos|>")

        audio_token_id = vocab[audio_token]
        audio_bos_id = vocab[audio_bos_token]
        audio_eos_id = vocab[audio_eos_token]

        out_mm_data = out_mm_kwargs.get_data()
        feature_attention_mask = out_mm_data.get("feature_attention_mask")
        if feature_attention_mask is None:
            audio_output_lengths = []
        else:
            assert isinstance(feature_attention_mask, torch.Tensor)
            _, audio_output_lens = _get_feat_extract_output_lengths(feature_attention_mask.sum(-1))

            audio_output_lengths = audio_output_lens.tolist()

        def get_replacement_mimo_audio(item_idx: int):
            if audio_output_lengths:
                num_features = audio_output_lengths[item_idx]
            else:
                audio_embeds = out_mm_data["audio_embeds"][item_idx]
                assert len(audio_embeds.shape) == 2, "audio_embeds must be a 2D tensor"
                num_features = audio_embeds.shape[0]

            if num_features == 0:
                audios = mm_items.get_items("audio", AudioProcessorItems)
                audio_len = audios.get_audio_length(item_idx)

                raise ValueError(f"The audio (len={audio_len}) is too short to be represented inside the model")

            audio_tokens = [audio_token_id] * num_features

            return PromptUpdateDetails.select_token_id(
                [audio_bos_id] + audio_tokens + [audio_eos_id],
                embed_token_id=audio_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=audio_token,
                replacement=get_replacement_mimo_audio,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MimoAudioMultiModalProcessor, info=MimoAudioProcessingInfo, dummy_inputs=MimoAudioDummyInputsBuilder
)
class MiMoAudioLLMForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # Special token IDs definition (corresponds to added_tokens.json)
        self.empty_token_id = 151667  # <|empty|>
        self.sostm_token_id = 151670  # <|sostm|>
        self.eostm_token_id = 151671  # <|eostm|>
        self.sosp_token_id = 151665  # <|sosp|>
        self.eosp_token_id = 151666  # <|eosp|>
        self.endoftext_token_id = 151643  # <|endoftext|>
        self.im_end_token_id = 151645  # <|im_end|>

        config = vllm_config.model_config.hf_config
        config = MiMoAudioConfig(**vars(config)) if isinstance(config, Qwen2Config) else config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        vllm_config.model_config.hf_config = self.config

        # Configure MRoPE parameters for multimodal rotary embeddings
        mrope_config = {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default",
        }
        setattr(vllm_config.model_config.hf_config, "rope_scaling", mrope_config)
        vllm_config.model_config.hf_config.rope_parameters.update(mrope_config)

        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            # hf_config=config,
            prefix=maybe_prefix(prefix, "model"),
            # hf_config=thinker_config.text_config,
            architectures=["Qwen2ForCausalLM"],
        )

        self.device = current_omni_platform.get_torch_device()
        self.global_sampler = MiMoSampler(do_sample=False, temperature=0.6, top_p=0.95)
        self.local_sampler = MiMoSampler(do_sample=False, temperature=0.9, top_p=0.95)
        self.removed_tokens = None

        self.speech_vocab_sizes = config.parsed_speech_vocab_sizes()
        self.speech_empty_ids = config.parsed_speech_empty_ids()
        self.delay_pattern = config.parsed_delay_pattern()
        self.group_size = config.group_size
        self.audio_channels = config.audio_channels

        self.local_config = config.local_config()
        self.input_local_config = config.input_local_config()

        self.speech_group_downcast = ColumnParallelLinear(
            self.input_local_config.hidden_size * config.group_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
        )
        self.hidden_states_downcast = ColumnParallelLinear(
            config.hidden_size,
            self.local_config.hidden_size,
            bias=False,
            return_bias=False,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            return_bias=False,
        )

        # Re-encode the sum of multi-layer RVQ embeddings to obtain true Audio Code Embeddings
        self.input_local_config = config.input_local_config()
        self.input_local_transformer = MiMoAudioQwen2Model(self.input_local_config)
        self.input_local_transformer.embed_tokens = None

        ###other parts

        # Used for multi-iteration computation: re-encode Audio Code Embeddings and convert back to
        # multi-layer RVQ codes in patch units
        self.local_transformer = MiMoAudioQwen2Model(self.local_config)
        self.local_transformer.embed_tokens = None
        self.local_transformer_lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    self.local_config.hidden_size,
                    self.speech_vocab_sizes[i],
                    bias=False,
                )
                for i in range(self.audio_channels)
            ]
        )

        self.speech_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    self.speech_vocab_sizes[i],
                    self.input_local_config.hidden_size,
                    padding_idx=self.speech_empty_ids[i],
                )
                for i in range(self.audio_channels)
            ]
        )

        if self.input_local_config.hidden_size != self.local_config.hidden_size:
            self.speech_embeddings_to_local = nn.Linear(
                self.input_local_config.hidden_size,
                self.local_config.hidden_size,
                bias=False,
            )
        else:
            self.speech_embeddings_to_local = None

        self._cached_new_audio_emb_by_req: dict[str, torch.Tensor] = {}

        # Pre-allocate audio_embeds buffer for CUDA graph capture to avoid dynamic allocation
        # Maximum sequence length set to 8192, can be adjusted according to actual needs
        self._max_audio_embeds_seq_len = 8192
        self.register_buffer(
            "_audio_embeds_buffer",
            torch.zeros((1, self._max_audio_embeds_seq_len, self.config.hidden_size), dtype=torch.bfloat16),
            persistent=False,
        )
        # Pre-allocate attention_mask buffer
        self._max_attn_len = 16384
        self.register_buffer(
            "_attention_mask_buffer",
            torch.ones((1, self._max_attn_len), dtype=torch.bool),
            persistent=False,
        )
        # Pre-allocate new_audio_emb buffer for processing after local_forward
        self._max_batch_size = 100
        self.register_buffer(
            "_new_audio_emb_buffer",
            torch.zeros(
                (self._max_batch_size, 1, self.group_size, self.input_local_config.hidden_size), dtype=torch.bfloat16
            ),
            persistent=False,
        )

        # CUDA Graph cache for local_forward (includes self.local_transformer inside base_local_forward).
        self.local_forward_cg_by_bs: dict[int, MiMoLocalDecodeCudaGraph] = {}
        self.local_forward_buf_by_bs: dict[int, MiMoLocalDecodeBuffer] = {}
        try:
            if torch.cuda.is_available() and next(self.hidden_states_downcast.parameters()).device.type == "cuda":
                for bs in MIMO_CUDAGRAPH_BATCH_SIZES:
                    try:
                        buf = MiMoLocalDecodeBuffer(self, max_batch_size=bs)
                        cg = MiMoLocalDecodeCudaGraph.capture(self, buf, batch_size=bs)
                        self.local_forward_buf_by_bs[bs] = buf
                        self.local_forward_cg_by_bs[bs] = cg
                        logger.info(f"Captured local_forward CUDA graph (batch_size={bs}).")
                    except Exception as e:
                        logger.warning(
                            f"Failed to capture local_forward CUDA graph (batch_size={bs}): {e}. Skip this bucket."
                        )
                if not self.local_forward_cg_by_bs:
                    logger.info("No local_forward CUDA graph buckets captured; falling back to eager local_forward.")
            else:
                logger.info("CUDA not available or model not on CUDA; skip local_forward CUDA graph capture.")
        except Exception as e:
            logger.warning(f"Failed to init local_forward CUDA graph cache: {e}. Falling back to eager local_forward.")
            self.local_forward_cg_by_bs.clear()
            self.local_forward_buf_by_bs.clear()

        # CUDA Graph cache for input_local_transformer (re-encode grouped RVQ embeddings).
        self.input_local_transformer_cg_by_bs: dict[int, MiMoInputLocalTransformerCudaGraph] = {}
        self.input_local_transformer_buf_by_bs: dict[int, MiMoInputLocalTransformerBuffer] = {}
        try:
            if torch.cuda.is_available() and next(self.input_local_transformer.parameters()).device.type == "cuda":
                for bs in MIMO_CUDAGRAPH_BATCH_SIZES:
                    try:
                        buf = MiMoInputLocalTransformerBuffer(self, max_batch_size=bs)
                        cg = MiMoInputLocalTransformerCudaGraph.capture(self, buf, batch_size=bs)
                        self.input_local_transformer_buf_by_bs[bs] = buf
                        self.input_local_transformer_cg_by_bs[bs] = cg
                        logger.info(f"Captured input_local_transformer CUDA graph (batch_size={bs}).")
                    except Exception as e:
                        logger.warning(
                            f"Failed to capture input_local_transformer CUDA graph (batch_size={bs}): {e}. "
                            "Skip this bucket."
                        )
                if not self.input_local_transformer_cg_by_bs:
                    logger.info("No input_local_transformer CUDA graph buckets captured; falling back to eager path.")
            else:
                logger.info("CUDA not available or model not on CUDA; skip input_local_transformer CUDA graph capture.")
        except Exception as e:
            logger.warning(f"Failed to init input_local_transformer CUDA graph cache: {e}. Falling back to eager path.")
            self.input_local_transformer_cg_by_bs.clear()
            self.input_local_transformer_buf_by_bs.clear()

    def _validate_and_reshape_mm_tensor(self, mm_input: object, name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return mm_input.reshape(-1, *mm_input.shape[2:])
        else:
            return mm_input
            # return torch.concat(mm_input)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> MimoAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of audio embeds. Got type: {type(audio_embeds)}")
            audio_embeds = self._validate_and_reshape_mm_tensor(audio_embeds, "audio_embeds")
            return MimoAudioEmbeddingInputs(type="audio_embeds", audio_embeds=audio_embeds)

        if input_features is not None:
            input_features = self._validate_and_reshape_mm_tensor(input_features, "input_features")
            return MimoAudioFeatureInputs(
                type="audio_features", input_features=input_features, feature_attention_mask=feature_attention_mask
            )

    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if kwargs.get("modality_preprocess") is None:
            mm_dummy_embeddings = []
            # TODO: audio_lengths is not the same. In vllm12, audio_length=tensor([[28],[28]...], device='cuda:0'),
            #  however, in vllm0.14, it becomes int array:[28, 28, 28, 28, 28, 28, ....]
            audio_lengths = kwargs.get("audio_lengths", [])
            for audio_length in audio_lengths:
                mm_dummy_embeddings.append(
                    torch.zeros(
                        (audio_length // self.group_size, self.config.hidden_size),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                )
            return tuple(mm_dummy_embeddings)

        if kwargs.get("mimo_audio_codes_processing") is None:
            kwargs["mimo_audio_codes_processing"] = True if kwargs.get("audio_embeds") is not None else False
        audio_input = self._parse_and_validate_audio_input(**kwargs)

        if audio_input is None:
            return []
        audio_features = self._prepare_input_audio_embeds(audio_input, **kwargs)
        return audio_features

    def _embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        inputs_embeds.masked_fill_(is_multimodal.unsqueeze(-1), 0.0)

        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = inputs_embeds + multimodal_embeddings

        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        return inputs_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def base_local_forward(
        self,
        local_embeds: torch.FloatTensor,  # [1, 1, hidden_size]
        tokens_dtype: torch.dtype = torch.int64,
        tokens_device: torch.device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        ),
        local_sampler: MiMoSampler | MiMoLocalSamplerTensor | None = None,
    ):
        B = local_embeds.shape[0]
        delay_iters = self.group_size + max(self.delay_pattern)

        local_tokens = torch.zeros(
            (B, self.group_size, self.audio_channels),
            dtype=tokens_dtype,
            device=tokens_device,
        )
        if local_sampler is None:
            local_sampler = MiMoSampler(do_sample=False, temperature=0.6, top_p=0.9)

        past_key_values = DynamicCache()
        for t in range(delay_iters):
            output = self.local_transformer(
                inputs_embeds=local_embeds,
                past_key_values=past_key_values,
                return_dict=True,
                use_cache=True,
            )
            hidden_state = output.last_hidden_state
            past_key_values = output.past_key_values

            local_embeds = torch.zeros_like(local_embeds)
            for idx in range(self.audio_channels):
                cur_start = self.delay_pattern[idx]
                cur_end = cur_start + self.group_size
                cur_empty = self.speech_empty_ids[idx]
                if cur_start <= t < cur_end:
                    cur_lm_head = self.local_transformer_lm_heads[idx]
                    cur_scores: torch.Tensor = cur_lm_head(hidden_state)[:, -1, :]

                    cur_token = local_sampler.sample(
                        cur_scores,
                        [cur_empty],
                    )

                    local_tokens[:, t - cur_start, idx] = cur_token
                    cur_input_embed = self.speech_embeddings[idx](cur_token.unsqueeze(1))

                    if self.speech_embeddings_to_local is not None:
                        cur_input_embed = self.speech_embeddings_to_local(cur_input_embed)
                    local_embeds += cur_input_embed

        return local_tokens  # [group_size, audio_channels]

    def local_forward(
        self,
        local_embeds: torch.FloatTensor,  # [1, 1, hidden_size]
        tokens_dtype: torch.dtype = torch.int64,
        tokens_device: torch.device = torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        ),
        local_sampler: MiMoSampler | None = None,
    ):
        if local_sampler is None:
            local_sampler = MiMoSampler(do_sample=False, temperature=0.6, top_p=0.9)

        b = int(local_embeds.shape[0])
        use_cg = (local_sampler.do_sample is None or local_sampler.do_sample is False) and bool(
            self.local_forward_cg_by_bs
        )
        if use_cg:
            # Pick the smallest bucket >= b.
            chosen_bs = None
            for bs in MIMO_CUDAGRAPH_BATCH_SIZES:
                if bs >= b and bs in self.local_forward_cg_by_bs:
                    chosen_bs = bs
                    break
            if chosen_bs is not None:
                logger.debug(f"Using CUDA graph for local_forward (b={b}, bucket={chosen_bs}).")
                return self.local_forward_cg_by_bs[chosen_bs].forward(local_embeds, local_sampler)

        return self.base_local_forward(
            local_embeds=local_embeds,
            tokens_dtype=tokens_dtype,
            tokens_device=tokens_device,
            local_sampler=local_sampler,
        )

    def _collect_merge_mm_embedding_info(
        self,
        input_ids: torch.Tensor,
        *,
        request_ids: list[str],
        query_start_loc: torch.Tensor,
        kwargs: dict,
    ) -> tuple[dict[str, any], dict]:
        has_merge_mm_embedding = False
        merge_mm_embedding_info: dict[str, any] = {}
        seq_len = input_ids.shape[1] if input_ids.ndim == 2 else input_ids.shape[0]

        for req_idx, req_id in enumerate(request_ids):
            query_start_loc_by_req = int(query_start_loc[req_idx].item())
            query_end_loc_by_req = int(query_start_loc[req_idx + 1].item())
            input_ids_by_req = input_ids[query_start_loc_by_req:query_end_loc_by_req]
            seq_len_by_req = input_ids_by_req.shape[1] if input_ids_by_req.ndim == 2 else input_ids_by_req.shape[0]

            if seq_len_by_req == 1 and bool(input_ids_by_req == self.empty_token_id):
                merge_mm_embedding_info[req_id] = {
                    "query_start_loc": query_start_loc_by_req,
                    "query_end_loc": query_end_loc_by_req,
                    "merge_mm_embedding": True,
                }

        has_merge_mm_embedding = len(merge_mm_embedding_info) > 0

        # Only for multimodal inputs audio generation processing(Input_ids=151667), inputs_embeds will be all zeros
        kwargs["audio_embeds"] = self._audio_embeds_buffer[:, :seq_len, :].zero_()
        kwargs["mimo_audio_codes_processing"] = False
        kwargs["modality_preprocess"] = False

        return merge_mm_embedding_info, has_merge_mm_embedding, kwargs

    def _load_cached_state(self) -> tuple[DynamicCache | None, dict[str, torch.Tensor]]:
        prev_new_audio_emb_by_req: dict[str, torch.Tensor] = {}

        if hasattr(self, "_cached_new_audio_emb_by_req"):
            for req_id, cached_emb in self._cached_new_audio_emb_by_req.items():
                if req_id not in prev_new_audio_emb_by_req:
                    prev_new_audio_emb_by_req[req_id] = cached_emb

        return prev_new_audio_emb_by_req

    def _prepare_multimodal_embeddings_with_cache(
        self,
        input_ids: torch.Tensor,
        merge_mm_embedding_info: dict[str, any],
        prev_new_audio_emb_by_req: dict[str, torch.Tensor],
        kwargs: dict,
    ) -> torch.Tensor:
        # This multimodal_embeddings is zero-valued and full sequence length,
        # will later retrieve previously generated audio codes embeddings
        multimodal_embeddings = self.embed_multimodal(**kwargs)

        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            _mm_embeddings = multimodal_embeddings[0]  # [seq_len, hidden_size]
        else:
            seq_len = input_ids.shape[-1] if input_ids.dim() > 0 else len(input_ids)
            _mm_embeddings = self._audio_embeds_buffer[0, :seq_len, :].zero_()

        # If previous new_audio_emb exists for each request, add it to multimodal_embeddings
        # In multi-request scenarios, need to select corresponding prev_new_audio_emb based on request_id
        for req_id, info in merge_mm_embedding_info.items():
            if info.get("merge_mm_embedding", False) and prev_new_audio_emb_by_req:
                start_loc = info.get("query_start_loc", None)
                end_loc = info.get("query_end_loc", None)

                if (prev_new_audio_emb := prev_new_audio_emb_by_req.get(req_id)) is not None:
                    _mm_embeddings[start_loc:end_loc] += prev_new_audio_emb  # [seq_len, hidden_size]

        inputs_embeds = self._embed_input_ids(
            input_ids, _mm_embeddings, is_multimodal=(input_ids == self.empty_token_id)
        )
        return inputs_embeds

    def _generate_speech_tokens_and_audio_embeddings(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        B = hidden_states.shape[0]
        next_speech_tokens = None
        new_audio_emb = None

        # if id is empty_token_id, then will be use hs to do local forward
        hs_downsampled = self.hidden_states_downcast(hidden_states[:, -1:, :])
        next_speech_tokens = self.local_forward(
            local_embeds=hs_downsampled,
            local_sampler=self.local_sampler,
        )

        # 4,8,4096 - Use pre-allocated buffer and zero it to avoid dynamic allocation
        new_audio_emb = self._new_audio_emb_buffer[:B].zero_()
        B, T_groups, group_size, hidden_size = new_audio_emb.shape

        next_speech_tokens = next_speech_tokens.to(torch.int32).transpose(1, 2).unsqueeze(1)
        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]

            cur_speech_ids = next_speech_tokens[:, :, idx, :]
            cur_speech_embeds: torch.Tensor = cur_embed(cur_speech_ids)

            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)

            new_audio_emb += cur_speech_embeds

        input_local_in = new_audio_emb.reshape(B * T_groups, group_size, hidden_size)
        use_cg = bool(self.input_local_transformer_cg_by_bs)
        new_audio_emb_last_hidden: torch.Tensor
        if use_cg:
            bt = int(input_local_in.shape[0])
            chosen_bs = None
            for bs in MIMO_CUDAGRAPH_BATCH_SIZES:
                if bs >= bt and bs in self.input_local_transformer_cg_by_bs:
                    chosen_bs = bs
                    break
            if chosen_bs is not None:
                logger.debug(f"Using CUDA graph for input_local_transformer (b={bt}, bucket={chosen_bs}).")
                new_audio_emb_last_hidden = self.input_local_transformer_cg_by_bs[chosen_bs].forward(input_local_in)
            else:
                use_cg = False

        if not use_cg:
            out = self.input_local_transformer(inputs_embeds=input_local_in, return_dict=True, is_causal=False)
            new_audio_emb_last_hidden = out.last_hidden_state

        new_audio_emb_last = new_audio_emb_last_hidden.reshape(B, T_groups, group_size, hidden_size)

        new_audio_emb_downcast = self.speech_group_downcast(new_audio_emb_last.view(B, T_groups, -1))
        new_audio_emb = new_audio_emb_downcast.clone()

        return next_speech_tokens, new_audio_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        _forward_context = get_forward_context()
        _default_query_start_loc = torch.tensor([0, input_ids.shape[-1]], device=input_ids.device)
        query_start_loc = (
            next(iter(_forward_context.attn_metadata.values())).query_start_loc
            if _forward_context.attn_metadata is not None
            else _default_query_start_loc
        )

        runtime_additional_information = kwargs.get("runtime_additional_information", [])
        if runtime_additional_information:
            request_ids = [info.get("req_id", str(i)) for i, info in enumerate(runtime_additional_information)]
        else:
            request_ids = [str(i) for i in range(len(query_start_loc[1:]))] if query_start_loc is not None else []
        num_reqs = len(request_ids)
        is_capturing = torch.cuda.is_current_stream_capturing()

        merge_mm_embedding_info, has_merge_mm_embedding, kwargs = self._collect_merge_mm_embedding_info(
            input_ids,
            request_ids=request_ids,
            query_start_loc=query_start_loc,
            kwargs=kwargs,
        )

        prev_new_audio_emb_by_req = self._load_cached_state()

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        if has_merge_mm_embedding:
            inputs_embeds = self._prepare_multimodal_embeddings_with_cache(
                input_ids, merge_mm_embedding_info, prev_new_audio_emb_by_req, kwargs
            )

        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

        logits = self.compute_logits(hidden_states)
        logits_indices = query_start_loc[1:] - 1
        next_ids = self.global_sampler.sample(logits[logits_indices], removed_tokens=self.removed_tokens)

        new_audio_emb_by_req: dict[str, torch.Tensor] = {}
        batch_next_speech_tokens: torch.Tensor | None = None

        if not is_capturing and next_ids is not None and num_reqs > 0:
            if (next_ids == self.empty_token_id).any():
                batch_hs_list = []
                valid_mask = []

                for req_idx in range(num_reqs):
                    start = int(query_start_loc[req_idx].item())
                    end = int(query_start_loc[req_idx + 1].item())
                    hs_req = hidden_states[start:end][-1:, :]
                    is_empty = bool(next_ids[req_idx] == self.empty_token_id)
                    valid_mask.append(is_empty)

                    if not is_empty:
                        hs_req = torch.zeros_like(hs_req)
                    batch_hs_list.append(hs_req)

                batch_hs = torch.stack(batch_hs_list, dim=0)
                batch_next_speech_tokens, batch_new_audio_emb = self._generate_speech_tokens_and_audio_embeddings(
                    batch_hs
                )

                for req_idx, is_valid in enumerate(valid_mask):
                    if is_valid:
                        req_id = request_ids[req_idx] if request_ids is not None else str(req_idx)
                        if batch_new_audio_emb is not None:
                            new_audio_emb_by_req[req_id] = batch_new_audio_emb[req_idx]
                    else:
                        batch_next_speech_tokens[req_idx] = torch.zeros_like(batch_next_speech_tokens[req_idx])

        self._update_request_caches(request_ids, new_audio_emb_by_req)

        return batch_next_speech_tokens, hidden_states

    def _update_request_caches(
        self,
        request_ids: list[str] | None,
        new_audio_emb_by_req: dict[str, torch.Tensor] | None,
    ) -> None:
        # If new_audio_emb_by_req is generated, need to store it for next round use
        # In multi-request scenarios, need to store each request's new_audio_emb based on request_id
        if new_audio_emb_by_req is not None:
            if request_ids:
                for req_id in request_ids:
                    if req_id in new_audio_emb_by_req:
                        self._cached_new_audio_emb_by_req[req_id] = new_audio_emb_by_req[req_id]
            else:
                # Case without request_ids (backward compatibility)
                # Use default key or first available key
                if not self._cached_new_audio_emb_by_req:
                    default_key = "default"
                else:
                    default_key = next(iter(self._cached_new_audio_emb_by_req.keys()))
                self._cached_new_audio_emb_by_req[default_key] = next(iter(new_audio_emb_by_req.values()))

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if hidden_states.ndim == 2:
            text_logits: torch.Tensor = self.lm_head(hidden_states)
            logits = text_logits.clone()
        else:
            text_logits: torch.Tensor = self.lm_head(hidden_states[:, -1:, :])
            logits = text_logits[:, -1, :].clone()
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = "model." + name
            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if name.startswith("input_local_transformer.") or name.startswith("local_transformer."):
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def apply_input_local_transformer(self, speech_embeddings: torch.Tensor) -> torch.Tensor:
        B, T_groups, group_size, hidden_size = speech_embeddings.shape

        # Process each group independently: [B*T_groups, group_size, hidden_size]
        input_embeddings = speech_embeddings.reshape(B * T_groups, group_size, hidden_size)

        # Apply input local transformer
        output = self.input_local_transformer(
            inputs_embeds=input_embeddings.to(speech_embeddings.device).to(torch.bfloat16),
            return_dict=True,
            is_causal=not self.config.input_full_attention,
        )
        encoded_embeddings = output.last_hidden_state

        # Reshape back to [B, T_groups, group_size, hidden_size]
        return encoded_embeddings.reshape(B, T_groups, group_size, hidden_size)

    def _prepare_input_audio_embeds(
        self,
        audio_input: MimoAudioInputs,  # [B, audio_channels + 1, new_T]
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        prompt_ids = kwargs.get("prompt_ids", None)
        _is_first_audio_codes = False if prompt_ids is None else True
        # Original TTS correct running logic
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
        if (
            not kwargs.get("mimo_audio_codes_processing")
            or (isinstance(audio_embeds, torch.Tensor) and audio_embeds.shape[0] > 1)
            or not _is_first_audio_codes
        ):
            return tuple([audio_embeds])

        prompt_ids_length = len(prompt_ids.tolist())
        prompt_ids_expand = self._expand_ids_4x_pad_and_nonpad(
            prompt_ids,
            empty_token_id=self.empty_token_id,
            ignore_id=-100,
        )
        T_groups = prompt_ids_length
        group_size = self.group_size
        mm_offset = kwargs.get("mm_offset")
        audio_lengths = [x // 4 for x in kwargs.get("audio_lengths", [])]
        audio_codes_list = audio_embeds

        # Convert list-format audio codes to tensor format [B, C, T]
        # Input may be nested list [[c0_codes], [c1_codes], ...] or tensor
        converted_audio_codes_list = []
        for codes in audio_codes_list:
            if isinstance(codes, (list, tuple)):
                codes_tensor = torch.tensor(codes, dtype=torch.long, device=self.device)
                if codes_tensor.dim() == 2:
                    codes_tensor = codes_tensor.unsqueeze(0)  # [C, T] -> [1, C, T]
                converted_audio_codes_list.append(codes_tensor)
            else:
                if codes.dim() == 2:
                    codes = codes.unsqueeze(0)  # [C, T] -> [1, C, T]
                converted_audio_codes_list.append(codes)
        audio_codes_list = converted_audio_codes_list

        dtype = audio_codes_list[0].dtype
        B = audio_codes_list[0].shape[0]

        speech_input_ids = torch.zeros(
            (B, self.audio_channels, prompt_ids_length * group_size), dtype=dtype, device=self.device
        )
        for i, idx in enumerate(self.speech_empty_ids):
            speech_input_ids[:, i, :] = idx

        speech_input_ids = self._overlay_audio_codes_by_prompt_pad_positions(
            speech_input_ids, prompt_ids_expand, audio_codes_list, mm_offset, self.device
        )

        speech_input_ids = speech_input_ids[:, :, : T_groups * group_size].view(
            B, self.audio_channels, T_groups, group_size
        )

        # Transpose to [B, T_groups, audio_channels, group_size]
        speech_input_ids = speech_input_ids.transpose(1, 2)

        # Determine which positions are speech (text token == empty_idx)
        is_speech = (prompt_ids == self.empty_token_id).unsqueeze(0).expand(B, -1)  # [B, T_groups]

        # Initialize speech embeddings: [B, T_groups, group_size, hidden_size]
        speech_embeds = torch.zeros(
            (B, T_groups, group_size, self.input_local_config.hidden_size),
            device=self.device,
            dtype=torch.bfloat16,
        )

        # Process each audio channel
        for idx in range(self.audio_channels):
            cur_empty = self.speech_empty_ids[idx]
            cur_embed = self.speech_embeddings[idx]

            # Get speech tokens for this channel: [B, T_groups, group_size]
            cur_speech_ids = speech_input_ids[:, :, idx, :]

            # Convert to embeddings: [B, T_groups, group_size, hidden_size]
            cur_speech_embeds: torch.Tensor = cur_embed(cur_speech_ids)

            # Mask out empty tokens
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)

            # Accumulate embeddings across channels
            speech_embeds += cur_speech_embeds

        # Apply mask to zero out non-speech positions
        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        # Apply input local transformer if configured
        speech_embeds = self.apply_input_local_transformer(speech_embeds)

        # Re-apply mask after transformer
        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        # Downcast grouped speech embeddings: [B, T_groups, hidden_size]
        speech_grouped_embeds: torch.Tensor = self.speech_group_downcast(
            speech_embeds.view(B, speech_embeds.shape[1], -1)
        )

        speech_embeds_split = self._split_grouped_embeds_by_speech_flag(
            speech_grouped_embeds=speech_grouped_embeds,  # [B, T_groups, H]
            is_speech_1d=(prompt_ids == self.empty_token_id),
            seg_lengths=audio_lengths,
            device=self.device,
        )

        # To pass sanity_check_mm_encoder_outputs check with dim = 2
        audio_embeds_list = [
            speech_embeds_grouped.reshape(B * speech_embeds_grouped.shape[1], -1)
            for speech_embeds_grouped in speech_embeds_split
        ]

        return tuple(audio_embeds_list)

    def _expand_ids_4x_pad_and_nonpad(
        self,
        prompt_ids: torch.Tensor,
        empty_token_id: int,
        ignore_id: int = -100,
    ) -> torch.Tensor:
        device = prompt_ids.device
        dtype = prompt_ids.dtype

        repeats = torch.full_like(prompt_ids, 4, dtype=torch.long, device=device)
        expanded = torch.repeat_interleave(prompt_ids, repeats)  # [4*T]

        within = torch.arange(expanded.numel(), device=device) % 4

        is_nonpad_expanded = expanded != empty_token_id
        mask_ignore = is_nonpad_expanded & (within != 0)

        expanded = expanded.clone()
        expanded[mask_ignore] = torch.tensor(ignore_id, device=device, dtype=dtype)
        return expanded

    def _overlay_audio_codes_by_prompt_pad_positions(
        self,
        speech_input_ids: torch.Tensor,  # [B, C, L]
        prompt_ids_expand: torch.Tensor,  # [L]  (L == speech_input_ids.shape[-1])
        audio_codes_list: list[torch.Tensor],  # each [B, C, T_i]
        mm_offset_groups: torch.Tensor | None = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        B, C, L = speech_input_ids.shape
        assert prompt_ids_expand.numel() == L, (
            f"Length mismatch: prompt_ids_expand={prompt_ids_expand.numel()} vs L={L}"
        )

        prompt_ids_expand = prompt_ids_expand.to(device)

        pad_positions = (prompt_ids_expand == self.empty_token_id).nonzero(as_tuple=True)[0]  # [K]

        # (Avoid list order not being chronological)
        if mm_offset_groups is not None:
            mm_offset_groups = mm_offset_groups.to(device).long()
            assert mm_offset_groups.numel() == len(audio_codes_list), (
                f"mm_offset_groups ({mm_offset_groups.numel()}) != num_segs ({len(audio_codes_list)})"
            )

            order = torch.argsort(mm_offset_groups)
            order_indices = order.flatten().tolist()
            audio_codes_list = [audio_codes_list[int(i)] for i in order_indices]

        cat_codes = torch.cat(audio_codes_list, dim=2).to(device)

        assert cat_codes.shape[0] == B, f"Batch mismatch: cat_codes.B={cat_codes.shape[0]} vs B={B}"
        assert cat_codes.shape[1] == C, f"Channel mismatch: cat_codes.C={cat_codes.shape[1]} vs C={C}"

        K = pad_positions.numel()
        T_total = cat_codes.shape[2]
        N = min(K, T_total)

        if N <= 0:
            return speech_input_ids

        speech_input_ids[:, :, pad_positions[:N]] = cat_codes[:, :, :N]

        return speech_input_ids

    def _split_grouped_embeds_by_speech_flag(
        self,
        speech_grouped_embeds: torch.Tensor,
        is_speech_1d: torch.Tensor,
        seg_lengths: list[int],
        device: torch.device = None,
    ) -> list[torch.Tensor]:
        assert speech_grouped_embeds.dim() == 3, f"expect [B,T,H], got {speech_grouped_embeds.shape}"
        B, T_groups, H = speech_grouped_embeds.shape

        assert is_speech_1d.dim() == 1 and is_speech_1d.numel() == T_groups, (
            f"is_speech_1d should be [T_groups], got {is_speech_1d.shape} vs T_groups={T_groups}"
        )

        is_speech_1d = is_speech_1d.to(device)

        speech_pos = is_speech_1d.nonzero(as_tuple=True)[0]  # [K]
        K = speech_pos.numel()

        segments: list[torch.Tensor] = []
        cursor = 0
        for seg_len in seg_lengths:
            seg_len = int(seg_len)
            if seg_len <= 0:
                continue

            end = cursor + seg_len
            if cursor >= K:
                break

            end = min(end, K)

            pos = speech_pos[cursor:end]
            seg = torch.index_select(speech_grouped_embeds, dim=1, index=pos)
            segments.append(seg)

            cursor = end

        return segments

    def _get_past_len(self, past_key_values: DynamicCache | None) -> int:
        if past_key_values is None:
            return 0

        if hasattr(past_key_values, "get_seq_length"):
            try:
                pl = past_key_values.get_seq_length()
                if pl is not None:
                    return int(pl)
            except Exception as e:
                logger.error(f"error happened: {e}")

        if hasattr(past_key_values, "seen_tokens"):
            try:
                return int(past_key_values.seen_tokens)
            except Exception as e:
                logger.error(f"error happened: {e}")

        try:
            if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
                k = past_key_values.layers[0].keys
                return int(k.shape[-2])

            if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
                k = past_key_values.key_cache[0]
                return int(k.shape[-2])
        except Exception as e:
            logger.error(f"error happened: {e}")

        return 0
