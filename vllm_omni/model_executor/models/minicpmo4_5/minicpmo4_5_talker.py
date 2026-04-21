from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaConfig
from transformers import LlamaModel as HFLlamaModel
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.minicpmo import MultiModalProjector
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)
E2E_ARTIFACT_DIR_ENV = "MINICPMO45_E2E_OUTPUT_DIR"


def _async_debug_request_dir(request_id: str | None) -> Path | None:
    artifact_root = os.environ.get(E2E_ARTIFACT_DIR_ENV, "").strip()
    if not artifact_root:
        return None

    safe_request_id = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id or "unknown_request").strip("_")
    if not safe_request_id:
        safe_request_id = "unknown_request"

    request_dir = Path(artifact_root) / "debug" / "minicpmo4_5_async_chunk" / safe_request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _ordered_unique(values: list[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _resolve_global_request_id(info_dict: dict[str, Any]) -> str | None:
    for key in ("global_request_id", "request_id", "external_req_id"):
        value = info_dict.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value is not None:
            value_str = str(value).strip()
            if value_str:
                return value_str
    return None


@dataclass
class _MiniCPMOAsyncConditionChunk:
    index: int
    tensor: torch.Tensor
    text_finished: bool


@dataclass
class _MiniCPMOAsyncTalkerState:
    request_id: str | None = None
    last_ingested_chunk_id: int | None = None
    pending_condition_chunks: list[_MiniCPMOAsyncConditionChunk] = field(default_factory=list)
    all_conditions: list[torch.Tensor] = field(default_factory=list)
    all_generated_tokens: list[torch.Tensor] = field(default_factory=list)
    pending_audio_token_ids: list[int] = field(default_factory=list)
    pending_audio_token_condition_indices: list[int | None] = field(default_factory=list)
    pending_audio_token_condition_shapes: list[list[int] | None] = field(default_factory=list)
    pending_audio_token_text_finished: list[bool] = field(default_factory=list)
    past_key_values: Any = None
    text_start_pos: int = 0
    text_finished: bool = False
    stream_finished: bool = False
    last_generated_token: torch.Tensor | None = None
    next_condition_index: int = 0
    current_condition_index: int | None = None
    current_condition_shape: list[int] | None = None
    current_condition_text_finished: bool = False
    debug_codec_chunk_index: int = 0
    debug_pending_emitted_audio_tokens: list[int] = field(default_factory=list)
    debug_pending_emitted_condition_indices: list[int | None] = field(default_factory=list)
    debug_pending_emitted_condition_shapes: list[list[int] | None] = field(default_factory=list)
    debug_pending_emitted_text_finished: list[bool] = field(default_factory=list)


class MiniCPMO4_5TalkerForConditionalGeneration(nn.Module, SupportsPP):
    """Native non-streaming MiniCPM-o 4.5 talker.

    This stage consumes thinker-produced text token ids + hidden states, builds
    the one-time TTS conditioning prompt during prefill, then uses the sampled
    audio token ids themselves as the next-step inputs for native vLLM decode.

    The current native port only supports the non-streaming ``num_vq == 1``
    path, which is enough for the MiniCPM-o 4.5 checkpoint we are targeting.
    """

    ASYNC_ENQUEUED_TOKEN_COUNT_KEY = "async_condition_tokens_enqueued"
    ASYNC_PENDING_STEPS_KEY = "async_pending_condition_steps"
    ASYNC_FINISHED_ENQUEUED_KEY = "async_finished_enqueued"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.root_config = vllm_config.model_config.hf_config
        stage_config_name = getattr(vllm_config.model_config, "hf_config_name", None)
        stage_config = getattr(self.root_config, stage_config_name, None) if stage_config_name else None
        if stage_config_name and stage_config is None:
            logger.warning(
                "MiniCPMO4_5 talker could not find hf_config.%s; falling back to root hf_config.",
                stage_config_name,
            )
        self.config = stage_config if stage_config is not None else self.root_config

        self.num_vq = int(getattr(self.config, "num_vq", 1))
        if self.num_vq != 1:
            raise NotImplementedError(
                f"MiniCPMO4_5 native talker currently supports num_vq == 1 only, got {self.num_vq}."
            )

        self.have_multimodal_outputs = False
        self.has_preprocess = True
        self.has_postprocess = False
        self.requires_raw_input_tokens = True

        self.audio_bos_token_id = int(self.config.audio_bos_token_id)
        self.text_eos_token_id = int(self.config.text_eos_token_id)

        self.emb_text = nn.Embedding(int(self.config.num_text_tokens), int(self.config.hidden_size))
        self.projector_semantic = MultiModalProjector(int(self.config.llm_dim), int(self.config.hidden_size))
        self.projector_spk = MultiModalProjector(int(self.config.llm_dim), int(self.config.hidden_size))

        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(self.config.num_audio_tokens), int(self.config.hidden_size)) for _ in range(self.num_vq)]
        )
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(
                        int(self.config.hidden_size),
                        int(self.config.num_audio_tokens),
                        bias=False,
                    )
                )
                for _ in range(self.num_vq)
            ]
        )

        llama_config = LlamaConfig(
            hidden_size=int(self.config.hidden_size),
            intermediate_size=int(self.config.intermediate_size),
            num_attention_heads=int(self.config.num_attention_heads),
            num_hidden_layers=int(self.config.num_hidden_layers),
            num_key_value_heads=int(self.config.num_key_value_heads),
            max_position_embeddings=int(self.config.max_position_embeddings),
        )
        self.model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix="model",
            hf_config=llama_config,
            architectures=["LlamaModel"],
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self._async_chunk_enabled = bool(getattr(vllm_config.model_config, "async_chunk", False))
        self._async_hf_model: HFLlamaModel | None = None
        self._async_backbone_synced = False
        self._async_stream_state: _MiniCPMOAsyncTalkerState | None = None
        self._async_eos_token_id = int(self.config.num_audio_tokens) - 1
        self._async_sampling_temperature = 0.8
        self._async_sampling_top_p = 0.85
        self._async_sampling_top_k = 25
        self._async_sampling_min_p = 0.01
        self._async_sampling_repetition_penalty = 1.05
        self._async_max_prefetch_tokens = 25

        connector_cfg = getattr(vllm_config.model_config, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            connector_extra = connector_cfg.get("extra", connector_cfg) or {}
        else:
            connector_extra = getattr(connector_cfg, "extra", {}) or {}
        self._async_codec_chunk_frames = int(connector_extra.get("codec_chunk_frames", 25))

        if self._async_chunk_enabled:
            async_llama_config = LlamaConfig(
                hidden_size=int(self.config.hidden_size),
                intermediate_size=int(self.config.intermediate_size),
                num_attention_heads=int(self.config.num_attention_heads),
                num_hidden_layers=int(self.config.num_hidden_layers),
                num_key_value_heads=int(self.config.num_key_value_heads),
                max_position_embeddings=int(self.config.max_position_embeddings),
                attention_bias=False,
            )
            async_llama_config._attn_implementation = "eager"
            object.__setattr__(self, "_async_hf_model", HFLlamaModel(async_llama_config))
            self._async_hf_model.eval()

        self._async_logits_processors = (
            [RepetitionPenaltyLogitsProcessor(self._async_sampling_repetition_penalty)]
            if self._async_chunk_enabled and self._async_sampling_repetition_penalty != 1.0
            else []
        )
        self._async_logits_warpers = []
        if self._async_chunk_enabled and self._async_sampling_top_p < 1.0:
            self._async_logits_warpers.append(TopPLogitsWarper(self._async_sampling_top_p))
        if self._async_chunk_enabled and self._async_sampling_top_k > 0:
            self._async_logits_warpers.append(TopKLogitsWarper(self._async_sampling_top_k))
        if self._async_chunk_enabled and self._async_sampling_min_p > 0.0:
            self._async_logits_warpers.append(MinPLogitsWarper(self._async_sampling_min_p))

        self.hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "tts.model.": "model.model.",
                "tts.emb_text.": "emb_text.",
                "tts.projector_semantic.": "projector_semantic.",
                "tts.projector_spk.": "projector_spk.",
                "tts.emb_code.": "emb_code.",
                "tts.head_code.": "head_code.",
            }
        )

    def embed_multimodal(self, **kwargs: Any) -> list[torch.Tensor]:
        if not kwargs:
            return []

        logger.warning(
            "MiniCPM talker received multimodal encoder inputs during profile run; "
            "returning dummy embeddings because Stage 1 does not consume them."
        )

        hidden_size = int(self.config.hidden_size)
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        num_items = 0

        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                num_items = int(value.shape[0]) if value.ndim > 0 else 1
                break
            if isinstance(value, list):
                num_items = len(value)
                if value and isinstance(value[0], torch.Tensor):
                    device = value[0].device
                break

        return [torch.zeros((1, hidden_size), device=device, dtype=dtype) for _ in range(num_items)]

    @staticmethod
    def estimate_prompt_len_from_additional_information(additional_information: dict[str, Any] | None) -> int:
        """Length-only mirror of ``prepare_condition_inputs`` for stage input sizing."""
        info = additional_information or {}
        llm_tokens = info.get("llm_tokens")
        if isinstance(llm_tokens, torch.Tensor):
            text_len = int(llm_tokens.numel())
        elif isinstance(llm_tokens, list):
            text_len = len(llm_tokens)
        else:
            text_len = 0

        spk_embeds = info.get("spk_embeds")
        if isinstance(spk_embeds, torch.Tensor) and spk_embeds.ndim >= 2:
            spk_len = int(spk_embeds.shape[0])
        else:
            spk_len = 0

        # Mirror the HF non-streaming prompt:
        #   [speaker_embeds] + [text-conditioned embeddings] + [text_eos] + [audio_bos]
        return spk_len + text_len + 2

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[],
        )
        loaded = loader.load_weights(
            ((name, tensor) for name, tensor in weights if name.startswith("tts.")),
            mapper=self.hf_to_vllm_mapper,
        )
        # vLLM wraps the backbone in LlamaForCausalLM, which adds lm_head.
        # MiniCPM TTS consumes hidden states directly and never uses it.
        loaded.add("model.lm_head.weight")
        if self._async_chunk_enabled:
            self._sync_async_backbone_from_vllm()
        logger.info("Loaded %d weights for MiniCPMO4_5TalkerForConditionalGeneration", len(loaded))
        return loaded

    def _sync_async_backbone_from_vllm(self) -> None:
        if not self._async_chunk_enabled or self._async_hf_model is None:
            return

        source_model = getattr(self.model, "model", self.model)
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        self._async_hf_model.to(device=device, dtype=dtype)

        src_params = dict(source_model.named_parameters())
        src_buffers = dict(source_model.named_buffers())

        missing_params: list[str] = []
        for name, param in self._async_hf_model.named_parameters():
            src = self._lookup_async_source_tensor(name, param, src_params)
            if src is None:
                missing_params.append(name)
                continue
            param.data.copy_(src.detach().to(device=param.device, dtype=param.dtype))

        for name, buffer in self._async_hf_model.named_buffers():
            src = src_buffers.get(name)
            if src is None:
                continue
            buffer.data.copy_(src.detach().to(device=buffer.device, dtype=buffer.dtype))

        if missing_params:
            raise RuntimeError(
                f"MiniCPM async HF talker backbone is missing parameters when syncing from vLLM: {missing_params[:8]}"
            )

        self._async_hf_model.eval()
        self._async_backbone_synced = True

    def _lookup_async_source_tensor(
        self,
        target_name: str,
        target_tensor: torch.Tensor,
        source_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        direct = source_tensors.get(target_name)
        if direct is not None:
            return direct

        if ".self_attn.q_proj." in target_name:
            fused_name = target_name.replace(".self_attn.q_proj.", ".self_attn.qkv_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            return fused.narrow(0, 0, int(target_tensor.shape[0]))

        if ".self_attn.k_proj." in target_name:
            fused_name = target_name.replace(".self_attn.k_proj.", ".self_attn.qkv_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            q_rows = int(self.config.hidden_size)
            return fused.narrow(0, q_rows, int(target_tensor.shape[0]))

        if ".self_attn.v_proj." in target_name:
            fused_name = target_name.replace(".self_attn.v_proj.", ".self_attn.qkv_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            q_rows = int(self.config.hidden_size)
            kv_rows = int(target_tensor.shape[0])
            return fused.narrow(0, q_rows + kv_rows, kv_rows)

        if ".mlp.gate_proj." in target_name:
            fused_name = target_name.replace(".mlp.gate_proj.", ".mlp.gate_up_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            return fused.narrow(0, 0, int(target_tensor.shape[0]))

        if ".mlp.up_proj." in target_name:
            fused_name = target_name.replace(".mlp.up_proj.", ".mlp.gate_up_proj.")
            fused = source_tensors.get(fused_name)
            if fused is None:
                return None
            rows = int(target_tensor.shape[0])
            return fused.narrow(0, rows, rows)

        return None

    def _reset_async_stream_state(self) -> None:
        self._async_stream_state = None

    def _append_async_debug_event(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        file_name: str,
        payload: dict[str, Any],
    ) -> None:
        request_dir = _async_debug_request_dir(state.request_id)
        if request_dir is None:
            return
        _append_jsonl(request_dir / file_name, payload)

    def _flush_async_debug_codec_chunk(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        is_last_audio_chunk: bool,
    ) -> None:
        if not state.debug_pending_emitted_audio_tokens:
            return

        condition_indices = list(state.debug_pending_emitted_condition_indices)
        condition_shapes = list(state.debug_pending_emitted_condition_shapes)
        text_finished_flags = list(state.debug_pending_emitted_text_finished)

        ordered_condition_indices = _ordered_unique([int(idx) for idx in condition_indices if idx is not None])
        ordered_condition_shapes = _ordered_unique([list(shape) for shape in condition_shapes if shape is not None])
        text_finished_unique = _ordered_unique([bool(flag) for flag in text_finished_flags])

        payload: dict[str, Any] = {
            "chunk_index": int(state.debug_codec_chunk_index),
            "condition_index": ordered_condition_indices[0] if len(ordered_condition_indices) == 1 else None,
            "condition_index_start": next((idx for idx in condition_indices if idx is not None), None),
            "condition_index_end": next((idx for idx in reversed(condition_indices) if idx is not None), None),
            "condition_indices": ordered_condition_indices,
            "condition_shape": ordered_condition_shapes[0] if len(ordered_condition_shapes) == 1 else None,
            "condition_shape_start": next((shape for shape in condition_shapes if shape is not None), None),
            "condition_shape_end": next((shape for shape in reversed(condition_shapes) if shape is not None), None),
            "condition_shapes": ordered_condition_shapes,
            "audio_token_ids": [int(tok) for tok in state.debug_pending_emitted_audio_tokens],
            "audio_token_count": int(len(state.debug_pending_emitted_audio_tokens)),
            "is_last_audio_chunk": bool(is_last_audio_chunk),
            "text_finished": text_finished_unique[0] if len(text_finished_unique) == 1 else any(text_finished_flags),
            "text_finished_flags": text_finished_unique,
        }
        self._append_async_debug_event(
            state,
            file_name="talker_codec_condition_chunks.jsonl",
            payload=payload,
        )

        state.debug_codec_chunk_index += 1
        state.debug_pending_emitted_audio_tokens.clear()
        state.debug_pending_emitted_condition_indices.clear()
        state.debug_pending_emitted_condition_shapes.clear()
        state.debug_pending_emitted_text_finished.clear()

    def _record_async_emitted_audio_token(
        self,
        state: _MiniCPMOAsyncTalkerState,
        *,
        token_id: int,
        condition_index: int | None,
        condition_shape: list[int] | None,
        text_finished: bool,
    ) -> None:
        state.debug_pending_emitted_audio_tokens.append(int(token_id))
        state.debug_pending_emitted_condition_indices.append(None if condition_index is None else int(condition_index))
        state.debug_pending_emitted_condition_shapes.append(None if condition_shape is None else list(condition_shape))
        state.debug_pending_emitted_text_finished.append(bool(text_finished))

        if int(token_id) == self._async_eos_token_id:
            self._flush_async_debug_codec_chunk(state, is_last_audio_chunk=True)
            return

        if len(state.debug_pending_emitted_audio_tokens) >= int(self._async_codec_chunk_frames):
            self._flush_async_debug_codec_chunk(state, is_last_audio_chunk=False)

    def _ensure_async_stream_state(self, *, allow_existing: bool) -> _MiniCPMOAsyncTalkerState:
        if self._async_stream_state is None:
            self._async_stream_state = _MiniCPMOAsyncTalkerState()
            return self._async_stream_state
        if not allow_existing and not self._async_stream_state.stream_finished:
            raise RuntimeError("MiniCPM async talker only supports a single active streaming request.")
        return self._async_stream_state

    @staticmethod
    def _async_cache_length(past_key_values: Any) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            return int(past_key_values.get_seq_length())
        return int(past_key_values[0][0].shape[2])

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        audio_ids = input_ids.to(device=self.emb_code[0].weight.device, dtype=torch.long)
        return self.emb_code[0](audio_ids)

    @staticmethod
    def _num_condition_tokens(llm_tokens: Any) -> int:
        if isinstance(llm_tokens, torch.Tensor):
            return int(llm_tokens.numel())
        if isinstance(llm_tokens, list):
            return len(llm_tokens)
        return 0

    @staticmethod
    def _info_flag(value: Any) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            return bool(value.reshape(-1)[0].item())
        return bool(value)

    def _get_text_eos_embed(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.text_eos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    def _get_audio_bos_embed(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.emb_text(torch.tensor([self.audio_bos_token_id], device=device, dtype=torch.long)).to(dtype=dtype)

    def _build_condition_embeds(
        self,
        llm_tokens: Any,
        hidden_states: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        llm_tokens = torch.as_tensor(llm_tokens, dtype=torch.long, device=device).reshape(-1)
        if llm_tokens.numel() == 0:
            return torch.empty((0, int(self.config.hidden_size)), device=device, dtype=dtype)

        hidden_states = torch.as_tensor(hidden_states, device=device, dtype=dtype).reshape(
            llm_tokens.shape[0], int(self.config.llm_dim)
        )

        hidden_embeds = self.projector_semantic(hidden_states)
        if bool(getattr(self.config, "normalize_projected_hidden", False)):
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

        llm_embeds = self.emb_text(llm_tokens)
        return llm_embeds + hidden_embeds

    def prepare_condition_inputs(
        self,
        additional_information: dict[str, Any],
        *,
        include_text_eos: bool = True,
        include_audio_bos: bool = True,
    ) -> torch.Tensor:
        """
        Build the one-time non-streaming MiniCPM TTS prompt.

        Expected keys from thinker2talker():
        - ``llm_tokens``: LongTensor [T_text]
        - ``tts_hidden_states``: Tensor [T_text, llm_dim]
        """
        llm_tokens = additional_information["llm_tokens"]
        hidden_states = additional_information["tts_hidden_states"]

        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype

        spk_embeds = torch.empty((0, int(self.config.hidden_size)), device=device, dtype=dtype)
        tts_embeds = self._build_condition_embeds(
            llm_tokens=llm_tokens,
            hidden_states=hidden_states,
            device=device,
            dtype=dtype,
        )

        pieces = [spk_embeds, tts_embeds]
        if include_text_eos:
            pieces.append(self._get_text_eos_embed(device=device, dtype=dtype))
        if include_audio_bos:
            pieces.append(self._get_audio_bos_embed(device=device, dtype=dtype))
        return torch.cat(pieces, dim=0)

    def _maybe_enqueue_async_condition_chunk(
        self,
        info_dict: dict[str, Any],
        *,
        is_prefill: bool,
    ) -> None:
        if not self._async_chunk_enabled:
            return

        chunk_id = info_dict.get("async_tts_chunk_id")
        if chunk_id is None:
            return

        if is_prefill and self._async_stream_state is not None and self._async_stream_state.stream_finished:
            self._reset_async_stream_state()

        state = self._ensure_async_stream_state(allow_existing=True if not is_prefill else True)
        if state.request_id is None:
            state.request_id = _resolve_global_request_id(info_dict)
        chunk_id = int(chunk_id)
        if state.last_ingested_chunk_id == chunk_id:
            return

        if is_prefill and state.last_ingested_chunk_id is not None and not state.stream_finished:
            raise RuntimeError("MiniCPM async talker only supports a single active streaming request.")

        llm_tokens = info_dict.get("llm_tokens")
        hidden_states = info_dict.get("tts_hidden_states")
        finished = self._info_flag(info_dict.get("finished", False))
        condition = self.prepare_condition_inputs(
            {
                "llm_tokens": [] if llm_tokens is None else llm_tokens,
                "tts_hidden_states": hidden_states
                if hidden_states is not None
                else torch.empty((0, int(self.config.llm_dim)), dtype=self.emb_text.weight.dtype),
            },
            include_text_eos=finished,
            include_audio_bos=True,
        )
        condition = condition.detach().to("cpu").contiguous()
        condition_index = int(state.next_condition_index)
        state.next_condition_index += 1
        condition_shape = (
            [1, int(condition.shape[0]), int(condition.shape[1])] if condition.ndim == 2 else list(condition.shape)
        )
        state.pending_condition_chunks.append(
            _MiniCPMOAsyncConditionChunk(
                index=condition_index,
                tensor=condition,
                text_finished=finished,
            )
        )
        state.all_conditions.append(condition)
        state.last_ingested_chunk_id = chunk_id
        state.text_finished = finished
        self._append_async_debug_event(
            state,
            file_name="talker_condition_events.jsonl",
            payload={
                "event": "enqueue",
                "condition_index": condition_index,
                "condition_shape": condition_shape,
                "pending_condition_queue_size": int(len(state.pending_condition_chunks)),
                "pending_audio_token_buffer_size": int(len(state.pending_audio_token_ids)),
                "text_finished": bool(finished),
                "async_tts_chunk_id": chunk_id,
            },
        )

    def _forward_async_hf_chunk(
        self,
        state: _MiniCPMOAsyncTalkerState,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self._async_hf_model is None:
            raise RuntimeError("MiniCPM async HF talker backbone is not initialized.")
        if not self._async_backbone_synced:
            self._sync_async_backbone_from_vllm()

        seq_len = int(inputs_embeds.shape[1])
        position_ids = torch.arange(
            state.text_start_pos,
            state.text_start_pos + seq_len,
            dtype=torch.long,
            device=inputs_embeds.device,
        ).unsqueeze(0)
        outputs = self._async_hf_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        state.past_key_values = outputs.past_key_values
        state.text_start_pos = self._async_cache_length(state.past_key_values)
        return outputs.last_hidden_state

    def _sample_async_next_token(
        self,
        state: _MiniCPMOAsyncTalkerState,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.head_code[0](hidden_state).reshape(1, -1).to(dtype=torch.float32)
        logits = logits / self._async_sampling_temperature

        if state.all_generated_tokens:
            generated = torch.cat(state.all_generated_tokens, dim=1).to(device=logits.device, dtype=torch.long)
            for processor in self._async_logits_processors:
                logits = processor(generated, logits)
            for warper in self._async_logits_warpers:
                logits = warper(generated, logits)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.reshape(1, 1).to(dtype=torch.long)

    def _top_up_async_token_buffer(
        self,
        *,
        min_tokens: int,
    ) -> None:
        if not self._async_chunk_enabled:
            return

        state = self._async_stream_state
        if state is None or state.stream_finished:
            return

        target = max(int(min_tokens), int(self._async_codec_chunk_frames), int(self._async_max_prefetch_tokens))
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype

        safety_steps = 0
        while len(state.pending_audio_token_ids) < target and not state.stream_finished:
            safety_steps += 1
            if safety_steps > 512:
                raise RuntimeError("MiniCPM async talker private decoder exceeded safety budget while prefetching.")

            if state.pending_condition_chunks:
                condition_chunk = state.pending_condition_chunks.pop(0)
                condition = condition_chunk.tensor.to(device=device, dtype=dtype, non_blocking=True)
                inputs_embeds = condition.unsqueeze(0)
                state.current_condition_index = int(condition_chunk.index)
                state.current_condition_shape = [1, int(condition.shape[0]), int(condition.shape[1])]
                state.current_condition_text_finished = bool(condition_chunk.text_finished)
                self._append_async_debug_event(
                    state,
                    file_name="talker_condition_events.jsonl",
                    payload={
                        "event": "consume",
                        "condition_index": int(condition_chunk.index),
                        "condition_shape": list(state.current_condition_shape),
                        "pending_condition_queue_size": int(len(state.pending_condition_chunks)),
                        "pending_audio_token_buffer_size": int(len(state.pending_audio_token_ids)),
                        "text_finished": bool(condition_chunk.text_finished),
                    },
                )
            else:
                if state.last_generated_token is None:
                    break
                inputs_embeds = self.embed_input_ids(state.last_generated_token.reshape(-1)).reshape(1, 1, -1)
                inputs_embeds = inputs_embeds.to(device=device, dtype=dtype)

            hidden_states = self._forward_async_hf_chunk(state, inputs_embeds)
            next_token = self._sample_async_next_token(state, hidden_states[:, -1:, :])
            token_id = int(next_token.reshape(-1)[0].item())

            state.last_generated_token = next_token.detach().to(device=device, dtype=torch.long)
            state.all_generated_tokens.append(state.last_generated_token.clone())
            state.pending_audio_token_ids.append(token_id)
            state.pending_audio_token_condition_indices.append(state.current_condition_index)
            state.pending_audio_token_condition_shapes.append(
                None if state.current_condition_shape is None else list(state.current_condition_shape)
            )
            state.pending_audio_token_text_finished.append(bool(state.current_condition_text_finished))
            if token_id == self._async_eos_token_id:
                state.stream_finished = True
                break

    def preprocess_decode_async_chunk(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._maybe_enqueue_async_condition_chunk(info_dict, is_prefill=False)
        self._top_up_async_token_buffer(min_tokens=1)

        audio_embeds = self.embed_input_ids(input_ids.reshape(-1)).reshape(input_ids.shape[0], -1)
        return input_ids, audio_embeds.to(device=input_ids.device, dtype=self.emb_text.weight.dtype), {}

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            if input_embeds is None:
                input_embeds = self.embed_input_ids(input_ids)
            return input_ids, input_embeds, {}

        # Prefill: runner schedules a placeholder prompt span, and we replace it
        # with the exact talker prompt embeddings slice for this request.
        if span_len > 1:
            prompt_embeds_buf = info_dict.get("talker_prompt_embeds")
            dev = input_ids.device
            is_async_chunk = bool(getattr(self.vllm_config.model_config, "async_chunk", False))
            include_text_eos = True
            if is_async_chunk:
                include_text_eos = self._info_flag(info_dict.get("finished", False))

            is_first_prefill = not isinstance(prompt_embeds_buf, torch.Tensor) or prompt_embeds_buf.ndim != 2
            if is_first_prefill:
                prompt_embeds = (
                    self.prepare_condition_inputs(
                        info_dict,
                        include_text_eos=include_text_eos,
                        include_audio_bos=True,
                    )
                    .detach()
                    .to("cpu")
                    .contiguous()
                )
                if not prompt_embeds.is_pinned():
                    prompt_embeds = prompt_embeds.pin_memory()
                offset = 0
            else:
                prompt_embeds = prompt_embeds_buf
                offset = int(info_dict.get("talker_prefill_offset", 0) or 0)

            total_prompt_len = int(prompt_embeds.shape[0])
            s = max(0, min(offset, total_prompt_len))
            e = max(0, min(offset + span_len, total_prompt_len))
            take = prompt_embeds[s:e]
            if int(take.shape[0]) < span_len:
                pad_n = int(span_len - int(take.shape[0]))
                pad = (
                    take[-1:].expand(pad_n, -1)
                    if take.numel() > 0
                    else torch.zeros(
                        (pad_n, int(self.config.hidden_size)),
                        dtype=prompt_embeds.dtype,
                    )
                )
                take = torch.cat([take, pad], dim=0)

            update_dict: dict[str, Any] = {"talker_prefill_offset": int(offset + span_len)}
            next_offset = offset + span_len
            if next_offset < total_prompt_len:
                update_dict["talker_prompt_embeds"] = prompt_embeds

            if is_async_chunk:
                self._maybe_enqueue_async_condition_chunk(info_dict, is_prefill=is_first_prefill)
                self._top_up_async_token_buffer(min_tokens=1)

            prompt_slice = take.to(device=dev, dtype=self.emb_text.weight.dtype, non_blocking=True)
            return input_ids.clone(), prompt_slice, update_dict

        if bool(getattr(self.vllm_config.model_config, "async_chunk", False)):
            return self.preprocess_decode_async_chunk(input_ids, input_embeds, **info_dict)

        # Decode: vLLM supplies the previously sampled audio token id as input_id.
        audio_embeds = self.embed_input_ids(input_ids.reshape(-1)).reshape(span_len, -1)
        return input_ids, audio_embeds.to(device=input_ids.device, dtype=self.emb_text.weight.dtype), {}

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._async_chunk_enabled and self._async_stream_state is not None:
            self._top_up_async_token_buffer(min_tokens=1)
            state = self._async_stream_state
            if state is None or not state.pending_audio_token_ids:
                raise RuntimeError("MiniCPM async talker private decoder has no buffered audio token to emit.")
            token_id = int(state.pending_audio_token_ids.pop(0))
            condition_index = (
                state.pending_audio_token_condition_indices.pop(0)
                if state.pending_audio_token_condition_indices
                else None
            )
            condition_shape = (
                state.pending_audio_token_condition_shapes.pop(0)
                if state.pending_audio_token_condition_shapes
                else None
            )
            text_finished = (
                state.pending_audio_token_text_finished.pop(0)
                if state.pending_audio_token_text_finished
                else bool(state.current_condition_text_finished)
            )
            self._record_async_emitted_audio_token(
                state,
                token_id=token_id,
                condition_index=condition_index,
                condition_shape=condition_shape,
                text_finished=bool(text_finished),
            )
            logits = torch.full(
                (hidden_states.shape[0], int(self.config.num_audio_tokens)),
                float("-inf"),
                device=hidden_states.device,
                dtype=torch.float32,
            )
            logits[:, token_id] = 0.0
            if token_id == self._async_eos_token_id:
                self._reset_async_stream_state()
            return logits
        return self.head_code[0](hidden_states)
