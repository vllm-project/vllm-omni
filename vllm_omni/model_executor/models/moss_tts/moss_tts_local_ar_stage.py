# Stage 0 of MOSS-TTS-Local: text to 32-channel RVQ codes.

import copy
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function
from transformers.models.qwen3 import Qwen3Config
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.moss_tts._local_stage0_timing import get_timer
from vllm_omni.model_executor.models.output_templates import OmniOutput

# Record the time for each process if MOSS_TTS_TIMING=1 is set in env.
_TIMER = get_timer()

logger = logging.getLogger(__name__)
_LOCAL_KV_DEBUG = os.environ.get("MOSS_TTS_LOCAL_KV_DEBUG", "0") == "1"
_LOCAL_KV_CACHE_ENABLED = os.environ.get("MOSS_TTS_LOCAL_KV_CACHE", "0") == "1"


def _parse_local_cudagraph_batch_sizes(value: str) -> tuple[int, ...]:
    batch_sizes: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        batch_size = int(item)
        if batch_size <= 0:
            raise ValueError(
                "MOSS_TTS_LOCAL_CUDAGRAPH_BATCH_SIZES must contain positive "
                f"integers, got {batch_size}."
            )
        if batch_size not in batch_sizes:
            batch_sizes.append(batch_size)
    return tuple(batch_sizes) or (1,)


_LOCAL_CUDAGRAPH_ENABLED = os.environ.get("MOSS_TTS_LOCAL_CUDAGRAPH", "0") == "1"
_LOCAL_CUDAGRAPH_DEFAULT_BATCH_SIZES = "1,2,4,8,16,32"
_LOCAL_CUDAGRAPH_BATCH_SIZES = _parse_local_cudagraph_batch_sizes(
    os.environ.get(
        "MOSS_TTS_LOCAL_CUDAGRAPH_BATCH_SIZES",
        _LOCAL_CUDAGRAPH_DEFAULT_BATCH_SIZES,
    )
)
_LOCAL_CUDAGRAPH_WARMUPS = int(
    os.environ.get("MOSS_TTS_LOCAL_CUDAGRAPH_WARMUPS", "3")
)


def _apply_top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    remove_mask = torch.zeros_like(sorted_mask)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return logits.masked_fill(remove_mask, float("-inf"))

# =======================================================================================
#  Lightweight modules cloned from MOSS-TTS - MUST be identical for weight compatibility
# =======================================================================================
class MossTTSRMSNorm(nn.Module):
    # Root-mean-square layer norm
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MossTTSMLP(nn.Module):
    # SwiGLU feed-forward network
    def __init__(self, input_size: int, ffn_hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.up_proj   = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MossTTSAttentionWithoutPositionalEmbedding(nn.Module):
    """Qwen3-style local attention without RoPE, matching the MOSS reference."""

    def __init__(self, config: Any, layer_idx: int):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        self.attn = Qwen3Attention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        attn = self.attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        past_key_values = kwargs.pop("past_key_values", None)
        use_cache = bool(kwargs.pop("use_cache", False))

        query_states = attn.q_norm(
            attn.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = attn.k_norm(
            attn.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if use_cache:
            if past_key_values is None:
                raise RuntimeError(
                    "MOSS-TTS local transformer cache was requested without "
                    "a cache object."
                )
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                attn.layer_idx,
            )

        attn_impl = getattr(attn.config, "_attn_implementation", None) or "eager"
        attention_interface = eager_attention_forward
        if attn_impl != "eager":
            if (
                attn_impl == "sdpa"
                and kwargs.get("output_attentions", False)
            ):
                logger.warning(
                    "`scaled_dot_product_attention` does not support "
                    "`output_attentions=True`; falling back to eager attention "
                    "for MOSS-TTS local transformer."
                )
            else:
                if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
                    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                        attn_impl,
                        eager_attention_forward,
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attn_output, _ = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            is_causal=not use_cache,
            # The MOSS reference local transformer deliberately ignores the
            # constructed mask here and lets the attention implementation apply
            # causal masking. Passing the mask changes local-channel logits.
            attention_mask=None,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return attn.o_proj(attn_output), None


class MossTTSLocalKVCache:
    """Small per-local-frame KV cache for the MOSS local transformer.

    This cache is intentionally not vLLM PagedAttention state. It only covers
    the short local channel loop inside one global AR step, then gets discarded.
    """

    def __init__(self, num_layers: int, max_seq_len: int = 64):
        self.max_seq_len = max_seq_len
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers
        self.seq_lens: list[int] = [0] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step = int(key_states.shape[-2])
        start = self.seq_lens[layer_idx]
        end = start + step
        if end > self.max_seq_len:
            raise RuntimeError(
                "MOSS-TTS local transformer KV cache exceeded max_seq_len="
                f"{self.max_seq_len}."
            )

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        if key_cache is None or value_cache is None:
            cache_shape = (
                *key_states.shape[:-2],
                self.max_seq_len,
                key_states.shape[-1],
            )
            key_cache = torch.empty(
                cache_shape,
                dtype=key_states.dtype,
                device=key_states.device,
            )
            value_cache = torch.empty(
                cache_shape,
                dtype=value_states.dtype,
                device=value_states.device,
            )
            self.key_cache[layer_idx] = key_cache
            self.value_cache[layer_idx] = value_cache

        key_cache[..., start:end, :].copy_(key_states)
        value_cache[..., start:end, :].copy_(value_states)
        self.seq_lens[layer_idx] = end
        return key_cache[..., :end, :], value_cache[..., :end, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_lens[layer_idx]


class MossTTSNativeLocalTransformer(nn.Module):
    """In-tree local transformer with MOSS reference-equivalent semantics.

    The local channel transformer is a short side computation inside one global
    AR decode step. It intentionally has no RoPE and does not use vLLM's main
    scheduler-managed KV blocks. The optional cache path below is a small
    per-local-frame cache and should remain opt-in until audio parity is proven.
    """

    supports_kv_cache = True

    def __init__(self, config: Any):
        super().__init__()
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3DecoderLayer,
            Qwen3RMSNorm,
        )

        self.config = config
        self.max_cache_len = int(getattr(config, "max_local_cache_len", 64))
        self.create_causal_mask = create_causal_mask
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn = MossTTSAttentionWithoutPositionalEmbedding(
                config, layer_idx
            )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: MossTTSLocalKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, MossTTSLocalKVCache | None]:
        if past_key_values is not None and not use_cache:
            raise RuntimeError(
                "MOSS-TTS local transformer received past_key_values with "
                "use_cache=False."
            )
        if use_cache and inputs_embeds.shape[1] != 1:
            raise RuntimeError(
                "MOSS-TTS local transformer cache path only supports "
                "single-token incremental local steps."
            )
        if use_cache and past_key_values is None:
            past_key_values = MossTTSLocalKVCache(
                self.config.num_hidden_layers,
                max_seq_len=self.max_cache_len,
            )

        seq_len = inputs_embeds.shape[1]
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len,
            device=inputs_embeds.device,
            dtype=torch.long,
        )
        position_ids = cache_position.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        mask_kwargs = {
            "config": self.config,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        import inspect

        mask_params = inspect.signature(self.create_causal_mask).parameters
        if "inputs_embeds" in mask_params:
            mask_kwargs["inputs_embeds"] = inputs_embeds
        else:
            mask_kwargs["input_embeds"] = inputs_embeds
        mask_kwargs = {
            key: value for key, value in mask_kwargs.items()
            if key in mask_params
        }
        causal_mask = self.create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=None,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
            )
        return self.norm(hidden_states), past_key_values


class MossTTSLocalTransformerWrapper(nn.Module):
    # Compatibility holder for the checkpoint prefix used by this branch.
    def __init__(self, local_qwen3_config, model_path: str | None = None):
        super().__init__()
        self.transformer = MossTTSNativeLocalTransformer(local_qwen3_config)
        self.supports_kv_cache = _LOCAL_KV_CACHE_ENABLED
        logger.info(
            "[MossTTS Local] Using in-tree local transformer; local KV cache "
            "enabled=%s.",
            self.supports_kv_cache,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        return self.transformer(
            inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

# =======================================================================================
#  Per-request FSM state
# =======================================================================================
@dataclass
class MossTTSLocalRequestState:
    n_vq: int
    audio_pad_code: int
    is_audio: bool = False
    audio_steps_generated: int = 0
    pending_audio_row: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        # init to pad so it is a no-op on prefill or non-audio decode steps.
        self.pending_audio_row = torch.full(
            (self.n_vq,), self.audio_pad_code, dtype=torch.long
        )

    def store_next_audio_row(self, row: torch.Tensor) -> None:
        # cache the [n_vq] codes just sampled, to be summed into the next decode embedding.
        self.pending_audio_row = row.detach().to(dtype=torch.long).reshape(self.n_vq)
        self.audio_steps_generated += 1


# =======================================================================================
#  AR Stage Model
# =======================================================================================
class MossTTSARStageModel(nn.Module, SupportsPP):
    have_multimodal_outputs = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.config = cfg
        self.n_vq: int            = cfg.n_vq
        self.channels: int        = 1 + self.n_vq
        self.audio_vocab_size: int = cfg.audio_vocab_size
        self.audio_pad_code: int  = cfg.audio_pad_code
        self.gen_slot_id: int     = cfg.audio_assistant_gen_slot_token_id
        self.audio_end_id: int    = cfg.audio_end_token_id

        # Tokens needed by the FSM / logits gating
        self.pad_token_id: int          = getattr(cfg, "pad_token_id", -1)
        self.im_end_token_id: int       = getattr(cfg, "im_end_token_id", -1)
        self.audio_start_token_id: int  = getattr(cfg, "audio_start_token_id", -1)
        self.audio_user_slot_token_id: int = getattr(
            cfg, "audio_user_slot_token_id", -1
        )
        # logger.info(
        #     "[MossTTS Local] FSM token ids: pad=%d im_end=%d audio_start=%d "
        #     "gen_slot=%d audio_end=%d",
        #     self.pad_token_id, self.im_end_token_id,
        #     self.audio_start_token_id, self.gen_slot_id, self.audio_end_id,
        # )
        lang_cfg = cfg.language_config # Qwen3Config
        if isinstance(lang_cfg, dict):
            lang_cfg = Qwen3Config(**lang_cfg)
        self.hidden_size: int = lang_cfg.hidden_size

        # Global Qwen3 backbone
        qwen3_vllm_config = copy.deepcopy(vllm_config)
        object.__setattr__(qwen3_vllm_config.model_config, "hf_config", lang_cfg)

        from vllm.model_executor.models.qwen3 import Qwen3Model
        backbone_prefix = f"{prefix}model.language_model" if prefix else "model.language_model"
        self.backbone = Qwen3Model(
            vllm_config=qwen3_vllm_config,
            prefix=backbone_prefix,
        )

        # Multi-channel embeddings
        self.embedding_list = nn.ModuleList()
        self.embedding_list.append(
            nn.Embedding(
                lang_cfg.vocab_size,
                self.hidden_size,
                padding_idx=cfg.pad_token_id,
            )
        )
        for _ in range(self.n_vq):
            self.embedding_list.append(
                nn.Embedding(
                    self.audio_vocab_size + 1,
                    self.hidden_size,
                    padding_idx=self.audio_pad_code,
                )
            )

        gpt2_cfg = getattr(cfg, "gpt2_config", None)
        local_num_layers = getattr(cfg, "local_num_layers", None)
        if local_num_layers is None and gpt2_cfg is not None:
            local_num_layers = getattr(gpt2_cfg, "n_layer", None)
        local_hidden_size = getattr(cfg, "local_hidden_size", None)
        if local_hidden_size is None and gpt2_cfg is not None:
            local_hidden_size = getattr(gpt2_cfg, "n_embd", None)
        local_ffn_hidden_size = getattr(cfg, "local_ffn_hidden_size", None)
        if local_ffn_hidden_size is None and gpt2_cfg is not None:
            local_ffn_hidden_size = getattr(gpt2_cfg, "n_inner", None)
        additional_mlp_ffn_hidden_size = getattr(cfg, "additional_mlp_ffn_hidden_size", None)
        if additional_mlp_ffn_hidden_size is None:
            additional_mlp_ffn_hidden_size = local_ffn_hidden_size or self.hidden_size
        if local_num_layers is None or local_hidden_size is None or local_ffn_hidden_size is None:
            raise ValueError("MOSS-TTS Local config is missing local transformer dimensions.")

        # Local Transformer - built using the local Qwen3 sub-config.
        local_cfg = copy.deepcopy(lang_cfg)
        local_cfg.num_hidden_layers = local_num_layers
        local_cfg.hidden_size = local_hidden_size
        local_cfg.intermediate_size = local_ffn_hidden_size
        local_cfg.max_local_cache_len = self.channels
        self.local_transformer = MossTTSLocalTransformerWrapper(
            local_cfg, model_path=vllm_config.model_config.model
        )

        # Projection: global hidden to local hidden
        self.speech_embedding_to_local_mlp = MossTTSMLP(
            input_size=self.hidden_size,
            ffn_hidden_size=additional_mlp_ffn_hidden_size,
            output_size=local_hidden_size,
        )

        # local hidden to global hidden
        self.local_to_speech_embedding_mlps = nn.ModuleList([
            MossTTSMLP(
                input_size=local_hidden_size,
                ffn_hidden_size=additional_mlp_ffn_hidden_size,
                output_size=self.hidden_size,
            )
            for _ in range(self.channels)
        ])

        self.layer_norm_before_lm_heads = nn.ModuleList([
            MossTTSRMSNorm(self.hidden_size) for _ in range(self.channels)
        ])

        self.lm_heads = nn.ModuleList()
        self.lm_heads.append(nn.Linear(self.hidden_size, lang_cfg.vocab_size, bias=False))
        for _ in range(self.n_vq):
            self.lm_heads.append(
                nn.Linear(self.hidden_size, self.audio_vocab_size + 1, bias=False)
            )

        # vllm sampler for text channel
        self.logits_processor = LogitsProcessor(lang_cfg.vocab_size)
        self.sampler = Sampler()

        # each request has its own FSM state tracking to check audio mode and prev audio codes
        self._request_states: dict[str, MossTTSLocalRequestState] = {}
        self._last_request_ids: list[str] = []
        self._last_seq_lens: list[int] = []

        self._pending_text_logits: dict[str, torch.Tensor] = {}

        self._local_cudagraph_enabled = _LOCAL_CUDAGRAPH_ENABLED
        self._local_cudagraphs = None
        if self._local_cudagraph_enabled:
            from vllm_omni.model_executor.models.moss_tts.moss_tts_local_cuda_graph import (
                MossTTSLocalCUDAGraphManager,
            )

            self._local_cudagraphs = MossTTSLocalCUDAGraphManager(
                model=self,
                batch_sizes=_LOCAL_CUDAGRAPH_BATCH_SIZES,
                warmups=_LOCAL_CUDAGRAPH_WARMUPS,
            )
            logger.info(
                "[MossTTS Local] Local CUDA graph enabled for batch sizes %s.",
                _LOCAL_CUDAGRAPH_BATCH_SIZES,
            )

    # =======================================================================================
    #  FSM helpers (per-request audio-mode tracking)
    # =======================================================================================

    def _new_request_state(self) -> MossTTSLocalRequestState:
        return MossTTSLocalRequestState(
            n_vq=self.n_vq,
            audio_pad_code=self.audio_pad_code,
        )

    def _should_run_local_forward_during_outer_capture(self) -> bool:
        return False

    def _advance_state_with_text_token(
        self,
        state: MossTTSLocalRequestState,
        token_id: int,
    ) -> None:
        # if currently not in audio mode, enter if "audio_start" or "gen_slot"
        # if currently in audio mode, exit if anything other than "gen_slot"
        if state.is_audio:
            if token_id == self.audio_end_id:
                state.is_audio = False
                state.pending_audio_row = torch.full_like(
                    state.pending_audio_row,
                    self.audio_pad_code,
                    dtype=torch.long,
                )
            return

        entry_tokens = {self.gen_slot_id}
        if self.audio_start_token_id >= 0:
            entry_tokens.add(self.audio_start_token_id)
        if token_id in entry_tokens:
            state.is_audio = True

    def _force_token(self, logits: torch.Tensor, token_id: int) -> torch.Tensor:
        if token_id < 0 or token_id >= logits.shape[-1]:
            return logits
        forced = torch.full_like(logits, float("-inf"))
        forced[:, token_id] = logits[:, token_id]
        return forced

    def _reset_prefill_state(
        self,
        request_id: str,
        prompt_tokens: torch.Tensor,
    ) -> MossTTSLocalRequestState:
        state = self._new_request_state()
        tokens = prompt_tokens.reshape(-1).tolist()

        if self.audio_user_slot_token_id >= 0:
            if any(int(t) == self.audio_user_slot_token_id for t in tokens):
                logger.warning(
                    "[MossTTS Local] Request %s contains continuation prompt tokens. "
                    "Phase-1 only validates direct TTS prompts.",
                    request_id,
                )

        for token in tokens:
            self._advance_state_with_text_token(state, int(token))

        self._request_states[request_id] = state
        return state

    def _prepare_request_states(
        self,
        input_ids: torch.Tensor,
        request_ids: list[str],
        seq_lens: list[int],
    ) -> tuple[list[int], list[MossTTSLocalRequestState]]:
        decode_positions: list[int] = []
        decode_states: list[MossTTSLocalRequestState] = []

        offset = 0
        for request_id, seq_len in zip(request_ids, seq_lens):
            req_tokens = input_ids[offset : offset + seq_len].reshape(-1)
            state = self._request_states.get(request_id)

            if seq_len > 1 or state is None:
                # Prefill (or first time we've seen this request) — rebuild FSM.
                state = self._reset_prefill_state(request_id, req_tokens)
            else:
                # Decode step — advance FSM by the single newly-sampled token.
                self._advance_state_with_text_token(state, int(req_tokens[-1].item()))

            if seq_len == 1:
                decode_positions.append(offset)
                decode_states.append(state)
            offset += seq_len

        self._last_request_ids = list(request_ids)
        self._last_seq_lens = list(seq_lens)
        return decode_positions, decode_states

    # =======================================================================================
    #  Build multi-channel embeddings
    # =======================================================================================

    def embed_input_ids(
        self,
        input_ids: torch.Tensor, # [L] flat 1-D token IDs (text channel)
        multimodal_embeddings=None,
        is_multimodal: bool = False,
        request_ids: list[str] | None = None,
        seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        # Channel 0: text embedding
        embeds = self.embedding_list[0](input_ids)  # [L, D]

        # Add pre-computed audio embeddings if provided by the pipeline
        if multimodal_embeddings is not None:
            embeds = embeds + multimodal_embeddings

        if not request_ids or not seq_lens:
            return embeds

        offset = 0
        for request_id, slen in zip(request_ids, seq_lens):
            if slen == 1:
                state = self._request_states.get(request_id)
                # inject audio embeddings if the request is currently producing audio
                if state is not None and state.is_audio:
                    row = state.pending_audio_row.to(embeds.device)
                    for ch_idx in range(self.n_vq):
                        ch_code = row[ch_idx].unsqueeze(0)
                        ch_emb  = self.embedding_list[ch_idx + 1](ch_code)
                        embeds[offset] = embeds[offset] + ch_emb[0]
            offset += slen

        return embeds

    # =======================================================================================
    #  Local transformer: predict n_vq RVQ codes from one global step
    # =======================================================================================

    def _local_channel_logits_eager(
        self,
        ch: int,
        current_proj: torch.Tensor,
        local_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_ctx = torch.cat([local_ctx, current_proj.unsqueeze(1)], dim=1)
        local_out, _ = self.local_transformer(local_ctx)
        last_h = local_out[:, -1, :]
        proj_out = self.local_to_speech_embedding_mlps[ch](last_h)
        normed = self.layer_norm_before_lm_heads[ch](proj_out)
        logits = self.lm_heads[ch](normed)
        return logits, local_ctx

    @torch.no_grad()
    def _local_forward(
        self,
        global_hidden: torch.Tensor, # [B, D_global]  (B = num decode seqs)
        forced_text_token_id: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B     = global_hidden.shape[0]
        dev   = global_hidden.device
        dtype = global_hidden.dtype
        local_dim = self.config.local_hidden_size

        # Project global hidden state to first local transformer input
        current_proj = self.speech_embedding_to_local_mlp(global_hidden) # [B, local_D]

        local_ctx = torch.zeros(B, 0, local_dim, device=dev, dtype=dtype)
        local_cache: MossTTSLocalKVCache | None = None
        use_local_cache = bool(
            getattr(self.local_transformer, "supports_kv_cache", False)
        )

        audio_codes: list[torch.Tensor] = []
        text_logits: torch.Tensor | None = None
        debug_input_lengths: list[int] | None = [] if _LOCAL_KV_DEBUG else None

        # ch = 0 (text), 1..32 (audio)
        for ch in range(self.channels):
            if use_local_cache:
                with record_function(f"local/ch_{ch:02d}_transformer"), \
                     _TIMER.gpu("local/transformer_per_ch"):
                    # Incremental path: cache K/V inside this one global AR
                    # frame. The cache is reset before the next audio frame.
                    local_out, local_cache = self.local_transformer(
                        current_proj.unsqueeze(1),
                        past_key_values=local_cache,
                        use_cache=True,
                    )
                    if debug_input_lengths is not None:
                        cache_len = (
                            local_cache.get_seq_length()
                            if local_cache is not None
                            else ch + 1
                        )
                        debug_input_lengths.append(cache_len)

                last_h = local_out[:, -1, :]
                with record_function(f"local/ch_{ch:02d}_head"), \
                     _TIMER.gpu("local/proj_norm_head_per_ch"):
                    proj_out = self.local_to_speech_embedding_mlps[ch](last_h)
                    normed   = self.layer_norm_before_lm_heads[ch](proj_out)
                    logits   = self.lm_heads[ch](normed)
            else:
                # Recompute path: feed the entire local channel context so far.
                # The optional CUDA graph captures only this tensor-only
                # transformer/head block; sampling remains eager below.
                with record_function(f"local/ch_{ch:02d}_transformer_head"), \
                     _TIMER.gpu("local/transformer_head_per_ch"):
                    graph_result = None
                    if self._local_cudagraphs is not None:
                        graph_result = self._local_cudagraphs.replay_channel(
                            channel=ch,
                            current_proj=current_proj,
                            local_ctx=local_ctx,
                            logits_dim=self.lm_heads[ch].out_features,
                        )
                    if graph_result is None:
                        logits, local_ctx = self._local_channel_logits_eager(
                            ch=ch,
                            current_proj=current_proj,
                            local_ctx=local_ctx,
                        )
                    else:
                        logits, local_ctx = graph_result
                    if debug_input_lengths is not None:
                        debug_input_lengths.append(int(local_ctx.shape[1]))

            with record_function(f"local/ch_{ch:02d}_sample"), \
                 _TIMER.gpu("local/sample_per_ch"):
                if ch == 0:
                    # Text channel
                    text_logits = logits.clone()
                    next_token = torch.full(
                        (B,), forced_text_token_id, dtype=torch.long, device=dev,
                    )
                else:
                    # Audio channel: prevent the pad code from being sampled.
                    logits[:, self.audio_pad_code] = float("-inf")
                    if temperature > 0.0:
                        logits = logits / temperature
                        if top_k > 0:
                            top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values
                            logits[logits < top_k_vals[..., -1:]] = float("-inf")
                        logits = _apply_top_p_filter(logits, top_p)
                        probs      = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        next_token = logits.argmax(dim=-1)
                    audio_codes.append(next_token)

            # Re-embed sampled token for next local step's input
            with record_function(f"local/ch_{ch:02d}_embed"), \
                 _TIMER.gpu("local/embed_next_per_ch"):
                emb          = self.embedding_list[ch](next_token)
                current_proj = self.speech_embedding_to_local_mlp(emb)

        # stack: [B, n_vq]
        if audio_codes:
            codes = torch.stack(audio_codes, dim=1).to(torch.long)
        else:
            codes = torch.zeros(B, self.n_vq, dtype=torch.long, device=dev)

        if text_logits is None:
            text_logits = torch.zeros(B, self.lm_heads[0].out_features, device=dev, dtype=dtype)

        if debug_input_lengths is not None:
            logger.info(
                "[MossTTS Local] mode=recompute channels=%d input_lengths=%s "
                "total_input_tokens=%d",
                self.channels,
                debug_input_lengths,
                sum(debug_input_lengths),
            )

        return codes, text_logits

    def _extract_request_info(
        self,
        runtime_additional_information: list[dict] | None = None,
    ) -> tuple[list[str], list[int], list[int]]:
        try:
            from vllm.forward_context import get_forward_context
            ctx = get_forward_context()
            attn_meta_dict = ctx.attn_metadata
            if not attn_meta_dict:
                return [], [], []
            if isinstance(attn_meta_dict, list):
                attn_meta_dict = attn_meta_dict[0]
            attn_meta = next(iter(attn_meta_dict.values()))
            qsl = attn_meta.query_start_loc.cpu().tolist()
            num_reqs = len(qsl) - 1
            seq_lens = [qsl[i + 1] - qsl[i] for i in range(num_reqs)]
            decode_positions = [qsl[i] for i, s in enumerate(seq_lens) if s == 1]
        except Exception as exc:
            logger.warning(
                "[MossTTS AR] _extract_request_info failed; falling back to "
                "ungrouped logits. This may force audio_start to avoid a Local "
                "FSM deadlock. Error: %r",
                exc,
                exc_info=True,
            )
            return [], [], []

        if runtime_additional_information:
            request_ids = [
                info.get("req_id", str(i))
                for i, info in enumerate(runtime_additional_information)
            ]
        else:
            request_ids = [str(i) for i in range(len(seq_lens))]
        return request_ids, seq_lens, decode_positions

    def _clear_warmup_state(self) -> None:
        """Clear any state accumulated during the vLLM profiling / warmup pass."""
        self._request_states.clear()
        self._last_request_ids = []
        self._last_seq_lens = []
        # Drop warmup samples so the post-warmup timing report is clean.
        _TIMER.reset()

    def on_requests_finished(self, request_ids) -> None:
        finished = {str(request_id) for request_id in request_ids}
        for request_id in finished:
            self._request_states.pop(request_id, None)
            self._pending_text_logits.pop(request_id, None)
        active = [
            (request_id, seq_len)
            for request_id, seq_len in zip(self._last_request_ids, self._last_seq_lens)
            if request_id not in finished
        ]
        self._last_request_ids = [request_id for request_id, _ in active]
        self._last_seq_lens = [seq_len for _, seq_len in active]

    # =======================================================================================
    #  Forward Pass
    # =======================================================================================

    def forward(
        self,
        input_ids: torch.Tensor | None              = None,
        positions: torch.Tensor | None              = None,
        kv_caches: list | None                      = None,
        attn_metadata                                  = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None          = None,
        **kwargs,
    ) -> OmniOutput:
        with record_function("stage0/forward"), _TIMER.gpu("stage0/forward_total"):
            # Step 0: obtain information from vLLM context
            with record_function("stage0/extract_info"), _TIMER.cpu("stage0/extract_info"):
                request_ids, seq_lens_per_req, decode_positions = self._extract_request_info(
                    kwargs.get("model_intermediate_buffer")
                    or kwargs.get("runtime_additional_information"),
                )
            self._pending_text_logits = {}

            # Step 1. Per-request FSM bookkeeping
            with record_function("stage0/fsm_prep"), _TIMER.cpu("stage0/fsm_prep"):
                decode_states: list[MossTTSLocalRequestState] = []
                if request_ids and seq_lens_per_req and input_ids is not None:
                    decode_positions, decode_states = self._prepare_request_states(
                        input_ids=input_ids,
                        request_ids=request_ids,
                        seq_lens=seq_lens_per_req,
                    )

            # Step 2. Build multi-channel embeddings
            with record_function("stage0/embed"), _TIMER.gpu("stage0/embed"):
                if inputs_embeds is None and input_ids is not None:
                    inputs_embeds = self.embed_input_ids(
                        input_ids,
                        multimodal_embeddings=kwargs.get("multimodal_embeddings"),
                        request_ids=request_ids if request_ids else None,
                        seq_lens=seq_lens_per_req if seq_lens_per_req else None,
                    )

            # Step 3. Global Qwen3 backbone
            with record_function("stage0/backbone"), _TIMER.gpu("stage0/backbone"):
                hidden_states = self.backbone(
                    input_ids=None,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )

            # Step 4. Local transformer
            multimodal_outputs: dict[str, Any] = {}

            run_local_forward = (
                not torch.cuda.is_current_stream_capturing()
                or self._should_run_local_forward_during_outer_capture()
            )
            if decode_positions and decode_states and run_local_forward:
                audio_mask = [s.is_audio for s in decode_states]
                audio_positions = [
                    p for p, m in zip(decode_positions, audio_mask) if m
                ]
                audio_states = [s for s, m in zip(decode_states, audio_mask) if m]
                decode_request_ids = [
                    r for r, sl in zip(request_ids, seq_lens_per_req) if sl == 1
                ]
                audio_request_ids = [
                    r for r, m in zip(decode_request_ids, audio_mask) if m
                ]

                if audio_positions:
                    pos_t = torch.tensor(audio_positions, device=hidden_states.device)
                    decode_hidden = hidden_states[pos_t]

                    with record_function("stage0/local_forward"), _TIMER.gpu("stage0/local_forward"):
                        codes, text_logits = self._local_forward(
                            decode_hidden,
                            forced_text_token_id=self.gen_slot_id,
                            temperature=kwargs.get("audio_temperature", 1.0),
                            top_k=kwargs.get("audio_top_k", 50),
                            top_p=kwargs.get("audio_top_p", 0.95),
                        )

                    with record_function("stage0/store_codes"), _TIMER.gpu("stage0/store_codes"):
                        for state, row in zip(audio_states, codes):
                            state.store_next_audio_row(row)

                        # Cache channel-0 text logits so `compute_logits`
                        # returns the local-pipeline-processed distribution.
                        for req_id, tl in zip(audio_request_ids, text_logits):
                            self._pending_text_logits[req_id] = tl

                    B = codes.shape[0]
                    multimodal_outputs = {
                        "code_predictor_codes": codes.reshape(B, 1, self.n_vq, 1),
                        "audio_pad_code": self.audio_pad_code,
                    }

            return OmniOutput(
                text_hidden_states=hidden_states,
                multimodal_outputs=multimodal_outputs,
            )

    # Minimum number of audio frames to emit before allowing the model to sample `audio_end`
    MIN_AUDIO_FRAMES: int = 10

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        with record_function("stage0/compute_logits"), _TIMER.gpu("stage0/compute_logits"):
            logits = self.lm_heads[0](hidden_states)   # [L, vocab_size]
            if not self._last_request_ids:
                if self.audio_start_token_id >= 0:
                    return self._force_token(logits, self.audio_start_token_id)
                logger.warning(
                    "[MossTTS AR] compute_logits has no request FSM state and "
                    "audio_start_token_id is unavailable; returning raw logits."
                )
                return logits

            neg_inf = float("-inf")
            for row_idx, request_id in enumerate(self._last_request_ids):
                if row_idx >= logits.shape[0]:
                    break
                state = self._request_states.get(request_id)
                if state is None:
                    continue

                # Substitute local-pipeline text logits when available.
                cached_tl = self._pending_text_logits.get(request_id)
                if cached_tl is not None:
                    logits[row_idx] = cached_tl.to(logits.dtype)

                row = logits[row_idx]

                if state.is_audio:
                    if state.audio_steps_generated < self.MIN_AUDIO_FRAMES:
                        keep = row[self.gen_slot_id].clone()
                        row.fill_(neg_inf)
                        row[self.gen_slot_id] = keep
                    else:
                        gen_keep = row[self.gen_slot_id].clone()
                        end_keep = row[self.audio_end_id].clone()
                        row.fill_(neg_inf)
                        row[self.gen_slot_id]  = gen_keep
                        row[self.audio_end_id] = end_keep
                else:
                    if self.audio_start_token_id >= 0:
                        keep = row[self.audio_start_token_id].clone()
                        row.fill_(neg_inf)
                        row[self.audio_start_token_id] = keep
                    else:
                        if self.pad_token_id >= 0:
                            row[self.pad_token_id] = neg_inf
                        row[self.gen_slot_id]  = neg_inf
                        row[self.audio_end_id] = neg_inf

            return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def make_omni_output(self, model_output: Any, **kwargs) -> OmniOutput:
        if isinstance(model_output, OmniOutput):
            return model_output
        empty = torch.zeros((0,), dtype=torch.float32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [empty]},
        )

    # =======================================================================================
    #  Load Weights
    # =======================================================================================

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params: dict[str, torch.nn.Parameter] = dict(self.named_parameters())
        loaded_module_names: set[str] = set()

        # Qwen3 backbone: checkpoint has separate q/k/v and gate/up projections but vLLM merges them.
        BACKBONE_STACKED = [
            ("self_attn.q_proj", "self_attn.qkv_proj", "q"),
            ("self_attn.k_proj", "self_attn.qkv_proj", "k"),
            ("self_attn.v_proj", "self_attn.qkv_proj", "v"),
            ("mlp.gate_proj",    "mlp.gate_up_proj",   0),
            ("mlp.up_proj",      "mlp.gate_up_proj",   1),
        ]

        for ckpt_name, tensor in weights:
            mapped: str | None = None
            shard_id = None

            if ckpt_name.startswith("model.language_model."):
                relative = ckpt_name[len("model.language_model."):]
                # Check stacked (merged) params first
                for ckpt_sfx, mod_sfx, s_id in BACKBONE_STACKED:
                    if ckpt_sfx in relative:
                        mapped = "backbone." + relative.replace(ckpt_sfx, mod_sfx)
                        shard_id = s_id
                        break
                if mapped is None:
                    mapped = "backbone." + relative

            elif ckpt_name.startswith("model.embedding_list."):
                mapped = "embedding_list." + ckpt_name[len("model.embedding_list."):]

            elif ckpt_name.startswith("local_transformer."):
                relative = ckpt_name[len("local_transformer."):]
                if ".self_attn." in relative:
                    relative = relative.replace(".self_attn.", ".self_attn.attn.")
                mapped = "local_transformer.transformer." + relative

            else:
                # speech_embedding_to_local_mlp.*, local_to_speech_embedding_mlps.*,
                # layer_norm_before_lm_heads.*, lm_heads.*  — no prefix change
                mapped = ckpt_name

            if mapped not in params:
                logger.debug(
                    "[MossTTS AR] Unused checkpoint key %s (mapped to %s)",
                    ckpt_name, mapped,
                )
                continue

            param = params[mapped]

            if shard_id is not None:
                # Use the registered weight_loader for proper shard merging
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, tensor, shard_id)
                    loaded_module_names.add(mapped)
                else:
                    logger.warning(
                        "[MossTTS AR] No weight_loader on %s, cannot merge shard %s — skipping",
                        mapped, shard_id,
                    )
            else:
                if param.data.shape != tensor.shape:
                    logger.warning(
                        "[MossTTS AR] Shape mismatch for %s: ckpt %s vs model %s — skipping.",
                        ckpt_name, tuple(tensor.shape), tuple(param.data.shape),
                    )
                    continue
                wl = getattr(param, "weight_loader", default_weight_loader)
                wl(param, tensor)
                loaded_module_names.add(mapped)

        missing = set(params.keys()) - loaded_module_names
        if missing:
            logger.warning(
                "[MossTTS AR] Parameters not loaded from checkpoint:\n%s",
                "\n".join(sorted(missing)[:20]),
            )
        return loaded_module_names
