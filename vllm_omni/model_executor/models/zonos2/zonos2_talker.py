# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 Stage-0 talker: 28-layer GQA + sonic-EDA-MoE backbone emitting
9-codebook DAC tokens per AR step.

Registered under ``Zonos2ForConditionalGeneration`` (canonical HF arch) and
``Zonos2TalkerForConditionalGeneration`` (explicit alias).

Numerical backbone scope:
  * Module tree mirrors the official checkpoint layout 1:1 (507 tensors,
    see tools/convert_zonos2_to_safetensors.py manifest) so dummy loading
    exercises the full parameter surface.
  * Attention runs on vLLM's PagedAttention (KV cache) with interleaved RoPE;
    QK RMSNorm / per-head temperature / headwise sigmoid gate are structurally
    present and get numerically validated in M2a.
  * The sonic EDA router preserves canonical checkpoint parameters while
    expert computation uses vLLM's fused MoE kernels.
  * Sampling is request-local with independent controls and reproducible
    replay from the complete nine-codebook history; lifecycle tokens carry
    continue/stop only, while multimodal output carries the codec frames.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.utils import make_layers
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_keys import (
    FRAMES,
    SCHEDULED_SPAN,
    SPEAKER_EMBEDDING,
    SPEAKER_POSITION,
    STATE,
    TERMINAL,
)
from vllm_omni.model_executor.models.zonos2.zonos2_moe import Zonos2TritonExperts
from vllm_omni.model_executor.models.zonos2.zonos2_norm import (
    zonos2_cuda_fused_add_rmsnorm,
    zonos2_cuda_rmsnorm,
)
from vllm_omni.model_executor.models.zonos2.zonos2_sampler import (
    CONTINUE_TOKEN,
    STOP_TOKEN,
    Zonos2RequestState,
    Zonos2SamplingParams,
    sample_frame,
)


class _WeightModule(nn.Module):
    """Bare holder exposing a single ``weight`` Parameter with a custom shape."""

    def __init__(self, *shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(*shape))


class Zonos2SpeakerLDAProjection(nn.Linear):
    """Preserve the frozen LDA weight's column-major storage.

    Safetensors stores its values contiguously, but the reference checkpoint
    uses stride (1, out_features). That layout selects a different BF16 GEMM
    reduction; rounding differences propagate through speaker conditioning.
    """

    weight: nn.Parameter

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.weight = nn.Parameter(self.weight.detach().t().contiguous().t())


class Zonos2RMSNorm(RMSNorm):
    """Keep the official FP32 residual sum through weighted normalization."""

    def forward_native(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        combined = x.float()
        if residual is not None:
            combined = combined + residual.float()
        variance = combined.square().mean(dim=-1, keepdim=True)
        normalized = combined * torch.rsqrt(variance + self.variance_epsilon)
        normalized = (normalized * self.weight.float()).to(x.dtype)
        if residual is None:
            return normalized
        return normalized, combined.to(residual.dtype)

    def forward_cuda(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return zonos2_cuda_rmsnorm(x, self.weight, self.variance_epsilon)
        zonos2_cuda_fused_add_rmsnorm(x, residual, self.weight, self.variance_epsilon)
        return x, residual


class Zonos2RotaryEmbedding(RotaryEmbedding):
    """Official interleaved RoPE with an FP32 trigonometric cache.

    Ordinary vLLM RoPE casts this cache to the activation dtype. ZONOS2's
    official FlashInfer path keeps it FP32 until the final rotated Q/K cast.
    """

    cos_sin_cache: torch.Tensor

    def __init__(self, config: Zonos2Config):
        self.use_flashinfer = True
        super().__init__(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_seqlen,
            base=config.rope_theta,
            is_neox_style=False,
            dtype=torch.float32,
        )

    def _apply(self, fn, recurse: bool = True):
        # Module.to(bfloat16) must not round the nonpersistent FP32 cache.
        cache = self.cos_sin_cache
        result = super()._apply(fn, recurse=recurse)
        self.cos_sin_cache = cache.to(device=self.cos_sin_cache.device, dtype=torch.float32)
        return result

    def forward_native(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query_dtype = query.dtype
        key_dtype = key.dtype if key is not None else None
        query, key = self.forward_static(
            positions,
            query.float(),
            key.float() if key is not None else None,
            self.head_size,
            self.rotary_dim,
            self.cos_sin_cache.to(device=query.device),
            self.is_neox_style,
        )
        return query.to(query_dtype), key.to(key_dtype) if key is not None else None


class Zonos2Attention(nn.Module):
    """GQA attention with fused wkv, QK RMSNorm, per-head temperature and
    headwise sigmoid gate, on top of vLLM PagedAttention."""

    def __init__(self, config: Zonos2Config, prefix: str):
        super().__init__()
        self.n_heads = config.n_heads  # 16
        self.n_kv_heads = config.n_kv_heads  # 4
        self.head_dim = config.head_dim  # 128
        self.norm_eps = config.norm_eps

        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wkv = _WeightModule(2, config.n_kv_heads * config.head_dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.gater = nn.Linear(config.dim, config.n_heads, bias=False)
        # per-head learnable temperature, checkpoint key ``attention.temp``
        self.temp = nn.Parameter(torch.ones(1, config.n_heads, 1))

        self.rotary = Zonos2RotaryEmbedding(config)
        self.attn = Attention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,  # default 1/sqrt(head_dim); temp handles per-head scaling
            num_kv_heads=self.n_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        q = self.wq(x)  # [T, 2048]
        kv = F.linear(x, self.wkv.weight.reshape(2 * self.n_kv_heads * self.head_dim, -1))
        k, v = kv.chunk(2, dim=-1)  # [T, 512] each
        # QK normalization materializes K; make V's row stride match it for
        # the native KV cache store, as in the official ChunkedLinear path.
        v = v.contiguous()

        q = q.view(t, self.n_heads, self.head_dim)
        k = k.view(t, self.n_kv_heads, self.head_dim)
        v = v.view(t, self.n_kv_heads, self.head_dim)

        # QK RMSNorm (weight-free; official ckpt carries no qk-norm params).
        # Official uses eps=1e-6 here (attention-only), not the model-wide 1e-5.
        q = F.rms_norm(q, (self.head_dim,), eps=1e-6)
        k = F.rms_norm(k, (self.head_dim,), eps=1e-6)

        # Per-head temperature: q scaled by abs(temp) (absolute value per the
        # official implementation; verified in M2a).
        q = q * self.temp.abs()

        q, k = self.rotary(positions, q.flatten(-2), k.flatten(-2))
        q = q.view(t, self.n_heads, self.head_dim)
        k = k.view(t, self.n_kv_heads, self.head_dim)
        attn_out = self.attn(q, k, v)  # [T, 16*128]

        # Headwise sigmoid gate (Qwen gated-attention style).
        gate = torch.sigmoid(self.gater(x))  # [T, 16]
        attn_out = attn_out.view(t, self.n_heads, self.head_dim) * gate.unsqueeze(-1)
        return self.wo(attn_out.view(t, -1))


class Zonos2DenseFFN(nn.Module):
    """Dense SwiGLU FFN for layers 0-2 and 27 (checkpoint: w_in/w_out)."""

    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.w_in = _WeightModule(2, 3072, config.dim)
        self.w_out = _WeightModule(config.dim, 3072)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official w_in layout: first half = up (h), second half = gate;
        # y = h * silu(gate) = up * silu(gate).
        h_gate = F.linear(x, self.w_in.weight.reshape(2 * 3072, -1))
        h, gate = h_gate.chunk(2, dim=-1)
        return F.linear(h * F.silu(gate), self.w_out.weight)


class _GraphReplay(Protocol):
    def replay(self) -> None: ...


@dataclass
class _RouterCapture:
    input: torch.Tensor
    previous: torch.Tensor | None
    graph: _GraphReplay
    output: tuple[torch.Tensor, torch.Tensor]


class Zonos2SonicRouter(nn.Module):
    """Sonic EDA router: down_proj -> EDA state mix -> RMSNorm -> 3-layer GeLU
    MLP, with aux-loss-free balancing bias on the top-k selection."""

    def __init__(self, config: Zonos2Config, has_prev_state: bool):
        super().__init__()
        self._replay_enabled = os.environ.get("VLLM_ZONOS2_ROUTER_REPLAY", "0") == "1"
        self._captures: dict[bool, _RouterCapture] = {}
        rd = config.moe_router_dim
        self.down_proj = nn.Linear(config.dim, rd, bias=True)
        # RMSNorm over router_dim (checkpoint key ``rmsnorm_eda.weight``).
        self.rmsnorm_eda = Zonos2RMSNorm(rd, eps=config.norm_eps)
        self.router_mlp = nn.ModuleList(
            [
                nn.Linear(rd, rd, bias=True),
                nn.GELU(),
                nn.Linear(rd, rd, bias=True),
                nn.GELU(),
                nn.Linear(rd, config.moe_n_experts, bias=False),
            ]
        )
        self.balancing_biases = nn.Parameter(torch.zeros(config.moe_n_experts))
        if has_prev_state:
            # Layers 4..26 mix in the previous MoE layer's router state.
            self.router_states_scale = nn.Parameter(torch.zeros(rd))

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Zonos2SonicRouter:
        # Captures hold parameter addresses; a device/dtype move invalidates them.
        self._captures.clear()
        super()._apply(fn, recurse=recurse)
        return self

    def forward(self, x: torch.Tensor, prev_router_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._replay_enabled:
            return self._forward_eager(x, prev_router_state)
        previous_matches = prev_router_state is None or (
            prev_router_state.shape == (1, self.down_proj.out_features)
            and prev_router_state.dtype == x.dtype
            and prev_router_state.device == x.device
            and prev_router_state.is_contiguous()
        )
        # Only the qualified contiguous BF16 single-row inference profile uses
        # replay. Prefill/batched/training/CPU paths retain the existing kernels.
        qualified = (
            self._replay_enabled
            and not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape == (1, self.down_proj.in_features)
            and x.is_contiguous()
            and previous_matches
        )
        if not qualified:
            return self._forward_eager(x, prev_router_state)
        key = prev_router_state is not None
        capture = self._captures.get(key)
        if capture is None:
            capture = self._capture(x, prev_router_state)
            self._captures[key] = capture
        # Private static storage is required by CUDA replay. Returned tensors
        # are consumed within this forward, before the next invocation.
        capture.input.copy_(x)
        if capture.previous is not None and prev_router_state is not None:
            capture.previous.copy_(prev_router_state)
        capture.graph.replay()
        return capture.output

    def _capture(self, x: torch.Tensor, previous: torch.Tensor | None) -> _RouterCapture:
        cuda = torch.get_device_module("cuda")
        static_x = torch.empty_like(x)
        static_previous = torch.empty_like(previous) if previous is not None else None
        static_x.copy_(x)
        if static_previous is not None and previous is not None:
            static_previous.copy_(previous)
        stream = cuda.Stream(device=x.device)
        stream.wait_stream(cuda.current_stream(x.device))
        with cuda.stream(stream):
            for _ in range(3):
                self._forward_eager(static_x, static_previous)
        cuda.current_stream(x.device).wait_stream(stream)
        graph = cuda.CUDAGraph()
        # Capture errors propagate; this never pretends a failed candidate ran.
        with cuda.graph(graph, stream=stream):
            output = self._forward_eager(static_x, static_previous)
        return _RouterCapture(static_x, static_previous, graph, output)

    def _forward_eager(
        self, x: torch.Tensor, prev_router_state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (expert_prob, next_router_state).

        Official order: down_proj -> EDA blend (scale * prev_state) -> keep the
        pre-norm state for the next MoE layer -> rmsnorm_eda -> router_mlp ->
        softmax (float32). Top-k selection happens in the caller on
        ``expert_prob + balancing_biases`` (legacy strategy); routing weights
        are the pre-bias probabilities.
        """
        h = self.down_proj(x)  # [T, 128]
        if prev_router_state is not None and hasattr(self, "router_states_scale"):
            h = h + self.router_states_scale * prev_router_state
        state = h  # pre-norm; carried to the next MoE layer's EDA blend
        h = self.rmsnorm_eda(h)
        for layer in self.router_mlp:
            h = layer(h)
        expert_prob = F.softmax(h.float(), dim=-1)  # [T, 16]
        return expert_prob, state


class Zonos2MoE(nn.Module):
    """MoE FFN: 16 experts (fused w13/w2), top-k per layer from config."""

    def __init__(self, config: Zonos2Config, layer_id: int):
        super().__init__()
        self.topk = config.router_topk(layer_id)
        self.n_experts = config.moe_n_experts
        self.experts = nn.Module()
        self.experts.w13 = nn.Parameter(torch.zeros(self.n_experts, 2 * 3072, config.dim))
        self.experts.w2 = nn.Parameter(torch.zeros(self.n_experts, config.dim, 3072))
        self.router = Zonos2SonicRouter(config, has_prev_state=layer_id > config.moe_start_from_layer)
        # Keep canonical interleaved checkpoint parameters for L0 attestation.
        # vLLM's fused kernel consumes contiguous gate/up halves instead.
        self.register_buffer("_packed_w13", None, persistent=False)
        self._experts_kernel: Zonos2TritonExperts | None = None

    def prepare_expert_weights(self) -> None:
        self._packed_w13 = torch.cat([self.experts.w13[:, 0::2], self.experts.w13[:, 1::2]], dim=1).contiguous()

    def forward(self, x: torch.Tensor, prev_router_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        expert_prob, state = self.router(x, prev_router_state)
        # Legacy aux-loss-free balancing: top-k on prob + bias, routing weights
        # are the pre-bias probabilities (no renormalization).
        scores = expert_prob + self.router.balancing_biases.float()
        topk_idx = torch.topk(scores, self.topk, dim=-1).indices  # [T, k]
        weights = expert_prob.gather(-1, topk_idx)  # [T, k]

        if x.device.type != "cpu":
            if self._packed_w13 is None:
                self.prepare_expert_weights()
            if self._experts_kernel is None:
                self._experts_kernel = Zonos2TritonExperts.create(x, self._packed_w13, self.topk)
            out = self._experts_kernel.forward(
                x.contiguous(),
                self._packed_w13,
                self.experts.w2,
                weights.float().contiguous(),
                topk_idx.int().contiguous(),
            )
            return out, state

        # CPU reference for single-layer tests. Activation and weighted down
        # projection retain FP32 intermediates until their BF16 boundaries.
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (topk_idx == e).any(dim=-1)
            xe = x[mask]
            w13 = self.experts.w13[e]
            gate = F.linear(xe, w13[0::2])
            up = F.linear(xe, w13[1::2])
            activated = (F.silu(gate.float()) * up.float()).to(x.dtype)
            ye = F.linear(activated.float(), self.experts.w2[e].float())
            w = (weights[mask] * (topk_idx[mask] == e).float()).sum(dim=-1, keepdim=True)
            out[mask] = out[mask] + (ye * w).to(x.dtype)
        return out, state


class Zonos2DecoderLayer(nn.Module):
    def __init__(self, config: Zonos2Config, layer_id: int, prefix: str):
        super().__init__()
        self.layer_id = layer_id
        self.attention_norm = Zonos2RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = Zonos2Attention(config, prefix=f"{prefix}.attention")
        self.ffn_norm = Zonos2RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward: nn.Module
        if config.is_moe_layer(layer_id):
            self.feed_forward = Zonos2MoE(config, layer_id)
        else:
            self.feed_forward = Zonos2DenseFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        prev_router_state: torch.Tensor | None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Preserve the official dual hidden/residual streams: fused norms
        # normalize the FP32 sum before the residual branch rounds to BF16.
        if residual is None:
            residual = x
            x = self.attention_norm(x)
        else:
            x, residual = self.attention_norm(x, residual)
        x = self.attention(x, positions)
        x, residual = self.ffn_norm(x, residual)
        if isinstance(self.feed_forward, Zonos2MoE):
            x, state = self.feed_forward(x, prev_router_state)
        else:
            x = self.feed_forward(x)
            state = None
        return x, residual, state


class Zonos2MultiEmbedder(nn.Module):
    """10-column embedding tables (9 audio codebooks + 1 text), summed.

    Audio tables: [codebook_size+2=1026, dim], padding_idx=audio_pad_id (1025).
    Text table:   [text_vocab+1=520, dim], padding_idx=text_vocab (519).
    Padding indices match the official embedding tables. Loaded pad rows are
    read unchanged, as in the official MultiEmbedding implementation.
    """

    def __init__(self, config: Zonos2Config):
        super().__init__()
        audio_vocab = config.codebook_vocab_size  # 1026
        text_table_vocab = config.text_vocab + 1  # 520
        self.embedders = nn.ModuleList(
            [nn.Embedding(audio_vocab, config.dim, padding_idx=config.audio_pad_id) for _ in range(config.n_codebooks)]
            + [nn.Embedding(text_table_vocab, config.dim, padding_idx=config.text_vocab)]
        )
        self.frame_width = config.frame_width

    def forward(self, frame_ids: torch.Tensor) -> torch.Tensor:
        if frame_ids.ndim != 2 or frame_ids.shape[1] != self.frame_width:
            raise ValueError(f"ZONOS2 frame ids must have shape [T, {self.frame_width}], got {tuple(frame_ids.shape)}")
        if frame_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("ZONOS2 frame ids must use torch.int32 or torch.int64")
        # Sum in the official column order: audio CB0..CB8, then text.
        out = self.embedders[0](frame_ids[:, 0])
        for col, table in enumerate(self.embedders[1:], start=1):
            out = out + table(frame_ids[:, col])
        return out


class Zonos2TalkerForConditionalGeneration(nn.Module):
    """Stage-0 AR backbone for ZONOS2 (see module docstring for M1 scope)."""

    # Runner hooks
    have_multimodal_outputs: bool = True
    prefer_model_sampler: bool = True
    has_postprocess: bool = True
    has_preprocess: bool = True
    requires_request_sampling_params: bool = True
    requires_request_sample_eligibility: bool = True
    postprocess_uses_hidden_states: bool = False
    postprocess_uses_multimodal_outputs: bool = False
    gpu_resident_buffer_keys = {(STATE, key) for key in ("history", "eos_frame", "countdown", "stopped")}

    _emb_norm_weight: torch.Tensor | None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if vllm_config.scheduler_config.async_scheduling:
            raise ValueError("ZONOS2 request-local sampling currently requires async_scheduling=False")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("ZONOS2 codec-history replay currently requires enable_prefix_caching=False")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("ZONOS2 request-local sampling currently requires enforce_eager=True")
        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, Zonos2Config):
            self.config = hf_config
        else:
            self.config = Zonos2Config(**hf_config.to_dict())
        cfg = self.config

        self.multi_embedder = Zonos2MultiEmbedder(cfg)
        self.register_buffer("_emb_norm_weight", None, persistent=False)
        _, _, self.layers = make_layers(
            cfg.n_layers,
            lambda prefix: Zonos2DecoderLayer(cfg, layer_id=int(prefix.split(".")[-1]), prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.out_norm = Zonos2RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.multi_output = nn.Linear(cfg.dim, cfg.n_codebooks * cfg.codebook_vocab_size, bias=False)

        # Speaker chain: Qwen3 voice embedding (2048) -> LDA (1024) -> hidden.
        self.speaker_lda_projection = Zonos2SpeakerLDAProjection(cfg.speaker_embedding_dim, cfg.speaker_lda_dim)
        self.speaker_projection = nn.Linear(cfg.speaker_lda_dim, cfg.dim, bias=True)

        # LM lifecycle channel reuses the codebook-0 logits slice.
        self.logits_processor = LogitsProcessor(cfg.codebook_vocab_size)

        self._last_audio_codes: torch.Tensor | None = None
        self._postprocess_cursor: int = 0
        self._request_states: dict[str, Zonos2RequestState] = {}
        self._sampling_plan: list[tuple[str, bool, dict[str, Any]]] = []
        self._step_codes: dict[str, torch.Tensor] = {}
        self._step_payload: dict[str, Any] | None = None

    # ------------------------------------------------------------------ embed
    def _embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed frozen [T, 10] frames or the legacy [T] text-only prompt.

        Frame ids are consumed unchanged, including all delay/shear and
        conditioning tokens. This method returns the raw embedding sum;
        speaker replacement and the single emb_norm belong in forward().
        The text-only path remains a skeleton convenience, not a text frontend.
        """
        if input_ids.ndim == 2:
            return self.multi_embedder(input_ids)
        if input_ids.ndim != 1:
            raise ValueError("ZONOS2 input ids must be [T] text ids or [T, 10] frame ids")
        cfg = self.config
        frame_ids = torch.full(
            (input_ids.shape[0], cfg.frame_width), cfg.audio_pad_id, dtype=input_ids.dtype, device=input_ids.device
        )
        frame_ids[:, cfg.n_codebooks] = input_ids.clamp(min=0, max=cfg.text_vocab)
        return self.multi_embedder(frame_ids)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self._embed_input_ids(input_ids)

    def _inject_speaker(
        self, hidden: torch.Tensor, speaker_embeddings: torch.Tensor, speaker_positions: torch.Tensor
    ) -> torch.Tensor:
        if speaker_embeddings.ndim != 2 or speaker_embeddings.shape[1] != self.config.speaker_embedding_dim:
            raise ValueError(f"speaker_embeddings must have shape [S, {self.config.speaker_embedding_dim}]")
        if speaker_positions.ndim != 1 or speaker_positions.shape[0] != speaker_embeddings.shape[0]:
            raise ValueError("speaker_positions must contain one input row index per speaker embedding")
        if speaker_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("speaker_positions must use torch.int32 or torch.int64")
        speaker = speaker_embeddings.to(device=hidden.device, dtype=self.speaker_lda_projection.weight.dtype)
        speaker = self.speaker_lda_projection(speaker).to(dtype=self.speaker_projection.weight.dtype)
        speaker = self.speaker_projection(speaker).to(dtype=hidden.dtype)
        return hidden.index_copy(0, speaker_positions.to(device=hidden.device, dtype=torch.long), speaker)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """The runner sends the same IDs for finish, cancellation and abort."""
        finished = {str(rid) for rid in finished_req_ids}
        for key in finished:
            self._request_states.pop(key, None)
            self._step_codes.pop(key, None)
        self._sampling_plan = [p for p in self._sampling_plan if p[0] not in finished]
        if not self._sampling_plan:
            self._step_payload = None

    def _managed_preprocess(
        self, input_ids: torch.Tensor, frames: torch.Tensor, info: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        key = str(info.get("_omni_req_id") or info["request_id"])
        params = Zonos2SamplingParams.from_runtime(info["_omni_sampling_params"])
        offset = int(info["_omni_num_computed_tokens"])
        span = input_ids.shape[0]
        saved = info.get(STATE) or {}
        previous = self._request_states.get(key)
        history = saved.get("history")
        if history is None:
            history = (
                previous.history
                if previous is not None
                else torch.empty((0, 9), device=input_ids.device, dtype=torch.int32)
            )
        if not isinstance(history, torch.Tensor):
            raise TypeError("ZONOS2 reconstruction needs full tensor code history")
        history = history.to(device=input_ids.device)
        lifecycle_history = info.get("_omni_output_token_ids", ())
        committed = len(lifecycle_history)
        terminal = bool(committed and lifecycle_history[-1] == STOP_TOKEN) or committed >= params.max_tokens
        if committed > len(history):
            raise ValueError("Cannot reconstruct nine-codebook history from lifecycle token ids alone")
        history = history[:committed]
        # Absolute replay offset and full code history are authoritative. No
        # prompt-only reset, slot key or incremental global cursor is used.
        if offset == 0 or previous is None or len(previous.history) != committed or previous.params != params:
            seed = params.seed if params.seed is not None else saved.get("seed", previous.seed if previous else None)
            state = Zonos2RequestState.rebuild(key, params, history, seed)
            self._request_states[key] = state
        else:
            state = previous
        prompt_length = len(frames)
        if offset < 0 or offset + span > prompt_length + len(history):
            raise ValueError("Scheduled replay span exceeds prompt plus full code history")
        # Slice before device transfer and concatenate only a crossing span.
        # Ordinary decode consumes one prior frame, not a rebuilt P+H stream.
        pieces = []
        if offset < prompt_length:
            stop = min(offset + span, prompt_length)
            pieces.append(frames[offset:stop].to(device=input_ids.device))
        if offset + span > prompt_length:
            start = max(0, offset - prompt_length)
            stop = offset + span - prompt_length
            audio = history[start:stop]
            text = torch.full((len(audio), 1), self.config.text_vocab, device=audio.device, dtype=audio.dtype)
            pieces.append(torch.cat((audio, text), dim=1))
        selected = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        hidden = self._embed_input_ids(selected)
        speaker = info.get(SPEAKER_EMBEDDING)
        position = int(info.get(SPEAKER_POSITION, 0))
        if speaker is not None and offset <= position < offset + span:
            hidden = self._inject_speaker(
                hidden,
                speaker.reshape(1, -1),
                torch.tensor([position - offset], device=input_ids.device, dtype=torch.long),
            )
        return (
            input_ids,
            hidden,
            {
                "_omni_req_id": key,
                SCHEDULED_SPAN: span,
                TERMINAL: terminal,
                STATE: state.payload(),
            },
        )

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Map one scheduled request span to raw ten-column embeddings.

        The runner supplies the absolute computed-token offset, so chunked
        prefill needs no model-owned cursor. Decode reads the existing
        per-request codes.audio payload; sampling/EOS logic is unchanged.
        """
        frames = info.get(FRAMES)
        if frames is None:
            # Retain the P1 numeric-id skeleton and direct legacy callers.
            return input_ids, self._embed_input_ids(input_ids), {}
        if not isinstance(frames, torch.Tensor) or frames.ndim != 2 or frames.shape[1] != 10:
            raise ValueError("zonos2_frames must have shape [T,10]")
        if "_omni_sampling_params" in info and ("_omni_req_id" in info or "request_id" in info):
            try:
                return self._managed_preprocess(input_ids, frames, info)
            except Exception:
                self.on_requests_finished([str(info.get("_omni_req_id") or info["request_id"])])
                raise
        offset = int(info["_omni_num_computed_tokens"])
        is_prefill = bool(info["_omni_is_prefill"])
        if offset < 0:
            raise ValueError("ZONOS2 computed-token offset must be nonnegative")
        span = input_ids.shape[0]
        if is_prefill:
            selected = frames[offset : offset + span]
            if selected.shape[0] != span:
                raise ValueError("Scheduled prefill span exceeds zonos2_frames")
            selected = selected.to(device=input_ids.device)
        else:
            if span != 1:
                raise ValueError("ZONOS2 decode expects one frame per scheduled request")
            codes = (info.get("codes") or {}).get("audio")
            if not isinstance(codes, torch.Tensor) or codes.numel() != self.config.n_codebooks:
                raise ValueError("ZONOS2 decode requires the previous nine codes.audio values")
            if codes.dtype not in (torch.int32, torch.int64):
                raise TypeError("ZONOS2 previous codes must use int32 or int64")
            selected = torch.cat(
                (
                    codes.reshape(1, self.config.n_codebooks).to(device=input_ids.device, dtype=torch.long),
                    torch.full((1, 1), self.config.text_vocab, device=input_ids.device, dtype=torch.long),
                ),
                dim=1,
            )
        hidden = self._embed_input_ids(selected)
        speaker = info.get(SPEAKER_EMBEDDING)
        if is_prefill and speaker is not None:
            position = int(info.get(SPEAKER_POSITION, 0))
            if offset <= position < offset + span:
                if not isinstance(speaker, torch.Tensor):
                    raise TypeError("zonos2_speaker_embedding must be a tensor")
                hidden = self._inject_speaker(
                    hidden,
                    speaker.reshape(1, -1),
                    torch.tensor([position - offset], device=input_ids.device, dtype=torch.long),
                )
        return input_ids, hidden, {}

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        speaker_embeddings: torch.Tensor | None = None,
        speaker_positions: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        infos = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        eligible = kwargs.get("request_sample_eligible")
        if infos and any(FRAMES in i for i in infos) and not all("_omni_req_id" in i and FRAMES in i for i in infos):
            raise ValueError("ZONOS2 cannot mix legacy numeric prompts with request-local framed prompts")
        if infos and all("_omni_req_id" in i and FRAMES in i for i in infos):
            self._sampling_plan = [
                (str(info["_omni_req_id"]), bool(eligible[i]) if eligible is not None else True, info)
                for i, info in enumerate(infos)
            ]
        else:
            self._sampling_plan = []
        self._step_codes = {}
        self._step_payload = None
        if inputs_embeds is None:
            hidden = self._embed_input_ids(input_ids)
        else:
            hidden = inputs_embeds
        if hidden.ndim != 2 or hidden.shape[1] != self.config.dim:
            raise ValueError(f"ZONOS2 raw embeddings must have shape [T, {self.config.dim}]")
        if positions.ndim != 1 or positions.shape[0] != hidden.shape[0]:
            raise ValueError("ZONOS2 positions must contain one position per input frame")
        if (speaker_embeddings is None) != (speaker_positions is None):
            raise ValueError("speaker_embeddings and speaker_positions must be supplied together")
        if speaker_embeddings is not None and speaker_positions is not None:
            # Official semantics replace the entire slot, before emb_norm.
            hidden = self._inject_speaker(hidden, speaker_embeddings, speaker_positions)
        # emb_norm: weight-free RMSNorm applied once at this common boundary,
        # whatever the embedding source (token ids / frame ids / inputs_embeds).
        # Callers producing inputs_embeds must NOT pre-normalize.
        # The reference weight-free norm also uses the CUDA JIT kernel;
        # torch RMSNorm can differ by a BF16 rounding boundary.
        if hidden.device.type == "cuda":
            weight = self._emb_norm_weight
            if weight is None or weight.dtype != hidden.dtype or weight.device != hidden.device:
                weight = torch.ones(self.config.dim, dtype=hidden.dtype, device=hidden.device)
                self._emb_norm_weight = weight
            hidden = zonos2_cuda_rmsnorm(hidden, weight, self.config.norm_eps)
        else:
            hidden = F.rms_norm(hidden, (self.config.dim,), eps=self.config.norm_eps)
        prev_router_state: torch.Tensor | None = None
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden, residual, prev_router_state = layer(hidden, positions, prev_router_state, residual)
        if residual is None:
            return self.out_norm(hidden)
        return self.out_norm(hidden, residual)[0]

    # ------------------------------------------------------------ logits/sample
    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        fused = self.multi_output(hidden_states)  # [N, 9*1026]
        fused = fused.view(-1, self.config.n_codebooks, self.config.codebook_vocab_size)
        # tanh softcap tau=15 on all codebook logits
        cap = self.config.loss_softcap
        fused = cap * torch.tanh(fused / cap)
        self._last_fused_logits = fused
        # LM lifecycle channel: codebook-0 logits.
        lm_logits = fused[:, 0, :].float()
        return lm_logits

    def fused_audio_logits(self) -> torch.Tensor | None:
        """Public accessor for the full [N, 9, 1026] softcapped audio logits
        from the most recent ``compute_logits`` call. Used by the M2a
        teacher-forced parity harness so tests never touch private state."""
        return getattr(self, "_last_fused_logits", None)

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        fused = self.fused_audio_logits()
        if fused is None:
            return None
        plan = getattr(self, "_sampling_plan", [])
        if not plan:
            # Numerical parity and engine warmup have no request context.
            self._last_audio_codes = fused.argmax(-1).to(torch.long)
            self._postprocess_cursor = 0
            return None
        if len(plan) != fused.shape[0]:
            raise ValueError("ZONOS2 sampling rows must match stable request IDs")
        lifecycle = torch.full((len(plan), 1), CONTINUE_TOKEN, device=fused.device, dtype=torch.int64)
        try:
            for index, (key, eligible, info) in enumerate(plan):
                if not eligible:
                    continue  # partial replay/prefill must not emit or consume RNG
                state = self._request_states[key]
                if info.get(TERMINAL):
                    lifecycle[index, 0] = STOP_TOKEN
                    if self._step_payload is not None:
                        self._step_payload["meta"]["eos_frame"][index] = state.eos_frame.reshape(1)
                        self._step_payload["meta"]["is_final"][index] = state.stopped.reshape(1)
                    continue
                target = int(info["_omni_num_computed_tokens"]) + int(info.get(SCHEDULED_SPAN, 1)) - len(info[FRAMES])
                if target != len(state.history):
                    raise ValueError("ZONOS2 sampler history is inconsistent with scheduled position")
                row = sample_frame(fused[index], state.history, state.params, state.seed, len(state.history))
                lifecycle[index, 0] = state.append(row)
                self._step_codes[key] = row.reshape(1, 9)
                if self._step_payload is not None:
                    self._step_payload["codes"]["audio"][index] = row.reshape(1, 9)
                    self._step_payload["meta"]["eos_frame"][index] = state.eos_frame.reshape(1)
                    self._step_payload["meta"]["is_final"][index] = state.stopped.reshape(1)
            return SamplerOutput(sampled_token_ids=lifecycle, logprobs_tensors=None)
        except Exception:
            self.on_requests_finished([p[0] for p in plan])
            raise

    def postprocess(self, hidden_states_slice: torch.Tensor, multimodal_outputs: Any = None, **req_infos: Any) -> dict:
        key = req_infos.get("_omni_req_id")
        if key is not None:
            key = str(key)
            state = self._request_states.get(key)
            if state is None or key not in self._step_codes:
                return {}
            return {"codes": {"audio": self._step_codes[key]}, STATE: state.payload()}
        codes = getattr(self, "_last_audio_codes", None)
        if codes is None:
            return {}
        cursor = int(getattr(self, "_postprocess_cursor", 0))
        if cursor >= len(codes):
            return {}
        self._postprocess_cursor = cursor + 1
        return {"codes": {"audio": codes[cursor : cursor + 1].to(torch.int32)}}

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        plan = getattr(self, "_sampling_plan", [])
        if plan:
            # Runner retains this payload by reference until its post-sampling
            # output builder. sample() fills these row-aligned slots in the same
            # step, including the terminal frame; no one-step output lag.
            self._step_payload = {
                "codes": {"audio": [model_outputs.new_empty((0, 9), dtype=torch.int32) for _ in plan]},
                "meta": {
                    "eos_frame": [model_outputs.new_full((1,), -1, dtype=torch.long) for _ in plan],
                    "is_final": [model_outputs.new_zeros((1,), dtype=torch.bool) for _ in plan],
                },
            }
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=self._step_payload)
        infos = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        codes = [(info.get("codes") or {}).get("audio") for info in infos]
        if any(isinstance(c, torch.Tensor) and c.numel() for c in codes):
            return OmniOutput(
                text_hidden_states=model_outputs,
                multimodal_outputs={
                    "codes": {
                        "audio": [c if isinstance(c, torch.Tensor) else torch.empty(0, dtype=torch.long) for c in codes]
                    }
                },
            )
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={})

    # ------------------------------------------------------------------- load
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the converted safetensors (keys mirror this module tree 1:1).

        L0 completeness contract (hard-fail): every checkpoint tensor must be
        consumed exactly once (nothing skipped, nothing duplicated) and every
        model parameter must be initialized (nothing missing); shape and dtype
        must match exactly. Any violation raises immediately — a partially
        loaded model must never reach the forward pass.
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        skipped: list[str] = []
        duplicates: list[str] = []
        for name, w in weights:
            if name in loaded:
                duplicates.append(name)
                continue
            target = params.get(name)
            if target is None:
                skipped.append(name)
                continue
            if tuple(target.shape) != tuple(w.shape):
                raise RuntimeError(
                    f"[Zonos2] L0 shape mismatch for {name}: checkpoint {tuple(w.shape)} vs model {tuple(target.shape)}"
                )
            if target.dtype != w.dtype:
                raise RuntimeError(
                    f"[Zonos2] L0 dtype mismatch for {name}: checkpoint {w.dtype} vs model {target.dtype}"
                )
            target.data.copy_(w)
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        if skipped or missing or duplicates:
            raise RuntimeError(
                f"[Zonos2] L0 weight completeness failed: "
                f"skipped={len(skipped)} missing={len(missing)} duplicates={len(duplicates)}\n"
                f"  skipped (unexpected checkpoint keys): {skipped[:20]}\n"
                f"  missing (uninitialized model params): {missing[:20]}\n"
                f"  duplicates: {duplicates[:20]}"
            )
        for layer in self.layers:
            if isinstance(layer.feed_forward, Zonos2MoE):
                layer.feed_forward.prepare_expert_weights()
        print(f"[Zonos2] load_weights L0 OK: consumed={len(loaded)} tensors, no residue", flush=True)
        return loaded
