from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)

from vllm_omni.platforms import current_omni_platform

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

logger = init_logger(__name__)


# ===================================================================
#  HF-numerics-compatible layers for code predictor
# ===================================================================
#
# These use plain PyTorch ops (nn.Linear, manual RMSNorm in float32,
# rotate_half RoPE) to produce outputs numerically identical to the
# HuggingFace reference.  vLLM's fused kernels (RMSNorm, QKVParallel,
# get_rope) introduce small precision differences that compound across
# the 15 autoregressive steps of the code predictor, causing severe
# audio quality degradation (UTMOS ~4.26 → ~2.66).
#
# See: https://github.com/vllm-project/vllm-omni/issues/2274


class _RMSNorm(nn.Module):
    """RMSNorm matching HuggingFace's Qwen3TTSRMSNorm exactly.

    Computes variance in float32 to avoid bfloat16 precision loss.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _RotaryEmbedding(nn.Module):
    """RoPE matching HuggingFace's Qwen3TTSRotaryEmbedding exactly.

    Forces float32 computation for cos/sin, matching HF's torch.autocast(enabled=False).
    """

    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig) -> None:
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        # Standard default RoPE
        rope_theta = getattr(config, "rope_theta", 10000.0)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 (matching HF)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class _CodePredictorAttention(nn.Module):
    """Standalone multi-head attention for code predictor.

    Uses F.scaled_dot_product_attention with HF-compatible RoPE and RMSNorm.
    Input: [B, seq_len, hidden_size], output: [B, seq_len, hidden_size].
    """

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        # Separate q/k/v projections matching HF (no fused packing)
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
        hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)

        cos, sin = position_embeddings
        # cos/sin are [batch, seq_len, head_dim], need unsqueeze at dim=1 for heads
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self._use_gqa,
        )

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(attn_out)
        return output


class _CodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching HF's Qwen3TTSTalkerTextMLP."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer for code predictor (SDPA, no KV cache)."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = _CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = _CodePredictorMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ===================================================================
#  Code Predictor Transformer Model
# ===================================================================


class Qwen3TTSTalkerCodePredictorModelVLLM(nn.Module):
    """Transformer model for the code predictor (re-prefill, no KV cache)."""

    def __init__(
        self,
        config: Qwen3TTSTalkerCodePredictorConfig,
        *,
        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [_CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{i}") for i in range(config.num_hidden_layers)]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

        # Codec embeddings: one per residual group. Stored in talker hidden dim
        # (some checkpoints use talker_hidden_size != code_predictor hidden_size).
        emb_dim = int(talker_hidden_size) if talker_hidden_size is not None else int(config.hidden_size)
        self.codec_embedding = nn.ModuleList(
            [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
        )

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# ===================================================================
#  Code Predictor Wrapper (optimized re-prefill + torch.compile)
# ===================================================================


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(nn.Module):
    """vLLM-native code_predictor for the AR talker (residual codebooks).

    Re-prefill approach: each AR step forwards the full growing sequence
    through the 5-layer transformer. No KV cache needed. This trades
    ~O(T^2) extra attention FLOPs (negligible for T=16, 5 layers) for
    zero KV cache management overhead and a simpler execution model.

    Uses HF-compatible layers (plain nn.Linear, float32 RMSNorm, rotate_half
    RoPE) to ensure numerical fidelity with the reference implementation.
    Precision matters here because small errors compound across 15 AR steps.

    Optimizations preserved:
      1. torch.compile on model forward -- fuses small kernel launches.
      2. Pre-allocated embedding buffer [B, max_seq, H] -- no torch.cat per step.
      3. Projection caching -- each token projected once and cached.
      4. Pre-allocated position_ids -- no torch.arange per step.
      5. Inline sampling -- no custom op / forward_context overhead.
      6. Cached module references -- bypass nn.Module.__call__ overhead.
      7. CUDA graphs per batch-size bucket.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = config
        self.talker_config = talker_config

        self.model = Qwen3TTSTalkerCodePredictorModelVLLM(
            config,
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=f"{prefix}.model",
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_code_groups - 1)]
        )

        if config.hidden_size != talker_config.hidden_size:
            self.small_to_mtp_projection = nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        self._num_groups = int(config.num_code_groups)
        self._talker_hidden = int(talker_config.hidden_size)
        self._cp_hidden = int(config.hidden_size)

        # Pre-allocated buffers (lazily initialized on first forward).
        self._proj_buf: torch.Tensor | None = None

        # torch.compile + warmup state (lazily initialized in _setup_compile).
        self._compiled_model_fwd = None
        self._bucket_sizes: list[int] = []
        self._bucket_pos_ids: dict[int, torch.Tensor] = {}
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None
        self._cuda_graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        with set_current_vllm_config(self._vllm_config):
            loaded: set[str] = set()
            model_weights: list[tuple[str, torch.Tensor]] = []
            other_weights: list[tuple[str, torch.Tensor]] = []
            for name, w in weights:
                if name.startswith("model."):
                    model_weights.append((name[len("model.") :], w))
                else:
                    other_weights.append((name, w))

            loaded_model = self.model.load_weights(model_weights)
            loaded |= {f"model.{n}" for n in loaded_model}

            params = dict(self.named_parameters(remove_duplicate=False))
            for name, w in other_weights:
                if name not in params:
                    continue
                default_weight_loader(params[name], w)
                loaded.add(name)

            return loaded

    # ------------------------------------------------------------------
    #  Pre-allocated buffer management
    # ------------------------------------------------------------------

    def _ensure_buffers(self, device: torch.device, dtype: torch.dtype) -> None:
        max_seq = self._num_groups + 1
        if self._proj_buf is not None and self._proj_buf.device == device and self._proj_buf.dtype == dtype:
            return
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        self._proj_buf = torch.zeros(
            max_bsz,
            max_seq,
            self._cp_hidden,
            dtype=dtype,
            device=device,
        )

    def _setup_compile(self) -> None:
        """Lazily set up torch.compile with manual CUDA graph capture."""
        if self._compiled_model_fwd is not None:
            return
        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)
        if not current_omni_platform.supports_torch_inductor():
            logger.warning_once("code_predictor: torch.compile disabled")
            self._compiled_model_fwd = self.model.forward
            return

        # torch.compile fuses RMSNorm/RoPE in ways that lose float32
        # precision, compounding across 15 AR steps.  Use torch.compile
        # with options that disable the problematic fusions while still
        # getting kernel fusion benefits for the linear layers and SDPA.
        self._compiled_model_fwd = torch.compile(
            self.model.forward,
            dynamic=False,
            options={
                "epilogue_fusion": False,
            },
        )
        self._warmup_buckets()
        self._capture_cuda_graphs()
        logger.info("code_predictor: torch.compile (no epilogue fusion) + CUDA graphs")

    def _padded_bsz(self, bsz: int) -> int:
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    def _warmup_buckets(self) -> None:
        """Warmup power-of-2 batch-size buckets to front-load Inductor compilation."""
        max_bsz = self._vllm_config.scheduler_config.max_num_seqs
        bucket_sizes = [1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz]
        if max_bsz not in bucket_sizes:
            bucket_sizes.append(max_bsz)
        self._bucket_sizes = sorted(bucket_sizes)

        max_seq = self._num_groups + 1
        device = next(self.model.parameters()).device

        proj_buf = self._proj_buf
        for bsz in self._bucket_sizes:
            # position_ids: [batch, seq_len] for HF-style RoPE
            pos_ids = torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            self._bucket_pos_ids[bsz] = pos_ids
            for _ in range(3):
                self._compiled_model_fwd(proj_buf[:bsz, :max_seq, :], pos_ids)
        logger.info("code_predictor: warmup done for buckets %s", self._bucket_sizes)

    def _capture_cuda_graphs(self) -> None:
        """Capture a CUDA graph per bucket using vLLM's global graph pool."""
        from vllm.platforms import current_platform

        pool = current_platform.get_global_graph_pool()

        max_seq = self._num_groups + 1
        proj_buf = self._proj_buf

        for bsz in self._bucket_sizes:
            static_input = proj_buf[:bsz, :max_seq, :]
            pos_ids = self._bucket_pos_ids[bsz]

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                static_output = self._compiled_model_fwd(static_input, pos_ids)

            self._cuda_graphs[bsz] = (g, static_output)

        logger.info("code_predictor: captured CUDA graphs for buckets %s", self._bucket_sizes)

    # ------------------------------------------------------------------
    #  Optimized forward: re-prefill + torch.compile + projection cache
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Predict residual codebooks 1..Q-1 autoregressively via re-prefill.

        torch.compile fuses the ~60 small kernel launches per step into fewer
        fused kernels, reducing kernel launch overhead by ~75%.

        Projection caching: each token is projected once via small_to_mtp_projection
        and cached in _proj_buf, avoiding redundant re-projection of past tokens.
        """
        bsz = int(layer0_code.shape[0])
        num_groups = self._num_groups
        device = layer0_code.device
        dtype = layer0_embed.dtype

        all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
        all_codes[:, 0] = layer0_code.reshape(bsz)

        self._ensure_buffers(device, dtype)
        self._setup_compile()

        proj_buf = self._proj_buf
        max_seq = self._num_groups + 1

        projection = self.small_to_mtp_projection
        model_fwd = self._compiled_model_fwd
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        use_sampling = do_sample and temperature > 0
        inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
        if use_sampling and top_p != 1.0:
            raise NotImplementedError(
                "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
            )

        padded_bsz = self._padded_bsz(bsz)
        proj_buf[:padded_bsz].zero_()

        proj_buf[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1)).reshape(bsz, -1)
        proj_buf[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)
        full_pos_ids = self._bucket_pos_ids.get(padded_bsz)
        if full_pos_ids is None:
            full_pos_ids = torch.arange(max_seq, device=device, dtype=torch.long).unsqueeze(0).expand(padded_bsz, -1)

        # Use captured CUDA graph if available, otherwise call compiled fn.
        cuda_graph_entry = self._cuda_graphs.get(padded_bsz)

        for step in range(1, num_groups):
            if cuda_graph_entry is not None:
                cuda_graph_entry[0].replay()
                hidden_out = cuda_graph_entry[1]
            else:
                hidden_out = model_fwd(proj_buf[:padded_bsz, :max_seq, :], full_pos_ids)
            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])

            if use_sampling:
                scaled = logits * inv_temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)

            all_codes[:, step] = next_ids.reshape(bsz)

            if step < num_groups - 1:
                new_embed = codec_embeds[step - 1](next_ids)
                proj_buf[:bsz, step + 1, :] = projection(new_embed.reshape(bsz, 1, -1)).reshape(bsz, -1)

        return all_codes
