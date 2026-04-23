"""Qwen3 Code Predictor -- optimized re-prefill, no KV cache.

Shared by Qwen3-Omni and Qwen3-TTS talker models.

* SDPA attention (F.scaled_dot_product_attention) with native GQA support
* HF-compatible numerics (float32 RMSNorm, float32 RoPE, separate linear layers)
* ``@support_torch_compile`` on the wrapper so the (projection + transformer)
  block gets captured by vLLM's cudagraph dispatcher; the AR loop + sampling
  live outside the compiled region and call ``self(...)`` per step.
* Persistent scratch buffers sized to ``max_num_batched_tokens`` provide the
  stable addresses required across cudagraph replays.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


# ===================================================================
# HF-numerics-compatible layers for code predictor
# ===================================================================
#
# These use plain PyTorch ops (nn.Linear, manual RMSNorm in float32,
# rotate_half RoPE) to produce outputs numerically identical to the
# HuggingFace reference. vLLM's fused kernels (RMSNorm, QKVParallel,
# get_rope) introduce small precision differences that compound across
# the autoregressive steps of the code predictor, causing severe
# audio quality degradation.
#
# See: https://github.com/vllm-project/vllm-omni/issues/2274


class _RMSNorm(nn.Module):
    """RMSNorm matching HuggingFace's implementation exactly.

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


# Gumbel-max categorical draw: argmax(logits + Gumbel(0, 1)). Equivalent in
# distribution to softmax+multinomial, but cudagraph-safe (only uniform_ / log /
# argmax) and avoids the device-side assert that ``torch.multinomial`` raises on
# all-zero probability rows during warmup.


def _gumbel_sample(logits: torch.Tensor) -> torch.Tensor:
    u = torch.empty_like(logits).uniform_(1e-20, 1.0 - 1e-20)
    return (logits - torch.log(-torch.log(u))).argmax(dim=-1)


def _multinomial_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


class _RotaryEmbedding(nn.Module):
    """RoPE matching HuggingFace's implementation exactly.

    Forces float32 computation for cos/sin, matching HF's torch.autocast(enabled=False).
    """

    def __init__(self, config) -> None:
        super().__init__()
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
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


# ===================================================================
#  Attention
# ===================================================================


class CodePredictorAttention(nn.Module):
    """Multi-head self-attention for code predictor.

    Uses ``F.scaled_dot_product_attention`` with HF-compatible RoPE and RMSNorm.
    No KV cache -- the code predictor always re-prefills the full (short)
    sequence each AR step.

    Input : [B, seq_len, hidden_size]
    Output: [B, seq_len, hidden_size]
    """

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5
        self._use_gqa = self.num_kv_heads != self.num_heads

        # Separate q/k/v projections matching HF (no fused packing)
        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
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
        return self.o_proj(attn_out)


# ===================================================================
#  MLP
# ===================================================================


class CodePredictorMLP(nn.Module):
    """SiLU-gated MLP for code predictor, matching HF's implementation."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ===================================================================
#  Decoder Layer
# ===================================================================


class CodePredictorDecoderLayer(nn.Module):
    """Transformer decoder layer (SDPA, no KV cache)."""

    def __init__(self, config, *, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = CodePredictorAttention(config, prefix=f"{prefix}.self_attn")
        self.mlp = CodePredictorMLP(config, prefix=f"{prefix}.mlp")
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
#  Base Transformer Model (re-prefill, no KV cache)
# ===================================================================


class CodePredictorBaseModel(nn.Module):
    """Inner transformer for code predictor.

    Signature: ``forward(inputs_embeds, position_ids) -> hidden_states``
    """

    def __init__(
        self,
        config,
        *,
        embedding_dim: int | None = None,
        use_parallel_embedding: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        emb_dim = int(embedding_dim) if embedding_dim is not None else int(config.hidden_size)
        if use_parallel_embedding:
            self.codec_embedding = nn.ModuleList(
                [VocabParallelEmbedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )
        else:
            self.codec_embedding = nn.ModuleList(
                [nn.Embedding(config.vocab_size, emb_dim) for _ in range(config.num_code_groups - 1)]
            )

        self.layers = nn.ModuleList(
            [
                CodePredictorDecoderLayer(config, prefix=f"{prefix}.layers.{idx}")
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RotaryEmbedding(config)

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
#  Wrapper Configuration
# ===================================================================


@dataclasses.dataclass
class CodePredictorWrapperConfig:
    """Controls behavioral differences between model-specific code predictors."""

    use_parallel_embedding: bool = False
    use_projection: bool = False
    return_proj_buf: bool = False
    sampling_mode: str = "stored"
    # Use Gumbel-max for the categorical draw (cudagraph-safe, warmup-safe).
    # Set to False to fall back to softmax+multinomial.
    use_gumbel: bool = True


# ===================================================================
#  Code Predictor Wrapper (torch.compile + vLLM cudagraph dispatch)
# ===================================================================


# ``dynamic_arg_dims`` is explicit because this module uses PEP 563
# (``from __future__ import annotations``), which defeats annotation-based
# inference in ``@support_torch_compile``.
@support_torch_compile(dynamic_arg_dims={"inputs_embeds": 0, "position_ids": 0})
class CodePredictorWrapper(nn.Module):
    """Optimized code predictor -- re-prefill approach, no KV cache.

    Each AR step forwards the full growing sequence (len 2 -> num_code_groups+1)
    through the transformer. The extra O(T^2) FLOPs are negligible for
    short sequences, and this avoids all KV-cache management overhead.

    ``forward(inputs_embeds, position_ids)`` is the compiled/captured region
    (projection + transformer + norm). ``generate(...)`` owns the AR loop and
    sampling, calling ``self(...)`` per step to hit the compiled path.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        cp_config,
        wrapper_config: CodePredictorWrapperConfig,
        talker_hidden_size: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._vllm_config = vllm_config
        self.config = cp_config
        self._wrapper_config = wrapper_config
        self.prefix = prefix

        self._num_groups = int(cp_config.num_code_groups)
        self._cp_hidden = int(cp_config.hidden_size)
        self._max_seq = self._num_groups + 1

        # For Omni backward compat (accessed by the talker)
        self.num_code_groups = self._num_groups

        _talker_hidden = int(talker_hidden_size) if talker_hidden_size is not None else self._cp_hidden

        self.model = CodePredictorBaseModel(
            cp_config,
            embedding_dim=_talker_hidden,
            use_parallel_embedding=wrapper_config.use_parallel_embedding,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        self.lm_head = nn.ModuleList(
            [nn.Linear(cp_config.hidden_size, cp_config.vocab_size, bias=False) for _ in range(self._num_groups - 1)]
        )

        # Projection: Identity when hidden sizes match or not needed
        if wrapper_config.use_projection and _talker_hidden != self._cp_hidden:
            self.small_to_mtp_projection = nn.Linear(_talker_hidden, self._cp_hidden, bias=True)
        else:
            self.small_to_mtp_projection = nn.Identity()

        # Sampling defaults for "stored" mode
        self._top_k: int = 50
        self._top_p: float = 0.8

        # Persistent scratch buffers sized to max_num_batched_tokens so
        # addresses stay stable across cudagraph replays. forward()/generate()
        # slice ``[:B]`` out of these.
        # NOTE: real runtime + cudagraph capture never push B above
        # max_num_seqs (predictor only sees decode positions, each request
        # contributes 1 decode token, and _get_decode_idxs bisects into
        # cudagraph_capture_sizes). The only path that needs the full
        # max_num_batched_tokens extent is the profile dummy run
        # (is_profile=True, attn_metadata=None -> pure-decode fallback in
        # the talker). If profile is reworked to cap B at max_num_seqs,
        # this buffer can be shrunk to max_num_seqs.
        max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        dtype = vllm_config.model_config.dtype
        self.register_buffer(
            "_cp_inputs_embeds",
            torch.zeros(max_num_tokens, self._max_seq, _talker_hidden, dtype=dtype),
            persistent=False,
        )
        pos_row = torch.arange(self._max_seq, dtype=torch.long)
        self.register_buffer(
            "_cp_position_ids",
            pos_row.unsqueeze(0).expand(max_num_tokens, -1).contiguous(),
            persistent=False,
        )

        # Cached on first generate() call so weights/dtype are finalized.
        self._lm_heads_list: list[nn.Module] | None = None
        self._codec_embeds_list: list[nn.Module] | None = None

    def get_input_embeddings(self) -> nn.ModuleList:
        return self.model.get_input_embeddings()

    def set_sampling_params(self, top_k: int = 50, top_p: float = 0.8) -> None:
        """Configure sampling parameters for ``stored`` sampling mode."""
        self._top_k = top_k
        self._top_p = top_p
        logger.debug("Sampling parameters updated: top_k=%d, top_p=%.2f", top_k, top_p)

    # ------------------------------------------------------------------
    #  Compiled region: projection + inner transformer
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled region: projection + inner transformer.

        ``inputs_embeds``: ``[B, max_seq, talker_hidden]``;
        ``position_ids``: ``[B, max_seq]``. Returns ``[B, max_seq, cp_hidden]``.
        """
        projected = self.small_to_mtp_projection(inputs_embeds)
        return self.model(projected, position_ids)

    # ------------------------------------------------------------------
    #  Non-compiled AR loop + sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict residual codebooks 1..G-1 autoregressively via re-prefill.

        Returns ``codes`` (``[B, num_groups]``) or ``(codes, inputs_embeds_buf)``
        depending on ``return_proj_buf``.
        """
        bsz = int(layer0_code.shape[0])
        num_groups = self._num_groups
        device = layer0_code.device

        if self._lm_heads_list is None:
            self._lm_heads_list = list(self.lm_head)
        if self._codec_embeds_list is None:
            self._codec_embeds_list = list(self.model.codec_embedding)
        lm_heads = self._lm_heads_list
        codec_embeds = self._codec_embeds_list

        inputs_buf = self._cp_inputs_embeds[:bsz]
        pos_ids = self._cp_position_ids[:bsz]
        buf_dtype = inputs_buf.dtype

        # Zero the active slice (persistent buffer may carry stale rows).
        inputs_buf.zero_()
        inputs_buf[:, 0, :] = last_talker_hidden.reshape(bsz, -1).to(buf_dtype)
        inputs_buf[:, 1, :] = layer0_embed.reshape(bsz, -1).to(buf_dtype)

        stored_mode = self._wrapper_config.sampling_mode == "stored"
        use_gumbel = self._wrapper_config.use_gumbel
        if stored_mode:
            s_top_k = self._top_k
            s_top_p = self._top_p
        else:
            use_sampling = do_sample and temperature > 0
            inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0
            if use_sampling and top_p != 1.0:
                raise NotImplementedError(
                    "top_p sampling is not implemented for the vLLM-native code predictor; please set top_p=1.0."
                )

        def _draw(filtered_logits: torch.Tensor) -> torch.Tensor:
            if use_gumbel:
                return _gumbel_sample(filtered_logits).unsqueeze(-1)
            return _multinomial_sample(filtered_logits).unsqueeze(-1)

        if self._wrapper_config.return_proj_buf:
            all_codes = torch.empty(bsz, num_groups, 1, dtype=torch.int64, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz, -1)[:, :1]
        else:
            all_codes = torch.empty(bsz, num_groups, dtype=torch.long, device=device)
            all_codes[:, 0] = layer0_code.reshape(bsz)

        for step in range(1, num_groups):
            hidden_out = self(inputs_buf, pos_ids)
            logits = lm_heads[step - 1](hidden_out[:, step, :])

            if stored_mode:
                if s_top_k > 0:
                    topk_vals, _ = logits.topk(s_top_k, dim=-1)
                    logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
                if s_top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    remove_mask = (cumulative_probs - sorted_probs) >= s_top_p
                    sorted_logits[remove_mask] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                code = _draw(logits)
            elif use_sampling:
                scaled = logits * inv_temperature
                if top_k > 0:
                    topk_vals, _ = scaled.topk(top_k, dim=-1)
                    scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
                code = _draw(scaled)
            else:
                code = logits.argmax(dim=-1, keepdim=True)

            if self._wrapper_config.return_proj_buf:
                all_codes[:, step] = code
            else:
                all_codes[:, step] = code.reshape(bsz)

            if step < num_groups - 1 or self._wrapper_config.return_proj_buf:
                new_embed = codec_embeds[step - 1](code.reshape(bsz))
                inputs_buf[:, step + 1, :] = new_embed.to(buf_dtype)

        if self._wrapper_config.return_proj_buf:
            return all_codes, inputs_buf.clone()
        return all_codes

    # ------------------------------------------------------------------
    #  Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Wrapped in set_current_vllm_config so VocabParallelEmbedding weight
        # loaders (Qwen3-Omni subclass) can read TP metadata.
        with set_current_vllm_config(self._vllm_config):
            loaded: set[str] = set()
            model_weights: list[tuple[str, torch.Tensor]] = []
            other_weights: list[tuple[str, torch.Tensor]] = []

            for name, w in weights:
                if "rotary_emb.inv_freq" in name:
                    continue
                if name.startswith("model."):
                    model_weights.append((name[len("model.") :], w))
                else:
                    other_weights.append((name, w))

            loaded_model = self.model.load_weights(model_weights)
            loaded |= {f"model.{n}" for n in loaded_model}

            params = dict(self.named_parameters(remove_duplicate=False))
            for name, w in other_weights:
                param = params.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, w)
                loaded.add(name)

            return loaded
