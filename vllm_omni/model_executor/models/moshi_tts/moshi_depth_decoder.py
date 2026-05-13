from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F

from .configuration_moshi import MoshiDepthConfig

logger = logging.getLogger(__name__)


class _MoshiRMSNorm(nn.Module):
    """RMSNorm matching HF's MoshiRMSNorm (upcasts to float32 for the norm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * self.weight.float()).type_as(x)


def _make_depth_norm(config: MoshiDepthConfig) -> nn.Module:
    """Create the appropriate norm layer based on config."""
    if config.norm_type == "layer_norm":
        return nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
    return _MoshiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MoshiFlexibleLinear(nn.Module):
    """Per-codebook linear layer with 3D weight [num_layers, out_size, in_size].

    Each codebook step uses a different weight matrix. During the AR loop,
    the codebook index selects which slice to use.
    """

    def __init__(self, input_size: int, output_size: int, num_layers: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_layers, output_size, input_size))
        self.num_layers = num_layers
        self.output_size = output_size
        self.input_size = input_size

    def forward(self, x: torch.Tensor, layer_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Apply per-codebook linear transformation.

        Args:
            x: [batch, seq_len, input_size] or [batch, input_size]
            layer_idx: [seq_len] tensor of codebook indices, or None to use
                       sequential indices 0..seq_len-1.

        Returns:
            [batch, seq_len, output_size] or [batch, output_size]
        """
        squeezed = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeezed = True

        if layer_idx is not None:
            selected_weights = torch.index_select(self.weight, 0, layer_idx)
        else:
            seq_len = x.shape[1]
            selected_weights = self.weight[:seq_len]

        # selected_weights: [seq_len, out, in]
        # x: [batch, seq_len, in]
        # Batched matmul: [batch, seq_len, 1, in] @ [1, seq_len, in, out]
        #              -> [batch, seq_len, 1, out] -> [batch, seq_len, out]
        w_t = selected_weights.transpose(1, 2).unsqueeze(0)  # [1, seq_len, in, out]
        out = torch.matmul(x.unsqueeze(2), w_t).squeeze(2)  # [batch, seq_len, out]

        if squeezed:
            out = out.squeeze(1)
        return out

    def forward_single(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Single-position fast path: F.linear instead of batched matmul.

        Args:
            x: [B, input_size] — no seq dimension
            idx: Python int — weight index for this position
        Returns:
            [B, output_size]
        """
        return F.linear(x, self.weight[idx])


class _DepthAttention(nn.Module):
    """Multi-head attention with per-codebook weights. No RoPE."""

    def __init__(self, config: MoshiDepthConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_codebooks = config.num_codebooks
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Number of unique weight sets (may be < num_codebooks with weight sharing)
        schedule = config.weights_per_step_schedule
        num_ws = max(schedule) + 1 if schedule else config.num_codebooks

        # Per-codebook q/k/v/o projections via FlexibleLinear
        self.q_proj = MoshiFlexibleLinear(self.hidden_size, self.q_size, num_ws)
        self.k_proj = MoshiFlexibleLinear(self.hidden_size, self.kv_size, num_ws)
        self.v_proj = MoshiFlexibleLinear(self.hidden_size, self.kv_size, num_ws)
        self.o_proj = MoshiFlexibleLinear(self.q_size, self.hidden_size, num_ws)

        # KV cache (lazily initialized)
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        """Re-prefill forward: processes full sequence, no KV cache.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            weight_indices: [seq_len] codebook weight indices for FlexibleLinear
        """
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states, weight_indices)
        k = self.k_proj(hidden_states, weight_indices)
        v = self.v_proj(hidden_states, weight_indices)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # No RoPE — depth decoder doesn't use positional embeddings

        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scaling, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.o_proj(attn_out, weight_indices)
        return out

    # ------------------------------------------------------------------
    #  KV cache methods
    # ------------------------------------------------------------------

    def init_kv_cache(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        """Pre-allocate KV cache for one frame's codebook steps."""
        max_seq = self.num_codebooks
        if (
            self._k_cache is not None
            and self._k_cache.shape[0] >= bsz
            and self._k_cache.device == device
            and self._k_cache.dtype == dtype
        ):
            return
        self._k_cache = torch.zeros(bsz, self.num_kv_heads, max_seq, self.head_dim, device=device, dtype=dtype)
        self._v_cache = torch.zeros(bsz, self.num_kv_heads, max_seq, self.head_dim, device=device, dtype=dtype)

    def forward_cached(self, hidden_states: torch.Tensor, wi: int, step: int) -> torch.Tensor:
        """Forward with KV cache. Only processes 1 new position.

        Args:
            hidden_states: [B, 1, hidden_size] — single new position
            wi: Python int — mapped weight index for this position
            step: current codebook step (0..num_codebooks-1)
        """
        bsz = hidden_states.shape[0]
        h = hidden_states.squeeze(1)  # [B, H]

        # Project Q/K/V via F.linear (cuBLAS gemm, no index_select copy)
        q = self.q_proj.forward_single(h, wi)  # [B, q_size]
        k = self.k_proj.forward_single(h, wi)  # [B, kv_size]
        v = self.v_proj.forward_single(h, wi)  # [B, kv_size]

        q = q.view(bsz, self.num_heads, 1, self.head_dim)  # [B, H, 1, D]
        k = k.view(bsz, self.num_kv_heads, 1, self.head_dim)  # [B, Hkv, 1, D]
        v = v.view(bsz, self.num_kv_heads, 1, self.head_dim)  # [B, Hkv, 1, D]

        # Append to cache at position `step`
        self._k_cache[:bsz, :, step : step + 1, :] = k
        self._v_cache[:bsz, :, step : step + 1, :] = v

        # Attend to all cached K/V up to and including current step
        cached_k = self._k_cache[:bsz, :, : step + 1, :]  # [B, Hkv, step+1, D]
        cached_v = self._v_cache[:bsz, :, : step + 1, :]  # [B, Hkv, step+1, D]

        # Q length is 1 attending to step+1 K/V positions — no causal mask needed
        attn_out = F.scaled_dot_product_attention(q, cached_k, cached_v, scale=self.scaling)
        attn_out = attn_out.reshape(bsz, -1)  # [B, q_size]

        out = self.o_proj.forward_single(attn_out, wi)  # [B, H]
        return out.unsqueeze(1)  # [B, 1, H]


class _DepthGatingMLP(nn.Module):
    """SiLU-gated MLP with per-codebook weights."""

    def __init__(self, config: MoshiDepthConfig) -> None:
        super().__init__()
        self.num_codebooks = config.num_codebooks
        schedule = config.weights_per_step_schedule
        num_ws = max(schedule) + 1 if schedule else config.num_codebooks
        self.fc1 = MoshiFlexibleLinear(config.hidden_size, config.ffn_dim, num_ws)
        self.fc2 = MoshiFlexibleLinear(config.intermediate_size, config.hidden_size, num_ws)

    def forward(self, hidden_states: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states, weight_indices)
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, 2, -1)
        hidden_states = F.silu(hidden_states[..., 0, :]) * hidden_states[..., 1, :]
        hidden_states = self.fc2(hidden_states, weight_indices)
        return hidden_states

    def forward_cached(self, hidden_states: torch.Tensor, wi: int) -> torch.Tensor:
        """Single-position fast path: [B, 1, H] in/out."""
        h = hidden_states.squeeze(1)  # [B, H]
        h = self.fc1.forward_single(h, wi)  # [B, ffn_dim]
        h = h.view(h.shape[0], 2, -1)  # [B, 2, intermediate]
        h = F.silu(h[:, 0, :]) * h[:, 1, :]  # [B, intermediate]
        h = self.fc2.forward_single(h, wi)  # [B, H]
        return h.unsqueeze(1)  # [B, 1, H]


class _DepthDecoderLayer(nn.Module):
    """Transformer layer with per-codebook FlexibleLinear weights."""

    def __init__(self, config: MoshiDepthConfig) -> None:
        super().__init__()
        self.self_attn = _DepthAttention(config)
        self.mlp = _DepthGatingMLP(config)
        # Depth transformer layer norms always use RMSNorm, matching the main
        # transformer. Kyutai's depformer_norm only controls the per-codebook
        # output norms, NOT the transformer layer norms.
        self.input_layernorm = _MoshiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _MoshiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        """Re-prefill forward (full sequence)."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, weight_indices)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, weight_indices)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_cached(self, hidden_states: torch.Tensor, wi: int, step: int) -> torch.Tensor:
        """KV-cached forward (single position) with F.linear fast path."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_cached(hidden_states, wi, step)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.forward_cached(hidden_states, wi)
        hidden_states = residual + hidden_states
        return hidden_states


class MoshiDepthDecoder(nn.Module):
    """Moshi depth decoder for multi-codebook audio token generation.

    At each main transformer step, generates N audio codes autoregressively.
    Supports two modes:
      - Re-prefill: forwards growing sequence (simple, O(N²) work)
      - KV cache: forwards 1 token per step (fast, O(N) work)
    """

    def __init__(self, config: MoshiDepthConfig) -> None:
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        self.audio_vocab_size = config.audio_vocab_size
        self.hidden_size = config.hidden_size

        # Weight sharing schedule: maps step index → weight index.
        schedule = config.weights_per_step_schedule
        if schedule is not None:
            num_weight_sets = max(schedule) + 1
            self.register_buffer("_step_schedule", torch.tensor(schedule, dtype=torch.long), persistent=False)
        else:
            num_weight_sets = config.num_codebooks
            self._step_schedule = None

        # Text embedding (position 0 in AR sequence).
        # When demux_second_stream is active the input token is muxed as
        # ``(second + 1) * card + main``; _text_embed() handles the split.
        self.text_embed_tokens = nn.Embedding(config.vocab_size + 1, config.hidden_size)
        self.text_embed_tokens_second_stream = nn.Embedding(config.vocab_size + 1, config.hidden_size)

        # Audio codebook embeddings (positions 1..num_codebooks-1)
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(config.audio_vocab_size + 1, config.hidden_size) for _ in range(config.num_codebooks - 1)]
        )

        # Per-codebook projection: main_hidden → depth_hidden
        self.input_projections = MoshiFlexibleLinear(config.input_size, config.hidden_size, num_weight_sets)

        # Transformer layers
        self.layers = nn.ModuleList([_DepthDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Per-codebook output heads
        self.lm_heads = MoshiFlexibleLinear(config.hidden_size, config.audio_vocab_size, config.num_codebooks)

        # Per-codebook output norms (Hibiki: LayerNorm, Moshi: none)
        if config.norm_type == "layer_norm":
            self.output_norms: nn.ModuleList | None = nn.ModuleList(
                [nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(config.num_codebooks)]
            )
        else:
            self.output_norms = None

        # Pre-allocated buffers (lazily initialized)
        self._embed_buf: torch.Tensor | None = None
        self._weight_indices: torch.Tensor | None = None

        # KV cache mode flag
        self._use_kv_cache: bool = config.cuda_graphs

        # CUDA graph state (lazily initialized)
        self._cuda_graphs_enabled: bool = False
        self._cuda_graphs_max_bsz: int = 0
        self._cuda_graphs_ready: bool = False
        self._use_torch_compile: bool = False
        # Per-step graphs: _step_graphs[step][padded_bsz] = (graph, static_output)
        self._step_graphs: list[dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]]] = []
        self._bucket_sizes: list[int] = []
        # Stored tensors to prevent GC (CUDA graphs reference this memory)
        self._full_weight_indices: torch.Tensor | None = None
        self._compiled_run_layers_cached: callable | None = None

    def _map_step_to_weight_idx(self, step_positions: torch.Tensor) -> torch.Tensor:
        """Map step indices to weight indices via the schedule (if present)."""
        if self._step_schedule is not None:
            return self._step_schedule[step_positions]
        return step_positions

    def _ensure_buffers(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        max_seq = self.num_codebooks + 1
        if (
            self._embed_buf is not None
            and self._embed_buf.shape[0] >= bsz
            and self._embed_buf.device == device
            and self._embed_buf.dtype == dtype
        ):
            return
        self._embed_buf = torch.zeros(bsz, max_seq, self.hidden_size, dtype=dtype, device=device)
        self._weight_indices = torch.arange(max_seq, dtype=torch.long, device=device)
        if self._step_schedule is not None:
            self._step_schedule = self._step_schedule.to(device=device)

    def _init_kv_caches(self, bsz: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize KV caches for all attention layers."""
        for layer in self.layers:
            layer.self_attn.init_kv_cache(bsz, device, dtype)

    def _run_layers(self, hidden_states: torch.Tensor, weight_indices: torch.Tensor) -> torch.Tensor:
        """Re-prefill: forward full sequence through all layers."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, weight_indices)
        return hidden_states

    def _run_layers_cached(self, hidden_states: torch.Tensor, wi: int, step: int) -> torch.Tensor:
        """KV cache: forward single position through all layers."""
        for layer in self.layers:
            hidden_states = layer.forward_cached(hidden_states, wi, step)
        return hidden_states

    # ------------------------------------------------------------------
    #  CUDA graph support (on top of KV cache)
    # ------------------------------------------------------------------

    def enable_cuda_graphs(self, max_batch_size: int, *, compile: bool = False) -> None:
        """Enable KV cache + CUDA graphs, optionally with torch.compile.

        Args:
            max_batch_size: Maximum batch size for graph capture buckets.
            compile: If True, also use torch.compile for kernel fusion.
                     Adds warmup time but gives additional speedup (~2x on
                     top of CUDA graphs alone).
        """
        self._use_kv_cache = True
        self._cuda_graphs_enabled = True
        self._cuda_graphs_max_bsz = max_batch_size
        self._use_torch_compile = compile
        logger.info(
            "Depth decoder: CUDA graphs enabled (max_batch_size=%d, compile=%s)",
            max_batch_size,
            compile,
        )

    def _padded_bsz(self, bsz: int) -> int:
        for bucket in self._bucket_sizes:
            if bsz <= bucket:
                return bucket
        return bsz

    def _setup_cuda_graphs(self) -> None:
        """Capture per-step CUDA graphs for KV-cached _run_layers_cached.

        With KV cache, each step processes [bsz, 1, hidden] — fixed shape.
        We capture one graph per (step, bucket) to also fix the step-dependent
        weight_idx and KV cache write position.

        torch.compile is used with dynamic=False since all graphs share the
        same [bsz, 1, hidden] input shape (only weight_idx differs, which
        is a captured constant).
        """
        if self._cuda_graphs_ready or not self._cuda_graphs_enabled:
            return

        device = self._embed_buf.device
        if device.type != "cuda":
            logger.warning("Depth decoder: CUDA graphs require CUDA device")
            self._cuda_graphs_enabled = False
            return

        num_cb = self.num_codebooks
        max_bsz = self._cuda_graphs_max_bsz
        self._bucket_sizes = sorted({1 << i for i in range(max_bsz.bit_length()) if (1 << i) <= max_bsz} | {max_bsz})

        # Store weight indices as instance attr to prevent GC
        self._full_weight_indices = self._map_step_to_weight_idx(self._weight_indices[: self.num_codebooks])

        if self._use_torch_compile:
            # Raise recompile limit: dynamic=False specializes on `step` int
            # (KV cache slice step:step+1) AND batch size, so needs
            # num_codebooks × num_buckets specializations.
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, num_cb * len(self._bucket_sizes) + 4
            )
            self._compiled_run_layers_cached = torch.compile(
                self._run_layers_cached,
                dynamic=False,
                options={"epilogue_fusion": False},
            )
            run_fn = self._compiled_run_layers_cached
            logger.info("Depth decoder: using torch.compile for CUDA graph capture")
        else:
            run_fn = self._run_layers_cached
            self._compiled_run_layers_cached = run_fn

        # Ensure KV caches are allocated for max batch size
        self._init_kv_caches(max_bsz, device, self._embed_buf.dtype)

        # Pre-allocate a static input buffer for graph capture
        # (embed_buf position 0 is reused as the static input for all steps)
        static_embed = torch.zeros(max_bsz, 1, self.hidden_size, device=device, dtype=self._embed_buf.dtype)

        # Warmup: run forward for each (step, bucket) to trigger compilation
        for step in range(num_cb):
            wi = int(self._full_weight_indices[step].item())
            for bsz in self._bucket_sizes:
                for _ in range(3):
                    run_fn(static_embed[:bsz], wi, step)
        torch.cuda.synchronize(device)
        logger.info("Depth decoder: warmup done (%d steps × %d buckets)", num_cb, len(self._bucket_sizes))

        # Capture per-step CUDA graphs
        try:
            from vllm.platforms import current_platform

            pool = current_platform.get_global_graph_pool()
        except (ImportError, AttributeError):
            pool = None

        capture_kwargs = {"pool": pool} if pool is not None else {}
        self._step_graphs = []
        for step in range(num_cb):
            wi = int(self._full_weight_indices[step].item())
            step_graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
            for bsz in self._bucket_sizes:
                inp = static_embed[:bsz]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, **capture_kwargs):
                    out = run_fn(inp, wi, step)
                step_graphs[bsz] = (graph, out)
            self._step_graphs.append(step_graphs)

        # Store static_embed so it stays alive and we can write into it before replay
        self._static_embed = static_embed

        self._cuda_graphs_ready = True
        logger.info("Depth decoder: captured %d CUDA graphs", num_cb * len(self._bucket_sizes))

    # ------------------------------------------------------------------
    #  Re-prefill AR step (original path)
    # ------------------------------------------------------------------

    def _text_embed(self, text_token_id: torch.Tensor) -> torch.Tensor:
        """Embed a (possibly muxed) text token.

        When ``second_stream_ahead > 0`` the token is muxed as
        ``(second + 1) * card + main``.  We split it, look up the two
        embedding tables, and sum — matching the main model's
        ``embed_input_ids`` logic.
        """
        card = self.config.vocab_size + 1
        main_ids = text_token_id % card
        second_ids = text_token_id // card - 1  # -1 when no second stream
        main_emb = self.text_embed_tokens(main_ids)
        second_zero = (second_ids < 0).unsqueeze(-1)
        second_emb = self.text_embed_tokens_second_stream(second_ids.clamp(min=0))
        return main_emb + torch.where(second_zero, torch.zeros_like(second_emb), second_emb)

    def _ar_step(
        self,
        step: int,
        bsz: int,
        main_hidden: torch.Tensor,
        text_token_id: torch.Tensor,
        prev_code: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one AR step with re-prefill (growing sequence)."""
        embed_buf = self._embed_buf
        wi = self._weight_indices
        seq_len = step + 1
        weight_pos = self._map_step_to_weight_idx(wi[:seq_len])

        if step == 0:
            text_embed = self._text_embed(text_token_id)
            proj = self.input_projections(main_hidden.unsqueeze(1), weight_pos[0:1])
            embed_buf[:bsz, 0, :] = text_embed + proj.squeeze(1)
        else:
            audio_embed = self.embed_tokens[step - 1](prev_code)
            proj = self.input_projections(main_hidden.unsqueeze(1), weight_pos[step : step + 1])
            embed_buf[:bsz, step, :] = audio_embed + proj.squeeze(1)

        step_input = embed_buf[:bsz, :seq_len, :]

        hidden_out = self._run_layers(step_input, weight_pos)

        last_hidden = hidden_out[:, -1:, :]
        if self.output_norms is not None:
            last_hidden = self.output_norms[step](last_hidden)
        raw_idx = wi[step : step + 1]
        logits = self.lm_heads(last_hidden, raw_idx)
        return logits.squeeze(1)

    # ------------------------------------------------------------------
    #  KV-cached AR step (fast path)
    # ------------------------------------------------------------------

    def _get_weight_idx_int(self, step: int) -> int:
        """Get the mapped weight index as a Python int for step."""
        idx = step
        if self._step_schedule is not None:
            idx = int(self._step_schedule[step].item())
        return idx

    def _ar_step_cached(
        self,
        step: int,
        bsz: int,
        main_hidden: torch.Tensor,
        text_token_id: torch.Tensor,
        prev_code: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one AR step with KV cache + F.linear fast path."""
        wi = self._get_weight_idx_int(step)

        # Embed single position via F.linear
        proj = self.input_projections.forward_single(main_hidden, wi)  # [B, H]
        if step == 0:
            text_embed_val = self._text_embed(text_token_id)
            embed = text_embed_val + proj
        else:
            embed = self.embed_tokens[step - 1](prev_code) + proj

        # Forward through layers with KV cache — [B, 1, H]
        hidden_out = self._run_layers_cached(embed.unsqueeze(1), wi, step)

        # Extract logits via F.linear
        last_h = hidden_out.squeeze(1)  # [B, H]
        if self.output_norms is not None:
            last_h = self.output_norms[step](last_h.unsqueeze(1)).squeeze(1)
        logits = self.lm_heads.forward_single(last_h, step)  # lm_heads uses raw step, not mapped
        return logits

    # ------------------------------------------------------------------
    #  CUDA graph + KV cache AR step
    # ------------------------------------------------------------------

    def _ar_step_cuda_graph(
        self,
        step: int,
        bsz: int,
        padded_bsz: int,
        main_hidden: torch.Tensor,
        text_token_id: torch.Tensor,
        prev_code: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one AR step using CUDA graph replay (KV cache + F.linear)."""
        wi = self._get_weight_idx_int(step)

        # Embed single position via F.linear
        proj = self.input_projections.forward_single(main_hidden, wi)
        if step == 0:
            embed = self._text_embed(text_token_id) + proj
        else:
            embed = self.embed_tokens[step - 1](prev_code) + proj

        # Copy into static buffer and replay graph
        graph_entry = self._step_graphs[step].get(padded_bsz)
        if graph_entry is not None:
            self._static_embed[:bsz, 0, :] = embed
            if padded_bsz > bsz:
                self._static_embed[bsz:padded_bsz, 0, :].zero_()
            graph_entry[0].replay()
            hidden_out = graph_entry[1][:bsz]
        else:
            hidden_out = self._run_layers_cached(embed.unsqueeze(1), wi, step)

        # Extract logits via F.linear
        last_h = hidden_out.squeeze(1)  # [B, H]
        if self.output_norms is not None:
            last_h = self.output_norms[step](last_h.unsqueeze(1)).squeeze(1)
        logits = self.lm_heads.forward_single(last_h, step)
        return logits

    # ------------------------------------------------------------------
    #  Main forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(
        self,
        main_hidden: torch.Tensor,
        text_token_id: torch.Tensor,
        *,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 25,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate audio codebook tokens autoregressively.

        Args:
            main_hidden: [B, main_hidden_size] last hidden from main transformer.
            text_token_id: [B] or [B, 1] sampled text token IDs.
            do_sample: Whether to sample (True) or use greedy decoding (False).
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.

        Returns:
            all_codes: [B, num_codebooks] audio codes.
        """
        bsz = main_hidden.shape[0]
        num_cb = self.num_codebooks
        device = main_hidden.device
        dtype = main_hidden.dtype

        text_token_id = text_token_id.reshape(bsz)
        all_codes = torch.empty(bsz, num_cb, dtype=torch.long, device=device)

        self._ensure_buffers(bsz, device, dtype)

        use_sampling = do_sample and temperature > 0
        inv_temperature = 1.0 / max(temperature, 1e-6) if use_sampling else 0.0

        if self._cuda_graphs_enabled:
            self._init_kv_caches(max(bsz, self._cuda_graphs_max_bsz), device, dtype)
            self._setup_cuda_graphs()
            padded_bsz = self._padded_bsz(bsz) if self._cuda_graphs_ready else bsz
            for step in range(num_cb):
                prev_code = all_codes[:, step - 1] if step > 0 else None
                logits = self._ar_step_cuda_graph(step, bsz, padded_bsz, main_hidden, text_token_id, prev_code)
                all_codes[:, step] = self._sample(logits, use_sampling, inv_temperature, top_k, top_p).reshape(bsz)
        elif self._use_kv_cache:
            self._init_kv_caches(bsz, device, dtype)
            for step in range(num_cb):
                prev_code = all_codes[:, step - 1] if step > 0 else None
                logits = self._ar_step_cached(step, bsz, main_hidden, text_token_id, prev_code)
                all_codes[:, step] = self._sample(logits, use_sampling, inv_temperature, top_k, top_p).reshape(bsz)
        else:
            for step in range(num_cb):
                prev_code = all_codes[:, step - 1] if step > 0 else None
                logits = self._ar_step(step, bsz, main_hidden, text_token_id, prev_code)
                all_codes[:, step] = self._sample(logits, use_sampling, inv_temperature, top_k, top_p).reshape(bsz)

        return all_codes

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        use_sampling: bool,
        inv_temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample or argmax from logits."""
        if use_sampling:
            scaled = logits * inv_temperature
            if top_k > 0:
                topk_vals, _ = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1)
                scaled = scaled.masked_fill(scaled < topk_vals[:, -1:], float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                scaled = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return logits.argmax(dim=-1, keepdim=True)

    def get_logits(
        self,
        main_hidden: torch.Tensor,
        text_token_id: torch.Tensor,
        audio_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Get logits for all codebook positions (for testing/comparison).

        Args:
            main_hidden: [B, main_hidden_size]
            text_token_id: [B]
            audio_codes: [B, num_codebooks] known audio codes

        Returns:
            logits: [B, num_codebooks, audio_vocab_size]
        """
        bsz = main_hidden.shape[0]
        num_cb = self.num_codebooks
        device = main_hidden.device
        dtype = main_hidden.dtype

        text_token_id = text_token_id.reshape(bsz)
        self._ensure_buffers(bsz, device, dtype)

        all_logits = torch.empty(bsz, num_cb, self.audio_vocab_size, dtype=dtype, device=device)

        if self._use_kv_cache:
            self._init_kv_caches(bsz, device, dtype)
            for step in range(num_cb):
                prev_code = audio_codes[:, step - 1] if step > 0 else None
                all_logits[:, step, :] = self._ar_step_cached(step, bsz, main_hidden, text_token_id, prev_code)
        else:
            for step in range(num_cb):
                prev_code = audio_codes[:, step - 1] if step > 0 else None
                all_logits[:, step, :] = self._ar_step(step, bsz, main_hidden, text_token_id, prev_code)

        return all_logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load depth decoder weights from HF checkpoint.

        Expects weights with 'depth_decoder.' prefix already stripped.
        FlexibleLinear weights are 3D [num_codebooks, out, in].
        Attention weights come through MoshiLinear: strip '.linear.' infix.
        """
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        for name, tensor in weights:
            # Strip .linear. infix from MoshiLinear-wrapped attention projections
            name = name.replace(".q_proj.linear.", ".q_proj.")
            name = name.replace(".k_proj.linear.", ".k_proj.")
            name = name.replace(".v_proj.linear.", ".v_proj.")
            name = name.replace(".o_proj.linear.", ".o_proj.")

            if name not in params_dict:
                logger.warning("Depth decoder: skipping unknown weight %s", name)
                continue

            param = params_dict[name]
            if param.shape != tensor.shape:
                logger.warning(
                    "Depth decoder: shape mismatch for %s: expected %s, got %s",
                    name,
                    param.shape,
                    tensor.shape,
                )
                continue

            param.data.copy_(tensor)
            loaded.add(name)

        # Report loading summary
        all_params = set(params_dict.keys())
        missing = all_params - loaded
        if missing:
            logger.warning(
                "Depth decoder: %d/%d params NOT loaded (random init): %s",
                len(missing),
                len(all_params),
                sorted(missing)[:20],
            )
        logger.info("Depth decoder: loaded %d/%d params", len(loaded), len(all_params))

        # EOS → PAD replacement for Hibiki (prevents early termination)
        if self.output_norms is not None:  # Hibiki indicator
            self.text_embed_tokens.weight.data[2] = self.text_embed_tokens.weight.data[3].clone()
            logger.info("Depth decoder: replaced EOS (2) with PAD (3) in text embeddings (Hibiki)")

        return loaded
