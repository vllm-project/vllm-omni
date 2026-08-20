# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex depformer: the per-step audio code predictor.

The depformer is the small autoregressive head that, conditioned on one temporal
hidden state, predicts the ``dep_q`` audio codebooks of a single Mimi frame. It
is a faithful eager port of Moshi's depformer (``moshi.models.lm`` +
``moshi.modules.transformer``), kept deliberately outside vLLM's paged-attention
engine: its attention KV is reset every frame and only spans the ``dep_q`` inner
steps, so paging buys nothing here. This mirrors how the existing TTS code
predictors live alongside, not inside, the AR talker.

Faithful-port details (all measured from ``nvidia/personaplex-7b-v1``):

* **Per-step weights** (``weights_per_step``): each inner step ``t`` uses its own
  attention in/out projection and its own gating MLP, selected by index (Moshi's
  ``multi_linear``). There are ``dep_q`` weight sets per layer.
* **No positional embedding** (``depformer_pos_emb="none"``): no RoPE, no sin.
* **Causal attention over all inner steps**: Moshi forces the depformer's
  ``context`` to None, so each step attends to every prior inner step (KV
  capacity == ``weights_per_step``); the KV cache is rebuilt every temporal frame.
* **fp32 RMSNorm** (``rms_norm_f32``, eps 1e-8) with the weight stored as
  ``alpha`` of shape ``[1, 1, dim]``.
* **SiLU gating** MLP (Moshi ``ActivationGating``): ``linear_in`` projects to
  ``2 * hidden``; the halves are combined as ``silu(a) * b`` then ``linear_out``.
* **ScaledEmbedding** with ``zero_idx = -1`` (the special token maps to the zero
  vector); audio table is ``card + 1`` rows, text table ``text_card + 1``.

The checkpoint ships ``dep_q = 8`` weight sets; the production loader (Moshi's
``get_moshi_lm``) overrides ``dep_q = 16`` and fills sets 8..15 by copying 0..7.
:meth:`PersonaPlexDepformer.load_weights` replicates that expansion so the module
loads directly from the raw ``model.safetensors`` as well as from an
already-expanded Moshi instance.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexDepformerConfig,
)

__all__ = ["PersonaPlexDepformer"]

# Moshi's special "no token / zero embedding" sentinel (``LMModel.zero_token_id``).
_ZERO_IDX = -1

# A builder takes the source state dict and returns the tensor for one target
# param, or None when the source lacks it.
_Builder = Callable[[dict], "torch.Tensor | None"]


def _rms_norm_f32(x: torch.Tensor, alpha: torch.Tensor, eps: float) -> torch.Tensor:
    """Moshi ``_rms_norm`` with the fp32 accumulation path.

    Variance and the alpha multiply run in float32 before casting back, matching
    ``moshi.modules.transformer._rms_norm`` exactly (``dtype=torch.float``).
    """
    x_dtype = x.dtype
    x_f32 = x.float()
    var = eps + x_f32.pow(2).mean(dim=-1, keepdim=True)
    y = x_f32 * (alpha.float() * torch.rsqrt(var))
    return y.to(x_dtype)


class _ScaledEmbedding(nn.Embedding):
    """``nn.Embedding`` matching Moshi's ``ScaledEmbedding`` (no norm, zero_idx).

    Inputs equal to ``zero_idx`` produce the zero vector; all other inputs are
    clamped to ``>= 0`` before the lookup. ``norm_emb`` is False for PersonaPlex,
    so no embedding norm is applied.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, zero_idx: int = _ZERO_IDX) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.zero_idx = zero_idx

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        is_zero = idx == self.zero_idx
        y = super().forward(idx.clamp(min=0))
        zero = torch.zeros(1, dtype=y.dtype, device=y.device)
        return torch.where(is_zero[..., None], zero, y)


class _DepformerLayer(nn.Module):
    """One depformer transformer layer with per-step attention + gating.

    Attention and gating weights carry a leading ``dep_q`` axis; the active inner
    step selects the slice (Moshi ``multi_linear`` / ``gating[t]``). KV is supplied
    per call so the caller controls the per-frame reset and the sliding window.
    """

    def __init__(self, config: PersonaPlexDepformerConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.dim = dim
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dep_q = config.dep_q
        self.eps = config.rms_norm_eps

        # Per-step fused QKV and output projections, stored exactly as Moshi does:
        # ``in_proj_weight`` is [dep_q * 3*dim, dim]; ``out_proj`` is [dep_q*dim, dim].
        self.in_proj_weight = nn.Parameter(torch.empty(self.dep_q * 3 * dim, dim))
        self.out_proj_weight = nn.Parameter(torch.empty(self.dep_q * dim, dim))

        # fp32 RMSNorm weights (Moshi ``alpha``, shape [1, 1, dim]).
        self.norm1_alpha = nn.Parameter(torch.ones(1, 1, dim))
        self.norm2_alpha = nn.Parameter(torch.ones(1, 1, dim))

        # Per-step SiLU gating MLP. ``linear_in`` -> [2*hidden, dim]; ``linear_out`` -> [dim, hidden].
        hidden = config.intermediate_size
        self.gating_in = nn.Parameter(torch.empty(self.dep_q, 2 * hidden, dim))
        self.gating_out = nn.Parameter(torch.empty(self.dep_q, dim, hidden))

    def _in_proj(self, step: int) -> torch.Tensor:
        return self.in_proj_weight.view(self.dep_q, 3 * self.dim, self.dim)[step]

    def _out_proj(self, step: int) -> torch.Tensor:
        return self.out_proj_weight.view(self.dep_q, self.dim, self.dim)[step]

    def forward(
        self,
        x: torch.Tensor,
        step: int,
        kv: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Run inner step ``step`` for a ``[B, 1, dim]`` input, updating ``kv`` in place."""
        # --- self-attention (per-step weights, causal sliding window, no RoPE) ---
        residual = x
        h = _rms_norm_f32(x, self.norm1_alpha, self.eps).squeeze(1)  # [B, dim]
        qkv = F.linear(h, self._in_proj(step))  # [B, 3*dim]
        q, k, v = qkv.split(self.dim, dim=-1)
        b = q.shape[0]
        q = q.view(b, self.num_heads, 1, self.head_dim)
        k = k.view(b, self.num_heads, 1, self.head_dim)
        v = v.view(b, self.num_heads, 1, self.head_dim)

        if kv["k"] is None:
            k_hist, v_hist = k, v
        else:
            k_hist = torch.cat([kv["k"], k], dim=2)
            v_hist = torch.cat([kv["v"], v], dim=2)
        kv["k"], kv["v"] = k_hist, v_hist

        # Moshi forces the depformer's ``context`` to None (lm.py: kwargs_dep[
        # "context"] = None), so attention spans ALL prior inner steps -- the KV
        # capacity is ``weights_per_step`` (== dep_q), never windowed. The query is
        # the newest step and every key is causal w.r.t. it, so no mask is needed.
        attn = F.scaled_dot_product_attention(q, k_hist, v_hist)  # [B, H, 1, Dh]
        attn = attn.reshape(b, self.dim)
        out = F.linear(attn, self._out_proj(step)).unsqueeze(1)  # [B, 1, dim]
        x = residual + out

        # --- feed-forward (per-step SiLU gating) ---
        residual = x
        h = _rms_norm_f32(x, self.norm2_alpha, self.eps).squeeze(1)  # [B, dim]
        gated = F.linear(h, self.gating_in[step])  # [B, 2*hidden]
        a, b_half = gated.chunk(2, dim=-1)
        update = F.linear(F.silu(a) * b_half, self.gating_out[step]).unsqueeze(1)
        return residual + update


class PersonaPlexDepformer(nn.Module):
    """Moshi depformer as a standalone per-frame code predictor.

    Call :meth:`forward` once per temporal frame with the frame's text token and
    temporal hidden state; it runs ``dep_q`` inner AR steps (greedy by default,
    matching Moshi's ``use_sampling=False``) and returns the ``dep_q`` audio codes.
    """

    def __init__(
        self,
        config: PersonaPlexDepformerConfig,
        temporal_hidden_size: int,
        text_card: int = 32000,
    ) -> None:
        super().__init__()
        self.config = config
        self.dep_q = config.dep_q
        self.card = config.card
        dim = config.hidden_size

        # Per-codebook projection of the temporal hidden state (``depformer_multi_linear``).
        self.depformer_in = nn.ModuleList([nn.Linear(temporal_hidden_size, dim, bias=False) for _ in range(self.dep_q)])
        # Step-0 conditions on the text token; steps 1..dep_q-1 on the previous audio code.
        self.depformer_text_emb = _ScaledEmbedding(text_card + 1, dim)
        self.depformer_emb = nn.ModuleList([_ScaledEmbedding(self.card + 1, dim) for _ in range(self.dep_q - 1)])
        self.layers = nn.ModuleList([_DepformerLayer(config) for _ in range(config.num_hidden_layers)])
        # Per-codebook output heads (Moshi ``linears``).
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=False) for _ in range(self.dep_q)])

    @torch.inference_mode()
    def forward(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor | None = None,
        audio_provided: torch.Tensor | None = None,
        return_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict the ``dep_q`` audio codes for one frame.

        Args:
            text_token: ``[B]`` sampled text token for the frame.
            transformer_out: ``[B, 1, temporal_hidden]`` temporal hidden state.
            audio_tokens: optional ``[B, dep_q]`` teacher-forcing codes.
            audio_provided: optional ``[B, dep_q]`` bool mask; where True the
                corresponding ``audio_tokens`` entry overrides the sampled code as
                the next-step input (partial teacher forcing, as in Moshi).
            return_logits: also return the per-step logits ``[B, dep_q, card]``.

        Returns:
            ``[B, dep_q]`` long tensor of sampled (greedy) audio codes, plus the
            per-step logits when ``return_logits`` is set.
        """
        if transformer_out.dim() != 3 or transformer_out.shape[1] != 1:
            raise ValueError(f"transformer_out must be [B, 1, H]; got {tuple(transformer_out.shape)}")
        prev = text_token
        kv = [{"k": None, "v": None} for _ in self.layers]
        codes: list[torch.Tensor] = []
        logits_all: list[torch.Tensor] = []
        for step in range(self.dep_q):
            x = self.depformer_in[step](transformer_out)  # [B, 1, dim]
            if step == 0:
                cond = self.depformer_text_emb(prev.unsqueeze(1))
            else:
                cond = self.depformer_emb[step - 1](prev.unsqueeze(1))
            x = x + cond
            for li, layer in enumerate(self.layers):
                x = layer(x, step, kv[li])
            logits = self.linears[step](x).squeeze(1)  # [B, card]
            sampled = logits.float().argmax(dim=-1)  # greedy [B]
            codes.append(sampled)
            if return_logits:
                logits_all.append(logits.float())
            if audio_provided is not None and audio_tokens is not None:
                prev = torch.where(audio_provided[:, step], audio_tokens[:, step], sampled)
            else:
                prev = sampled
        stacked = torch.stack(codes, dim=1)  # [B, dep_q]
        if return_logits:
            return stacked, torch.stack(logits_all, dim=1)  # [B, dep_q, card]
        return stacked

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]] | Mapping[str, torch.Tensor],
    ) -> set[str]:
        """Load Moshi depformer weights, expanding ``dep_q`` 8 -> 16 if needed.

        Accepts the raw checkpoint names (``depformer.*``, ``depformer_in.N``,
        ``depformer_emb.N``, ``depformer_text_emb``, ``linears.N``). Per-step
        modules present only for steps 0..7 are copied to 8..15, and the fused
        attention projections are tiled, exactly as Moshi's ``get_moshi_lm`` does.
        """
        src = dict(weights.items() if isinstance(weights, Mapping) else weights)
        src = {k: v for k, v in src.items() if self._is_depformer_key(k)}
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        for tgt, build in self._weight_plan().items():
            tensor = build(src)
            if tensor is None:
                continue
            param = params[tgt]
            if tensor.shape != param.shape:
                raise ValueError(
                    f"shape mismatch for {tgt}: checkpoint {tuple(tensor.shape)} vs param {tuple(param.shape)}"
                )
            param.data.copy_(tensor.to(param.dtype))
            loaded.add(tgt)
        return loaded

    @staticmethod
    def _is_depformer_key(name: str) -> bool:
        return name.startswith(("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb", "linears."))

    def _weight_plan(self) -> dict[str, _Builder]:
        """Map each of this module's params to a builder over the source state dict."""
        dim = self.config.hidden_size

        def step_or_copy(prefix: str, step: int, suffix: str = "weight") -> _Builder:
            # cb 8..15 fall back to cb (step-8) when absent (Moshi copy_missing_weights).
            def build(src: dict[str, torch.Tensor]) -> torch.Tensor | None:
                key = f"{prefix}.{step}.{suffix}"
                if key in src:
                    return src[key]
                if step >= 8:
                    return src.get(f"{prefix}.{step - 8}.{suffix}")
                return None

            return build

        def expand_attn(layer: int, kind: str, per_step_rows: int) -> _Builder:
            # ``in_proj_weight`` [steps*per_step_rows, dim] / ``out_proj.weight`` likewise;
            # tile an 8-step checkpoint block up to dep_q (Moshi concat-copy expansion).
            def build(src: dict[str, torch.Tensor]) -> torch.Tensor | None:
                key = f"depformer.layers.{layer}.self_attn.{kind}"
                t = src.get(key)
                if t is None:
                    return None
                steps = t.shape[0] // per_step_rows
                if steps < self.dep_q:
                    reps = (self.dep_q + steps - 1) // steps
                    t = t.repeat(reps, 1)[: self.dep_q * per_step_rows]
                return t

            return build

        plan: dict[str, _Builder] = {}
        # depformer_in / linears / depformer_emb / text_emb
        for step in range(self.dep_q):
            plan[f"depformer_in.{step}.weight"] = step_or_copy("depformer_in", step)
            plan[f"linears.{step}.weight"] = step_or_copy("linears", step)
        plan["depformer_text_emb.weight"] = lambda src: src.get("depformer_text_emb.weight")
        for step in range(self.dep_q - 1):
            plan[f"depformer_emb.{step}.weight"] = step_or_copy("depformer_emb", step)

        # per-layer attention + norms + gating
        for layer in range(self.config.num_hidden_layers):
            lp = f"depformer.layers.{layer}"
            plan[f"layers.{layer}.in_proj_weight"] = expand_attn(layer, "in_proj_weight", 3 * dim)
            plan[f"layers.{layer}.out_proj_weight"] = expand_attn(layer, "out_proj.weight", dim)
            plan[f"layers.{layer}.norm1_alpha"] = self._reshape_alpha(f"{lp}.norm1.alpha")
            plan[f"layers.{layer}.norm2_alpha"] = self._reshape_alpha(f"{lp}.norm2.alpha")
            plan[f"layers.{layer}.gating_in"] = self._stack_gating(lp, "linear_in")
            plan[f"layers.{layer}.gating_out"] = self._stack_gating(lp, "linear_out")
        return plan

    @staticmethod
    def _reshape_alpha(key: str) -> _Builder:
        def build(src: dict[str, torch.Tensor]) -> torch.Tensor | None:
            t = src.get(key)
            return None if t is None else t.reshape(1, 1, -1)

        return build

    def _stack_gating(self, layer_prefix: str, which: str) -> _Builder:
        """Stack the dep_q per-step gating weights into one [dep_q, ...] tensor.

        Steps 8..15 fall back to step (n-8) when the checkpoint only ships 8
        (Moshi ``copy_missing_weights``).
        """

        def build(src: dict[str, torch.Tensor]) -> torch.Tensor | None:
            slabs: list[torch.Tensor] = []
            for step in range(self.dep_q):
                key = f"{layer_prefix}.gating.{step}.{which}.weight"
                if key not in src and step >= 8:
                    key = f"{layer_prefix}.gating.{step - 8}.{which}.weight"
                t = src.get(key)
                if t is None:
                    return None
                slabs.append(t)
            return torch.stack(slabs, dim=0)

        return build
