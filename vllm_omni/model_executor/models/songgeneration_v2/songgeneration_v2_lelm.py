# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SongGeneration v2 LeLM Stage 0 — per-step AR under vLLM LLM_AR runner.

Architecture:
  - self.model: vLLM LlamaModel (36L) — cond path, PagedAttention
  - self.null_model: HF LlamaModel (36L) — null path for CFG, explicit KV cache
  - self.lm_head: ParallelLMHead — stream-0 logits
  - CFG combine in compute_logits: guided = uncond + cfg_coef * (cond - uncond)
  - transformer2 + linears: side computation for streams 1+2
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.songgeneration_v2.ar_state import ArState
    from vllm_omni.model_executor.models.songgeneration_v2.configuration_songgeneration_v2 import (
        SongGenerationV2LeLMConfig,
    )

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .runtime_assets import resolve_songgeneration_runtime_assets

logger = init_logger(__name__)


def _resolve_upstream_repo_for_assets(model_path: str | None) -> str | None:
    """Resolve repo path for tokenizer/runtime assets.

    HF deployments may pass the LeLM checkpoint snapshot as ``model_path`` and
    keep ckpt/third_party assets in the companion runtime repo.
    """
    return resolve_songgeneration_runtime_assets(model_path)


class SongGenV2LeLMNullModel(nn.Module):
    """HF LlamaModel for the CFG null path with explicit past_key_values.

    Uses standard HuggingFace transformers attention (not PagedAttention) so
    that KV can be cached and reused across AR decode steps without requiring
    vLLM scheduler integration. Shares weights with the cond path's LlamaModel
    via load_weights (same checkpoint tensors, different runtime KV state).
    """

    def __init__(self, config: SongGenerationV2LeLMConfig):
        super().__init__()
        from transformers import LlamaConfig as HFLlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel as HFLlamaModel

        hf_cfg = HFLlamaConfig(
            hidden_size=int(config.dim),
            intermediate_size=int(config.intermediate_size),
            num_attention_heads=int(config.num_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            num_hidden_layers=int(config.num_layers),
            max_position_embeddings=int(config.max_position_embeddings),
            rope_theta=float(config.rope_theta),
            vocab_size=int(config.code_size) + 1,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
        )
        self.model = HFLlamaModel(hf_cfg)
        self._past_key_values: tuple | None = None

    def reset_cache(self) -> None:
        self._past_key_values = None

    @torch.no_grad()
    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Run null-condition forward with HF-style KV cache.

        Args:
            inputs_embeds: [S, dim] token embeddings for this step.

        Returns:
            Hidden states [S, dim] — caller applies lm_head.
        """
        # HF expects [batch, seq, dim]
        embeds_3d = inputs_embeds.unsqueeze(0) if inputs_embeds.ndim == 2 else inputs_embeds
        # Ensure dtype matches model weights (conditioner may output fp32)
        model_dtype = next(self.model.parameters()).dtype
        if embeds_3d.dtype != model_dtype:
            embeds_3d = embeds_3d.to(model_dtype)

        outputs = self.model(
            inputs_embeds=embeds_3d,
            past_key_values=self._past_key_values,
            use_cache=True,
            return_dict=True,
        )

        self._past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state  # [1, S, dim]
        return hidden.squeeze(0)  # [S, dim]


class SongGenerationV2LeLMForConditionalGeneration(nn.Module):
    """SongGeneration v2 LeLM Stage 0 — per-step AR with PagedAttention + CFG.

    Uses VoxCPM2 pattern:
      - Cond path: PagedAttention via self.model (LlamaModel)
      - Null path: self-managed KV via self.null_model
      - CFG combine in compute_logits
    """

    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True
    requires_raw_input_tokens = True
    enable_update_additional_information = True
    prefer_model_sampler = True
    engine_output_type = "codes"

    # Maps upstream checkpoint keys (after stripping "audiolm." prefix) to
    # this module's parameter names. Same pattern as Qwen3-TTS talker.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Main 36L transformer → self.model (vLLM LlamaModel)
            "transformer.model.layers.": "model.layers.",
            "transformer.model.norm.": "model.norm.",
            "transformer.model.embed_tokens.": "model.embed_tokens.",
            "transformer.lm_head.": "lm_head.",
            # Null model shares same weights (loaded separately below)
            # Stream-0 embedding, streams 1+2 heads, MLP — direct match
            "emb.": "emb.",
            "layer2_emb.": "layer2_emb.",
            "linears.": "linears.",
            "mlp.": "mlp.",
            "out_norm.": "out_norm.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        import copy

        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config

        if hasattr(config, "lelm_config"):
            config = config.lelm_config
        self.config = config
        self.model_path = vllm_config.model_config.model

        # Tensor parallelism is unsupported: stream-0 logits are computed by
        # hand from self.lm_head.weight (a local shard under TP>1) and the
        # null/transformer2 paths are plain replicated HF LlamaModels.
        tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
        assert tp_size == 1, (
            f"SongGenerationV2LeLM only supports tensor_parallel_size == 1; got {tp_size}. "
            "The CFG/null-path and hand-computed lm_head logits assume an unsharded vocab."
        )

        # Single-request only: the CFG null path keeps one past_key_values
        # swapped per request, and the force-EOS / repetition-penalty /
        # audio_codes paths use one request's state for the whole batch.
        max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        if max_num_seqs > 1:
            raise ValueError(
                f"SongGenerationV2LeLM currently requires max_num_seqs == 1; got {max_num_seqs}. "
                "Multi-request batching needs per-row keyed state in compute_logits / "
                "make_omni_output and a per-request null-model KV pool (see README "
                "'Known constraints')."
            )

        self.code_depth = int(config.code_depth)
        self.code_size = int(config.code_size) + 1
        self.input_emb_dim = int(config.code_size) + 2
        self.dim = int(config.dim)
        self.cfg_coef = float((config.reference_generation_params or {}).get("cfg_coef", 1.5))
        self.top_k_per_stream: list[int] = list(getattr(config, "top_k_per_stream", [5000, 1, 1]))

        # Substitute the LeLM sub-config so LlamaModel sees the right
        # hidden_size / num_hidden_layers / vocab_size. The backbone uses silu
        # (upstream builds its Llama with LlamaConfig's default and vLLM's
        # LlamaMLP only supports silu), so hidden_act is forced below.
        # config.activation ("gelu") is the MLP fuser's (self.mlp uses nn.GELU);
        # guard it so a future checkpoint that changes the fuser fails loudly.
        fuser_act = str(getattr(config, "activation", "gelu")).lower()
        if fuser_act != "gelu":
            raise ValueError(
                "SongGenerationV2LeLM hardcodes a GELU MLP fuser (self.mlp) and a SiLU "
                f"Llama backbone, but upstream config.activation={fuser_act!r}. Review the "
                "fuser/backbone activation wiring before running this checkpoint."
            )
        llama_config = copy.copy(config)
        llama_config.hidden_act = "silu"
        lelm_vllm_config = copy.copy(vllm_config)
        lelm_model_config = copy.copy(vllm_config.model_config)
        lelm_model_config.hf_config = llama_config
        lelm_vllm_config.model_config = lelm_model_config

        # --- Main transformer (cond path) — PagedAttention ---
        self.model = LlamaModel(
            vllm_config=lelm_vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # --- LM head for stream 0 ---
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.code_size,
                self.dim,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            from vllm.model_executor.models.utils import PPMissingLayer

            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(self.code_size)

        # --- Null model (CFG uncond path) — HF LlamaModel with past_key_values ---
        self.null_model = SongGenV2LeLMNullModel(config)

        # --- Stream-0 embedding ---
        self.emb = nn.ModuleList([nn.Embedding(self.input_emb_dim, self.dim)])

        # --- Streams 1+2 components ---
        self.layer2_emb = nn.ModuleList([nn.Embedding(self.input_emb_dim, self.dim) for _ in range(self.code_depth)])
        if self.code_depth > 1:
            self.linears = nn.ModuleList(
                [nn.Linear(self.dim, self.code_size, bias=False) for _ in range(self.code_depth - 1)]
            )

        # MLP fuser for transformer2 input: concat(stream12_emb, hidden1) → dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

        # --- Transformer2 (12L sub-LM for streams 1+2) — HF LlamaModel ---
        if self.code_depth > 1:
            from transformers import LlamaConfig as HFLlamaConfig
            from transformers.models.llama.modeling_llama import (
                LlamaModel as HFLlamaModel,
            )

            t2_cfg = HFLlamaConfig(
                hidden_size=self.dim,
                intermediate_size=int(config.intermediate_size),
                num_attention_heads=int(config.num_heads),
                num_key_value_heads=int(config.num_key_value_heads),
                num_hidden_layers=int(config.num_layers_sub),
                max_position_embeddings=int(config.max_position_embeddings_sub),
                rope_theta=float(config.rope_theta_sub),
                vocab_size=self.code_size,
                rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-5)),
                tie_word_embeddings=False,
            )
            self.transformer2 = HFLlamaModel(t2_cfg)
        else:
            self.transformer2 = None

        # out_norm: loaded from checkpoint but not used at inference
        self.out_norm = nn.LayerNorm(self.dim) if getattr(config, "norm_first", False) else None

        # --- Condition provider + fuser (for CFG) ---
        from .condition_fuser import ConditionFuser

        fuser_cfg = config.fuser or {"sum": [], "prepend": []}
        self.fuser = ConditionFuser(
            fuse2cond={
                "sum": list(fuser_cfg.get("sum", []) or []),
                "prepend": list(fuser_cfg.get("prepend", []) or []),
            }
        )

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Tell vLLM's DefaultModelLoader where to find the weight file
        self.allow_patterns_overrides = ["songgeneration_v2_large/model.pt"]

        # Delayed pattern config
        cp = config.codebooks_pattern or {}
        self._delays: list[int] = list((cp.get("delay") or {}).get("delays") or [0, 0, 0])

        # Keep hidden_states.last on GPU to avoid CPU round-trip
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("hidden_states", "last"),
        }

        # Per-request null KV caches: {request_id: past_key_values}
        self._null_kv_states: dict[str, tuple | None] = {}
        # Per-request transformer2 KV caches (cond + null for CFG):
        self._t2_kv_states: dict[str, tuple | None] = {}
        self._t2_null_kv_states: dict[str, tuple | None] = {}
        # Pending transformer2 prefill inputs (per-request, consumed in postprocess)
        self._pending_t2_prefill: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # Per-request null path hidden1 cache (for transformer2 CFG)
        self._null_hidden1_cache: dict[str, torch.Tensor] = {}
        # Full null hidden sequence (only during prefill, for t2 null prefill)
        self._null_hidden1_full: dict[str, torch.Tensor] = {}
        # Per-request AR state machines: {request_id: ArState}
        self._ar_states: dict[str, ArState] = {}
        # Requests that have finished generation — force EOS on next compute_logits
        self._force_eos_reqs: set[str] = set()
        # Queue of (request_id, null_embeds) for the current batch,
        # filled by preprocess and consumed by forward.
        self._null_embeds_queue: deque[tuple[str, torch.Tensor]] = deque()
        # Null hidden states from the current forward, consumed in compute_logits
        self._null_hidden_states: torch.Tensor | None = None

        # No fused-kernel patching: forcing RMSNorm/activation to native paths
        # was not needed (the claimed NaN/incorrect-output does not reproduce;
        # it only shifts the AR trajectory, like the attention backend choice),
        # so this model uses the same kernels as every other model.

    @property
    def special_token_id(self) -> int:
        return self.code_size

    @property
    def eos_token_id(self) -> int:
        return self.code_size - 1

    # ---- vLLM model interface ----

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Return stream-0 embeddings for input token IDs."""
        return self.emb[0](input_ids.clamp(0, self.input_emb_dim - 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Run cond path (PagedAttention) + null path (side computation).

        Cond hidden states are returned normally for compute_logits.
        Null hidden states are stored in self._null_hidden_states.
        """
        # --- Cond path (PagedAttention, standard vLLM forward) ---
        # Clone inputs_embeds to prevent aliasing issues with runner buffer
        model_inputs = inputs_embeds.clone() if inputs_embeds is not None else None

        hidden = self.model(input_ids, positions, intermediate_tensors, model_inputs)

        if isinstance(hidden, IntermediateTensors):
            return hidden

        # --- Null path (HF LlamaModel, per-request KV cache) ---
        if self._null_embeds_queue:
            null_hidden_parts: list[torch.Tensor] = []
            while self._null_embeds_queue:
                req_id, embeds = self._null_embeds_queue.popleft()
                # Restore this request's KV state
                self.null_model._past_key_values = self._null_kv_states.get(req_id)
                # Forward through null model
                h = self.null_model(embeds)
                # Save updated KV state
                self._null_kv_states[req_id] = self.null_model._past_key_values
                # Cache null hidden1 last token for transformer2 CFG (decode steps)
                self._null_hidden1_cache[req_id] = h[-1].detach()
                # If this is a prefill (multi-token), cache full sequence for t2 prefill
                if h.shape[0] > 1:
                    self._null_hidden1_full[req_id] = h.detach()
                # Keep last token's hidden (aligned with cond path output)
                null_hidden_parts.append(h[-1:])

            self._null_hidden_states = torch.cat(null_hidden_parts, dim=0)
        else:
            self._null_hidden_states = None

        return hidden

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Compute CFG-guided stream-0 logits.

        CFG formula: guided = uncond + cfg_coef * (cond - uncond)
        - cond logits: from hidden_states (PagedAttention cond path)
        - uncond logits: from self._null_hidden_states (self-managed null path)
        """
        # Force EOS for requests that have finished internal generation
        if self._force_eos_reqs:
            self._force_eos_reqs.clear()
            self._null_hidden_states = None
            eos_logits = torch.full(
                (hidden_states.shape[0], self.code_size),
                -float("inf"),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            eos_logits[:, self.eos_token_id] = 0.0
            return eos_logits

        # Hand-compute logits from the lm_head weight (CFG needs raw cond/null
        # logits before combining). Safe because TP == 1 is asserted in __init__.
        lm_weight = self.lm_head.weight  # [vocab, dim]
        cond_logits = torch.nn.functional.linear(hidden_states, lm_weight)

        null_hidden = self._null_hidden_states
        if null_hidden is not None and self.cfg_coef != 1.0:
            if null_hidden.shape[0] == cond_logits.shape[0]:
                uncond_logits = torch.nn.functional.linear(null_hidden, lm_weight)
                guided_logits = uncond_logits + self.cfg_coef * (cond_logits - uncond_logits)
            elif null_hidden.shape[0] == 1 and cond_logits.shape[0] > 1:
                # Prefill case: null model kept only the last hidden state,
                # while cond path has all positions. Apply CFG to the last
                # position only (which is the one vLLM samples from).
                uncond_logits_last = torch.nn.functional.linear(null_hidden, lm_weight)
                cond_logits_last = cond_logits[-1:]
                guided_last = uncond_logits_last + self.cfg_coef * (cond_logits_last - uncond_logits_last)
                guided_logits = cond_logits.clone()
                guided_logits[-1:] = guided_last
            elif null_hidden.shape[0] > cond_logits.shape[0]:
                null_hidden = null_hidden[-cond_logits.shape[0] :]
                uncond_logits = torch.nn.functional.linear(null_hidden, lm_weight)
                guided_logits = uncond_logits + self.cfg_coef * (cond_logits - uncond_logits)
            else:
                logger.warning(
                    "Null hidden (%d) vs cond (%d) size mismatch, skipping CFG",
                    null_hidden.shape[0],
                    cond_logits.shape[0],
                )
                guided_logits = cond_logits
        else:
            guided_logits = cond_logits

        # Repetition penalty: exact match to official lm_levo.py:520-522
        #   q_count = torch.bincount(torch.unique(sampled_token_pool[q]))
        #   logits[:, q, :tmp] /= (1.1 ** q_count[:tmp])
        # Official generate.py uses record_window=50 (overrides model default of 150).
        if self._ar_states:
            state = next(iter(self._ar_states.values()))
            record_window = 50
            s0_step = state.step
            if s0_step > 0:
                start = max(0, s0_step - record_window)
                recent_s0 = state.gen_codes[0, start:s0_step]
                recent_s0 = recent_s0[recent_s0 >= 0]
                if recent_s0.numel() > 0:
                    unique_toks = torch.unique(recent_s0)
                    q_count = torch.bincount(unique_toks)
                    tmp = min(q_count.shape[-1], self.code_size - 1)
                    guided_logits[:, :tmp] /= (1.1 ** q_count[:tmp].to(guided_logits.dtype)).unsqueeze(0)

        self._null_hidden_states = None

        return guided_logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
    ) -> Any:
        """Sample stream-0 token from guided logits."""
        from vllm.v1.sample.sampler import Sampler

        if not hasattr(self, "_sampler"):
            self._sampler = Sampler()
        return self._sampler(logits, sampling_metadata)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare per-request conditioning, null-path embeddings, and AR state.

        Contract (aligned with Qwen3-TTS / Fish Speech):
          - NEVER trust runner's input_embeds (may be uninitialized buffer)
          - ALWAYS return non-zero embeddings (even for warmup/dummy)
          - input_ids_out must be in-vocab [0, code_size+1]
        """
        ai = info_dict.get("additional_information")
        if isinstance(ai, dict):
            merged: dict[str, Any] = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in ai.items():
                merged.setdefault(k, v)
            info_dict = merged

        # Detect dummy/warmup: no lyric/lyrics AND (_is_dummy or missing request context)
        is_dummy = info_dict.get("_is_dummy", False)
        has_content = bool(info_dict.get("lyric") or info_dict.get("lyrics") or info_dict.get("descriptions"))
        if is_dummy or (not has_content and not info_dict.get("request_id")):
            # Warmup/profile run: return valid non-zero embeddings, no state mutation
            input_ids_out = input_ids.clamp(0, self.code_size - 1)
            embeds = self.embed_input_ids(input_ids_out)
            return input_ids_out, embeds, {}

        info_update: dict[str, Any] = {
            k: info_dict[k]
            for k in ("lyric", "lyrics", "descriptions", "gen_type", "prompt_audio_codes", "seed", "max_gen_len")
            if k in info_dict
        }

        req_id = info_dict.get("request_id", "default")
        is_prefill_raw = info_dict.get("_omni_is_prefill")
        if isinstance(is_prefill_raw, bool):
            is_prefill = is_prefill_raw
        else:
            try:
                is_prefill = int(info_dict["_omni_num_computed_tokens"]) < int(info_dict["_omni_prompt_len"])
            except Exception:
                is_prefill = input_ids.shape[0] > 1

        if is_prefill:
            # Always compute our own embeddings — never use runner buffer
            own_embeds = self.embed_input_ids(input_ids)
            cond_embeds, null_embeds = self._prepare_cfg_embeddings(own_embeds, info_dict, req_id=req_id)
            # Reset KV caches for this request (new generation)
            self._null_kv_states[req_id] = None
            self._t2_kv_states[req_id] = None
            self._t2_null_kv_states[req_id] = None
            self._null_hidden1_cache.pop(req_id, None)
            # Queue null embeds for null_model side computation in forward()
            self._null_embeds_queue.append((req_id, null_embeds))

            # Initialize the delayed-pattern AR state machine
            from .ar_state import ArState

            # Defense-in-depth for the single-request constraint (also checked
            # at engine init): no other request may be in flight.
            other_active = [r for r in self._ar_states if r != req_id]
            assert not other_active, (
                "SongGenerationV2LeLM only supports max_num_seqs == 1; "
                f"got concurrent requests {other_active + [req_id]}. Batched "
                "decode is unsupported (CFG/repetition-penalty/EOS/audio_codes "
                "paths assume a single in-flight request)."
            )

            max_dur = int(self.config.max_position_embeddings)
            max_gen_len = int(info_dict.get("max_gen_len", max_dur * 0.7))
            self._ar_states[req_id] = ArState.create(
                code_depth=self.code_depth,
                code_size=self.code_size,
                max_gen_len=max_gen_len,
                delays=self._delays,
                device=input_ids.device,
            )
            info_update["_ar_state_req_id"] = req_id
            span_len = input_ids.shape[0]
            cond_len = cond_embeds.shape[0]
            if cond_len != span_len:
                # On a mismatch only min(span_len, cond_len) condition tokens
                # reach PagedAttention, producing undefined output; not
                # recoverable, so raise rather than emit garbage audio.
                raise ValueError(
                    f"prompt_token_ids length ({span_len}) != condition_embeds length "
                    f"({cond_len}); only {min(span_len, cond_len)} of {cond_len} condition "
                    f"tokens would enter PagedAttention. Set prompt_token_ids = [0] * {cond_len} "
                    "(use compute_condition_prompt_len to size it)."
                )
            # Return in-vocab input_ids (bookkeeping only; embeddings carry semantics)
            input_ids_out = input_ids.clamp(0, self.code_size - 1)
            return input_ids_out, cond_embeds, info_update
        else:
            # Decode step: always compute own embedding
            input_embeds = self.embed_input_ids(input_ids)

            # Advance AR state with previously sampled token
            state = self._ar_states.get(req_id)
            if state is not None and not state.finished:
                # input_ids[0] is the token vLLM fed back (= previously sampled stream-0 token)
                sampled_tok = int(input_ids[0].item())
                state.record_stream0_token(sampled_tok)
                state.fill_inactive_streams()

                # Run transformer2 EVERY step (matching official: transformer2 runs
                # from step 0, accumulating KV cache even when streams 1+2 are
                # still inactive and using special_token_id as input).
                hs = info_dict.get("hidden_states", {})
                hidden1_last = hs.get("last") if isinstance(hs, dict) else None
                if isinstance(hidden1_last, torch.Tensor):
                    hidden1_last = hidden1_last.to(
                        device=input_embeds.device,
                        dtype=input_embeds.dtype,
                    )
                    stream12_tokens = self._compute_stream12(req_id, state, hidden1_last)
                    if stream12_tokens is not None and state.should_compute_stream12():
                        state.record_stream12_tokens(stream12_tokens)

                state.advance()

                if state.finished:
                    info_update["audio_codes"] = state.get_output_codes()
                    self._force_eos_reqs.add(req_id)
                    self._null_kv_states.pop(req_id, None)
                    self._t2_kv_states.pop(req_id, None)
                    self._t2_null_kv_states.pop(req_id, None)
                    self._null_hidden1_cache.pop(req_id, None)
                    self._ar_states.pop(req_id, None)

            # Queue null embeds for null_model side computation
            self._null_embeds_queue.append((req_id, input_embeds.clone()))
            info_update["_ar_state_req_id"] = req_id
            return input_ids, input_embeds, info_update

    def _prepare_cfg_embeddings(
        self,
        token_embeds: torch.Tensor,
        info_dict: dict[str, Any],
        req_id: str = "default",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build condition-fused embeddings for both cond and null paths.

        Returns (cond_embeds, null_embeds) where each may have prepended
        condition tokens (sequence dim > token_embeds.shape[0]).
        """
        from .prompt_builder import build_condition_tensors

        if not hasattr(self, "condition_provider"):
            raise RuntimeError(
                "condition_provider was not loaded; load_weights() must run before inference. "
                "Check that the model checkpoint contains audiolm.condition_provider.* keys."
            )

        lyrics = [info_dict.get("lyric") or info_dict.get("lyrics")]
        descriptions = [info_dict.get("descriptions")]
        prompt_codes = info_dict.get("prompt_audio_codes")
        gen_type = info_dict.get("gen_type", "mixed")

        # Build CFG-paired conditions: first B rows = real, last B rows = null
        cond_tensors, B = build_condition_tensors(
            self.condition_provider,
            lyrics=lyrics,
            descriptions=descriptions,
            prompt_audio_codes=prompt_codes,
            gen_type=gen_type,
            cfg_pair=True,
        )

        # Split condition_tensors into cond and null halves
        cond_half: dict = {}
        null_half: dict = {}
        for name, (e1, e2, mask) in cond_tensors.items():
            cond_half[name] = (e1[:B], e2[:B], mask[:B])
            null_half[name] = (e1[B:], e2[B:], mask[B:])

        # The fuser prepends condition tokens to the input. To match the
        # official AR loop's first step, we feed only ONE token embedding
        # (the "start" token = special_token_id) to the fuser. The fuser
        # then prepends all condition tokens in front, producing a sequence
        # of length (cond_prefix_len + 1).
        #
        # The caller must ensure prompt_token_ids has length == (cond_prefix_len + 1)
        # so the returned embeddings exactly fill the vLLM-scheduled positions.
        device = token_embeds.device

        # Use only the LAST token embedding as the actual "start token" input.
        # (In the official loop, the start token is special_token_id embedded
        # through emb[0]; all preceding prompt_token_ids are just placeholders.)
        start_emb = token_embeds[-1:].unsqueeze(0)  # [1, 1, dim]

        # Build stream12 embedding for the single start token position
        special_ids = torch.full(
            (1, 1),
            min(self.special_token_id, self.input_emb_dim - 1),
            dtype=torch.long,
            device=device,
        )
        input2_emb = sum(self.layer2_emb[k](special_ids) for k in range(1, self.code_depth))  # [1, 1, dim]

        # Fuse conditions (prepend on first step) for BOTH paths
        cond_fused1, cond_fused2 = self.fuser(start_emb, input2_emb, cond_half, first_step=True)
        null_fused1, null_fused2 = self.fuser(start_emb, input2_emb, null_half, first_step=True)

        # Store fused2 for transformer2 prefill (consumed in postprocess, per-request)
        self._pending_t2_prefill[req_id] = (cond_fused2, null_fused2)

        # Remove batch dim back to [S+P, dim]
        return cond_fused1.squeeze(0), null_fused1.squeeze(0)

    @torch.no_grad()
    def _compute_stream12(
        self,
        req_id: str,
        state: ArState,
        hidden1_last: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run transformer2 to predict streams 1+2 tokens at current step.

        Architecture (matches original's forward exactly):
          1. Embed streams 1+2 previous tokens → sum
          2. Concat with hidden1 (main transformer cond output) → MLP → [1, 1, dim]
          3. Feed through transformer2 (cond path) with KV cache
          4. If CFG active: also run null path transformer2 (with null hidden1)
          5. Apply CFG: guided_logits = uncond + cfg_coef * (cond - uncond)
          6. linears[k] → logits → greedy sample (top_k=1 in original)

        Returns [K-1] tensor of token IDs, or None if transformer2 unavailable.
        """
        if self.transformer2 is None:
            return None

        device = hidden1_last.device
        K = self.code_depth

        # 1. Embed previous streams 1+2 tokens (sum of embeddings)
        stream12_emb = torch.zeros(1, 1, self.dim, device=device, dtype=hidden1_last.dtype)
        for k in range(1, K):
            prev_frame = state.stream_frame(k) - 1
            if prev_frame >= 0:
                prev_tok = state.gen_codes[k, prev_frame].item()
            else:
                prev_tok = state.special_token_id
            tok_id = min(prev_tok, self.input_emb_dim - 1)
            tok_tensor = torch.tensor([[tok_id]], dtype=torch.long, device=device)
            stream12_emb = stream12_emb + self.layer2_emb[k](tok_tensor)

        # 2. Cond path: concat with cond hidden1, project through MLP
        hidden1_3d = hidden1_last.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        fused_cond = torch.cat([stream12_emb, hidden1_3d], dim=-1)  # [1, 1, 2*dim]
        t2_input_cond = self.mlp(fused_cond)  # [1, 1, dim]

        # 3. Forward through transformer2 (cond path)
        past_kv = self._t2_kv_states.get(req_id)
        outputs_cond = self.transformer2(
            inputs_embeds=t2_input_cond,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        self._t2_kv_states[req_id] = outputs_cond.past_key_values
        hidden2_cond = outputs_cond.last_hidden_state  # [1, 1, dim]

        # 4. Null path for CFG (if cfg_coef != 1.0)
        if self.cfg_coef != 1.0:
            null_h1 = self._get_null_hidden1_last(req_id, device, hidden1_last.dtype)
            fused_null = torch.cat([stream12_emb, null_h1.unsqueeze(0).unsqueeze(0)], dim=-1)
            t2_input_null = self.mlp(fused_null)

            past_kv_null = self._t2_null_kv_states.get(req_id)
            outputs_null = self.transformer2(
                inputs_embeds=t2_input_null,
                past_key_values=past_kv_null,
                use_cache=True,
                return_dict=True,
            )
            self._t2_null_kv_states[req_id] = outputs_null.past_key_values
            hidden2_null = outputs_null.last_hidden_state  # [1, 1, dim]
        else:
            hidden2_null = None

        # 5. Apply linears + CFG combine + repetition penalty + greedy decode
        # Repetition penalty: exact match to official lm_levo.py:520-522
        #   q_count = torch.bincount(torch.unique(sampled_token_pool[q]))
        #   logits[:, q, :tmp] /= (1.1 ** q_count[:tmp])
        # bincount(unique(x)) is binary (0 or 1) per token id.
        # Official generate.py uses record_window=50.
        record_window = 50
        tokens = []
        for k in range(K - 1):
            cond_logits_k = self.linears[k](hidden2_cond[:, -1, :])  # [1, code_size]
            if hidden2_null is not None:
                uncond_logits_k = self.linears[k](hidden2_null[:, -1, :])
                guided_logits_k = uncond_logits_k + self.cfg_coef * (cond_logits_k - uncond_logits_k)
            else:
                guided_logits_k = cond_logits_k

            q = k + 1  # stream index (streams 1, 2)
            frame = state.stream_frame(q)
            if frame > 0:
                start = max(0, frame - record_window)
                recent = state.gen_codes[q, start:frame]
                recent = recent[recent >= 0]
                if recent.numel() > 0:
                    unique_toks = torch.unique(recent)
                    q_count = torch.bincount(unique_toks)
                    tmp = min(q_count.shape[-1], self.code_size - 1)
                    guided_logits_k[0, :tmp] /= 1.1 ** q_count[:tmp].to(guided_logits_k.dtype)

            tok = torch.argmax(guided_logits_k, dim=-1).item()
            tokens.append(tok)

        return torch.tensor(tokens, dtype=torch.long, device=device)

    def _get_null_hidden1_last(self, req_id: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get the null path's last hidden state for transformer2 CFG.

        During decode, the null model's hidden state from the previous step
        is stored after each forward pass in the null model.
        """
        # The null hidden from the last forward is stored on each step.
        # We cache it in _null_hidden1_last per request.
        h = self._null_hidden1_cache.get(req_id)
        if h is not None:
            return h.to(device=device, dtype=dtype)
        return torch.zeros(self.dim, device=device, dtype=dtype)

    def postprocess(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Save last hidden state and run transformer2 prefill if pending.

        The actual AR state advancement happens in preprocess of the next step,
        where the sampled token is available from the request's output history.
        """
        if hidden_states.numel() == 0:
            return {}

        last = hidden_states[-1, :].detach()
        result: dict[str, Any] = {"hidden_states": {"last": last}}

        # If we have a pending transformer2 prefill (from _prepare_cfg_embeddings),
        # run it now that we have hidden1 from main transformer.
        req_id = kwargs.get("_ar_state_req_id")
        if req_id and self.transformer2 is not None:
            pending = self._pending_t2_prefill.pop(req_id, None)
            if pending is not None:
                t2_cond, t2_null = pending
                self._run_t2_prefill(req_id, hidden_states, t2_cond, t2_null)

        return result

    @torch.no_grad()
    def _run_t2_prefill(
        self,
        req_id: str,
        cond_hidden1: torch.Tensor,
        cond_fused2: torch.Tensor,
        null_fused2: torch.Tensor | None,
    ) -> None:
        """Run transformer2 prefill to establish its initial KV cache.

        In the original model, transformer2 sees:
          concat(fused_input2, hidden1) → MLP → transformer2
        on the first (prefill) step, where fused_input2 includes condition
        embeddings prepended to stream12 embeddings.

        We run this for both cond and null paths to enable CFG on streams 1+2.

        IMPORTANT: Only prefill the condition prefix (positions 0..P-1),
        NOT the start token at position P. The start token position will be
        handled by the first _compute_stream12 call at state.step=0, which
        ensures transformer2's position IDs align with the official model.
        Official: position P = step 0 output, position P+1 = step 1 output.
        If we included position P here, _compute_stream12 would start at P+1,
        causing an off-by-one in RoPE position encoding.
        """
        # cond_hidden1: [S+P, dim] from vLLM main transformer (cond path)
        # cond_fused2: [1, S+P, dim] condition-prepended stream12 embeddings (cond)
        # null_fused2: [1, S+P, dim] null-condition-prepended stream12 embeddings
        h1 = cond_hidden1.unsqueeze(0)  # [1, S+P, dim]
        cond_fused2 = cond_fused2.to(device=h1.device, dtype=h1.dtype)

        # Match sequence lengths
        seq_len = min(h1.shape[1], cond_fused2.shape[1])
        h1 = h1[:, :seq_len, :]
        cond_fused2 = cond_fused2[:, :seq_len, :]

        # Exclude the last position (start token) — it will be processed
        # by _compute_stream12 at state.step=0, matching the official's
        # step 0 at position P.
        if seq_len > 1:
            h1 = h1[:, :-1, :]
            cond_fused2 = cond_fused2[:, :-1, :]

        # --- Cond path: concat(cond_fused_input2, cond_hidden1) + MLP ---
        cat_cond = torch.cat([cond_fused2, h1], dim=-1)  # [1, P, 2*dim]
        t2_input_cond = self.mlp(cat_cond)

        outputs_cond = self.transformer2(
            inputs_embeds=t2_input_cond,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        self._t2_kv_states[req_id] = outputs_cond.past_key_values

        # --- Null path for CFG ---
        if self.cfg_coef != 1.0 and null_fused2 is not None:
            null_fused2 = null_fused2.to(device=h1.device, dtype=h1.dtype)
            null_fused2 = null_fused2[:, :seq_len, :]
            if seq_len > 1:
                null_fused2 = null_fused2[:, :-1, :]

            # Use the full null hidden sequence from the null model's prefill
            null_h1_full = self._null_hidden1_full.pop(req_id, None)
            if null_h1_full is not None:
                null_h1_seq = null_h1_full.unsqueeze(0).to(device=h1.device, dtype=h1.dtype)
                null_h1_seq = null_h1_seq[:, : h1.shape[1], :]
            else:
                null_h1_seq = torch.zeros_like(h1)

            cat_null = torch.cat([null_fused2, null_h1_seq], dim=-1)
            t2_input_null = self.mlp(cat_null)

            outputs_null = self.transformer2(
                inputs_embeds=t2_input_null,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            self._t2_null_kv_states[req_id] = outputs_null.past_key_values

    def cleanup_request(self, request_id: str) -> None:
        """Free per-request state when request completes."""
        self._null_kv_states.pop(request_id, None)
        self._t2_kv_states.pop(request_id, None)
        self._t2_null_kv_states.pop(request_id, None)
        self._null_hidden1_cache.pop(request_id, None)
        self._null_hidden1_full.pop(request_id, None)
        self._pending_t2_prefill.pop(request_id, None)
        self._ar_states.pop(request_id, None)
        self._force_eos_reqs.discard(request_id)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Hook called by the AR runner when requests terminate (normal or cancelled).

        Ensures per-request state is freed even for abnormally terminated requests
        (cancel, timeout, preempt) that bypass the normal finish path in preprocess.
        """
        for req_id in finished_req_ids:
            self.cleanup_request(str(req_id))

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> Any:
        """Wrap forward outputs into OmniOutput with audio_codes when available.

        The runner calls this after forward(). model_intermediate_buffer is
        a list[dict] in batch order. If any request has finished generation
        (audio_codes stored in buffer via preprocess), we emit them through
        multimodal_outputs so the downstream Stage 1 (Flow1dVAE) receives them.

        NOTE: Current implementation assumes max_num_seqs=1 (enforced via
        deploy YAML engine_args). If multiple requests
        finish in the same scheduler step, torch.cat would merge their codes
        into one tensor that cannot be split back. Supporting batched
        completions requires restructuring into a per-request keyed payload.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information") or []

        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                audio_codes_list.append(ac.to(torch.long))

        if audio_codes_list:
            audio_codes = torch.cat(audio_codes_list, dim=0)
            mm: dict[str, Any] = {
                "audio_codes": audio_codes,
                "sr": torch.tensor([int(self.config.sample_rate)], dtype=torch.int32),
            }
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs=mm)

        return OmniOutput(
            text_hidden_states=hidden if isinstance(hidden, torch.Tensor) else None,
            multimodal_outputs={},
        )

    # ---- Weight loading ----

    def _audiolm_checkpoint_path(self) -> str:
        """Resolve upstream model.pt relative to the SongGeneration repo."""
        base = self.model_path or ""
        for rel in (
            os.path.join("songgeneration_v2_large", "model.pt"),
            "model.pt",
        ):
            path = os.path.join(base, rel) if base else rel
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"SongGeneration v2 LeLM checkpoint not found under {base}. Expected songgeneration_v2_large/model.pt."
        )

    def _strip_audiolm_prefix(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        """Strip 'audiolm.' prefix and skip modules loaded separately or unused."""
        for name, tensor in weights:
            if not name.startswith("audiolm."):
                continue
            name = name[len("audiolm.") :]
            if name.startswith(
                (
                    "att_dropout.",
                    "cfg_dropout.",
                    "transformer2.",
                    "condition_provider.",
                )
            ):
                continue
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader + WeightsMapper (Qwen3-TTS pattern).

        Upstream checkpoint: songgeneration_v2_large/model.pt with 'audiolm.' prefix.
        After stripping prefix, WeightsMapper handles the rest:
          - transformer.model.layers.* → model.layers.*
          - transformer.lm_head.* → lm_head.*
          - emb.*, layer2_emb.*, linears.*, mlp.*, out_norm.* → direct match
        """
        # Drain the iterator (vLLM's loader provides it from safetensors/pt)
        # and load from the actual .pt checkpoint ourselves.
        for _ in weights:
            pass

        ckpt_path = self._audiolm_checkpoint_path()
        logger.info("Loading SongGenerationV2LeLM weights from %s", ckpt_path)
        # weights_only=True: the checkpoint is downloaded from an external repo,
        # and upstream model.pt is a plain state_dict of tensors, so no
        # arbitrary-code unpickling is needed.
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]

        # Load cond model + embeddings + heads via AutoWeightsLoader
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            self._strip_audiolm_prefix(raw.items()),
            mapper=self.hf_to_vllm_mapper,
        )

        # Load null_model (HF LlamaModel) from same checkpoint weights
        null_state = {}
        for name, tensor in self._strip_audiolm_prefix(raw.items()):
            if name.startswith("transformer.model."):
                hf_name = name[len("transformer.model.") :]
                null_state[hf_name] = tensor
        info = self.null_model.model.load_state_dict(null_state, strict=False)
        if info.unexpected_keys:
            logger.debug("null_model unexpected keys: %s", info.unexpected_keys[:5])
        loaded |= {f"null_model.model.{k}" for k in null_state if k not in info.unexpected_keys}

        # Load transformer2 (HF LlamaModel, 12L sub) for streams 1+2
        if self.transformer2 is not None:
            t2_state = {}
            prefix = "audiolm.transformer2.model."
            for name, tensor in raw.items():
                if name.startswith(prefix):
                    hf_name = name[len(prefix) :]
                    t2_state[hf_name] = tensor
            if t2_state:
                info_t2 = self.transformer2.load_state_dict(t2_state, strict=False)
                if info_t2.unexpected_keys:
                    logger.debug("transformer2 unexpected keys: %s", info_t2.unexpected_keys[:5])
                loaded |= {f"transformer2.{k}" for k in t2_state if k not in info_t2.unexpected_keys}
                loaded_t2 = len(t2_state) - len(info_t2.unexpected_keys)
                logger.info("SongGenerationV2LeLM: loaded %d transformer2 params", loaded_t2)

        # Load condition_provider eagerly (NOT lazily at first request).
        # The checkpoint has audiolm.condition_provider.* weights for
        # output_proj, structure_emb, prompt_audio embeddings, EOT_emb, etc.
        provider_state = {
            name[len("audiolm.condition_provider.") :]: tensor
            for name, tensor in raw.items()
            if name.startswith("audiolm.condition_provider.")
        }
        if provider_state:
            from .conditioners import build_conditioner_provider as _build_conditioner_provider

            dev = next(self.model.parameters()).device
            self.condition_provider = _build_conditioner_provider(
                self.config,
                upstream_repo_path=_resolve_upstream_repo_for_assets(self.model_path),
            ).to(dev)
            info_cp = self.condition_provider.load_state_dict(provider_state, strict=False)
            if info_cp.missing_keys:
                logger.warning("condition_provider missing keys: %s", info_cp.missing_keys[:10])
            if info_cp.unexpected_keys:
                logger.debug("condition_provider unexpected keys: %s", info_cp.unexpected_keys[:10])
            n_loaded = len(provider_state) - len(info_cp.unexpected_keys)
            loaded |= {f"condition_provider.{k}" for k in provider_state if k not in info_cp.unexpected_keys}
            logger.info("SongGenerationV2LeLM: loaded %d condition_provider params", n_loaded)

        logger.info("SongGenerationV2LeLM: loaded %d weight params total", len(loaded))

        # Free the three backbones' built-in embed_tokens (~67MB each in bf16):
        # all paths run on inputs_embeds (stream-0 uses self.emb[0]), so these
        # tables are dead weight. Done post-load so weight loading is undisturbed.
        import gc

        freed: list[str] = []
        for path, mod in (
            ("null_model.model", getattr(self.null_model, "model", None)),
            ("transformer2", self.transformer2),
            ("model", self.model),
        ):
            if mod is not None and getattr(mod, "embed_tokens", None) is not None:
                mod.embed_tokens = None
                freed.append(path)
        if freed:
            gc.collect()
            if torch.accelerator.is_available():
                torch.accelerator.empty_cache()
            logger.info("SongGenerationV2LeLM: freed bypassed embed_tokens on %s", freed)

        return loaded

    def compute_condition_prompt_len(
        self,
        lyrics: list[str | None],
        descriptions: list[str | None] | None = None,
        prompt_audio_codes: torch.Tensor | None = None,
        gen_type: str = "mixed",
    ) -> int:
        """Pre-compute the total prompt length (condition prefix + 1 start token).

        Call this BEFORE constructing the engine request to determine the
        correct `prompt_token_ids` length. The returned length equals the
        output of fuser.forward(..., first_step=True) on dim-1.

        Usage:
            prompt_len = model.compute_condition_prompt_len(lyrics=[lyric], ...)
            prompt_token_ids = [SPECIAL_TOKEN_ID] * prompt_len
        """
        from .prompt_builder import build_condition_tensors

        if not hasattr(self, "condition_provider"):
            raise RuntimeError("condition_provider not loaded yet; call after load_weights()")

        with torch.no_grad():
            cond_tensors, B = build_condition_tensors(
                self.condition_provider,
                lyrics=lyrics,
                descriptions=descriptions,
                prompt_audio_codes=prompt_audio_codes,
                gen_type=gen_type,
                cfg_pair=True,
            )

            # Split to get cond-half only
            cond_half: dict = {}
            for name, (e1, e2, mask) in cond_tensors.items():
                cond_half[name] = (e1[:B], e2[:B], mask[:B])

            # Simulate fuser prepend on a single token (the start token)
            device = next(self.condition_provider.parameters()).device
            dummy_input = torch.zeros(1, 1, self.dim, device=device)
            dummy_input2 = torch.zeros(1, 1, self.dim, device=device)
            fused1, _ = self.fuser(dummy_input, dummy_input2, cond_half, first_step=True)
            return int(fused1.shape[1])

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        """Placeholder conditioning for vLLM KV-cache profiling / dummy runs."""
        return [
            {
                "lyric": "[verse] dummy",
                "gen_type": "mixed",
                "seed": 42,
                "max_gen_len": 8,
                "_is_dummy": True,
            }
            for _ in range(num_reqs)
        ]
