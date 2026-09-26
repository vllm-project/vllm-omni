"""TADA TTS -- Stage 0: AR text generation + per-token flow-matching diffusion.

Each decode step runs the VibeVoice diffusion head on the last hidden state to
produce a 512-dim acoustic feature, stored in model_intermediate_buffer and fed
back as conditioning for the next step.  make_omni_output() emits the previous
step's feature so the output processor accumulates acoustic_features [T, 512].

Weights come from HumeAI/tada-1b or tada-3b-ml.  _decoder.* keys are skipped;
the codec decoder is loaded lazily by TadaVocoder from HumeAI/tada-codec.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _decode_gray_code(bits: torch.Tensor) -> torch.Tensor:
    """Decode gray-coded bits [B, n_bits] → integer values [B]."""
    n = bits.shape[-1]
    result = bits[..., 0].clone().long()
    for i in range(1, n):
        result = result * 2 + (bits[..., i].long() ^ (result & 1))
    return result


def _build_time_schedule(num_steps: int, schedule: str, device: torch.device) -> torch.Tensor:
    """Build the ODE time discretisation in [0, 1] for the given schedule."""
    if schedule == "cosine":
        u = torch.linspace(0, 1, num_steps + 1, device=device)
        return 0.5 * (1 - torch.cos(math.pi * u))
    if schedule == "logsnr":
        # Uniform in log-SNR space, denser near t=0 (denoising onset).
        log_snr = torch.linspace(5.0, -5.0, num_steps + 1, device=device)
        t_span = torch.sigmoid(-log_snr / 2)
        t_span[0] = 0.0
        t_span[-1] = 1.0
        return t_span
    return torch.linspace(0, 1, num_steps + 1, device=device)


def _scheduled_cfg(base_scale: float, t: float, schedule: str) -> float:
    """Return the effective CFG scale at ODE timestep t for the given schedule."""
    if schedule == "constant" or base_scale == 1.0:
        return base_scale
    if schedule == "linear":
        return 1.0 + (base_scale - 1.0) * (1.0 - t)
    if schedule == "cosine":
        return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))
    return base_scale


def _get_vibevoice_head(
    hidden_size: int,
    acoustic_dim: int,
    time_dim: int,
    head_layers: int,
    head_ffn_ratio: float,
    bottleneck_dim: int | None,
) -> nn.Module:
    from .codec import VibeVoiceDiffusionHead, VibeVoiceDiffusionHeadConfig

    cfg = VibeVoiceDiffusionHeadConfig(
        diffusion_type="ddpm",
        head_ffn_ratio=head_ffn_ratio,
        head_layers=head_layers,
        hidden_size=hidden_size if bottleneck_dim is None else bottleneck_dim,
        latent_size=acoustic_dim + time_dim,
        model_type="vibevoice_diffusion_head",
        rms_norm_eps=1e-5,
        speech_vae_dim=acoustic_dim + time_dim,
    )
    return VibeVoiceDiffusionHead(cfg)


class TadaARStageForConditionalGeneration(nn.Module):
    """Stage 0: AR text generation + per-token flow-matching acoustic synthesis."""

    # The stage walks the input text one token per step, producing one acoustic frame each;
    # it does not free-generate. preprocess() forces each decode step to the next text token
    # and rebuilds the input embedding with one-step acoustic/time feedback.
    has_preprocess = True
    has_postprocess = True
    have_multimodal_outputs = True

    # Per-request feedback state lives in the runner's intermediate buffer; preprocess moves
    # the feedback frame back to the device each step.
    gpu_resident_buffer_keys: set[str] = set()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config

        self.hidden_size: int = int(cfg.hidden_size)
        self.acoustic_dim: int = int(getattr(cfg, "acoustic_dim", 512))
        self.num_time_classes: int = int(getattr(cfg, "num_time_classes", 1024))
        self.num_time_bits: int = math.ceil(math.log2(self.num_time_classes))
        self.time_dim: int = 2 * self.num_time_bits
        self.shift_acoustic: int = int(getattr(cfg, "shift_acoustic", 5))
        self.bottleneck_dim: int | None = getattr(cfg, "bottleneck_dim", None)

        self.model = LlamaModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                int(cfg.vocab_size),
                self.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            # tada-1b has tie_word_embeddings=true and ships NO lm_head.* weights;
            # without tying, lm_head stays randomly initialised → garbage text logits.
            if getattr(cfg, "tie_word_embeddings", False):
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(int(cfg.vocab_size))
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # acoustic_proj uses a bias term (the checkpoint ships ``acoustic_proj.bias``).
        self.acoustic_proj = nn.Linear(self.acoustic_dim, self.hidden_size, bias=True)
        # 0 = no acoustic (prompt), 1 = real acoustic
        self.acoustic_mask_emb = nn.Embedding(2, self.hidden_size)
        self.acoustic_mask_emb.weight.data.fill_(0.0)
        self.time_start_embed = nn.Embedding(self.num_time_classes, self.hidden_size)
        self.time_end_embed = nn.Embedding(self.num_time_classes, self.hidden_size)

        if self.bottleneck_dim is not None:
            self.bottleneck_proj = nn.Linear(self.hidden_size, self.bottleneck_dim, bias=False)
        else:
            self.bottleneck_proj = nn.Identity()

        self.prediction_head = _get_vibevoice_head(
            hidden_size=self.hidden_size,
            acoustic_dim=self.acoustic_dim,
            time_dim=self.time_dim,
            head_layers=int(getattr(cfg, "head_layers", 4)),
            head_ffn_ratio=float(getattr(cfg, "head_ffn_ratio", 3.0)),
            bottleneck_dim=self.bottleneck_dim,
        )

        # Flow-matching hyperparameters (overridable via the model config).
        self.num_flow_steps: int = int(getattr(cfg, "num_flow_matching_steps", 10))
        self.acoustic_cfg_scale: float = float(getattr(cfg, "acoustic_cfg_scale", 1.6))
        self.duration_cfg_scale: float = float(getattr(cfg, "duration_cfg_scale", 1.0))
        self.noise_temperature: float = float(getattr(cfg, "noise_temperature", 0.9))
        self.cfg_schedule: str = str(getattr(cfg, "cfg_schedule", "cosine"))
        self.time_schedule: str = str(getattr(cfg, "time_schedule", "logsnr"))

        # Diffusion runs in normalised space; features are de-normalised before decoding.
        self.acoustic_mean: float = float(getattr(cfg, "acoustic_mean", 0.0))
        self.acoustic_std: float = float(getattr(cfg, "acoustic_std", 1.5))

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    @torch.no_grad()
    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **kw: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Walk the input text, producing one acoustic frame per token.

        On **prefill** this builds the prompt-region embeddings (``[BOS]`` + chat-template
        headers, plus the optional reference-audio acoustic features for voice cloning). On
        each **decode** step it takes the next text id from ``tada_walk_ids`` and rebuilds the
        input embedding with one-step acoustic/time feedback:
        ``embed_tokens(id) + acoustic_proj(prev_feat) + acoustic_mask_emb(mask)
        + time_start_embed(time_before) + time_end_embed(time_after)``.

        Returns ``(req_input_ids, req_embeds, update_dict)``; the runner writes both back into
        the flat ``input_ids``/``inputs_embeds`` for this request's span and merges
        ``update_dict`` into the per-request buffer. The sampled token is unused; the fixed
        length is enforced by ``max_tokens`` in the request's sampling params.
        """
        is_prefill = bool(kw.get("_omni_is_prefill", False))
        base = self.model.embed_tokens(input_ids)  # [span, H]
        device, dtype = base.device, base.dtype

        if is_prefill:
            # Substitute known reference-audio acoustic features over the prompt region for
            # voice cloning (no-op when no reference is provided).
            prompt_ac = kw.get("tada_prompt_acoustic")
            if isinstance(prompt_ac, torch.Tensor) and prompt_ac.numel() > 0:
                base = self._add_prompt_acoustic_embeds(base, kw, device, dtype)
            return input_ids, base, {}

        # ---- decode: one forced text token + one-step feedback ----
        walk_ids = kw.get("tada_walk_ids") or []
        offset = int(kw.get("tada_offset", 0))
        if offset < len(walk_ids):
            tok_id = int(walk_ids[offset])
        elif walk_ids:
            tok_id = int(walk_ids[-1])  # bounded by max_tokens; reached only as a safety fallback
        else:
            tok_id = int(input_ids.reshape(-1)[-1].item())
        tok = torch.tensor([tok_id], device=device, dtype=input_ids.dtype)
        emb = self.model.embed_tokens(tok)  # [1, H]

        af = kw.get("acoustic_feat_last")
        zeros_t = torch.zeros(1, dtype=torch.long, device=device)
        if isinstance(af, torch.Tensor) and af.numel() > 0:
            # Feedback from the previous step's acoustic frame + predicted durations.
            af = af.to(device=device, dtype=dtype).reshape(1, -1)
            tb = kw.get("time_before_last")
            ta = kw.get("time_after_last")
            tb_i = (
                tb.to(device=device).reshape(1).clamp(0, self.num_time_classes - 1)
                if isinstance(tb, torch.Tensor) and tb.numel() > 0
                else zeros_t
            )
            ta_i = (
                ta.to(device=device).reshape(1).clamp(0, self.num_time_classes - 1)
                if isinstance(ta, torch.Tensor) and ta.numel() > 0
                else tb_i
            )
            emb = (
                emb
                + self.acoustic_proj(af)
                + self.acoustic_mask_emb(torch.ones(1, dtype=torch.long, device=device))
                + self.time_start_embed(tb_i)
                + self.time_end_embed(ta_i)
            )
        else:
            # Warm-up step (no previous frame yet): acoustic, mask and time are all zero.
            zeros_ac = torch.zeros(1, self.acoustic_dim, device=device, dtype=dtype)
            emb = (
                emb
                + self.acoustic_proj(zeros_ac)
                + self.acoustic_mask_emb(zeros_t)
                + self.time_start_embed(zeros_t)
                + self.time_end_embed(zeros_t)
            )

        return tok, emb, {"tada_offset": offset + 1}

    def _add_prompt_acoustic_embeds(
        self, base: torch.Tensor, kw: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Add the reference-audio acoustic features over the prompt region during prefill,
        putting the reference voice into the KV cache. Position ``t`` is conditioned on
        ``prompt_acoustic[t - shift - 1]`` (the acoustic stream lags the text by
        ``shift_acoustic``). Assumes the prompt arrives in a single prefill chunk. ``base`` is
        the token embedding for the whole prefill span [P, H].
        """
        pa = kw.get("tada_prompt_acoustic")
        if not (isinstance(pa, torch.Tensor) and pa.numel() > 0):
            return base
        P = base.shape[0]
        shift = self.shift_acoustic
        pa = pa.to(device=device, dtype=dtype)
        pm = kw.get("tada_prompt_masks")
        pm = (
            pm.to(device=device).long()
            if isinstance(pm, torch.Tensor) and pm.numel() > 0
            else torch.ones(pa.shape[0], dtype=torch.long, device=device)
        )

        acoustic_full = torch.zeros(P, self.acoustic_dim, device=device, dtype=dtype)
        masks_full = torch.zeros(P, dtype=torch.long, device=device)
        n_ac = min(P - shift - 1, pa.shape[0])
        if n_ac > 0:
            acoustic_full[shift + 1 : shift + 1 + n_ac] = pa[:n_ac]
            masks_full[shift + 1 : shift + 1 + n_ac] = pm[:n_ac]

        tb_full = torch.zeros(P, dtype=torch.long, device=device)
        ta_full = torch.zeros(P, dtype=torch.long, device=device)
        ptb = kw.get("tada_prompt_tb")
        pta = kw.get("tada_prompt_ta")
        if isinstance(ptb, torch.Tensor) and isinstance(pta, torch.Tensor) and ptb.numel() > 1:
            ptb = ptb.to(device=device).long().clamp(0, self.num_time_classes - 1)
            pta = pta.to(device=device).long().clamp(0, self.num_time_classes - 1)
            n_t = min(P - shift - 1, ptb.shape[0] - 1)
            if n_t > 0:
                tb_full[shift + 1 : shift + 1 + n_t] = ptb[1 : 1 + n_t]
                ta_full[shift + 1 : shift + 1 + n_t] = pta[1 : 1 + n_t]

        return (
            base
            + self.acoustic_proj(acoustic_full)
            + self.acoustic_mask_emb(masks_full)
            + self.time_start_embed(tb_full)
            + self.time_end_embed(ta_full)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        # preprocess() builds the fed-back inputs_embeds per request; here we only run
        # the LLM backbone. The embed_tokens fallback covers profiling / graph-capture
        # dummy runs where preprocess is not invoked.
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        return self.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        **_: Any,
    ) -> OmniOutput:
        """Emit the per-step acoustic features for the output processor to accumulate.

        Output is laid out on the token-aligned leading dimension (``[total_tokens, feat_dim]``)
        so the runner slices each request's own rows via ``request_token_spans``. The leading
        dimension must equal the total scheduled token count, otherwise the features are dropped
        before the vocoder in mixed prefill/decode steps.
        """
        hidden = model_outputs
        if isinstance(model_outputs, OmniOutput):
            hidden = model_outputs.text_hidden_states

        if not model_intermediate_buffer:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        if not isinstance(hidden, torch.Tensor) or hidden.numel() == 0:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        total_tokens = hidden.shape[0]

        if request_token_spans is None:
            logger.warning_once(
                "[TADA] request_token_spans unavailable in make_omni_output; "
                "falling back to batch-index row mapping, which is only correct "
                "when each request contributes a single scheduled token."
            )

        acoustic_feats = torch.zeros(total_tokens, self.acoustic_dim, dtype=torch.float32)
        text_token_mask = torch.zeros(total_tokens, dtype=torch.long)
        # Per-token predicted duration (frames), used by the vocoder to expand the sequence.
        time_before = torch.zeros(total_tokens, dtype=torch.long)
        have_feats = False

        for i, buf in enumerate(model_intermediate_buffer):
            if not isinstance(buf, dict):
                continue
            af = buf.get("acoustic_feat_last")
            if not (isinstance(af, torch.Tensor) and af.numel() > 0):
                continue
            # Place this request's frame on its own decode-token row so the
            # runner slices it back to the right request.
            if request_token_spans is not None and i < len(request_token_spans):
                row_start, row_end = request_token_spans[i]
                # Emit only for single-token decode steps. The prompt/prefill region is
                # processed in a multi-token span; its frames are warm-up/prompt and must not
                # enter the audio, keeping the emitted stream aligned to the walked text tokens.
                if int(row_end) - int(row_start) > 1:
                    continue
                row = min(int(row_end), total_tokens) - 1
            else:
                row = i
            if row < 0 or row >= total_tokens:
                continue
            # Skip the first ``tada_trim_lead`` decode steps: they walk the reference
            # transcript tail to smooth the prompt→synthesis boundary and are fed back but
            # not emitted (they are transcript continuation, not the requested text).
            trim_lead = int(buf.get("tada_trim_lead", 0) or 0)
            if trim_lead and int(buf.get("tada_offset", 0) or 0) <= trim_lead:
                continue
            acoustic_feats[row] = af.reshape(self.acoustic_dim).to(dtype=torch.float32).cpu()
            text_token_mask[row] = 1
            tb = buf.get("time_before_last")
            if isinstance(tb, torch.Tensor) and tb.numel() > 0:
                time_before[row] = int(tb.reshape(-1)[0].item())
            have_feats = True

        if not have_feats:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "acoustic_features": acoustic_feats,  # [total_tokens, feat_dim]
                "text_token_mask": text_token_mask,  # [total_tokens]
                "time_before": time_before,  # [total_tokens]
            },
        )

    @torch.no_grad()
    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Run flow-matching diffusion on the last token's hidden state.

        Returns the acoustic feature and the before/after durations for this step.
        """
        if hidden_states_slice is None or hidden_states_slice.numel() == 0:
            return {}

        # Run flow-matching in the diffusion head's dtype to avoid a matmul dtype mismatch.
        pdtype = next(self.prediction_head.parameters()).dtype
        last_h = hidden_states_slice[-1:].to(pdtype)  # [1, H]
        device = last_h.device
        cond = self.bottleneck_proj(last_h)

        total_dim = self.acoustic_dim + self.time_dim
        speech = torch.randn(1, total_dim, device=device, dtype=pdtype) * self.noise_temperature

        # ODE time schedule + per-step CFG decay.
        t_span = _build_time_schedule(self.num_flow_steps, self.time_schedule, device)
        neg_cond = torch.zeros_like(cond)

        for step_i in range(self.num_flow_steps):
            t = t_span[step_i]
            dt = t_span[step_i + 1] - t
            t_val = float(t.item())
            a_cfg = _scheduled_cfg(self.acoustic_cfg_scale, t_val, self.cfg_schedule)
            d_cfg = _scheduled_cfg(self.duration_cfg_scale, t_val, self.cfg_schedule)
            velocity = self._compute_velocity(speech, t, cond, neg_cond, a_cfg, d_cfg)
            speech = speech + dt * velocity

        acoustic_feat = speech[..., : self.acoustic_dim].detach()  # [1, 512]
        time_gray = (speech[..., self.acoustic_dim :] > 0).float()

        bits_b = time_gray[..., : self.num_time_bits]
        bits_a = time_gray[..., self.num_time_bits :]
        time_before = _decode_gray_code(bits_b).clamp(0, self.num_time_classes - 1)  # [1]
        time_after = _decode_gray_code(bits_a).clamp(0, self.num_time_classes - 1)  # [1]

        return {
            "acoustic_feat_last": acoustic_feat,
            "time_before_last": time_before,
            "time_after_last": time_after,
        }

    def _compute_velocity(
        self,
        speech: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
        acoustic_cfg: float,
        duration_cfg: float,
    ) -> torch.Tensor:
        """CFG-scaled flow-matching velocity.

        ``acoustic_cfg`` guides the acoustic dimensions and ``duration_cfg`` the time/duration
        dimensions. ``cond`` is already bottleneck-projected.
        """
        t_scalar = t.reshape(1).to(speech)
        if acoustic_cfg != 1.0:
            speech_cat = torch.cat([speech, speech], dim=0)
            t_cat = t_scalar.repeat(2)
            cond_cat = torch.cat([cond, neg_cond], dim=0)
            vel_cat = self.prediction_head(
                speech_cat, t_cat, condition=cond_cat.squeeze(1) if cond_cat.dim() == 3 else cond_cat
            )
            vel_pos, vel_neg = vel_cat.chunk(2, dim=0)
            acoustic_part = vel_neg[..., : self.acoustic_dim] + acoustic_cfg * (
                vel_pos[..., : self.acoustic_dim] - vel_neg[..., : self.acoustic_dim]
            )
            time_part = vel_neg[..., self.acoustic_dim :] + duration_cfg * (
                vel_pos[..., self.acoustic_dim :] - vel_neg[..., self.acoustic_dim :]
            )
            return torch.cat([acoustic_part, time_part], dim=-1)
        else:
            cond_in = cond.squeeze(1) if cond.dim() == 3 else cond
            return self.prediction_head(speech, t_scalar.expand(speech.shape[0]), condition=cond_in)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a TadaForCausalLM checkpoint in a single pass.

        Key mapping:
          model.*            → self.model (LlamaModel, vLLM native)
          lm_head.*          → self.lm_head
          acoustic_proj.*    → self.acoustic_proj
          acoustic_mask_emb.*→ self.acoustic_mask_emb
          time_start_embed.* → self.time_start_embed
          time_end_embed.*   → self.time_end_embed
          bottleneck_proj.*  → self.bottleneck_proj
          prediction_head.*  → self.prediction_head
          _decoder.*         → skipped (loaded by TadaVocoder)
          _tokenizer.*       → skipped
          _encoder.*         → skipped

        The checkpoint iterator is consumed exactly once: ``model.*`` tensors are
        streamed (renamed) straight into ``self.model.load_weights`` while every
        other tensor is dispatched inline, so large (1B+) checkpoints never get
        materialized into an intermediate list.
        """
        _SKIP_PREFIXES = ("_decoder.", "_tokenizer.", "_encoder.", "_acoustic_spkr_verf.")

        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        have_model_loader = hasattr(self.model, "load_weights")
        model_params: dict[str, torch.Tensor] | None = (
            None if have_model_loader else dict(self.model.named_parameters(remove_duplicate=False))
        )

        def stream_model_weights() -> Iterable[tuple[str, torch.Tensor]]:
            """Yield renamed ``model.*`` tensors; dispatch all others inline."""
            for name, tensor in weights:
                if any(name.startswith(p) for p in _SKIP_PREFIXES):
                    continue
                if name.startswith("model."):
                    key = name[len("model.") :]  # strip "model." prefix
                    if have_model_loader:
                        yield key, tensor
                    elif model_params is not None and key in model_params:
                        default_weight_loader(model_params[key], tensor)
                        loaded.add(name)
                    continue
                if name in params:
                    default_weight_loader(params[name], tensor)
                    loaded.add(name)

        if have_model_loader:
            loaded_llama = self.model.load_weights(stream_model_weights())
            if loaded_llama:
                loaded |= {f"model.{n}" for n in loaded_llama}
        else:
            # Drive the generator to completion (no sub-loader to consume it).
            for _ in stream_model_weights():
                pass

        return loaded

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Any:
        from vllm.model_executor.layers.sampler import get_sampler

        sampler = get_sampler()
        return sampler(logits, sampling_metadata)
