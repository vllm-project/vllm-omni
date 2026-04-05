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
import torch.nn.functional as F
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


def _get_vibevoice_head(hidden_size: int, acoustic_dim: int, time_dim: int,
                        head_layers: int, head_ffn_ratio: float,
                        bottleneck_dim: int | None) -> nn.Module:
    try:
        from tada.nn.vibevoice import VibeVoiceDiffusionHead, VibeVoiceDiffusionHeadConfig
    except ImportError as e:
        raise ImportError(
            "hume-tada package is required for TADA TTS. "
            "Install with: pip install hume-tada"
        ) from e

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

    has_preprocess = False
    has_postprocess = True
    have_multimodal_outputs = True

    gpu_resident_buffer_keys: set[str] = {"acoustic_feat_last", "time_feat_last"}

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
        _head_size: int = self.hidden_size if self.bottleneck_dim is None else self.bottleneck_dim

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
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(int(cfg.vocab_size))
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.acoustic_proj = nn.Linear(self.acoustic_dim, self.hidden_size, bias=False)
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

        self.num_flow_steps: int = 10
        self.acoustic_cfg_scale: float = 1.6
        self.noise_temperature: float = 0.9

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        is_decode = (
            inputs_embeds is None
            and model_intermediate_buffer is not None
            and any(b.get("acoustic_feat_last") is not None for b in model_intermediate_buffer)
        )

        if inputs_embeds is None:
            # Base token embeddings [total_tokens, H]
            base_emb = self.model.embed_tokens(input_ids)

            if is_decode and model_intermediate_buffer is not None:
                acoustic_add = torch.zeros_like(base_emb)
                time_add = torch.zeros_like(base_emb)
                # model_intermediate_buffer[i] corresponds to batch-position i.
                for i, buf in enumerate(model_intermediate_buffer):
                    if i >= base_emb.shape[0]:
                        break
                    af = buf.get("acoustic_feat_last")
                    if isinstance(af, torch.Tensor) and af.numel() > 0:
                        af_dev = af.to(device=base_emb.device, dtype=base_emb.dtype)
                        acoustic_add[i] = self.acoustic_proj(af_dev.reshape(1, -1)).squeeze(0)
                        mask_emb = self.acoustic_mask_emb(
                            torch.ones(1, dtype=torch.long, device=base_emb.device)
                        ).squeeze(0)
                        acoustic_add[i] += mask_emb

                    tf_b = buf.get("time_before_last")
                    tf_a = buf.get("time_after_last")
                    if isinstance(tf_b, torch.Tensor) and tf_b.numel() > 0:
                        tb = tf_b.to(device=base_emb.device).reshape(1).clamp(0, self.num_time_classes - 1)
                        ta = tf_a.to(device=base_emb.device).reshape(1).clamp(0, self.num_time_classes - 1) \
                            if isinstance(tf_a, torch.Tensor) and tf_a.numel() > 0 else tb
                        time_add[i] = (
                            self.time_start_embed(tb).squeeze(0)
                            + self.time_end_embed(ta).squeeze(0)
                        )

                inputs_embeds = base_emb + acoustic_add + time_add
            else:
                inputs_embeds = base_emb

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
        **_: Any,
    ) -> OmniOutput:
        """Emit previous step's acoustic features for output processor accumulation."""
        hidden = model_outputs
        if isinstance(model_outputs, OmniOutput):
            hidden = model_outputs.text_hidden_states

        if not model_intermediate_buffer:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        feat_list: list[torch.Tensor] = []
        mask_list: list[torch.Tensor] = []

        for buf in model_intermediate_buffer:
            if not isinstance(buf, dict):
                continue
            af = buf.get("acoustic_feat_last")
            if isinstance(af, torch.Tensor) and af.numel() > 0:
                feat_list.append(af.reshape(1, self.acoustic_dim).cpu())
                mask_list.append(torch.ones(1, dtype=torch.long))

        if not feat_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

        acoustic_feats = torch.cat(feat_list, dim=0)   # [B, 512]
        text_token_mask = torch.cat(mask_list, dim=0)  # [B]

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={
                "acoustic_features": acoustic_feats,
                "text_token_mask": text_token_mask,
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

        Returns acoustic_feat_last [1, 512], time_before_last [1], time_after_last [1].
        """
        if hidden_states_slice is None or hidden_states_slice.numel() == 0:
            return {}

        last_h = hidden_states_slice[-1:].float()   # [1, H]
        device = last_h.device
        cond = self.bottleneck_proj(last_h)

        total_dim = self.acoustic_dim + self.time_dim
        noise = torch.randn(1, total_dim, device=device, dtype=last_h.dtype) * self.noise_temperature
        speech = noise.clone()

        t_span = torch.linspace(0.0, 1.0, self.num_flow_steps + 1, device=device, dtype=last_h.dtype)
        neg_cond = torch.zeros_like(cond)

        for step_i in range(self.num_flow_steps):
            t = t_span[step_i]
            dt = t_span[step_i + 1] - t
            velocity = self._compute_velocity(speech, t, cond, neg_cond)
            speech = speech + dt * velocity

        acoustic_feat = speech[..., : self.acoustic_dim].detach()  # [1, 512]
        time_gray = (speech[..., self.acoustic_dim:] > 0).float()

        bits_b = time_gray[..., : self.num_time_bits]
        bits_a = time_gray[..., self.num_time_bits:]
        time_before = _decode_gray_code(bits_b).clamp(0, self.num_time_classes - 1)  # [1]
        time_after  = _decode_gray_code(bits_a).clamp(0, self.num_time_classes - 1)  # [1]

        return {
            "acoustic_feat_last": acoustic_feat,  # kept on GPU via gpu_resident_buffer_keys
            "time_before_last": time_before,
            "time_after_last": time_after,
        }

    def _compute_velocity(
        self,
        speech: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        neg_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CFG-scaled flow-matching velocity."""
        t_scalar = t.reshape(1).to(speech)
        if self.acoustic_cfg_scale != 1.0:
            speech_cat = torch.cat([speech, speech], dim=0)
            t_cat = t_scalar.repeat(2)
            cond_cat = torch.cat([cond, neg_cond], dim=0)
            vel_cat = self.prediction_head(
                speech_cat, t_cat, condition=cond_cat.squeeze(1) if cond_cat.dim() == 3 else cond_cat
            )
            vel_pos, vel_neg = vel_cat.chunk(2, dim=0)
            # Apply CFG to acoustic dims; keep identity for time dims.
            acoustic_part = vel_neg[..., :self.acoustic_dim] + self.acoustic_cfg_scale * (
                vel_pos[..., :self.acoustic_dim] - vel_neg[..., :self.acoustic_dim]
            )
            time_part = vel_neg[..., self.acoustic_dim:] + self.acoustic_cfg_scale * (
                vel_pos[..., self.acoustic_dim:] - vel_neg[..., self.acoustic_dim:]
            )
            return torch.cat([acoustic_part, time_part], dim=-1)
        else:
            cond_in = cond.squeeze(1) if cond.dim() == 3 else cond
            return self.prediction_head(speech, t_scalar.expand(speech.shape[0]), condition=cond_in)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a TadaForCausalLM checkpoint.

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
        """
        _SKIP_PREFIXES = ("_decoder.", "_tokenizer.", "_encoder.", "_acoustic_spkr_verf.")

        stashed: list[tuple[str, torch.Tensor]] = [
            (n, t) for n, t in weights if not any(n.startswith(p) for p in _SKIP_PREFIXES)
        ]

        llama_weights: list[tuple[str, torch.Tensor]] = []
        other_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in stashed:
            if name.startswith("model."):
                llama_weights.append((name, tensor))
            else:
                other_weights.append((name, tensor))

        loaded: set[str] = set()
        if hasattr(self.model, "load_weights"):
            loaded_llama = self.model.load_weights(llama_weights)
            if loaded_llama:
                loaded |= {f"model.{n}" for n in loaded_llama}
        else:
            params = dict(self.model.named_parameters(remove_duplicate=False))
            for name, tensor in llama_weights:
                key = name[len("model."):]  # strip "model." prefix
                if key in params:
                    default_weight_loader(params[key], tensor)
                    loaded.add(name)

        params = dict(self.named_parameters(remove_duplicate=False))
        for name, tensor in other_weights:
            if name not in params:
                continue
            default_weight_loader(params[name], tensor)
            loaded.add(name)

        return loaded

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Any:
        from vllm.model_executor.layers.sampler import get_sampler
        sampler = get_sampler()
        return sampler(logits, sampling_metadata)
