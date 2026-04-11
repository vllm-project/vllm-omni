# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VoxCPM2 native AR talker — uses native MiniCPM4 base_lm directly.

Uses native VoxCPM2 modules (no PagedAttention, manual KV cache).
Each AR decode step:
  feat_encoder → base_lm → FSQ → residual_lm → LocDiT → stop

TODO(PagedAttention): The base_lm is a MiniCPM4 variant (GQA + LongRoPE,
use_mup=False).  vllm's MiniCPMModel already supports the architecture
(LongRoPE via Phi3LongRoPEScaledRotaryEmbedding, muP via config), but
two issues block replacing the native base_lm with a vllm MiniCPM4Model:
  1. Per-request state isolation — residual_lm and LocDiT diffusion use
     shared native KV caches; concurrent requests clobber each other.
     Fix: save/restore residual_lm cache per request, or pool N instances.
  2. Streaming audio — make_omni_output re-decodes all patches each step.
     Fix: sliding-window VAE decode (decode_pad pattern from nanovllm).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.minicpm import MiniCPMModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .voxcpm2_import_utils import import_voxcpm2_core

logger = init_logger(__name__)


class VoxCPM2TalkerForConditionalGeneration(nn.Module):
    """VoxCPM2 talker using native MiniCPM4 base_lm.

    Loads the full VoxCPM2 model natively and decomposes the AR loop:
    each vllm decode step runs one iteration of the native generate loop.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # Flags for OmniGPUModelRunner
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self._accumulated_patches: list[torch.Tensor] = []

        # vllm MiniCPMModel scaffold — needed for warmup/profiling/KV cache
        # sizing. Not used for actual computation (native modules are used).
        self.model = MiniCPMModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Placeholder — actual native model loaded in load_weights
        self._tts: nn.Module | None = None
        self._device = "cuda"
        self._side_dtype = torch.bfloat16

        # Config values
        self._patch_size = getattr(self.config, "patch_size", 4)
        self._feat_dim = getattr(self.config, "feat_dim", 64)
        self._inference_timesteps = 10
        self._cfg_value = 2.0

        # TODO: implement sliding-window VAE decode (nanovllm pattern)
        # for O(1) per-step streaming. Current impl re-decodes all patches.

    @property
    def tts(self) -> nn.Module:
        assert self._tts is not None, "Model not loaded yet"
        return self._tts

    # -------------------- vllm hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Embed input IDs using native base_lm with scale_emb."""
        embeds = self.tts.base_lm.embed_tokens(input_ids)
        return embeds * self.tts.config.lm_config.scale_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Full VoxCPM2 AR step: base_lm → FSQ → residual_lm → diffusion."""
        # Always run scaffold model to keep FlashInfer/attention happy
        model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(model_output, IntermediateTensors):
            return model_output
        scaffold_hidden = model_output
        if isinstance(scaffold_hidden, tuple):
            scaffold_hidden = scaffold_hidden[0]

        # Real computation: use native modules
        has_infos = bool(getattr(self, "_current_step_infos", None))
        is_prefill = scaffold_hidden.shape[0] > 1

        if is_prefill and has_infos:
            self._forward_prefill(inputs_embeds, scaffold_hidden.device)
            # Return scaffold output (right shape for engine) — our side
            # computation results are stored in instance state
            return scaffold_hidden

        if not is_prefill and hasattr(self, "_prev_feat_embed"):
            self._forward_decode(inputs_embeds, scaffold_hidden.device)
            return scaffold_hidden

        return scaffold_hidden

    def _build_prefill_inputs(self, text: str, dev: Any):
        """Build text_token / audio_feat / masks like native _generate_with_prompt_cache.

        Returns a dict with keys: text_token, audio_feat, text_mask, audio_mask,
        prefix_feat_cond. Handles zero-shot, reference (voice clone), continuation,
        and ref_continuation modes.
        """
        tts = self.tts
        dtype = self._side_dtype
        cache = getattr(self, "_prompt_cache", None)
        mode = cache.get("mode", "continuation") if cache else "zero_shot"

        if cache is not None and mode in ("continuation", "ref_continuation"):
            full_text = cache.get("prompt_text", "") + text
        else:
            full_text = text

        text_token = torch.LongTensor(tts.text_tokenizer(full_text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor([tts.audio_start_token], dtype=torch.int32, device=text_token.device),
            ],
            dim=-1,
        )
        text_length = text_token.shape[0]
        latent_dim = tts.audio_vae.latent_dim
        patch_size = tts.patch_size

        if mode in ("zero_shot", "continuation"):
            prompt_audio_feat = (
                cache["audio_feat"] if cache else torch.empty((0, patch_size, latent_dim), dtype=torch.float32)
            )
            audio_length = prompt_audio_feat.size(0)
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32)
            text_pad_feat = torch.zeros((text_length, patch_size, latent_dim), dtype=torch.float32)
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [
                    torch.ones(text_length, dtype=torch.int32),
                    torch.zeros(audio_length, dtype=torch.int32),
                ]
            )
            audio_mask = torch.cat(
                [
                    torch.zeros(text_length, dtype=torch.int32),
                    torch.ones(audio_length, dtype=torch.int32),
                ]
            )
        elif mode == "reference":
            ref_audio_feat = cache["ref_audio_feat"]
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = tts._make_ref_prefix(ref_audio_feat, text_token.device)
            text_pad_feat = torch.zeros((text_length, patch_size, latent_dim), dtype=torch.float32)
            text_token = torch.cat([ref_tokens.cpu(), text_token])
            audio_feat = torch.cat([ref_feats.cpu(), text_pad_feat], dim=0)
            text_mask = torch.cat([ref_t_mask.cpu(), torch.ones(text_length, dtype=torch.int32)])
            audio_mask = torch.cat([ref_a_mask.cpu(), torch.zeros(text_length, dtype=torch.int32)])
        else:
            # ref_continuation
            ref_audio_feat = cache["ref_audio_feat"]
            prompt_audio_feat = cache["audio_feat"]
            prompt_audio_length = prompt_audio_feat.size(0)
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = tts._make_ref_prefix(ref_audio_feat, text_token.device)
            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32)
            text_pad_feat = torch.zeros((text_length, patch_size, latent_dim), dtype=torch.float32)
            text_token = torch.cat([ref_tokens.cpu(), text_token, prompt_pad_token])
            audio_feat = torch.cat([ref_feats.cpu(), text_pad_feat, prompt_audio_feat], dim=0)
            text_mask = torch.cat(
                [
                    ref_t_mask.cpu(),
                    torch.ones(text_length, dtype=torch.int32),
                    torch.zeros(prompt_audio_length, dtype=torch.int32),
                ]
            )
            audio_mask = torch.cat(
                [
                    ref_a_mask.cpu(),
                    torch.zeros(text_length, dtype=torch.int32),
                    torch.ones(prompt_audio_length, dtype=torch.int32),
                ]
            )

        return {
            "text_token": text_token.unsqueeze(0).to(dev),
            "audio_feat": audio_feat.unsqueeze(0).to(dev).to(dtype),
            "text_mask": text_mask.unsqueeze(0).to(dev),
            "audio_mask": audio_mask.unsqueeze(0).to(dev),
        }

    def _forward_prefill(self, inputs_embeds: torch.Tensor, dev: Any) -> torch.Tensor:
        """Prefill: build combined embeds, run base_lm + residual_lm + first diffusion.

        Uses the same path as native ``VoxCPM2Model._inference`` so zero-shot,
        voice cloning (reference), continuation, and ref_continuation modes
        all share the same code.
        """
        tts = self.tts
        dtype = self._side_dtype
        text = getattr(self, "_prefill_text", None)
        if text is None:
            # Fallback (should not hit at runtime; preprocess sets this)
            text = ""

        inputs = self._build_prefill_inputs(text, dev)
        text_token = inputs["text_token"]
        feat = inputs["audio_feat"]
        text_mask = inputs["text_mask"]
        feat_mask = inputs["audio_mask"]

        # Compose combined_embed exactly like native _inference
        feat_embed = tts.feat_encoder(feat)
        feat_embed = tts.enc_to_lm_proj(feat_embed)
        scale_emb = tts.config.lm_config.scale_emb if tts.config.lm_config.use_mup else 1.0
        text_embed = tts.base_lm.embed_tokens(text_token) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        # last audio patch becomes initial prefix_feat_cond (zeros for zero-shot,
        # last reference/prompt patch for voice clone / continuation)
        prefix_feat_cond = (
            feat[:, -1, ...]
            if feat.shape[1] > 0
            else torch.zeros(1, tts.patch_size, tts.feat_dim, device=dev, dtype=dtype)
        )

        # Base LM prefill
        tts.base_lm.setup_cache(1, 4096, dev, dtype)
        enc_out, enc_kv = tts.base_lm(inputs_embeds=combined_embed, is_causal=True)
        tts.base_lm.kv_cache.fill_caches(enc_kv)

        # FSQ: identity on text positions, quantized on audio positions
        enc_outputs = tts.fsq_layer(enc_out) * feat_mask.unsqueeze(-1) + enc_out * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]  # [1, H]

        logger.info(
            "PREFILL: enc shape=%s last_norm=%.4f",
            enc_outputs.shape,
            lm_hidden.norm().item(),
        )

        # Residual LM prefill
        tts.residual_lm.setup_cache(1, 4096, dev, dtype)
        residual_input = tts.fusion_concat_proj(torch.cat([enc_outputs, feat_mask.unsqueeze(-1) * feat_embed], dim=-1))
        res_out, res_kv = tts.residual_lm(inputs_embeds=residual_input, is_causal=True)
        tts.residual_lm.kv_cache.fill_caches(res_kv)
        residual_hidden = res_out[:, -1, :]  # [1, H]

        # Precompute stop logits for first compute_logits call
        stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden)))
        self._precomputed_stop_logits = stop_logits.detach()
        logger.info("PREFILL stop: %s", stop_logits[0].tolist())

        # First diffusion step
        dit_h = torch.cat(
            [
                tts.lm_to_dit_proj(lm_hidden),
                tts.res_to_dit_proj(residual_hidden),
            ],
            dim=-1,
        )
        pred_feat = tts.feat_decoder(
            mu=dit_h,
            patch_size=tts.patch_size,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).transpose(1, 2)  # [1, P, D]

        with torch.no_grad():
            curr_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)

        # Store state for decode steps
        self._curr_embed_for_next = curr_embed.detach()
        self._prev_feat_embed = curr_embed.detach()
        self._curr_prefix_feat_cond = pred_feat[0].detach()
        self._last_audio_patch = pred_feat.reshape(1, -1).detach().cpu().float()

        logger.info(
            "PREFILL patch: norm=%.4f first3=%s",
            pred_feat.norm().item(),
            pred_feat[0, 0, :3].tolist(),
        )

        return lm_hidden.to(dtype)

    def _forward_decode(self, inputs_embeds: torch.Tensor | None, dev: Any) -> torch.Tensor:
        """Decode step: base_lm → FSQ → residual_lm → diffusion."""
        tts = self.tts
        dtype = self._side_dtype

        # 1. Base LM step with curr_embed from previous diffusion
        curr_embed = self._curr_embed_for_next.to(dev, dtype=dtype)
        if curr_embed.ndim == 2:
            curr_embed_3d = curr_embed.unsqueeze(0)  # [1, 1, H]
        else:
            curr_embed_3d = curr_embed

        step_pos = torch.tensor([tts.base_lm.kv_cache.step()], device=dev)
        new_hidden = tts.base_lm.forward_step(curr_embed_3d[:, 0, :], step_pos).clone()

        # 2. FSQ
        new_lm_hidden = tts.fsq_layer(new_hidden)
        if new_lm_hidden.ndim == 1:
            new_lm_hidden = new_lm_hidden.unsqueeze(0)

        # 3. Residual LM step
        prev_fe = self._prev_feat_embed.to(dtype)
        if prev_fe.ndim == 1:
            prev_fe = prev_fe.unsqueeze(0)
        res_input = tts.fusion_concat_proj(torch.cat([new_lm_hidden, prev_fe], dim=-1))
        res_step_pos = torch.tensor([tts.residual_lm.kv_cache.step()], device=dev)
        new_res_hidden = tts.residual_lm.forward_step(res_input, res_step_pos).clone()
        if new_res_hidden.ndim == 1:
            new_res_hidden = new_res_hidden.unsqueeze(0)

        # 4. Diffusion
        p = self._patch_size
        pfc = self._curr_prefix_feat_cond.to(dtype).unsqueeze(0)

        dit_h = torch.cat(
            [
                tts.lm_to_dit_proj(new_lm_hidden),
                tts.res_to_dit_proj(new_res_hidden),
            ],
            dim=-1,
        )
        pred_feat = tts.feat_decoder(
            mu=dit_h,
            patch_size=p,
            cond=pfc.transpose(1, 2).contiguous(),
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).transpose(1, 2)  # [1, P, D]

        # 5. feat_encoder → curr_embed
        with torch.no_grad():
            curr_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)

        # 6. Stop logits
        stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(new_lm_hidden)))
        self._precomputed_stop_logits = stop_logits.detach()

        # 7. Store state
        self._curr_embed_for_next = curr_embed.detach()
        self._prev_feat_embed = curr_embed.detach()
        self._curr_prefix_feat_cond = pred_feat[0].detach()
        self._last_audio_patch = pred_feat.reshape(1, -1).detach().cpu().float()

        return new_lm_hidden[-1:].detach()

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        precomputed = getattr(self, "_precomputed_stop_logits", None)
        if precomputed is not None:
            self._precomputed_stop_logits = None
            raw_logits = precomputed[: hidden_states.shape[0]]
        else:
            # Fallback for warmup
            bsz = hidden_states.shape[0]
            raw_logits = torch.zeros(bsz, 2, device=hidden_states.device)
            raw_logits[:, 0] = 1.0  # continue

        bsz = raw_logits.shape[0]
        full_logits = torch.full(
            (bsz, self.config.vocab_size),
            float("-inf"),
            device=raw_logits.device,
            dtype=raw_logits.dtype,
        )
        full_logits[:, 0] = raw_logits[:, 0]  # continue
        full_logits[:, 1] = raw_logits[:, 1]  # stop
        return full_logits

    # -------------------- Omni output --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        hidden = model_outputs
        patch = getattr(self, "_last_audio_patch", None)
        mm: dict[str, Any] = {}

        if patch is not None:
            self._last_audio_patch = None
            self._accumulated_patches.append(patch.clone())

        # Decode all accumulated patches → full audio waveform.
        # TODO: implement sliding-window VAE decode (nanovllm pattern)
        # for O(1) per-step streaming instead of O(N) re-decode.
        if self._accumulated_patches:
            all_p = torch.cat(self._accumulated_patches, dim=0)
            d = self._feat_dim
            from einops import rearrange

            feat = rearrange(all_p.float().reshape(1, -1, d), "b t d -> b d t")
            with torch.no_grad():
                audio = self.tts.audio_vae.decode(feat.to(self._device)).reshape(-1).detach().cpu().float()

            mm["model_outputs"] = [audio]
            mm["sr"] = [torch.tensor(48000, dtype=torch.int32)]

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=mm,
        )

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional_information = info_dict.get("additional_information")
        if isinstance(additional_information, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional_information.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        dev = input_ids.device

        if span_len > 1:
            # ---- Prefill ----
            # Decode the text from input_ids for native-matching tokenization.
            # Speech API tokenizes with BOS; we use the detokenized string so
            # native's ``text_tokenizer`` produces the exact same tokens as
            # ``generate()``.
            ids = input_ids.tolist()
            if ids and ids[0] == self.config.bos_token_id:
                ids = ids[1:]
            text = self.tts.text_tokenizer.tokenizer.decode(ids, skip_special_tokens=True)
            self._prefill_text = text

            # Voice clone / continuation: build prompt cache from info_dict.
            ref_audio = info_dict.get("reference_audio") or info_dict.get("ref_audio")
            prompt_audio = info_dict.get("prompt_audio")
            prompt_text = info_dict.get("prompt_text")
            if isinstance(ref_audio, list):
                ref_audio = ref_audio[0] if ref_audio else None
            if isinstance(prompt_audio, list):
                prompt_audio = prompt_audio[0] if prompt_audio else None
            if isinstance(prompt_text, list):
                prompt_text = prompt_text[0] if prompt_text else None

            self._prompt_cache = None
            if ref_audio or (prompt_audio and prompt_text):
                try:
                    self._prompt_cache = self.tts.build_prompt_cache(
                        prompt_text=prompt_text,
                        prompt_wav_path=prompt_audio,
                        reference_wav_path=ref_audio,
                    )
                except Exception as e:
                    logger.warning("build_prompt_cache failed: %s; falling back to zero-shot", e)
                    self._prompt_cache = None

            # Reset per-request state (fresh generation)
            self._accumulated_patches = []
            if hasattr(self, "_prev_feat_embed"):
                del self._prev_feat_embed
            if hasattr(self, "_curr_embed_for_next"):
                del self._curr_embed_for_next

            # Store info for forward
            self._current_step_infos = [{"is_prefill": True}]

            # The scaffold model still needs embeddings sized to span_len for
            # its warmup/attention bookkeeping. Native modules use the full
            # (potentially longer) sequence internally. Pass zeros — scaffold
            # output is discarded.
            embeds = torch.zeros(
                span_len,
                self.config.hidden_size,
                device=dev,
                dtype=self._side_dtype,
            )

            return input_ids, embeds, {}

        # ---- Decode ----
        curr_embed = getattr(self, "_curr_embed_for_next", None)
        if curr_embed is not None:
            inputs_embeds = curr_embed.to(dev, dtype=self._side_dtype).reshape(1, -1)
        else:
            inputs_embeds = torch.zeros(
                1,
                self.config.hidden_size,
                device=dev,
                dtype=self._side_dtype,
            )

        self._current_step_infos = [{}]
        return input_ids, inputs_embeds, {}

    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        return {}

    # -------------------- Weight loading --------------------

    # Weight mapping for vllm scaffold
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"base_lm.": "model."})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load scaffold weights via vllm + native model for computation."""

        # Filter: only pass base_lm weights to the scaffold
        def _base_lm_only(ws):
            for name, tensor in ws:
                if name.startswith("base_lm."):
                    yield name, tensor

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_base_lm_only(weights), mapper=self.hf_to_vllm_mapper)

        # Load the full native model for actual computation
        model_path = self.vllm_config.model_config.model
        VoxCPM = import_voxcpm2_core()
        native = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
        self._tts = native.tts_model.to("cuda")
        self._side_dtype = self._tts.fusion_concat_proj.weight.dtype
        self._device = "cuda"

        self._patch_size = self._tts.patch_size
        self._feat_dim = self._tts.feat_dim

        logger.info(
            "Loaded native VoxCPM2 (patch_size=%d, feat_dim=%d, dtype=%s)",
            self._patch_size,
            self._feat_dim,
            self._side_dtype,
        )
        return loaded
