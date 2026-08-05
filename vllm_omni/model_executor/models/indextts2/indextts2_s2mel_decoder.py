# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IndexTTS2 Stage 1: S2Mel decoder + BigVGAN vocoder.

Receives mel_codes + latent from Stage 0 (GPT AR talker), runs flow matching
to synthesize mel spectrogram, then BigVGAN to produce waveform audio.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.model_loader.hub_prefetch import _repo_prefetch_lock
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_indextts2 import IndexTTS2Config
from .preprocess_utils import load_semantic_codec, resolve_model_file
from .s2mel.modules.commons import AttrDict, MyModel

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy loaders for external models
# ---------------------------------------------------------------------------

_bigvgan_models: dict[tuple[str, str], nn.Module] = {}


def _patch_bigvgan_compat(cls):
    """Monkey-patch BigVGAN._from_pretrained for huggingface_hub>=1.0 compat."""
    if getattr(cls._from_pretrained, "_hf1_patched", False):
        return
    orig = cls._from_pretrained.__func__

    @classmethod
    def _compat(klass, *, proxies=None, resume_download=False, **kw):
        return orig(klass, proxies=proxies, resume_download=resume_download, **kw)

    _compat._hf1_patched = True
    cls._from_pretrained = _compat


def _strip_weight_norm(root: nn.Module) -> list[str]:
    """Bake legacy weight_norm (weight_g/weight_v) into plain weights.

    The DiT final-layer WaveNet (and any other S2Mel submodule) keeps
    torch.nn.utils.weight_norm forward pre-hooks after loading, so every
    CFM estimator step recomputes the normalization (~217 calls/request
    in profiling). Inference never updates these weights, so we fold the
    parametrization once at load time. Hooks live on the innermost conv
    (e.g. SConv1d.conv.conv), hence the generic module walk instead of
    the per-class remove_weight_norm() helpers.

    Returns the qualified names (relative to ``root``) of the plain
    weight parameters created by the fold, so callers can mark them as
    checkpoint-initialized for vLLM's loader validation.
    """
    from torch.nn.utils import remove_weight_norm

    folded: list[str] = []
    for mod_name, module in root.named_modules():
        for hook in list(module._forward_pre_hooks.values()):
            if hook.__class__.__name__ == "WeightNorm":
                remove_weight_norm(module, name=hook.name)
                folded.append(f"{mod_name}.{hook.name}" if mod_name else hook.name)
    return folded


def _load_bigvgan(vocoder_name: str, device: torch.device, dtype: torch.dtype = torch.float32):
    cache_key = (vocoder_name, str(device), str(dtype))
    if cache_key in _bigvgan_models:
        return _bigvgan_models[cache_key]

    lock = _repo_prefetch_lock(vocoder_name) if not os.path.isdir(vocoder_name) else None
    if lock is not None:
        lock.__enter__()
    try:
        try:
            from .s2mel.modules import bigvgan as bigvgan_mod

            _patch_bigvgan_compat(bigvgan_mod.BigVGAN)
            bigvgan_model = bigvgan_mod.BigVGAN.from_pretrained(vocoder_name)
        except (ImportError, ModuleNotFoundError):
            import bigvgan

            _patch_bigvgan_compat(bigvgan.BigVGAN)
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_name)
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    # One-time weight cast (vs per-call autocast, which re-casts every conv's
    # fp32 weights on each forward and is slower than fp32 on BigVGAN).
    bigvgan_model = bigvgan_model.to(device=device, dtype=dtype).eval()
    bigvgan_model.remove_weight_norm()
    for p in bigvgan_model.parameters():
        p.requires_grad_(False)
    _bigvgan_models[cache_key] = bigvgan_model
    return bigvgan_model


# ---------------------------------------------------------------------------
# Stage 1 Decoder
# ---------------------------------------------------------------------------


class IndexTTS2S2MelDecoder(nn.Module):
    """S2Mel + BigVGAN decoder for IndexTTS2 Stage 1.

    Receives from Stage 0:
      - mel_codes: [T] mel code sequence
      - latent: [T, 1280] hidden states from GPT AR
      - S_ref: speaker semantic embeddings (from Wav2Vec2 → RepCodec)
      - ref_mel: [80, T_ref] reference mel spectrogram
      - style: [192] CAMPPlus style vector
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.config: IndexTTS2Config = vllm_config.model_config.hf_config  # type: ignore[assignment]

        # --- Flags for vLLM-Omni framework ---
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # --- Build S2Mel model (CFM + LengthRegulator + gpt_layer) ---
        s2mel_cfg = self.config.s2mel
        s2mel_args = AttrDict(s2mel_cfg)
        # Ensure nested dicts are also AttrDicts
        for key in ["DiT", "wavenet", "length_regulator", "style_encoder", "preprocess_params"]:
            if key in s2mel_args and isinstance(s2mel_args[key], dict):
                s2mel_args[key] = AttrDict(s2mel_args[key])

        self.s2mel = MyModel(s2mel_args, use_emovec=False, use_gpt_latent=True)

        # Diffusion config — configurable via deploy YAML hf_overrides
        self.diffusion_steps: int = getattr(self.config, "diffusion_steps", 25)
        self.inference_cfg_rate: float = getattr(self.config, "inference_cfg_rate", 0.7)
        self.mel_code_to_frame_ratio: float = 1.72

        # Run the DiT estimator under bf16 autocast (flash attention + faster
        # GEMMs); the CFM Euler solver stays float32. Disable via deploy YAML
        # hf_overrides `s2mel_dit_bf16: false` if quality regresses.
        self.s2mel_dit_bf16: bool = getattr(self.config, "s2mel_dit_bf16", True)
        # Run BigVGAN vocoding under bf16 autocast (conv-heavy, ~2x faster).
        # Disable via deploy YAML hf_overrides `s2mel_vocoder_bf16: false`.
        self.s2mel_vocoder_bf16: bool = getattr(self.config, "s2mel_vocoder_bf16", True)
        # Capture BigVGAN into bucketed CUDA graphs (collapses the Snake/conv
        # small-op launch storm). Enable via `s2mel_vocoder_cuda_graph: true`.
        self.s2mel_vocoder_cuda_graph: bool = getattr(self.config, "s2mel_vocoder_cuda_graph", False)
        self._vocoder_graph: Any = None
        # Capture the DiT transformer core into per-shape CUDA graphs.
        # Enable via `s2mel_dit_cuda_graph: true`.
        self.s2mel_dit_cuda_graph: bool = getattr(self.config, "s2mel_dit_cuda_graph", False)
        self.s2mel_dit_cuda_graph_max_graphs: int = int(getattr(self.config, "s2mel_dit_cuda_graph_max_graphs", 8))
        self.s2mel_vocoder_capture_sizes: list[int] | None = getattr(self.config, "s2mel_vocoder_capture_sizes", None)
        self.s2mel_vocoder_compile_shapes: list[int] | None = getattr(self.config, "s2mel_vocoder_compile_shapes", None)
        logger.info(
            "[S2Mel] dit_bf16=%s, vocoder_bf16=%s, "
            "vocoder_cuda_graph=%s, dit_cuda_graph=%s, dit_graph_lru=%d, vocoder_buckets=%s, "
            "vocoder_compile_shapes=%s",
            self.s2mel_dit_bf16,
            self.s2mel_vocoder_bf16,
            self.s2mel_vocoder_cuda_graph,
            self.s2mel_dit_cuda_graph,
            self.s2mel_dit_cuda_graph_max_graphs,
            self.s2mel_vocoder_capture_sizes,
            self.s2mel_vocoder_compile_shapes,
        )

    # ------------------------------------------------------------------
    # vLLM hooks
    # ------------------------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | OmniOutput:
        """Run S2Mel flow matching + BigVGAN vocoding."""
        model_intermediate_buffer = kwargs.get("model_intermediate_buffer")
        if model_intermediate_buffer is None:
            model_intermediate_buffer = runtime_additional_information
        if runtime_additional_information is not None and "model_intermediate_buffer" not in kwargs:
            logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")

        request_infos: list[dict[str, Any]] = []
        if model_intermediate_buffer:
            request_infos = [info for info in model_intermediate_buffer if isinstance(info, dict)]
        additional_information: dict[str, Any] = request_infos[0] if request_infos else {}

        device = input_ids.device
        model_dtype = self.s2mel.models["gpt_layer"][0].weight.dtype

        logger.debug(
            "[S2Mel forward] model_intermediate_buffer len=%d, keys=%s",
            len(model_intermediate_buffer) if model_intermediate_buffer else 0,
            list(additional_information.keys()) if additional_information else [],
        )
        for k, v in additional_information.items():
            if isinstance(v, torch.Tensor):
                logger.debug("[S2Mel forward] %s: shape=%s, dtype=%s", k, v.shape, v.dtype)

        # Extract Stage 0 outputs and pad multiple ready Stage-1 requests into
        # one S2Mel batch. The runner can split list-valued multimodal outputs
        # back to per-request payloads, so never silently drop batched items.
        if len(request_infos) > 1:
            mel_codes, latent, code_lens, code_lens_list, s_ref, ref_mel, style = self._build_batched_inputs(
                request_infos,
                device=device,
                model_dtype=model_dtype,
            )
            code_lens_raw = code_lens
        else:
            # Extract Stage 0 outputs and cast to model dtype
            latent = self._get_tensor(additional_information, "latent", device, model_dtype)
            mel_codes = self._get_tensor(additional_information, "mel_codes", device)
            code_lens_raw = additional_information.get("code_lens")
            s_ref = self._get_tensor(additional_information, "S_ref", device)
            ref_mel = self._get_tensor(additional_information, "ref_mel", device, model_dtype)
            style = self._get_tensor(additional_information, "style", device, model_dtype)

        if mel_codes is None or latent is None:
            logger.warning("S2Mel decoder received empty mel_codes or latent")
            return OmniOutput(
                text_hidden_states=torch.zeros(1, device=device),
                multimodal_outputs={"audio": torch.zeros(1, device=device), "sr": 22050},
            )

        logger.debug(
            "[S2Mel] extracted tensors — mel_codes=%s latent=%s code_lens=%s s_ref=%s ref_mel=%s style=%s",
            mel_codes.shape if mel_codes is not None else None,
            latent.shape if latent is not None else None,
            code_lens_raw,
            s_ref.shape if isinstance(s_ref, torch.Tensor) else None,
            ref_mel.shape if isinstance(ref_mel, torch.Tensor) else None,
            style.shape if isinstance(style, torch.Tensor) else None,
        )

        # Ensure batch dimension
        if mel_codes.ndim == 1:
            mel_codes = mel_codes.unsqueeze(0)
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)

        # Compute code_lens
        if code_lens_raw is not None:
            if isinstance(code_lens_raw, torch.Tensor):
                if "code_lens_list" not in locals():
                    code_lens_list = [int(x) for x in code_lens_raw.detach().cpu().reshape(-1).tolist()]
                code_lens = code_lens_raw.to(device=device, dtype=torch.long)
            else:
                code_lens_list = [int(x) for x in code_lens_raw]
                code_lens = torch.tensor(code_lens_list, device=device, dtype=torch.long)
        else:
            # Strip stop token (8193) and compute actual length
            stop_token = self.config.gpt.get("stop_mel_token", 8193)
            code_lens_list = []
            for i in range(mel_codes.shape[0]):
                stop_mask = (mel_codes[i] == stop_token).nonzero(as_tuple=False)
                if stop_mask.numel() > 0:
                    code_lens_list.append(stop_mask[0].item())
                else:
                    code_lens_list.append(mel_codes.shape[1])
            code_lens = torch.tensor(code_lens_list, device=device, dtype=torch.long)

        # Trim to actual length
        max_len = max(code_lens_list)
        mel_codes = mel_codes[:, :max_len]
        latent = latent[:, :max_len, :]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[S2Mel] after trim — mel_codes=%s latent=%s code_lens=%s max_len=%d",
                mel_codes.shape,
                latent.shape,
                code_lens.tolist(),
                max_len,
            )

        # --- S2Mel pipeline ---
        # 1. Project GPT latent: 1280 → 1024
        latent = latent.to(device=device, dtype=model_dtype)
        latent = self.s2mel.forward_gpt(latent).to(dtype=model_dtype)  # [B, T, 1024]
        logger.debug("[S2Mel] step1 forward_gpt → latent=%s dtype=%s", latent.shape, latent.dtype)

        # 2. Embed mel codes via semantic codec vq2emb
        semantic_codec = load_semantic_codec(self.model_path, self.config.semantic_codec, device)
        codebook_size = self.config.semantic_codec.get("codebook_size", 8192)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "mel_codes debug: shape=%s, min=%d, max=%d",
                mel_codes.shape,
                mel_codes.min().item(),
                mel_codes.max().item(),
            )
        if getattr(self.config, "s2mel_validate_code_range", False) and torch.any(
            (mel_codes < 0) | (mel_codes >= codebook_size)
        ):
            raise ValueError(f"IndexTTS2 generated mel code outside semantic codebook range [0, {codebook_size - 1}]")
        with torch.no_grad():
            S_infer = semantic_codec.quantizer.vq2emb(mel_codes.unsqueeze(1))  # [B, T, 1024]
        S_infer = S_infer.transpose(1, 2).to(dtype=model_dtype)  # [B, T, 1024]
        S_infer = S_infer + latent
        logger.debug("[S2Mel] step2 vq2emb → S_infer=%s dtype=%s", S_infer.shape, S_infer.dtype)

        # 3. Length regulate: codes → mel frames
        target_lengths_list = [int(length * self.mel_code_to_frame_ratio) for length in code_lens_list]
        target_lengths = torch.tensor(target_lengths_list, device=device, dtype=torch.long)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[S2Mel] step3 length_regulator target_lengths=%s", target_lengths_list)
        cond = self.s2mel.models["length_regulator"](S_infer, ylens=target_lengths, n_quantizers=3, f0=None)[
            0
        ]  # [B, T_mel, 512]
        logger.debug("[S2Mel] step3 length_regulator → cond=%s", cond.shape)

        # 4. Reference prompt conditioning
        if s_ref is not None and ref_mel is not None:
            if ref_mel.ndim == 2:
                ref_mel = ref_mel.unsqueeze(0)

            if s_ref.ndim == 3:
                # Official infer_v2.py uses the quantized embedding returned by
                # semantic_codec.quantize() directly as S_ref.
                s_ref_emb = s_ref.to(device=device, dtype=model_dtype)
            else:
                # Backward-compatible path for older payloads that carried
                # codebook indices instead of quantized embeddings.
                s_ref_codes = s_ref.long()
                if s_ref_codes.ndim == 1:
                    s_ref_codes = s_ref_codes.unsqueeze(0)  # [1, T]
                codebook_size = self.config.semantic_codec.get("codebook_size", 8192)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "S_ref debug: shape=%s, min=%d, max=%d, codebook_size=%d",
                        s_ref_codes.shape,
                        s_ref_codes.min().item(),
                        s_ref_codes.max().item(),
                        codebook_size,
                    )
                s_ref_codes = s_ref_codes.clamp(0, codebook_size - 1)
                with torch.no_grad():
                    s_ref_emb = semantic_codec.quantizer.vq2emb(s_ref_codes.unsqueeze(1))  # [B, 1024, T]
                s_ref_emb = s_ref_emb.transpose(1, 2).to(dtype=model_dtype)  # [B, T, 1024]

            ref_target_lengths = self._infer_ref_lengths(ref_mel)
            prompt_condition = self.s2mel.models["length_regulator"](
                s_ref_emb, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            logger.debug(
                "[S2Mel] step4 ref conditioning — s_ref_emb=%s prompt_condition=%s cat_condition=%s",
                s_ref_emb.shape,
                prompt_condition.shape,
                cat_condition.shape,
            )
        else:
            prompt_condition = None
            cat_condition = cond
            ref_mel = torch.zeros(cond.shape[0], 80, 0, device=device, dtype=model_dtype)

        if style is None:
            style = torch.zeros(cat_condition.shape[0], 192, device=device, dtype=model_dtype)
        if style.ndim == 1:
            style = style.unsqueeze(0)

        logger.debug(
            "[S2Mel] step5 CFM input — cat_condition=%s ref_mel=%s style=%s steps=%d cfg_rate=%.2f",
            cat_condition.shape,
            ref_mel.shape,
            style.shape,
            self.diffusion_steps,
            self.inference_cfg_rate,
        )

        # 5. Flow matching inference (Euler ODE steps, CFG rate configurable)
        # CFM ODE solver runs in float32 — fp16 Euler steps diverge to noise.
        # The DiT estimator itself may run under bf16 autocast (see below).
        cfm = self.s2mel.models["cfm"]
        cfm.estimator_autocast_dtype = torch.bfloat16 if self.s2mel_dit_bf16 else None
        cond = cond.float()
        if prompt_condition is not None:
            prompt_condition = prompt_condition.float()
        cat_condition = cat_condition.float()
        ref_mel = ref_mel.float()
        style = style.float()

        if prompt_condition is None:
            ref_lengths = torch.zeros(cat_condition.shape[0], device=device, dtype=torch.long)
        else:
            ref_lengths = self._infer_ref_lengths(ref_mel)
        ref_lengths_list = [int(length) for length in ref_lengths.tolist()]
        if len(ref_lengths_list) == 1 and cat_condition.shape[0] > 1:
            ref_lengths_list = ref_lengths_list * int(cat_condition.shape[0])
            ref_lengths = ref_lengths.expand(int(cat_condition.shape[0]))
        max_ref_len = int(ref_mel.size(-1))
        ref_len = ref_lengths_list[0] if ref_lengths_list else max_ref_len
        x_lens = target_lengths + ref_lengths

        estimator = cfm.estimator
        required_cache_batch = max(2, int(cat_condition.shape[0]) * (2 if self.inference_cfg_rate > 0 else 1))
        if (
            estimator.transformer.freqs_cis is None
            or getattr(estimator.transformer, "max_batch_size", -1) < required_cache_batch
        ):
            estimator.setup_caches(max_batch_size=required_cache_batch, max_seq_length=16384)
            logger.info("[S2Mel] DiT caches initialized (freqs_cis + causal_mask)")

        cat_len = int(cat_condition.size(1))
        x_lens_list = [target_len + ref_len_i for target_len, ref_len_i in zip(target_lengths_list, ref_lengths_list)]
        same_ref_len = len(set(ref_lengths_list)) <= 1
        same_target_len = len(set(target_lengths_list)) <= 1
        dit_graph_full_mask = same_ref_len and same_target_len and all(length == cat_len for length in x_lens_list)
        if dit_graph_full_mask:
            self._enable_dit_cuda_graph(estimator)
            with torch.no_grad():
                mel = cfm.inference(
                    cat_condition,
                    x_lens,
                    ref_mel,
                    style,
                    None,  # f0
                    self.diffusion_steps,
                    inference_cfg_rate=self.inference_cfg_rate,
                )
            # Strip reference portion
            mel = mel[:, :, ref_len:]
        else:
            # Batched Stage-1 requests are padded to a shared length. The DiT
            # CUDA graph fast path assumes a full mask, so rebuild each request
            # with its exact reference/target lengths before enabling the graph.
            mels_list: list[torch.Tensor] = []
            for i in range(cat_condition.shape[0]):
                self._enable_dit_cuda_graph(estimator)
                ref_len_i = ref_lengths_list[i]
                target_len_i = target_lengths_list[i]
                if prompt_condition is not None:
                    prompt_i = prompt_condition[i : i + 1, :ref_len_i]
                    target_i = cond[i : i + 1, :target_len_i]
                    cond_i = torch.cat([prompt_i, target_i], dim=1)
                else:
                    cond_i = cat_condition[i : i + 1, :target_len_i]
                x_lens_i = torch.tensor([ref_len_i + target_len_i], device=device, dtype=torch.long)
                ref_mel_i = ref_mel[i : i + 1, :, :ref_len_i] if ref_mel.ndim == 3 else ref_mel[:, :ref_len_i]
                style_i = style[i : i + 1] if style.ndim == 2 else style
                with torch.no_grad():
                    mel_i = cfm.inference(
                        cond_i,
                        x_lens_i,
                        ref_mel_i,
                        style_i,
                        None,
                        self.diffusion_steps,
                        inference_cfg_rate=self.inference_cfg_rate,
                    )
                mels_list.append(mel_i[0, :, ref_len_i:])
            mel = mels_list
        dit_graph_info = getattr(getattr(estimator, "_cuda_graph_runner", None), "last_call_info", None)
        mel_shape = [tuple(m.shape) for m in mel] if isinstance(mel, list) else tuple(mel.shape)
        logger.debug("[S2Mel] step5 CFM done → mel=%s (stripped ref %d frames)", mel_shape, ref_len)

        # 6. BigVGAN vocoding
        vocoder_cfg = self.config.vocoder
        vocoder_name = vocoder_cfg.get("name", "nvidia/bigvgan_v2_22khz_80band_256x")
        voc_dtype = torch.bfloat16 if (self.s2mel_vocoder_bf16 and device.type == "cuda") else torch.float32
        bigvgan = _load_bigvgan(vocoder_name, device, voc_dtype)
        vocode = self._get_vocoder_graph(bigvgan, device, voc_dtype)
        upsample_factor = int(math.prod(bigvgan.h.upsample_rates))
        with torch.no_grad():
            wavs: list[torch.Tensor] = []
            vocoder_infos: list[dict[str, Any]] = []
            if isinstance(mel, list):
                # Per-request CFM output: variable-length mels. Batch them
                # for BigVGAN by padding to the max length.
                batch_size = len(mel)
                if batch_size == 1:
                    mel_i = mel[0:1].to(voc_dtype) if mel[0].ndim == 3 else mel[0].unsqueeze(0).to(voc_dtype)
                    wav_i = vocode(mel_i).float().squeeze(0).squeeze(0)
                    wavs.append(torch.clamp(wav_i, -1.0, 1.0))
                    last_call_info = getattr(vocode, "last_call_info", None)
                    if isinstance(last_call_info, dict):
                        vocoder_infos.append(dict(last_call_info))
                else:
                    for i in range(batch_size):
                        mel_i = mel[i].unsqueeze(0).to(voc_dtype) if mel[i].ndim == 2 else mel[i : i + 1].to(voc_dtype)
                        wav_i = vocode(mel_i).float().squeeze(0).squeeze(0)
                        wavs.append(torch.clamp(wav_i, -1.0, 1.0))
                        last_call_info = getattr(vocode, "last_call_info", None)
                        if isinstance(last_call_info, dict):
                            vocoder_infos.append(dict(last_call_info))
            else:
                batch_size = mel.shape[0]
                if batch_size == 1:
                    mel_i = mel[:, :, : target_lengths_list[0]].to(voc_dtype)
                    wav_i = vocode(mel_i).float().squeeze(0).squeeze(0)
                    wavs.append(torch.clamp(wav_i, -1.0, 1.0))
                    last_call_info = getattr(vocode, "last_call_info", None)
                    if isinstance(last_call_info, dict):
                        vocoder_infos.append(dict(last_call_info))
                else:
                    for i in range(batch_size):
                        mel_i = mel[i : i + 1, :, : target_lengths_list[i]].to(voc_dtype)
                        wav_i = vocode(mel_i).float().squeeze(0).squeeze(0)
                        wavs.append(torch.clamp(wav_i, -1.0, 1.0))
                        last_call_info = getattr(vocode, "last_call_info", None)
                        if isinstance(last_call_info, dict):
                            vocoder_infos.append(dict(last_call_info))
        wav: torch.Tensor | list[torch.Tensor] = wavs[0] if len(wavs) == 1 else wavs

        # Keep the public vLLM-Omni audio contract as normalized float. This is
        # numerically equivalent to official infer_v2.py before its int16 save
        # step, while avoiding double scaling in OpenAI/soundfile encoders.
        if isinstance(wav, torch.Tensor) and wav.ndim == 0:
            wav = wav.unsqueeze(0)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[S2Mel] step6 BigVGAN done → wav=%s min=%.1f max=%.1f sr=22050",
                wav.shape if isinstance(wav, torch.Tensor) else [w.shape for w in wav],
                wav.min().item() if isinstance(wav, torch.Tensor) else min(w.min().item() for w in wav),
                wav.max().item() if isinstance(wav, torch.Tensor) else max(w.max().item() for w in wav),
            )

        audio_cpu = [w.cpu() for w in wav] if isinstance(wav, list) else wav.cpu()

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "audio": audio_cpu,
                "sr": torch.tensor(22050, dtype=torch.int32),
            },
        )

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_batched_inputs(
        self,
        infos: list[dict[str, Any]],
        *,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        stop_token = self.config.gpt.get("stop_mel_token", 8193)
        mel_items: list[torch.Tensor] = []
        latent_items: list[torch.Tensor] = []
        code_lens: list[int] = []
        s_ref_items: list[torch.Tensor | None] = []
        ref_mel_items: list[torch.Tensor | None] = []
        style_items: list[torch.Tensor | None] = []

        for idx, info in enumerate(infos):
            mel = self._get_tensor(info, "mel_codes", device)
            latent = self._get_tensor(info, "latent", device, model_dtype)
            if mel is None or latent is None:
                raise ValueError(f"S2Mel batched input {idx} is missing mel_codes or latent")

            mel = mel.to(device=device, dtype=torch.long).reshape(-1)
            if latent.ndim == 3 and latent.shape[0] == 1:
                latent = latent[0]
            elif latent.ndim != 2:
                latent = latent.reshape(-1, latent.shape[-1])

            raw_len = info.get("code_lens")
            if isinstance(raw_len, torch.Tensor) and raw_len.numel() > 0:
                code_len = int(raw_len.reshape(-1)[0].item())
            elif raw_len is not None:
                code_len = int(raw_len[0] if isinstance(raw_len, list) else raw_len)
            else:
                stop_mask = (mel == stop_token).nonzero(as_tuple=False)
                code_len = int(stop_mask[0].item()) if stop_mask.numel() > 0 else int(mel.shape[0])
            code_len = max(0, min(code_len, int(mel.shape[0]), int(latent.shape[0])))
            mel_items.append(mel[:code_len])
            latent_items.append(latent[:code_len])
            code_lens.append(code_len)

            s_ref = self._get_tensor(info, "S_ref", device)
            if isinstance(s_ref, torch.Tensor) and s_ref.ndim == 3 and s_ref.shape[0] == 1:
                s_ref = s_ref[0]
            s_ref_items.append(s_ref)

            ref_mel = self._get_tensor(info, "ref_mel", device, model_dtype)
            if isinstance(ref_mel, torch.Tensor) and ref_mel.ndim == 3 and ref_mel.shape[0] == 1:
                ref_mel = ref_mel[0]
            ref_mel_items.append(ref_mel)

            style = self._get_tensor(info, "style", device, model_dtype)
            if isinstance(style, torch.Tensor) and style.ndim == 2 and style.shape[0] == 1:
                style = style[0]
            style_items.append(style)

        max_code_len = max(code_lens) if code_lens else 0
        latent_dim = latent_items[0].shape[-1]
        # Padding reaches vq2emb before length masking, so it must be a valid
        # semantic-code index. Actual sequence lengths are still carried by
        # code_lens/target_lengths; stop_token (8193) is outside the codebook.
        mel_batch = torch.zeros((len(mel_items), max_code_len), dtype=torch.long, device=device)
        latent_batch = torch.zeros(len(latent_items), max_code_len, latent_dim, device=device, dtype=model_dtype)
        for i, (mel, latent) in enumerate(zip(mel_items, latent_items)):
            mel_batch[i, : mel.shape[0]] = mel
            latent_batch[i, : latent.shape[0]] = latent.to(dtype=model_dtype)

        s_ref_batch = self._stack_optional_sequence(s_ref_items, device=device, dtype=model_dtype)
        ref_mel_batch = self._stack_optional_ref_mel(ref_mel_items, device=device, dtype=model_dtype)
        style_batch = self._stack_optional_style(style_items, device=device, dtype=model_dtype)
        return (
            mel_batch,
            latent_batch,
            torch.tensor(code_lens, device=device, dtype=torch.long),
            code_lens,
            s_ref_batch,
            ref_mel_batch,
            style_batch,
        )

    @staticmethod
    def _stack_optional_sequence(
        items: list[torch.Tensor | None],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not items or any(item is None for item in items):
            return None
        tensors = [item.to(device=device) for item in items if item is not None]
        max_len = max(int(t.shape[0]) for t in tensors)
        if tensors[0].ndim == 1:
            out = torch.zeros(len(tensors), max_len, device=device, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                out[i, : t.shape[0]] = t
            return out
        out = torch.zeros(len(tensors), max_len, tensors[0].shape[-1], device=device, dtype=dtype)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0], :] = t.to(dtype=dtype)
        return out

    @staticmethod
    def _stack_optional_ref_mel(
        items: list[torch.Tensor | None],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not items or any(item is None for item in items):
            return None
        tensors = [item.to(device=device, dtype=dtype) for item in items if item is not None]
        max_len = max(int(t.shape[-1]) for t in tensors)
        out = torch.zeros(len(tensors), tensors[0].shape[-2], max_len, device=device, dtype=dtype)
        for i, t in enumerate(tensors):
            out[i, :, : t.shape[-1]] = t
        return out

    @staticmethod
    def _stack_optional_style(
        items: list[torch.Tensor | None],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not items or any(item is None for item in items):
            return None
        return torch.stack(
            [item.to(device=device, dtype=dtype).reshape(-1) for item in items if item is not None],
            dim=0,
        )

    @staticmethod
    def _infer_ref_lengths(ref_mel: torch.Tensor) -> torch.Tensor:
        if ref_mel.ndim == 2:
            return torch.tensor([ref_mel.size(-1)], device=ref_mel.device, dtype=torch.long)
        nonzero = ref_mel.abs().sum(dim=1) > 0
        lengths = nonzero.long().sum(dim=1)
        return torch.clamp(lengths, min=1).to(device=ref_mel.device, dtype=torch.long)

    @staticmethod
    def _get_tensor(
        info: dict,
        key: str,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """Extract tensor from additional_information, handling nested dicts."""
        val = info.get(key)
        if val is None:
            for sub_key in ["codes", "hidden_states", "meta"]:
                sub = info.get(sub_key)
                if isinstance(sub, dict) and key in sub:
                    val = sub[key]
                    break
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.to(device=device, dtype=dtype) if dtype else val.to(device=device)
        if isinstance(val, list):
            if val and isinstance(val[0], torch.Tensor):
                return val[0].to(device=device, dtype=dtype) if dtype else val[0].to(device=device)
        return None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from s2mel.pth checkpoint.

        Bypasses vllm's default .pt iterator (same reason as talker: raw
        tensor .pt files in model dir). Loads s2mel.pth directly and
        flattens the nested ``state["net"]`` structure:

        - ``net["cfm"][key]`` → ``s2mel.models.cfm.{key}``
        - ``net["length_regulator"][key]`` → ``s2mel.models.length_regulator.{key}``
        - ``net["gpt_layer"][key]`` → ``s2mel.models.gpt_layer.{key}``
        """
        _ = weights

        ckpt_path = resolve_model_file(self.model_path, self.config.s2mel_checkpoint)
        if ckpt_path is None:
            raise FileNotFoundError(f"IndexTTS2 S2Mel checkpoint {self.config.s2mel_checkpoint!r} not found")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        net = state["net"] if "net" in state else state

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        prefix_map = {
            "cfm.": "s2mel.models.cfm.",
            "length_regulator.": "s2mel.models.length_regulator.",
            "gpt_layer.": "s2mel.models.gpt_layer.",
        }

        # Flatten nested ModuleDict state: {module: {param: tensor}} → flat iter
        flat_items: Iterable[tuple[str, torch.Tensor]]
        if net and isinstance(next(iter(net.values())), dict):
            flat_items = (
                (f"{mod}.{param}", tensor) for mod, mod_state in net.items() for param, tensor in mod_state.items()
            )
        else:
            flat_items = net.items()

        for name, loaded_weight in flat_items:
            mapped_name = name
            for old_prefix, new_prefix in prefix_map.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix) :]
                    break

            if mapped_name not in params_dict:
                logger.debug("Skipping unrecognized S2Mel weight: %s → %s", name, mapped_name)
                continue

            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped_name)

        logger.info("Loaded %d weights for IndexTTS2S2MelDecoder", len(loaded_params))
        required_prefixes = [
            "s2mel.models.cfm.",
            "s2mel.models.length_regulator.",
            "s2mel.models.gpt_layer.",
        ]
        missing_prefixes = [
            prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in loaded_params)
        ]
        if missing_prefixes:
            raise RuntimeError(
                "IndexTTS2 S2Mel checkpoint did not load required parameter groups: " + ", ".join(missing_prefixes)
            )

        stripped = _strip_weight_norm(self.s2mel)
        if stripped:
            # The fold replaces weight_g/weight_v with a plain `weight`
            # parameter; register it as loaded so vLLM's initialization
            # check does not flag it as missing from the checkpoint.
            for name in stripped:
                loaded_params.add(f"s2mel.{name}")
            logger.info("Folded %d weight_norm parametrizations in S2Mel", len(stripped))

        # Keep the CFM ODE solver in float32 once after loading instead of
        # recursively re-casting the module on every request.
        cfm = self.s2mel.models["cfm"]
        cfm.float()
        cfm.estimator_autocast_dtype = torch.bfloat16 if self.s2mel_dit_bf16 else None

        self._warmup_external_models()
        return loaded_params

    def _warmup_external_models(self) -> None:
        """Preload vocoder + semantic codec to eliminate first-request latency."""
        device = next((p.device for p in self.s2mel.parameters()), torch.device("cpu"))
        try:
            vocoder_name = self.config.vocoder.get("name", "nvidia/bigvgan_v2_22khz_80band_256x")
            voc_dtype = torch.bfloat16 if (self.s2mel_vocoder_bf16 and device.type == "cuda") else torch.float32
            bigvgan = _load_bigvgan(vocoder_name, device, voc_dtype)
            load_semantic_codec(self.model_path, self.config.semantic_codec, device)
            # Warm up the CUDA graph + torch.compile wrapper at load time so the
            # ~95s compile cost is paid during startup rather than on the first
            # request. Mirrors the Qwen3-TTS pattern (enable_cudagraph in
            # load_weights). _get_vocoder_graph constructs the wrapper and calls
            # warmup() on first access.
            if self.s2mel_vocoder_cuda_graph and device.type == "cuda":
                self._get_vocoder_graph(bigvgan, device, voc_dtype)
            logger.info("BigVGAN + semantic codec preloaded")
        except Exception as e:
            logger.warning("Failed to preload Stage 1 models: %s", e)

    def _get_vocoder_graph(self, bigvgan: nn.Module, device: torch.device, dtype: torch.dtype):
        """Return the CUDA-graph vocoder wrapper, or the eager model when disabled.

        Warmup is lazy (first request) following the GLM-TTS DiT wrapper
        pattern; on warmup failure the wrapper is disabled and all
        subsequent calls fall back to the eager model.
        """
        if not self.s2mel_vocoder_cuda_graph or device.type != "cuda":
            return bigvgan
        if self._vocoder_graph is None:
            from vllm_omni.model_executor.models.indextts2.bigvgan_cuda_graph import (
                CUDAGraphBigVGANWrapper,
            )

            self._vocoder_graph = CUDAGraphBigVGANWrapper(
                bigvgan,
                capture_sizes=self.s2mel_vocoder_capture_sizes,
                compile_shapes=self.s2mel_vocoder_compile_shapes,
            )
            try:
                self._vocoder_graph.warmup(device, dtype)
            except Exception:
                logger.warning("Disabling BigVGAN CUDA graphs after warmup failure", exc_info=True)
                self._vocoder_graph.enabled = False
        return self._vocoder_graph

    def _enable_dit_cuda_graph(self, estimator: nn.Module) -> None:
        """Attach a per-shape CUDA graph runner to the DiT transformer core.

        Sets `_assume_full_mask` on the attention modules: this decoder
        always passes x_lens == T, so the padding mask is all-True and
        dense attention is identical — the flag removes the backend's
        per-step ``torch.any(~mask)`` D2H sync, which is also a
        stream-capture blocker.
        """
        if not self.s2mel_dit_cuda_graph:
            return
        if getattr(estimator, "_cuda_graph_runner", None) is not None:
            return
        from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import CUDAGraphDiTRunner

        for layer in estimator.transformer.layers:
            layer.attention._assume_full_mask = True
        estimator._cuda_graph_runner = CUDAGraphDiTRunner(
            estimator.transformer,
            max_graphs=self.s2mel_dit_cuda_graph_max_graphs,
        )
        logger.info(
            "S2Mel DiT CUDA graph runner enabled (per-shape capture, LRU=%d)",
            self.s2mel_dit_cuda_graph_max_graphs,
        )
