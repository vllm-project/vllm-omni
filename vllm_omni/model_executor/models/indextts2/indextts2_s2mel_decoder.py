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
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.model_loader.hub_prefetch import _repo_prefetch_lock
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .configuration_indextts2 import (
    INDEXTTS25_MAX_DURATION_FACTOR,
    INDEXTTS25_MIN_DURATION_FACTOR,
    IndexTTS2Config,
)
from .preprocess_utils import (
    load_semantic_codec,
    resolve_model_directory,
    resolve_model_file,
)
from .s2mel.modules.commons import AttrDict, MyModel
from .s2mel.modules.flow_matching import build_cfm_unpad_data

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Lazy loaders for external models
# ---------------------------------------------------------------------------

_bigvgan_models: dict[tuple[str, str], nn.Module] = {}


def _raise_if_cuda_runtime_error(error: BaseException) -> None:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        raise error
    if not isinstance(error, RuntimeError):
        return
    message = str(error).lower()
    cuda_markers = (
        "cuda error",
        "device-side assert",
        "illegal memory access",
        "out of memory",
        "cublas",
        "cudnn",
        "cufft",
    )
    if any(marker in message for marker in cuda_markers):
        raise error


def _resolve_bigvgan_source(model_path: str, vocoder_name: str) -> str:
    if os.path.isdir(vocoder_name):
        return vocoder_name
    local_dir = resolve_model_directory(
        model_path,
        "bigvgan",
        os.path.join("hf_cache", "bigvgan"),
    )
    if local_dir is not None:
        return local_dir
    if os.path.isdir(model_path):
        raise FileNotFoundError(f"IndexTTS BigVGAN assets are missing from local bundle {model_path!r}")
    return vocoder_name


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


def _precast_cfm_estimator_bf16(
    estimator: nn.Module,
    *,
    enabled: bool,
) -> int:
    """Pre-cast CFM GEMM/conv parameters while keeping other layers in fp32."""
    if not enabled:
        return 0

    parameter_count = 0
    for module in estimator.modules():
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            continue
        for parameter in module.parameters(recurse=False):
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
            parameter_count += parameter.numel()
    return parameter_count


def _load_bigvgan(vocoder_name: str, device: torch.device, dtype: torch.dtype = torch.float32):
    cache_key = (vocoder_name, str(device), str(dtype))
    if cache_key in _bigvgan_models:
        return _bigvgan_models[cache_key]

    lock = _repo_prefetch_lock(vocoder_name) if not os.path.isdir(vocoder_name) else None
    if lock is not None:
        lock.__enter__()
    try:
        with set_default_torch_dtype(torch.float32):
            try:
                from .s2mel.modules import bigvgan as bigvgan_mod

                _patch_bigvgan_compat(bigvgan_mod.BigVGAN)
                bigvgan_model = bigvgan_mod.BigVGAN.from_pretrained(vocoder_name)
            except (ImportError, ModuleNotFoundError):
                import bigvgan

                _patch_bigvgan_compat(bigvgan.BigVGAN)
                bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_name)
            # Fold weight norm while its normalization is still computed in
            # fp32. The vLLM model loader may otherwise leave the process-wide
            # default dtype set to the stage's low-precision model dtype here.
            bigvgan_model.remove_weight_norm()
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    bigvgan_model = bigvgan_model.to(device=device, dtype=dtype).eval()
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
        self._resolved_vocoder_source: str | None = None
        self.use_gpt_latent = bool(getattr(self.config, "use_gpt_latent", True))
        self.semantic_codec_type = str(getattr(self.config, "semantic_codec_type", "repcodec"))

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

        self.s2mel = MyModel(
            s2mel_args,
            use_emovec=False,
            use_gpt_latent=self.use_gpt_latent,
        )

        # Diffusion config — configurable via deploy YAML hf_overrides
        self.diffusion_steps: int = getattr(self.config, "diffusion_steps", 25)
        self.inference_cfg_rate: float = getattr(self.config, "inference_cfg_rate", 0.7)
        self.mel_code_to_frame_ratio: float = 1.72
        self.s2mel_cfm_batch_size: int = int(getattr(self.config, "s2mel_cfm_batch_size", 1))
        if self.s2mel_cfm_batch_size < 1:
            raise ValueError("s2mel_cfm_batch_size must be at least 1")

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
        self.s2mel_vocoder_torch_compile: bool = getattr(
            self.config,
            "s2mel_vocoder_torch_compile",
            False,
        )
        self._vocoder_graph: Any = None
        self._compiled_vocoder: Any = None
        # Capture the DiT transformer core into per-shape CUDA graphs.
        # Enable via `s2mel_dit_cuda_graph: true`.
        self.s2mel_dit_cuda_graph: bool = getattr(self.config, "s2mel_dit_cuda_graph", False)
        self.s2mel_dit_torch_compile: bool = getattr(
            self.config,
            "s2mel_dit_torch_compile",
            False,
        )
        self.s2mel_dit_torch_compile_mode: str = str(
            getattr(
                self.config,
                "s2mel_dit_torch_compile_mode",
                "default",
            )
        )
        self._dit_runtime_configured = False
        self.s2mel_dit_cuda_graph_max_graphs: int = int(getattr(self.config, "s2mel_dit_cuda_graph_max_graphs", 8))
        self.s2mel_vocoder_capture_sizes: list[int] | None = getattr(self.config, "s2mel_vocoder_capture_sizes", None)
        self.s2mel_vocoder_compile_shapes: list[int] | None = getattr(self.config, "s2mel_vocoder_compile_shapes", None)
        logger.info(
            "[S2Mel] dit_bf16=%s, vocoder_bf16=%s, "
            "vocoder_cuda_graph=%s, vocoder_torch_compile=%s, "
            "dit_cuda_graph=%s, dit_torch_compile=%s, dit_compile_mode=%s, "
            "dit_graph_lru=%d, cfm_batch_size=%d, vocoder_buckets=%s, "
            "vocoder_compile_shapes=%s",
            self.s2mel_dit_bf16,
            self.s2mel_vocoder_bf16,
            self.s2mel_vocoder_cuda_graph,
            self.s2mel_vocoder_torch_compile,
            self.s2mel_dit_cuda_graph,
            self.s2mel_dit_torch_compile,
            self.s2mel_dit_torch_compile_mode,
            self.s2mel_dit_cuda_graph_max_graphs,
            self.s2mel_cfm_batch_size,
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
        self._validate_payload_policy(request_infos)
        self._validate_payload_contract(request_infos)

        device = input_ids.device
        model_dtype = self._s2mel_model_dtype()

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
            (
                mel_codes,
                latent,
                code_lens,
                code_lens_list,
                s_ref,
                ref_mel,
                style,
                ref_lengths_list,
            ) = self._build_batched_inputs(
                request_infos,
                device=device,
                model_dtype=model_dtype,
            )
            code_lens_raw = code_lens
        else:
            # Extract Stage 0 outputs and cast to model dtype
            latent = (
                self._get_tensor(additional_information, "latent", device, model_dtype) if self.use_gpt_latent else None
            )
            mel_codes = self._get_tensor(additional_information, "mel_codes", device)
            code_lens_raw = additional_information.get("code_lens")
            s_ref = self._get_tensor(additional_information, "S_ref", device)
            ref_mel = self._get_tensor(additional_information, "ref_mel", device, model_dtype)
            style = self._get_tensor(additional_information, "style", device, model_dtype)
            ref_lengths_list = self._ref_mel_lengths([ref_mel])

        if mel_codes is None or (self.use_gpt_latent and latent is None):
            if not request_infos:
                # vLLM profiles generation stages once with synthetic token
                # inputs and no upstream payload during engine initialization.
                return OmniOutput(
                    text_hidden_states=torch.zeros(1, device=device),
                    multimodal_outputs={"audio": torch.zeros(1, device=device), "sr": 22050},
                )
            raise ValueError("S2Mel decoder requires mel_codes" + (" and latent" if self.use_gpt_latent else ""))

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
        if latent is not None and latent.ndim == 2:
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
        duration_factors = self._duration_factors(request_infos, len(code_lens_list))

        # Trim to actual length
        max_len = max(code_lens_list)
        mel_codes = mel_codes[:, :max_len]
        if latent is not None:
            latent = latent[:, :max_len, :]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[S2Mel] after trim — mel_codes=%s latent=%s code_lens=%s max_len=%d",
                mel_codes.shape,
                latent.shape if latent is not None else None,
                code_lens.tolist(),
                max_len,
            )

        # --- S2Mel pipeline ---
        semantic_codec = load_semantic_codec(
            self.model_path,
            self.config.semantic_codec,
            device,
            codec_type=self.semantic_codec_type,
            checkpoint_name=getattr(
                self.config,
                "semantic_codec_checkpoint",
                None,
            ),
        )
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
        S_infer = self._build_semantic_embedding(
            codec=semantic_codec,
            mel_codes=mel_codes,
            latent=latent,
            dtype=model_dtype,
        )
        logger.debug(
            "[S2Mel] semantic decode → S_infer=%s dtype=%s codec=%s latent=%s",
            S_infer.shape,
            S_infer.dtype,
            self.semantic_codec_type,
            self.use_gpt_latent,
        )

        # 3. Length regulate: codes → mel frames
        semantic_time_scale = (
            int(getattr(semantic_codec, "downsample_scale", 1) or 1) if self.semantic_codec_type == "enhanced" else 1
        )
        target_lengths_list = self._target_lengths(
            code_lens_list,
            semantic_time_scale=semantic_time_scale,
            mel_code_to_frame_ratio=self.mel_code_to_frame_ratio,
            duration_factors=duration_factors,
        )
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
                if self.semantic_codec_type != "repcodec":
                    raise ValueError("IndexTTS 2.5 expects W2V-BERT reference embeddings, not semantic-code indices")
                with torch.no_grad():
                    s_ref_emb = semantic_codec.quantizer.vq2emb(s_ref_codes.unsqueeze(1))
                s_ref_emb = s_ref_emb.transpose(1, 2).to(dtype=model_dtype)

            ref_batch_size = int(ref_mel.shape[0])
            if len(ref_lengths_list) == 1 and ref_batch_size > 1:
                ref_lengths_list = ref_lengths_list * ref_batch_size
            if len(ref_lengths_list) != ref_batch_size:
                raise ValueError(
                    "IndexTTS reference-mel length/batch mismatch: "
                    f"lengths={len(ref_lengths_list)} batch={ref_batch_size}"
                )
            ref_target_lengths = torch.tensor(
                ref_lengths_list,
                device=device,
                dtype=torch.long,
            )
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
            raise RuntimeError("IndexTTS Stage 1 payload validation allowed missing S_ref or ref_mel")

        if style is None:
            raise RuntimeError("IndexTTS Stage 1 payload validation allowed missing style")
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
        self._configure_dit_runtime(device)
        cfm = self.s2mel.models["cfm"]
        cfm.estimator_autocast_dtype = torch.bfloat16 if self.s2mel_dit_bf16 else None
        cond = cond.float()
        if prompt_condition is not None:
            prompt_condition = prompt_condition.float()
        cat_condition = cat_condition.float()
        ref_mel = ref_mel.float()
        style = style.float()

        cfm_batch_size = int(cat_condition.shape[0])
        if prompt_condition is None:
            ref_lengths_list = [0] * cfm_batch_size
        elif len(ref_lengths_list) != cfm_batch_size:
            raise ValueError(
                f"IndexTTS CFM reference-length/batch mismatch: lengths={len(ref_lengths_list)} batch={cfm_batch_size}"
            )

        estimator = cfm.estimator
        max_cfm_group_size = min(cfm_batch_size, self.s2mel_cfm_batch_size)
        required_cache_batch = max(2, max_cfm_group_size * (2 if self.inference_cfg_rate > 0 else 1))
        if (
            estimator.transformer.freqs_cis is None
            or getattr(estimator.transformer, "max_batch_size", -1) < required_cache_batch
        ):
            estimator.setup_caches(max_batch_size=required_cache_batch, max_seq_length=16384)
            logger.info("[S2Mel] DiT caches initialized (freqs_cis + causal_mask)")
        cfm_seeds = self._cfm_request_seeds(
            request_infos,
            batch_size=int(cat_condition.shape[0]),
        )
        mel = self._run_cfm_groups(
            cfm=cfm,
            cond=cond,
            prompt_condition=prompt_condition,
            ref_mel=ref_mel,
            style=style,
            target_lengths=target_lengths_list,
            ref_lengths=ref_lengths_list,
            seeds=cfm_seeds,
            max_group_size=self.s2mel_cfm_batch_size,
        )
        logger.debug("[S2Mel] step5 CFM done → mel=%s", [tuple(item.shape) for item in mel])

        # 6. BigVGAN vocoding
        vocoder_name = self._get_resolved_vocoder_source()
        voc_dtype = torch.bfloat16 if (self.s2mel_vocoder_bf16 and device.type == "cuda") else torch.float32
        bigvgan = _load_bigvgan(vocoder_name, device, voc_dtype)
        vocode = self._get_vocoder_runner(bigvgan, device, voc_dtype)
        with torch.no_grad():
            wavs = self._vocode_mels(
                mels=mel,
                vocode=vocode,
                voc_dtype=voc_dtype,
            )
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

    def _s2mel_model_dtype(self) -> torch.dtype:
        if self.use_gpt_latent:
            return self.s2mel.models["gpt_layer"][0].weight.dtype
        return next(self.s2mel.models["length_regulator"].parameters()).dtype

    def _validate_payload_policy(self, infos: list[dict[str, Any]]) -> None:
        for index, info in enumerate(infos):
            if "use_gpt_latent" not in info:
                raise ValueError(f"IndexTTS Stage 1 input {index} is missing use_gpt_latent")
            payload_value = info["use_gpt_latent"]
            if bool(payload_value) != self.use_gpt_latent:
                raise ValueError(
                    "IndexTTS Stage 1 use_gpt_latent mismatch for input "
                    f"{index}: payload={bool(payload_value)} "
                    f"stage_config={self.use_gpt_latent}"
                )

    @staticmethod
    def _validate_payload_contract(infos: list[dict[str, Any]]) -> None:
        for index, info in enumerate(infos):
            for key in ("mel_codes", "S_ref", "ref_mel", "style"):
                value = info.get(key)
                if not isinstance(value, torch.Tensor) or value.numel() == 0:
                    raise ValueError(f"IndexTTS Stage 1 input {index} is missing non-empty {key}")

    @staticmethod
    def _duration_factors(infos: list[dict[str, Any]], batch_size: int) -> list[float]:
        factors = [float(info.get("duration_factor", 1.0)) for info in infos] if infos else [1.0] * batch_size
        if len(factors) != batch_size:
            raise ValueError(f"IndexTTS duration-factor batch mismatch: factors={len(factors)} batch={batch_size}")
        if any(
            not math.isfinite(factor) or not INDEXTTS25_MIN_DURATION_FACTOR <= factor <= INDEXTTS25_MAX_DURATION_FACTOR
            for factor in factors
        ):
            raise ValueError("IndexTTS duration_factor values must be finite and between 0.5 and 2.0")
        return factors

    @staticmethod
    def _target_lengths(
        code_lens: list[int],
        *,
        semantic_time_scale: int,
        mel_code_to_frame_ratio: float,
        duration_factors: list[float],
    ) -> list[int]:
        if len(code_lens) != len(duration_factors):
            raise ValueError(
                f"IndexTTS target-length batch mismatch: codes={len(code_lens)} factors={len(duration_factors)}"
            )
        return [
            int(length * semantic_time_scale * mel_code_to_frame_ratio * factor)
            for length, factor in zip(code_lens, duration_factors)
        ]

    def _run_cfm_groups(
        self,
        *,
        cfm: Any,
        cond: torch.Tensor,
        prompt_condition: torch.Tensor | None,
        ref_mel: torch.Tensor,
        style: torch.Tensor,
        target_lengths: list[int],
        ref_lengths: list[int],
        seeds: list[int | None],
        max_group_size: int,
    ) -> list[torch.Tensor]:
        """Run tightly packed CFM groups and restore original request order."""
        batch_size = int(cond.shape[0])
        if max_group_size < 1:
            raise ValueError("IndexTTS CFM max_group_size must be at least 1")
        if not (len(target_lengths) == len(ref_lengths) == len(seeds) == batch_size):
            raise ValueError(
                "IndexTTS CFM input batch mismatch: "
                f"batch={batch_size} target={len(target_lengths)} "
                f"ref={len(ref_lengths)} seeds={len(seeds)}"
            )
        if batch_size == 0:
            return []

        total_lengths = [ref_len + target_len for ref_len, target_len in zip(ref_lengths, target_lengths)]
        # Python's sort is stable, so equal total lengths preserve arrival order.
        sorted_indices = sorted(
            range(batch_size),
            key=total_lengths.__getitem__,
        )
        outputs: list[torch.Tensor | None] = [None] * batch_size
        estimator = cfm.estimator
        runtime_estimator = getattr(estimator, "_orig_mod", estimator)
        wavenet = getattr(runtime_estimator, "wavenet", None)
        min_ragged_length = int(getattr(wavenet, "ragged_min_length", 1))
        groups = self._cfm_group_indices(
            sorted_indices=sorted_indices,
            total_lengths=total_lengths,
            max_group_size=max_group_size,
            min_ragged_length=min_ragged_length,
        )

        for group_indices in groups:
            group_ref_lengths = [ref_lengths[index] for index in group_indices]
            group_target_lengths = [target_lengths[index] for index in group_indices]
            group_x_lengths = [
                ref_len + target_len for ref_len, target_len in zip(group_ref_lengths, group_target_lengths)
            ]
            group_length = max(group_x_lengths)
            group_prompt_length = max(group_ref_lengths)

            packed_condition = cond.new_zeros(
                len(group_indices),
                group_length,
                cond.shape[-1],
            )
            packed_prompt = ref_mel.new_zeros(
                len(group_indices),
                ref_mel.shape[1],
                group_prompt_length,
            )
            for row, (index, ref_len, target_len) in enumerate(
                zip(group_indices, group_ref_lengths, group_target_lengths)
            ):
                if ref_len:
                    if prompt_condition is None:
                        raise ValueError("IndexTTS CFM reference length requires prompt conditioning")
                    packed_condition[row, :ref_len].copy_(prompt_condition[index, :ref_len])
                    packed_prompt[row, :, :ref_len].copy_(ref_mel[index, :, :ref_len])
                packed_condition[row, ref_len : ref_len + target_len].copy_(cond[index, :target_len])

            group_index_tensor = torch.tensor(
                group_indices,
                device=style.device,
                dtype=torch.long,
            )
            packed_style = style.index_select(0, group_index_tensor)
            x_lens = torch.tensor(
                group_x_lengths,
                device=cond.device,
                dtype=torch.long,
            )
            prompt_lens = torch.tensor(
                group_ref_lengths,
                device=cond.device,
                dtype=torch.long,
            )
            full_mask = all(length == group_length for length in group_x_lengths)
            self._set_dit_full_mask_fast_path(estimator, enabled=full_mask)
            if full_mask:
                self._enable_dit_cuda_graph(estimator)
            unpad_data = None
            if not full_mask:
                token_offset = int(bool(getattr(runtime_estimator, "style_as_token", False))) + int(
                    bool(getattr(runtime_estimator, "time_as_token", False))
                )
                unpad_data = build_cfm_unpad_data(
                    group_x_lengths,
                    padded_length=group_length,
                    token_offset=token_offset,
                    repeat_for_cfg=self.inference_cfg_rate > 0,
                    device=cond.device,
                )

            initial_noise = self._sample_cfm_noise(
                seeds=[seeds[index] for index in group_indices],
                channels=cfm.in_channels,
                length=group_length,
                lengths=group_x_lengths,
                device=cond.device,
                dtype=cond.dtype,
            )
            with torch.no_grad():
                group_mel = cfm.inference(
                    packed_condition,
                    x_lens,
                    packed_prompt,
                    packed_style,
                    None,
                    self.diffusion_steps,
                    inference_cfg_rate=self.inference_cfg_rate,
                    initial_noise=initial_noise,
                    prompt_lens=prompt_lens,
                    unpad_data=unpad_data,
                )

            for row, (index, ref_len, target_len) in enumerate(
                zip(group_indices, group_ref_lengths, group_target_lengths)
            ):
                outputs[index] = group_mel[row, :, ref_len : ref_len + target_len]

        if any(output is None for output in outputs):
            raise RuntimeError("IndexTTS CFM failed to produce every request output")
        return [output for output in outputs if output is not None]

    @staticmethod
    def _cfm_group_indices(
        *,
        sorted_indices: list[int],
        total_lengths: list[int],
        max_group_size: int,
        min_ragged_length: int,
    ) -> list[list[int]]:
        """Build packed groups while preserving exact short-input semantics."""
        groups: list[list[int]] = []
        start = 0
        while start < len(sorted_indices):
            index = sorted_indices[start]
            group_size = 1 if total_lengths[index] < min_ragged_length else max_group_size
            group = sorted_indices[start : start + group_size]
            groups.append(group)
            start += len(group)
        return groups

    @staticmethod
    def _cfm_request_seeds(
        infos: list[dict[str, Any]],
        *,
        batch_size: int,
    ) -> list[int | None]:
        seeds: list[int | None] = []
        for info in infos:
            seed = info.get("seed")
            if isinstance(seed, torch.Tensor):
                seed = seed.reshape(-1)[0].item() if seed.numel() else None
            elif isinstance(seed, (list, tuple)):
                seed = seed[0] if seed else None
            seeds.append(int(seed) if seed is not None else None)

        if not seeds:
            return [None] * batch_size
        if len(seeds) != batch_size:
            raise ValueError(f"IndexTTS CFM seed/request batch mismatch: seeds={len(seeds)} batch={batch_size}")
        return seeds

    @staticmethod
    def _sample_cfm_noise(
        *,
        seeds: list[int | None],
        channels: int,
        length: int,
        lengths: list[int] | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not seeds or all(seed is None for seed in seeds):
            return None
        if lengths is not None and len(lengths) != len(seeds):
            raise ValueError(f"IndexTTS CFM noise length/seed mismatch: lengths={len(lengths)} seeds={len(seeds)}")

        rows: list[torch.Tensor] = []
        for index, seed in enumerate(seeds):
            row_length = lengths[index] if lengths is not None else length
            if row_length < 0 or row_length > length:
                raise ValueError(f"IndexTTS CFM noise length out of range: row={row_length} padded={length}")
            generator = None
            if seed is not None:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
            row = torch.randn(
                (1, channels, row_length),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            rows.append(F.pad(row, (0, length - row_length)))
        return torch.cat(rows, dim=0)

    def _build_semantic_embedding(
        self,
        *,
        codec: Any,
        mel_codes: torch.Tensor,
        latent: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.semantic_codec_type == "enhanced":
            semantic = codec.decode(mel_codes)
        else:
            semantic = codec.quantizer.vq2emb(mel_codes.unsqueeze(1))
            semantic = semantic.transpose(1, 2)
        semantic = semantic.to(device=mel_codes.device, dtype=dtype)

        if not self.use_gpt_latent:
            return semantic
        if latent is None:
            raise ValueError("IndexTTS GPT latent mode requires a latent tensor")

        projected = self.s2mel.forward_gpt(latent.to(device=mel_codes.device, dtype=dtype)).to(dtype=dtype)
        if projected.shape[1] != semantic.shape[1]:
            # EnhancedCodec restores the encoder's temporal downsampling with
            # nearest-neighbor upsampling. Mirror that time mapping for the
            # optional GPT residual before addition.
            projected = F.interpolate(
                projected.transpose(1, 2),
                size=semantic.shape[1],
                mode="nearest",
            ).transpose(1, 2)
        if projected.shape != semantic.shape:
            raise ValueError(
                "IndexTTS semantic/GPT latent shape mismatch after alignment: "
                f"semantic={tuple(semantic.shape)} latent={tuple(projected.shape)}"
            )
        return semantic + projected

    def _build_batched_inputs(
        self,
        infos: list[dict[str, Any]],
        *,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        list[int],
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        list[int],
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
            latent = self._get_tensor(info, "latent", device, model_dtype) if self.use_gpt_latent else None
            if mel is None or (self.use_gpt_latent and latent is None):
                raise ValueError(
                    f"S2Mel batched input {idx} is missing mel_codes" + (" or latent" if self.use_gpt_latent else "")
                )

            mel = mel.to(device=device, dtype=torch.long).reshape(-1)
            if latent is not None:
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
            code_len = max(0, min(code_len, int(mel.shape[0])))
            if latent is not None:
                if int(latent.shape[0]) != int(mel.shape[0]):
                    raise ValueError(
                        f"S2Mel batched input {idx} mel/latent length mismatch: "
                        f"mel={mel.shape[0]} latent={latent.shape[0]}"
                    )
            mel_items.append(mel[:code_len])
            if latent is not None:
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
        # Padding reaches vq2emb before length masking, so it must be a valid
        # semantic-code index. Actual sequence lengths are still carried by
        # code_lens/target_lengths; stop_token (8193) is outside the codebook.
        mel_batch = torch.zeros((len(mel_items), max_code_len), dtype=torch.long, device=device)
        latent_batch: torch.Tensor | None = None
        if self.use_gpt_latent:
            latent_dim = latent_items[0].shape[-1]
            latent_batch = torch.zeros(
                len(latent_items),
                max_code_len,
                latent_dim,
                device=device,
                dtype=model_dtype,
            )
        for i, mel in enumerate(mel_items):
            mel_batch[i, : mel.shape[0]] = mel
            if latent_batch is not None:
                latent = latent_items[i]
                latent_batch[i, : latent.shape[0]] = latent.to(dtype=model_dtype)

        s_ref_batch = self._stack_optional_sequence(s_ref_items, device=device, dtype=model_dtype)
        ref_lengths = self._ref_mel_lengths(ref_mel_items)
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
            ref_lengths,
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
    def _ref_mel_lengths(items: list[torch.Tensor | None]) -> list[int]:
        """Return unpadded reference lengths without inspecting CUDA values."""
        if not items or any(item is None for item in items):
            return []
        return [int(item.shape[-1]) for item in items if item is not None]

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
        ]
        if self.use_gpt_latent:
            required_prefixes.append("s2mel.models.gpt_layer.")
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
        precast_parameter_count = _precast_cfm_estimator_bf16(
            cfm.estimator,
            enabled=self.s2mel_dit_bf16,
        )
        if self.s2mel_dit_bf16:
            logger.info(
                "Pre-cast %d CFM estimator Linear/Conv1d parameters to bfloat16",
                precast_parameter_count,
            )
        cfm.estimator_autocast_dtype = torch.bfloat16 if self.s2mel_dit_bf16 else None

        self._warmup_external_models()
        return loaded_params

    def _configure_dit_runtime(self, device: torch.device) -> None:
        if self._dit_runtime_configured:
            return
        if self.s2mel_dit_torch_compile and self.s2mel_dit_cuda_graph:
            raise ValueError(
                "s2mel_dit_torch_compile and s2mel_dit_cuda_graph are mutually exclusive",
            )
        if self.s2mel_dit_torch_compile:
            if device.type == "cuda":
                self.s2mel.enable_torch_compile(
                    mode=self.s2mel_dit_torch_compile_mode,
                )
                logger.info(
                    "S2Mel DiT torch.compile enabled with mode=%s",
                    self.s2mel_dit_torch_compile_mode,
                )
            else:
                logger.info(
                    "S2Mel DiT torch.compile skipped on device type %s",
                    device.type,
                )
        self._dit_runtime_configured = True

    def _warmup_external_models(self) -> None:
        """Preload vocoder + semantic codec to eliminate first-request latency."""
        device = next((p.device for p in self.s2mel.parameters()), torch.device("cpu"))
        try:
            vocoder_name = self._get_resolved_vocoder_source()
            voc_dtype = torch.bfloat16 if (self.s2mel_vocoder_bf16 and device.type == "cuda") else torch.float32
            bigvgan = _load_bigvgan(vocoder_name, device, voc_dtype)
            load_semantic_codec(
                self.model_path,
                self.config.semantic_codec,
                device,
                codec_type=self.semantic_codec_type,
                checkpoint_name=getattr(
                    self.config,
                    "semantic_codec_checkpoint",
                    None,
                ),
            )
            if (self.s2mel_vocoder_cuda_graph or self.s2mel_vocoder_torch_compile) and device.type == "cuda":
                self._get_vocoder_runner(bigvgan, device, voc_dtype)
            logger.info("BigVGAN + semantic codec preloaded")
        except Exception as e:
            _raise_if_cuda_runtime_error(e)
            raise RuntimeError("Failed to preload required IndexTTS Stage 1 models") from e

    def _get_resolved_vocoder_source(self) -> str:
        if self._resolved_vocoder_source is None:
            vocoder_name = self.config.vocoder.get(
                "name",
                "nvidia/bigvgan_v2_22khz_80band_256x",
            )
            self._resolved_vocoder_source = _resolve_bigvgan_source(
                self.model_path,
                vocoder_name,
            )
        return self._resolved_vocoder_source

    def _vocode_mels(
        self,
        *,
        mels: list[torch.Tensor],
        vocode: Any,
        voc_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Vocode each already-cropped request at its exact mel length."""
        wavs: list[torch.Tensor] = []
        for mel in mels:
            if mel.ndim == 2:
                mel_input = mel.unsqueeze(0)
            elif mel.ndim == 3 and mel.shape[0] == 1:
                mel_input = mel
            else:
                raise ValueError(f"IndexTTS BigVGAN expects one mel per request, got {tuple(mel.shape)}")
            wav = vocode(mel_input.to(voc_dtype)).float().squeeze(0).squeeze(0)
            wavs.append(torch.clamp(wav, -1.0, 1.0))
        return wavs

    def _get_vocoder_runner(
        self,
        bigvgan: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Return the selected exact-length BigVGAN execution path."""
        if self.s2mel_vocoder_torch_compile and self.s2mel_vocoder_cuda_graph:
            raise ValueError(
                "s2mel_vocoder_torch_compile and s2mel_vocoder_cuda_graph are mutually exclusive",
            )
        if self.s2mel_vocoder_cuda_graph:
            return self._get_vocoder_graph(bigvgan, device, dtype)
        if not self.s2mel_vocoder_torch_compile or device.type != "cuda":
            return bigvgan
        if self._compiled_vocoder is None:
            self._compiled_vocoder = torch.compile(
                bigvgan.forward,
                mode="default",
                fullgraph=False,
                dynamic=True,
            )
            logger.info(
                "BigVGAN dynamic torch.compile enabled with exact-length inputs",
            )
        return self._compiled_vocoder

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

    @staticmethod
    def _set_dit_full_mask_fast_path(
        estimator: nn.Module,
        *,
        enabled: bool,
    ) -> None:
        """Select dense attention and physical WaveNet reflect boundaries."""
        runtime_estimator = getattr(estimator, "_orig_mod", estimator)
        for layer in runtime_estimator.transformer.layers:
            layer.attention._assume_full_mask = enabled
        if hasattr(runtime_estimator, "wavenet"):
            runtime_estimator.wavenet._assume_full_mask = enabled

        # A graph captured with the dense-attention branch must never replay
        # for a padded input. Keep its cache resident, but route this call
        # through eager execution until a later full-mask call re-enables it.
        graph_runner = getattr(runtime_estimator, "_cuda_graph_runner", None)
        if graph_runner is not None and not enabled:
            graph_runner.enabled = False

    def _enable_dit_cuda_graph(self, estimator: nn.Module) -> None:
        """Attach a per-shape CUDA graph runner to the DiT transformer core.

        Full-mask semantics are enabled independently by
        ``_set_dit_full_mask_fast_path`` so eager execution also avoids
        the per-layer ``torch.any(~mask)`` host synchronization.
        """
        if not self.s2mel_dit_cuda_graph:
            return
        graph_runner = getattr(estimator, "_cuda_graph_runner", None)
        if graph_runner is not None:
            graph_runner.enabled = True
            return
        from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import CUDAGraphDiTRunner

        estimator._cuda_graph_runner = CUDAGraphDiTRunner(
            estimator.transformer,
            max_graphs=self.s2mel_dit_cuda_graph_max_graphs,
        )
        logger.info(
            "S2Mel DiT CUDA graph runner enabled (per-shape capture, LRU=%d)",
            self.s2mel_dit_cuda_graph_max_graphs,
        )
