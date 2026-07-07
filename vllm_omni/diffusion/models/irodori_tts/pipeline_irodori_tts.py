# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .codec import DACVAECodec, patchify_latent, unpatchify_latent
from .duration import build_duration_features
from .irodori_tts_transformer import IrodoriTTSTransformer
from .rf import sample_euler_rf_cfg

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# 1. Pretrained Text Tokenizer Wrapper
# ----------------------------------------------------------------------------


class PretrainedTextTokenizer:
    """
    Hugging Face tokenizer wrapper for Irodori TTS conditioning.
    Provides standardized right-padding and special BOS token prepending.
    """

    def __init__(self, tokenizer, add_bos: bool = True) -> None:
        self.tokenizer = tokenizer
        self.add_bos = bool(add_bos)
        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError("Tokenizer has no pad_token_id (and no eos_token fallback).")

        if self.add_bos and self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer has no bos_token_id but add_bos=True.")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        add_bos: bool = True,
        local_files_only: bool = False,
    ) -> PretrainedTextTokenizer:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for pretrained text tokenization. "
                "Install with `pip install transformers sentencepiece`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        return cls(tokenizer=tokenizer, add_bos=add_bos)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("pad_token_id is unexpectedly None.")
        return int(pad_id)

    def encode(self, text: str, add_bos: bool | None = None) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        use_bos = self.add_bos if add_bos is None else bool(add_bos)
        if use_bos:
            bos_id = self.bos_token_id
            if bos_id is None:
                raise ValueError("Tokenizer has no bos_token_id but BOS prepend was requested.")
            token_ids.insert(0, int(bos_id))
        return torch.tensor(token_ids, dtype=torch.long)

    def batch_encode(
        self,
        texts: Iterable[str],
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t) for t in texts]
        if max_length is None:
            max_length = max(max(x.numel(), 1) for x in encoded)
        if max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length}")

        batch = torch.full(
            (len(encoded), max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        mask = torch.zeros((len(encoded), max_length), dtype=torch.bool)
        for i, seq in enumerate(encoded):
            n = min(max_length, seq.numel())
            if n > 0:
                batch[i, :n] = seq[:n]
                mask[i, :n] = True
        return batch, mask


# ----------------------------------------------------------------------------
# 2. Main Integrated Pipeline
# ----------------------------------------------------------------------------


class IrodoriTTSPipeline(nn.Module, SupportAudioOutput):
    """
    Unified Pipeline Orchestrator for serving Irodori TTS v3 inside vLLM Omni.
    """

    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 48000

    # Metadata for offloading, HSDP, and CPU sharding controllers
    _dit_modules = ["transformer"]
    _encoder_modules = []
    _vae_modules = ["vae"]

    def __init__(self, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.float16)

        # 1. Directly resolve local model.safetensors file and parse embedded config metadata
        import json
        import os
        from dataclasses import fields

        from safetensors import safe_open

        from .irodori_tts_transformer import ModelConfig

        cfg_dict = {}
        try:
            model_path = self.od_config.model
            if not os.path.isdir(model_path):
                # Resolve local HF hub cache directory for the repo ID
                repo_id_escaped = model_path.replace("/", "--")
                cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_id_escaped}/snapshots")
                if os.path.exists(cache_dir):
                    snapshots = sorted(os.listdir(cache_dir))
                    if snapshots:
                        model_path = os.path.join(cache_dir, snapshots[-1])

            if os.path.isdir(model_path):
                safetensors_file = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_file):
                    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                        metadata = f.metadata()
                        if metadata is not None and "config_json" in metadata:
                            cfg_dict = json.loads(metadata["config_json"])
        except Exception as e:
            logger.warning("Could not parse config_json metadata from local safetensors file: %s", e)

        # 2. Filter parsed options to strictly match ModelConfig dataclass fields
        valid_fields = {f.name for f in fields(ModelConfig)}
        filtered_kwargs = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        cfg = ModelConfig(**filtered_kwargs)

        # A. Core Denoising Transformer
        self.transformer = IrodoriTTSTransformer(cfg=cfg, od_config=od_config)

        # B. Autotokenizer Wrapper
        self.tokenizer = PretrainedTextTokenizer.from_pretrained(
            self.transformer.cfg.text_tokenizer_repo,
            add_bos=self.transformer.cfg.text_add_bos,
            local_files_only=getattr(od_config, "local_files_only", False),
        )

        # C. DACVAE Codec / VAE Loader (default Aratako Japanese 32dim)
        self.vae = DACVAECodec.load(
            repo_id="Aratako/Semantic-DACVAE-Japanese-32dim",
            device=str(self.device),
            dtype=self.dtype,
        )

        # D. Move all layers and weights to the local device & dtype
        self.to(device=self.device, dtype=self.dtype)

        # E. Define weights source for diffusers loader tracking
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.od_config.model,
                subfolder=None,
                revision=getattr(self.od_config, "revision", None),
                prefix="",
                fall_back_to_pt=False,
                allow_patterns_overrides=[
                    "model.safetensors",
                ],
            ),
        ]

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Execute request-level text-to-latent voice cloning.

        Args:
            req: OmniDiffusionRequest containing generations params
        Returns:
            DiffusionOutput holding unpatched raw VAE-ready latent tensor
        """
        # 1. Parse text prompts from the unified request
        prompt = [
            p if isinstance(p, str) else (p.get("prompt") or p.get("input") or p.get("text") or "") for p in req.prompts
        ]
        batch_size = len(prompt)
        device = self.device
        dtype = self.dtype

        # 2. Extract scheduling and guidance hyperparameters
        num_inference_steps = req.sampling_params.num_inference_steps
        if num_inference_steps is None:
            num_inference_steps = 40  # default linear steps matching baseline

        seed = req.sampling_params.seed if req.sampling_params.seed is not None else 0

        # Optional extras passed via extra_args
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        ref_wav_path = extra_args.get("ref_wav")
        duration_scale = float(extra_args.get("duration_scale", 1.0))
        min_seconds = float(extra_args.get("min_seconds", 0.0))
        max_seconds = float(extra_args.get("max_seconds", 300.0))

        cfg_scale_text = float(extra_args.get("cfg_scale_text", 3.0))
        cfg_scale_speaker = float(extra_args.get("cfg_scale_speaker", 5.0))
        cfg_scale_caption = float(extra_args.get("cfg_scale_caption", 3.0))
        cfg_guidance_mode = str(extra_args.get("cfg_guidance_mode", "independent")).strip().lower()
        speaker_uncond_mode = str(extra_args.get("speaker_uncond_mode", "mask")).strip().lower()

        # 3. Tokenize prompts
        text_ids, text_mask = self.tokenizer.batch_encode(prompt)
        text_ids = text_ids.to(device=device)
        text_mask = text_mask.to(device=device)

        # 4. Process reference audio voice cloning conditioning
        ref_latent = None
        ref_mask = None
        has_speaker = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Check for inline resolved ref_audio tuple (waveform, sr) in prompts
        inline_ref_audio = None
        for p in req.prompts:
            if isinstance(p, dict) and p.get("ref_audio") is not None:
                inline_ref_audio = p["ref_audio"]
                break

        if inline_ref_audio is not None:
            import numpy as np

            ref_data, ref_sr = inline_ref_audio
            if isinstance(ref_data, np.ndarray):
                ref_waveform = torch.from_numpy(ref_data).float().to(device=device)
            else:
                ref_waveform = ref_data.float().to(device=device)

            if ref_waveform.ndim == 1:
                ref_waveform = ref_waveform.unsqueeze(0)  # (1, T)
            elif ref_waveform.ndim == 2 and ref_waveform.shape[0] > ref_waveform.shape[1]:
                ref_waveform = ref_waveform.transpose(0, 1)  # (C, T)

            # Preprocess reference audio via VAE encoder (output is unpatched)
            ref_latent_raw = self.vae.encode_waveform(ref_waveform, ref_sr)

            # Patchify latents to speaker dimension format expected by transformer reference encoder
            ref_latent = patchify_latent(ref_latent_raw, self.transformer.cfg.speaker_patch_size)
            ref_latent = ref_latent.to(dtype=dtype, device=device)

            if ref_latent.shape[0] == 1 and batch_size > 1:
                ref_latent = ref_latent.expand(batch_size, -1, -1)

            ref_mask = torch.ones(ref_latent.shape[:2], dtype=torch.bool, device=device)
            has_speaker = torch.ones(batch_size, dtype=torch.bool, device=device)
        elif ref_wav_path is not None:
            import soundfile as sf

            data, ref_sr = sf.read(ref_wav_path)
            ref_waveform = torch.from_numpy(data).float().to(device=device)
            if ref_waveform.ndim == 1:
                ref_waveform = ref_waveform.unsqueeze(0)  # (1, T)
            else:
                ref_waveform = ref_waveform.transpose(0, 1)  # (C, T)

            # Preprocess reference audio via VAE encoder (output is unpatched)
            ref_latent_raw = self.vae.encode_waveform(ref_waveform, ref_sr)

            # Patchify latents to speaker dimension format expected by transformer reference encoder
            ref_latent = patchify_latent(ref_latent_raw, self.transformer.cfg.speaker_patch_size)
            ref_latent = ref_latent.to(dtype=dtype, device=device)

            if ref_latent.shape[0] == 1 and batch_size > 1:
                ref_latent = ref_latent.expand(batch_size, -1, -1)

            ref_mask = torch.ones(ref_latent.shape[:2], dtype=torch.bool, device=device)
            has_speaker = torch.ones(batch_size, dtype=torch.bool, device=device)
        elif req.is_dummy_run():
            # Generate dummy reference speaker conditioning for the warmup pass
            ref_latent = torch.zeros(
                (batch_size, 4, self.transformer.cfg.patched_latent_dim),
                dtype=dtype,
                device=device,
            )
            ref_mask = torch.ones((batch_size, 4), dtype=torch.bool, device=device)
            has_speaker = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 5. Predict speech frames length (Durations Predictor)
        hop_length = int(self.vae.model.hop_length)
        if self.transformer.duration_predictor is not None:
            token_counts = text_mask.sum(dim=1)
            duration_features = build_duration_features(
                prompt,
                token_counts=token_counts,
                max_text_len=text_ids.shape[1],
                has_speaker=has_speaker,
            ).to(device=device, dtype=dtype)

            # Retrieve conditions
            duration_text_state, duration_text_mask, duration_speaker_state, _duration_speaker_mask, _, _ = (
                self.transformer.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                )
            )

            pred_log_frames = self.transformer.predict_duration_log_frames(
                text_state=duration_text_state,
                text_mask=duration_text_mask,
                speaker_state=duration_speaker_state,
                speaker_mask=_duration_speaker_mask,
                duration_features=duration_features,
                has_speaker=has_speaker,
            )

            pred_frames = torch.expm1(pred_log_frames).float().mean().item()
            scaled_frames = pred_frames * duration_scale

            min_frames = max(1, math.ceil(min_seconds * self.vae.sample_rate / hop_length))
            max_frames = max(1, math.floor(max_seconds * self.vae.sample_rate / hop_length))
            latent_steps = int(round(scaled_frames))
            latent_steps = max(min_frames, min(max_frames, latent_steps))
        else:
            fallback_seconds = 30.0
            target_samples = int(fallback_seconds * self.vae.sample_rate)
            latent_steps = math.ceil(target_samples / hop_length)

        patched_steps = math.ceil(latent_steps / self.transformer.cfg.latent_patch_size)

        # 6. Run ODE Euler Flow CFG Denoising Scheduler Loop
        z_patched = sample_euler_rf_cfg(
            model=self.transformer,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            sequence_length=patched_steps,
            speaker_uncond_mode=speaker_uncond_mode,
            num_steps=num_inference_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_scale_caption=cfg_scale_caption,
            cfg_guidance_mode=cfg_guidance_mode,
            seed=seed,
            use_context_kv_cache=True,
        )

        # 7. Unpatchify final latents back to sequence space
        z_unpatched = unpatchify_latent(
            z_patched,
            self.transformer.cfg.latent_patch_size,
            self.transformer.cfg.latent_dim,
        )

        # 8. Decode unpatched final latents to audio waveform using VAE
        # Output waveform is shape (B, 1, samples)
        audio = self.vae.decode_latent(z_unpatched)

        return DiffusionOutput(output=audio)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Directly map and load checkpoint weights into the pipeline parameters and buffers."""
        loaded: set[str] = set()
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())

        for name, tensor in weights:
            # Map name to its prefix "transformer." in the pipeline
            new_name = f"transformer.{name}"

            if new_name in params:
                param = params[new_name]
                param.data.copy_(tensor)
                loaded.add(new_name)
            elif new_name in buffers:
                buffers[new_name].data.copy_(tensor)
                loaded.add(new_name)

        logger.info(
            "IrodoriTTSPipeline load_weights: successfully loaded %d weights from root checkpoint",
            len(loaded),
        )
        return loaded


# ----------------------------------------------------------------------------
# 3. Post-Processing builder function for Registry
# ----------------------------------------------------------------------------


def get_irodori_tts_post_process_func(od_config: OmniDiffusionConfig):
    """
    Create post-processing function for Irodori TTS output.
    Converts raw waveform tensors to numpy arrays for serving response serialization.
    """

    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio

        # Convert torch tensor to standard numpy array
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func
