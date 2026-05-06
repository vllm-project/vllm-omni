# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Wan2.2 Speech-to-Video (S2V) Pipeline for vLLM-Omni.

Migrated from Wan2.2/wan/speech2video.py (WanS2V class).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel, Wav2Vec2ForCTC, Wav2Vec2Processor
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal.media.audio import load_audio

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    load_transformer_config,
    retrieve_latents,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import (
    create_s2v_transformer_from_config,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

# Default negative prompt from Wan2.2 S2V config (shared_config + s2v_14B)
_S2V_DEFAULT_NEG_PROMPT = (
    "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# ---------------------------------------------------------------------------
# Audio utility helpers (ported from Wan2.2/wan/modules/s2v/audio_encoder.py)
# ---------------------------------------------------------------------------

_AUDIO_VIDEO_RATE = 30


def _linear_interpolation(
    features: torch.Tensor,
    input_fps: float,
    output_fps: float,
    output_len: int | None = None,
) -> torch.Tensor:
    """Resample audio features from *input_fps* to *output_fps*.

    Args:
        features: Shape ``[1, T, D]``.
        input_fps: Source frame-rate of *features*.
        output_fps: Target frame-rate.
        output_len: If given, force output length; otherwise derived from
            ``features.shape[1] / input_fps * output_fps``.

    Returns:
        Resampled tensor with shape ``[1, output_len, D]``.
    """
    features = features.transpose(1, 2)  # [1, D, T]
    if output_len is None:
        seq_len = features.shape[2] / float(input_fps)
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(features, size=output_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)  # [1, output_len, D]


def _get_sample_indices(
    original_fps: float,
    total_frames: int,
    target_fps: float,
    num_sample: int,
    fixed_start: int | None = None,
) -> np.ndarray:
    """Compute frame indices to sample *num_sample* frames at *target_fps*."""
    required_duration = num_sample / target_fps
    required_origin_frames = int(np.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")

    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = np.random.randint(0, max_start + 1)
    start_time = start_frame / original_fps

    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)

    frame_indices = np.round(np.array(time_points) * original_fps).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    return frame_indices


def _get_audio_embed_bucket_fps(
    audio_embed: torch.Tensor,
    fps: int = 16,
    batch_frames: int = 80,
    m: int = 0,
) -> tuple[torch.Tensor, int, int]:
    """Bucket audio embeddings to align with video frames at *fps*.

    Args:
        audio_embed: Shape ``[num_layers, audio_frame_num, audio_dim]``.
        fps: Video frame-rate.
        batch_frames: Number of video frames per clip (``infer_frames``).
        m: Context window radius for each audio sample.

    Returns:
        ``(audio_embed_bucket, num_repeat, target_video_frames)`` where
        *num_repeat* is the number of clips needed and *target_video_frames*
        is the exact number of video frames to match audio duration.
    """
    num_layers, audio_frame_num, audio_dim = audio_embed.shape
    return_all_layers = num_layers > 1

    scale = _AUDIO_VIDEO_RATE / fps
    # Calculate exact video frames needed to match audio duration
    target_video_frames = int(audio_frame_num / scale)
    # Number of clips needed (use ceil to cover all audio)
    min_batch_num = math.ceil(target_video_frames / batch_frames)
    # Ensure at least 1 clip
    min_batch_num = max(1, min_batch_num)
    bucket_num = min_batch_num * batch_frames

    pad_audio_num = math.ceil(min_batch_num * batch_frames / fps * _AUDIO_VIDEO_RATE) - audio_frame_num
    batch_idx = _get_sample_indices(
        original_fps=_AUDIO_VIDEO_RATE,
        total_frames=audio_frame_num + pad_audio_num,
        target_fps=fps,
        num_sample=bucket_num,
        fixed_start=0,
    )
    audio_sample_stride = int(_AUDIO_VIDEO_RATE / fps)

    batch_audio_eb = []
    for bi in batch_idx:
        if bi < audio_frame_num:
            chosen_idx = list(
                range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride)
            )
            chosen_idx = [max(0, c) for c in chosen_idx]
            chosen_idx = [min(audio_frame_num - 1, c) for c in chosen_idx]

            if return_all_layers:
                frame_audio_embed = audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
            else:
                frame_audio_embed = audio_embed[0][chosen_idx].flatten()
        else:
            if return_all_layers:
                frame_audio_embed = torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
            else:
                frame_audio_embed = torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device)
        batch_audio_eb.append(frame_audio_embed)

    batch_audio_eb = torch.stack(batch_audio_eb, dim=0)
    return batch_audio_eb, min_batch_num, target_video_frames


# ---------------------------------------------------------------------------
# Size calculation helper (ported from WanS2V.get_size_less_than_area)
# ---------------------------------------------------------------------------


def _get_size_less_than_area(
    height: int,
    width: int,
    target_area: int = 1024 * 704,
    divisor: int = 64,
) -> tuple[int, int]:
    """Compute (H, W) that fits within *target_area*, padded to *divisor*."""
    if height * width <= target_area:
        max_upper_area = target_area
        min_scale = 0.1
        max_scale = 1.0
    else:
        max_upper_area = target_area
        d = divisor - 1
        b = d * (height + width)
        a = height * width
        c = d**2 - max_upper_area
        min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)
        max_scale = math.sqrt(max_upper_area / (height * width))

    for i in range(100):
        scale = max_scale - (max_scale - min_scale) * i / 100
        new_height, new_width = int(height * scale), int(width * scale)
        pad_height = (divisor - new_height % divisor) % divisor
        pad_width = (divisor - new_width % divisor) % divisor
        padded_height, padded_width = new_height + pad_height, new_width + pad_width
        if padded_height * padded_width <= max_upper_area:
            return padded_height, padded_width

    # Fallback
    aspect_ratio = width / height
    target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
    target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
    if target_width >= width or target_height >= height:
        target_width = int(width // divisor * divisor)
        target_height = int(height // divisor * divisor)
    return target_height, target_width


# ---------------------------------------------------------------------------
# Pre / Post process functions (registered via registry.py)
# ---------------------------------------------------------------------------


def get_wan22_s2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        output,
        output_type: str = "np",
    ):
        # output is (video_tensor, audio_waveform_np, audio_sample_rate)
        if isinstance(output, tuple) and len(output) == 3:
            video, audio_waveform, audio_sr = output
        else:
            video = output
            audio_waveform = None
            audio_sr = None

        if output_type == "latent":
            return video

        processed_video = video_processor.postprocess_video(video, output_type=output_type)

        if audio_waveform is not None:
            return {
                "video": processed_video,
                "audio": audio_waveform,
                "audio_sample_rate": audio_sr,
                "fps": 16,
            }
        return processed_video

    return post_process_func


def get_wan22_s2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for S2V: load ref image, compute target size.

    Expects ``multi_modal_data`` to contain:
      - ``"image"``: reference image (PIL.Image or file path)
      - ``"audio"``: audio file path (str)

    Optionally:
      - ``"pose_video"``: pose conditioning video path (str)
      - ``"init_first_frame"``: bool, use ref image as first frame
    """

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            # -- Reference image --
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if raw_image is None:
                raise ValueError(
                    "No reference image provided. S2V requires a reference image. "
                    'Set `"multi_modal_data": {"image": <path or PIL.Image>, ...}`'
                )
            if isinstance(raw_image, str):
                image = PIL.Image.open(raw_image).convert("RGB")
            elif isinstance(raw_image, PIL.Image.Image):
                image = raw_image
            else:
                raise TypeError(f"Unsupported image type {type(raw_image)}")

            # -- Audio --
            raw_audio = multi_modal_data.get("audio", None) if multi_modal_data is not None else None
            if raw_audio is None:
                raise ValueError(
                    "No audio provided. S2V requires an audio file path. "
                    'Set `"multi_modal_data": {"audio": "<path>", ...}`'
                )

            # -- Compute target size --
            max_area = 720 * 1280
            if request.sampling_params.height is not None and request.sampling_params.width is not None:
                height, width = request.sampling_params.height, request.sampling_params.width
            else:
                ref_h, ref_w = image.height, image.width
                height, width = _get_size_less_than_area(ref_h, ref_w, target_area=max_area)
                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width

            # Resize + center-crop reference image to target size
            resize_op = transforms.Resize(min(height, width))
            crop_op = transforms.CenterCrop((height, width))
            ref_pil = crop_op(resize_op(image))

            prompt["multi_modal_data"]["image"] = ref_pil
            prompt["additional_information"]["audio_path"] = raw_audio
            prompt["additional_information"]["pose_video"] = (
                multi_modal_data.get("pose_video", None) if multi_modal_data is not None else None
            )
            prompt["additional_information"]["init_first_frame"] = (
                multi_modal_data.get("init_first_frame", False) if multi_modal_data is not None else False
            )
            request.prompts[i] = prompt

        return request

    return pre_process_func


# ---------------------------------------------------------------------------
# Model format detection
# ---------------------------------------------------------------------------


def _is_diffusers_format(model_path: str) -> bool:
    """Check if a model directory uses diffusers subfolder layout."""
    return os.path.isdir(os.path.join(model_path, "tokenizer")) and os.path.isdir(os.path.join(model_path, "vae"))


def _resolve_model_path(model: str) -> str:
    """Resolve a HF repo ID or local path to a local directory.

    For local paths, returns as-is. For HF repo IDs, downloads needed files
    and returns the snapshot directory.
    """
    if os.path.isdir(model):
        return model
    from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

    # S2V models use a non-standard repo layout with files at root level.
    # Download only the files needed for inference:
    # - Transformer weights (sharded safetensors + index)
    # - VAE weights
    # - T5 encoder weights + tokenizer config
    # - Audio encoder (wav2vec2) weights + config
    # - Model config files
    allow_patterns = [
        "*.safetensors",  # Transformer weights (sharded)
        "*.safetensors.index.json",  # Weight shard index
        "*.pth",  # VAE and T5 weights
        "*.json",  # Config files (config.json, configuration.json)
        "google/**",  # T5 tokenizer/config directory
        "wav2vec2-large-xlsr-53-english/**",  # Audio encoder directory
    ]
    return download_weights_from_hf_specific(model, None, allow_patterns)


# ---------------------------------------------------------------------------
# Wan T5 → UMT5EncoderModel weight conversion
# ---------------------------------------------------------------------------

# UMT5Config matching the Wan2.2 T5 encoder architecture (umt5-xxl variant).
_WAN_UMT5_CONFIG = UMT5Config(
    vocab_size=256384,
    d_model=4096,
    d_kv=64,
    d_ff=10240,
    num_heads=64,
    num_layers=24,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    dense_act_fn="gelu_new",
    is_gated_act=True,
    is_encoder_decoder=False,
)


def _convert_wan_t5_state_dict(wan_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a Wan2.2 T5 encoder state dict to HuggingFace UMT5 format.

    The Wan checkpoint uses flat ``blocks.{i}.attn.*`` naming whereas
    HuggingFace ``UMT5EncoderModel`` expects ``encoder.block.{i}.layer.*``
    prefixed keys.  All tensor shapes are identical; only key names change.
    """
    hf_sd: dict[str, torch.Tensor] = {}

    # Embeddings (Wan has 1 copy; UMT5 ties shared + encoder.embed_tokens)
    hf_sd["shared.weight"] = wan_sd["token_embedding.weight"]
    hf_sd["encoder.embed_tokens.weight"] = wan_sd["token_embedding.weight"]

    # Final layer norm
    hf_sd["encoder.final_layer_norm.weight"] = wan_sd["norm.weight"]

    # Per-block weights
    num_layers = _WAN_UMT5_CONFIG.num_layers
    for i in range(num_layers):
        src = f"blocks.{i}"
        dst = f"encoder.block.{i}"

        # Self-attention
        for proj in ("q", "k", "v", "o"):
            hf_sd[f"{dst}.layer.0.SelfAttention.{proj}.weight"] = wan_sd[f"{src}.attn.{proj}.weight"]
        hf_sd[f"{dst}.layer.0.SelfAttention.relative_attention_bias.weight"] = wan_sd[
            f"{src}.pos_embedding.embedding.weight"
        ]
        hf_sd[f"{dst}.layer.0.layer_norm.weight"] = wan_sd[f"{src}.norm1.weight"]

        # Gated feed-forward (gate.0 → wi_0, fc1 → wi_1, fc2 → wo)
        hf_sd[f"{dst}.layer.1.DenseReluDense.wi_0.weight"] = wan_sd[f"{src}.ffn.gate.0.weight"]
        hf_sd[f"{dst}.layer.1.DenseReluDense.wi_1.weight"] = wan_sd[f"{src}.ffn.fc1.weight"]
        hf_sd[f"{dst}.layer.1.DenseReluDense.wo.weight"] = wan_sd[f"{src}.ffn.fc2.weight"]
        hf_sd[f"{dst}.layer.1.layer_norm.weight"] = wan_sd[f"{src}.norm2.weight"]

    return hf_sd


def _load_wan_t5_as_umt5(
    model: UMT5EncoderModel,
    checkpoint_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> UMT5EncoderModel:
    """Load a Wan2.2 T5 ``.pth`` checkpoint as a ``UMT5EncoderModel``."""
    logger.info("Loading Wan T5 checkpoint: %s", checkpoint_path)
    wan_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    hf_sd = _convert_wan_t5_state_dict(wan_sd)

    model.load_state_dict(hf_sd, assign=True)
    model = model.to(dtype=dtype).eval()

    return model


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Wan22S2VPipeline(
    nn.Module, SupportImageInput, SupportAudioInput, CFGParallelMixin, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    """
    Wan2.2 Speech-to-Video Pipeline.

    Migrated from ``Wan2.2/wan/speech2video.py`` (``WanS2V``).

    Key differences from I2V:
      - Single transformer (``WanModel_S2V``), no MoE boundary switching.
      - Audio conditioning via wav2vec2 features injected into transformer.
      - Motion frame autoregressive chaining across multiple clips.
      - Optional pose video conditioning.
      - Reference image encoded as separate ``ref_latents`` tokens (not
        channel-concatenated like I2V).
    """

    # Default config values from Wan2.2/wan/configs/wan_s2v_14B.py
    _DEFAULT_MOTION_FRAMES = 73
    _DEFAULT_INFER_FRAMES = 80
    _DEFAULT_FPS = 16

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        model_path = _resolve_model_path(model)
        is_diffusers = _is_diffusers_format(model_path)

        if is_diffusers:
            self._init_diffusers_format(model_path, dtype)
        else:
            self._init_original_format(model_path, dtype)

        # -- Scheduler --
        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 3.0  # S2V default
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        # -- VAE scale factors --
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8

        # -- Resolution divisor for padding (VAE spatial scale × transformer patch size) --
        # S2V uses patch_size=(1,2,2) in patch_embedding (3D conv with stride 2 in spatial dims)
        # Total divisor = vae_scale_factor_spatial × patch_size_spatial = 8 × 2 = 16
        self.resolution_divisor = self.vae_scale_factor_spatial * 2  # 2 from patch_size spatial stride

        # -- S2V specific config --
        # These mirror Wan2.2/wan/configs/wan_s2v_14B.py
        self.motion_frames = getattr(od_config, "motion_frames", self._DEFAULT_MOTION_FRAMES)
        self.drop_first_motion = getattr(od_config, "drop_first_motion", True)
        self.fps = getattr(od_config, "fps", self._DEFAULT_FPS) or self._DEFAULT_FPS
        self.audio_sample_m = 0

        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # ------------------------------------------------------------------
    # Format-specific initialization
    # ------------------------------------------------------------------

    def _init_diffusers_format(self, model_path: str, dtype: torch.dtype) -> None:
        """Initialize from a diffusers-format model with subfolders."""
        # -- Weight sources for transformer (loaded via load_weights) --
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # -- Text encoder --
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)

        # -- VAE --
        self.vae = DistributedAutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype).to(
            self.device
        )

        # -- Audio encoder (wav2vec2) --
        wav2vec_path = os.path.join(model_path, "wav2vec2-large-xlsr-53-english")
        if not os.path.isdir(wav2vec_path):
            wav2vec_path = "facebook/wav2vec2-large-xlsr-53"
        self.audio_processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        self.audio_model = Wav2Vec2ForCTC.from_pretrained(wav2vec_path).to(self.device)
        self.audio_model.eval()

        # -- Transformer (S2V variant, native vllm-omni ops) --
        transformer_config = load_transformer_config(model_path, "transformer", local_files_only=True)
        self.transformer = create_s2v_transformer_from_config(transformer_config)

    def _init_original_format(self, model_path: str, dtype: torch.dtype) -> None:
        """Initialize from the original Wan2.2 checkpoint layout.

        Expected structure::

            model_path/
            ├── config.json                            # transformer config
            ├── diffusion_pytorch_model-*.safetensors  # transformer weights
            ├── Wan2.1_VAE.pth                         # VAE weights
            ├── models_t5_umt5-xxl-enc-bf16.pth        # T5 encoder weights
            ├── google/umt5-xxl/                       # tokenizer
            └── wav2vec2-large-xlsr-53-english/         # audio encoder
        """
        # -- Weight sources: transformer weights live at root --
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # -- Text encoder (original Wan2.2 T5, converted to UMT5EncoderModel) --
        tokenizer_path = os.path.join(model_path, "google", "umt5-xxl")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        t5_checkpoint = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
        self.text_encoder = UMT5EncoderModel(_WAN_UMT5_CONFIG)
        self.text_encoder = _load_wan_t5_as_umt5(self.text_encoder, t5_checkpoint, dtype=dtype).to(self.device)

        # -- VAE (original Wan2.1 VAE, loaded via diffusers from_single_file) --
        vae_pth = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = DistributedAutoencoderKLWan.from_single_file(vae_pth, torch_dtype=dtype)
        self.vae.init_distributed()
        self.vae = self.vae.to(self.device)

        # -- Audio encoder (wav2vec2) --
        wav2vec_path = os.path.join(model_path, "wav2vec2-large-xlsr-53-english")
        if not os.path.isdir(wav2vec_path):
            wav2vec_path = "facebook/wav2vec2-large-xlsr-53"
        self.audio_processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        self.audio_model = Wav2Vec2ForCTC.from_pretrained(wav2vec_path).to(self.device)
        self.audio_model.eval()

        # -- Transformer (S2V variant, native vllm-omni ops) --
        transformer_config = load_transformer_config(model_path, ".", local_files_only=True)
        self.transformer = create_s2v_transformer_from_config(transformer_config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    # ------------------------------------------------------------------
    # VAE latent normalization
    # ------------------------------------------------------------------

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply (latent - mean) * (1/std) normalization.

        The original Wan2_1_VAE does this internally, but diffusers'
        AutoencoderKLWan stores the values in config without applying them.
        """
        if not hasattr(self.vae, "config") or not hasattr(self.vae.config, "latents_mean"):
            return latents
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return (latents - mean) * inv_std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse normalization before VAE decode: latent / (1/std) + mean."""
        if not hasattr(self.vae, "config") or not hasattr(self.vae.config, "latents_mean"):
            return latents
        mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        return latents / inv_std + mean

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text prompts using T5 text encoder.

        Identical to Wan22I2VPipeline.encode_prompt.
        """
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def encode_audio(
        self,
        audio_path: str | np.ndarray,
        infer_frames: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        """Extract wav2vec2 audio features and bucket them to video frame-rate.

        Ported from ``WanS2V.encode_audio`` and ``AudioEncoder``.

        Args:
            audio_path: Path to audio file, or raw numpy audio array (16 kHz).
            infer_frames: Number of video frames per clip.

        Returns:
            ``(audio_embed_bucket, num_repeat, target_video_frames)`` — audio
            embeddings aligned to video frames with shape
            ``[1, num_layers, C_a, T_total]``, the number of clips needed, and
            the exact number of video frames to match audio duration.
        """
        device = device or self.device
        dtype = dtype or self.transformer.dtype

        # Extract wav2vec2 features
        if isinstance(audio_path, np.ndarray):
            audio_input = audio_path.astype(np.float32)
            sample_rate = 16000
        else:
            audio_input, sample_rate = load_audio(audio_path, sr=16000, mono=True)
        input_values = self.audio_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
        # Match dtype of audio model weights (may be bfloat16 in bundled checkpoints)
        audio_model_dtype = next(self.audio_model.parameters()).dtype
        res = self.audio_model(
            input_values.to(device=self.audio_model.device, dtype=audio_model_dtype),
            output_hidden_states=True,
        )
        feat = torch.cat(res.hidden_states)  # [num_layers, T, D]
        feat = _linear_interpolation(feat, input_fps=50, output_fps=_AUDIO_VIDEO_RATE)

        # Bucket to video fps
        audio_embed_bucket, num_repeat, target_video_frames = _get_audio_embed_bucket_fps(
            feat.to(torch.float32),
            fps=self.fps,
            batch_frames=infer_frames,
            m=self.audio_sample_m,
        )
        audio_embed_bucket = audio_embed_bucket.to(device, dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)  # [1, T_bucket, ...]

        # Permute to match model expectation: [B, num_layers, C_a, T_a]
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)

        return audio_embed_bucket, num_repeat, target_video_frames

    def encode_ref_image(
        self,
        image: PIL.Image.Image,
        height: int,
        width: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """VAE-encode the reference image into latent space.

        Ported from ``WanS2V.generate`` lines 492-499.

        Args:
            image: Reference PIL image (already resized/cropped).
            height: Target height.
            width: Target width.

        Returns:
            ``ref_latents`` with shape ``[1, C, 1, H_lat, W_lat]``.
        """
        device = device or self.device

        tensor_trans = transforms.ToTensor()
        ref_pixel_values = tensor_trans(image)  # [C, H, W]
        ref_pixel_values = ref_pixel_values.unsqueeze(1).unsqueeze(0) * 2 - 1.0  # [1, C, 1, H, W]
        ref_pixel_values = ref_pixel_values.to(dtype=self.vae.dtype, device=device)
        ref_latents = retrieve_latents(self.vae.encode(ref_pixel_values), sample_mode="argmax")
        return self._normalize_latents(ref_latents)

    def prepare_motion_latents(
        self,
        motion_pixels: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """VAE-encode motion frame pixels into latent space.

        Args:
            motion_pixels: Shape ``[1, C, T_motion, H, W]``, pixel values in
                ``[-1, 1]``.

        Returns:
            ``motion_latents`` with shape ``[1, C_lat, T_lat, H_lat, W_lat]``.
        """
        device = device or self.device
        motion_pixels = motion_pixels.to(dtype=self.vae.dtype, device=device)
        motion_latents = retrieve_latents(self.vae.encode(motion_pixels), sample_mode="argmax")
        return self._normalize_latents(motion_latents)

    def prepare_latents(
        self,
        infer_frames: int,
        height: int,
        width: int,
        motion_frames: int,
        lat_motion_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate random noise latents for one S2V clip.

        Ported from ``WanS2V.generate`` lines 544-556.

        The target latent shape accounts for both the denoised portion and
        excludes the motion prefix (which is prepended separately at decode
        time).

        Returns:
            Noise tensor with shape ``[C, T_lat, H_lat, W_lat]`` (no batch
            dim, matching the original ``WanModel_S2V`` convention).
        """
        lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
        lat_h = height // self.vae_scale_factor_spatial
        lat_w = width // self.vae_scale_factor_spatial
        shape = (16, lat_target_frames, lat_h, lat_w)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        prompt: str | None,
        image: PIL.Image.Image | None,
        audio_path: str | None,
        height: int,
        width: int,
        prompt_embeds: torch.Tensor | None = None,
    ):
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if image is None:
            raise ValueError("Reference image is required for S2V generation.")
        if audio_path is None:
            raise ValueError("Audio path is required for S2V generation.")
        if height % self.resolution_divisor != 0 or width % self.resolution_divisor != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {self.resolution_divisor}, got {height} and {width}."
            )

    # ------------------------------------------------------------------
    # Noise prediction
    # ------------------------------------------------------------------

    def predict_noise(self, current_model: nn.Module | None = None, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the S2V transformer to predict noise.

        The S2V model returns a list of tensors (one per batch element).
        We take the first element since S2V processes one sample at a time.
        """
        if current_model is None:
            current_model = self.transformer
        # WanModel_S2V's norm layers compute in float32; autocast ensures
        # the float32→bfloat16 casts happen automatically (matching the
        # original Wan2.2 inference path).
        param_dtype = next(current_model.parameters()).dtype
        with torch.amp.autocast(self.device.type, dtype=param_dtype):
            result = current_model(**kwargs)
        # WanModel_S2V.forward returns a list of tensors
        if isinstance(result, list):
            return result[0]
        return result[0] if isinstance(result, tuple) else result

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        clip_generator: torch.Generator,
        dtype: torch.dtype,
        device: torch.device,
        max_seq_len: int,
        cond_latents: torch.Tensor,
        input_motion_latents: torch.Tensor,
        ref_latents: torch.Tensor,
        motion_frames: list[int],
        drop_first_motion: bool,
        positive_audio_emb: torch.Tensor,
        negative_audio_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Denoising diffusion loop for one S2V clip.

        Args:
            latents: Initial noise tensor [C, T, H, W]
            timesteps: Denoising timesteps
            prompt_embeds: Text embeddings [1, seq_len, dim]
            negative_prompt_embeds: Negative text embeddings (optional)
            guidance_scale: CFG scale
            clip_generator: Random generator for this clip
            dtype: Data type for computation
            device: Device for computation
            max_seq_len: Maximum sequence length for transformer
            cond_latents: Pose condition latents [1, 16, T, H, W]
            input_motion_latents: Motion latents from previous clip
            ref_latents: Reference image latents
            motion_frames: Motion frame counts [pixel_frames, latent_frames]
            drop_first_motion: Whether to drop first motion frames (first clip only)
            positive_audio_emb: Precomputed audio embeddings for positive prompt
            negative_audio_emb: Precomputed audio embeddings for negative prompt (optional)

        Returns:
            Denoised latents [C, T, H, W]
        """
        do_true_cfg = self.do_classifier_free_guidance and negative_prompt_embeds is not None

        with self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                self._current_timestep = t

                latent_model_input = [latents.to(device)]
                timestep = t.unsqueeze(0).to(device) if t.dim() == 0 else t.to(device)

                # -- Positive (conditional) prediction kwargs --
                positive_kwargs = {
                    "x": latent_model_input,
                    "t": timestep,
                    "context": prompt_embeds[0:1],
                    "seq_len": max_seq_len,
                    "cond_states": cond_latents,
                    "motion_latents": input_motion_latents,
                    "ref_latents": ref_latents,
                    "motion_frames": motion_frames,
                    "drop_motion_frames": drop_first_motion,
                    "audio_emb": positive_audio_emb,
                }

                if do_true_cfg:
                    negative_kwargs = {
                        "x": latent_model_input,
                        "t": timestep,
                        "context": negative_prompt_embeds[0:1],
                        "seq_len": max_seq_len,
                        "cond_states": cond_latents,
                        "motion_latents": input_motion_latents,
                        "ref_latents": ref_latents,
                        "motion_frames": motion_frames,
                        "drop_motion_frames": drop_first_motion,
                        "audio_emb": negative_audio_emb,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                # Scheduler step
                latents = self.scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents.unsqueeze(0),
                    return_dict=False,
                    generator=clip_generator,
                )[0].squeeze(0)

                pbar.update()

        self._current_timestep = None
        return latents

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        image: PIL.Image.Image | None = None,
        audio_path: str | None = None,
        height: int = 704,
        width: int = 1024,
        infer_frames: int | None = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        num_repeat: int | None = None,
        pose_video: str | None = None,
        init_first_frame: bool = False,
        output_type: str | None = "np",
        generator: torch.Generator | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        """Run S2V generation — may produce multiple autoregressive clips.

        This method mirrors ``WanS2V.generate()``, reorganized into the
        vLLM-Omni pipeline structure.
        """
        # ---- Extract params from request ----
        if len(req.prompts) > 1:
            raise ValueError("S2V only supports a single prompt per request.")

        if len(req.prompts) == 1:
            prompt_data = req.prompts[0]
            prompt = prompt_data if isinstance(prompt_data, str) else prompt_data.get("prompt")
            negative_prompt = None if isinstance(prompt_data, str) else prompt_data.get("negative_prompt")

            multi_modal_data = prompt_data.get("multi_modal_data", {}) if not isinstance(prompt_data, str) else {}
            additional_info = prompt_data.get("additional_information", {}) if not isinstance(prompt_data, str) else {}

            if image is None:
                raw_image = multi_modal_data.get("image", None)
                if isinstance(raw_image, str):
                    image = PIL.Image.open(raw_image).convert("RGB")
                elif isinstance(raw_image, PIL.Image.Image):
                    image = raw_image

            if audio_path is None:
                audio_path = additional_info.get("audio_path")
                if audio_path is None:
                    audio_path = multi_modal_data.get("audio")

            if pose_video is None:
                pose_video = additional_info.get("pose_video")

            init_first_frame = additional_info.get("init_first_frame", init_first_frame)

        if negative_prompt is None:
            negative_prompt = _S2V_DEFAULT_NEG_PROMPT

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        # num_frames defaults to 1 (image-model sentinel); S2V needs many frames.
        req_num_frames = req.sampling_params.num_frames
        infer_frames = (
            infer_frames
            or (req_num_frames if req_num_frames and req_num_frames > 1 else None)
            or self._DEFAULT_INFER_FRAMES
        )
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        # ---- Validate ----
        self.check_inputs(prompt, image, audio_path, height, width, prompt_embeds)

        device = self.device
        dtype = self.transformer.dtype if hasattr(self.transformer, "dtype") else torch.bfloat16

        if generator is None:
            generator = req.sampling_params.generator
        seed = req.sampling_params.seed
        if generator is None and seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        if seed is None:
            seed = 0

        # ---- 1. Text encoding ----
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        # ---- 2. Audio encoding ----
        audio_emb, audio_num_repeat, target_video_frames = self.encode_audio(
            audio_path, infer_frames=infer_frames, device=device, dtype=dtype
        )
        if num_repeat is None or num_repeat > audio_num_repeat:
            num_repeat = audio_num_repeat

        # Load raw audio waveform for muxing into output video
        if isinstance(audio_path, np.ndarray):
            raw_audio_waveform = audio_path.astype(np.float32)
            raw_audio_sr = 16000
        else:
            raw_audio_waveform, raw_audio_sr = load_audio(audio_path, sr=None, mono=True)

        # ---- 3. Reference image encoding ----
        ref_latents = self.encode_ref_image(image, height, width, device=device).to(dtype=dtype)

        # ---- 4. Initial motion latents (zeros) ----
        motion_frames = self.motion_frames
        motion_pixels = torch.zeros(
            [1, 3, motion_frames, height, width],
            dtype=dtype,
            device=device,
        )
        drop_first_motion = self.drop_first_motion
        if init_first_frame:
            drop_first_motion = False
            # Place reference image in the last 6 frames of motion
            tensor_trans = transforms.ToTensor()
            ref_tensor = tensor_trans(image).to(device=device, dtype=dtype) * 2 - 1.0  # [C, H, W]
            motion_pixels[:, :, -6:] = ref_tensor.unsqueeze(0).unsqueeze(2).expand(-1, -1, 6, -1, -1)

        motion_latents = self.prepare_motion_latents(motion_pixels, device=device).to(dtype=dtype)

        lat_motion_frames = (motion_frames + 3) // 4

        # ---- 5. Pose conditioning (optional) ----
        # Pose conditioning is encoded via VAE ahead of the denoising loop.
        # When no pose video is provided, zero-valued cond_latents are used.
        # Full pose video reading/encoding (WanS2V.load_pose_cond) would be
        # ported here in a future iteration when pose support is needed.
        # For now, we support the default zero-condition path.

        # ---- 6. Multi-clip autoregressive denoising ----
        clips: list[torch.Tensor] = []
        # Keep a pixel-space buffer of the trailing motion_frames for the
        # autoregressive connection between clips.
        videos_last_frames = torch.zeros([1, 3, motion_frames, height, width], dtype=dtype, device=device)

        for r in range(num_repeat):
            # Per-clip seed
            clip_seed = seed + r
            clip_generator = torch.Generator(device=device).manual_seed(clip_seed)

            # -- Noise --
            latents = self.prepare_latents(
                infer_frames=infer_frames,
                height=height,
                width=width,
                motion_frames=motion_frames,
                lat_motion_frames=lat_motion_frames,
                dtype=dtype,
                device=device,
                generator=clip_generator,
            )

            # -- Scheduler --
            self.scheduler.set_timesteps(num_steps, device=device)
            timesteps = self.scheduler.timesteps
            self._num_timesteps = len(timesteps)

            # -- Slice audio for this clip --
            left_idx = r * infer_frames
            right_idx = left_idx + infer_frames
            audio_input = audio_emb[..., left_idx:right_idx]

            # -- Pose condition for this clip --
            # Default: zero condition (no pose driving)
            # Shape: [B=1, C=16, T, H, W] — passed to transformer which iterates over batch dim
            lat_h = height // self.vae_scale_factor_spatial
            lat_w = width // self.vae_scale_factor_spatial
            lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
            cond_latents = torch.zeros([1, 16, lat_target_frames, lat_h, lat_w], dtype=dtype, device=device)

            # -- Clone motion latents for this clip --
            input_motion_latents = motion_latents.clone()

            # Max sequence length for the transformer
            max_seq_len = int(np.prod([lat_target_frames, lat_h, lat_w]) // 4)

            # -- Precompute audio embeddings once per clip --
            mf = [motion_frames, lat_motion_frames]
            # When CPU offload is active the transformer lives on CPU until
            # its __call__ hook fires.  encode_audio() bypasses __call__,
            # so move the audio sub-module to GPU explicitly.
            _audio_enc = getattr(self.transformer, "casual_audio_encoder", None)
            _audio_on_cpu = _audio_enc is not None and next(_audio_enc.parameters()).device.type == "cpu"
            if _audio_on_cpu:
                _audio_enc.to(device)
            param_dtype = next(self.transformer.parameters()).dtype
            with torch.amp.autocast(device.type, dtype=param_dtype):
                positive_audio_emb = self.transformer.encode_audio(audio_input, mf)
            do_true_cfg = self.do_classifier_free_guidance and negative_prompt_embeds is not None
            if do_true_cfg:
                with torch.amp.autocast(device.type, dtype=param_dtype):
                    negative_audio_emb = self.transformer.encode_audio(0.0 * audio_input, mf)
            else:
                negative_audio_emb = None
            if _audio_on_cpu:
                _audio_enc.to("cpu")

            # -- Denoising loop --
            latents = self.diffuse(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                clip_generator=clip_generator,
                dtype=dtype,
                device=device,
                max_seq_len=max_seq_len,
                cond_latents=cond_latents,
                input_motion_latents=input_motion_latents,
                ref_latents=ref_latents,
                motion_frames=mf,
                drop_first_motion=drop_first_motion and r == 0,
                positive_audio_emb=positive_audio_emb,
                negative_audio_emb=negative_audio_emb,
            )

            # ---- Decode this clip ----
            latents_for_decode = latents.unsqueeze(0)  # [1, C, T, H, W]
            if not (drop_first_motion and r == 0):
                decode_latents = torch.cat([motion_latents, latents_for_decode], dim=2)
            else:
                decode_latents = torch.cat([ref_latents, latents_for_decode], dim=2)

            decode_latents = self._denormalize_latents(decode_latents)
            decode_latents = decode_latents.to(self.vae.dtype)
            clip_video = self.vae.decode(decode_latents, return_dict=False)[0]  # [1, C, T, H, W]

            # Handle VAE patch parallel: only rank 0 gets result, broadcast to all ranks
            # This is needed for S2V's autoregressive loop where all ranks need the decoded frames
            if clip_video.numel() == 0:
                # Non-rank0 received empty tensor from patch parallel decode
                # Use the same broadcast mechanism as the VAE patch parallel code
                import torch.distributed as dist

                total_frames = decode_latents.shape[2]

                # Create buffer for broadcast
                clip_video = torch.empty(
                    (1, 3, total_frames, height, width),
                    device=decode_latents.device,
                    dtype=decode_latents.dtype,
                )

                # Get the VAE's patch parallel group (same one used in decode)
                vae_pp_group = getattr(self.vae, "_vae_pp_group", None)
                if vae_pp_group is not None:
                    # Broadcast using the same group as VAE patch parallel
                    dist.broadcast(clip_video, src=0, group=vae_pp_group)

            # Trim to the infer_frames of interest
            clip_video = clip_video[:, :, -infer_frames:]
            if drop_first_motion and r == 0:
                # Drop the first 3 frames (artifact from ref_latents prepend)
                clip_video = clip_video[:, :, 3:]

            # ---- Update motion for next clip (autoregressive) ----
            overlap_frames_num = min(motion_frames, clip_video.shape[2])
            videos_last_frames = torch.cat(
                [videos_last_frames[:, :, overlap_frames_num:], clip_video[:, :, -overlap_frames_num:]],
                dim=2,
            )
            videos_last_frames = videos_last_frames.to(dtype=dtype, device=device)
            # Only prepare motion latents if there's another clip coming
            if r < num_repeat - 1:
                motion_latents = self.prepare_motion_latents(videos_last_frames, device=device).to(dtype=dtype)

            clips.append(clip_video.cpu())

            # Free VRAM between clips
            if current_omni_platform.is_available():
                current_omni_platform.empty_cache()

        # ---- Concatenate all clips ----
        output = torch.cat(clips, dim=2)  # [1, C, T_total, H, W]

        return DiffusionOutput(
            output=(output, raw_audio_waveform, raw_audio_sr),
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
