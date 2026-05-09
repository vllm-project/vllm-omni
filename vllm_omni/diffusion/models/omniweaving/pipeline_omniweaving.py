# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
    HunyuanVideo15Transformer3DModel,
)
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    extract_glyph_texts,
    format_text_input,
    get_hunyuan_video_15_post_process_func,
    retrieve_latents,
)
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.t5_encoder import T5EncoderModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

# Official OmniWeaving I2V `prompt_mode=2` system prompt (see `OmniWeaving/hyvideo/models/text_encoders/__init__.py`).
OMNIWEAVING_I2V_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text instruction should alter the image "
    "to introduce motion and evolution over time. Generate a video using this image as the first frame that "
    "meets the user's requirements, ensuring the specified elements evolve or move in a way that fulfills the "
    "text description while maintaining consistency."
)


def get_omniweaving_post_process_func(od_config: OmniDiffusionConfig):
    return get_hunyuan_video_15_post_process_func(od_config)


def get_omniweaving_pre_process_func(od_config: OmniDiffusionConfig):
    # Official aligned I2V 480p: 480×848 (26:15). Use the same area target when inferring WxH from the image.
    max_area = 480 * 848
    divisor = 16

    def pre_process_func(req: OmniDiffusionRequest) -> OmniDiffusionRequest:
        if not req.prompts:
            return req
        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            return req

        multi_modal_data = prompt_data.get("multi_modal_data", {})
        raw_image = multi_modal_data.get("image", None)
        if raw_image is None:
            return req
        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        if isinstance(raw_image, str):
            image = PIL.Image.open(raw_image).convert("RGB")
        elif isinstance(raw_image, PIL.Image.Image):
            image = raw_image.convert("RGB")
        else:
            return req

        height, width = req.sampling_params.height, req.sampling_params.width
        if height is None or width is None:
            w, h = image.size
            aspect = w / h
            target_h = int((max_area / aspect) ** 0.5)
            target_w = int(target_h * aspect)
            req.sampling_params.height = height or ((target_h + divisor - 1) // divisor * divisor)
            req.sampling_params.width = width or ((target_w + divisor - 1) // divisor * divisor)
        return req

    return pre_process_func


class OmniWeavingPipeline(
    nn.Module,
    CFGParallelMixin,
    SupportImageInput,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    support_image_input = True
    color_format = "RGB"

    # Default external sub-model paths
    DEFAULT_QWEN_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
    DEFAULT_SIGLIP_PATH = "google/siglip-so400m-patch14-384"
    DEFAULT_T2V_PATH = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

    @staticmethod
    def _resolve_external_model_path(
        od_config: OmniDiffusionConfig,
        arg_name: str,
        default_path: str,
    ) -> str:
        custom_args = od_config.custom_pipeline_args or {}
        if isinstance(custom_args, dict):
            value = custom_args.get(arg_name)
            if isinstance(value, str) and value:
                return value
        return default_path

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)
        self.qwen_path = self._resolve_external_model_path(od_config, "qwen_path", self.DEFAULT_QWEN_PATH)
        self.siglip_path = self._resolve_external_model_path(od_config, "siglip_path", self.DEFAULT_SIGLIP_PATH)
        self.t2v_path = self._resolve_external_model_path(od_config, "t2v_path", self.DEFAULT_T2V_PATH)
        # Only force offline HF when the *Qwen* path itself is a local directory/file; do not use the
        # OmniWeaving checkpoint's local_files_only (that would break Hub IDs when the main model is local).
        qwen_local_files_only: bool = os.path.isdir(self.qwen_path)

        custom_args = od_config.custom_pipeline_args if isinstance(od_config.custom_pipeline_args, dict) else {}
        self._mllm_i2v_use_vision: bool = bool(custom_args.get("mllm_i2v_use_vision", True))
        self._mllm_i2v_crop_start: int = int(custom_args.get("mllm_i2v_crop_start", 92))
        self._mllm_i2v_vision_token_budget: int = int(custom_args.get("mllm_i2v_vision_token_budget", 400))
        # `hunyuan_video_pipeline` throttles MLLM vision to max edge 560 before `prepare_input` (I2V).
        self._mllm_i2v_qwen_thumbnail_max: int = int(custom_args.get("mllm_i2v_qwen_thumbnail_max", 560))
        # `TextEncoder.encode(..., setclip=True)`: after crop_start, drop vision
        # prefix up to last `token_id` (Qwen2-VL).
        self._mllm_i2v_setclip: bool = bool(custom_args.get("mllm_i2v_setclip", True))
        self._mllm_i2v_setclip_token_id: int = int(custom_args.get("mllm_i2v_setclip_token_id", 151653))
        # Official-style bench: use slow Qwen2-VL image path unless explicitly
        # overridden (OMNIWEAVING_I2V_MI2V_PERF.md).
        self._qwen_image_processor_use_fast: bool = bool(custom_args.get("qwen_image_processor_use_fast", False))
        # VAE first-frame `cond_latents`: diffusers `resize_mode="default"` =
        # independent scale to H×W (aspect not kept).
        # Portrait or AR-mismatched refs get stretched, harming I2V temporal
        # consistency. Prefer letterboxing ("fill").
        self._i2v_cond_resize_mode: str = str(custom_args.get("i2v_cond_resize_mode", "fill"))
        trust = bool(getattr(od_config, "trust_remote_code", True))

        self.qwen_processor: Any | None = None
        try:
            self.qwen_processor = AutoProcessor.from_pretrained(
                self.qwen_path,
                local_files_only=qwen_local_files_only,
                trust_remote_code=trust,
            )
            self.tokenizer = self.qwen_processor.tokenizer
            _ip = getattr(self.qwen_processor, "image_processor", None)
            if _ip is not None and hasattr(_ip, "use_fast") and not self._qwen_image_processor_use_fast:
                _ip.use_fast = False
        except Exception as e:
            logger.warning(
                "AutoProcessor could not be loaded for Qwen2.5-VL (%s); I2V multimodal Qwen path disabled. %s",
                self.qwen_path,
                e,
            )
            self.qwen_processor = None
            self.tokenizer = Qwen2Tokenizer.from_pretrained(self.qwen_path, local_files_only=qwen_local_files_only)

        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.qwen_path,
            torch_dtype=dtype,
            local_files_only=qwen_local_files_only,
            trust_remote_code=trust,
        ).to(self.device)

        ckpt_dir = os.path.join(model, "text_encoder")
        safetensors_files = glob.glob(os.path.join(ckpt_dir, "**", "*.safetensors"), recursive=True)
        if safetensors_files:
            from safetensors.torch import load_file

            for f in safetensors_files:
                te_weights = load_file(f)
                if "__metadata__" in te_weights:
                    del te_weights["__metadata__"]
                self.mllm.load_state_dict(te_weights, strict=False)

        self._process_vision_info = None
        try:
            from qwen_vl_utils import process_vision_info as _pvi

            self._process_vision_info = _pvi
        except ImportError:
            logger.warning(
                "qwen_vl_utils is not installed; OmniWeaving I2V will not feed images into Qwen. "
                "Install with: pip install qwen-vl-utils"
            )

        try:
            self.tokenizer_2 = AutoTokenizer.from_pretrained(
                model, subfolder="tokenizer_2", local_files_only=local_files_only
            )
            t5_config = AutoConfig.from_pretrained(model, subfolder="text_encoder_2", local_files_only=local_files_only)
            self.text_encoder_2 = T5EncoderModel(t5_config, prefix="text_encoder_2").to(dtype=dtype, device=self.device)
            self.has_t5_2 = True
        except Exception:
            self.tokenizer_2 = None
            self.text_encoder_2 = None
            self.has_t5_2 = False
            self.t5_2_d_model = 1472

        try:
            self.image_encoder = SiglipVisionModel.from_pretrained(
                model,
                subfolder="image_encoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            ).to(self.device)
            self.feature_extractor = SiglipImageProcessor.from_pretrained(
                model,
                subfolder="feature_extractor",
                local_files_only=local_files_only,
            )
        except Exception:
            self.image_encoder = SiglipVisionModel.from_pretrained(self.siglip_path, torch_dtype=dtype).to(self.device)
            self.feature_extractor = SiglipImageProcessor.from_pretrained(self.siglip_path)

        self.vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            self.t2v_path, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.t2v_path, subfolder="scheduler")
        if od_config.flow_shift is not None:
            self.scheduler._shift = od_config.flow_shift
        # `Omni` sets `od_config.flow_shift` once; offline bench may vary per case via
        # `sampling_params.extra_args['flow_shift']`. Reset in `forward` before `set_timesteps`.
        self._default_scheduler_shift = float(getattr(self.scheduler, "_shift", 1.0))

        # When model_index.json is absent (Tencent-format repos), OmniDiffusionConfig falls back
        # to the top-level config.json which lacks transformer-specific keys like `deepstack`.
        # Patch only the missing keys from transformer/config.json into the existing tf_model_config.
        if not getattr(od_config.tf_model_config, "deepstack", None):
            import json as _json

            _tf_cfg_path = os.path.join(od_config.model, "transformer", "config.json")
            if os.path.isfile(_tf_cfg_path):
                _tf_cfg = _json.loads(open(_tf_cfg_path).read())
                if _tf_cfg.get("deepstack"):
                    od_config.tf_model_config.params["deepstack"] = _tf_cfg["deepstack"]
                    logger.info(
                        "OmniWeaving: patched deepstack=%s from transformer/config.json",
                        od_config.tf_model_config.get("deepstack"),
                    )

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, HunyuanVideo15Transformer3DModel)
        self.transformer = HunyuanVideo15Transformer3DModel(od_config=od_config, **transformer_kwargs)
        self.use_meanflow = getattr(od_config.tf_model_config, "use_meanflow", False)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="text_encoder_2",
                revision=None,
                prefix="text_encoder_2.",
                fall_back_to_pt=True,
            ),
        ]

        self.vae_scale_factor_temporal = (
            self.vae.temporal_compression_ratio if hasattr(self.vae, "temporal_compression_ratio") else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.spatial_compression_ratio if hasattr(self.vae, "spatial_compression_ratio") else 16
        )
        self.num_channels_latents = self.vae.config.latent_channels if hasattr(self.vae, "config") else 32

        self.system_message = (
            "You are a helpful assistant. Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
            "4. background environment, light, style and atmosphere. "
            "5. camera angles, movements, and transitions used in the video."
        )

        self.prompt_template_encode_start_idx = 108
        self.tokenizer_max_length = 1000
        self.tokenizer_2_max_length = 256
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )
        self._logged_i2v_text_only_mllm = False

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def _mllm_inputs_to_device(self, proc_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
        """Map processor outputs to Qwen2.5-VL forward kwargs; move tensors to device with correct dtype."""
        out: dict[str, Any] = {}
        mllm_dtype = next(self.mllm.model.parameters()).dtype
        for key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "video_grid_thw",
            "pixel_values_videos",
            "second_per_grid_ts",
            "position_ids",
        ):
            if key not in proc_inputs or proc_inputs[key] is None:
                continue
            v = proc_inputs[key]
            if not isinstance(v, torch.Tensor):
                out[key] = v
                continue
            v = v.to(device)
            if key in ("image_grid_thw", "video_grid_thw"):
                out[key] = v.long()
            elif key in ("pixel_values", "pixel_values_videos"):
                out[key] = v.to(dtype=mllm_dtype)
            else:
                out[key] = v
        return out

    @staticmethod
    def _mllm_forward_kwargs(model_kw: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "video_grid_thw",
            "pixel_values_videos",
            "second_per_grid_ts",
            "position_ids",
        }
        return {k: v for k, v in model_kw.items() if k in allowed and v is not None}

    @staticmethod
    def _resize_pil_for_qwen_mllm(img: PIL.Image.Image, max_edge: int) -> PIL.Image.Image:
        if max_edge <= 0:
            return img
        out = img.copy()
        out.thumbnail((max_edge, max_edge))
        return out

    @staticmethod
    def _apply_omniweaving_mllm_setclip(
        last_hidden: torch.Tensor,
        attention: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Port of `TextEncoder.encode(..., setclip=True)` post-crop path (official OmniWeaving)."""
        b, s, d = last_hidden.shape
        device, dtype_h = last_hidden.device, last_hidden.dtype
        idt = input_ids
        all_h: list[torch.Tensor] = []
        all_a: list[torch.Tensor] = []
        for k in range(b):
            mask = idt[k] == token_id
            if not mask.any():
                all_h.append(last_hidden[k])
                all_a.append(attention[k])
                continue
            last_pos = int(mask.nonzero()[-1, 0].item())
            if last_pos < s - 1:
                all_h.append(last_hidden[k, last_pos + 1 :])
                all_a.append(attention[k, last_pos + 1 :])
            else:
                all_h.append(last_hidden[k])
                all_a.append(attention[k])
        max_len = max(t.shape[0] for t in all_h)
        padded_h = torch.zeros(b, max_len, d, device=device, dtype=dtype_h)
        padded_a = torch.zeros(b, max_len, device=device, dtype=attention.dtype)
        for i in range(b):
            c = all_h[i].size(0)
            padded_h[i, :c] = all_h[i]
            padded_a[i, :c] = all_a[i]
        return padded_h, padded_a

    def _get_mllm_prompt_embeds(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        images: list[PIL.Image.Image] | None = None,
        num_hidden_layers_to_skip: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_i2v_vision = (
            self._mllm_i2v_use_vision
            and images is not None
            and len(images) == len(prompt)
            and all(im is not None for im in images)
            and self.qwen_processor is not None
            and self._process_vision_info is not None
        )
        if (
            not use_i2v_vision
            and images is not None
            and any(im is not None for im in images)
            and not self._logged_i2v_text_only_mllm
        ):
            self._logged_i2v_text_only_mllm = True
            logger.info(
                "OmniWeaving I2V: Qwen is running in text-only mode; install qwen-vl-utils and ensure "
                "AutoProcessor loads to align with the official prompt_mode=2 multimodal MLLM path."
            )
        if use_i2v_vision:
            return self._get_mllm_prompt_embeds_i2v_vision(prompt, device, dtype, images, num_hidden_layers_to_skip)
        return self._get_mllm_prompt_embeds_text_only(prompt, device, dtype, num_hidden_layers_to_skip)

    def _get_mllm_prompt_embeds_text_only(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        num_hidden_layers_to_skip: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_formatted = format_text_input(prompt, self.system_message)
        text_inputs = self.tokenizer.apply_chat_template(
            prompt_formatted,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)
        with torch.no_grad():
            outputs = self.mllm(
                input_ids=text_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        if not outputs.hidden_states:
            raise RuntimeError("Qwen2.5-VL forward did not return hidden_states (output_hidden_states=True).")
        prompt_embeds = outputs.hidden_states[-(num_hidden_layers_to_skip + 1)]
        crop_start = self.prompt_template_encode_start_idx
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]
        return (
            torch.clamp(prompt_embeds, min=-65504.0, max=65504.0).to(dtype=dtype),
            prompt_attention_mask,
        )

    def _get_mllm_prompt_embeds_i2v_vision(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        images: list[PIL.Image.Image],
        num_hidden_layers_to_skip: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.qwen_processor is not None and self._process_vision_info is not None
        system_block = {
            "role": "system",
            "content": [{"type": "text", "text": OMNIWEAVING_I2V_SYSTEM_MESSAGE}],
        }
        max_length = self.tokenizer_max_length + self._mllm_i2v_crop_start + self._mllm_i2v_vision_token_budget
        batch_conversations: list[list[dict[str, Any]]] = []
        text_for_proc: list[str] = []
        tmax = self._mllm_i2v_qwen_thumbnail_max
        for p, img in zip(prompt, images, strict=True):
            txt = p if p else " "
            vis = self._resize_pil_for_qwen_mllm(img, tmax)
            user_block: dict[str, Any] = {
                "role": "user",
                "content": [
                    {"type": "image", "image": vis},
                    {"type": "text", "text": txt},
                ],
            }
            conv = [system_block, user_block]
            batch_conversations.append(conv)
            text_for_proc.append(
                self.qwen_processor.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        assert self._process_vision_info is not None
        image_inputs, video_inputs = self._process_vision_info(batch_conversations)
        proc_inputs = self.qwen_processor(
            text=text_for_proc,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        model_kw = self._mllm_inputs_to_device(dict(proc_inputs), device)
        fwd_kw = self._mllm_forward_kwargs(model_kw)
        with torch.no_grad():
            outputs = self.mllm(
                **fwd_kw,
                output_hidden_states=True,
                return_dict=True,
            )
        if not outputs.hidden_states:
            raise RuntimeError("Qwen2.5-VL forward did not return hidden_states (output_hidden_states=True).")
        prompt_embeds = outputs.hidden_states[-(num_hidden_layers_to_skip + 1)]
        attention_mask = fwd_kw["attention_mask"]
        input_ids = fwd_kw["input_ids"]
        crop_start = self._mllm_i2v_crop_start
        if crop_start and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            attention_mask = attention_mask[:, crop_start:]
            input_ids = input_ids[:, crop_start:]
        if self._mllm_i2v_setclip and "pixel_values" in fwd_kw and input_ids.numel() > 0:
            prompt_embeds, attention_mask = self._apply_omniweaving_mllm_setclip(
                prompt_embeds,
                attention_mask,
                input_ids,
                self._mllm_i2v_setclip_token_id,
            )
        return (
            torch.clamp(prompt_embeds, min=-65504.0, max=65504.0).to(dtype=dtype),
            attention_mask,
        )

    def _extract_deepstack_from_outputs(
        self,
        outputs: Any,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor | None,
        crop_start: int,
        deepstack_indices: list[int],
        apply_setclip: bool,
        setclip_token_id: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Extract and process deepstack hidden states from Qwen outputs.

        Returns tensor of shape (num_deepstack_layers, B, L, D).
        Mirrors official TextEncoder.encode deepstack path (lines 506-573).
        """
        # Stack selected hidden layers: (num_layers, B, full_seq, D)
        stacked = torch.stack([outputs.hidden_states[i] for i in deepstack_indices], dim=0)
        # Crop system/template prefix
        stacked = stacked[:, :, crop_start:, :]
        # Zero-out padding positions using the (already-cropped) attention_mask
        stacked = stacked * attention_mask.unsqueeze(0).unsqueeze(-1)

        if apply_setclip and input_ids is not None:
            # Mirror official setclip path: find last occurrence of token_id per batch item,
            # drop everything up to and including it (vision prefix removal).
            num_layers, B, L, D = stacked.shape
            device = stacked.device
            all_slices: list[torch.Tensor] = []
            for k in range(B):
                mask = input_ids[k] == setclip_token_id
                if mask.any():
                    last_pos = int(mask.nonzero()[-1, 0].item())
                    if last_pos < L - 1:
                        all_slices.append(stacked[:, k, last_pos + 1 :])
                    else:
                        all_slices.append(stacked[:, k])
                else:
                    all_slices.append(stacked[:, k])
            max_len = max(s.shape[1] for s in all_slices)
            padded = torch.zeros(num_layers, B, max_len, D, device=device, dtype=stacked.dtype)
            for i, s in enumerate(all_slices):
                padded[:, i, : s.shape[1]] = s
            stacked = padded

        return stacked.to(dtype=dtype)

    def _get_mllm_deepstack_hidden_states_text_only(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        deepstack_indices: list[int],
        num_hidden_layers_to_skip: int = 2,
    ) -> torch.Tensor:
        prompt_formatted = format_text_input(prompt, self.system_message)
        text_inputs = self.tokenizer.apply_chat_template(
            prompt_formatted,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)
        with torch.no_grad():
            outputs = self.mllm(
                input_ids=text_input_ids,
                attention_mask=prompt_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        crop_start = self.prompt_template_encode_start_idx
        cropped_mask = prompt_attention_mask[:, crop_start:]
        return self._extract_deepstack_from_outputs(
            outputs,
            cropped_mask,
            None,
            crop_start,
            deepstack_indices,
            apply_setclip=False,
            setclip_token_id=0,
            dtype=dtype,
        )

    def _get_mllm_deepstack_hidden_states_i2v_vision(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        images: list[PIL.Image.Image],
        deepstack_indices: list[int],
        num_hidden_layers_to_skip: int = 2,
    ) -> torch.Tensor:
        assert self.qwen_processor is not None and self._process_vision_info is not None
        system_block = {
            "role": "system",
            "content": [{"type": "text", "text": OMNIWEAVING_I2V_SYSTEM_MESSAGE}],
        }
        max_length = self.tokenizer_max_length + self._mllm_i2v_crop_start + self._mllm_i2v_vision_token_budget
        batch_conversations: list[list[dict[str, Any]]] = []
        text_for_proc: list[str] = []
        tmax = self._mllm_i2v_qwen_thumbnail_max
        for p, img in zip(prompt, images, strict=True):
            txt = p if p else " "
            vis = self._resize_pil_for_qwen_mllm(img, tmax)
            user_block: dict[str, Any] = {
                "role": "user",
                "content": [
                    {"type": "image", "image": vis},
                    {"type": "text", "text": txt},
                ],
            }
            conv = [system_block, user_block]
            batch_conversations.append(conv)
            text_for_proc.append(
                self.qwen_processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            )
        image_inputs, video_inputs = self._process_vision_info(batch_conversations)
        proc_inputs = self.qwen_processor(
            text=text_for_proc,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        model_kw = self._mllm_inputs_to_device(dict(proc_inputs), device)
        fwd_kw = self._mllm_forward_kwargs(model_kw)
        with torch.no_grad():
            outputs = self.mllm(**fwd_kw, output_hidden_states=True, return_dict=True)
        crop_start = self._mllm_i2v_crop_start
        attention_mask = fwd_kw["attention_mask"]
        input_ids = fwd_kw["input_ids"]
        cropped_mask = attention_mask[:, crop_start:]
        cropped_ids = input_ids[:, crop_start:]
        apply_setclip = self._mllm_i2v_setclip and "pixel_values" in fwd_kw
        return self._extract_deepstack_from_outputs(
            outputs,
            cropped_mask,
            cropped_ids,
            crop_start,
            deepstack_indices,
            apply_setclip=apply_setclip,
            setclip_token_id=self._mllm_i2v_setclip_token_id,
            dtype=dtype,
        )

    def _get_mllm_deepstack_hidden_states(
        self,
        prompt: list[str],
        device: torch.device,
        dtype: torch.dtype,
        deepstack_indices: list[int],
        images: list[PIL.Image.Image] | None = None,
        num_hidden_layers_to_skip: int = 2,
    ) -> torch.Tensor:
        """Return deepstack tensor (num_layers, B, L, D) for the given prompts."""
        use_i2v_vision = (
            self._mllm_i2v_use_vision
            and images is not None
            and len(images) == len(prompt)
            and all(im is not None for im in images)
            and self.qwen_processor is not None
            and self._process_vision_info is not None
        )
        if use_i2v_vision:
            return self._get_mllm_deepstack_hidden_states_i2v_vision(
                prompt, device, dtype, images, deepstack_indices, num_hidden_layers_to_skip
            )
        return self._get_mllm_deepstack_hidden_states_text_only(
            prompt, device, dtype, deepstack_indices, num_hidden_layers_to_skip
        )

    def _get_t5_2_prompt_embeds(
        self, prompt: list[str], device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds_list, prompt_embeds_mask_list = [], []
        for p in prompt:
            glyph_text = extract_glyph_texts(p)
            if glyph_text is None or not self.has_t5_2:
                glyph_text_embeds = torch.zeros(
                    (1, self.tokenizer_2_max_length, self.t5_2_d_model),
                    device=device,
                    dtype=dtype,
                )
                glyph_text_embeds_mask = torch.ones((1, self.tokenizer_2_max_length), device=device, dtype=torch.int64)
            else:
                txt_tokens = self.tokenizer_2(
                    glyph_text,
                    padding="max_length",
                    max_length=self.tokenizer_2_max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(device)
                glyph_text_embeds = self.text_encoder_2(
                    input_ids=txt_tokens.input_ids,
                    attention_mask=txt_tokens.attention_mask.float(),
                )[0].to(device=device, dtype=dtype)
                glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)
            prompt_embeds_list.append(glyph_text_embeds)
            prompt_embeds_mask_list.append(glyph_text_embeds_mask)
        return torch.cat(prompt_embeds_list, dim=0), torch.cat(prompt_embeds_mask_list, dim=0)

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image | None,
        device: torch.device,
        dtype: torch.dtype,
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = False,
    ) -> tuple:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        mllm_images: list[PIL.Image.Image] | None = None
        if image is not None:
            mllm_images = [image] * len(prompt)
        prompt_embeds, prompt_embeds_mask = self._get_mllm_prompt_embeds(prompt, device, dtype, images=mllm_images)
        prompt_embeds_2, prompt_embeds_mask_2 = self._get_t5_2_prompt_embeds(prompt, device, dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(dtype=dtype)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(dtype=dtype)

        deepstack_indices: list[int] = list(getattr(self.transformer, "deep_stack", []))
        deepstack_states: torch.Tensor | None = None
        negative_deepstack_states: torch.Tensor | None = None
        if deepstack_indices:
            deepstack_states = self._get_mllm_deepstack_hidden_states(
                prompt, device, dtype, deepstack_indices, images=mllm_images
            )
            logger.info(
                "OmniWeaving deepstack: indices=%s, states_shape=%s",
                deepstack_indices,
                tuple(deepstack_states.shape),
            )

        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        negative_prompt_embeds_2 = None
        negative_prompt_embeds_mask_2 = None

        if do_classifier_free_guidance:
            neg_p = (
                [negative_prompt]
                if isinstance(negative_prompt, str)
                else (negative_prompt if negative_prompt else [""])
            )
            if mllm_images is not None and len(mllm_images) != len(neg_p):
                neg_mllm_images = [mllm_images[0]] * len(neg_p)
            else:
                neg_mllm_images = mllm_images
            (
                negative_prompt_embeds,
                negative_prompt_embeds_mask,
            ) = self._get_mllm_prompt_embeds(neg_p, device, dtype, images=neg_mllm_images)
            (
                negative_prompt_embeds_2,
                negative_prompt_embeds_mask_2,
            ) = self._get_t5_2_prompt_embeds(neg_p, device, dtype)
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(dtype=dtype)
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(dtype=dtype)
            if deepstack_indices:
                negative_deepstack_states = self._get_mllm_deepstack_hidden_states(
                    neg_p, device, dtype, deepstack_indices, images=neg_mllm_images
                )
        return (
            prompt_embeds,
            prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
            deepstack_states,
            negative_deepstack_states,
        )

    def _get_image_embeds(self, image: PIL.Image.Image, device: torch.device) -> torch.Tensor:
        image_encoder_dtype = next(self.image_encoder.parameters()).dtype
        pixel_values = self.feature_extractor(images=image, do_resize=True, return_tensors="pt").pixel_values.to(
            device=device, dtype=image_encoder_dtype
        )
        return torch.clamp(
            self.image_encoder(pixel_values).last_hidden_state,
            min=-65504.0,
            max=65504.0,
        )

    def _get_image_latents(self, image: PIL.Image.Image, height: int, width: int, device: torch.device) -> torch.Tensor:
        image_tensor = (
            VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            .preprocess(
                image,
                height=height,
                width=width,
                resize_mode=self._i2v_cond_resize_mode,
            )
            .unsqueeze(2)
            .to(device=device, dtype=self.vae.dtype)
        )
        return retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax") * self.vae.config.scaling_factor

    def prepare_cond_latents_and_mask(
        self,
        latents: torch.Tensor,
        image: PIL.Image.Image,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, frames, lat_height, lat_width = latents.shape
        latent_condition = self._get_image_latents(image, height, width, device).repeat(batch, 1, frames, 1, 1)
        latent_condition[:, :, 1:, :, :] = 0
        latent_mask = torch.zeros(batch, 1, frames, lat_height, lat_width, dtype=dtype, device=device)
        latent_mask[:, :, 0, :, :] = 1.0
        return latent_condition.to(device=device, dtype=dtype), latent_mask

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (
            batch_size,
            self.num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        return self.transformer(**kwargs)[0]

    def forward(
        self,
        req: OmniDiffusionRequest,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 121,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if not req.prompts:
            raise ValueError("Prompt is required.")
        prompt_data = req.prompts[0]
        prompt = prompt_data if isinstance(prompt_data, str) else prompt_data.get("prompt")
        negative_prompt = None if isinstance(prompt_data, str) else prompt_data.get("negative_prompt")
        multi_modal_data = prompt_data.get("multi_modal_data", {}) if not isinstance(prompt_data, str) else {}

        raw_image = multi_modal_data.get("image", None)
        if isinstance(raw_image, list):
            raw_image = raw_image[0]

        image = None
        if raw_image is not None:
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image.convert("RGB")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames_val = req.sampling_params.num_frames or num_frames
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale_provided else guidance_scale
        )
        do_cfg = guidance_scale > 1.0

        device = self.device
        dtype = self.transformer.transformer_blocks[0].norm1.linear.weight.dtype
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        self.scheduler._shift = self._default_scheduler_shift
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        if isinstance(extra, dict) and extra.get("flow_shift") is not None:
            self.scheduler._shift = float(extra["flow_shift"])

        enc_tuple = self.encode_prompt(prompt, image, device, dtype, negative_prompt, do_cfg)

        latents = self.prepare_latents(
            enc_tuple[0].shape[0],
            height,
            width,
            num_frames_val,
            dtype,
            device,
            generator,
            req.sampling_params.latents,
        )

        if image is not None:
            image_embeds = self._get_image_embeds(image, device).to(dtype=dtype)
            cond_latents, mask = self.prepare_cond_latents_and_mask(latents, image, height, width, dtype, device)
        else:
            batch, channels, frames, lat_height, lat_width = latents.shape
            cond_latents = torch.zeros_like(latents)
            mask = torch.zeros(batch, 1, frames, lat_height, lat_width, dtype=dtype, device=device)
            image_embeds = torch.zeros((batch, 729, 1152), dtype=dtype, device=device)

        self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, num_steps + 1)[:-1], device=device)
        timesteps = self.scheduler.timesteps

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents, cond_latents, mask], dim=1)
                timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
                timestep_r = torch.tensor([0.0], device=device) if i == len(timesteps) - 1 else timesteps[i + 1]
                timestep_r = timestep_r.expand(latents.shape[0]).to(latents.dtype) if self.use_meanflow else None

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "timestep_r": timestep_r,
                    "encoder_hidden_states": enc_tuple[0],
                    "encoder_attention_mask": enc_tuple[1],
                    "encoder_hidden_states_2": enc_tuple[2],
                    "encoder_attention_mask_2": enc_tuple[3],
                    "image_embeds": image_embeds,
                    "all_stack_text_states": enc_tuple[8],
                    "return_dict": False,
                }
                negative_kwargs = (
                    {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "timestep_r": timestep_r,
                        "encoder_hidden_states": enc_tuple[4],
                        "encoder_attention_mask": enc_tuple[5],
                        "encoder_hidden_states_2": enc_tuple[6],
                        "encoder_attention_mask_2": enc_tuple[7],
                        "image_embeds": image_embeds,
                        "all_stack_text_states": enc_tuple[9],
                        "return_dict": False,
                    }
                    if do_cfg and enc_tuple[4] is not None
                    else None
                )

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=bool(negative_kwargs),
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=req.sampling_params.cfg_normalize,
                )
                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg=bool(negative_kwargs))
                pbar.update()

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        if output_type == "latent":
            return DiffusionOutput(output=latents)
        return DiffusionOutput(
            output=self.vae.decode(
                latents.to(self.vae.dtype) / self.vae.config.scaling_factor,
                return_dict=False,
            )[0]
        )

    # NATIVE ARCHITECTURE ROUTER: Tencent Original -> HF Diffusers
    def _translate_tencent_to_hf(self, raw_tf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped_tf = {}

        def map_k(src: str, dst: str):
            if src in raw_tf:
                mapped_tf[dst] = raw_tf[src]

        map_k("img_in.proj.weight", "x_embedder.proj.weight")
        map_k("img_in.proj.bias", "x_embedder.proj.bias")
        map_k("final_layer.linear.weight", "proj_out.weight")
        map_k("final_layer.linear.bias", "proj_out.bias")
        map_k("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight")
        map_k("final_layer.adaLN_modulation.1.bias", "norm_out.linear.bias")
        map_k("txt_in.input_embedder.weight", "context_embedder.proj_in.weight")
        map_k("txt_in.input_embedder.bias", "context_embedder.proj_in.bias")

        # Time Text Embedder
        map_k(
            "txt_in.c_embedder.linear_1.weight",
            "context_embedder.time_text_embed.text_embedder.linear_1.weight",
        )
        map_k(
            "txt_in.c_embedder.linear_1.bias",
            "context_embedder.time_text_embed.text_embedder.linear_1.bias",
        )
        map_k(
            "txt_in.c_embedder.linear_2.weight",
            "context_embedder.time_text_embed.text_embedder.linear_2.weight",
        )
        map_k(
            "txt_in.c_embedder.linear_2.bias",
            "context_embedder.time_text_embed.text_embedder.linear_2.bias",
        )
        map_k(
            "txt_in.t_embedder.mlp.0.weight",
            "context_embedder.time_text_embed.timestep_embedder.linear_1.weight",
        )
        map_k(
            "txt_in.t_embedder.mlp.0.bias",
            "context_embedder.time_text_embed.timestep_embedder.linear_1.bias",
        )
        map_k(
            "txt_in.t_embedder.mlp.2.weight",
            "context_embedder.time_text_embed.timestep_embedder.linear_2.weight",
        )
        map_k(
            "txt_in.t_embedder.mlp.2.bias",
            "context_embedder.time_text_embed.timestep_embedder.linear_2.bias",
        )

        # Time Embedder
        map_k("time_in.mlp.0.weight", "time_embed.timestep_embedder.linear_1.weight")
        map_k("time_in.mlp.0.bias", "time_embed.timestep_embedder.linear_1.bias")
        map_k("time_in.mlp.2.weight", "time_embed.timestep_embedder.linear_2.weight")
        map_k("time_in.mlp.2.bias", "time_embed.timestep_embedder.linear_2.bias")
        map_k("cond_type_embedding.weight", "cond_type_embed.weight")

        # Vision Embedder
        vision_proj_map = {
            "vision_in.proj.0.weight": "image_embedder.norm_in.weight",
            "vision_in.proj.0.bias": "image_embedder.norm_in.bias",
            "vision_in.proj.1.weight": "image_embedder.linear_1.weight",
            "vision_in.proj.1.bias": "image_embedder.linear_1.bias",
            "vision_in.proj.3.weight": "image_embedder.linear_2.weight",
            "vision_in.proj.3.bias": "image_embedder.linear_2.bias",
            "vision_in.proj.4.weight": "image_embedder.norm_out.weight",
            "vision_in.proj.4.bias": "image_embedder.norm_out.bias",
        }
        for s_k, d_k in vision_proj_map.items():
            map_k(s_k, d_k)

        # T5-2 Projections
        t5_2_prefix = "by" + "t5_in"
        if f"{t5_2_prefix}.fc1.weight" in raw_tf:
            w1 = raw_tf[f"{t5_2_prefix}.fc1.weight"]
            mapped_tf["context_embedder_2.linear_1.weight"] = w1

            map_k(f"{t5_2_prefix}.fc1.bias", "context_embedder_2.linear_1.bias")
            map_k(f"{t5_2_prefix}.fc2.weight", "context_embedder_2.linear_2.weight")
            map_k(f"{t5_2_prefix}.fc2.bias", "context_embedder_2.linear_2.bias")
            map_k(f"{t5_2_prefix}.fc3.weight", "context_embedder_2.linear_3.weight")
            map_k(f"{t5_2_prefix}.fc3.bias", "context_embedder_2.linear_3.bias")
            map_k(f"{t5_2_prefix}.layernorm.weight", "context_embedder_2.norm.weight")
            map_k(f"{t5_2_prefix}.layernorm.bias", "context_embedder_2.norm.bias")

        # Refiner Blocks
        for layer in range(2):
            p = f"txt_in.individual_token_refiner.blocks.{layer}"
            if f"{p}.self_attn_qkv.weight" in raw_tf:
                q_w, k_w, v_w = raw_tf[f"{p}.self_attn_qkv.weight"].chunk(3, dim=0)
                q_b, k_b, v_b = raw_tf[f"{p}.self_attn_qkv.bias"].chunk(3, dim=0)
                pref = f"context_embedder.token_refiner.refiner_blocks.{layer}"

                mapped_tf[f"{pref}.attn.to_q.weight"] = q_w
                mapped_tf[f"{pref}.attn.to_k.weight"] = k_w
                mapped_tf[f"{pref}.attn.to_v.weight"] = v_w
                mapped_tf[f"{pref}.attn.to_q.bias"] = q_b
                mapped_tf[f"{pref}.attn.to_k.bias"] = k_b
                mapped_tf[f"{pref}.attn.to_v.bias"] = v_b

                map_k(f"{p}.self_attn_proj.weight", f"{pref}.attn.to_out.0.weight")
                map_k(f"{p}.self_attn_proj.bias", f"{pref}.attn.to_out.0.bias")
                map_k(f"{p}.norm1.weight", f"{pref}.norm1.weight")
                map_k(f"{p}.norm1.bias", f"{pref}.norm1.bias")
                map_k(f"{p}.norm2.weight", f"{pref}.norm2.weight")
                map_k(f"{p}.norm2.bias", f"{pref}.norm2.bias")
                map_k(f"{p}.mlp.fc1.weight", f"{pref}.ff.net.0.proj.weight")
                map_k(f"{p}.mlp.fc1.bias", f"{pref}.ff.net.0.proj.bias")
                map_k(f"{p}.mlp.fc2.weight", f"{pref}.ff.net.2.weight")
                map_k(f"{p}.mlp.fc2.bias", f"{pref}.ff.net.2.bias")
                map_k(f"{p}.adaLN_modulation.1.weight", f"{pref}.norm_out.linear.weight")
                map_k(f"{p}.adaLN_modulation.1.bias", f"{pref}.norm_out.linear.bias")

        # Double Blocks
        double_block_keys = set(k.split(".")[1] for k in raw_tf.keys() if k.startswith("double_blocks."))
        for i_str in double_block_keys:
            p = f"double_blocks.{i_str}"
            t_p = f"transformer_blocks.{i_str}"

            map_k(f"{p}.img_attn_q.weight", f"{t_p}.attn.to_q.weight")
            map_k(f"{p}.img_attn_k.weight", f"{t_p}.attn.to_k.weight")
            map_k(f"{p}.img_attn_v.weight", f"{t_p}.attn.to_v.weight")
            map_k(f"{p}.img_attn_q.bias", f"{t_p}.attn.to_q.bias")
            map_k(f"{p}.img_attn_k.bias", f"{t_p}.attn.to_k.bias")
            map_k(f"{p}.img_attn_v.bias", f"{t_p}.attn.to_v.bias")

            map_k(f"{p}.txt_attn_q.weight", f"{t_p}.attn.add_q_proj.weight")
            map_k(f"{p}.txt_attn_k.weight", f"{t_p}.attn.add_k_proj.weight")
            map_k(f"{p}.txt_attn_v.weight", f"{t_p}.attn.add_v_proj.weight")
            map_k(f"{p}.txt_attn_q.bias", f"{t_p}.attn.add_q_proj.bias")
            map_k(f"{p}.txt_attn_k.bias", f"{t_p}.attn.add_k_proj.bias")
            map_k(f"{p}.txt_attn_v.bias", f"{t_p}.attn.add_v_proj.bias")

            map_k(f"{p}.img_attn_q_norm.weight", f"{t_p}.attn.norm_q.weight")
            map_k(f"{p}.img_attn_k_norm.weight", f"{t_p}.attn.norm_k.weight")
            map_k(f"{p}.txt_attn_q_norm.weight", f"{t_p}.attn.norm_added_q.weight")
            map_k(f"{p}.txt_attn_k_norm.weight", f"{t_p}.attn.norm_added_k.weight")

            map_k(f"{p}.img_attn_proj.weight", f"{t_p}.attn.to_out.0.weight")
            map_k(f"{p}.img_attn_proj.bias", f"{t_p}.attn.to_out.0.bias")
            map_k(f"{p}.txt_attn_proj.weight", f"{t_p}.attn.to_add_out.weight")
            map_k(f"{p}.txt_attn_proj.bias", f"{t_p}.attn.to_add_out.bias")

            map_k(f"{p}.img_mlp.fc1.weight", f"{t_p}.ff.net.0.proj.weight")
            map_k(f"{p}.img_mlp.fc1.bias", f"{t_p}.ff.net.0.proj.bias")
            map_k(f"{p}.img_mlp.fc2.weight", f"{t_p}.ff.net.2.weight")
            map_k(f"{p}.img_mlp.fc2.bias", f"{t_p}.ff.net.2.bias")

            map_k(f"{p}.txt_mlp.fc1.weight", f"{t_p}.ff_context.net.0.proj.weight")
            map_k(f"{p}.txt_mlp.fc1.bias", f"{t_p}.ff_context.net.0.proj.bias")
            map_k(f"{p}.txt_mlp.fc2.weight", f"{t_p}.ff_context.net.2.weight")
            map_k(f"{p}.txt_mlp.fc2.bias", f"{t_p}.ff_context.net.2.bias")

            map_k(f"{p}.img_mod.linear.weight", f"{t_p}.norm1.linear.weight")
            map_k(f"{p}.img_mod.linear.bias", f"{t_p}.norm1.linear.bias")
            map_k(f"{p}.txt_mod.linear.weight", f"{t_p}.norm1_context.linear.weight")
            map_k(f"{p}.txt_mod.linear.bias", f"{t_p}.norm1_context.linear.bias")

        # OmniWeaving deepstack projection (mm_in) — same key names on both sides
        map_k("mm_in.linear_1.weight", "mm_in.linear_1.weight")
        map_k("mm_in.linear_1.bias", "mm_in.linear_1.bias")
        map_k("mm_in.linear_2.weight", "mm_in.linear_2.weight")
        map_k("mm_in.linear_2.bias", "mm_in.linear_2.bias")

        # Single Blocks
        single_block_keys = set(k.split(".")[1] for k in raw_tf.keys() if k.startswith("single_blocks."))
        for i_str in single_block_keys:
            p = f"single_blocks.{i_str}"
            t_p = f"single_transformer_blocks.{i_str}"

            map_k(f"{p}.attn_q.weight", f"{t_p}.attn.to_q.weight")
            map_k(f"{p}.attn_k.weight", f"{t_p}.attn.to_k.weight")
            map_k(f"{p}.attn_v.weight", f"{t_p}.attn.to_v.weight")
            map_k(f"{p}.attn_q.bias", f"{t_p}.attn.to_q.bias")
            map_k(f"{p}.attn_k.bias", f"{t_p}.attn.to_k.bias")
            map_k(f"{p}.attn_v.bias", f"{t_p}.attn.to_v.bias")

            map_k(f"{p}.attn_proj.weight", f"{t_p}.attn.to_out.0.weight")
            map_k(f"{p}.attn_proj.bias", f"{t_p}.attn.to_out.0.bias")
            map_k(f"{p}.mlp.fc1.weight", f"{t_p}.proj_mlp.weight")
            map_k(f"{p}.mlp.fc1.bias", f"{t_p}.proj_mlp.bias")
            map_k(f"{p}.mlp.fc2.weight", f"{t_p}.proj_out.weight")
            map_k(f"{p}.mlp.fc2.bias", f"{t_p}.proj_out.bias")
            map_k(f"{p}.mod.linear.weight", f"{t_p}.norm.linear.weight")
            map_k(f"{p}.mod.linear.bias", f"{t_p}.norm.linear.bias")

        return mapped_tf

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raw_tf, raw_t5, other_weights = {}, {}, []
        for k, v in weights:
            if k.startswith("transformer."):
                raw_tf[k[len("transformer.") :]] = v
            elif k.startswith("text_encoder_2."):
                raw_t5[k[len("text_encoder_2.") :]] = v
            else:
                other_weights.append((k, v))

        if any(k.startswith("double_blocks.") for k in raw_tf.keys()):
            logger.info("Detecting Tencent original weight format. Routing through native Auto-Translator...")
            tf_weights_to_load = self._translate_tencent_to_hf(raw_tf).items()
        else:
            tf_weights_to_load = raw_tf.items()

        loaded_keys = set()

        # Load Main Backbone
        if hasattr(self.transformer, "load_weights"):
            tf_loaded = self.transformer.load_weights(tf_weights_to_load)
            loaded_keys.update([f"transformer.{k}" for k in tf_loaded])
        else:
            loader = AutoWeightsLoader(self.transformer)
            tf_loaded = loader.load_weights(tf_weights_to_load)
            loaded_keys.update([f"transformer.{k}" for k in tf_loaded])

        # Load T5-2
        if self.has_t5_2 and raw_t5:
            t5_items = raw_t5.items()
            if hasattr(self.text_encoder_2, "load_weights"):
                t5_loaded = self.text_encoder_2.load_weights(t5_items)
                loaded_keys.update([f"text_encoder_2.{k}" for k in t5_loaded])
            else:
                loader = AutoWeightsLoader(self.text_encoder_2)
                t5_loaded = loader.load_weights(t5_items)
                loaded_keys.update([f"text_encoder_2.{k}" for k in t5_loaded])

        # Load Everything Else
        if other_weights:
            loader = AutoWeightsLoader(self)
            loaded_keys.update(loader.load_weights(other_weights))

        for name, p in self.transformer.named_parameters():
            if f"transformer.{name}" not in loaded_keys and (torch.isnan(p).any() or p.std().item() == 0.0):
                logger.warning(
                    f"Parameter transformer.{name} was not loaded from the checkpoint. "
                    "This might cause incorrect outputs."
                )

        return loaded_keys
