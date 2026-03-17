# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import logging
import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import (
    PipelineImageInput,
    VaeImageProcessor,
    is_valid_image_imagelist,
)
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.omnigen2.omnigen2_transformer import (
    OmniGen2RotaryPosEmbed,
    OmniGen2Transformer2DModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        dynamic_time_shift (`bool`, defaults to `True`):
            Whether to use dynamic time shifting for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, dynamic_time_shift: bool = True):
        timesteps = torch.linspace(0, 1, num_train_timesteps + 1, dtype=torch.float32)[:-1]

        self.timesteps = timesteps

        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self._timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: str | torch.device = None,
        timesteps: list[float] | None = None,
        num_tokens: int | None = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`list[float]`, *optional*):
                Custom timesteps to use. If provided, `num_inference_steps` is ignored.
            num_tokens (`int`, *optional*):
                Number of tokens, used for dynamic time shifting.
        """

        if timesteps is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]
            if self.config.dynamic_time_shift and num_tokens is not None:
                # when input resolution is 320*320, m = 1; when 1024*1024, m = 3.2
                m = np.sqrt(num_tokens) / 40
                timesteps = timesteps / (m - m * timesteps + timesteps)

        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)
        _timesteps = torch.cat([timesteps, torch.ones(1, device=timesteps.device)])

        self.timesteps = timesteps
        self._timesteps = _timesteps
        self._step_index = None
        self._begin_index = None

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is
                the sample tensor.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        t = self._timesteps[self.step_index]
        t_next = self._timesteps[self.step_index + 1]

        prev_sample = sample + (t_next - t) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


def get_omnigen2_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-processing function for OmniGen2Pipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8

    image_processor = OmniGen2ImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_resize=True)
    latent_channels = vae_config.get("z_dim", 16)

    def pre_process_func(
        request: OmniDiffusionRequest,
    ) -> OmniDiffusionRequest:
        """Pre-process requests for OmniGen2Pipeline."""
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None

            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is not None:
                if isinstance(raw_image, list):
                    images = [PIL.Image.open(img) if isinstance(img, str) else img for img in raw_image]
                elif isinstance(raw_image, str):
                    images = [PIL.Image.open(raw_image)]
                else:
                    images = [raw_image]

                first_raw = images[0]
                if isinstance(first_raw, PIL.Image.Image):
                    new_h, new_w = image_processor.get_new_height_width(
                        first_raw, max_pixels=1024 * 1024, max_side_length=1024
                    )
                    if request.sampling_params.height is None:
                        request.sampling_params.height = new_h
                    if request.sampling_params.width is None:
                        request.sampling_params.width = new_w

                preprocessed_images = []
                for image in images:
                    if not (
                        isinstance(image, torch.Tensor) and len(image.shape) > 1 and image.shape[1] == latent_channels
                    ):
                        image = image_processor.preprocess(image, max_pixels=1024 * 1024, max_side_length=1024)
                    preprocessed_images.append(image)

                prompt["additional_information"]["preprocessed_images"] = preprocessed_images

            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_omnigen2_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = OmniGen2ImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_resize=True)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


class OmniGen2ImageProcessor(VaeImageProcessor):
    """
    Image processor for OmniGen2 image resize and crop.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`. Can accept
            `height` and `width` arguments from [`image_processor.VaeImageProcessor.preprocess`] method.
        vae_scale_factor (`int`, *optional*, defaults to `16`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        max_pixels (`int`, *optional*, defaults to `1048576`):
            Maximum number of pixels allowed in the image. Images exceeding this limit are downscaled proportionally.
        max_side_length (`int`, *optional*, defaults to `1024`):
            Maximum length of the longer side of the image. Images exceeding this limit are downscaled proportionally.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `False`):
            Whether to binarize the image to 0/1.
        do_convert_grayscale (`bool`, *optional*, defaults to `False`):
            Whether to convert the images to grayscale format.
    """

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        resample: str = "lanczos",
        max_pixels: int = 1024 * 1024,
        max_side_length: int = 1024,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

        self.max_pixels = max_pixels
        self.max_side_length = max_side_length

    def get_new_height_width(
        self,
        image: PIL.Image.Image | np.ndarray | torch.Tensor,
        height: int | None = None,
        width: int | None = None,
        max_pixels: int | None = None,
        max_side_length: int | None = None,
    ) -> tuple[int, int]:
        r"""
        Returns the height and width of the image, downscaled to the next integer multiple of `vae_scale_factor`.

        Args:
            image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
                The image input, which can be a PIL image, NumPy array, or PyTorch tensor. If it is a NumPy array, it
                should have shape `[batch, height, width]` or `[batch, height, width, channels]`. If it is a PyTorch
                tensor, it should have shape `[batch, channels, height, width]`.
            height (`Optional[int]`, *optional*, defaults to `None`):
                The height of the preprocessed image. If `None`, the height of the `image` input will be used.
            width (`Optional[int]`, *optional*, defaults to `None`):
                The width of the preprocessed image. If `None`, the width of the `image` input will be used.

        Returns:
            `Tuple[int, int]`:
                A tuple containing the height and width, both resized to the nearest integer multiple of
                `vae_scale_factor`.
        """

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            else:
                height = image.shape[1]

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            else:
                width = image.shape[2]

        if max_side_length is None:
            max_side_length = self.max_side_length

        if max_pixels is None:
            max_pixels = self.max_pixels

        max_side_length_ratio = 1.0
        if max_side_length is not None:
            if height > width:
                max_side_length_ratio = max_side_length / height
            else:
                max_side_length_ratio = max_side_length / width

        cur_pixels = height * width
        max_pixels_ratio = (max_pixels / cur_pixels) ** 0.5
        ratio = min(max_pixels_ratio, max_side_length_ratio, 1.0)  # do not upscale input image

        new_height, new_width = (
            int(height * ratio) // self.config.vae_scale_factor * self.config.vae_scale_factor,
            int(width * ratio) // self.config.vae_scale_factor * self.config.vae_scale_factor,
        )
        return new_height, new_width

    def preprocess(
        self,
        image: PipelineImageInput,
        height: int | None = None,
        width: int | None = None,
        max_pixels: int | None = None,
        max_side_length: int | None = None,
        resize_mode: str = "default",  # "default", "fill", "crop"
        crops_coords: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Preprocess the image input.

        Args:
            image (`PipelineImageInput`):
                The image input, accepted formats are PIL images, NumPy arrays, PyTorch tensors; Also accept list of
                supported formats.
            height (`int`, *optional*):
                The height in preprocessed image. If `None`, will use the `get_default_height_width()` to get default
                height.
            width (`int`, *optional*):
                The width in preprocessed. If `None`, will use get_default_height_width()` to get the default width.
            resize_mode (`str`, *optional*, defaults to `default`):
                The resize mode, can be one of `default` or `fill`. If `default`, will resize the image to fit within
                the specified width and height, and it may not maintaining the original aspect ratio. If `fill`, will
                resize the image to fit within the specified width and height, maintaining the aspect ratio, and then
                center the image within the dimensions, filling empty with data from image. If `crop`, will resize the
                image to fit within the specified width and height, maintaining the aspect ratio, and then center the
                image within the dimensions, cropping the excess. Note that resize_mode `fill` and `crop` are only
                supported for PIL image input.
            crops_coords (`List[Tuple[int, int, int, int]]`, *optional*, defaults to `None`):
                The crop coordinates for each image in the batch. If `None`, will not crop the image.

        Returns:
            `torch.Tensor`:
                The preprocessed image.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        # Expand the missing dimension for 3-dimensional pytorch tensor or numpy array that represents grayscale image
        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                # if image is a pytorch tensor could have 2 possible shapes:
                #    1. batch x height x width: we should insert the channel dimension at position 1
                #    2. channel x height x width: we should insert batch dimension at position 0,
                #       however, since both channel and batch dimension has same size 1, it is same to insert at
                #       position 1
                #    for simplicity, we insert a dimension of size 1 at position 1 for both cases
                image = image.unsqueeze(1)
            else:
                # if it is a numpy array, it could have 2 possible shapes:
                #   1. batch x height x width: insert channel dimension on last position
                #   2. height x width x channel: insert batch dimension on first position
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support "
                f"{', '.join(str(x) for x in supported_formats)}"
            )
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            if self.config.do_resize:
                height, width = self.get_new_height_width(image[0], height, width, max_pixels, max_side_length)
                image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
            if self.config.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            elif self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_pt(image)  # to pt

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)

            image = self.numpy_to_pt(image)

            height, width = self.get_new_height_width(image, height, width, max_pixels, max_side_length)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)

            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)

            channel = image.shape[1]
            # don't need any preprocess if the image is latents
            if channel == self.config.vae_latent_channels:
                return image

            height, width = self.get_new_height_width(image, height, width, max_pixels, max_side_length)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. "
                "The expected value range for image tensor is [0,1] "
                f"when passing as pytorch tensor or numpy Array. "
                f"You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    **kwargs: Any,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` must be `None`.
        **kwargs (`Any`):
            Additional keyword arguments passed to `scheduler.set_timesteps`.

    Returns:
        timesteps (`torch.Tensor`): The timestep schedule from the scheduler.
        num_inference_steps (`int`): The number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OmniGen2Pipeline(nn.Module):
    """
    Pipeline for text-to-image generation using OmniGen2.

    This pipeline implements a text-to-image generation model that uses:
    - Qwen2.5-VL for text encoding
    - A custom transformer architecture for image generation
    - VAE for image encoding/decoding
    - FlowMatchEulerDiscreteScheduler for noise scheduling

    Args:
        od_config (OmniDiffusionConfig): The OmniDiffusion configuration.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        """
        Initialize the OmniGen2 pipeline.

        Args:
            transformer: The transformer model for image generation.
            vae: The VAE model for image encoding/decoding.
            scheduler: The scheduler for noise scheduling.
            text_encoder: The text encoder model.
            tokenizer: The tokenizer for text processing.
        """
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        model = od_config.model

        # Check if model is a local path
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )

        transformer_config_path = os.path.join(model, "transformer", "config.json")
        transformer_kwargs = {}

        if os.path.exists(transformer_config_path):
            with open(transformer_config_path) as f:
                transformer_config = json.load(f)

            param_mapping = {
                "patch_size": "patch_size",
                "in_channels": "in_channels",
                "out_channels": "out_channels",
                "hidden_size": "hidden_size",
                "num_layers": "num_layers",
                "num_refiner_layers": "num_refiner_layers",
                "num_attention_heads": "num_attention_heads",
                "num_kv_heads": "num_kv_heads",
                "multiple_of": "multiple_of",
                "ffn_dim_multiplier": "ffn_dim_multiplier",
                "norm_eps": "norm_eps",
                "axes_dim_rope": "axes_dim_rope",
                "axes_lens": "axes_lens",
                "text_feat_dim": "text_feat_dim",
                "timestep_scale": "timestep_scale",
            }

            for config_key, param_name in param_mapping.items():
                if config_key in transformer_config:
                    value = transformer_config[config_key]
                    # Handle tuple parameters (axes_dim_rope, axes_lens)
                    if isinstance(value, list) and param_name in (
                        "axes_dim_rope",
                        "axes_lens",
                    ):
                        value = tuple(value)
                    transformer_kwargs[param_name] = value
        self.transformer = OmniGen2Transformer2DModel(**transformer_kwargs)
        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, subfolder="mllm", local_files_only=local_files_only
        ).to(self.device)
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            model, subfolder="processor", local_files_only=local_files_only
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
        self.default_sample_size = 128

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        latents: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """
        Prepare the initial latents for the diffusion process.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of channels in the latent space.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type of the latents.
            device: The device to place the latents on.
            generator: The random number generator to use.
            latents: Optional pre-computed latents to use instead of random initialization.

        Returns:
            torch.FloatTensor: The prepared latents tensor.
        """
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def encode_vae(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode an image into the VAE latent space.

        Args:
            img: The input image tensor to encode.

        Returns:
            torch.FloatTensor: The encoded latent representation.
        """
        z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample()
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.vae.dtype)
        return z0

    def prepare_image(
        self,
        images: list[PIL.Image.Image] | PIL.Image.Image,
        batch_size: int,
        num_images_per_prompt: int,
        max_pixels: int,
        max_side_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.FloatTensor | None]:
        """
        Prepare input images for processing by encoding them into the VAE latent space.

        Args:
            images: Single image or list of images to process.
            batch_size: The number of images to generate per prompt.
            num_images_per_prompt: The number of images to generate for each prompt.
            device: The device to place the encoded latents on.
            dtype: The data type of the encoded latents.

        Returns:
            List[Optional[torch.FloatTensor]]: List of encoded latent representations for each image.
        """
        if batch_size == 1:
            images = [images]
        latents = []
        for i, img in enumerate(images):
            if img is not None and len(img) > 0:
                ref_latents = []
                for j, img_j in enumerate(img):
                    ref_latents.append(self.encode_vae(img_j.to(device=device)).squeeze(0))
            else:
                ref_latents = None
            for _ in range(num_images_per_prompt):
                latents.append(ref_latents)

        return latents

    def _get_qwen2_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings from the Qwen2 text encoder.

        Args:
            prompt: The prompt or list of prompts to encode.
            device: The device to place the embeddings on. If None, uses the pipeline's device.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The prompt embeddings tensor
                - The attention mask tensor

        Raises:
            Warning: If the input text is truncated due to sequence length limitations.
        """
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # text_inputs = self.processor.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=max_sequence_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.processor.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.processor.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because Qwen2.5-VL can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.mllm(
            text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]

        if self.mllm is not None:
            dtype = self.mllm.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    def _apply_chat_template(self, prompt: str):
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates high-quality images based on user instructions.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return prompt

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int = 1,
        device: torch.device | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                Lumina-T2I, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Lumina-T2I, it's should be the embeddings of the "" string.
            max_sequence_length (`int`, defaults to `256`):
                Maximum sequence length to use for the prompt.
        """
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [self._apply_chat_template(_prompt) for _prompt in prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=prompt, device=device, max_sequence_length=max_sequence_length
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get negative embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""

            # Normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = [self._apply_chat_template(_negative_prompt) for _negative_prompt in negative_prompt]

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )

            batch_size, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def cfg_range(self):
        return self._cfg_range

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        prompt_attention_mask: torch.LongTensor | None = None,
        negative_prompt_attention_mask: torch.LongTensor | None = None,
        max_sequence_length: int | None = 1024,
        input_images: list[PIL.Image.Image] | None = None,
        num_images_per_prompt: int = 1,
        height: int | None = None,
        width: int | None = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: tuple[float, float] = (0.0, 1.0),
        attention_kwargs: dict[str, Any] | None = None,
        timesteps: list[int] = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        verbose: bool = False,
        step_func=None,
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning(
                "OmniGen2 only supports a single prompt per request. "
                "Only the first prompt will be used; %d extra prompt(s) will be ignored.",
                len(req.prompts) - 1,
            )
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or prompt)
        negative_prompt = (
            None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt", negative_prompt)
        )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for OmniGen2 generation.")

        if not isinstance(first_prompt, str) and "preprocessed_images" in (
            additional_information := first_prompt.get("additional_information", {})
        ):
            input_images = additional_information.get("preprocessed_images")

        height = req.sampling_params.height or height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or width or self.default_sample_size * self.vae_scale_factor
        generator = req.sampling_params.generator or generator
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        if req.sampling_params.guidance_scale_provided:
            text_guidance_scale = req.sampling_params.guidance_scale
        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = (
            req.sampling_params.guidance_scale_2
            if req.sampling_params.guidance_scale_2 is not None
            else image_guidance_scale
        )
        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.text_guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )

        dtype = self.vae.dtype
        # 3. Prepare control image
        ref_latents = self.prepare_image(
            images=input_images,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_pixels=max_pixels,
            max_side_length=max_input_image_side_length,
            device=device,
            dtype=dtype,
        )

        if input_images is None:
            input_images = []

        if len(input_images) == 1 and align_res:
            width, height = (
                ref_latents[0][0].shape[-1] * self.vae_scale_factor,
                ref_latents[0][0].shape[-2] * self.vae_scale_factor,
            )
            ori_width, ori_height = width, height
        else:
            ori_width, ori_height = width, height

            cur_pixels = height * width
            ratio = (max_pixels / cur_pixels) ** 0.5
            ratio = min(ratio, 1.0)

            height, width = (
                int(height * ratio) // 16 * 16,
                int(width * ratio) // 16 * 16,
            )

        if len(input_images) == 0:
            self._image_guidance_scale = 1

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )

        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            device=device,
            dtype=dtype,
            verbose=verbose,
            step_func=step_func,
        )

        image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        return DiffusionOutput(output=image)

    def processing(
        self,
        latents,
        ref_latents,
        prompt_embeds,
        freqs_cis,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        num_inference_steps,
        timesteps,
        device,
        dtype,
        verbose,
        step_func=None,
    ):
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )
        self._num_timesteps = len(timesteps)

        for i, t in enumerate(timesteps):
            model_pred = self.predict(
                t=t,
                latents=latents,
                prompt_embeds=prompt_embeds,
                freqs_cis=freqs_cis,
                prompt_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=ref_latents,
            )
            text_guidance_scale = (
                self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
            )
            image_guidance_scale = (
                self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
            )

            if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                model_pred_ref = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )

                model_pred_uncond = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                )

                model_pred = (
                    model_pred_uncond
                    + image_guidance_scale * (model_pred_ref - model_pred_uncond)
                    + text_guidance_scale * (model_pred - model_pred_ref)
                )
            elif text_guidance_scale > 1.0:
                model_pred_uncond = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=negative_prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                )
                model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

            latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

            latents = latents.to(dtype=dtype)

            if step_func is not None:
                step_func(i, self._num_timesteps)

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        return image

    def predict(
        self,
        t,
        latents,
        prompt_embeds,
        freqs_cis,
        prompt_attention_mask,
        ref_image_hidden_states,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        batch_size, num_channels_latents, height, width = latents.shape

        optional_kwargs = {}
        if "ref_image_hidden_states" in set(inspect.signature(self.transformer.forward).parameters.keys()):
            optional_kwargs["ref_image_hidden_states"] = ref_image_hidden_states

        model_pred = self.transformer(
            latents,
            timestep,
            prompt_embeds,
            freqs_cis,
            prompt_attention_mask,
            **optional_kwargs,
        )
        return model_pred

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
