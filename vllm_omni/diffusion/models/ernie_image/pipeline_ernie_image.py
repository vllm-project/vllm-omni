# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ernie_image/pipeline_ernie_image.py

import json
import os
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import ErnieImageTransformer2DModel
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


def _resolve_model_path_for_optional_pe(model: str, revision: str | None) -> str:
    if os.path.exists(model):
        return model
    try:
        return download_weights_from_hf_specific(model, revision, ["pe/*", "pe_tokenizer/*"])
    except Exception as e:
        logger.debug("Failed to resolve ERNIE-Image PE files for %s: %s", model, e)
        return model


def get_ernie_image_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.output_type == "latent":
        return lambda x: x
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_normalize=False)

    def post_process_func(images: torch.Tensor):
        images = (images.clamp(-1, 1) + 1) / 2
        return image_processor.postprocess(images)

    return post_process_func


class ErnieImagePipeline(
    nn.Module, CFGParallelMixin, SupportImageInput, ProgressBarMixin, DiffusionPipelineProfilerMixin
):
    support_image_input = False

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
        is_distilled: bool = False,
    ):
        super().__init__()
        self.od_config = od_config
        self.is_distilled = is_distilled
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        logger.info("Model path for initialization: %s", model)
        local_files_only = os.path.exists(model)
        logger.info("Local files only: %s", local_files_only)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        self.text_encoder = AutoModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=od_config.dtype,
            local_files_only=local_files_only,
        ).to(self._execution_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        self.vae = AutoencoderKLFlux2.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=od_config.dtype,
            local_files_only=local_files_only,
        ).to(self._execution_device)

        # Load PE (Prompt Enhancement) model if available. For repo IDs,
        # resolve/download only PE files first so the existence check works.
        pe_base_path = _resolve_model_path_for_optional_pe(model, getattr(od_config, "revision", None))
        pe_model_path = os.path.join(pe_base_path, "pe")
        if os.path.exists(pe_model_path):
            try:
                self.pe_model = AutoModelForCausalLM.from_pretrained(
                    pe_model_path,
                    torch_dtype=od_config.dtype,
                    local_files_only=True,
                    trust_remote_code=True,
                ).to(self._execution_device)
                self.pe_tokenizer = AutoTokenizer.from_pretrained(
                    pe_base_path,
                    subfolder="pe_tokenizer",
                    local_files_only=True,
                    trust_remote_code=True,
                    use_fast=False,
                )
                self.use_pe = True
                logger.info("Loaded PE model from %s", pe_model_path)
            except Exception as e:
                logger.warning("Failed to load PE model: %s", e)
                self.pe_model = None
                self.pe_tokenizer = None
                self.use_pe = False
        else:
            self.pe_model = None
            self.pe_tokenizer = None
            self.use_pe = False

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, ErnieImageTransformer2DModel)
        self.transformer = ErnieImageTransformer2DModel(
            quant_config=od_config.quantization_config, **transformer_kwargs
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels)) if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        self._guidance_scale = None
        self._attention_kwargs = None
        self._num_timesteps = None
        self._current_timestep = None
        self._interrupt = False
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @staticmethod
    def _distributed_prompt_sync_state() -> tuple[bool, int]:
        try:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return False, 0
            if torch.distributed.get_world_size() <= 1:
                return False, 0
            return True, torch.distributed.get_rank()
        except Exception:
            return False, 0

    @staticmethod
    def _broadcast_enhanced_prompt(prompt: str | None) -> str | None:
        values = [prompt]
        torch.distributed.broadcast_object_list(values, src=0)
        return values[0]

    def _enhance_prompt(
        self,
        prompt: str,
        device: torch.device,
        width: int = 1024,
        height: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        if not self.use_pe or self.pe_model is None:
            return prompt

        sync_prompt, rank = self._distributed_prompt_sync_state()
        if sync_prompt and rank != 0:
            try:
                enhanced = self._broadcast_enhanced_prompt(None)
                return enhanced if enhanced else prompt
            except Exception as e:
                logger.warning("PE enhancement broadcast failed: %s", e)
                return prompt

        try:
            user_content = json.dumps(
                {"prompt": prompt, "width": width, "height": height},
                ensure_ascii=False,
            )
            messages = [{"role": "user", "content": user_content}]
            input_text = self.pe_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = self.pe_tokenizer(input_text, return_tensors="pt").to(device)
            output_ids = self.pe_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=temperature != 1.0 or top_p != 1.0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.pe_tokenizer.pad_token_id,
                eos_token_id=self.pe_tokenizer.eos_token_id,
            )
            output_ids = output_ids[0][inputs.input_ids.shape[1] :]
            result = self.pe_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            enhanced = result if result else prompt
        except Exception as e:
            logger.warning("PE enhancement failed: %s", e)
            enhanced = prompt

        if sync_prompt:
            try:
                synced = self._broadcast_enhanced_prompt(enhanced)
                return synced if synced else prompt
            except Exception as e:
                logger.warning("PE enhancement broadcast failed: %s", e)
        return enhanced

    @staticmethod
    def _is_warmup_request(req: OmniDiffusionRequest) -> bool:
        request_ids = getattr(req, "request_ids", None) or ()
        return len(request_ids) == 1 and request_ids[0] == "dummy_req_id"

    @staticmethod
    def _should_apply_pe(req: OmniDiffusionRequest) -> bool:
        if ErnieImagePipeline._is_warmup_request(req):
            return False
        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}
        return bool(extra_args.get("apply_pe", True))

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        num_images_per_prompt: int = 1,
        width: int = 1024,
        height: int = 1024,
        apply_pe: bool = True,
    ) -> list[torch.Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]

        text_hiddens = []

        for p in prompt:
            if apply_pe and self.use_pe and self.pe_model is not None:
                enhanced = self._enhance_prompt(p, device, width=width, height=height)
                logger.info("PE: original='%s...' enhanced='%s...'", p[:50], enhanced[:50])
                p = enhanced
            ids = self.tokenizer(
                p,
                add_special_tokens=True,
                truncation=True,
                padding=False,
            )["input_ids"]

            if len(ids) == 0:
                if self.tokenizer.bos_token_id is not None:
                    ids = [self.tokenizer.bos_token_id]
                else:
                    ids = [0]

            input_ids = torch.tensor([ids], device=device)
            with torch.no_grad():
                outputs = self.text_encoder(input_ids=input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-2][0]

            for _ in range(num_images_per_prompt):
                text_hiddens.append(hidden)

        return text_hiddens

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @staticmethod
    def _pad_text(text_hiddens: list[torch.Tensor], device: torch.device, dtype: torch.dtype, text_in_dim: int):
        B = len(text_hiddens)
        if B == 0:
            return torch.zeros((0, 0, text_in_dim), device=device, dtype=dtype), torch.zeros(
                (0,), device=device, dtype=torch.long
            )
        normalized = [
            th.squeeze(1).to(device).to(dtype) if th.dim() == 3 else th.to(device).to(dtype) for th in text_hiddens
        ]
        lens = torch.tensor([t.shape[0] for t in normalized], device=device, dtype=torch.long)
        Tmax = int(lens.max().item())
        text_bth = torch.zeros((B, Tmax, text_in_dim), device=device, dtype=dtype)
        for i, t in enumerate(normalized):
            text_bth[i, : t.shape[0], :] = t
        return text_bth, lens

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1 and not self.is_distilled

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _resize_dimensions(self, height: int, width: int) -> tuple[int, int]:
        resized_height = height - height % self.vae_scale_factor
        resized_width = width - width % self.vae_scale_factor
        return resized_height, resized_width

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        if (height is not None and height % self.vae_scale_factor != 0) or (
            width is not None and width % self.vae_scale_factor != 0
        ):
            resized_height, resized_width = self._resize_dimensions(height, width)
            logger.warning(
                "`height` and `width` have to be divisible by %s "
                "but are %s and %s. Dimensions will be resized to %s and %s accordingly",
                self.vae_scale_factor,
                height,
                width,
                resized_height,
                resized_width,
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in ["latents", "prompt_embeds"] for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError("`callback_on_step_end_tensor_inputs` must be a subset of ['latents', 'prompt_embeds'].")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please forward only one.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Must provide either `prompt` or `prompt_embeds`.")

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = "",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: list[torch.FloatTensor] | None = None,
        negative_prompt_embeds: list[torch.FloatTensor] | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            logger.warning("This model only supports a single prompt. Taking only the first.")
        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else guidance_scale
        )
        generator = req.sampling_params.generator or generator
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

        device = self._execution_device
        dtype = self.transformer.dtype

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        self.check_inputs(prompt=prompt, height=height, width=width, prompt_embeds=prompt_embeds)
        height, width = self._resize_dimensions(height, width)

        if prompt is not None:
            if isinstance(prompt, str):
                prompt = [prompt]

        if prompt_embeds is not None:
            text_hiddens = prompt_embeds
        else:
            text_hiddens = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                width=width,
                height=height,
                apply_pe=self._should_apply_pe(req),
            )

        if self.do_classifier_free_guidance:
            if negative_prompt_embeds is not None:
                uncond_text_hiddens = negative_prompt_embeds
            else:
                if negative_prompt is None:
                    negative_prompt = ""
                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt] * batch_size
                uncond_text_hiddens = self.encode_prompt(
                    negative_prompt, device, num_images_per_prompt, width=width, height=height, apply_pe=False
                )

        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = self.transformer.config.in_channels

        if latents is None:
            latents = randn_tensor(
                (batch_size * num_images_per_prompt, latent_channels, latent_h, latent_w),
                generator=generator,
                device=device,
                dtype=dtype,
            )

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        self.scheduler.set_timesteps(sigmas=sigmas[:-1], device=device)

        if self.do_classifier_free_guidance:
            cfg_text_hiddens = list(uncond_text_hiddens) + list(text_hiddens)
        else:
            cfg_text_hiddens = text_hiddens

        text_bth, text_lens = self._pad_text(
            text_hiddens=cfg_text_hiddens, device=device, dtype=dtype, text_in_dim=self.transformer.config.text_in_dim
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if self.do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents, latents], dim=0)
                    t_batch = torch.full(
                        (batch_size * num_images_per_prompt * 2,), t.item(), device=device, dtype=dtype
                    )
                else:
                    latent_model_input = latents
                    t_batch = torch.full((batch_size * num_images_per_prompt,), t.item(), device=device, dtype=dtype)

                pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    pred_uncond, pred_cond = pred.chunk(2, dim=0)
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                latents = self.scheduler.step(pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            return DiffusionOutput(output=latents)

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(device)

        latents = latents * bn_std + bn_mean

        latents = self._unpatchify_latents(latents)

        images = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=images, stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = set()
        transformer_weights = (
            (name.replace("transformer.", "", 1), weight) for name, weight in weights if name.startswith("transformer.")
        )
        loaded_weights |= {f"transformer.{name}" for name in self.transformer.load_weights(transformer_weights)}
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        return loaded_weights
