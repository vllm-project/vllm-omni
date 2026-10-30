# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-Image text-to-image pipeline for vLLM-Omni.

Native vLLM-Omni integration of the MOSS-Image T2I model (Stage3 dense DiT),
ported from the training repository's finalized standalone pipeline
``pipeline_moss_image.py``:

  * ``MOSSImageTransformer`` lives in ``mossimage_transformer.py`` and uses the
    vLLM-Omni ``Attention`` layer; weights are loaded by vLLM's
    ``AutoWeightsLoader`` from the diffusers checkpoint's ``transformer/``
    subfolder (state-dict names match exactly).
  * Components: Qwen3.5 text encoder (bf16), FLUX.2-klein VAE (fp32),
    ``MOSSImageScheduler`` (V2 nearest-grid Euler, shift 5.0 /
    training_shift 1.0).
  * The precision contract of the reference pipeline is preserved:
    fp32 initial noise, fp32 classifier-free guidance, fp32 Euler updates,
    fp32 VAE decode with BN-statistics denormalization and 2x2 unpacking.
  * Prompts are rendered captions; the Qwen chat template is applied with
    ``add_generation_prompt=True`` and ``enable_thinking=True``, and each
    sample is encoded unpadded at its true length.

Parallelism: tensor parallelism (QKV/FFN sharding), CFG parallelism
(CFGParallelMixin), and Ulysses sequence parallelism (_sp_plan) are
supported. Ring attention and TeaCache / Cache-DiT are not (ring attention
cannot express the joint key-padding mask).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoModel, AutoTokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.guidance import (
    FlowMatchingPredictionAdapter,
    SkipLayerGuidanceConfig,
    SkipLayerGuidanceMixin,
    prepare_skip_layer_guidance,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import (
    from_pretrained_with_prefetch,
    prefetch_subfolders,
)
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.mossimage.mossimage_transformer import MOSSImageTransformer
from vllm_omni.diffusion.offloader.config import TEXT_ENCODER_COMPONENT, resolve_offload
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

_DEFAULT_RESOLUTION = 1024
_DEFAULT_NUM_INFERENCE_STEPS = 50
_DEFAULT_GUIDANCE_SCALE = 4.0
_MAX_SEQUENCE_LENGTH = 4096


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler integration using the training project's V2, nearest-grid schedule.

    Ported from ``pipeline_moss_image.MOSSImageScheduler``. Loading explicitly
    through this class uses the native implementation while retaining
    diffusers' local-directory and Hub config resolution.
    """

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, training_shift: float = 1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps: int = 50, device=None, shift: float | None = None):
        shift = self.config.shift if shift is None else float(shift)
        if num_inference_steps < 1 or shift <= 0 or self.config.training_shift <= 0:
            raise ValueError("Steps and shifts must be positive.")
        total = self.config.num_train_timesteps

        def grid(count, shift):
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count)
            return shift * values / (1 + (shift - 1) * values)

        sigmas = grid(num_inference_steps, shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        sigmas = torch.unique_consecutive(training[nearest])
        # Our sigmas have already been shifted and snapped. Avoid a second shift.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


def _sampling_shift(extra_args: dict[str, Any]) -> float | None:
    for name in ("timestep_shift", "timesteps_shift", "flow_shift"):
        if extra_args.get(name) is not None:
            return float(extra_args[name])
    return None


def get_mossimage_pre_process_func(od_config: OmniDiffusionConfig):
    """Normalize model-owned sampler settings before batch admission."""

    def pre_process_func(request):
        extra_args = request.sampling_params.extra_args or {}
        slg = SkipLayerGuidanceConfig.from_extra_args(extra_args)
        request.batch_compatibility_key = ("mossimage", _sampling_shift(extra_args), slg.batch_key)
        return request

    return pre_process_func


def get_mossimage_post_process_func(od_config: OmniDiffusionConfig):
    """Convert the pipeline's clamped [-1, 1] image tensors to PIL images."""
    from diffusers.image_processor import VaeImageProcessor

    # The pipeline uses an 8x Flux.2 VAE and 2x2 channel packing. Decoded
    # images need only denormalization and PIL conversion, not weight loading.
    image_processor = VaeImageProcessor(vae_scale_factor=16)

    def post_process_func(images: Any):
        if isinstance(images, torch.Tensor):
            if images.ndim == 4 and images.shape[1] == 3:
                return image_processor.postprocess(images.detach().float().cpu(), output_type="pil")
            return images
        return images

    return post_process_func


class MOSSImagePipeline(nn.Module, SkipLayerGuidanceMixin, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery):
    """Native MOSS-Image pipeline with request batching and step execution."""

    # Per-forward a DiffusionRequestBatch; homogeneous requests (same sampler
    # key: resolution/CFG/steps/shift/num_outputs) are fused into one batch.
    supports_request_batch: ClassVar[bool] = True
    # Continuous batching via step-wise execution (prepare_encode /
    # denoise_step / step_scheduler / post_decode).
    supports_step_execution: ClassVar[bool] = True
    slg_prediction_adapter: ClassVar[FlowMatchingPredictionAdapter] = FlowMatchingPredictionAdapter()

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        block_attrs={"transformer": ("layers",)},
        offload_submodules={"text_embedder": "blocks"},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_component_types={"text_encoder": TEXT_ENCODER_COMPONENT},
        encoder_block_attrs={"text_encoder": ("language_model.layers",)},
        # AutoModel loads the complete Qwen3.5 encoder on each worker; its
        # weights are replicated even when the DiT uses TP or Ulysses.
        encoder_dlo_weight_replication=frozenset({"text_encoder"}),
    )

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self._raise_on_unsupported_features(od_config)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.isdir(model)

        mossimage_subfolders = ["scheduler", "text_encoder", "tokenizer", "vae"]
        prefetch_subfolders(
            model,
            mossimage_subfolders,
            local_files_only=local_files_only,
            revision=od_config.revision,
        )
        self.scheduler = MOSSImageScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
            revision=od_config.revision,
        )

        # Qwen3.5 base model; bf16 like the standalone release.
        text_dtype = od_config.dtype or torch.bfloat16
        encoder_device = (
            torch.device("cpu") if resolve_offload(od_config).offloads(TEXT_ENCODER_COMPONENT) else self.device
        )
        self.text_encoder = from_pretrained_with_prefetch(
            AutoModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=mossimage_subfolders,
            local_files_only=local_files_only,
            revision=od_config.revision,
            dtype=text_dtype,
        ).to(encoder_device)
        self.text_encoder.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
            revision=od_config.revision,
        )

        # FLUX.2-klein VAE; fp32 decode is part of the release precision contract.
        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLFlux2.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=mossimage_subfolders,
            local_files_only=local_files_only,
            revision=od_config.revision,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.vae.eval()

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, MOSSImageTransformer)
        self.transformer = MOSSImageTransformer(od_config=od_config, prefix="transformer", **transformer_kwargs)

        self.default_sample_size = _DEFAULT_RESOLUTION // 16
        self._interrupt = False
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    # ------------------------------------------------------------------
    # Validation guards
    # ------------------------------------------------------------------

    @staticmethod
    def _raise_on_unsupported_features(od_config: OmniDiffusionConfig) -> None:
        pc = od_config.parallel_config
        if getattr(pc, "ring_degree", 1) and getattr(pc, "ring_degree", 1) > 1:
            raise NotImplementedError(
                "Ring attention is not supported by the MOSSImage pipeline "
                "(ring attention cannot express the joint key-padding mask); use --ulysses-degree only."
            )
        if od_config.cache_backend not in (None, "none"):
            raise NotImplementedError(
                f"Cache backend '{od_config.cache_backend}' is not supported by the MOSSImage pipeline "
                "(TeaCache/Cache-DiT require transformer-block hooks)."
            )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @contextmanager
    def _dit_residency(self) -> Iterator[None]:
        """Keep optional resident DiT blocks live only for the denoise stage."""
        controller = getattr(self, "_dlo_residency_controller", None)
        if controller is None:
            yield
            return
        try:
            controller.load_resident_layers()
            yield
        finally:
            controller.offload_resident_layers()

    def encode_prompt(
        self,
        prompt: str | list[str],
        max_sequence_length: int = _MAX_SEQUENCE_LENGTH,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with Qwen3.5's chat template at true (unpadded) length."""
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")
        rows = []
        # Unpadded individual forwards avoid padding effects in Qwen GDN.
        for value in prompts:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length).to(
                self.device
            )
            output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
        for index, row in enumerate(rows):
            embeds[index, : row.shape[0]] = row
            mask[index, : row.shape[0]] = True
        return embeds, mask

    # ------------------------------------------------------------------
    # Diffusion loop (CFGParallelMixin protocol)
    # ------------------------------------------------------------------

    def predict_noise(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        _conditioning_cache: dict | None = None,
        skip_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """One velocity prediction; CFG combines fp32 predictions."""
        skip_kwargs = {"skip_layers": skip_layers} if skip_layers else {}
        return self.transformer(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            return_dict=False,
            _conditioning_cache=_conditioning_cache,
            **skip_kwargs,
        )[0].float()

    def scheduler_step(
        self,
        noise_pred: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        per_request_scheduler=None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        scheduler = per_request_scheduler if per_request_scheduler is not None else self.scheduler
        return scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """fp32 VAE decode with BN-statistics denormalization and 2x2 unpacking."""
        z = latents.to(dtype=self.vae.dtype)
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        return self.vae.decode(z, return_dict=False)[0].clamp(-1, 1)

    # ------------------------------------------------------------------
    # Step-wise execution (continuous batching)
    # ------------------------------------------------------------------

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Encode prompts, initialize latents and the per-request scheduler."""
        sampling = state.sampling
        prompt, negative_prompt = "", ""
        p = state.prompt
        if isinstance(p, str):
            prompt = p
        elif isinstance(p, dict):
            prompt = p.get("prompt") or ""
            negative_prompt = p.get("negative_prompt") or ""

        height = sampling.height or _DEFAULT_RESOLUTION
        width = sampling.width or _DEFAULT_RESOLUTION
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        num_inference_steps = sampling.num_inference_steps or _DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale is not None else _DEFAULT_GUIDANCE_SCALE
        num_images_per_prompt = max(sampling.num_outputs_per_prompt, 1)
        max_sequence_length = sampling.max_sequence_length or _MAX_SEQUENCE_LENGTH
        max_sequence_length = max(1, min(max_sequence_length, _MAX_SEQUENCE_LENGTH))
        extra_args = getattr(sampling, "extra_args", None) or {}
        shift = _sampling_shift(extra_args)
        slg_config = SkipLayerGuidanceConfig.from_extra_args(extra_args)

        device = self.device
        req_scheduler = MOSSImageScheduler.from_config(self.scheduler.config)
        req_scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        state.extra["slg"] = prepare_skip_layer_guidance(
            slg_config,
            transformer=self.transformer,
            sigmas=req_scheduler.sigmas[:-1],
        )
        positive, positive_mask = self.encode_prompt([prompt], max_sequence_length)
        positive = positive.repeat_interleave(num_images_per_prompt, 0)
        positive_mask = positive_mask.repeat_interleave(num_images_per_prompt, 0)
        do_cfg = guidance_scale != 1.0
        negative = negative_mask = None
        if do_cfg:
            negative, negative_mask = self.encode_prompt([negative_prompt], max_sequence_length)
            negative = negative.repeat_interleave(num_images_per_prompt, 0)
            negative_mask = negative_mask.repeat_interleave(num_images_per_prompt, 0)

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=sampling.generator_device or str(device)).manual_seed(sampling.seed)
        latents = (
            randn_tensor(
                (num_images_per_prompt, self.transformer.latent_channels, height // 16, width // 16),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            if sampling.latents is None
            else sampling.latents.to(device=device, dtype=torch.float32)
        )

        state.prompt_embeds = positive
        state.prompt_embeds_mask = positive_mask
        state.negative_prompt_embeds = negative
        state.negative_prompt_embeds_mask = negative_mask
        state.latents = latents
        state.timesteps = req_scheduler.timesteps
        state.scheduler = req_scheduler
        state.step_index = 0
        state.do_true_cfg = do_cfg
        # The runner batches requests only when CFG scalars match; forward()
        # uses ``guidance_scale`` for this model, so mirror it here.
        state.sampling.true_cfg_scale = guidance_scale
        return state

    def denoise_step(
        self, input_batch: InputBatch, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any
    ) -> torch.Tensor | None:
        """One CFG-aware velocity evaluation on the runner-assembled batch."""
        del kwargs
        if self._interrupt:
            return None
        positive_kwargs = {
            "hidden_states": input_batch.latents,
            "timestep": input_batch.timesteps,
            "encoder_hidden_states": input_batch.prompt_embeds,
            "encoder_attention_mask": input_batch.prompt_embeds_mask,
        }
        negative_kwargs = None
        if input_batch.do_true_cfg:
            negative_kwargs = {
                "hidden_states": input_batch.latents,
                "timestep": input_batch.timesteps,
                "encoder_hidden_states": input_batch.negative_prompt_embeds,
                "encoder_attention_mask": input_batch.negative_prompt_embeds_mask,
            }
        selected_states = input_batch.states or states or ()
        slg_states = [state.extra.get("slg") for state in selected_states]
        slg = next((state for state in slg_states if state is not None), None)
        active, sigmas = [], []
        if slg is not None:
            for state, slg_state in zip(selected_states, slg_states):
                if slg_state is None or slg_state.config.batch_key != slg.config.batch_key:
                    raise ValueError("MOSSImage step batches require identical SLG configurations.")
                count = state.latents.shape[0]
                active.extend([slg_state.active_steps[state.step_index]] * count)
                sigmas.extend([slg_state.sigmas[state.step_index]] * count)
        with self._dit_residency():
            return self.predict_noise_with_slg(
                input_batch.do_true_cfg,
                input_batch.true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                slg_config=slg.config if slg is not None else None,
                active=active,
                sigmas=sigmas,
                latents=input_batch.latents,
            )

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """One Euler update for a single request."""
        if self._interrupt:
            return
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            state.current_timestep,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Decode latents after the denoise loop."""
        return DiffusionOutput(
            output=self._vae_decode(state.latents),
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else {},
        )

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput | list[DiffusionOutput]:
        num_reqs = len(req.requests)
        sampling_list = req.sampling_params_list
        # Homogeneity (resolution / CFG / steps / num_outputs / shift) is
        # enforced by the scheduler's request-batch compatibility key.
        sampling = sampling_list[0]
        slg_configs = [SkipLayerGuidanceConfig.from_extra_args(sp.extra_args) for sp in sampling_list]
        slg_config = slg_configs[0]
        if any(config.batch_key != slg_config.batch_key for config in slg_configs[1:]):
            raise ValueError("MOSSImage request batches require identical SLG configurations.")

        prompts: list[str] = []
        negative_prompts: list[str] = []
        has_negative = False
        for p in req.prompts:
            if isinstance(p, str):
                prompts.append(p)
                negative_prompts.append("")
            elif isinstance(p, dict):
                prompts.append(p.get("prompt") or "")
                negative = p.get("negative_prompt")
                if negative is not None:
                    has_negative = True
                negative_prompts.append(negative if negative is not None else "")
            else:
                raise ValueError(f"Unsupported prompt type for MOSSImage: {type(p)}")
        assert len(prompts) == num_reqs, f"prompt count {len(prompts)} != request count {num_reqs}"

        device = self.device
        height = sampling.height or _DEFAULT_RESOLUTION
        width = sampling.width or _DEFAULT_RESOLUTION
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        num_inference_steps = sampling.num_inference_steps or _DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale is not None else _DEFAULT_GUIDANCE_SCALE
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError(f"Invalid guidance scale: {guidance_scale}")
        num_images_per_prompt = max(sampling.num_outputs_per_prompt, 1)
        max_sequence_length = sampling.max_sequence_length or _MAX_SEQUENCE_LENGTH
        max_sequence_length = max(1, min(max_sequence_length, _MAX_SEQUENCE_LENGTH))

        # Per-request scheduler shift override. Offline: --timesteps-shift;
        # serving: the OpenAI image request's ``flow_shift`` extension field.
        extra_args = getattr(sampling, "extra_args", None) or {}
        shift = _sampling_shift(extra_args)

        # Per-request generators (seeds are request-local by design).
        generators = []
        for sp in sampling_list:
            gen = sp.generator
            if gen is None and sp.seed is not None:
                gen = torch.Generator(device=sp.generator_device or str(device)).manual_seed(sp.seed)
            generators.append(gen)

        # 1. Text embeddings (positive / negative), repeated per output image.
        positive, positive_mask = self.encode_prompt(prompts, max_sequence_length)
        positive = positive.repeat_interleave(num_images_per_prompt, 0)
        positive_mask = positive_mask.repeat_interleave(num_images_per_prompt, 0)

        do_cfg = guidance_scale != 1.0
        negative = negative_mask = None
        if do_cfg:
            negative_prompts_use = negative_prompts if has_negative else [""] * len(prompts)
            negative, negative_mask = self.encode_prompt(negative_prompts_use, max_sequence_length)
            negative = negative.repeat_interleave(num_images_per_prompt, 0)
            negative_mask = negative_mask.repeat_interleave(num_images_per_prompt, 0)

        # 2. Initial latents (fp32 noise is part of the reference contract).
        #    Request batching: each request draws its own generator-driven noise,
        #    then rows are concatenated. num_images_per_prompt is homogeneous
        #    within a request batch (enforced by the scheduler key).
        num_images_per_prompt = max(sampling.num_outputs_per_prompt, 1)
        row_shape = (num_images_per_prompt, self.transformer.latent_channels, height // 16, width // 16)
        latent_rows = []
        for sp, gen in zip(sampling_list, generators):
            if sp.latents is not None:
                row = sp.latents.to(device=device, dtype=torch.float32)
                if tuple(row.shape) != row_shape:
                    raise ValueError(f"latents must have shape {row_shape}, got {tuple(row.shape)}.")
            else:
                row = randn_tensor(row_shape, generator=gen, device=device, dtype=torch.float32)
            latent_rows.append(row)
        latents = torch.cat(latent_rows, dim=0)
        batch = latents.shape[0]

        # 3. Denoising: CFG combines fp32 predictions even when the DiT is bf16.
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        slg = prepare_skip_layer_guidance(
            slg_config,
            transformer=self.transformer,
            sigmas=self.scheduler.sigmas[:-1],
        )
        self._num_timesteps = len(self.scheduler.timesteps)
        show_progress = (not torch.distributed.is_available()) or (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
        timesteps_iter = self.scheduler.timesteps
        if show_progress:
            try:
                from tqdm.auto import tqdm

                timesteps_iter = tqdm(
                    self.scheduler.timesteps, desc="MOSSImage denoising", unit="step", dynamic_ncols=True
                )
            except ImportError:
                pass
        # These caches live only for this immutable request batch. Stepwise
        # execution deliberately does not reuse them as batch membership changes.
        positive_cache: dict = {}
        negative_cache: dict = {}
        with self._dit_residency():
            for step_index, timestep in enumerate(timesteps_iter):
                if self._interrupt:
                    break
                positive_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep,
                    "encoder_hidden_states": positive,
                    "encoder_attention_mask": positive_mask,
                    "_conditioning_cache": positive_cache,
                }
                negative_kwargs = None
                if do_cfg:
                    negative_kwargs = {
                        "hidden_states": latents,
                        "timestep": timestep,
                        "encoder_hidden_states": negative,
                        "encoder_attention_mask": negative_mask,
                        "_conditioning_cache": negative_cache,
                    }
                prediction = self.predict_noise_with_slg(
                    do_true_cfg=do_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    slg_config=slg.config if slg is not None else None,
                    active=[slg.active_steps[step_index]] * batch if slg is not None else (),
                    sigmas=[slg.sigmas[step_index]] * batch if slg is not None else (),
                    latents=latents,
                )
                latents = self.scheduler_step_maybe_with_cfg(prediction, timestep, latents, do_cfg)

        # 4. VAE decode in fp32 with BN statistics and 2x2 channel unpacking.
        images = self._vae_decode(latents)
        stage_durations = self.stage_durations if hasattr(self, "stage_durations") else {}

        if num_reqs == 1:
            return DiffusionOutput(output=images, stage_durations=stage_durations)

        # Request-batch mode: return one DiffusionOutput per request (runner
        # validates the length). Requests share the batch, so slicing rows by
        # num_images_per_prompt recovers each request's output block.
        outputs: list[DiffusionOutput] = []
        pos = 0
        for sp in sampling_list:
            k = max(sp.num_outputs_per_prompt, 1)
            outputs.append(DiffusionOutput(output=images[pos : pos + k], stage_durations=stage_durations))
            pos += k
        return outputs

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
