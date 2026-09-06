from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

from .mammothmoda2_dit_model import SimpleQFormerImageRefiner, Transformer2DModel
from .rope_real import RotaryPosEmbedReal
from .schedulers import FlowMatchEulerDiscreteScheduler

logger = init_logger(__name__)


def _build_mammoth_config(od_config: OmniDiffusionConfig) -> Mammothmoda2Config:
    raw_config = od_config.tf_model_config.to_dict()
    if not raw_config:
        raise ValueError("MammothModa2 diffusion stage requires the root checkpoint config")
    return Mammothmoda2Config(**raw_config)


def _root_weight_source(
    od_config: OmniDiffusionConfig,
) -> DiffusersPipelineLoader.ComponentSource:
    if not od_config.model:
        raise ValueError("MammothModa2 diffusion stage requires a model path")
    return DiffusersPipelineLoader.ComponentSource(
        model_or_path=od_config.model,
        subfolder=None,
        revision=od_config.revision,
        prefix="",
        fall_back_to_pt=True,
    )


@dataclass(frozen=True)
class _MammothRequest:
    request_id: str
    full_hidden_states: torch.Tensor
    full_token_ids: list[int]
    answer_start_index: int
    height: int
    width: int
    text_guidance_scale: float
    cfg_range: tuple[float, float]
    num_inference_steps: int
    seed: int | None
    generator: torch.Generator | list[torch.Generator] | None


class MammothModa2DiTPipeline(nn.Module, SupportsComponentDiscovery):
    """
    MammothModa2 DiT + VAE generation stage (non-autoregressive).

    This stage expects "image condition token hidden states" from the upstream AR stage,
    and outputs image tensors via diffusion transformer + VAE decode.

    """

    _dit_modules: ClassVar[list[str]] = ["gen_transformer"]
    _encoder_modules: ClassVar[list[str]] = ["gen_image_condition_refiner"]
    _vae_modules: ClassVar[list[str]] = ["gen_vae"]

    supports_request_batch = False
    supports_step_execution = False

    # Load only gen_* weights; ignore llm_model.* to prevent loading the entire LLM backbone in the DiT stage.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm_model.": None,
            "gen_tokenizer.": None,
        }
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.config = _build_mammoth_config(od_config)
        self.weights_sources = [_root_weight_source(od_config)]

        # --- Build DiT / VAE modules (names must match checkpoint keys) ---
        if self.config.gen_vae_config is None or self.config.gen_dit_config is None:
            raise ValueError("Mammothmoda2Config.gen_vae_config / gen_dit_config must not be None")

        self.gen_vae = AutoencoderKL.from_config(self.config.gen_vae_config)
        self.gen_transformer = Transformer2DModel.from_config(self.config.gen_dit_config)

        # llm_config is a Mammothmoda2Qwen2_5_VLConfig which has nested text_config
        llm_hidden_size = 0
        text_config = self.config.get_text_config()
        if text_config is None:
            logger.warning("No text config; failed to infer llm_hidden_size.")
        elif not hasattr(text_config, "hidden_size"):
            logger.warning("Text config exists, but has no hidden_size attribute; failed to infer llm_hidden_size.")
        else:
            llm_hidden_size = int(text_config.hidden_size or 0)
        if llm_hidden_size <= 0:
            raise ValueError(
                "Failed to infer llm hidden_size from Mammothmoda2Config.llm_config.text_config.hidden_size"
            )
        self._reinit_caption_embedder(llm_hidden_size)

        # Optional image condition Q-Former. Preview stores it as a standalone
        # module; Dev stores it under the DiT timestep/caption embedder.
        llm_model_type = getattr(self.config.llm_config, "model_type", "")
        refiner_config = self.config.gen_image_condition_refiner_config
        if refiner_config is not None and llm_model_type == "mammothmoda2_qwen3_vl":
            dit_hidden_size = int(self.gen_transformer.hidden_size)
            self.gen_transformer.time_caption_embed.image_embedder = SimpleQFormerImageRefiner(
                hidden_size=llm_hidden_size,
                output_hidden_size=dit_hidden_size,
                num_heads=max(1, dit_hidden_size // 128),
                **refiner_config,
            )
            self.gen_image_condition_refiner = None
        elif refiner_config is not None:
            self.gen_image_condition_refiner = SimpleQFormerImageRefiner(
                hidden_size=llm_hidden_size,
                **refiner_config,
            )
        else:
            self.gen_image_condition_refiner = None

        # Precompute rotary freqs for diffusion transformer
        # IMPORTANT: follow upstream mammothmoda: use top-level `config.gen_axes_*`
        # (the checkpoint's `gen_dit_config.axes_lens` can be as small as 1024,
        # which is insufficient for vLLM dummy-run/cudagraph warmup).
        self.gen_freqs_cis = RotaryPosEmbedReal.get_freqs_real(
            tuple(self.config.gen_axes_dim_rope),
            tuple(self.config.gen_axes_lens),
            theta=10000,
        )

        self._llm_hidden_size = llm_hidden_size

    def _reinit_caption_embedder(self, in_features: int) -> None:
        # Align with upstream Mammothmoda2Model's `reinit_caption_embedder`:
        # Use Qwen2RMSNorm(in_features) + Linear(in_features -> out_features).
        out_features = int(getattr(self.gen_transformer, "hidden_size", 0) or self.gen_transformer.config.hidden_size)
        self.gen_transformer.time_caption_embed.caption_embedder = nn.Sequential(
            Qwen2RMSNorm(in_features, eps=1e-5),
            nn.Linear(in_features, out_features, bias=True),
        )

    def _parse_request(self, req: DiffusionRequestBatch) -> _MammothRequest:
        if req.num_reqs != 1:
            raise ValueError("MammothModa2 diffusion requires exactly one request")
        request_id = req.request_id
        sampling = req.sampling_params
        if sampling.num_outputs_per_prompt != 1:
            raise ValueError(f"MammothModa2 requires num_outputs_per_prompt=1 for request {request_id}")

        prompt = req.prompts[0]
        prompt = prompt if isinstance(prompt, dict) else {}
        info = prompt.get("additional_information")
        if req.is_dummy_run():
            full_hidden_states = torch.zeros((2, self._llm_hidden_size), dtype=torch.float32, device="cpu")
            full_token_ids = [0, int(self.config.llm_config.gen_vocab_start_index)]
            answer_start_index = 1
        else:
            if not isinstance(info, dict):
                raise ValueError(f"Missing additional_information AR conditions for request {request_id}")
            full_hidden_states = info.get("full_hidden_states")
            full_token_ids = info.get("full_token_ids")
            if not isinstance(full_hidden_states, torch.Tensor) or not isinstance(full_token_ids, list):
                raise ValueError(f"Expected full_hidden_states tensor and full_token_ids list for request {request_id}")
            try:
                answer_start_index = int(info.get("answer_start_index"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid answer_start_index for request {request_id}") from exc
            if full_hidden_states.ndim != 2:
                raise ValueError(f"Expected 2D full_hidden_states for request {request_id}")
            if full_hidden_states.shape[0] != len(full_token_ids):
                raise ValueError(f"AR hidden-state/token-count mismatch for request {request_id}")
            if not 0 <= answer_start_index <= len(full_token_ids):
                raise ValueError(f"answer_start_index outside token range for request {request_id}")
            try:
                full_token_ids = [int(token_id) for token_id in full_token_ids]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid full_token_ids for request {request_id}") from exc

        dimensions = []
        for name in ("height", "width"):
            value = DiffusionRequestBatch.get_prompt_field(prompt, name)
            if value is None:
                value = getattr(sampling, name)
            if value is None:
                value = 1024
            try:
                dimensions.append(int(value))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid image size {name}={value!r} for request {request_id}") from exc
        height, width = dimensions
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image size: {height}x{width} for request {request_id}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Image size must be multiples of 16, got {height}x{width} for request {request_id}")

        extra_args = sampling.extra_args or {}
        guidance = extra_args.get(
            "text_guidance_scale", sampling.guidance_scale if sampling.guidance_scale_provided else 9.0
        )
        try:
            text_guidance_scale = float(guidance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid text_guidance_scale for request {request_id}") from exc
        steps = extra_args.get(
            "num_inference_steps", sampling.num_inference_steps if sampling.num_inference_steps is not None else 50
        )
        try:
            num_inference_steps = int(steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid num_inference_steps for request {request_id}") from exc
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive for request {request_id}")
        cfg_range = extra_args.get("cfg_range", [0.0, 1.0])
        if not isinstance(cfg_range, (list, tuple)) or len(cfg_range) != 2:
            raise ValueError(f"cfg_range requires two values for request {request_id}")
        try:
            cfg_start, cfg_end = float(cfg_range[0]), float(cfg_range[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"cfg_range requires two values convertible to floats for request {request_id}") from exc
        if not 0 <= cfg_start <= cfg_end <= 1:
            raise ValueError(f"cfg_range must satisfy 0 <= start <= end <= 1 for request {request_id}")

        return _MammothRequest(
            request_id=request_id,
            full_hidden_states=full_hidden_states,
            full_token_ids=full_token_ids,
            answer_start_index=answer_start_index,
            height=height,
            width=width,
            text_guidance_scale=text_guidance_scale,
            cfg_range=(cfg_start, cfg_end),
            num_inference_steps=num_inference_steps,
            seed=sampling.seed,
            generator=sampling.generator,
        )

    def _split_ar_conditions(
        self,
        *,
        full_hidden_states: torch.Tensor,
        full_token_ids: list[int],
        answer_start_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split AR-stage hidden states into text / image condition embeds.

        The token ids that distinguish question (text) tokens, generated visual
        tokens, and multi-modal placeholder tokens are read from the model config
        (``gen_vocab_start_index`` and the vision placeholder token ids), so the
        caller no longer needs to pass them. Mirrors the masking the bespoke
        MammothModa2 example performed via ar2dit.
        """
        gen_vocab_start_index = int(self.config.llm_config.gen_vocab_start_index)
        visual_ids = [
            int(self.config.image_token_id),
            int(self.config.video_token_id),
            int(self.config.vision_start_token_id),
            int(self.config.vision_end_token_id),
        ]

        device = full_hidden_states.device
        token_ids = torch.tensor(full_token_ids, dtype=torch.long, device=device)
        positions = torch.arange(token_ids.shape[0], device=device)
        questions_mask = positions < answer_start_index
        answers_mask = ~questions_mask
        gen_token_mask = token_ids >= gen_vocab_start_index
        visual_token_mask = torch.isin(token_ids, torch.tensor(visual_ids, dtype=torch.long, device=device))
        text_mask = questions_mask & ~(visual_token_mask | gen_token_mask)
        image_mask = answers_mask & gen_token_mask

        text_cond = full_hidden_states[text_mask].to(dtype=torch.float32).contiguous()
        image_cond = full_hidden_states[image_mask].to(dtype=torch.float32).contiguous()
        return text_cond, image_cond

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        request = self._parse_request(req)
        text_cond, image_cond = self._split_ar_conditions(
            full_hidden_states=request.full_hidden_states,
            full_token_ids=request.full_token_ids,
            answer_start_index=request.answer_start_index,
        )
        if image_cond.shape[0] == 0:
            answer_token_ids = request.full_token_ids[request.answer_start_index :]
            raise ValueError(
                "MammothModa2 AR stage produced no visual-token hidden states for "
                f"request {request.request_id}; the DiT stage requires at least one generated visual token. "
                f"Generated token ids: {answer_token_ids[:32]}"
            )

        # Move to model device/dtype.
        model_device = next(self.parameters()).device
        if self.gen_image_condition_refiner is not None:
            target_dtype = next(self.gen_image_condition_refiner.parameters()).dtype
        else:
            target_dtype = next(self.gen_transformer.parameters()).dtype

        text_cond = text_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()
        image_cond = image_cond.to(device=model_device, dtype=target_dtype, non_blocking=True).contiguous()

        text_embeds = text_cond.unsqueeze(0)  # [1, T_text, H]
        text_attention_mask = torch.ones(
            (1, text_embeds.shape[1]),
            dtype=torch.bool,
            device=text_embeds.device,
        )

        image_embeds = image_cond.unsqueeze(0)  # [1, T_img, H]
        image_attention_mask = torch.ones(
            (1, image_embeds.shape[1]),
            dtype=torch.bool,
            device=image_embeds.device,
        )

        # Apply optional refiner ONLY on image condition tokens.
        if self.gen_image_condition_refiner is not None and image_embeds.shape[1] > 0:
            image_embeds = self.gen_image_condition_refiner(image_embeds, ~image_attention_mask.bool())
            image_attention_mask = torch.ones(
                image_embeds.shape[:2],
                dtype=torch.bool,
                device=image_embeds.device,
            )

        nested_image_embedder = getattr(self.gen_transformer.time_caption_embed, "image_embedder", None)
        if nested_image_embedder is None:
            prompt_embeds = torch.cat([text_embeds, image_embeds], dim=1)
            prompt_attention_mask = torch.cat([text_attention_mask, image_attention_mask], dim=1)
            ar_image_embeds = None
            ar_image_attention_mask = None
        else:
            prompt_embeds = text_embeds
            prompt_attention_mask = text_attention_mask
            ar_image_embeds = image_embeds
            ar_image_attention_mask = image_attention_mask

        # Empty unconditional prompt for classifier-free guidance.
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if request.text_guidance_scale > 1.0:
            hidden_size = int(prompt_embeds.shape[-1])
            negative_prompt_embeds = torch.zeros(
                (1, 0, hidden_size),
                dtype=target_dtype,
                device=prompt_embeds.device,
            )
            negative_prompt_attention_mask = torch.zeros(
                (1, 0),
                dtype=torch.bool,
                device=prompt_embeds.device,
            )

        generator = request.generator
        if generator is None and request.seed is not None:
            generator = torch.Generator(device=model_device).manual_seed(request.seed)

        # Output image size (px), passed from stage input processor.
        height, width = request.height, request.width
        vae_scale_factor = 16

        latent_channels = int(self.gen_transformer.config.in_channels)
        shape = (1, latent_channels, 2 * height // vae_scale_factor, 2 * width // vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=prompt_embeds.device, dtype=prompt_embeds.dtype)

        scheduler = FlowMatchEulerDiscreteScheduler()

        scheduler.set_timesteps(
            num_inference_steps=request.num_inference_steps,
            device=prompt_embeds.device,
            num_tokens=latents.shape[-2] * latents.shape[-1],
        )

        # Run diffusion loop (CFG supported when text_guidance_scale > 1.0)
        total_steps = max(1, len(scheduler.timesteps))
        for i, t in enumerate(scheduler.timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            model_pred = self.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=prompt_embeds,
                text_attention_mask=prompt_attention_mask,
                ref_image_hidden_states=None,
                ar_image_hidden_states=ar_image_embeds,
                ar_image_attention_mask=ar_image_attention_mask,
                freqs_cis=self.gen_freqs_cis,
            )
            guidance_scale = (
                request.text_guidance_scale if request.cfg_range[0] <= i / total_steps <= request.cfg_range[1] else 1.0
            )
            if guidance_scale > 1.0 and negative_prompt_embeds is not None:
                model_pred_uncond = self.gen_transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    text_hidden_states=negative_prompt_embeds,
                    text_attention_mask=negative_prompt_attention_mask,
                    ref_image_hidden_states=None,
                    freqs_cis=self.gen_freqs_cis,
                )
                model_pred = model_pred_uncond + guidance_scale * (model_pred - model_pred_uncond)
            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = latents.to(dtype=prompt_embeds.dtype)

        # VAE decode
        if self.gen_vae.config.scaling_factor is not None:
            latents = latents / self.gen_vae.config.scaling_factor
        if self.gen_vae.config.shift_factor is not None:
            latents = latents + self.gen_vae.config.shift_factor
        image = self.gen_vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
