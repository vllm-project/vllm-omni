# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.diffusion.data import DiffusionOutput

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState


class QwenImageStepwiseMixin:
    """Shared step-wise (continuous-batching) execution hooks for QwenImage pipelines.

    Provides the four hooks the DiffusionRunner calls each step plus two shared
    helpers.  Concrete classes must supply:
      - ``self.vae_scale_factor``, ``self.vae``, ``self.transformer``
      - ``self.interrupt``, ``self.attention_kwargs``, ``self._current_timestep``
      - ``self.default_sample_size``
      - ``predict_noise_maybe_with_cfg()`` / ``scheduler_step_maybe_with_cfg()``
        (from ``QwenImageCFGParallelMixin``)
      - ``_unpack_latents()`` (static method on the concrete class)
      - ``self.stage_durations`` (optional, for profiling)

    Only ``prepare_encode`` must be defined per-pipeline because it is the only
    hook that differs between T2I, single-image edit, and multi-image edit.
    """

    def _build_denoise_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list,
        txt_seq_lens: list[int] | None,
        do_true_cfg: bool,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        negative_txt_seq_lens: list[int] | None,
        image_latents: torch.Tensor | None = None,
        extra_transformer_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
        """Build positive/negative transformer kwargs and output_slice for one denoise step.

        When *image_latents* is provided (edit pipelines), the output latents
        are concatenated with image latents along the sequence dim and
        ``output_slice = latents.size(1)`` tells the caller where the
        output-only region ends so noise prediction can be cropped back.
        """
        extra_transformer_kwargs = extra_transformer_kwargs or {}
        t_for_model = timestep.expand(latents.shape[0]).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        positive_kwargs: dict[str, Any] = {
            "hidden_states": latent_model_input,
            "timestep": t_for_model / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            **extra_transformer_kwargs,
        }
        if do_true_cfg:
            negative_kwargs: dict[str, Any] | None = {
                "hidden_states": latent_model_input,
                "timestep": t_for_model / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": negative_txt_seq_lens,
                **extra_transformer_kwargs,
            }
        else:
            negative_kwargs = None

        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
    ) -> DiffusionOutput:
        """Unpack, normalize, and VAE-decode latents into a DiffusionOutput."""
        if output_type == "latent":
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def denoise_step(
        self,
        input_batch: "InputBatch",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """One denoise step: read from *input_batch*, delegate to CFGParallelMixin."""
        del kwargs
        if self.interrupt:
            return None

        t = input_batch.timesteps
        self._current_timestep = t
        self.transformer.do_true_cfg = input_batch.do_true_cfg

        extra_transformer_kwargs = {
            "attention_kwargs": self.attention_kwargs,
            "return_dict": False,
        }
        get_extra_kwargs = getattr(self, "_get_stepwise_transformer_kwargs", None)
        if callable(get_extra_kwargs):
            extra_transformer_kwargs.update(get_extra_kwargs(input_batch))

        positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
            latents=input_batch.latents,
            timestep=t,
            guidance=input_batch.guidance,
            prompt_embeds=input_batch.prompt_embeds,
            prompt_embeds_mask=input_batch.prompt_embeds_mask,
            img_shapes=input_batch.img_shapes,
            txt_seq_lens=input_batch.txt_seq_lens,
            do_true_cfg=input_batch.do_true_cfg,
            negative_prompt_embeds=input_batch.negative_prompt_embeds,
            negative_prompt_embeds_mask=input_batch.negative_prompt_embeds_mask,
            negative_txt_seq_lens=input_batch.negative_txt_seq_lens,
            image_latents=input_batch.image_latents,
            extra_transformer_kwargs=extra_transformer_kwargs,
        )

        return self.predict_noise_maybe_with_cfg(
            input_batch.do_true_cfg,
            input_batch.true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            input_batch.cfg_normalize,
            output_slice,
        )

    def step_scheduler(
        self,
        state: "DiffusionRequestState",
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One scheduler step: update ``state.latents`` and advance ``step_index``."""
        del kwargs
        if self.interrupt:
            return
        t = state.current_timestep
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1

    def post_decode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents from *state*."""
        self._current_timestep = None
        default_size = self.default_sample_size * self.vae_scale_factor
        height = state.extra.get("height") or state.sampling.height or default_size
        width = state.extra.get("width") or state.sampling.width or default_size
        output_type = kwargs.get("output_type") or state.sampling.output_type or "pil"
        if state.latents is None:
            raise ValueError(f"Request {state.request_id} has no latents to decode.")
        return self._decode_latents(state.latents, height, width, output_type)
