# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import torch.nn as nn

from vllm_omni.platforms import current_omni_platform

from .config import PID_SAMPLING_CONFIG
from .pid_net import PidNet
from .text_encoder import GemmaTextEncoder

logger = logging.getLogger(__name__)


class PidInferenceModel(nn.Module):
    """Inference-only PiD model (model-agnostic).

    Args:
        net_kwargs: Passed directly to PidNet.__init__.
        gemma_model_id: Local path or HF ID for gemma-2-2b-it.
        sampling_overrides: Optional overrides for PID_SAMPLING_CONFIG.
        precision: Compute precision preset, one of "float32" / "float16" /
            "bfloat16". Mirrors PixelDiTModelConfig.precision. For any
            non-float32 value, the tensor container stays float32 (matching
            the student's calibrated distribution) and matmuls run under
            ``torch.autocast(..., dtype=precision)``. ``"float32"`` disables
            autocast entirely (pure fp32 forward), used for precision
            baselines or checkpoints that overflow under bf16/fp16.
    """

    def __init__(
        self,
        net_kwargs: dict,
        gemma_model_id: str,
        sampling_overrides: dict | None = None,
        precision: str = "bfloat16",
        enforce_eager: bool = False,
    ):
        super().__init__()
        self.net = PidNet(**net_kwargs)
        self.text_encoder = GemmaTextEncoder(gemma_model_id, precision=precision)

        samp = dict(PID_SAMPLING_CONFIG)
        if sampling_overrides:
            samp.update(sampling_overrides)
        self._cfg = type("Cfg", (), samp)()

        # Replicate PixelDiTModel.__init__ precision resolution: the tensor
        # container is always float32; for non-float32 precision, matmuls run
        # under autocast(dtype=requested). precision="float32" disables
        # autocast entirely (pure fp32 forward).
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if precision not in _dtype_map:
            raise ValueError(f"precision must be one of {list(_dtype_map)}, got {precision!r}")
        requested_dtype = _dtype_map[precision]
        if requested_dtype != torch.float32:
            self.autocast_dtype = requested_dtype
        else:
            self.autocast_dtype = None
        self.precision = torch.float32
        self.tensor_kwargs = {"dtype": self.precision}

        # torch.compile support (opt-in via enable_compile()).
        # Compilation is lazy and cached per output resolution (H, W).
        self._compile_enabled = False
        self._compiled_nets: dict[tuple[int, int], torch.nn.Module] = {}
        if not enforce_eager:
            self.enable_compile()

        logger.debug(
            "PidInferenceModel: net params=%s, precision=%s (autocast=%s, container=%s)",
            f"{sum(p.numel() for p in self.net.parameters()):,}",
            precision,
            self.autocast_dtype,
            self.precision,
        )

    # ------------------------------------------------------------------
    # Net output <-> (x0, velocity) conversion
    # ------------------------------------------------------------------

    def _net_output_to_x0(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str = "velocity",
    ) -> torch.Tensor:
        """Convert net output to x0 estimate."""
        if prediction_type == "x0":
            return net_output.to(x_t.dtype)
        if prediction_type == "velocity":
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s)
            return (x_t.double() - t_shaped * net_output.double()).to(x_t.dtype)
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _net_output_to_velocity(
        self,
        x_t: torch.Tensor,
        net_output: torch.Tensor,
        t: torch.Tensor,
        prediction_type: str = "velocity",
    ) -> torch.Tensor:
        """Convert net output to velocity estimate."""
        if prediction_type == "velocity":
            return net_output
        if prediction_type == "x0":
            s = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
            t_shaped = t.double().view(*s).clamp(min=5e-2)
            return ((x_t.double() - net_output.double()) / t_shaped).to(x_t.dtype)
        raise ValueError(f"Invalid prediction_type: {prediction_type}")

    def _velocity_to_x0(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """velocity -> x0: x0 = x_t - t * v (uses config prediction_type)."""
        return self._net_output_to_x0(x_t, v, t, self._cfg.prediction_type)

    # ------------------------------------------------------------------
    # torch.compile (opt-in)
    # ------------------------------------------------------------------

    def enable_compile(self, mode: str = "default") -> None:
        """Arm torch.compile for :attr:`net`.

        Compilation is lazy — the actual ``torch.compile`` call happens on the
        first ``generate_samples_from_batch`` for each output resolution and
        is cached thereafter.  ``mode`` is passed directly to
        ``torch.compile``; use ``"max-autotune"`` for maximum throughput at
        the cost of a much slower first compile.
        """
        if not current_omni_platform.supports_torch_inductor():
            logger.warning(
                "PidInferenceModel: torch.compile skipped (platform %s does not support inductor); running eager.",
                current_omni_platform.device_name,
            )
            return
        self._compile_enabled = True
        self._compile_mode = mode
        logger.info("PidInferenceModel: torch.compile armed (lazy, per resolution).")

    def _maybe_compile_net(self, image_h: int, image_w: int, text_len: int, device: torch.device) -> torch.nn.Module:
        """Return compiled net for this shape, or eager net if compile is off."""
        if not self._compile_enabled:
            return self.net
        key = (int(image_h), int(image_w))
        compiled = self._compiled_nets.get(key)
        if compiled is None:
            logger.info(
                "PidInferenceModel: warming pos caches + compiling net for %dx%d",
                image_h,
                image_w,
            )
            self.net.precompute_positional_caches(
                image_height=image_h,
                image_width=image_w,
                text_length=text_len,
                device=device,
                pixel_dtype=self.precision,
            )
            compiled = torch.compile(self.net, mode=self._compile_mode, dynamic=False)
            self._compiled_nets[key] = compiled
        return compiled

    # ------------------------------------------------------------------
    # Timestep schedule
    # ------------------------------------------------------------------

    def _get_t_list(self, device, num_steps: int | None = None) -> torch.Tensor:
        target = num_steps or self._cfg.student_sample_steps
        full_t = torch.tensor(self._cfg.student_t_list, device=device, dtype=torch.float32)
        if target != len(full_t) - 1:
            indices = torch.linspace(0, len(full_t) - 1, target + 1).round().long()
            return full_t[indices]
        return full_t

    # ------------------------------------------------------------------
    # SDE sample loop
    # ------------------------------------------------------------------

    def _sample_loop(
        self,
        noise: torch.Tensor,
        t_list: torch.Tensor,
        caption_embs: torch.Tensor,
        lq_latent: torch.Tensor,
        degrade_sigma: torch.Tensor,
        net: torch.nn.Module | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        B = noise.shape[0]
        timescale = self._cfg.fm_timescale
        sample_type = getattr(self._cfg, "student_sample_type", "sde")
        prediction_type = getattr(self._cfg, "prediction_type", "velocity")
        autocast_ctx = (
            torch.autocast(noise.device.type, dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        )
        if net is None:
            net = self.net
        x = noise
        n_steps = len(t_list) - 1
        with autocast_ctx:
            for i, (t_cur, t_next) in enumerate(zip(t_list[:-1], t_list[1:])):
                t_cur_batch = t_cur.expand(B)
                t_scaled = t_cur_batch * timescale

                v_pred = net(
                    x,
                    t_scaled,
                    caption_embs,
                    lq_latent=lq_latent,
                    degrade_sigma=degrade_sigma,
                )

                if i < n_steps - 1:
                    if sample_type == "ode":
                        v_for_step = self._net_output_to_velocity(x, v_pred, t_cur_batch, prediction_type)
                        dt = t_next - t_cur
                        x = x + dt * v_for_step
                    else:
                        x0_pred = self._net_output_to_x0(x, v_pred, t_cur_batch, prediction_type)
                        eps_infer = torch.randn(
                            x0_pred.shape,
                            device=x0_pred.device,
                            dtype=x0_pred.dtype,
                            generator=generator,
                        )
                        s = [B] + [1] * (x.ndim - 1)
                        t_next_bcast = t_next.reshape(1).expand(s)
                        x = (1.0 - t_next_bcast) * x0_pred + t_next_bcast * eps_infer
                else:
                    x = self._net_output_to_x0(x, v_pred, t_cur_batch, prediction_type)

        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        lq_latent: torch.Tensor,  # (B, C_lq, zH, zW)
        caption: str | list[str],
        output_size: tuple[int, int],  # (H, W) pixel output
        degrade_sigma: float = 0.0,
        num_steps: int = 4,
        seed: int = 0,
    ) -> torch.Tensor:
        """Run PiD decode.

        Returns:
            (B, 3, H, W) tensor in [-1, 1].
        """
        if isinstance(caption, str):
            caption = [caption]
        # The runner passes one request's latents as a single batch ([n, C, zH,
        # zW] for num_outputs_per_prompt = n) with a single caption string, so
        # broadcast the caption to the latent batch size.
        B = int(lq_latent.shape[0])
        if len(caption) == 1 and B > 1:
            caption = caption * B
        elif len(caption) != B:
            raise ValueError(f"PiD decode: caption count ({len(caption)}) must be 1 or match latent batch size ({B})")

        # Use tensor_kwargs (dtype-only; device derived from lq_latent at
        # call time) to match the original PixelDiTModel: the student was
        # calibrated against a float32 container; autocast(...) inside
        # _sample_loop handles mixed-precision matmuls. Feeding bf16 directly
        # skips the float32 container and shifts the input distribution.
        device = lq_latent.device
        tensor_kwargs = {**self.tensor_kwargs, "device": device}

        caption_embs = self.text_encoder.encode(caption)
        caption_embs = caption_embs.to(**tensor_kwargs)

        lq_latent = lq_latent.to(**tensor_kwargs)
        degrade_sigma_tensor = torch.full((B,), float(degrade_sigma), **tensor_kwargs)

        gen = torch.Generator(device=device).manual_seed(int(seed))
        img_h, img_w = output_size
        noise = torch.randn(B, 3, img_h, img_w, device=device, generator=gen)

        # Resolve the net to use: compiled (per-resolution cache) or eager.
        text_len = min(caption_embs.shape[1], self.net.txt_max_length)
        net = self._maybe_compile_net(img_h, img_w, text_len, device)

        effective_steps = num_steps or self._cfg.student_sample_steps

        if effective_steps == 1:
            t_student = torch.full(
                (B,),
                self._cfg.student_t_list[0],
                **tensor_kwargs,
            )
            t_scaled = t_student * self._cfg.fm_timescale
            v = net(
                noise,
                t_scaled,
                caption_embs,
                lq_latent=lq_latent,
                degrade_sigma=degrade_sigma_tensor,
            )
            x0 = self._velocity_to_x0(noise, v, t_student)
        else:
            t_list = self._get_t_list(device, effective_steps)
            x0 = self._sample_loop(
                noise,
                t_list,
                caption_embs,
                lq_latent,
                degrade_sigma_tensor,
                net=net,
                generator=gen,
            )

        return x0.clamp(-1, 1)
