# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2026 Skywork AI and contributors.
# Adapted from SkyworkAI/Matrix-Game, revision
# 5446525c80da2a4a2f87801addbbc7faa8d2486e (Matrix-Game-4).
"""Native chunk execution with model-owned temporal KV and codec caches.

Sampling follows the reference scheduler, warped timesteps, noise generators and
float64 x0 calculation. Decoded pixels remain on device; the Omni model runner
owns presentation conversion and transport after the clean-KV pass is queued.
"""

import torch

from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.device_copy import to_device_nonblocking

from .sampling import Chunk, FlowMatchScheduler, warp_steps

CONTROL_PAD_CHANNELS = 52  # zero channels after the 48 control channels
TIMESTEP_SHIFT = 5.0
WINDOW = 21  # attention window in latent frames, sink included


class MatrixStream:
    """Accept CPU uint8 RGB [T, H, W, 3]: 1 frame first, then 4 per chunk.

    Racing: the sink is frame zero (absolute RoPE). Shooters: the sink is a KV-only prefix at position
    zero whose keys follow the current block at a fixed distance once the window is full.
    """

    def __init__(
        self,
        transformer,
        vae,
        prompt_embeds,
        *,
        seed=0,
        max_latents=8192,
        vae_backend="taehv",
        steps=(1000,),
        first_steps=(1000,),
        profile=False,
        sink_latent=None,
        sink_mode="first_frame",
        sink_rope_mode="absolute",
        frame_size=(480, 864),
    ):
        if vae_backend != "taehv":
            raise ValueError("Only the TAEHV backend is supported")
        if sink_mode not in ("kv_only", "first_frame") or sink_rope_mode not in ("absolute", "fixed"):
            raise ValueError("Invalid sink policy")
        if max_latents < 1:
            raise ValueError("max_latents must be positive")
        self.transformer, self.vae, self.profile, self.vae_backend = transformer, vae, profile, vae_backend
        self.device, self.dtype = torch.device("cuda"), torch.bfloat16
        self.max_latents = max_latents
        self.height, self.width = frame_size
        if any(type(size) is not int or size <= 0 or size % 32 for size in frame_size):
            raise ValueError("Frame height and width must be positive multiples of 32")
        self.shape = (1, 1, 48, self.height // 16, self.width // 16)
        self.frame_tokens = (self.height // 32) * (self.width // 32)
        if sink_latent is not None and tuple(sink_latent.shape) != self.shape:
            raise ValueError("Sink latent must match the configured frame size")
        self.sink_mode, self.sink_rope_mode = sink_mode, sink_rope_mode
        self.first_frame_latent = sink_latent if sink_mode == "first_frame" else None
        self.sink_offset = int(sink_latent is not None and sink_mode == "kv_only")
        transformer.ensure_rope_positions(max(8192, max_latents + self.sink_offset))
        self.scheduler = FlowMatchScheduler(shift=TIMESTEP_SHIFT, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.steps = warp_steps(self.scheduler, list(steps))
        self.steps_first = warp_steps(self.scheduler, list(first_steps))
        self.initial_rng = torch.Generator("cpu").manual_seed(seed + 1)
        self.racing_rng = torch.Generator("cpu").manual_seed(seed)
        self.renoise_rng = torch.Generator(self.device).manual_seed(seed)
        self.kv = transformer.init_kv_cache(1, self.frame_tokens, self.device, self.dtype)
        self.xc = transformer.init_crossattn_cache(1, self.device, self.dtype)
        self.set_prompt(prompt_embeds)
        self.index = self.frame = 0
        self.failed = self.closed = False
        if self.sink_offset:
            # Position zero holds the sink as clean history only; it is neither rendered nor controlled.
            with torch.no_grad():
                self._forward(sink_latent.to(self.device, self.dtype), torch.zeros((), device=self.device), None, 0)
        vae.clear_cache()

    def set_prompt(self, embeddings):
        """Change text conditioning at a chunk boundary without clearing visual history."""
        context = embeddings.to(self.device, self.dtype)
        if tuple(context.shape) != (1, 512, 4096):
            raise ValueError("prompt embeddings must be [1, 512, 4096]")
        self.context = context
        for cross in self.xc:
            cross.ready = False

    def _encode(self, x, param):
        """Control pixels [1, 3, T, H, W] -> one normalised latent frame [1, 1, 48, h, w], as CastleStream."""
        return self.vae.encode_stream(x).permute(0, 2, 1, 3, 4)

    def _sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Sigma of the scheduler grid point nearest to `timestep` (0 for a clean input), as in the wrapper."""
        timesteps = self.scheduler.timesteps.double().to(self.device)
        sigma = self.scheduler.sigmas.double().to(self.device)[torch.argmin((timesteps - timestep.double()).abs())]
        return torch.where(timestep.double() == 0, torch.zeros_like(sigma), sigma)

    def _forward(self, latent, timestep, control, position, kv_only=False):
        """latent [1, 1, 48, h, w] (B, F, C), control [1, 1, 48, h, w] or None -> flow [1, 1, 48, h, w]."""
        x = latent.permute(0, 2, 1, 3, 4)
        b, _, f, h, w = x.shape
        y = x.new_zeros(b, 48 + CONTROL_PAD_CHANNELS, f, h, w)
        if control is not None:
            y[:, :48] = control.permute(0, 2, 1, 3, 4)
        t = timestep.reshape(1, 1).to(self.device, torch.float32)
        fixed = self.sink_rope_mode == "fixed" and position > WINDOW
        with torch.autocast("cuda", dtype=self.dtype):
            flow = self.transformer(
                x,
                t,
                self.context,
                y,
                self.kv,
                self.xc,
                position * self.frame_tokens,
                kv_only=kv_only,
                sink_position=position - WINDOW if fixed else None,
            )
        return None if flow is None else flow.permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def push(self, pixels, *, frame_start, initial_noise=None):
        """Advance one chunk and return owned float32 TCHW device pixels."""
        if self.closed or self.failed:
            raise RuntimeError("Session is closed or failed; create a fresh session")
        count = 1 if self.index == 0 else 4
        if frame_start != self.frame or tuple(pixels.shape) != (count, self.height, self.width, 3):
            raise ValueError("Expected consecutive RGB frames: 1 initially, then 4 per chunk")
        if self.index >= self.max_latents:
            raise ValueError("Session reached its configured RoPE/frame budget")
        if pixels.dtype != torch.uint8 or pixels.device.type != "cpu":
            raise ValueError("RGB input must be CPU uint8")
        if initial_noise is not None and (
            tuple(initial_noise.shape) != self.shape or not torch.isfinite(initial_noise).all()
        ):
            raise ValueError("Initial noise must be finite and match one latent frame")
        try:
            events = []

            def mark():
                if self.profile:
                    events.append(torch.cuda.Event(enable_timing=True))
                    events[-1].record()

            mark()
            param = next(self.vae.parameters())
            # uint8 proves range/finiteness without scanning a float video on
            # the CPU. Reuse pinned staging and preserve the FP32 normalization
            # before the BF16 cast used by the reference.
            x = to_device_nonblocking(pixels, param.device).permute(3, 0, 1, 2).unsqueeze(0)
            x = (x.float() * (2.0 / 255.0) - 1.0).to(param.dtype)
            control = self._encode(x, param)
            mark()
            if tuple(control.shape) != self.shape:
                raise RuntimeError("VAE encoder output cadence changed")

            position = self.index + self.sink_offset
            if self.sink_mode == "first_frame":
                steps = self.steps_first if self.index == 0 else self.steps
            else:
                steps = self.steps_first if position < 2 else self.steps
            if self.index == 0 and self.first_frame_latent is not None:
                steps = ()
                pred = self.first_frame_latent.to(self.device, self.dtype).clone()
            else:
                if initial_noise is None:
                    rng = self.racing_rng if self.sink_mode == "first_frame" and self.index > 0 else self.initial_rng
                    initial_noise = torch.randn(self.shape, generator=rng, device="cpu", dtype=torch.float32)
                latent = initial_noise.to(self.device, self.dtype)
                for k, step in enumerate(steps):
                    timestep = torch.full((), float(step), device=self.device, dtype=torch.float32)
                    flow = self._forward(latent, timestep, control, position)
                    pred = (latent.double() - self._sigma(timestep) * flow.double()).to(self.dtype)
                    if k + 1 < len(steps):
                        flat = pred.flatten(0, 1).float()
                        noise = torch.randn(
                            flat.shape, dtype=flat.dtype, device=flat.device, generator=self.renoise_rng
                        )
                        latent = self.scheduler.add_noise(
                            flat, noise, torch.full((1,), float(steps[k + 1]), device=self.device)
                        )
                        latent = latent.unflatten(0, (1, 1)).to(self.dtype)
            mark()

            decoded = self.vae.decode_stream(pred.permute(0, 2, 1, 3, 4))[0].permute(1, 0, 2, 3)
            if tuple(decoded.shape) != (count, 3, self.height, self.width):
                raise RuntimeError("VAE decoder output cadence changed")
            mark()
            # The caller owns its output independently of the next codec invocation.
            decoded = decoded.to(dtype=torch.float32, copy=True)
            mark()
            self._forward(pred, torch.zeros((), device=self.device), None, position, kv_only=True)
            mark()
            if self.profile:
                events[-1].synchronize()
                names = ("encode", "denoise", "decode", "owned_pixels", "clean_kv")
                timings = {name: events[i].elapsed_time(events[i + 1]) for i, name in enumerate(names)}
            else:
                timings = {}
            result = Chunk(self.index, self.frame, self.frame + count - 1, decoded, pred, control, len(steps), timings)
            self.index += 1
            self.frame += count
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        current_omni_platform.synchronize()
        self.transformer.clear_graphs()
        self.kv, self.xc = [], []
        if self.vae_backend == "taehv":
            self.vae.clear_cache()
        self.closed = True
