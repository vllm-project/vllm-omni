# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/OpenMOSS/MOVA
"""
MOVA Flow Matching Schedulers.

Contains both the base FlowMatchScheduler and the paired variant
FlowMatchPairScheduler that supports independent visual/audio
denoising schedules via dual_sigma_shift.
"""

import math

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class FlowMatchScheduler:
    """
    Flow matching scheduler for diffusion models.

    Supports linear/exponential sigma shifting, inverse timesteps,
    reverse sigmas, and shift terminal scaling.
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.train_timesteps: torch.Tensor | None = None
        self.train_sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        self.set_timesteps(num_train_timesteps)
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: float | None = None,
        dynamic_shift_len: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        if shift is not None:
            self.shift = shift

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength

        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)

        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])

        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)

        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)

        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas

        self.timesteps = self.sigmas * self.num_train_timesteps

        # Cache the initial train timesteps/sigmas the first time
        if self.train_timesteps is None:
            self.train_timesteps = self.timesteps
            self.train_sigmas = self.sigmas

        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
            self.training = True
        else:
            self.training = False

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        to_final: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)

    def return_to_timestep(
        self,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        sample_stabilized: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        return (sample - sample_stabilized) / sigma

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor | float,
    ) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        return self.linear_timesteps_weights[timestep_id]

    def calculate_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b


class FlowMatchPairScheduler(FlowMatchScheduler):
    """
    Paired flow matching scheduler for dual-modality (video + audio) generation.

    Extends FlowMatchScheduler with a pairing interface that produces [N, 2]
    tensors of timesteps/sigmas. Supports independent visual/audio denoising
    schedules via the dual_sigma_shift post-processing mode.
    """

    def __init__(
        self,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
        exponential_shift: bool = False,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = None,
    ):
        self._pair_postprocess_fn = None
        self._pair_postprocess_requires_source = False
        self.pair_timesteps: torch.Tensor | None = None
        self.pair_sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
            exponential_shift=exponential_shift,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
        )

    def set_pair_postprocess(self, fn: object | None) -> None:
        """Set a post-processing function to customize pairing behavior."""
        if fn is not None and not callable(fn):
            raise TypeError("pair_postprocess must be a callable object or None")
        self._pair_postprocess_fn = fn
        self._pair_postprocess_requires_source = False if fn is None else bool(getattr(fn, "_requires_source", False))
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        self._refresh_pair_cache()

    def set_pair_postprocess_by_name(self, name: str | None, **kwargs) -> None:
        """
        Configure the post-processing function by name.
        Supports: None / "dual_sigma_shift".
        """
        if name is None or str(name).lower() in ("none", "off", "false", "no"):
            self.set_pair_postprocess(None)
            return

        if name == "dual_sigma_shift":
            visual_shift = float(kwargs.get("visual_shift", self.shift))
            audio_shift = float(kwargs.get("audio_shift", self.shift))
            visual_denoising_strength = float(kwargs.get("visual_denoising_strength", 1.0))
            audio_denoising_strength = float(kwargs.get("audio_denoising_strength", 1.0))
            visual_mu = kwargs.get("visual_exponential_shift_mu", self.exponential_shift_mu)
            audio_mu = kwargs.get("audio_exponential_shift_mu", self.exponential_shift_mu)

            def _dual_sigma_shift(pairs: torch.Tensor, *, source: str) -> torch.Tensor:
                if pairs.ndim != 2 or pairs.shape[1] != 2:
                    raise ValueError("pairs must have shape [N, 2]")
                if pairs.shape[0] == 0:
                    raise ValueError("pairs length must be greater than 0")
                if source not in ("timesteps", "sigmas"):
                    raise ValueError("source only supports 'timesteps' or 'sigmas'")

                num_steps = pairs.shape[0]
                device = pairs.device
                dtype = pairs.dtype

                def _build_column(
                    shift_value: float, denoising_strength: float, mu_override: float | None
                ) -> torch.Tensor:
                    sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
                    if self.extra_one_step:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps + 1, device=device, dtype=dtype)[
                            :-1
                        ]
                    else:
                        base = torch.linspace(sigma_start, self.sigma_min, num_steps, device=device, dtype=dtype)

                    if self.inverse_timesteps:
                        base = torch.flip(base, dims=[0])

                    if self.exponential_shift:
                        if mu_override is None:
                            raise RuntimeError("exponential_shift is enabled but exponential_shift_mu is not provided")
                        exp_mu = math.exp(float(mu_override))
                        base = exp_mu / (exp_mu + (1 / base - 1))
                    else:
                        base = shift_value * base / (1 + (shift_value - 1) * base)

                    if self.shift_terminal is not None:
                        one_minus_z = 1 - base
                        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
                        base = 1 - (one_minus_z / scale_factor)

                    if self.reverse_sigmas:
                        base = 1 - base

                    if source == "timesteps":
                        return base * self.num_train_timesteps
                    return base

                visual_column = _build_column(visual_shift, visual_denoising_strength, visual_mu)
                audio_column = _build_column(audio_shift, audio_denoising_strength, audio_mu)
                return torch.stack([visual_column, audio_column], dim=1)

            _dual_sigma_shift._requires_source = True  # type: ignore[attr-defined]
            self.set_pair_postprocess(_dual_sigma_shift)
            return

        raise ValueError(f"Unsupported pair postprocessing name: {name}")

    def _make_pairs_from_vector(self, vec: torch.Tensor) -> torch.Tensor:
        """Default pairing: each row is (t, t)."""
        if vec.ndim != 1:
            raise ValueError("input vector must be one-dimensional")
        return torch.stack([vec, vec], dim=1)

    def get_pairs(self, source: str = "timesteps") -> torch.Tensor:
        """
        Return paired timesteps/sigmas with shape [N, 2].
        Column 0 = visual, Column 1 = audio.
        """
        if source == "timesteps":
            pairs = self.pair_timesteps
        elif source == "sigmas":
            pairs = self.pair_sigmas
        else:
            raise ValueError("source only supports 'timesteps' or 'sigmas'")

        if pairs is None:
            raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
        return pairs

    @property
    def visual_timesteps(self) -> torch.Tensor:
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized")
        return self.pair_timesteps[:, 0]

    @property
    def audio_timesteps(self) -> torch.Tensor:
        if self.pair_timesteps is None:
            raise RuntimeError("Scheduler not initialized")
        return self.pair_timesteps[:, 1]

    def set_timesteps(self, *args, **kwargs) -> None:
        super().set_timesteps(*args, **kwargs)
        self._refresh_pair_cache()

    def timestep_to_sigma(self, timestep: torch.Tensor | float) -> torch.Tensor:
        """Return the sigma for a given timestep via nearest-neighbor lookup."""
        t_value = float(timestep)

        # Prefer the active paired schedule so visual/audio-specific shift overrides
        # map back to the sigma sequence actually used for denoising.
        schedule_timesteps = self.pair_timesteps
        schedule_sigmas = self.pair_sigmas

        if schedule_timesteps is not None and schedule_sigmas is not None:
            flat_timesteps = schedule_timesteps.reshape(-1)
            flat_sigmas = schedule_sigmas.reshape(-1)
            t_tensor = flat_timesteps.new_tensor(t_value)
            idx = torch.argmin((flat_timesteps - t_tensor).abs())
            return flat_sigmas[idx]

        if self.timesteps is not None and self.sigmas is not None:
            t_tensor = self.timesteps.new_tensor(t_value)
            idx = torch.argmin((self.timesteps - t_tensor).abs())
            return self.sigmas[idx]

        if self.train_timesteps is None or self.train_sigmas is None:
            raise RuntimeError("Scheduler has no cached timesteps/sigmas")

        t_tensor = self.train_timesteps.new_tensor(t_value)
        idx = torch.argmin((self.train_timesteps - t_tensor).abs())
        return self.train_sigmas[idx]

    def step_from_to(
        self,
        model_output: torch.Tensor,
        timestep_from: torch.Tensor | float,
        timestep_to: torch.Tensor | float | None,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Advance one denoising step using explicit (from, to) timestep pair:
            x_to = x_from + model_output * (sigma(to) - sigma(from))
        """
        sigma_from = self.timestep_to_sigma(timestep_from)
        if timestep_to is None:
            sigma_to = torch.tensor(
                1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0,
                device=sigma_from.device,
                dtype=sigma_from.dtype,
            )
        else:
            sigma_to = self.timestep_to_sigma(timestep_to)
        return sample + model_output * (sigma_to - sigma_from)

    def _refresh_pair_cache(self) -> None:
        """Rebuild pair caches from current timesteps/sigmas."""
        if self.timesteps is None or self.sigmas is None:
            return

        def _apply_postprocess(pairs: torch.Tensor, source: str) -> torch.Tensor:
            if self._pair_postprocess_fn is None:
                return pairs
            if self._pair_postprocess_requires_source:
                modified = self._pair_postprocess_fn(pairs, source=source)
            else:
                modified = self._pair_postprocess_fn(pairs)
            if not isinstance(modified, torch.Tensor):
                raise TypeError("pair_postprocess return value must be a torch.Tensor")
            if modified.shape != pairs.shape:
                raise ValueError("pair_postprocess return tensor shape must match input")
            return modified

        base_pairs_timesteps = self._make_pairs_from_vector(self.timesteps)
        base_pairs_sigmas = self._make_pairs_from_vector(self.sigmas)

        self.pair_timesteps = _apply_postprocess(base_pairs_timesteps, "timesteps")
        self.pair_sigmas = _apply_postprocess(base_pairs_sigmas, "sigmas")
