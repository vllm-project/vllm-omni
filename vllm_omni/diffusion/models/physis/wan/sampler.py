# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WanSamplerProfile:
    name: str
    sigmas: tuple[float, float, float, float, float]
    timesteps: tuple[float, float, float, float]


WAN21_1_3B = WanSamplerProfile(
    name="wan21_1_3b",
    sigmas=(1.0, 0.750257670879364, 0.499718636274338, 0.250000029802322, 0.0),
    timesteps=(1000.0, 750.0, 500.0, 250.0),
)

WAN21_14B = WanSamplerProfile(
    name="wan21_14b",
    sigmas=(0.700232386589050, 0.499718636274338, 0.400259941816330, 0.198347106575966, 0.0),
    timesteps=(700.0, 500.0, 400.0, 200.0),
)

WAN22_5B = WanSamplerProfile(
    name="wan22_5b",
    sigmas=(1.0, 0.9097071290016174, 0.7173174619674683, 0.0244140625, 0.0),
    timesteps=(1000.0, 909.7071290016174, 717.3174619674683, 24.4140625),
)


def mapped_seed(seed: int, chunk: int, step: int) -> int:
    return seed * 1_000_003 + chunk * 4096 + step + 1


def make_generator(device: torch.device | str, seed: int, chunk: int, step: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(mapped_seed(seed, chunk, step))
    return generator


@dataclass(frozen=True)
class WanFourStepSampler:
    profile: WanSamplerProfile

    def __post_init__(self) -> None:
        if len(self.profile.sigmas) != 5 or len(self.profile.timesteps) != 4:
            raise ValueError("Wan Physis sampling requires four timesteps and five sigma levels")
        if self.profile.sigmas[-1] != 0.0:
            raise ValueError("The final Wan sigma must be zero")

    @property
    def timesteps(self):
        return self.profile.timesteps

    def initial_noise(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        seed: int,
        chunk: int,
    ) -> torch.Tensor:
        generator = make_generator(device, seed, chunk, -1)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def advance(
        self,
        sample: torch.Tensor,
        prediction: torch.Tensor,
        *,
        step: int,
        seed: int,
        chunk: int,
    ) -> torch.Tensor:
        if not 0 <= step < 4:
            raise IndexError(f"step must be in [0, 3], got {step}")

        sigma = self.profile.sigmas[step]
        following_sigma = self.profile.sigmas[step + 1]
        clean = (sample.float() - sigma * prediction.float()).to(sample.dtype)
        if following_sigma == 0.0:
            return clean

        generator = make_generator(sample.device, seed, chunk, step)
        noise = torch.randn(
            sample.shape,
            generator=generator,
            device=sample.device,
            dtype=torch.float32,
        )
        return ((1.0 - following_sigma) * clean.float() + following_sigma * noise).to(sample.dtype)

    def sink_input(
        self,
        clean_latent: torch.Tensor,
        *,
        sink_sigma: float,
        seed: int,
        chunk: int = 0,
    ) -> torch.Tensor:
        if sink_sigma == 0.0:
            return clean_latent

        generator = make_generator(clean_latent.device, seed, chunk, 4)
        noise = torch.randn(
            clean_latent.shape,
            generator=generator,
            device=clean_latent.device,
            dtype=torch.float32,
        )
        return ((1.0 - sink_sigma) * clean_latent.float() + sink_sigma * noise).to(clean_latent.dtype)
