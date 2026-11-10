# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch


class LTXFourStepSampler:
    def __init__(self):
        shift = 5.0

        def warp(value):
            return shift * value / (1.0 + (shift - 1.0) * value)

        ladder = warp(torch.linspace(1.0, warp(0.001), 4, dtype=torch.float64, device="cpu"))
        self.sigmas = (*ladder.float().tolist(), 0.0)
        self.timesteps = tuple(1000.0 * sigma for sigma in self.sigmas[:-1])

    @staticmethod
    def generator(device, seed, chunk, step):
        return torch.Generator(device=device).manual_seed(seed * 1_000_003 + chunk * 4096 + step + 1)

    def initial_noise(self, shape, *, device, seed, chunk):
        return torch.randn(
            shape,
            generator=self.generator(device, seed, chunk, -1),
            device=device,
            dtype=torch.float32,
        )

    def advance(self, sample, prediction, *, step, seed, chunk):
        sigma, following = self.sigmas[step : step + 2]
        clean = sample.float() - sigma * prediction.float()
        if following == 0.0:
            return clean
        noise = torch.randn(
            sample.shape,
            generator=self.generator(sample.device, seed, chunk, step),
            device=sample.device,
            dtype=torch.float32,
        )
        return (1.0 - following) * clean + following * noise
