from typing import Protocol

import torch


class SupportsDiffusion(Protocol):

    def get_latents(
        self,
    ) -> torch.Tensor:
        """
        Create latents from mm input.
        """
        ...

    def diffuse(
        self,
    ) -> torch.Tensor:
        """
        Perform a diffusion step.
        """
        ...

    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to mm output.
        """
        ...
