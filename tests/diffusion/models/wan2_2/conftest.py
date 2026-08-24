from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn


class StubTransformer(nn.Module):
    def __init__(self, *, name: str = "transformer", in_channels: int = 4, out_channels: int = 4) -> None:
        super().__init__()
        self.name = name
        self.config = SimpleNamespace(
            patch_size=(1, 2, 2),
            in_channels=in_channels,
            out_channels=out_channels,
            image_dim=None,
        )

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def forward(self, **kwargs):
        hidden_states = kwargs["hidden_states"]
        return (torch.zeros_like(hidden_states[:, : self.config.out_channels]),)


class StubScheduler:
    def __init__(self, timesteps: list[int]) -> None:
        self.timesteps = torch.tensor(timesteps, dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.set_timesteps_calls: list[tuple[int, torch.device]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.set_timesteps_calls.append((num_steps, device))


class StubVAE:
    dtype = torch.float32

    def __init__(self, z_dim: int = 4) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=4,
            scale_factor_spatial=8,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        latent_frames = (video.shape[2] + self.config.scale_factor_temporal - 1) // self.config.scale_factor_temporal
        latent_height = video.shape[-2] // self.config.scale_factor_spatial
        latent_width = video.shape[-1] // self.config.scale_factor_spatial
        latents = torch.ones(
            video.shape[0],
            self.config.z_dim,
            latent_frames,
            latent_height,
            latent_width,
            dtype=video.dtype,
            device=video.device,
        )
        return SimpleNamespace(latents=latents)

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        del return_dict
        return (latents,)


@contextmanager
def noop_progress_bar(*args, **kwargs):
    del args, kwargs

    class Bar:
        def update(self) -> None:
            return None

    yield Bar()
