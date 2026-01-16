# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/hao-ai-lab/FastVideo
# Originally from https://github.com/huggingface/diffusers
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Base scheduler class for diffusion models."""

from abc import ABC, abstractmethod

import torch


class BaseScheduler(ABC):
    """
    Abstract base class for schedulers.

    Subclasses must define:
        - timesteps: torch.Tensor
        - order: int
        - num_train_timesteps: int
    """

    timesteps: torch.Tensor
    order: int
    num_train_timesteps: int

    def __init__(self):
        required_attrs = ["timesteps", "order", "num_train_timesteps"]
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Subclass {self.__class__.__name__} must define `{attr}` before calling super().__init__()"
                )

    @abstractmethod
    def set_shift(self, shift: float) -> None:
        """Set the shift parameter for the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def set_timesteps(self, *args, **kwargs) -> None:
        """Set the timesteps for the scheduler."""
        raise NotImplementedError

    @abstractmethod
    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor:
        """Scale the model input."""
        raise NotImplementedError
