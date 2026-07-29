# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""
ActionModule for AMD Micro-World models.

Processes keyboard (7-dim: W/A/S/D/Space/Shift/Ctrl) and mouse (2-dim) action
inputs through sliding-window grouped MLPs to produce per-frame action features.

Ported from: https://github.com/AMD-AGI/Micro-World
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Compute 1D sinusoidal positional embeddings.

    Args:
        dim: Embedding dimension.
        position: Position indices tensor.

    Returns:
        Sinusoidal embeddings of shape ``[len(position), dim]``.
    """
    half = dim // 2
    sinusoid = torch.outer(
        position.float(),
        torch.pow(10000, -torch.arange(half, device=position.device).float() / half),
    )
    return torch.cat([sinusoid.cos(), sinusoid.sin()], dim=-1)


class ActionModule(nn.Module):
    """Processes mouse and keyboard action inputs for Micro-World models.

    Mouse actions are grouped using a sliding window and projected via MLP.
    Keyboard actions are embedded, given sinusoidal positional encoding,
    grouped with the same sliding window, and projected via MLP.
    The two streams are concatenated and fused through a final MLP.

    Args:
        mouse_dim: Dimensionality of mouse input (default 2: dx, dy).
        keyboard_dim: Dimensionality of keyboard input (default 7: W/A/S/D/Space/Shift/Ctrl).
        action_dim: Output action feature dimension (default 1536).
        window_size: Number of latent frames in the sliding window context.
        temporal_ratio: Raw-frame-to-latent-frame ratio (e.g. 4 raw frames per latent).
        flatten_spatial: If True, repeat action features over H*W and flatten to
            ``[B, T*H*W, action_dim]`` (for ControlNet T2W variant).
            If False, return ``[B, T, action_dim]`` (for AdaLN I2W variant).
    """

    def __init__(
        self,
        mouse_dim: int = 2,
        keyboard_dim: int = 7,
        action_dim: int = 1536,
        window_size: int = 3,
        temporal_ratio: int = 4,
        flatten_spatial: bool = True,
    ):
        super().__init__()
        self.mouse_dim = mouse_dim
        self.keyboard_dim = keyboard_dim
        self.action_dim = action_dim
        self.window_size = window_size
        self.ratio = temporal_ratio
        self.flatten_spatial = flatten_spatial

        # Mouse stream: sliding-window grouped raw values -> MLP -> action_dim//2
        self.mouse_mlp = nn.Sequential(
            nn.Linear(self.ratio * window_size * mouse_dim, action_dim // 2),
            nn.GELU(),
            nn.Linear(action_dim // 2, action_dim // 2),
        )

        # Keyboard stream: embed -> sinusoidal pos enc -> sliding-window group -> MLP
        keyboard_embedding_dim = 64
        self.keyboard_embedding = nn.Sequential(
            nn.Linear(keyboard_dim, keyboard_embedding_dim),
            nn.GELU(),
            nn.Linear(keyboard_embedding_dim, keyboard_embedding_dim),
        )
        self.keyboard_mlp = nn.Sequential(
            nn.Linear(self.ratio * window_size * keyboard_embedding_dim, action_dim // 2),
            nn.GELU(),
            nn.Linear(action_dim // 2, action_dim // 2),
        )

        # Fusion MLP: concat(mouse, keyboard) -> action_dim
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_dim // 2),
            nn.GELU(),
            nn.Linear(action_dim // 2, action_dim),
        )

    def _group_with_sliding_window(
        self,
        features: torch.Tensor,
        t: int,
        feature_dim: int,
    ) -> torch.Tensor:
        """Group temporal features using a sliding window.

        For each latent frame index ``i``, gathers the surrounding raw-frame
        features within ``[max(1, ratio*(i-window_size)+1), i*ratio+1)`` and
        pads to a fixed length of ``ratio * window_size``.

        Args:
            features: ``[B, rn, D]`` raw-frame-level features.
            t: Number of latent frames (post-patch temporal dim).
            feature_dim: Last dimension size (used for flatten).

        Returns:
            Grouped features of shape ``[B, t, ratio*window_size*D]``.
        """
        batch_size = features.shape[0]
        target_len = self.ratio * self.window_size

        # Frame 0: repeat the first feature to fill the window
        p0 = features[:, :1].repeat(1, target_len, 1).reshape(batch_size, -1)
        grouped = [p0]

        for i in range(1, t):
            start_idx = max(1, self.ratio * (i - self.window_size) + 1)
            end_idx = i * self.ratio + 1
            window = features[:, start_idx:end_idx]

            pad_len = target_len - window.shape[1]
            if pad_len > 0:
                window = F.pad(window, (0, 0, pad_len, 0), mode="replicate")
            grouped.append(window.reshape(batch_size, -1))

        return torch.stack(grouped, dim=1)  # [B, t, target_len * D]

    def forward(
        self,
        mouse_actions: torch.Tensor,
        keyboard_actions: torch.Tensor,
        grid_sizes: tuple[int, int, int],
    ) -> torch.Tensor:
        """Compute action features from mouse and keyboard inputs.

        Args:
            mouse_actions: ``[B, rn, mouse_dim]`` raw mouse movement per raw frame.
            keyboard_actions: ``[B, rn, keyboard_dim]`` binary key states per raw frame.
            grid_sizes: ``(t, h, w)`` post-patch-embedding spatial dimensions.

        Returns:
            Action features. Shape depends on ``flatten_spatial``:
            - True: ``[B, t*h*w, action_dim]``
            - False: ``[B, t, action_dim]``
        """
        t, h, w = grid_sizes

        # --- Mouse stream ---
        grouped_mouse = self._group_with_sliding_window(mouse_actions, t, self.mouse_dim)
        # [B, t, ratio*window_size*mouse_dim]
        if self.flatten_spatial:
            grouped_mouse = grouped_mouse.unsqueeze(2).repeat(1, 1, h * w, 1)
            # [B, t, h*w, ratio*window_size*mouse_dim]
        mouse_features = self.mouse_mlp(grouped_mouse)  # [..., action_dim//2]

        # --- Keyboard stream ---
        keyboard_emb = self.keyboard_embedding(keyboard_actions)  # [B, rn, 64]

        # Add sinusoidal positional encoding
        positions = torch.arange(keyboard_emb.size(1), device=keyboard_emb.device)
        pos_enc = sinusoidal_embedding_1d(keyboard_emb.shape[-1], positions).to(dtype=keyboard_emb.dtype)
        pos_enc = pos_enc.unsqueeze(0).expand(keyboard_emb.size(0), -1, -1)
        keyboard_emb = keyboard_emb + pos_enc

        grouped_keyboard = self._group_with_sliding_window(keyboard_emb, t, keyboard_emb.shape[-1])
        # [B, t, ratio*window_size*64]
        if self.flatten_spatial:
            grouped_keyboard = grouped_keyboard.unsqueeze(2).repeat(1, 1, h * w, 1)
        keyboard_features = self.keyboard_mlp(grouped_keyboard)  # [..., action_dim//2]

        # --- Fusion ---
        action_features = torch.cat([mouse_features, keyboard_features], dim=-1)
        action_features = self.action_mlp(action_features)  # [..., action_dim]

        if self.flatten_spatial:
            action_features = action_features.flatten(1, 2)  # [B, t*h*w, action_dim]

        return action_features
