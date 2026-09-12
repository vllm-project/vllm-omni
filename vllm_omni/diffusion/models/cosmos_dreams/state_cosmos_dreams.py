# SPDX-License-Identifier: Apache-2.0
"""Non-KV per-session state and fail-closed request fingerprinting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


def append_dense_kv_history(
    history: list[tuple[torch.Tensor, torch.Tensor]] | None,
    current_kv: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    tokens_per_frame: int,
    sink_frames: int,
    window_frames: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Append a dense oracle block while retaining only one rebuilt layer."""

    if isinstance(tokens_per_frame, bool) or not isinstance(tokens_per_frame, int) or tokens_per_frame <= 0:
        raise ValueError("tokens_per_frame must be positive.")
    if (
        isinstance(sink_frames, bool)
        or not isinstance(sink_frames, int)
        or sink_frames < 0
        or isinstance(window_frames, bool)
        or not isinstance(window_frames, int)
        or window_frames <= 0
    ):
        raise ValueError("sink_frames must be non-negative and window_frames must be positive.")
    sink_tokens = sink_frames * tokens_per_frame
    tail_tokens = window_frames * tokens_per_frame
    max_tokens = sink_tokens + tail_tokens
    if not current_kv:
        raise ValueError("Cosmos-Dreams dense K/V update must contain at least one layer.")

    def validate_pair(key: torch.Tensor, value: torch.Tensor, *, label: str) -> None:
        if key.ndim < 2 or key.shape != value.shape:
            raise ValueError(
                f"Cosmos-Dreams {label} K/V must have matching rank-2+ shapes, "
                f"got {tuple(key.shape)} and {tuple(value.shape)}."
            )
        if key.device != value.device or key.dtype != value.dtype:
            raise ValueError(f"Cosmos-Dreams {label} K/V must have matching dtypes and devices.")
        if key.shape[1] <= 0 or key.shape[1] % tokens_per_frame:
            raise ValueError(
                f"Cosmos-Dreams {label} K/V token length must be a positive multiple "
                f"of tokens_per_frame={tokens_per_frame}, got {key.shape[1]}."
            )

    for layer_idx, (new_k, new_v) in enumerate(current_kv):
        validate_pair(new_k, new_v, label=f"current layer {layer_idx}")

    if history is None:
        # The transformer output already owns exactly the first committed
        # block. Retain its detached storage directly instead of copying it
        # through a concatenation with zero-length views.
        if any(key.shape[1] > max_tokens for key, _ in current_kv):
            raise ValueError("The initial Cosmos-Dreams dense K/V block exceeds the configured history window.")
        return [(key.detach(), value.detach()) for key, value in current_kv]
    if len(history) != len(current_kv):
        raise ValueError(
            "Cosmos-Dreams dense K/V layer count changed within a session: "
            f"history={len(history)}, current={len(current_kv)}."
        )

    # Validate every layer before mutating the list. Once this succeeds,
    # replacing each entry immediately releases that layer's previous history
    # instead of retaining complete old and next-history lists.
    for layer_idx, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(history, current_kv, strict=True)):
        validate_pair(old_k, old_v, label=f"history layer {layer_idx}")
        if (
            old_k.device != new_k.device
            or old_k.dtype != new_k.dtype
            or old_k.shape[:1] + old_k.shape[2:] != new_k.shape[:1] + new_k.shape[2:]
        ):
            raise ValueError(
                f"Cosmos-Dreams dense K/V layer {layer_idx} geometry, dtype, or device changed within a session."
            )

    def combined_prefix(old: torch.Tensor, new: torch.Tensor, count: int) -> list[torch.Tensor]:
        old_count = min(count, old.shape[1])
        parts = [old[:, :old_count]] if old_count else []
        remaining = count - old_count
        if remaining:
            parts.append(new[:, :remaining])
        return parts

    def combined_suffix(old: torch.Tensor, new: torch.Tensor, count: int) -> list[torch.Tensor]:
        new_count = min(count, new.shape[1])
        old_count = count - new_count
        parts = [old[:, -old_count:]] if old_count else []
        if new_count:
            parts.append(new[:, -new_count:])
        return parts

    def append_bounded(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old.shape[1] + new.shape[1] <= max_tokens:
            return torch.cat([old, new], dim=1)
        # Build the final sink+tail tensor directly. Appending the full history
        # and trimming it afterward creates another layer-sized transient at
        # the window boundary.
        parts = combined_prefix(old, new, sink_tokens)
        parts.extend(combined_suffix(old, new, tail_tokens))
        return parts[0].clone() if len(parts) == 1 else torch.cat(parts, dim=1)

    for layer_idx, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(history, current_kv, strict=True)):
        key = append_bounded(old_k, new_k)
        value = append_bounded(old_v, new_v)
        history[layer_idx] = (key.detach(), value.detach())
    return history


@dataclass(frozen=True)
class CosmosDreamsSessionFingerprint:
    prompt_hash: str
    real_text_kv_lengths: tuple[tuple[str, int], ...]
    height: int
    width: int
    fps: float
    domain_id: int
    embodiment: str
    action_contract_sha256: str
    checkpoint_id: str
    manifest_id: str
    sampler_id: str

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            raise ValueError("Cosmos-Dreams session fingerprint requires prompt_hash")
        if self.height <= 0 or self.width <= 0 or self.fps <= 0:
            raise ValueError(
                f"Cosmos-Dreams fingerprint resolution/FPS must be positive, got {self.height}x{self.width}@{self.fps}"
            )
        if self.domain_id < 0:
            raise ValueError(f"Cosmos-Dreams domain_id must be non-negative, got {self.domain_id}")
        if not self.embodiment:
            raise ValueError("Cosmos-Dreams session fingerprint requires an embodiment")
        if not self.action_contract_sha256:
            raise ValueError("Cosmos-Dreams session fingerprint requires action_contract_sha256")
        if not self.real_text_kv_lengths:
            raise ValueError("Cosmos-Dreams fingerprint requires at least one text KV branch")
        if any(length <= 0 for _, length in self.real_text_kv_lengths):
            raise ValueError(f"Cosmos-Dreams real text KV lengths must be positive, got {self.real_text_kv_lengths}")

    def text_length(self, branch: str) -> int:
        try:
            return dict(self.real_text_kv_lengths)[branch]
        except KeyError as exc:
            raise KeyError(f"Cosmos-Dreams fingerprint has no text length for branch {branch!r}") from exc


@dataclass
class CosmosDreamsSessionState:
    session_id: str
    fingerprint: CosmosDreamsSessionFingerprint | None = None
    next_frame_idx: int = 0
    terminal: bool = False
    tick_output_type: str | None = None
    text_kv_by_branch: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    dense_kv_by_branch: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    latents: list[torch.Tensor] = field(default_factory=list)
    vae_decoder_feat_cache: list[Any] | None = None
    vae_decoder_initialized: bool = False
    last_vae_decode_input_frames: int = 0
    max_vae_decode_input_frames: int = 0

    def initialize(
        self,
        fingerprint: CosmosDreamsSessionFingerprint,
        *,
        next_frame_idx: int = 0,
    ) -> None:
        if self.fingerprint is not None:
            raise RuntimeError(
                f"Cosmos-Dreams session {self.session_id!r} is already initialized; session reset required"
            )
        if next_frame_idx < 0:
            raise ValueError(f"Cosmos-Dreams next_frame_idx must be non-negative, got {next_frame_idx}")
        self.fingerprint = fingerprint
        self.next_frame_idx = int(next_frame_idx)

    def validate_request(
        self,
        fingerprint: CosmosDreamsSessionFingerprint,
        *,
        frame_idx: int,
    ) -> None:
        if self.fingerprint is None:
            raise RuntimeError(f"Cosmos-Dreams session {self.session_id!r} is not initialized; session reset required")
        if self.terminal:
            raise ValueError(
                f"Cosmos-Dreams session {self.session_id!r} already completed a full rollout; session reset required"
            )
        if fingerprint != self.fingerprint:
            changed = [
                field_name
                for field_name in self.fingerprint.__dataclass_fields__
                if getattr(self.fingerprint, field_name) != getattr(fingerprint, field_name)
            ]
            raise ValueError(
                f"Cosmos-Dreams session conditioning changed ({', '.join(changed)}); session reset required"
            )
        if frame_idx != self.next_frame_idx:
            raise ValueError(
                "Cosmos-Dreams request is out of order: "
                f"expected latent frame {self.next_frame_idx}, got {frame_idx}; session reset required"
            )

    def append_chunk(
        self,
        chunk: torch.Tensor,
        *,
        frame_start: int,
        retain_latent: bool = True,
    ) -> None:
        if chunk.ndim != 5 or chunk.shape[0] != 1:
            raise ValueError(f"Cosmos-Dreams session chunks must have shape [1,C,T,H,W], got {tuple(chunk.shape)}")
        if frame_start != self.next_frame_idx:
            raise ValueError(
                "Cosmos-Dreams cannot append an out-of-order chunk: "
                f"expected {self.next_frame_idx}, got {frame_start}; session reset required"
            )
        if retain_latent:
            self.latents.append(chunk.detach())
        self.next_frame_idx += int(chunk.shape[2])

    def record_incremental_decode(
        self,
        *,
        input_frames: int,
        feature_cache: list[Any],
    ) -> None:
        if input_frames <= 0:
            raise ValueError(f"Cosmos-Dreams decode input_frames must be positive, got {input_frames}")
        self.vae_decoder_feat_cache = feature_cache
        self.vae_decoder_initialized = True
        self.last_vae_decode_input_frames = int(input_frames)
        self.max_vae_decode_input_frames = max(self.max_vae_decode_input_frames, int(input_frames))

    def reset(self) -> None:
        self.fingerprint = None
        self.next_frame_idx = 0
        self.terminal = False
        self.tick_output_type = None
        self.text_kv_by_branch.clear()
        self.dense_kv_by_branch.clear()
        self.latents.clear()
        self.vae_decoder_feat_cache = None
        self.vae_decoder_initialized = False
        self.last_vae_decode_input_frames = 0
        self.max_vae_decode_input_frames = 0

    @property
    def accumulated_latents(self) -> torch.Tensor | None:
        if not self.latents:
            return None
        return torch.cat(self.latents, dim=2)
