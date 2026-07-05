# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CUDA graph wrapper for the MOSS-TTS-Local local-channel hot path."""

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalGraphKey:
    batch_size: int
    channel: int
    dtype: torch.dtype
    device_index: int


class LocalChannelGraph:
    """Captured graph for one fixed local-channel shape."""

    def __init__(
        self,
        key: LocalGraphKey,
        local_dim: int,
        logits_dim: int,
    ) -> None:
        self.key = key
        device = torch.device("cuda", key.device_index)
        self.ctx = torch.empty(
            key.batch_size,
            key.channel + 1,
            local_dim,
            device=device,
            dtype=key.dtype,
        )
        self.ctx.zero_()
        self.logits = torch.empty(
            key.batch_size,
            logits_dim,
            device=device,
            dtype=key.dtype,
        )
        self.graph = torch.cuda.CUDAGraph()
        self.captured = False
        self.lock = threading.Lock()

    def capture(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        warmups: int,
    ) -> None:
        device = self.ctx.device
        current_stream = torch.cuda.current_stream(device)
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            for _ in range(max(warmups, 0)):
                self.logits.copy_(fn(self.ctx))
        current_stream.wait_stream(warmup_stream)

        with torch.cuda.graph(self.graph):
            self.logits.copy_(fn(self.ctx))
        self.captured = True

    def replay(self, ctx_plus: torch.Tensor) -> torch.Tensor:
        active_batch = int(ctx_plus.shape[0])
        if active_batch > self.key.batch_size:
            raise ValueError(
                f"Expected batch size <= {self.key.batch_size}, got "
                f"{active_batch}."
            )
        with self.lock:
            self.ctx[:active_batch].copy_(ctx_plus)
            if active_batch < self.key.batch_size:
                self.ctx[active_batch:].zero_()
            self.graph.replay()
            return self.logits[:active_batch]


class MossTTSLocalCUDAGraphManager:
    """Lazy graph cache for local transformer + channel head replays.

    The manager intentionally captures only pure tensor work. Sampling, request
    state mutation, and multimodal output construction remain in eager Python.
    """

    def __init__(
        self,
        model: Any,
        batch_sizes: tuple[int, ...],
        warmups: int,
    ) -> None:
        self.model = model
        self.batch_sizes = tuple(sorted(set(batch_sizes)))
        self.warmups = warmups
        self._graphs: dict[LocalGraphKey, LocalChannelGraph] = {}
        self._failed_keys: set[LocalGraphKey] = set()

    def _select_bucket(self, batch_size: int) -> int | None:
        for bucket in self.batch_sizes:
            if batch_size <= bucket:
                return bucket
        return None

    def _device_index(self, device: torch.device) -> int:
        if device.type != "cuda":
            raise RuntimeError("MOSS-TTS local CUDA graph requires CUDA tensors.")
        return torch.cuda.current_device() if device.index is None else device.index

    def _make_key(
        self,
        batch_size: int,
        channel: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> LocalGraphKey:
        return LocalGraphKey(
            batch_size=batch_size,
            channel=channel,
            dtype=dtype,
            device_index=self._device_index(device),
        )

    def _logits_from_ctx(self, channel: int, ctx_plus: torch.Tensor) -> torch.Tensor:
        local_out, _ = self.model.local_transformer(ctx_plus)
        last_h = local_out[:, -1, :]
        proj_out = self.model.local_to_speech_embedding_mlps[channel](last_h)
        normed = self.model.layer_norm_before_lm_heads[channel](proj_out)
        return self.model.lm_heads[channel](normed)

    def _capture_graph(
        self,
        key: LocalGraphKey,
        local_dim: int,
        logits_dim: int,
    ) -> LocalChannelGraph | None:
        graph = LocalChannelGraph(
            key=key,
            local_dim=local_dim,
            logits_dim=logits_dim,
        )

        def fn(ctx: torch.Tensor) -> torch.Tensor:
            return self._logits_from_ctx(key.channel, ctx)

        try:
            graph.capture(fn, self.warmups)
        except RuntimeError as exc:
            self._failed_keys.add(key)
            logger.warning(
                "[MossTTS Local] CUDA graph capture failed for %s; "
                "falling back to eager local channel execution. Error: %r",
                key,
                exc,
                exc_info=True,
            )
            return None
        self._graphs[key] = graph
        logger.info("[MossTTS Local] Captured local CUDA graph for %s.", key)
        return graph

    @torch.no_grad()
    def replay_channel(
        self,
        channel: int,
        current_proj: torch.Tensor,
        local_ctx: torch.Tensor,
        logits_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        batch_size = int(current_proj.shape[0])
        bucket = self._select_bucket(batch_size)
        if bucket is None:
            return None
        if current_proj.device.type != "cuda":
            return None
        if torch.cuda.is_current_stream_capturing():
            return None

        local_ctx = torch.cat([local_ctx, current_proj.unsqueeze(1)], dim=1)
        key = self._make_key(
            batch_size=bucket,
            channel=channel,
            dtype=current_proj.dtype,
            device=current_proj.device,
        )
        if key in self._failed_keys:
            return None

        graph = self._graphs.get(key)
        if graph is None:
            graph = self._capture_graph(
                key=key,
                local_dim=int(current_proj.shape[-1]),
                logits_dim=logits_dim,
            )
            if graph is None:
                return None

        return graph.replay(local_ctx), local_ctx
