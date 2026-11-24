# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import _axis_from_sqrt_area, _video_t_grid, _video_t_span
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.models.taomate_h3.attention import (
    StreamContext,
    StreamMode,
    TaoMateH3StreamingAttention,
    _dense_attention,
    stream_context,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState, SegmentHandle

from .schedule import Window, WindowPlanner


@dataclass
class _CausalContext(StreamContext):
    read_history: bool = True


class InvokerRenderAttention(TaoMateH3StreamingAttention):
    def _streaming_attention(self, q, k, v, context: _CausalContext):
        if q.shape[0] != context.seq_len or context.media_span is None or self.layer_index is None:
            raise RuntimeError("Invalid Invoker-Render packed attention context")
        _, used = context.media_span
        keys, values = context.cache.assemble(self.layer_index, k[:0], v[:0], k[:used], v[:used])
        if not context.read_history:
            keys, values = keys[-used:], values[-used:]
        output = torch.zeros_like(q)
        output[:used] = _dense_attention(
            q[:used], keys, values, scale=float(self.softmax_scale), flash_attn_func=self._flash_attn_func
        )
        if context.mode is StreamMode.CLEAN_COMMIT:
            context.cache.stage(self.layer_index)
        return output


def generator(device: torch.device | str, *parts: int | str) -> torch.Generator:
    digest = hashlib.sha256("_".join(str(part) for part in parts).encode()).hexdigest()
    return torch.Generator(device=device).manual_seed(int(digest, 16) % (2**63))


def euler_step(
    state: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor, next_sigma: torch.Tensor
) -> torch.Tensor:
    # Compute x0 from the BF16-rounded input and keep the FP32 state for the residual.
    time = 1.0 - sigma.float()
    x0 = state.to(torch.bfloat16).float() + (1.0 - time) * velocity
    current = sigma.reshape((1,) * state.ndim)
    following = next_sigma.reshape((1,) * state.ndim)
    epsilon = (state - (1.0 - current) * x0) / current
    return ((1.0 - following) * x0 + following * epsilon).to(state.dtype)


@dataclass
class _Frame:
    proxy: torch.Tensor
    proxy_noise: torch.Tensor
    noise: torch.Tensor
    clean_noise: torch.Tensor
    clean: torch.Tensor | None = None


class CausalSession:
    def __init__(
        self,
        model: MiniMaxH3DiTModel,
        *,
        features: torch.Tensor,
        tags: torch.Tensor,
        image: torch.Tensor,
        seed: int,
        ulysses: int,
    ) -> None:
        self.model = model
        self.device = image.device
        self.features = features.to(self.device)
        self.tags = tags.to(self.device)
        self.image = image
        self.seed = seed
        self.ulysses = ulysses
        self.image_noise = torch.randn(image.shape, generator=generator("cpu", seed, "first_frame", "static")).to(
            self.device
        )
        sigmas = torch.arange(1, 0, -0.25, dtype=torch.float64, device=self.device)
        self.sigmas = torch.cat((12 * sigmas / (1 + 11 * sigmas), sigmas.new_zeros(1)))
        height, width = image.shape[-2:]
        area = (height * width) ** 0.5
        self.spatial = torch.stack(
            torch.meshgrid(_axis_from_sqrt_area(height, 2, area), _axis_from_sqrt_area(width, 2, area), indexing="ij"),
            dim=-1,
        ).reshape(-1, 2)
        self.planner = WindowPlanner()
        self.frames: dict[int, _Frame] = {}
        self.handles: list[tuple[int | None, SegmentHandle]] = []
        self.shift = 0.0
        self.source_clip = 0

    def append(self, proxy: torch.Tensor, *, eos: bool) -> tuple[Window, ...]:
        expected = 2 if eos else 5
        if proxy.shape != (self.image.shape[0], expected, *self.image.shape[-2:]):
            raise ValueError(f"Unexpected encoded proxy shape: {tuple(proxy.shape)}")
        start = self.planner.available
        plans = self.planner.append(eos=eos)
        proxy_noise = torch.randn(proxy.shape, generator=generator("cpu", self.seed, "proxy", self.source_clip)).to(
            self.device
        )
        rng = generator(self.device, self.seed, "cwm_rgb", 0, self.source_clip)
        noise = torch.randn(proxy.shape, device=self.device, generator=rng)
        clean_noise = torch.randn(proxy.shape, device=self.device, generator=rng)
        for index in range(expected):
            self.frames[start + index] = _Frame(
                *(value[:, index : index + 1] for value in (proxy, proxy_noise, noise, clean_noise))
            )
        self.source_clip += 1
        return plans

    def prepare(self, cache: ARDiffusionSegmentedKVState, window: Window) -> torch.Tensor:
        if window.start:
            shift = window.rope_shift_blocks * _video_t_span(5)
            if not self.handles or shift != self.shift:
                self.shift = shift
                self._forward(cache, (0,), static=True, commit=True, prefix=True)
            if window.start == 9:
                self._forward(cache, tuple(range(window.recent_start, window.start)), commit=True)
            recent = [
                (index, handle) for index, handle in self.handles if index is None or index >= window.recent_start
            ]
            cache.retain([handle for _, handle in recent])
            self.handles = recent
            self._forward(cache, tuple(range(window.start, window.stop)), video=False, commit=True)
        return torch.cat([self.frames[index].noise for index in range(window.start, window.stop)], dim=1)

    def predict(
        self, cache: ARDiffusionSegmentedKVState, window: Window, latents: torch.Tensor, step: int
    ) -> torch.Tensor:
        logits = self._forward(
            cache,
            tuple(range(window.start, window.stop)),
            static=window.start == 0,
            proxies=window.start == 0,
            latents=latents,
            sigma=self.sigmas[step],
        )
        channels, frames, height, width = latents.shape
        return minimax_h3_unpatchify_video_tokens(
            logits, latent_shape=(frames, height // 2, width // 2, channels), patch_size=(1, 2, 2)
        )[0]

    def commit(self, cache: ARDiffusionSegmentedKVState, window: Window, latents: torch.Tensor) -> None:
        if not bool(torch.isfinite(latents).all()):
            raise RuntimeError("Invoker-Render generated nonfinite latents")
        positions = tuple(range(window.start, window.stop))
        for i, index in enumerate(positions):
            # Detach a contiguous single-latent copy so retained history does not pin the whole window tensor.
            self.frames[index].clean = latents[:, i : i + 1].detach().clone()
        if window.start and not window.eos:
            start = max(1, window.stop - 12)
            self._forward(cache, positions, proxies=False, commit=True, retain_start=start)
        start = max(1, window.stop - 12)
        self.frames = {index: frame for index, frame in self.frames.items() if index == 0 or index >= start}

    def _forward(
        self,
        cache: ARDiffusionSegmentedKVState,
        positions: tuple[int, ...],
        *,
        static: bool = False,
        proxies: bool = True,
        video: bool = True,
        commit: bool = False,
        prefix: bool = False,
        retain_start: int | None = None,
        latents: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows, tags, times, rope = [], [], [], []
        image_rows, text_rows, output_rows = [], [], []
        segments: list[tuple[int | None, int]] = []
        text_len = len(self.features)
        cursor = 0

        def indices(count: int) -> torch.Tensor:
            return torch.arange(cursor, cursor + count, device=self.device)

        def add(value: torch.Tensor, noise: torch.Tensor, position: int, origin: float, *, target: bool = False):
            nonlocal cursor
            value = minimax_h3_patchify_video_latent(value[None], patch_size=(1, 2, 2)).to(torch.bfloat16)
            count = value.shape[0]
            time = 1.0 - sigma.float() if target else value.new_tensor(0.999, dtype=torch.float32)
            if target:
                rows.append(value.float())
                output_rows.append(indices(count))
            else:
                epsilon = minimax_h3_patchify_video_latent(noise[None], patch_size=(1, 2, 2)).to(torch.bfloat16)
                rows.append(time.expand(count, 1) * value + (1 - time.expand(count, 1)) * epsilon)
            times.append(time.expand(count))
            tags.append(torch.zeros(count, dtype=torch.long, device=self.device))
            time_pos = origin + position // 5 * _video_t_span(5) + _video_t_grid(5, 0.0)[position % 5]
            rope.append(torch.cat((self.spatial.new_full((count, 1), time_pos), self.spatial), dim=1))
            image_rows.append(indices(count))
            cursor += count

        if static:
            rows.append(torch.zeros(text_len, self.image.shape[0] * 4, device=self.device))
            times.append(torch.full((text_len,), 0.999, device=self.device))
            tags.append(self.tags)
            text_rows.append(indices(text_len))
            pos = torch.zeros(text_len, 3, dtype=torch.float64)
            pos[:, 0] = torch.arange(text_len, dtype=torch.float64) + self.shift
            rope.append(pos)
            cursor += text_len
            add(self.image, self.image_noise, 0, text_len + self.shift)
        for offset, index in enumerate(positions):
            previous = cursor
            frame = self.frames[index]
            if proxies:
                add(frame.proxy, frame.proxy_noise, index, text_len)
            if video:
                value = frame.clean if latents is None else latents[:, offset : offset + 1]
                if value is None:
                    raise RuntimeError("Missing Invoker-Render clean video history")
                add(value, frame.clean_noise, index, text_len, target=latents is not None)
            if not prefix:
                segments.append((index, cursor - previous))
        if prefix:
            segments.append((None, cursor))
        used = cursor
        padding = self.ulysses - used % self.ulysses if self.ulysses > 1 else 0
        if padding:
            rows.append(rows[0].new_zeros(padding, rows[0].shape[-1]))
            times.append(torch.full((padding,), 0.999, device=self.device))
            tags.append(torch.full((padding,), -1, device=self.device, dtype=torch.long))
            rope.append(torch.zeros(padding, 3, dtype=torch.float64))
        empty = torch.empty(0, device=self.device, dtype=torch.long)
        image_ids = torch.cat(image_rows) if image_rows else empty
        outputs = torch.cat(output_rows) if output_rows else empty
        unique, inverse = torch.unique(torch.cat(times), return_inverse=True)
        offsets = (0, used, used + padding) if padding else (0, used)
        cache.prepare_layer_views(used, 0)
        pending = cache.begin_commit([count for _, count in segments]) if commit else ()
        context = _CausalContext(
            cache,
            StreamMode.CLEAN_COMMIT if commit else StreamMode.NOISY,
            empty,
            image_ids,
            used + padding,
            media_span=(0, used),
            read_history=not prefix,
        )
        try:
            with stream_context(context):
                logits, _ = self.model(
                    x=torch.cat(rows)[None],
                    audio_x=rows[0].new_zeros(1, used + padding, self.model.arch.audio_latents_dim),
                    img_position_ids=torch.cat(rope)[None].to(self.device),
                    unique_timesteps=unique,
                    inverse_indices=inverse,
                    token_tags=torch.cat(tags),
                    prompt_embeds=self.features if static else self.features[:0],
                    img_pos_info={"position_ids": image_ids},
                    text_pos_info={"position_ids": text_rows[0] if text_rows else empty},
                    audio_pos_info={"position_ids": empty},
                    img_pos_for_infer_output_info={"position_ids": outputs},
                    update_mask=torch.ones(outputs.numel(), dtype=torch.bool, device=self.device),
                    packed_seq_params={
                        "cu_seqlens_q": torch.tensor(offsets, dtype=torch.int32, device=self.device),
                        "cu_seqlens_q_host": offsets,
                        "max_seqlen_q": used,
                    },
                    refiner_packed_seq_params={
                        "cu_seqlens_q": torch.tensor(
                            (0, text_len if static else 0), dtype=torch.int32, device=self.device
                        ),
                        "max_seqlen_q": text_len if static else 0,
                    },
                )
            if commit:
                new = [(index, handle) for (index, _), handle in zip(segments, pending, strict=True)]
                if prefix:
                    retained = new + [(index, handle) for index, handle in self.handles if index is not None]
                else:
                    retained = self.handles + new
                if retain_start is not None:
                    retained = [(index, handle) for index, handle in retained if index is None or index >= retain_start]
                cache.commit(retain=[handle for _, handle in retained])
                self.handles = retained
        except BaseException:
            if commit:
                cache.rollback()
            raise
        return logits
