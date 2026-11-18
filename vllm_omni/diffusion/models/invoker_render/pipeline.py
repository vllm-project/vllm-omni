# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import av
import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import Qwen3VLProcessor
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import ReferenceVideoDecodeSpec
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionSegmentedKVSpec,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.model_executor.models.minimax_h3.preprocessing import minimax_h3_multi_image_presentation

from .assets import FPS, PACKET_FRAMES, TAIL_FRAMES, VIDEO_SIZE, InvokerRenderAssets
from .causal import CausalSession, InvokerRenderAttention, euler_step
from .codec import CausalVideoDecoder, DecodeState, encode_rgb


def validate_proxy(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (*size[::-1], 3):
        raise OmniClientError(f"Invoker-Render proxy must have uint8 THWC frames on the {size[0]}x{size[1]} canvas")
    return np.ascontiguousarray(rgb)


def read_proxy_video(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    try:
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1:
                raise OmniClientError("Invoker-Render requires one proxy video stream")
            stream = container.streams.video[0]
            if (stream.width, stream.height) != size or stream.average_rate != FPS:
                raise OmniClientError(f"Invoker-Render requires a {size[0]}x{size[1]} proxy video at 24 FPS")
            frames = []
            start = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise OmniClientError("Invoker-Render source frames must have timestamps")
                pts = frame.pts * frame.time_base
                if start is None:
                    start = pts
                if abs((pts - start) * FPS - len(frames)) > FPS * frame.time_base:
                    raise OmniClientError("Invoker-Render proxy frame timestamps must follow the 24 FPS timeline")
                frames.append(frame.to_ndarray(format="rgb24"))
    except (OSError, av.FFmpegError) as exc:
        raise OmniClientError(f"Could not decode the Invoker-Render proxy: {exc}") from exc
    if not frames:
        raise OmniClientError("Invoker-Render proxy video is empty")
    return validate_proxy(np.stack(frames), size)


def get_invoker_render_pre_process_func(od_config: OmniDiffusionConfig):
    assets = InvokerRenderAssets.read(od_config.model)

    def preprocess(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        text = prompt.get("prompt") if isinstance(prompt, Mapping) else None
        if not isinstance(text, str) or text.strip() != assets.system_prompt:
            raise OmniClientError("Invoker-Render requires the frozen prompt from its artifact")
        data = prompt.get("multi_modal_data") or {}
        image = data.get("image")
        if not isinstance(image, Image.Image) or image.size != VIDEO_SIZE:
            raise OmniClientError("Invoker-Render requires one 1344x768 reference image")
        proxy = data.get("video")
        if isinstance(proxy, list) and len(proxy) == 1:
            proxy = proxy[0]
        if isinstance(proxy, (str, Path)):
            proxy = read_proxy_video(proxy, assets.proxy_size)
        else:
            proxy = validate_proxy(proxy, assets.proxy_size)
        count = len(proxy)
        if count < 2 * PACKET_FRAMES + TAIL_FRAMES or count % PACKET_FRAMES != TAIL_FRAMES:
            raise OmniClientError("Invoker-Render requires 17*N+5 proxy frames and at least 39 frames")
        sampling = copy(request.sampling_params)
        extra = sampling.extra_args or {}
        if (
            sampling.width not in (None, VIDEO_SIZE[0])
            or sampling.height not in (None, VIDEO_SIZE[1])
            or sampling.fps not in (None, FPS)
            or sampling.frame_rate not in (None, FPS)
            or sampling.num_frames not in (1, count)
            or sampling.num_inference_steps not in (None, 4)
            or sampling.guidance_scale not in (None, 1.0)
            or sampling.guidance_scale_2 not in (None, 1.0)
            or sampling.true_cfg_scale not in (None, 1.0)
            or sampling.num_outputs_per_prompt != 1
            or sampling.lora_request is not None
            or sampling.timesteps is not None
            or sampling.sigmas is not None
            or sampling.enable_frame_interpolation
            or extra.get("flow_shift", 12) != 12
            or prompt.get("negative_prompt")
            or set(data) != {"image", "video"}
        ):
            raise OmniClientError("Invoker-Render uses fixed geometry, 24 FPS and four Euler steps with shift 12")
        sampling.width, sampling.height = VIDEO_SIZE
        sampling.fps = sampling.frame_rate = FPS
        sampling.num_frames = count
        sampling.num_inference_steps = 4
        sampling.extra_args = {**extra, "session_id": request.request_id, "reset": True, "close_session": True}
        request.prompt = {**prompt, "multi_modal_data": {"image": image.convert("RGB"), "video": proxy}}
        request.sampling_params = sampling
        return request

    return preprocess


@dataclass
class _Session:
    causal: CausalSession
    decoder: DecodeState
    delivered: int = 0


class InvokerRenderPipeline(nn.Module):
    support_image_input: ClassVar[bool] = True
    dummy_run_num_frames: ClassVar[int] = 0

    @staticmethod
    def reference_video_decode_spec(*, num_frames: int | None, extra_args: dict) -> ReferenceVideoDecodeSpec:
        return ReferenceVideoDecodeSpec()

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.device = get_local_device()
        self.assets = InvokerRenderAssets.read(od_config.model)
        parallel = od_config.parallel_config
        self.ulysses = parallel.ulysses_degree
        if parallel.tensor_parallel_size != 1 or parallel.ring_degree != 1 or od_config.quantization_config is not None:
            raise ValueError("Invoker-Render requires TP1 with pure Ulysses and unquantized weights")
        self.transformer = MiniMaxH3DiTModel(od_config, diffusers_weights=True, attention_cls=InvokerRenderAttention)
        for index, block in enumerate(self.transformer.blocks):
            block.attn.layer_name = f"blocks.{index}.attn"
            block.attn.layer_index = index
        self.processor = Qwen3VLProcessor.from_pretrained(self.assets.processor, local_files_only=True)
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(self.assets.text_encoder, device=self.device, load_model=True)
        self.vae = MiniMaxH3VideoVAE(
            self.assets.video_vae,
            device=self.device,
            encode_only=True,
            trust_remote_code=od_config.trust_remote_code,
        ).to(dtype=torch.bfloat16)
        self.vae.model.parallel_tiling = False
        self.decoder = CausalVideoDecoder(
            self.assets.decoder_config, self.assets.decoder_weights, self.vae, self.device
        )
        sample = self.processor.image_processor(images=[Image.new("RGB", VIDEO_SIZE)], return_tensors="pt")
        self.text_rows = len(self._presentation(sample)[0])
        self._ar_state: ARDiffusionKVState | None = None
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                self.assets.dit_dir, None, None, "transformer.", fall_back_to_pt=False
            ),
            DiffusersPipelineLoader.ComponentSource(
                self.assets.text_encoder, None, None, "text_encoder.", fall_back_to_pt=False
            ),
        ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def _presentation(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        grid = inputs["image_grid_thw"]
        count = int(grid[0].prod()) // self.processor.image_processor.merge_size**2
        return minimax_h3_multi_image_presentation(
            self.processor.tokenizer,
            prompt=self.assets.system_prompt,
            image_token_counts=[count],
        )

    def _start(self, image: Image.Image, seed: int | None) -> _Session:
        inputs = self.processor.image_processor(images=[image], return_tensors="pt")
        ids, tags = self._presentation(inputs)
        if len(ids) != self.text_rows:
            raise RuntimeError("Invoker-Render reference image changed the frozen token geometry")
        hidden = self.text_encoder.encode_ids(
            ids, pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]
        )
        latent = encode_rgb(self.vae, np.asarray(image)[None].copy(), proxy_resize=None, eos=False)
        causal = CausalSession(
            self.transformer,
            features=hidden,
            tags=tags,
            image=latent,
            seed=0 if seed is None else seed,
            ulysses=self.ulysses,
        )
        return _Session(causal, self.decoder.start())

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        width, height = VIDEO_SIZE
        rows = width // 32 * (height // 32)
        # 固定前缀、最近 12 对历史以及当前最多 5 个 proxy 同时可见。
        retained = 32 * rows + self.text_rows
        pending = 19 * rows + self.text_rows
        latent_pixels = width // 16 * (height // 16)
        decoder_bytes = (
            2 * len(self.decoder.blocks) * 4 * latent_pixels * self.decoder.heads * self.decoder.head_dim * 2
        )
        recent_latent_bytes = 24 * latent_pixels * 32 * (2 + 4 * 4)
        return ARDiffusionKVCacheSpec(
            num_layers=len(self.transformer.blocks),
            num_kv_heads=self.transformer.blocks[0].attn.num_sp_heads,
            head_size=self.transformer.arch.attention_head_dim,
            tokens_per_frame=rows,
            frames_per_block=1,
            window_frames=1,
            kv_branches=(ARDiffusionKVBranchSpec("video", 0),),
            session_capacity=1,
            segmented=ARDiffusionSegmentedKVSpec(
                block_size=256,
                max_retained_tokens=retained,
                max_retained_segments=30,
                max_pending_tokens=pending,
                max_pending_segments=9,
                initial_staging_tokens=retained + pending,
            ),
            model_owned_state_bytes_per_session=decoder_bytes + recent_latent_bytes,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_state is not None or session_id != state.session_id:
            raise RuntimeError("Invalid Invoker-Render AR Diffusion state binding")
        self._ar_state = state
        try:
            yield
        finally:
            self._ar_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self.close_ar_diffusion_session(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        pass

    def _cache(self):
        if self._ar_state is None:
            raise RuntimeError("Invoker-Render must run with the AR Diffusion engine")
        return self._ar_state.segmented("video")

    def _append(self, session: _Session, pixels: np.ndarray, *, eos: bool):
        proxy = encode_rgb(self.vae, pixels, proxy_resize=self.assets.proxy_resize, eos=eos)
        return session.causal.append(proxy, eos=eos)

    def _decode(self, session: _Session, window, latents: torch.Tensor) -> torch.Tensor:
        session.causal.commit(self._cache(), window, latents)
        video = self.decoder.decode(session.decoder, latents, eos=window.eos)
        session.delivered += len(video)
        return video

    def forward(self, batch: DiffusionRequestBatch) -> DiffusionOutput:
        if batch.num_reqs != 1:
            raise ValueError("Invoker-Render handles one request per worker group")
        request = batch.requests[0]
        data = request.prompt["multi_modal_data"]
        proxy = data["video"]
        session = self._start(data["image"], request.sampling_params.seed)
        cache = self._cache()
        frames = []
        for start in range(0, len(proxy), PACKET_FRAMES):
            eos = start + PACKET_FRAMES >= len(proxy)
            for window in self._append(session, proxy[start : start + PACKET_FRAMES], eos=eos):
                latents = session.causal.prepare(cache, window)
                for step in range(4):
                    velocity = session.causal.predict(cache, window, latents, step)
                    latents = euler_step(
                        latents, velocity, session.causal.sigmas[step], session.causal.sigmas[step + 1]
                    )
                frames.append(self._decode(session, window, latents))
        if session.delivered != len(proxy):
            raise RuntimeError(f"Invoker-Render produced {session.delivered} frames for {len(proxy)} source frames")
        return DiffusionOutput(output={"payload": {"video": torch.cat(frames)}, "metadata": {"video": {"fps": FPS}}})
