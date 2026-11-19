# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import av
import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import Qwen3VLProcessor
from vllm.distributed import get_tp_group
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import ReferenceVideoDecodeSpec
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState, StepWait
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
from .interaction import ProxyQueue
from .proxy import BOOTSTRAP_KEY, PROXY_MODALITY
from .schedule import Window


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
        if not isinstance(data, Mapping):
            raise OmniClientError("Invoker-Render requires structured media inputs")
        image = data.get("image")
        if not isinstance(image, Image.Image) or image.size != VIDEO_SIZE:
            raise OmniClientError("Invoker-Render requires one 1344x768 reference image")
        live = BOOTSTRAP_KEY in data
        media_key = BOOTSTRAP_KEY if live else "video"
        proxy = data.get(media_key)
        if live:
            if not od_config.streaming_output or not isinstance(proxy, list) or len(proxy) != 2:
                raise OmniClientError(
                    "Invoker-Render live requests require two bootstrap packets and the streaming profile"
                )
            queue = ProxyQueue(assets.proxy_size)
            try:
                for packet in proxy:
                    queue.append(queue.packet(packet))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(str(exc)) from exc
            count = 1
        else:
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
            or set(data) != {"image", media_key}
        ):
            raise OmniClientError("Invoker-Render uses fixed geometry, 24 FPS and four Euler steps with shift 12")
        sampling.width, sampling.height = VIDEO_SIZE
        sampling.fps = sampling.frame_rate = FPS
        sampling.num_frames = count
        sampling.num_inference_steps = 4
        sampling.extra_args = {**extra, "session_id": request.request_id, "reset": True, "close_session": True}
        request.prompt = {**prompt, "multi_modal_data": {"image": image.convert("RGB"), media_key: proxy}}
        request.sampling_params = sampling
        request.use_step_execution = live
        return request

    return preprocess


@dataclass
class _Session:
    causal: CausalSession
    decoder: DecodeState
    delivered: int = 0
    windows: deque[Window] = field(default_factory=deque)
    active: Window | None = None


class InvokerRenderPipeline(nn.Module, InteractionMixin):
    support_image_input: ClassVar[bool] = True
    dummy_run_num_frames: ClassVar[int] = 0
    supports_step_execution: ClassVar[bool] = True

    @staticmethod
    def reference_video_decode_spec(*, num_frames: int | None, extra_args: dict) -> ReferenceVideoDecodeSpec:
        return ReferenceVideoDecodeSpec()

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.device = get_local_device()
        self.assets = InvokerRenderAssets.read(od_config.model)
        parallel = od_config.parallel_config
        self.ulysses = parallel.ulysses_degree
        if parallel.ring_degree != 1 or od_config.quantization_config is not None:
            raise ValueError("Invoker-Render requires unquantized weights and ring_degree=1")
        self.transformer = MiniMaxH3DiTModel(od_config, diffusers_weights=True, attention_cls=InvokerRenderAttention)
        for index, block in enumerate(self.transformer.blocks):
            block.attn.layer_name = f"blocks.{index}.attn"
            block.attn.layer_index = index
        self.processor = Qwen3VLProcessor.from_pretrained(self.assets.processor, local_files_only=True)
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(
            self.assets.text_encoder,
            device=self.device,
            load_model=True,
            encoder_group=get_tp_group() if parallel.tensor_parallel_size > 1 else None,
        )
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
        # 前缀包含文本、参考图和首帧的 proxy 与目标。历史最多保留 12 对。
        prefix = 3 * rows + self.text_rows
        history = 24 * rows
        retained = prefix + history + 5 * rows
        pending = 19 * rows + self.text_rows
        # 去噪使用当前最多 5 个目标帧。更新 RoPE 时，新旧前缀同时驻留。
        staging = max(retained + 5 * rows, 2 * prefix + history)
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
                initial_staging_tokens=staging,
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

    def prepare_encode(self, state: StepRequestState, **kwargs) -> StepRequestState:
        if not isinstance(state.prompt, Mapping):
            raise RuntimeError("Invoker-Render live request has no prepared proxy inputs")
        data = state.prompt["multi_modal_data"]
        session = self._start(data["image"], state.sampling.seed)
        queue = ProxyQueue(self.assets.proxy_size)
        for event in data[BOOTSTRAP_KEY]:
            queue.append(queue.packet(event))
        state.extra[PROXY_MODALITY] = queue
        state.extra["invoker_session"] = session
        state.total_chunks = None
        state.chunk_num_steps = 4
        state.timesteps = (1 - session.causal.sigmas[:-1]).float()
        state.prompt = None
        return state

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        session: _Session = state.extra["invoker_session"]
        queue: ProxyQueue = state.extra[PROXY_MODALITY]
        while not session.windows and queue.pending:
            packet = queue.take()
            session.windows.extend(self._append(session, packet.as_frames(queue.size).copy(), eos=packet.eos))
        if not session.windows:
            session.active = None
            return
        session.active = session.windows.popleft()
        state.latents = session.causal.prepare(self._cache(), session.active)
        state.step_index = 0
        state.step_in_chunk = 0

    def ar_diffusion_step_wait(self, state: StepRequestState) -> StepWait | None:
        session: _Session = state.extra["invoker_session"]
        return StepWait() if session.active is None else None

    @torch.inference_mode()
    def ar_diffusion_resume_step(self, state: StepRequestState, reason: str) -> None:
        if reason != "interaction":
            raise RuntimeError("Invoker-Render proxy stream ended while waiting for input")
        self.apply_interaction_at_chunk_boundary(state)
        self.prepare_next_chunk(state)

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        session: _Session = state.extra["invoker_session"]
        window = session.active
        if window is None:
            raise ValueError("Invoker-Render has no ready proxy window")
        count = sum(1 if index % 5 == 0 else 4 for index in range(window.start, window.stop))
        return ChunkMediaSpec(num_media_frames=count, num_latent_frames=window.stop - window.start, fps=FPS)

    def denoise_step(self, input_batch, *, states: Sequence[StepRequestState], **kwargs) -> torch.Tensor:
        if len(states) != 1:
            raise ValueError("Invoker-Render handles one stream per worker group")
        state = states[0]
        session: _Session = state.extra["invoker_session"]
        if session.active is None or state.latents is None:
            raise RuntimeError("Invoker-Render cannot denoise without a ready proxy window")
        return session.causal.predict(self._cache(), session.active, state.latents, state.step_in_chunk)

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs) -> None:
        session: _Session = state.extra["invoker_session"]
        step = state.step_in_chunk
        assert state.latents is not None
        state.latents = euler_step(
            state.latents, noise_pred, session.causal.sigmas[step], session.causal.sigmas[step + 1]
        )
        state.step_in_chunk += 1
        state.step_index += 1

    def post_decode(self, state: StepRequestState, **kwargs) -> DiffusionOutput:
        session: _Session = state.extra["invoker_session"]
        queue: ProxyQueue = state.extra[PROXY_MODALITY]
        window = session.active
        assert window is not None and state.latents is not None
        source_start = session.delivered
        video = self._decode(session, window, state.latents)
        chunk_index = state.chunk_index
        state.chunk_index += 1
        state.stream_input_drained = window.eos
        if window.eos and session.delivered != queue.sequence.accepted:
            raise RuntimeError("Invoker-Render EOS left proxy frames unaccounted for")
        return DiffusionOutput(
            output={
                "payload": {"video": video},
                "metadata": {
                    "video": {"fps": FPS},
                    "stream": {
                        "source_frame_start": source_start,
                        "source_frame_count": len(video),
                        "source_frames_consumed": queue.sequence.consumed,
                    },
                },
            },
            chunk_index=chunk_index,
            total_chunks=state.chunk_index if window.eos else None,
            finished=window.eos,
        )

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
