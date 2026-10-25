# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.models.hy_nomas.interaction import HyNomasInteractionSession, validate_camera_payload
from vllm_omni.diffusion.models.hy_nomas.request import HyNomasRequest, normalize_request
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionSegmentedKVSpec,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.segmented import ARDiffusionSegmentedKVState
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata, ARDiffusionTickRequest


@dataclass(frozen=True)
class HyNomasARConfig:
    height: int = 704
    width: int = 1280
    chunk_latent_frames: int = 16
    max_latent_frames: int = 120
    max_segments: int = 16
    session_capacity: int = 1

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> HyNomasARConfig:
        values = {name: args.get(f"ar_{name}", info.default) for name, info in cls.__dataclass_fields__.items()}
        for name, value in values.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"Hy-Nomas ar_{name} must be a positive integer.")
        if values["height"] % 32 or values["width"] % 32:
            raise ValueError("Hy-Nomas ar_height and ar_width must be multiples of 32.")
        if values["chunk_latent_frames"] > values["max_latent_frames"]:
            raise ValueError("Hy-Nomas ar_chunk_latent_frames must not exceed ar_max_latent_frames.")
        return cls(**values)


@dataclass(frozen=True)
class HyNomasChunk:
    segment: int
    start: int
    end: int
    pixel_frames: int


def plan_chunks(request: HyNomasRequest, chunk_latents: int) -> list[HyNomasChunk]:
    boundaries = [value // 4 for value in request.boundaries[:-1]] + [(request.num_frames - 1) // 4 + 1]
    chunks = []
    for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        for left in range(start, end, chunk_latents):
            right = min(left + chunk_latents, end)
            pixels = min(request.num_frames, 4 * right) - 4 * left
            chunks.append(HyNomasChunk(segment, left, right, pixels))
    return chunks


@dataclass
class HyNomasSession:
    height: int
    width: int
    seed: int
    noise: torch.Tensor
    generator: torch.Generator
    decoder: Any
    history_mask: torch.Tensor
    latent_frames: int = 0
    segments: int = 0
    tick_index: int = 0
    emitted_frames: int = 0
    # Original masks are retained separately from per-forward visibility masks.
    action_tails: list[tuple[int, int, int]] = field(default_factory=list)


def reference_cache_views(cache: ARDiffusionSegmentedKVState, current_tokens: int) -> list[dict[str, torch.Tensor]]:
    """Expose runner-owned history to the reference attention implementation.

    The unused suffix is scratch until the reference clean forward writes it.
    Only that successful forward can publish rows through ``commit_cache``.
    No view survives the runner's state binding.
    """
    cache.prepare_layer_views(current_tokens, 0)
    scratch = torch.empty(
        (current_tokens, cache.cache.num_kv_heads, cache.cache.head_size),
        dtype=cache.cache.dtype,
        device=cache.cache.device,
    )
    empty = scratch[:0]
    result = []
    for layer in range(cache.cache.num_layers):
        key, value = cache.assemble(layer, empty, empty, scratch, scratch)
        result.append({"k": key.unsqueeze(0), "v": value.unsqueeze(0)})
    return result


def commit_cache(cache: ARDiffusionSegmentedKVState, views: list[dict[str, torch.Tensor]]) -> None:
    start = cache.history_tokens
    for layer, view in enumerate(views):
        key, value = view["k"][0, start:], view["v"][0, start:]
        cache.assemble(layer, key[:0], value[:0], key, value)
        cache.stage(layer)


class HyNomasAR(InteractionMixin):
    supports_step_execution: ClassVar[bool] = True
    supports_chunk_step_grouping: ClassVar[bool] = True
    pipe: Any
    device: torch.device
    sp_size: int

    def initialize_ar(self, args: Mapping[str, Any]) -> None:
        self.ar_config = HyNomasARConfig.from_args(args)
        self._ar_state: ARDiffusionKVState | None = None
        self._ar_sessions: dict[str, HyNomasSession] = {}

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        config, pipe = self.ar_config, self.pipe
        frame_tokens = (config.height // pipe.s_ratio // pipe.patch_size[1]) * (
            config.width // pipe.s_ratio // pipe.patch_size[2]
        )
        text_tokens = int(pipe.args.text_len)
        retained = config.max_latent_frames * frame_tokens + config.max_segments * text_tokens
        pending = config.chunk_latent_frames * frame_tokens + text_tokens
        latent_elements = config.max_latent_frames * (config.height // pipe.s_ratio) * (config.width // pipe.s_ratio)
        # Fixed noise, the decoder's latent timeline, pending pixels and text.
        model_bytes = latent_elements * (pipe.in_channels + pipe.transformer.config.out_channels) * 2
        model_bytes += config.height * config.width * 3 * 4 * 40 + config.max_segments * text_tokens * 3584 * 2
        return ARDiffusionKVCacheSpec(
            num_layers=pipe.num_double_blocks,
            num_kv_heads=pipe.heads_num // self.sp_size,
            head_size=pipe.hidden_size // pipe.heads_num,
            tokens_per_frame=frame_tokens,
            frames_per_block=1,
            window_frames=1,
            kv_branches=(ARDiffusionKVBranchSpec("cond", 0),),
            session_capacity=config.session_capacity,
            segmented=ARDiffusionSegmentedKVSpec(
                block_size=256,
                max_retained_tokens=retained,
                max_retained_segments=config.max_latent_frames,
                max_pending_tokens=pending,
                max_pending_segments=1,
                initial_staging_tokens=retained + pending,
            ),
            model_owned_state_bytes_per_session=model_bytes,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_state is not None or state.session_id != session_id:
            raise RuntimeError("Hy-Nomas AR state binding is nested or has a different session ID.")
        self._ar_state = state
        try:
            yield
        finally:
            self._ar_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self.reset_ar_diffusion_session(session_id)

    def _bound(self) -> ARDiffusionKVState:
        if self._ar_state is None:
            raise RuntimeError("Hy-Nomas streaming and sessions require the ARDiffusionEngine backend.")
        return self._ar_state

    def _session(self) -> HyNomasSession:
        return self._ar_sessions[self._bound().session_id]

    @torch.no_grad()
    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        bound = self._bound()
        tick = ARDiffusionTickRequest.from_extra_args(state.sampling.extra_args)
        prompt = tick.prompt if tick is not None and tick.prompt is not None else state.prompt
        request = normalize_request(prompt, state.sampling)
        config = self.ar_config
        if request.height > config.height or request.width > config.width:
            raise OmniClientError("Hy-Nomas request exceeds ar_height/ar_width; increase the deployment limits.")
        frame_tokens = (request.height // self.pipe.s_ratio // self.pipe.patch_size[1]) * (
            request.width // self.pipe.s_ratio // self.pipe.patch_size[2]
        )
        if frame_tokens % self.sp_size:
            raise OmniClientError("Hy-Nomas spatial tokens per frame must be divisible by ulysses_degree.")
        frames = (request.num_frames - 1) // 4 + 1
        if frames > config.max_latent_frames:
            raise OmniClientError("Hy-Nomas request exceeds ar_max_latent_frames.")
        chunks = plan_chunks(request, config.chunk_latent_frames)
        if tick is not None and len(chunks) != 1:
            raise OmniClientError("An AR-Diffusion tick must fit one Hy-Nomas chunk and contain one prompt segment.")
        extra = state.sampling.extra_args
        persistent = tick is not None or "session_id" in extra
        if persistent and not state.extra.get("hy_nomas_request_mode"):
            raise OmniClientError(
                "Persistent Hy-Nomas ticks require step_execution=false; live streams own one request."
            )
        finish = not persistent or (tick.close_session if tick is not None else extra.get("close_session", False))
        if not persistent:
            bound.reset()
            self.reset_ar_diffusion_session(bound.session_id)
        session = self._ar_sessions.get(bound.session_id)
        if session is None:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)
            noise = torch.randn(
                (
                    1,
                    self.pipe.in_channels,
                    config.max_latent_frames,
                    request.height // self.pipe.s_ratio,
                    request.width // self.pipe.s_ratio,
                ),
                generator=generator,
                dtype=self.pipe.target_dtype,
            ).to(self.device)
            session = HyNomasSession(
                request.height,
                request.width,
                request.seed,
                noise,
                generator,
                self.pipe.vae.create_streaming_decode_session(to_cpu=True, retain_history=False),
                torch.empty((1, 0), dtype=torch.bool, device=self.device),
            )
            self._ar_sessions[bound.session_id] = session
        if (session.height, session.width, session.seed) != (request.height, request.width, request.seed):
            raise OmniClientError("A Hy-Nomas continuation must keep its height, width and seed; reset to change them.")
        if session.latent_frames + frames > config.max_latent_frames:
            raise OmniClientError("Hy-Nomas session exceeds ar_max_latent_frames; close or reset the session.")
        if tick is not None and (tick.session_id != bound.session_id or tick.chunk_index != session.tick_index):
            raise OmniClientError("Hy-Nomas tick session/chunk_index does not match its persistent state.")
        if tick is not None:
            for control in tick.controls:
                if control.track != "camera" or control.schema != "hy_nomas.camera.v1":
                    raise OmniClientError("Hy-Nomas accepts only camera controls with schema hy_nomas.camera.v1.")
                validate_camera_payload(control.data)
                state.extra["hy_nomas_camera_override"] = control.data["camera_trajectory"]
        state.extra.update(
            hy_nomas_request=request,
            hy_nomas_chunks=chunks,
            hy_nomas_tick=tick,
            hy_nomas_finish=finish,
            hy_nomas_request_latent_start=session.latent_frames,
            hy_nomas_segment_key=None,
        )
        state.total_chunks = len(chunks)
        state.chunk_index = state.step_index = state.step_in_chunk = 0
        state.chunk_num_steps = request.steps
        state.do_true_cfg = False
        return state

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        chunk = state.extra["hy_nomas_chunks"][state.chunk_index]
        return ChunkMediaSpec(num_media_frames=chunk.pixel_frames, fps=24.0, num_latent_frames=chunk.end - chunk.start)

    @torch.no_grad()
    def prepare_next_chunk(self, state: StepRequestState) -> None:
        session, pipe = self._session(), self.pipe
        request = state.extra["hy_nomas_request"]
        chunk = state.extra["hy_nomas_chunks"][state.chunk_index]
        text = state.extra.get("hy_nomas_prompt_override", request.prompts[chunk.segment])
        segment_key = (chunk.segment, state.extra.get("hy_nomas_prompt_version", 0))
        first = segment_key != state.extra["hy_nomas_segment_key"]
        if first:
            if session.segments >= self.ar_config.max_segments:
                raise OmniClientError("Hy-Nomas session exceeds ar_max_segments; close or reset the session.")
            encoded = pipe._encode_prompt(text, request.negative_prompt, encode_negative=False)
            state.prompt_embeds = encoded["text_states"]
            # The reference owns padding and action-token coordinates. The
            # generic batcher must not trim that conditioning to mask.sum().
            state.prompt_embeds_mask = None
            state.extra["hy_nomas_text_mask"] = encoded["text_mask"]
            state.txt_seq_lens = [int(state.prompt_embeds.shape[1])]
            if state.txt_seq_lens[0] > int(pipe.args.text_len):
                raise OmniClientError("Hy-Nomas prompt exceeds the deployed text token limit.")
            prompt_session = state.interaction_sessions.setdefault("prompt", HyNomasInteractionSession())
            assert isinstance(prompt_session, HyNomasInteractionSession)
            with prompt_session.lock:
                prompt_session.version += 1
            session.segments += 1
            state.extra["hy_nomas_segment_key"] = segment_key
            state.extra["hy_nomas_segment_id"] = session.segments
            state.extra["hy_nomas_action_start"] = (
                pipe._compute_action_start(text) if pipe._cross_segment_text_head_only_enabled() else -1
            )
        offsets = windows = None
        if first and pipe._intra_seg_enabled():
            offsets = pipe._compute_action_text_offsets(text)
            if offsets is not None:
                segment_start = request.boundaries[chunk.segment] // 4
                if "hy_nomas_prompt_override" in state.extra:
                    segment_start = chunk.start
                segment_end = (request.num_frames - 1) // 4 + 1
                if chunk.segment + 1 < len(request.prompts):
                    segment_end = request.boundaries[chunk.segment + 1] // 4
                windows = pipe._compute_action_video_windows(
                    text, seg_latent_start=segment_start, seg_latent_end=segment_end
                )
        state.extra["hy_nomas_intra_offsets"] = [offsets] if offsets is not None and windows is not None else None
        state.extra["hy_nomas_intra_windows"] = windows
        state.extra["hy_nomas_cache_text"] = first
        start = state.extra["hy_nomas_request_latent_start"] + chunk.start
        end = start + chunk.end - chunk.start
        state.latents = session.noise[:, : pipe.transformer.config.out_channels, start:end].clone()
        state.scheduler = pipe.omni_scheduler(request.steps)
        state.timesteps = state.scheduler.timesteps
        state.step_index = state.step_in_chunk = 0
        camera = state.extra.get("hy_nomas_camera_override", request.camera_trajectory)
        overridden = "hy_nomas_camera_override" in state.extra
        camera_key = (camera, overridden)
        if state.extra.get("hy_nomas_camera_key") != camera_key:
            origin = chunk.start if overridden else 0
            with pipe._dit_denoise_residency():
                state.extra["hy_nomas_camera_full"] = pipe._prepare_camera_keyboard(
                    camera,
                    num_frames=request.num_frames - 4 * origin,
                    latent_h=request.height // pipe.s_ratio,
                    latent_w=request.width // pipe.s_ratio,
                    segment_frame_boundaries=None if overridden else request.boundaries,
                )
            state.extra["hy_nomas_camera_key"] = camera_key
            state.extra["hy_nomas_camera_origin"] = origin
        origin = state.extra["hy_nomas_camera_origin"]
        state.extra["hy_nomas_camera"], state.extra["hy_nomas_camera_mask"] = pipe._slice_chunk_camera(
            camera_embedding_full=state.extra["hy_nomas_camera_full"],
            cam_mode=pipe.transformer.camera_injection_mode,
            chunk_latent_start=chunk.start - origin,
            chunk_latent_end=chunk.end - origin,
            seg_idx=chunk.segment,
            segment_has_camera=None if overridden else request.segment_has_camera,
            guidance_branches=1,
        )

    def _forward_chunk(
        self, state: StepRequestState, latents: torch.Tensor, timestep: torch.Tensor, *, commit: bool
    ) -> torch.Tensor:
        session, pipe = self._session(), self.pipe
        cache = self._bound().segmented("cond")
        if state.prompt_embeds is None:
            raise RuntimeError("Hy-Nomas requires encoded text before a DiT forward.")
        first = state.extra["hy_nomas_cache_text"]
        image_tokens = (
            (latents.shape[2] // pipe.patch_size[0])
            * (latents.shape[3] // pipe.patch_size[1])
            * (latents.shape[4] // pipe.patch_size[2])
        )
        current_tokens = image_tokens + (state.prompt_embeds.shape[1] if first else 0)
        views = reference_cache_views(cache, current_tokens)
        history = cache.history_tokens
        mask = torch.cat(
            (session.history_mask, torch.zeros((1, current_tokens), device=self.device, dtype=torch.bool)), 1
        )
        for segment, start, end in session.action_tails:
            if segment != state.extra["hy_nomas_segment_id"]:
                mask[:, start:end] = False
        if latents.shape[1] < pipe.in_channels:
            latents = torch.cat(
                (
                    latents,
                    latents.new_zeros((latents.shape[0], pipe.in_channels - latents.shape[1], *latents.shape[2:])),
                ),
                1,
            )
        pending = cache.begin_commit((current_tokens,)) if commit else ()
        try:
            prediction, written = pipe.omni_forward_with_cache(
                hidden_states=latents,
                timestep=timestep,
                text_states=state.prompt_embeds,
                text_mask=state.extra["hy_nomas_text_mask"],
                kv_cache_list=views,
                cache_valid_len=history,
                update_cache=commit,
                global_frame_counter=session.latent_frames,
                guidance=None,
                cache_txt=first,
                cache_mask_list=[mask] * len(views),
                camera_embedding=state.extra["hy_nomas_camera"],
                camera_has_mask=state.extra["hy_nomas_camera_mask"],
                intra_seg_txt_offsets=state.extra["hy_nomas_intra_offsets"],
                intra_seg_vid_windows=state.extra["hy_nomas_intra_windows"],
                camera_rope_time_offset=0,
            )
            if commit:
                if written != current_tokens:
                    raise RuntimeError("Hy-Nomas clean forward wrote an unexpected KV token count.")
                commit_cache(cache, views)
                cache.commit(retain=(*cache.handles, *pending))
                session.history_mask = torch.cat((session.history_mask, mask[:, history:].clone()), 1)
                action_start = state.extra["hy_nomas_action_start"]
                if first and action_start is not None and 0 <= action_start < state.prompt_embeds.shape[1]:
                    session.action_tails.append(
                        (
                            state.extra["hy_nomas_segment_id"],
                            history + image_tokens + action_start,
                            history + current_tokens,
                        )
                    )
            return prediction
        except BaseException:
            if commit:
                cache.rollback()
            raise

    @torch.no_grad()
    def denoise_step(
        self, input_batch: Any, *, states: Sequence[StepRequestState] | None = None, **kwargs: Any
    ) -> torch.Tensor:
        del input_batch, kwargs
        if states is None or len(states) != 1:
            raise ValueError("Hy-Nomas step execution supports one request at a time.")
        state = states[0]
        if state.latents is None or state.scheduler is None:
            raise RuntimeError("Hy-Nomas prepare_next_chunk must precede denoise_step.")
        self.pipe.transformer._sparse_step_index = state.step_in_chunk
        return self._forward_chunk(
            state, state.latents, state.scheduler.timesteps[state.step_in_chunk].expand(1), commit=False
        )

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        if state.scheduler is None or state.latents is None:
            raise RuntimeError("Hy-Nomas prepare_next_chunk must precede step_scheduler.")
        state.latents = state.scheduler.step(
            model_output=noise_pred,
            timestep=state.scheduler.timesteps[state.step_in_chunk],
            sample=state.latents,
            return_dict=False,
        )[0].to(self.pipe.target_dtype)
        state.step_in_chunk += 1
        state.step_index += 1

    def _decode_chunk(
        self, session: HyNomasSession, latents: torch.Tensor, *, pixel_frames: int, finish: bool
    ) -> np.ndarray:
        pipe = self.pipe
        error = None
        video = None
        try:
            with pipe._component_context(pipe.vae):
                session.decoder.final_segment = finish
                pixels = session.decoder.push(pipe._prepare_vae_decode_latents(latents), pixel_frames=pixel_frames)
                if finish:
                    tail = session.decoder.finish()
                    if tail.numel():
                        pixels = torch.cat((pixels, tail), 2) if pixels.numel() else tail
            if pixels.numel():
                pixels = pipe._postprocess_vae_video(pixels)
                video = pixels[0].permute(1, 2, 3, 0).float().mul(255).round().to(torch.uint8).cpu().numpy()
            else:
                video = np.empty((0, session.height, session.width, 3), dtype=np.uint8)
        except Exception as exc:
            error = exc
        # PixelUMM runs independently on each SP rank. All peers must reject
        # a failed decode before any of them enters the next DiT collective.
        if self.sp_size > 1:
            failed = torch.tensor([error is not None], dtype=torch.int32, device=self.device)
            if get_sp_group().all_reduce(failed).item() and error is None:
                error = RuntimeError("Hy-Nomas PixelUMM decoding failed on another SP rank.")
        if error is not None:
            raise error
        assert video is not None
        return video

    @torch.no_grad()
    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        if state.latents is None or not state.chunk_denoise_completed:
            raise RuntimeError("Hy-Nomas must complete every denoising step before committing a chunk.")
        session, pipe = self._session(), self.pipe
        chunk = state.extra["hy_nomas_chunks"][state.chunk_index]
        cache_latents = state.latents
        sigma = pipe.cache_noise_sigma
        if sigma:
            epsilon = torch.randn(cache_latents.shape, generator=session.generator, dtype=cache_latents.dtype).to(
                self.device
            )
            cache_latents = ((1 - sigma) * cache_latents.float() + sigma * epsilon.float()).to(cache_latents.dtype)
        self._forward_chunk(
            state,
            cache_latents,
            torch.full((1,), sigma * 1000, dtype=torch.float32, device=self.device),
            commit=True,
        )
        session.latent_frames += chunk.end - chunk.start
        frame_start = session.emitted_frames
        completed = state.chunk_index
        state.chunk_index += 1
        video = self._decode_chunk(
            session,
            state.latents,
            pixel_frames=chunk.pixel_frames,
            finish=state.request_denoise_completed and state.extra["hy_nomas_finish"],
        )
        session.emitted_frames += len(video)
        tick = state.extra["hy_nomas_tick"]
        identity = (
            ARDiffusionChunkMetadata.from_tick(tick)
            if tick is not None
            else ARDiffusionChunkMetadata(
                session_id=self._bound().session_id,
                request_id=state.request_id,
                chunk_index=completed,
            )
        )
        if state.request_denoise_completed:
            session.tick_index += 1
        return DiffusionOutput(
            output={
                "payload": {"video": video},
                "metadata": {
                    "video": {"fps": 24.0},
                    "ar_diffusion": identity.to_dict(),
                    "hy_nomas": {
                        "frame_start": frame_start,
                        "num_frames": len(video),
                        "generated_frames": session.decoder.available_pixel_frames,
                        "withheld_frames": session.decoder.available_pixel_frames - session.emitted_frames,
                    },
                },
            },
            chunk_index=completed,
            total_chunks=state.total_chunks,
            finished=state.request_denoise_completed,
        )

    @torch.no_grad()
    def forward_ar(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if request.num_reqs != 1:
            raise ValueError("Hy-Nomas processes one request at a time.")
        req = request.requests[0]
        state = StepRequestState(request_id=req.request_id, sampling=req.sampling_params, prompt=req.prompt)
        state.extra["hy_nomas_request_mode"] = True
        self.prepare_encode(state)
        frames = []
        frame_start = self._session().emitted_frames
        while not state.request_denoise_completed:
            self.prepare_next_chunk(state)
            assert state.chunk_num_steps is not None
            for _ in range(state.chunk_num_steps):
                self.step_scheduler(state, self.denoise_step(None, states=[state]))
            output = self.post_decode(state)
            frames.append(output.output["payload"]["video"])
        output.output["payload"]["video"] = np.concatenate(frames, axis=0)
        output.output["metadata"]["hy_nomas"].update(
            frame_start=frame_start, num_frames=len(output.output["payload"]["video"])
        )
        return output
