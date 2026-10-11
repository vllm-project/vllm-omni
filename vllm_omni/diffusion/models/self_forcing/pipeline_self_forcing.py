# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Self-Forcing causal DMD text-to-video pipeline.

One AR block is ``num_frames_per_block`` latent frames, denoised by four DMD
probes at the checkpoint's warped timesteps and then written to the causal
K/V by a clean ``t=0`` pass. Request mode rolls out a whole clip against a
request-local cache; step execution streams one block per chunk and keeps the
K/V on the AR-Diffusion runner's paged session instead.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.interaction.mixin import InteractionMixin
from vllm_omni.diffusion.interaction.modality_handlers.prompt import prompt_update_versions
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery, SupportsStepExecution
from vllm_omni.diffusion.models.lingbot_world.transformer import LingBotAttentionCache, LingBotTransformerCache
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.self_forcing.transformer import SelfForcingTransformer3DModel
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.experimental.ar_diffusion.streaming_decode import StreamingDecodeState, WanStreamingDecoder
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata, ARDiffusionTickRequest

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmProgressBar

    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

logger = init_logger(__name__)

# Raw DMD step labels on the unshifted [1, 1/1000] training lattice and the
# shift the generator was distilled with; see ``self_forcing_dmd_schedule``.
SELF_FORCING_DMD_TIMESTEPS = (1000, 750, 500, 250)
SELF_FORCING_FLOW_SHIFT = 5.0
SELF_FORCING_DEFAULT_HEIGHT = 480
SELF_FORCING_DEFAULT_WIDTH = 832
SELF_FORCING_DEFAULT_NUM_FRAMES = 81
SELF_FORCING_FPS = 16
_MAX_SEQUENCE_LENGTH = 512
_MAX_PIXEL_AREA = SELF_FORCING_DEFAULT_HEIGHT * SELF_FORCING_DEFAULT_WIDTH
# Streaming Wan2.1 VAE decoder cache per output pixel in fp32, measured on the
# Wan2.1-T2V-1.3B VAE; the runner sizes session admission with it.
_STREAMING_DECODE_BYTES_PER_PIXEL_FP32 = 37832 * 1024 / (64 * 64)


def self_forcing_dmd_schedule(flow_shift: float = SELF_FORCING_FLOW_SHIFT) -> tuple[tuple[float, float], ...]:
    """Return the warped ``(timestep, sigma)`` pair of each DMD step.

    Self-Forcing indexes the shifted 1000-step flow-matching table with its raw
    labels, which is the closed form ``sigma = s * t / (1 + (s - 1) * t)``; the
    timestep fed to the transformer is ``1000 * sigma``.
    """
    base = torch.tensor(SELF_FORCING_DMD_TIMESTEPS, dtype=torch.float64) / 1000
    sigmas = flow_shift * base / (1 + (flow_shift - 1) * base)
    return tuple((float(sigma * 1000), float(sigma)) for sigma in sigmas.tolist())


def _uint8_frames(video: torch.Tensor) -> np.ndarray:
    """Turn one decoded block ``(1, 3, F, H, W)`` in ``[-1, 1]`` into ``(F, H, W, 3)`` uint8."""
    frames = (video[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 3, 0)
    return np.ascontiguousarray(frames.float().mul_(255).round_().to(torch.uint8).cpu().numpy())


def get_self_forcing_post_process_func(od_config: OmniDiffusionConfig) -> Callable[..., Any]:
    """Convert request-mode video tensors to frames; pass AR payloads through."""
    del od_config
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor | dict[str, Any], output_type: str = "np", sampling_params: Any | None = None
    ) -> Any:
        if isinstance(video, dict) and isinstance(video.get("payload"), dict):
            return video
        if sampling_params is not None:
            output_type = getattr(sampling_params, "output_type", None) or output_type
        if output_type == "latent":
            return video
        return {"payload": {"video": video_processor.postprocess_video(video, output_type=output_type)}, "metadata": {}}

    return post_process_func


@dataclass(frozen=True)
class _SelfForcingRequestInputs:
    prompt: str
    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    output_type: str
    generator: torch.Generator


@dataclass
class _SelfForcingSessionState:
    """Model-owned per-session state; attention K/V stays runner-owned."""

    next_chunk_index: int = 0
    # Key of the prompt embedding the runner-owned text K/V was built from.
    # ``text_cache_dirty`` stays raised while a build is in flight so a failed
    # build forces the next chunk to rebuild instead of reading partial K/V.
    text_cache_key: tuple[int, ...] | None = None
    text_cache_dirty: bool = False


@dataclass(frozen=True)
class _ARBlockContext:
    """The runner-bound session one block runs against."""

    state: ARDiffusionKVState
    cross_attention: list[LingBotAttentionCache]


class SelfForcingPipeline(
    nn.Module,
    SupportsComponentDiscovery,
    SupportsStepExecution,
    InteractionMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
):
    """Self-Forcing Wan2.1-T2V-1.3B with request-local or session-paged causal K/V."""

    supports_step_execution: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    # The generic warmup request (two steps, 512x512) is not a valid DMD rollout.
    dummy_run_num_frames: ClassVar[int] = 0
    _AR_BRANCH = "main"
    _AR_TEXT_CACHE = "text"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        _validate_parallel_config(od_config)
        self.od_config = od_config
        model_config = getattr(od_config, "model_config", None) or {}
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model = od_config.model
        local_files_only = os.path.exists(model)
        managed_placement = resolve_offload_strategy(od_config) in (
            OffloadStrategy.MODEL_LEVEL,
            OffloadStrategy.LAYER_WISE,
        )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only)
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model,
            subfolder="tokenizer",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        if not managed_placement:
            self.text_encoder = self.text_encoder.to(self.device)
            self.vae = self.vae.to(self.device)

        self.transformer = SelfForcingTransformer3DModel.from_config(
            load_transformer_config(model, "transformer", local_files_only),
            quant_config=getattr(od_config, "quantization_config", None),
            prefix="transformer",
        )
        self.vae_scale_factor_temporal = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        self.vae_scale_factor_spatial = int(getattr(self.vae.config, "scale_factor_spatial", 8))
        self._schedule = self_forcing_dmd_schedule()
        self._ar_height = int(model_config.get("ar_diffusion_height", SELF_FORCING_DEFAULT_HEIGHT))
        self._ar_width = int(model_config.get("ar_diffusion_width", SELF_FORCING_DEFAULT_WIDTH))
        self._ar_diffusion_kv_state: ARDiffusionKVState | None = None
        self._ar_sessions: dict[str, _SelfForcingSessionState] = {}
        self._streaming_decode_states: dict[str, StreamingDecodeState] = {}
        self._streaming_decoder_instance: WanStreamingDecoder | None = None
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=["text_encoder.forward", "_generate_block", "vae.decode", "_decode_chunk_to_pixels"],
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    # ── AR-Diffusion capability ───────────────────────────────────────────

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        """Describe the fixed worker-local cache geometry for realtime sessions."""
        spatial = self.vae_scale_factor_spatial
        latent_height = self._ar_height // spatial
        latent_width = self._ar_width // spatial
        config = self.transformer.config
        return ARDiffusionKVCacheSpec(
            num_layers=config.num_layers,
            num_kv_heads=int(self.transformer.blocks[0].self_attn.num_sp_heads),
            head_size=config.attention_head_dim,
            tokens_per_frame=self.transformer.tokens_per_frame(latent_height, latent_width),
            frames_per_block=config.num_frames_per_block,
            window_frames=config.window_frames - config.sink_frames,
            sink_frames=config.sink_frames,
            kv_branches=(ARDiffusionKVBranchSpec(self._AR_BRANCH, 0),),
            session_capacity=2,
            cross_attention=(
                ARDiffusionCrossAttentionKVSpec(
                    self._AR_TEXT_CACHE,
                    _MAX_SEQUENCE_LENGTH,
                    num_kv_heads=int(self.transformer.blocks[0].cross_attn.num_local_heads),
                ),
            ),
            model_owned_state_bytes_per_session=self._streaming_decoder().declared_state_bytes(
                height=self._ar_height, width=self._ar_width, dtype=self.vae.dtype
            ),
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("Self-Forcing AR-Diffusion state is already bound.")
        if state.session_id != session_id:
            raise ValueError(f"Self-Forcing bound session mismatch: {state.session_id!r} != {session_id!r}.")
        self._ar_diffusion_kv_state = state
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)
        self._release_streaming_decode_state(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._ar_sessions.pop(session_id, None)
        self._release_streaming_decode_state(session_id)

    # ── request parsing and conditioning ─────────────────────────────────

    def _parse_request(self, req: DiffusionRequestBatch) -> _SelfForcingRequestInputs:
        if req.num_reqs != 1 or len(req.prompts) != 1:
            raise ValueError("Self-Forcing supports a single prompt request, not request batching.")
        sampling = req.sampling_params
        if int(sampling.num_outputs_per_prompt or 1) != 1:
            raise ValueError("Self-Forcing requires num_outputs_per_prompt=1.")
        generator = getattr(sampling, "generator", None)
        if not isinstance(generator, torch.Generator):
            raise ValueError("Self-Forcing requires one runner-provided torch.Generator.")
        if getattr(sampling, "latents", None) is not None:
            raise ValueError("Self-Forcing does not support caller-provided latents.")
        if (sampling.guidance_scale or 0.0) > 1.0:
            raise ValueError("Self-Forcing is a distilled DMD generator without classifier-free guidance.")

        prompt_value = req.prompts[0]
        if isinstance(prompt_value, dict):
            if prompt_value.get("multi_modal_data"):
                raise ValueError("Self-Forcing is text-to-video and accepts no multi_modal_data.")
            if prompt_value.get("negative_prompt"):
                raise ValueError("Self-Forcing does not use a negative prompt.")
            prompt_value = prompt_value.get("prompt")
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ValueError("prompt must contain non-empty text.")

        num_inference_steps = getattr(sampling, "num_inference_steps", None)
        if num_inference_steps not in (None, len(SELF_FORCING_DMD_TIMESTEPS)):
            raise ValueError(
                f"num_inference_steps must be {len(SELF_FORCING_DMD_TIMESTEPS)} for the Self-Forcing DMD "
                f"timesteps {list(SELF_FORCING_DMD_TIMESTEPS)}, got {num_inference_steps}."
            )
        max_sequence_length = getattr(sampling, "max_sequence_length", None)
        if max_sequence_length not in (None, _MAX_SEQUENCE_LENGTH):
            raise ValueError(f"max_sequence_length must be {_MAX_SEQUENCE_LENGTH}.")

        height = getattr(sampling, "height", None) or SELF_FORCING_DEFAULT_HEIGHT
        width = getattr(sampling, "width", None) or SELF_FORCING_DEFAULT_WIDTH
        divisor = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        if height % divisor or width % divisor:
            raise ValueError(f"height and width must be multiples of {divisor}, got {height}x{width}.")
        if height * width > _MAX_PIXEL_AREA:
            raise ValueError("height * width must not exceed the trained 480 * 832 pixel area.")

        num_frames = getattr(sampling, "num_frames", None)
        if num_frames in (None, 1):
            num_frames = SELF_FORCING_DEFAULT_NUM_FRAMES
        temporal = self.vae_scale_factor_temporal
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or (num_frames - 1) % temporal:
            raise ValueError(f"num_frames must satisfy (num_frames - 1) % {temporal} == 0, got {num_frames!r}.")
        num_latent_frames = (num_frames - 1) // temporal + 1
        block_frames = self.transformer.config.num_frames_per_block
        if num_latent_frames % block_frames:
            raise ValueError(
                f"num_frames={num_frames} maps to {num_latent_frames} latent frames, which is not a whole number "
                f"of {block_frames}-frame AR blocks."
            )
        return _SelfForcingRequestInputs(
            prompt=prompt_value.strip(),
            height=height,
            width=width,
            num_frames=num_frames,
            num_latent_frames=num_latent_frames,
            output_type=getattr(sampling, "output_type", None) or "np",
            generator=generator,
        )

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        do_classifier_free_guidance: bool = False,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Encode ``prompt`` with UMT5, zeroing the padding past its tokens.

        The signature matches the prompt-interaction handler; Self-Forcing has
        no negative branch, so only the positive embedding is returned.
        """
        del device
        if negative_prompt is not None or do_classifier_free_guidance or num_videos_per_prompt != 1:
            raise ValueError("Self-Forcing encodes one positive prompt without classifier-free guidance.")
        max_sequence_length = max_sequence_length or _MAX_SEQUENCE_LENGTH
        dtype = dtype or self.transformer.dtype
        text = " ".join(prompt.strip().split())
        text_inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        prompt_embeds = self.text_encoder(input_ids, attention_mask).last_hidden_state.to(dtype=dtype)
        return prompt_embeds * attention_mask.unsqueeze(-1).to(dtype=dtype), None

    def _ar_text_caches(self, prompt_embeds: torch.Tensor, *, invalidate: bool) -> list[LingBotAttentionCache]:
        """Runner-owned text K/V of the bound session, projected once per prompt embedding."""
        state = self._require_bound_ar_state()
        if invalidate:
            state.clear_cross_attention()
        if not state.is_cross_attention_populated(self._AR_BRANCH, self._AR_TEXT_CACHE):
            projected_text = self.transformer.text_embedding(prompt_embeds)

            def layer_kv() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
                for block in self.transformer.blocks:
                    cross_attention = block.cross_attn
                    shape = (cross_attention.num_local_heads, cross_attention.head_dim)
                    key = cross_attention.norm_k(cross_attention.k(projected_text)).unflatten(2, shape)
                    value = cross_attention.v(projected_text).unflatten(2, shape)
                    yield key, value

            state.populate_cross_attention(self._AR_BRANCH, self._AR_TEXT_CACHE, layer_kv())
        return [
            LingBotAttentionCache(
                key=layer["k"],
                value=layer["v"],
                end=prompt_embeds.shape[1],
                absolute_end=prompt_embeds.shape[1],
                last_start=0,
            )
            for layer in state.get_cross_attention_kv(self._AR_BRANCH, self._AR_TEXT_CACHE)
        ]

    def _session_text_caches(
        self,
        prompt_embeds: torch.Tensor,
        text_key: tuple[int, ...],
        session_state: _SelfForcingSessionState,
    ) -> list[LingBotAttentionCache]:
        """Text K/V for the next block, rebuilt whenever the prompt embedding changed."""
        previous = session_state.text_cache_key
        invalidate = session_state.text_cache_dirty or (previous is not None and previous != text_key)
        session_state.text_cache_dirty = True
        session_state.text_cache_key = None
        caches = self._ar_text_caches(prompt_embeds, invalidate=invalidate)
        session_state.text_cache_key = text_key
        session_state.text_cache_dirty = False
        return caches

    def _require_bound_ar_state(self) -> ARDiffusionKVState:
        if self._ar_diffusion_kv_state is None:
            raise RuntimeError("Self-Forcing AR-Diffusion execution requires a runner-bound session.")
        return self._ar_diffusion_kv_state

    # ── one AR block ──────────────────────────────────────────────────────
    #
    # Request mode loops ``_generate_block``; step execution calls the probe,
    # transition and commit below one scheduler step at a time, so both modes
    # share one copy of the DMD math.

    def _block_cache(
        self,
        *,
        cache: LingBotTransformerCache | None,
        ar: _ARBlockContext | None,
        seq_len: int,
        commit_current: bool,
    ) -> LingBotTransformerCache:
        if ar is None:
            return cast(LingBotTransformerCache, cache)
        return LingBotTransformerCache(
            self_attention=ar.state.get_kv_caches(self._AR_BRANCH, seq_len=seq_len, commit_current=commit_current),
            cross_attention=ar.cross_attention,
        )

    def _probe(
        self,
        latents: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar: _ARBlockContext | None,
        start_frame: int,
        step_index: int,
    ) -> torch.Tensor:
        """Predict flow for one DMD step of the current block."""
        if not self.od_config.enforce_eager:
            torch.compiler.cudagraph_mark_step_begin()
        set_forward_context_denoise_step_idx(step_index)
        timestep = torch.full((1,), self._schedule[step_index][0], device=self.device, dtype=torch.float32)
        return self.transformer(
            latents.to(dtype=self.transformer.dtype),
            timestep,
            prompt_embeds,
            cache=self._block_cache(cache=cache, ar=ar, seq_len=self._block_tokens(latents), commit_current=False),
            start_frame=start_frame,
            update_cache=False,
        )

    def _transition(
        self,
        latents: torch.Tensor,
        flow: torch.Tensor,
        step_index: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Invert the flow to ``x0`` and re-noise it at the next DMD sigma."""
        x0 = latents - self._schedule[step_index][1] * flow.float()
        if step_index + 1 == len(self._schedule):
            return x0
        next_sigma = self._schedule[step_index + 1][1]
        noise = randn_tensor(latents.shape, generator=generator, device=self.device, dtype=torch.float32)
        return (1.0 - next_sigma) * x0 + next_sigma * noise

    def _commit(
        self,
        latents: torch.Tensor,
        *,
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar: _ARBlockContext | None,
        start_frame: int,
    ) -> None:
        """Write the finished block's clean K/V with the ``t=0`` pass."""
        if not self.od_config.enforce_eager:
            torch.compiler.cudagraph_mark_step_begin()
        self.transformer(
            latents.to(dtype=self.transformer.dtype),
            torch.zeros(1, device=self.device, dtype=torch.float32),
            prompt_embeds,
            cache=self._block_cache(cache=cache, ar=ar, seq_len=self._block_tokens(latents), commit_current=True),
            start_frame=start_frame,
            update_cache=True,
        )
        if ar is not None:
            ar.state.commit_paged_context(self._AR_BRANCH)

    def _block_tokens(self, latents: torch.Tensor) -> int:
        return latents.shape[2] * self.transformer.tokens_per_frame(latents.shape[-2], latents.shape[-1])

    def _generate_block(
        self,
        *,
        latent_shape: tuple[int, int],
        prompt_embeds: torch.Tensor,
        cache: LingBotTransformerCache | None,
        ar: _ARBlockContext | None,
        start_frame: int,
        generator: torch.Generator,
        progress_bar: TqdmProgressBar[Any],
    ) -> torch.Tensor:
        """Request mode: all DMD probes of one block, then its commit."""
        shape = (1, self.transformer.config.out_channels, self.transformer.config.num_frames_per_block, *latent_shape)
        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=torch.float32)
        for step_index in range(len(self._schedule)):
            flow = self._probe(
                latents,
                prompt_embeds=prompt_embeds,
                cache=cache,
                ar=ar,
                start_frame=start_frame,
                step_index=step_index,
            )
            latents = self._transition(latents, flow, step_index, generator=generator)
            progress_bar.update()
        self._commit(latents, prompt_embeds=prompt_embeds, cache=cache, ar=ar, start_frame=start_frame)
        return latents

    # ── request mode ──────────────────────────────────────────────────────

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if (
            ARDiffusionTickRequest.from_extra_args(req.sampling_params.extra_args) is not None
            or self._ar_diffusion_kv_state is not None
        ):
            raise ValueError(
                "Self-Forcing streams through step execution (vllm_omni/deploy/self_forcing_stepwise.yaml); "
                "AR-Diffusion ticks are not supported."
            )
        inputs = self._parse_request(req)
        dtype = self.transformer.dtype
        prompt_embeds, _ = self.encode_prompt(inputs.prompt, max_sequence_length=_MAX_SEQUENCE_LENGTH, dtype=dtype)
        latent_shape = (
            inputs.height // self.vae_scale_factor_spatial,
            inputs.width // self.vae_scale_factor_spatial,
        )
        cache = self.transformer.allocate_cache(
            latent_height=latent_shape[0], latent_width=latent_shape[1], device=self.device, dtype=dtype
        )
        block_frames = self.transformer.config.num_frames_per_block
        num_blocks = inputs.num_latent_frames // block_frames
        blocks: list[torch.Tensor] = []
        with self.progress_bar(total=num_blocks * len(self._schedule)) as progress_bar:
            for block_index in range(num_blocks):
                blocks.append(
                    self._generate_block(
                        latent_shape=latent_shape,
                        prompt_embeds=prompt_embeds,
                        cache=cache,
                        ar=None,
                        start_frame=block_index * block_frames,
                        generator=inputs.generator,
                        progress_bar=progress_bar,
                    )
                )
        latents = torch.cat(blocks, dim=2)
        # Release the request-local K/V before the decoder allocates.
        del cache
        if inputs.output_type == "latent":
            output = latents
        else:
            output = self.vae.decode(self._denormalize_latents(latents), return_dict=False)[0]
        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    # ── decoding ──────────────────────────────────────────────────────────

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1, 1, 1)
        mean = torch.as_tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(*shape)
        std = torch.as_tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(*shape)
        return (latents * std + mean).to(dtype=self.vae.dtype)

    def _streaming_decoder(self) -> WanStreamingDecoder:
        if self._streaming_decoder_instance is None:
            self._streaming_decoder_instance = WanStreamingDecoder(
                self.vae, bytes_per_pixel_fp32=_STREAMING_DECODE_BYTES_PER_PIXEL_FP32
            )
        return self._streaming_decoder_instance

    def _vae_would_tile(self, latents: torch.Tensor) -> bool:
        if not bool(getattr(self.vae, "use_tiling", False)):
            return False
        compression = int(getattr(self.vae, "spatial_compression_ratio", None) or self.vae_scale_factor_spatial)
        min_height = int(getattr(self.vae, "tile_sample_min_height", None) or 0) // compression
        min_width = int(getattr(self.vae, "tile_sample_min_width", None) or 0) // compression
        return bool(latents.shape[-2] > min_height or latents.shape[-1] > min_width)

    def _release_streaming_decode_state(self, session_id: str) -> None:
        state = self._streaming_decode_states.pop(session_id, None)
        if state is not None:
            state.release()

    def _decode_chunk_to_pixels(self, latents: torch.Tensor, *, session_id: str) -> np.ndarray:
        """Decode one block through the session's temporal cache so chunks join seamlessly."""
        vae_latents = self._denormalize_latents(latents)
        if self._vae_would_tile(vae_latents):
            return _uint8_frames(self.vae.decode(vae_latents, return_dict=False)[0])
        decoder = self._streaming_decoder()
        state = self._streaming_decode_states.get(session_id)
        if state is None:
            state = decoder.new_decode_state(session_id)
            self._streaming_decode_states[session_id] = state
        try:
            return _uint8_frames(decoder.decode_chunk(vae_latents, state))
        except Exception:
            # A half-advanced temporal cache cannot be resumed.
            self._release_streaming_decode_state(session_id)
            raise

    # ── step execution (streaming sessions) ──────────────────────────────

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        self._require_bound_ar_state()
        req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(prompt=state.prompt, sampling_params=state.sampling, request_id=state.request_id)
            ]
        )
        inputs = self._parse_request(req)
        if ARDiffusionTickRequest.from_extra_args(state.sampling.extra_args) is not None:
            raise ValueError("Self-Forcing step execution does not accept AR-Diffusion ticks.")
        if (inputs.height, inputs.width) != (self._ar_height, self._ar_width):
            raise ValueError(
                f"Self-Forcing step execution resolution must match the cache geometry "
                f"{self._ar_height}x{self._ar_width}."
            )
        block_frames = self.transformer.config.num_frames_per_block
        state.prompt_embeds, _ = self.encode_prompt(
            inputs.prompt, max_sequence_length=_MAX_SEQUENCE_LENGTH, dtype=self.transformer.dtype
        )
        state.chunk_index = 0
        state.step_index = 0
        state.step_in_chunk = 0
        state.total_chunks = inputs.num_latent_frames // block_frames
        state.chunk_num_steps = len(self._schedule)
        state.extra = {"inputs": inputs}
        self._ar_sessions[state.request_id] = _SelfForcingSessionState()
        return state

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        """Draw the next block's noise and refresh text K/V after any prompt update."""
        if state.request_denoise_completed:
            return
        inputs: _SelfForcingRequestInputs = state.extra["inputs"]
        session_state = self._ar_sessions[state.request_id]
        if session_state.next_chunk_index != state.chunk_index:
            raise ValueError("Self-Forcing chunks must be contiguous within a session.")
        # Every prompt-update boundary that changes ``state.prompt_embeds``
        # (including each blended transition step) bumps the session version.
        state.extra["ar_cross_attention"] = self._session_text_caches(
            cast(torch.Tensor, state.prompt_embeds), prompt_update_versions([state]), session_state
        )
        block_frames = self.transformer.config.num_frames_per_block
        state.extra["start_frame"] = state.chunk_index * block_frames
        state.latents = randn_tensor(
            (
                1,
                self.transformer.config.out_channels,
                block_frames,
                inputs.height // self.vae_scale_factor_spatial,
                inputs.width // self.vae_scale_factor_spatial,
            ),
            generator=inputs.generator,
            device=self.device,
            dtype=torch.float32,
        )
        state.timesteps = torch.tensor([timestep for timestep, _ in self._schedule], device=self.device)
        state.step_in_chunk = 0
        state.step_index = 0

    def _step_block_context(self, state: StepRequestState) -> _ARBlockContext:
        # The runner binds the KV state per invocation, so it is never kept on
        # the request state; only the text K/V views of this chunk are.
        return _ARBlockContext(state=self._require_bound_ar_state(), cross_attention=state.extra["ar_cross_attention"])

    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del input_batch, kwargs
        if states is None or len(states) != 1:
            raise ValueError("Self-Forcing step execution supports a single request, not a batch.")
        state = states[0]
        if state.latents is None:
            raise RuntimeError("Self-Forcing step execution requires prepared latents before denoise_step.")
        return self._probe(
            state.latents,
            prompt_embeds=cast(torch.Tensor, state.prompt_embeds),
            cache=None,
            ar=self._step_block_context(state),
            start_frame=int(state.extra["start_frame"]),
            step_index=state.step_in_chunk,
        )

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        if state.latents is None:
            raise RuntimeError("Self-Forcing step execution requires latents before step_scheduler.")
        inputs: _SelfForcingRequestInputs = state.extra["inputs"]
        state.latents = self._transition(state.latents, noise_pred, state.step_in_chunk, generator=inputs.generator)
        state.step_in_chunk += 1
        state.step_index = state.step_in_chunk

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        if state.latents is None:
            raise RuntimeError("Self-Forcing step execution requires latents before post_decode.")
        self._commit(
            state.latents,
            prompt_embeds=cast(torch.Tensor, state.prompt_embeds),
            cache=None,
            ar=self._step_block_context(state),
            start_frame=int(state.extra["start_frame"]),
        )
        inputs: _SelfForcingRequestInputs = state.extra["inputs"]
        completed_chunk_index = state.chunk_index
        if inputs.output_type == "latent":
            payload: dict[str, Any] = {"latents": state.latents}
        else:
            payload = {"video": self._decode_chunk_to_pixels(state.latents, session_id=state.request_id)}
        metadata = ARDiffusionChunkMetadata(
            session_id=state.request_id,
            request_id=state.request_id,
            chunk_index=completed_chunk_index,
        )
        self._ar_sessions[state.request_id].next_chunk_index += 1
        state.chunk_index += 1
        return DiffusionOutput(
            output={"payload": payload, "metadata": {"ar_diffusion": metadata.to_dict()}},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            chunk_index=completed_chunk_index,
            total_chunks=state.total_chunks,
            finished=state.request_denoise_completed,
        )

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        """Report the next chunk's decoded frame count for interaction pacing."""
        block_frames = self.transformer.config.num_frames_per_block
        decode_state = self._streaming_decode_states.get(state.request_id)
        if decode_state is not None and decode_state.started:
            num_media_frames = block_frames * self.vae_scale_factor_temporal
        else:
            num_media_frames = (block_frames - 1) * self.vae_scale_factor_temporal + 1
        fps = state.sampling.fps
        return ChunkMediaSpec(
            num_media_frames=num_media_frames,
            fps=float(fps) if fps else float(SELF_FORCING_FPS),
            num_latent_frames=block_frames,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return cast(set[str], AutoWeightsLoader(self).load_weights(weights))


def _validate_parallel_config(od_config: OmniDiffusionConfig) -> None:
    """Reject distributed layouts this single-GPU port has not been validated with."""
    parallel_config = getattr(od_config, "parallel_config", None)
    if parallel_config is None:
        return
    for field in (
        "tensor_parallel_size",
        "sequence_parallel_size",
        "ulysses_degree",
        "ring_degree",
        "cfg_parallel_size",
        "pipeline_parallel_size",
        "vae_patch_parallel_size",
    ):
        size = getattr(parallel_config, field, 1) or 1
        if size > 1:
            raise NotImplementedError(f"Self-Forcing runs on one GPU; {field}={size} is not supported.")
    if getattr(parallel_config, "use_hsdp", False):
        raise NotImplementedError("Self-Forcing does not support HSDP.")
