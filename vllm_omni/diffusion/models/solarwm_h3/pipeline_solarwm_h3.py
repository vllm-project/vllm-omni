# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native camera-conditioned, autoregressive SolarWM-H3 Stage2 inference."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.interaction.types import ChunkMediaSpec
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    _broadcast_rank0_exception,
    _broadcast_tensor,
    _dit_rank_world,
    _SingleRankEncoderGroup,
)
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionKVCacheSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.raw import RawKVWindow
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionChunkMetadata
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.minimax_h3.preprocessing import minimax_h3_multi_image_presentation

from .attention import SolarAttentionContext
from .cache import SolarRawKVCache
from .checkpoint import install_static_adapter, load_stage2_adapter
from .decode import SolarH3DecodeState, SolarH3StreamingDecoder
from .layout import H3PackedLayout, build_layout, patchify_video, unpatchify_video
from .request import SolarRequest, parse_request
from .sampling import sample_audio_timestep, sample_chunk, scale_noise, stage2_timesteps
from .transformer import SolarWMH3Transformer
from .vae import SolarH3VideoVAE

_BASE = "SolarWM-h3-33B-base"
_STAGE2 = "SolarWM-h3-33B-sgf-stage2-158f-fix"


@dataclass
class _SolarSession:
    inputs: SolarRequest
    prompt: torch.Tensor
    anchor: torch.Tensor
    audio: torch.Tensor
    audio_t: torch.Tensor
    layout: H3PackedLayout
    generator: torch.Generator
    initial_noise: torch.Tensor
    decode: SolarH3DecodeState
    cache: SolarRawKVCache | None
    next_chunk: int = 0


class SolarWMH3Pipeline(nn.Module, SupportImageInput, SupportsComponentDiscovery):
    """BF16 SGF baseline with bounded clean history and temporal video decode.

    A normal request walks the whole trajectory. AR ticks execute one chunk;
    step execution emits decoded segments as soon as their lookahead is ready.
    The Qwen encoder lives on CPU between conditioning calls; it is never
    re-encoded for a continuation of the same session.
    """

    supports_step_execution: ClassVar[bool] = True
    supports_chunk_step_grouping: ClassVar[bool] = True
    dummy_run_num_frames: ClassVar[int] = 0
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.parallel_config = od_config.parallel_config
        if self.parallel_config.data_parallel_size not in (None, 1):
            raise ValueError("SolarWM-H3 currently requires data_parallel_size=1")
        if od_config.quantization_config is not None:
            raise ValueError("SolarWM-H3 requires unquantized BF16 weights")
        if od_config.cache_backend not in (None, "none"):
            raise ValueError("SolarWM-H3 does not support approximate DiT caching")
        if not od_config.enforce_eager:
            raise ValueError("SolarWM-H3 currently requires enforce_eager=true")
        if int(self.parallel_config.text_encoder_tp_size) != 1:
            raise ValueError("SolarWM-H3 currently requires text_encoder_tp_size=1")
        if resolve_offload(od_config).strategy is not OffloadStrategy.NONE:
            raise ValueError("SolarWM-H3 does not support configured model/layer CPU offload")
        if od_config.max_num_seqs != 1:
            raise ValueError("SolarWM-H3 requires max_num_seqs=1")
        if od_config.step_execution and not od_config.streaming_output:
            raise ValueError("SolarWM-H3 step execution requires streaming_output=true")
        if od_config.dtype != torch.bfloat16:
            raise ValueError("SolarWM-H3 requires dtype=bfloat16")
        if not all(od_config.model_loaded.get(name, True) for name in ("text_encoder", "vae_encoder", "vae_decoder")):
            raise ValueError("SolarWM-H3 currently requires local text and VAE components")
        config = od_config.model_config or {}
        self.height = int(config.get("ar_diffusion_height", 768))
        self.width = int(config.get("ar_diffusion_width", 1344))
        if min(self.height, self.width) <= 0 or self.height % 32 or self.width % 32:
            raise ValueError("SolarWM-H3 output dimensions must be positive multiples of 32")
        self.max_text_tokens = int(config.get("solarwm_max_text_tokens", 4096))
        self.session_capacity = int(config.get("ar_diffusion_session_capacity", 1))
        if self.max_text_tokens <= 0 or self.session_capacity <= 0:
            raise ValueError("SolarWM-H3 text/session limits must be positive")
        model_root = Path(od_config.model)
        if not model_root.is_dir():
            model_root = Path(
                download_weights_from_hf_specific(
                    model_name_or_path=str(od_config.model),
                    cache_dir=None,
                    require_all=True,
                    revision=od_config.revision,
                    allow_patterns=[
                        f"{_BASE}/**",
                        f"{_STAGE2}/**",
                        "transformer/*",
                        "text_encoder/*",
                        "tokenizer/*",
                        "processor/*",
                        "vae/*",
                        "audio_vae/*",
                    ],
                )
            )
        base = model_root / _BASE if (model_root / _BASE).is_dir() else model_root
        checkpoint = config.get("solarwm_checkpoint_path") or model_root / _STAGE2
        self._adapter = load_stage2_adapter(checkpoint)
        od_config.tf_model_config = TransformerConfig.from_dict(
            json.loads((base / "transformer" / "config.json").read_text())
        )
        # AdaLN's exact projection cache is compatible; automatic projection
        # offload would mutate native weights that our fixed sidecars use.
        od_config.cache_config.minimax_h3_adaln_offload = False
        self.transformer = SolarWMH3Transformer(od_config, diffusers_weights=True)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(base),
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        _, self._rank, _ = _dit_rank_world()
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(
            str(base / "text_encoder"),
            device=self.device,
            load_model=self._rank == 0,
            encoder_group=_SingleRankEncoderGroup(self._rank),
        )
        self.video_vae = None
        self.decoder = None
        self.tokenizer = self.processor = None
        self.register_buffer("silence", torch.empty(0, dtype=torch.bfloat16), persistent=False)
        error = None
        silence = None
        if self._rank == 0:
            try:
                from diffusers import AutoencoderKLMiniMaxH3
                from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

                self.tokenizer = Qwen2TokenizerFast.from_pretrained(str(base / "tokenizer"))
                self.processor = Qwen3VLProcessor.from_pretrained(str(base / "processor"))
                vae = AutoencoderKLMiniMaxH3.from_pretrained(str(base / "vae"), torch_dtype=torch.float32)
                vae.set_attention_backend("native")
                self.video_vae = SolarH3VideoVAE(vae.to(self.device).eval())
                self.decoder = SolarH3StreamingDecoder(self.video_vae, device=self.device)
                silence = self._load_silence(base, config.get("solarwm_silence_path"))
                self.weights_sources.append(
                    DiffusersPipelineLoader.ComponentSource(
                        model_or_path=str(base),
                        subfolder="text_encoder",
                        revision=od_config.revision,
                        prefix="text_encoder.",
                        fall_back_to_pt=False,
                    )
                )
            except Exception as exc:
                error = exc
        _broadcast_rank0_exception(error)
        self.silence = _broadcast_tensor(silence, dtype=torch.bfloat16, device=self.device)
        self._sessions: dict[str, _SolarSession] = {}
        self._ar_state: ARDiffusionKVState | None = None

    def _load_silence(self, base: Path, artifact: str | None) -> torch.Tensor:
        if artifact:
            from safetensors.torch import load_file

            tensor = load_file(str(artifact), device="cpu").get("silence_158f")
        else:
            from diffusers import AutoencoderKLMiniMaxH3Audio

            audio_vae = (
                AutoencoderKLMiniMaxH3Audio.from_pretrained(
                    str(base / "audio_vae"),
                    torch_dtype=torch.float32,
                )
                .eval()
                .to(self.device)
            )
            audio_vae.set_attention_backend("native")
            with torch.inference_mode():
                waveform = torch.zeros((2, 1, 263 * 800), device=self.device, dtype=torch.float32)
                tensor = audio_vae.encode(waveform, return_dict=False)[0].mode().float().cpu()
            mean = torch.tensor(audio_vae.config.latents_mean, dtype=torch.float32, device="cpu").view(1, 32, 1)
            std = torch.tensor(audio_vae.config.latents_std, dtype=torch.float32, device="cpu").view(1, 32, 1)
            tensor = ((tensor - mean) / std).to(torch.bfloat16)
            del audio_vae
        if (
            tensor is None
            or tensor.shape != (2, 32, 263)
            or tensor.dtype != torch.bfloat16
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("SolarWM-H3 silence must be finite BF16 silence_158f [2,32,263]")
        return tensor.contiguous()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        seen: set[str] = set()
        for prefix, stream in groupby(weights, key=lambda item: item[0].partition(".")[0]):
            if prefix not in ("transformer", "text_encoder") or prefix in seen:
                raise ValueError(f"unexpected or noncontiguous SolarWM-H3 source {prefix!r}")
            seen.add(prefix)
            component = getattr(self, prefix)
            expected = set(dict(component.named_parameters()))
            names = component.load_weights((name[len(prefix) + 1 :], value) for name, value in stream)
            if missing := expected - names:
                raise ValueError(f"SolarWM-H3 {prefix} parameters were not loaded: {sorted(missing)[:8]}")
            loaded.update(f"{prefix}.{name}" for name in names)
        required = {"transformer", "text_encoder"} if self._rank == 0 else {"transformer"}
        if seen != required:
            raise ValueError(f"SolarWM-H3 weight sources differ: expected {required}, got {seen}")
        self.transformer.post_load_weights()
        install_static_adapter(self.transformer, self._adapter)
        self._adapter = None
        if self._rank == 0:
            self.text_encoder.offload_to_cpu()
            loaded.update(f"video_vae.{name}" for name, _ in self.video_vae.named_parameters())
        return loaded

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        heads = self.transformer.arch.num_attention_heads // int(self.parallel_config.ulysses_degree or 1)
        area = (self.height // 32) * (self.width // 32)
        # Prompt, anchor, fixed audio, current FP32 sample and decoder overlap.
        # Full camera metadata stays on CPU, independent of HBM admission.
        owned = 2 * (self.max_text_tokens * 5120 + area * 96 + 526 * 32)
        owned += 4 * 24 * 5 * (self.height // 16) * (self.width // 16)
        owned += 4 * (6 * 24 * (self.height // 16) * (self.width // 16) + 5 * 3 * self.height * self.width)
        # Bound layout/index tensors and condition FP32 promotion conservatively.
        owned += 8 * 1024**2
        # The exact projection cache is shared by serial sessions, but charging
        # its complete cap to each session also covers cold-cache admission.
        owned += self.transformer.adaln_cache.max_bytes if self.transformer.adaln_cache is not None else 0
        return ARDiffusionKVCacheSpec(
            num_layers=self.transformer.arch.num_layers,
            num_kv_heads=heads,
            head_size=128,
            tokens_per_frame=area,
            frames_per_block=5,
            window_frames=25,
            kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
            session_capacity=self.session_capacity,
            model_owned_state_bytes_per_session=owned,
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id: str, state: ARDiffusionKVState) -> Iterator[None]:
        if self._ar_state is not None or state.session_id != session_id:
            raise ValueError("SolarWM-H3 AR state is already bound or has a different session_id")
        self._ar_state = state
        try:
            yield
        finally:
            self._ar_state = None

    def close_ar_diffusion_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.decode.close()
            if session.cache is not None:
                session.cache.clear()

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self.close_ar_diffusion_session(session_id)

    def _new_session(self, session_id: str, inputs: SolarRequest) -> _SolarSession:
        prompt = anchor = tags = None
        error = None
        if self._rank == 0:
            try:
                processed = self.processor.image_processor(images=[inputs.image], return_tensors="pt")
                grid = processed["image_grid_thw"]
                count = int(grid[0].prod()) // int(self.processor.image_processor.merge_size) ** 2
                ids, tags = minimax_h3_multi_image_presentation(
                    self.tokenizer,
                    prompt=inputs.prompt,
                    image_token_counts=[count],
                )
                if len(ids) > self.max_text_tokens:
                    raise ValueError(f"SolarWM-H3 joint image/text prefix exceeds {self.max_text_tokens} tokens")
                self.text_encoder.load_to_device()
                try:
                    prompt = self.text_encoder.encode_ids(
                        ids,
                        pixel_values=processed["pixel_values"],
                        image_grid_thw=grid,
                    )
                finally:
                    self.text_encoder.offload_to_cpu()
                anchor = self.video_vae.encode_image(inputs.image, height=self.height, width=self.width)
            except Exception as exc:
                error = exc
        _broadcast_rank0_exception(error)
        prompt = _broadcast_tensor(prompt, dtype=torch.bfloat16, device=self.device)
        anchor = _broadcast_tensor(anchor, dtype=torch.bfloat16, device=self.device).reshape(1, -1, 96)
        tags = _broadcast_tensor(tags, dtype=torch.long, device=self.device)
        layout = build_layout(tags, latent_height=self.height // 16, latent_width=self.width // 16, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(inputs.seed)
        anchor = unpatchify_video(anchor, 1, self.height // 16, self.width // 16).float()
        anchor_noise = torch.randn(anchor.shape, device=self.device, dtype=torch.float32, generator=generator)
        anchor = patchify_video(scale_noise(anchor, anchor_noise, 0.999))
        audio_t = sample_audio_timestep(device=self.device, generator=generator)
        silence = self.silence.permute(0, 2, 1).reshape(1, 526, 32).float()
        audio_noise = torch.randn(silence.shape, device=self.device, dtype=torch.float32, generator=generator)
        audio = scale_noise(silence, audio_noise, audio_t)
        # Draw the complete initial trajectory before any stochastic SGF step,
        # preserving the reference RNG order and channel/time memory layout.
        # Host storage keeps persistent CUDA state independent of the horizon.
        initial_noise = torch.randn(
            (1, 24, inputs.camera.geometry.rollout_latents, self.height // 16, self.width // 16),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        ).cpu()
        return _SolarSession(
            inputs,
            prompt,
            anchor,
            audio,
            audio_t,
            layout,
            generator,
            initial_noise,
            SolarH3DecodeState(session_id, max_frames=inputs.num_frames),
            None if self._ar_state is not None else SolarRawKVCache(self.transformer.arch.num_layers),
        )

    def _session(self, session_id: str, inputs: SolarRequest) -> _SolarSession:
        session = self._sessions.get(session_id)
        if session is None:
            session = self._new_session(session_id, inputs)
            self._sessions[session_id] = session
        else:
            previous = session.inputs
            if (inputs.prompt, inputs.image_digest, inputs.num_frames, inputs.seed, inputs.output_type, inputs.fps) != (
                previous.prompt,
                previous.image_digest,
                previous.num_frames,
                previous.seed,
                previous.output_type,
                previous.fps,
            ):
                raise ValueError("SolarWM-H3 session prompt, anchor, horizon, seed and output settings are fixed")
            committed = 1 + session.next_chunk * 5
            if not torch.equal(inputs.camera.viewmats[:, :committed], previous.camera.viewmats[:, :committed]):
                raise ValueError("SolarWM-H3 camera controls cannot change already committed poses")
            if not torch.equal(inputs.camera.intrinsics[:, :committed], previous.camera.intrinsics[:, :committed]):
                raise ValueError("SolarWM-H3 camera controls cannot change already committed intrinsics")
            session.inputs = inputs
        return session

    def _cache(self, session: _SolarSession) -> SolarRawKVCache | RawKVWindow:
        if self._ar_state is not None:
            return RawKVWindow(self._ar_state, "main", frames_per_chunk=5, history_chunks=5)
        if session.cache is None:
            raise RuntimeError("SolarWM-H3 runner session must be bound for every invocation")
        return session.cache

    def _predict(
        self,
        session: _SolarSession,
        latent: torch.Tensor,
        time: torch.Tensor,
        *,
        cache: SolarRawKVCache | RawKVWindow,
        commit: bool = False,
    ) -> torch.Tensor:
        context = SolarAttentionContext(
            session.layout,
            session.inputs.camera.viewmats,
            session.inputs.camera.intrinsics,
            session.next_chunk,
            cache,
            commit_cache=commit,
        )
        return self.transformer.predict_video(
            session.prompt,
            session.anchor,
            latent,
            session.audio,
            session.layout,
            time,
            session.audio_t,
            context,
        )

    def _noise(self, session: _SolarSession) -> torch.Tensor:
        return torch.randn(
            (1, 24, 5, self.height // 16, self.width // 16),
            device=self.device,
            dtype=torch.float32,
            generator=session.generator,
        )

    def _commit(self, session: _SolarSession, latent: torch.Tensor) -> None:
        cache = self._cache(session)
        try:
            self._predict(session, latent, latent.new_tensor(1.0), cache=cache, commit=True)
            cache.commit_chunk(session.next_chunk)
        except BaseException:
            cache.abort()
            raise

    def _generate(self, session: _SolarSession) -> torch.Tensor:
        cache = self._cache(session)
        return sample_chunk(
            lambda latent, time: self._predict(session, latent, time, cache=cache),
            self._initial_noise(session),
            commit=lambda latent, time: self._commit(session, latent),
            generator=session.generator,
            chunk_index=session.next_chunk,
        )

    def _initial_noise(self, session: _SolarSession) -> torch.Tensor:
        start = session.next_chunk * 5
        return session.initial_noise[:, :, start : start + 5].to(self.device).contiguous()

    def _decode(self, session: _SolarSession, latent: torch.Tensor, *, final: bool) -> torch.Tensor | None:
        if self._rank != 0 or session.inputs.output_type == "latent":
            return None
        # The 158-frame release samples 50 target latents, but decodes 47.
        remaining = session.inputs.camera.geometry.decode_latents - session.next_chunk * 5
        visible = latent[:, :, : min(5, remaining)]
        first = self.decoder.push(session.decode, visible)
        last = self.decoder.flush(session.decode) if final else None
        values = [value for value in (first, last) if value is not None]
        return torch.cat(values, dim=2) if values else None

    def _chunk_output(
        self,
        session_id: str,
        session: _SolarSession,
        latent: torch.Tensor,
        *,
        request_id: str,
        event_ids: tuple[int, ...] = (),
    ) -> DiffusionOutput:
        index = session.next_chunk
        total = session.inputs.camera.geometry.rollout_latents // 5
        final = index + 1 == total
        pixels = self._decode(session, latent, final=final)
        payload: dict[str, Any] = {}
        if session.inputs.output_type == "latent":
            remaining = session.inputs.camera.geometry.decode_latents - index * 5
            payload["latents"] = latent[:, :, : min(5, remaining)]
        elif pixels is not None:
            payload["video"] = [self._pixels(pixels, session.inputs.output_type)]
        session.next_chunk += 1
        return DiffusionOutput(
            output={
                "payload": payload,
                "metadata": {
                    "video": {"fps": session.inputs.fps},
                    "ar_diffusion": ARDiffusionChunkMetadata(
                        session_id=session_id,
                        request_id=request_id,
                        chunk_index=index,
                        applied_event_ids=event_ids,
                    ).to_dict(),
                },
            },
            chunk_index=index,
            total_chunks=total,
            finished=final,
        )

    @staticmethod
    def _pixels(pixels: torch.Tensor, output_type: str) -> Any:
        # Match the existing video formatter's unbatched THWC payload.
        video = pixels[0].permute(1, 2, 3, 0).contiguous().cpu()
        if output_type == "pt":
            return video
        return (video.clamp(0, 1).mul(255).round().to(torch.uint8)).numpy()

    @torch.inference_mode()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if request.num_reqs != 1:
            raise ValueError("SolarWM-H3 supports one request at a time")
        inputs = parse_request(request.prompts[0], request.sampling_params, height=self.height, width=self.width)
        tick = inputs.tick
        if tick is not None:
            if self._ar_state is None or self._ar_state.session_id != tick.session_id:
                raise ValueError("SolarWM-H3 ticks require ARDiffusionEngine session binding")
            session = self._session(tick.session_id, inputs)
            if tick.chunk_index != session.next_chunk:
                raise ValueError(f"SolarWM-H3 expected chunk_index={session.next_chunk}, got {tick.chunk_index}")
            if session.next_chunk >= inputs.camera.geometry.rollout_latents // 5:
                raise ValueError("SolarWM-H3 trajectory is complete; reset before another rollout")
            return self._chunk_output(
                tick.session_id,
                session,
                self._generate(session),
                request_id=tick.request_id,
                event_ids=tick.applied_event_ids,
            )
        if self._ar_state is not None:
            raise ValueError("SolarWM-H3 full rollouts require AR step execution or the standard diffusion engine")
        session_id = request.request_id
        session = self._session(session_id, inputs)
        videos, latents = [], []
        try:
            total = inputs.camera.geometry.rollout_latents // 5
            while session.next_chunk < total:
                chunk = self._generate(session)
                final = session.next_chunk + 1 == total
                if inputs.output_type == "latent":
                    latents.append(chunk.cpu())
                else:
                    pixels = self._decode(session, chunk, final=final)
                    if pixels is not None:
                        videos.append(pixels.cpu())
                session.next_chunk += 1
            payload = (
                {"latents": torch.cat(latents, dim=2)[:, :, : inputs.camera.geometry.decode_latents]}
                if inputs.output_type == "latent"
                else {"video": [self._pixels(torch.cat(videos, dim=2), inputs.output_type)]}
                if videos
                else {}
            )
            return DiffusionOutput(output={"payload": payload, "metadata": {"video": {"fps": inputs.fps}}})
        finally:
            self.close_ar_diffusion_session(session_id)

    @torch.inference_mode()
    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        if self._ar_state is None:
            raise RuntimeError("SolarWM-H3 step execution requires ARDiffusionEngine")
        inputs = parse_request(state.prompt, state.sampling, height=self.height, width=self.width)
        if inputs.tick is not None:
            raise ValueError("SolarWM-H3 typed ticks use request execution, not step execution")
        session_id = self._ar_state.session_id
        session = self._session(session_id, inputs)
        if session.next_chunk != 0:
            raise ValueError("SolarWM-H3 stepwise rollout requires a new or reset session")
        state.extra["solar_session_id"] = session_id
        state.total_chunks = inputs.camera.geometry.rollout_latents // 5
        state.chunk_index = state.step_index = 0
        state.chunk_num_steps = 4
        state.timesteps = stage2_timesteps(self.device).repeat(state.total_chunks)
        self.prepare_next_chunk(state)
        return state

    @torch.inference_mode()
    def denoise_step(self, input_batch: Any, *, states: Sequence[StepRequestState], **kwargs: Any) -> torch.Tensor:
        del input_batch, kwargs
        if len(states) != 1:
            raise ValueError("SolarWM-H3 step execution requires one active request")
        state = states[0]
        session = self._sessions[state.extra["solar_session_id"]]
        return self._predict(session, state.latents, state.current_timestep, cache=self._cache(session))

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        session = self._sessions[state.extra["solar_session_id"]]
        clean = state.latents + (1.0 - state.current_timestep) * noise_pred.float()
        if not bool(torch.isfinite(clean).all()):
            raise FloatingPointError("non-finite SolarWM-H3 stepwise clean sample")
        state.step_index += 1
        state.step_in_chunk += 1
        if state.step_in_chunk < 4:
            state.latents = scale_noise(clean, self._noise(session), state.current_timestep)
        else:
            state.latents = clean

    @torch.inference_mode()
    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        session_id = state.extra["solar_session_id"]
        session = self._sessions[session_id]
        self._commit(session, state.latents)
        output = self._chunk_output(session_id, session, state.latents, request_id=state.request_id)
        state.chunk_index += 1
        return output

    def prepare_next_chunk(self, state: StepRequestState) -> None:
        if not state.request_denoise_completed:
            session = self._sessions[state.extra["solar_session_id"]]
            state.latents = self._initial_noise(session)
            state.step_in_chunk = 0

    def peek_chunk_media(self, state: StepRequestState) -> ChunkMediaSpec:
        # This positive-only interaction contract describes the nominal advance.
        # Solar registers no interaction handlers: typed camera ticks own updates.
        # Temporal lookahead emits zero frames first and adds the final overlap
        # on flush, so this value must not be used to pace decoded output.
        session = self._sessions[state.extra["solar_session_id"]]
        return ChunkMediaSpec(num_media_frames=17, fps=session.inputs.fps, num_latent_frames=5)

    def apply_interaction_at_chunk_boundary(self, state: StepRequestState) -> None:
        if state.interaction_sessions:
            raise ValueError("SolarWM-H3 camera changes require versioned AR ticks")
