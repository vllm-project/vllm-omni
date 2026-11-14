# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resident Realization video/audio codecs with explicit model generation geometry.

The duo_5_17 mode uses five video codes and 29 audio codes per block,
decoded immediately with native temporal boundary padding. Legacy
single-code decoding remains an explicit, separate compatibility mode.
"""

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from vllm_omni.model_executor.models.duo.duplex.av_output import encode_av_chunk
from vllm_omni.platforms import current_omni_platform

from .condition import RealizationConditionTicket
from .vae_bridge import cache_bf16_decoder_weights, load_codec, set_tile_group


def block_clock(block, frames=4):
    if type(block) is not int or block < 0:
        raise ValueError("Invalid codec block clock")
    start = (block * frames * 32000 + 23) // 24
    stop = ((block + 1) * frames * 32000 + 23) // 24
    return dict(
        frame_start=block * frames,
        frame_stop=(block + 1) * frames,
        audio_start_sample=start,
        audio_stop_sample=stop,
        sample_rate=32000,
        fps=24,
    )


@dataclass(frozen=True)
class DuoCodecBlock:
    session_id: str
    fence: int
    block: int
    video: torch.Tensor  # normalized [24,video_codes,H/16,W/16]
    audio: torch.Tensor  # normalized [2,32,audio_codes]
    reference: torch.Tensor | None = None
    final: bool = False
    tenant: str = "local"
    dispatch_ns: int | None = None
    codec_mode: str = "stream_4frame"
    reset_media_cache: bool = False


def validate_codec_block(c):
    if not isinstance(c, DuoCodecBlock):
        raise ValueError("Native codec needs typed DuoCodecBlock")
    RealizationConditionTicket(c.session_id, c.fence, c.block, c.tenant)
    if type(c.final) is not bool:
        raise ValueError("Invalid codec final flag")
    if c.codec_mode not in ("stream_4frame", "old", "duo_5_17"):
        raise ValueError("Unknown codec geometry")
    nv, na = (5, 29) if c.codec_mode == "duo_5_17" else (1, 7)
    if (
        c.video.ndim != 4
        or tuple(c.video.shape[:2]) != (24, nv)
        or any(s < 2 or s % 2 for s in c.video.shape[2:])
        or tuple(c.audio.shape) != (2, 32, na)
    ):
        raise ValueError("Invalid Realization codec latent geometry")
    if c.block == 0:
        if c.reference is None or c.reference.shape != (24, 1, *c.video.shape[-2:]):
            raise ValueError("First codec block requires matching reference latent")
    elif c.reference is not None:
        raise ValueError("Codec reference is immutable and supplied only at block zero")
    for z in (c.video, c.audio, c.reference):
        if z is not None and (not z.is_floating_point() or not torch.isfinite(z).all()):
            raise ValueError("Nonfinite/nonfloating normalized codec latent")


def unpack_duo_latents(payload, *, height, width, codec_mode="stream_4frame"):
    """Inverse spatial/stereo packing; original VAE math is unchanged."""
    if codec_mode == "duo_5_17":
        frames, steps = 5, 29
    elif codec_mode in ("stream_4frame", "old"):
        frames, steps = 1, 7
    else:
        raise ValueError("Unknown Realization codec geometry")
    h, w = height // 16, width // 16
    video, audio = payload["video_latent"], payload["audio_latent"]
    if (
        height % 32
        or width % 32
        or video.shape != (1, frames * (h // 2) * (w // 2), 96)
        or audio.shape != (1, 2 * steps, 32)
    ):
        raise ValueError("Native Realization packed output geometry mismatch")
    v = video.reshape(frames, h // 2, w // 2, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(24, frames, h, w)
    a = audio.reshape(2, steps, 32).permute(0, 2, 1)
    return v.contiguous(), a.contiguous()


@dataclass
class _CodecSession:
    fence: int
    tenant: str
    reference: torch.Tensor
    history: deque
    next_block: int = 0
    touched: float = 0
    exact_decoder: object = None
    pending_audio: dict | None = None


class DuoCodecPipeline(nn.Module):
    _dit_modules = []
    _encoder_modules = []
    _vae_modules = ["video", "audio"]
    weights_sources = ()
    dummy_run_num_frames = 0  # startup has no session/reference; never fabricate one

    def __init__(self, od_config, *, tile_group=None):
        super().__init__()
        pc = od_config.parallel_config
        if (
            od_config.dtype != torch.float32
            or od_config.step_execution
            or pc.tensor_parallel_size not in (1, 2, 4, 8)
            or pc.sequence_parallel_size != 1
            or od_config.enable_cpu_offload
            or od_config.enable_layerwise_offload
            or od_config.quantization_config is not None
            or od_config.enable_sleep_mode
        ):
            raise ValueError(
                "Realization codec requires resident FP32 weights, request execution, "
                "TP1 or TP2/TP4/TP8 tile workers, SP1, no offload; "
                f"actual dtype={od_config.dtype}, step={od_config.step_execution}, tp={pc.tensor_parallel_size}, "
                f"sp={pc.sequence_parallel_size}, cpu_offload={od_config.enable_cpu_offload}, "
                f"layerwise_offload={od_config.enable_layerwise_offload}, quant={od_config.quantization_config}, "
                f"sleep={od_config.enable_sleep_mode}"
            )
        self.od_config = od_config
        self.options = od_config.model_config.get("duo_codec", {})
        self.max_sessions = self.options.get("max_sessions", 4)
        self.max_retired_sessions = self.options.get("max_retired_sessions", 4096)
        self.idle_seconds = self.options.get("session_idle_seconds", 300)
        if (
            type(self.max_sessions) is not int
            or self.max_sessions < 1
            or self.idle_seconds <= 0
            or type(self.max_retired_sessions) is not int
            or self.max_retired_sessions < 1
        ):
            raise ValueError("Invalid codec session bounds")
        # The official loader constructs on CPU; preserve the execution
        # device context and move both resident codecs after loading.
        device = torch.empty(0).device
        root = Path(od_config.model)
        if root.name in ("FL2VA", "Ref2VA"):
            root = root.parent
        self.video = load_codec("video", root, trust_remote_code=od_config.trust_remote_code).to(device)
        if self.options.get("codec_mode") == "duo_5_17":
            cache_bf16_decoder_weights(self.video)
        self._streaming_video = None
        if self.options.get("decode_backend", "native") == "solarx":
            self.max_sessions = 1
        self.sessions, self.retired = {}, {}
        self._video_graph = None
        self._audio_graph = None
        self._audio_stream = None
        self._tile_group = None
        self.video.stack_tiling = self.options.get("stack_tiling", False)
        video_group = None
        tensor_group = None
        tensor_rank = 0
        tensor_size = self.options.get("tile_tensor_parallel_size", 1)
        if pc.tensor_parallel_size in (2, 4, 8):
            from vllm.distributed import get_tp_group

            self._tile_group = get_tp_group() if tile_group is None else tile_group
            video_group = self._tile_group
            if tensor_size > 1:
                from .tile_tp import tile_tensor_groups

                video_group, tensor_group, tensor_rank = tile_tensor_groups(self._tile_group, tensor_size)
            set_tile_group(self.video, video_group)

        if self.options.get("decode_backend", "native") == "solarx":
            from .streaming_video import make_streaming_runtime

            if tensor_size > 1:
                from .tile_tp import shard_decoder

                self.video.decoder.to(dtype=torch.bfloat16)
                self.video.post_quant_conv.to(dtype=torch.bfloat16)
                shard_decoder(self.video, tensor_group, tensor_rank, tensor_size)
            self._streaming_video = make_streaming_runtime(
                self.video, video_group, use_graph=self.options.get("use_cuda_graph", True)
            )
            height, width = self.options["height"], self.options["width"]
            with torch.inference_mode():
                for h, w in dict.fromkeys(((height, width), (width, height))):
                    self._streaming_video.reset()
                    z = torch.zeros((1, 24, 1, h // 16, w // 16), device=device)
                    for _ in range(13):
                        self._streaming_video.push(z)
            self._streaming_video.reset()

        if self._tile_group is not None and self._tile_group.rank_in_group != 0:
            self._vae_modules = ["video"]
            return

        self.audio = load_codec("audio", root, trust_remote_code=od_config.trust_remote_code).to(device)
        headroom_db = self.options.get("audio_headroom_db", 0.0)
        if headroom_db:
            # The checkpoint clamps directly after BigVGAN's last convolution.
            # Apply fixed stereo gain before that clamp, also during graph replay.
            gain = 10 ** (-headroom_db / 20)
            self.audio.decoder.conv_post.register_forward_hook(lambda _m, _args, wave: wave.mul(gain))
        if self.options.get("codec_mode") == "duo_5_17":
            # Audio geometry is fixed; capture before admitting live chunks.
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                self._decode_audio_window(torch.zeros((2, 32, 29), device=device))

    def _decode_with_graph(self, z, decode, name):
        if not self.options.get("use_cuda_graph", True):
            return decode(z)
        cached = getattr(self, name)
        if cached is None:
            stream = torch.cuda.Stream(device=z.device)
            stream.wait_stream(torch.cuda.current_stream(z.device))
            with torch.cuda.stream(stream):
                static = z.clone()
                for _ in range(3):
                    output = decode(static)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = decode(static)
            torch.cuda.current_stream(z.device).wait_stream(stream)
            cached = (static, output, graph)
            setattr(self, name, cached)
        static, output, graph = cached
        if static.shape != z.shape or static.dtype != z.dtype:
            raise ValueError("Resident codec graph geometry changed")
        static.copy_(z)
        graph.replay()
        return output

    def _decode_video_window(self, z):
        return self._decode_with_graph(z, self.video._adaptive_decode, "_video_graph")

    def _decode_audio_window(self, z):
        return self._decode_with_graph(z, self.audio.decode, "_audio_graph")

    def load_weights(self, weights):
        if any(True for _ in weights):
            raise ValueError("Original codec weights were already strictly loaded")
        return {name for name, _ in self.named_parameters()}

    @property
    def device(self):
        return next(self.video.parameters()).device

    def close_session(self, session_id, fence):
        session = self.sessions.get(session_id)
        if session is not None:
            if session.fence != fence:
                raise ValueError("Stale codec fence")
            self.sessions.pop(session_id)
        # A reservation may be cancelled before the codec sees its first block.
        # Unknown closes allocate no codec fence state; scheduler owns tickets.
        if session is not None or session_id in self.retired:
            self.retired[session_id] = max(fence, self.retired.get(session_id, -1))

    def encode_reference(self, rgb, seed=42, *, codec_mode=None):
        """Cold bootstrap using the encoder semantics of the model protocol."""
        mode = codec_mode or self.options.get("codec_mode", "stream_4frame")
        device = self.device
        if mode == "duo_5_17":
            pixels = torch.as_tensor(rgb.copy()).permute(2, 0, 1)[None, :, None].to(device).float() / 255
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                x = self.video.processor.transform_tensor(pixels)
                moments = self.video._adaptive_encode(x)
                z = moments.chunk(2, dim=1)[0][0].float()
        elif mode in ("stream_4frame", "old"):
            with (
                torch.random.fork_rng(devices=[device.index]),
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.float16),
            ):
                torch.manual_seed(seed)
                z = self.video.encode_images([rgb])[0].float()
        else:
            raise ValueError("Unknown reference codec mode")
        z = (z - self.video.latents_mean[:, None, None, None]) / self.video.latents_std[:, None, None, None]
        if z.shape[1] != 1 or not torch.isfinite(z).all():
            raise ValueError("Invalid original Realization reference encoding")
        return z.cpu().to(torch.bfloat16)

    def decode_block(self, c):
        codec_started_ns = time.perf_counter_ns()
        validate_codec_block(c)
        if c.codec_mode == "duo_5_17":
            return self._decode_exact_block(c)
        now = time.monotonic()
        for sid, live in tuple(self.sessions.items()):
            if now - live.touched > self.idle_seconds:
                self.close_session(sid, live.fence)
        session = self.sessions.get(c.session_id)
        # The official manager fences cancellation before admitting a new epoch.
        # Retire the interrupted epoch if its last chunk already completed.
        if session is not None and c.block == 0 and c.fence > session.fence:
            self.close_session(c.session_id, session.fence)
            session = None
        if session is None:
            if c.block != 0 or c.fence <= self.retired.get(c.session_id, -1):
                raise ValueError("Codec requires block zero with fresh fence")
            if len(self.sessions) >= self.max_sessions:
                raise BufferError("Codec session limit")
            if (
                c.session_id not in self.retired
                and len(set(self.retired) | set(self.sessions)) >= self.max_retired_sessions
            ):
                raise BufferError("Codec fence registry full; start a new engine epoch")
            session = _CodecSession(
                c.fence, c.tenant, c.reference.detach().to(self.device).clone(), deque(maxlen=4), touched=now
            )
        if (c.fence, c.tenant, c.block) != (
            session.fence,
            session.tenant,
            session.next_block,
        ) or c.video.shape != session.reference.shape:
            raise ValueError("Stale/out-of-order codec block or changed geometry")
        # Commit history only after BOTH codec outputs have been acknowledged.
        # A model failure retires the fence: never allow retry with partial state.
        self.sessions[c.session_id] = session
        metrics = {}

        def timed(name, call):
            start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start_event.record()
            value = call()
            stop_event.record()
            stop_event.synchronize()
            metrics[name + "_gpu_ms"] = start_event.elapsed_time(stop_event)
            metrics[name + "_wall_s"] = time.perf_counter() - wall
            return value

        try:
            with torch.inference_mode(), current_omni_platform.device(self.device):
                current = c.video.detach().to(self.device).clone()
                padding = 4 - len(session.history)
                window = torch.cat([session.reference] * padding + list(session.history) + [current], dim=1)[
                    None
                ].float()

                group = self._tile_group
                audio_root = group is None or group.rank_in_group == 0
                async_audio = self.options.get("async_audio", True)
                staged_audio = {}
                if audio_root and async_audio:
                    za = (
                        c.audio.to(self.device).float() * self.audio.latents_std[None, :, None]
                        + self.audio.latents_mean[None, :, None]
                    )
                    if self._audio_stream is None:
                        self._audio_stream = torch.cuda.Stream(device=self.device)
                    # Depend on inputs, before the video replay enters this stream.
                    self._audio_stream.wait_stream(torch.cuda.current_stream(self.device))

                def video_decode():
                    z = (
                        window * self.video.latents_std[None, :, None, None, None]
                        + self.video.latents_mean[None, :, None, None, None]
                    )
                    with torch.autocast("cuda", dtype=torch.float16):
                        raw = self._decode_video_window(z)
                    if audio_root and async_audio:
                        with torch.cuda.stream(self._audio_stream), torch.jit.optimized_execution(False):
                            staged_audio["wave"] = self._decode_audio_window(za).squeeze(1).float()
                            staged_audio["ready"] = torch.cuda.Event()
                            staged_audio["ready"].record()
                    if raw.shape[2] != 20 or not self.video.causal_encoder:
                        raise ValueError("Expected 20 execution frames and causal trim")
                    pixels = self.video.trim_output(raw, 17)
                    pixels = self.video.processor.revert_tensor(pixels.float())[:, :, 13:17]
                    if pixels.shape[2] != 4 or not torch.isfinite(pixels).all():
                        raise ValueError("Invalid five-code video decode")
                    return (pixels[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu()

                rgb = timed("video_decode", video_decode)
                clock = block_clock(c.block)
                samples = clock["audio_stop_sample"] - clock["audio_start_sample"]

                def audio_decode():
                    if audio_root:
                        if async_audio:
                            torch.cuda.current_stream(self.device).wait_event(staged_audio["ready"])
                            wave = staged_audio["wave"]
                        else:
                            za = (
                                c.audio.to(self.device).float() * self.audio.latents_std[None, :, None]
                                + self.audio.latents_mean[None, :, None]
                            )
                            with torch.jit.optimized_execution(False):
                                wave = self._decode_audio_window(za).squeeze(1).float()
                    else:
                        wave = torch.empty((2, 5600), device=self.device, dtype=torch.float32)
                    if group is not None:
                        torch.distributed.broadcast(wave, src=group.ranks[0], group=group.device_group)
                    if wave.shape != (2, 5600) or not torch.isfinite(wave).all():
                        raise ValueError("Invalid decoded stereo audio")
                    return wave[:, :samples].cpu()

                wave = timed("audio_decode", audio_decode)
                session.history.append(current)
                session.next_block += 1
                session.touched = time.monotonic()
                if c.final:
                    self.close_session(c.session_id, c.fence)
                metrics.update(
                    video_window_latents=5,
                    video_target_frames=17,
                    retained_video_frames=4,
                    startup_pad_latents=padding,
                    history_latents=len(session.history),
                )
                return dict(
                    payload={"rgb": rgb, "waveform": wave},
                    metadata={
                        "session_id": c.session_id,
                        "fence": c.fence,
                        "block": c.block,
                        "tile_parallel_size": self.od_config.parallel_config.tensor_parallel_size,
                        "stack_tiling": self.video.stack_tiling,
                        "codec_started_ns": codec_started_ns,
                        "codec_finished_ns": time.perf_counter_ns(),
                        "dispatch_ns": c.dispatch_ns,
                        **clock,
                        **metrics,
                    },
                )
        except BaseException:
            self.close_session(c.session_id, c.fence)
            raise

    def _new_video_decoder(self, block=0):
        if self._streaming_video is not None:
            from .streaming_video import DuoStreamingVideoDecoder

            return DuoStreamingVideoDecoder(self.video, runtime=self._streaming_video, block=block)
        from .exact_video import DuoExactVideoDecoder

        decoder = DuoExactVideoDecoder(
            self.video,
            cache_temporal_overlap=self.options.get("cache_temporal_overlap", True),
            next_block_lookahead=self.options.get("next_block_lookahead", False),
        )
        decoder.received = decoder.emitted = block
        return decoder

    def _decode_exact_block(self, c):
        started = time.perf_counter_ns()
        captures_before = self._streaming_video.capture_count if self._streaming_video is not None else 0
        now = time.monotonic()
        for sid, live in tuple(self.sessions.items()):
            if now - live.touched > self.idle_seconds:
                self.close_session(sid, live.fence)
        session = self.sessions.get(c.session_id)
        if session is not None and c.block == 0 and c.fence > session.fence:
            self.close_session(c.session_id, session.fence)
            session = None
        if session is None:
            if c.block != 0 or c.fence <= self.retired.get(c.session_id, -1):
                raise ValueError("Exact codec requires fresh block zero")
            if len(self.sessions) >= self.max_sessions:
                raise BufferError("Codec session limit")
            if (
                c.session_id not in self.retired
                and len(set(self.retired) | set(self.sessions)) >= self.max_retired_sessions
            ):
                raise BufferError("Codec fence registry full; start a new engine epoch")
            session = _CodecSession(
                c.fence,
                c.tenant,
                c.reference.detach().cpu().clone(),
                deque(maxlen=2),
                touched=now,
                exact_decoder=self._new_video_decoder(),
                pending_audio={},
            )
            self.sessions[c.session_id] = session
        if (
            (session.fence, session.tenant, session.next_block) != (c.fence, c.tenant, c.block)
            or session.exact_decoder is None
            or c.video.shape[-2:] != session.reference.shape[-2:]
        ):
            raise ValueError("Stale/noncontiguous exact codec block")
        try:
            with torch.inference_mode(), current_omni_platform.device(self.device):
                group = self._tile_group
                root = group is None or group.rank_in_group == 0
                # Every rank executes video tiling, then rank zero decodes audio.
                if c.reset_media_cache:
                    if session.pending_audio:
                        raise RuntimeError("Media reset cannot discard unpublished audio")
                    session.exact_decoder = self._new_video_decoder(c.block)
                    session.history.clear()
                decoded = session.exact_decoder.push(c.video)
                if c.final:
                    decoded += session.exact_decoder.finish()
                if root:
                    z = c.audio.to(self.device).float() * self.audio.latents_std[None, :, None]
                    z = z + self.audio.latents_mean[None, :, None]
                    with torch.jit.optimized_execution(False):
                        wave = self._decode_audio_window(z).squeeze(1).float()
                else:
                    wave = torch.empty((2, 23200), device=self.device, dtype=torch.float32)
                if group is not None:
                    torch.distributed.broadcast(wave, src=group.ranks[0], group=group.device_group)
                if wave.shape != (2, 23200) or not torch.isfinite(wave).all():
                    raise ValueError("Invalid exact Realization decoded audio")
                session.pending_audio[c.block] = wave.cpu().clone()
                packets = []
                for packet in decoded:
                    block = packet["block"]
                    clock = block_clock(block, frames=17)
                    n = clock["audio_stop_sample"] - clock["audio_start_sample"]
                    packets.append(
                        dict(
                            payload=dict(rgb=packet["frames"], waveform=session.pending_audio.pop(block)[:, :n]),
                            metadata=dict(session_id=c.session_id, fence=c.fence, block=block, **clock),
                        )
                    )
                session.next_block += 1
                session.touched = time.monotonic()
                metadata = dict(
                    session_id=c.session_id,
                    fence=c.fence,
                    block=c.block,
                    codec_mode="duo_5_17",
                    buffered=not packets,
                    video_input_latents=5,
                    decode_backend=self.options.get("decode_backend", "native"),
                    video_lookahead_latents=2 if session.exact_decoder.next_block_lookahead else 0,
                    codec_started_ns=started,
                    codec_finished_ns=time.perf_counter_ns(),
                    video_graph_captures_total=self._streaming_video.capture_count
                    if self._streaming_video is not None
                    else 0,
                    video_graph_captures_in_block=(self._streaming_video.capture_count - captures_before)
                    if self._streaming_video is not None
                    else 0,
                    dispatch_ns=c.dispatch_ns,
                    tile_parallel_size=self.od_config.parallel_config.tensor_parallel_size,
                    tile_tensor_parallel_size=self.options.get("tile_tensor_parallel_size", 1),
                    spatial_tiles=len(self._streaming_video._plan["tiles"])
                    if self._streaming_video is not None
                    else None,
                )
                if c.final:
                    if session.pending_audio:
                        raise RuntimeError("Exact codec EOF left unmatched audio")
                    self.close_session(c.session_id, c.fence)
                return dict(payload=dict(packets=packets), metadata=metadata)
        except BaseException:
            self.close_session(c.session_id, c.fence)
            raise

    def forward(self, batch):
        from vllm_omni.diffusion.data import DiffusionOutput

        outputs = []
        for request in batch.requests:
            result = self.decode_block(request.prepared_layout)
            if result["metadata"].get("codec_mode") == "duo_5_17":
                group = self._tile_group
                if group is None or group.rank_in_group == 0:
                    for packet in result["payload"]["packets"]:
                        packet["payload"] = encode_av_chunk(packet["payload"], packet["metadata"])
                else:
                    # Only the executor reply rank transports AV packets.
                    result["payload"]["packets"] = []
                result["metadata"]["codec_encoded_finished_ns"] = time.perf_counter_ns()
            outputs.append(DiffusionOutput(output=result))
        return outputs

    def close(self):
        self.sessions.clear()
        self.retired.clear()


def get_duo_codec_pre_process_func(od_config):
    def preprocess(request):
        if request.prepared_layout is None:
            from .wire import codec_from_wire

            request.prepared_layout = codec_from_wire(request.prompt["additional_information"]["duo_codec"])
        c = request.prepared_layout
        validate_codec_block(c)
        request.batch_compatibility_key = ("duo_codec", tuple(c.video.shape), str(c.video.dtype))
        return request

    return preprocess
