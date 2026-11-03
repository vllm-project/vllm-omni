# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import io
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnxruntime
import s3tokenizer
import torch
import torch.nn as nn
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from hyperpyyaml import load_hyperpyyaml

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal.audio import AudioResampler
from vllm.multimodal.media.audio import load_audio
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator
from vllm_omni.model_executor.models.cosyvoice3.utils import mel_spectrogram
from vllm_omni.model_executor.models.minicpmo_4_5.config_flags import read_bool_flag
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    DEFAULT_STREAM_CONFIG,
    DEFAULT_TOKEN2WAV_CONFIG,
    STEP_AUDIO2_DEFAULT_PROMPT_WAV,
    STREAM_SOURCE_CACHE_LEN,
)

logger = init_logger(__name__)

# Cache key for the default prompt's solved features inside the opt-in
# BatchedToken2Wav backend (``prepare_prompt`` is content-addressed by it).
_BATCHED_PROMPT_CACHE_ID = "step_audio2_default_prompt"


@contextmanager
def _stage_matmul_policy(tf32_enabled: bool) -> Iterator[None]:
    """Apply this stage's TF32 matmul policy for the enclosed work (CUDA only).

    Token2wav runs in fp32, so without TF32 every flow-DiT GEMM executes on
    SIMT cores; ordinary TF32 puts the dense QKV/MLP and Triton CFM attention
    dots on SM80+ Tensor Cores while HiFT stays IEEE FP32. This stage owns the
    vocoder process, so the previous policy is restored on exit (same contract
    as the MiniCPM-o facade's matmul policy).
    """
    from vllm_omni.platforms import current_omni_platform

    if not tf32_enabled or not current_omni_platform.is_cuda():
        yield
        return
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _prepare_s3_quantize_graph_runtime() -> None:
    """Select graph-capturable ACLNN kernels before S3 quantize capture.

    Mirrors ``prepare_code2wav_graph_runtime`` (minicpmo_4_5_code2wav) and
    ``_prepare_moss_codec_graph_runtime``: the NPU platform keeps
    ``allow_internal_format`` on for quantized LLM kernels, but NPUGraph
    capture of the conv kernels requires it off, plus ``jit_compile=False``.
    Configure it right here, only when capture is actually enabled, so
    this path never relies on another stage having done it.
    """
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError(
            "S3 quantize NPUGraph capture is incompatible with "
            "ASCEND_LAUNCH_BLOCKING=1; unset it or set it to 0 before startup."
        )
    npu = torch.npu
    npu.config.allow_internal_format = False
    npu.set_compile_mode(jit_compile=False)
    logger.info("Configured S3 quantize NPUGraph runtime (allow_internal_format=False, jit_compile=False)")


# Long-audio boundary of s3tokenizer's quantize() (max_frames = 3000 in
# s3tokenizer/model.py). The short-audio path below must stay on the same side
# of that boundary as the library call, so this constant mirrors it; if the
# library changes its value, change this one with it.
_S3_MAX_FRAMES = 3000


def _h2d_async(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Staging-copy to pinned memory, then async H2D.

    A plain ``.to(device)`` on pageable memory is a synchronous copy: the host
    blocks until every kernel already queued on the stream finishes, which
    costs ~100ms per call when setup runs behind a c8 decode backlog. The
    pinned staging copy is pure host work; the device copy overlaps with
    whatever is already executing on the stream.
    """
    pinned = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    pinned.copy_(t)
    return pinned.to(device, non_blocking=True)


def _get_left_context_size(info: dict[str, Any]) -> Any | None:
    """Return the async-chunk marker from a nested metadata payload."""
    meta = info.get("meta")
    if not isinstance(meta, dict):
        return None
    return meta.get("left_context_size")


def fade_in_out(
    fade_in_mel: torch.Tensor,
    fade_out_mel: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """Cross-fade two overlapping waveform segments using a Hamming window.

    The window is split in half: the first half ramps *up* (fade-in) and
    the second half ramps *down* (fade-out).  The overlap region of
    ``fade_in_mel`` is blended with the tail of ``fade_out_mel``.
    """
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel


@dataclass
class _StreamState:
    """Per-request streaming state for concurrent Token2Wav sessions."""

    setup_done: bool = False
    stream_cache: dict | None = None
    hift_cache_dict: dict = field(default_factory=dict)
    finished: bool = False


class _ConvRNNF0Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        for in_channels in [80, 512, 512, 512, 512]:
            layers.extend([weight_norm(nn.Conv1d(in_channels, 512, 3, padding=1)), nn.ELU()])
        self.condnet = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.condnet(x).transpose(1, 2)).squeeze(-1).abs()


def drop_upstream_chunk_att_buffers(flow: nn.Module) -> int:
    """Drop upstream cosyvoice2's unused chunk-streaming ``att_cache_buffer`` tensors (~2.1 GiB); returns bytes freed.

    Only ``flow.setup_cache`` / ``flow.inference_chunk`` read them, and then
    fail instead of computing on a stand-in.
    """
    released = 0
    decoder = getattr(flow, "decoder", None)
    for module in (decoder, getattr(decoder, "estimator", None)):
        buffer = module._buffers.get("att_cache_buffer") if isinstance(module, nn.Module) else None
        if isinstance(buffer, torch.Tensor):
            released += buffer.numel() * buffer.element_size()
            del module._buffers["att_cache_buffer"]
    return released


def _build_hift() -> HiFTGenerator:
    return HiFTGenerator(
        sampling_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
        f0_predictor=_ConvRNNF0Predictor(),
    )


class StepAudio2Token2WavCore(nn.Module):
    """Core Token2Wav model - handles token to waveform conversion"""

    def __init__(
        self,
        model_path: str,
        float16: bool = False,
        device: str = "cuda",
        n_timesteps: int = DEFAULT_TOKEN2WAV_CONFIG.n_timesteps,
        drop_upstream_chunk_att_buffers: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        # A backend that never runs ``flow.setup_cache`` / ``flow.inference_chunk``
        # (MiniCPM-o's ``BatchedToken2Wav``) can drop their 2.1 GiB of buffers.
        self._drop_chunk_att_buffers = bool(drop_upstream_chunk_att_buffers)
        self.float16 = float16
        self.device = torch.device(device)
        self.n_timesteps = n_timesteps

        self._models_loaded = False
        self._audio_tokenizer = None
        self._spk_model = None
        self._flow = None
        self._hift = None
        # Private stream for the campplus forward (see _forward_spk_embedding).
        self._spk_stream = None
        self._spk_event = None
        # S3 quantize graph state (see _quantize_s3_graph): declared up front
        # so concurrent first calls share one lock/dict instead of each
        # creating their own outside any lock.
        self._s3_graphs: dict = {}
        self._s3_failed: set = set()
        self._s3_graph_lock = threading.Lock()
        self._s3_graph_done = None
        self._s3_cuda_graph: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
        # Replay/eager counters, logged on capture and at milestones so a
        # benchmark log shows the realized replay rate per run.
        self._s3_replay_calls = 0
        self._s3_eager_calls = 0

        self.cache = {}

        # Streaming state (constants from centralised config)
        self.mel_cache_len = DEFAULT_STREAM_CONFIG.mel_cache_len
        self.source_cache_len = STREAM_SOURCE_CACHE_LEN
        self.speech_window: torch.Tensor | None = None  # created lazily on device

        # On Ascend the talker worker never routes through
        # ``NPUOmniPlatform.set_device``, so gate on ``current_omni_platform``
        # and apply the Token2Wav NPU patch here (idempotent) before
        # ``_ensure_models_loaded`` runs. Covers every
        # construction site (Step-Audio2 wrapper + MiniCPM-o facade): patched
        # ``_ensure_models_loaded`` replaces HiFT's failing linear downsample,
        # while patched ``forward`` expands CosyVoice DiT masks and forces
        # MATH SDPA (avoids FA 161001).
        from vllm_omni.platforms import current_omni_platform

        if current_omni_platform.is_npu():
            from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
                apply_step_audio2_token2wav_npu_patch,
            )

            apply_step_audio2_token2wav_npu_patch()

    def _ensure_models_loaded(self):
        """Lazy load models on first use"""
        if self._models_loaded:
            return

        from vllm.logger import init_logger

        logger = init_logger(__name__)
        logger.info(f"Loading Token2Wav models from: {self.model_path}")

        self._audio_tokenizer = (
            s3tokenizer.load_model(f"{self.model_path}/speech_tokenizer_v2_25hz.onnx").to(self.device).eval()
        )
        # Cache the rotary table on device: AudioEncoderV2.forward does
        # ``self.freqs_cis.to(x.device)`` on every call — a pageable (synchronous)
        # H2D that blocks the host behind queued decode work. With the table
        # already on device that ``.to`` is a no-op.
        _enc = getattr(self._audio_tokenizer, "encoder", None)
        if _enc is not None and isinstance(getattr(_enc, "freqs_cis", None), torch.Tensor):
            _enc.freqs_cis = _enc.freqs_cis.to(self.device)

        with open(f"{self.model_path}/flow.yaml") as f:
            configs = load_hyperpyyaml(f)
        self._flow = configs["flow"]
        if self._drop_chunk_att_buffers:
            # Before ``.to(self.device)``: the buffers never reach the device.
            released = drop_upstream_chunk_att_buffers(self._flow)
            logger.info("Dropped the upstream flow chunk attention buffers (%.1f MiB)", released / 2**20)
        if self.float16:
            self._flow.half()
        self._flow.load_state_dict(
            torch.load(f"{self.model_path}/flow.pt", map_location="cpu", weights_only=True), strict=True
        )
        self._flow.to(self.device).eval()
        self._hift = _build_hift()

        hift_state_dict = {
            k.replace("generator.", ""): v
            for k, v in torch.load(f"{self.model_path}/hift.pt", map_location="cpu", weights_only=True).items()
        }
        self._hift.load_state_dict(hift_state_dict, strict=True)
        self._hift.to(self.device).eval()
        # Fold after loading and device placement so the normalized weight is
        # computed once instead of on every chunk. Folding rewrites
        # ``state_dict`` keys, so reloading the original checkpoint into this
        # instance is unsupported. ``CosyVoice3Code2Wav`` already does this;
        # MiniCPM-o and StepAudio2 reach HiFT through this core.
        folded = self._hift.remove_weight_norm()
        if folded:
            logger.info("Folded %d HiFT weight-norm layers", folded)
        else:
            logger.warning("HiFT generator had no weight-norm layers to fold; check config drift")

        self._models_loaded = True
        logger.info("Token2Wav models loaded successfully")

    @property
    def audio_tokenizer(self):
        """Lazy-loaded audio tokenizer"""
        self._ensure_models_loaded()
        return self._audio_tokenizer

    @property
    def spk_model(self):
        """Lazy-loaded speaker model.

        The CPU onnxruntime fallback is built on first *use*: the device paths
        (:meth:`enable_trt_spk_embedding` / :meth:`enable_device_spk_embedding`)
        install a callable right after the eager facade init, so the temporary
        CPU session is never allocated there (review: eager load + immediate
        replacement wasted one full onnx parse per replica).
        """
        self._ensure_models_loaded()
        if self._spk_model is None:
            option = onnxruntime.SessionOptions()
            option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            option.intra_op_num_threads = 1
            self._spk_model = onnxruntime.InferenceSession(
                f"{self.model_path}/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"]
            )
        return self._spk_model

    @property
    def flow(self):
        """Lazy-loaded flow model"""
        self._ensure_models_loaded()
        return self._flow

    @property
    def hift(self):
        """Lazy-loaded HiFT vocoder"""
        self._ensure_models_loaded()
        return self._hift

    def _forward_spk_embedding(self, spk_feat: torch.Tensor) -> torch.Tensor:
        """Run campplus on ``[T, 80]`` fbank features; returns ``[1, 192]`` on ``self.device``."""
        spk_model = self.spk_model
        if isinstance(spk_model, onnxruntime.InferenceSession):
            return torch.tensor(
                spk_model.run(None, {spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0],
                device=self.device,
            )
        # CampplusTRT / CampplusTorch (any callable taking [T, 80] -> [1, 192]).
        stream = self._spk_stream
        if stream is None or torch.is_grad_enabled():
            return spk_model(spk_feat).to(self.device)
        # Launching the ~50 campplus kernels on the decode stream blocks the
        # host whenever that queue is saturated (~100 ms/request at c8). A
        # private stream lets them run alongside decode; the dependency is
        # attached with an event instead of an implicit host wait.
        #
        # ``out`` is allocated on the private stream but read (and cached) by
        # the caller on the decode stream, so tell the caching allocator about
        # that second consumer -- otherwise the block may be reused while the
        # decode stream still reads it.
        decode_stream = torch.npu.current_stream(self.device)
        with torch.npu.stream(stream):
            out = spk_model(spk_feat)
            out.record_stream(decode_stream)
        self._spk_event = stream.record_event()
        return out.to(self.device)

    def enable_trt_spk_embedding(self, *, cpu_reference: bool = False) -> None:
        """Swap the onnxruntime campplus session for the shared TensorRT engine.

        Reuses the CosyVoice3 CAM++ engine wrapper — MiniCPM-o / Step-Audio2
        ship the same campplus.onnx architecture and I/O contract.

        Deliberately does *not* call ``_ensure_models_loaded()``: callers may
        run inside vLLM's ``set_default_torch_dtype(model_config.dtype)`` block
        (e.g. the standalone wrapper's ``__init__``), and forcing the load
        there would eagerly build s3tokenizer/flow/HiFT under the stage dtype
        instead of on first forward (review).
        """
        from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import get_campplus_trt

        self._spk_model = get_campplus_trt(
            f"{self.model_path}/campplus.onnx", device=self.device, cpu_reference=cpu_reference
        )

    def enable_device_spk_embedding(self, *, use_trt: bool = True) -> None:
        """Swap the CPU onnxruntime campplus session for an on-device model.

        CUDA defaults to the shared TensorRT engine. Set ``use_trt=False``
        to use torch without a TensorRT dependency.
        Other devices (e.g. Ascend NPU) use torch CAMPPlus loaded from the same
        campplus.onnx weights — same architecture and I/O contract, no
        per-request ``.cpu().numpy()`` host round-trip.

        Like :meth:`enable_trt_spk_embedding`, this only installs the callable;
        the remaining token2wav modules stay lazy and load on first forward.
        """
        if self.device.type == "cuda" and use_trt:
            self.enable_trt_spk_embedding()
            return
        from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import get_campplus_torch

        self._spk_model = get_campplus_torch(f"{self.model_path}/campplus.onnx", device=self.device)
        # Run campplus on its own stream so its ~50 kernel launches do not line
        # up behind the decode backlog (see _forward_spk_embedding).
        npu_mod = getattr(torch, "npu", None)
        try:
            if npu_mod is not None and npu_mod.is_available():
                self._spk_stream = npu_mod.Stream(device=self.device)
        except Exception:
            self._spk_stream = None

    # Per-frame-count NPU graphs for the S3 quantize forward (short-audio
    # path), capped: Seed-TTS references span many lengths (179 distinct over
    # 200 wavs), so only the first ``_MAX_S3_GRAPHS`` frame counts seen get
    # captured and every later distinct length runs eager — the cap guards
    # against pathological callers capturing a graph per length.
    _MAX_S3_GRAPHS = 2

    @torch.inference_mode()
    def _quantize_s3_graph(self, mels: torch.Tensor, mels_lens: torch.Tensor):
        """S3 quantize (short audio, device tensors) with an NPU graph.

        Equivalent to ``encoder + quantizer.encode`` (the library else-branch),
        but replays a captured graph: eager is ~13ms on NPU while the output
        is input-length invariant (61..249 frames all ~11ms) — launch-bound.
        Graph replay is ~6ms with bit-identical tokens.  Eager fallback
        off-NPU / on capture failure / bucket overflow.
        """
        if self._s3_cuda_graph is not None:
            return self._s3_cuda_graph(mels, mels_lens)
        tok = self.audio_tokenizer
        frames = int(mels.shape[2])
        with self._s3_graph_lock:
            entry = self._s3_graphs.get(frames)
            if entry is None:
                if (
                    self.device.type != "npu"
                    or frames in self._s3_failed
                    or len(self._s3_graphs) >= self._MAX_S3_GRAPHS
                ):
                    # Eager on freshly allocated tensors: no shared static
                    # buffers are touched, so no event chaining is needed
                    # (chaining here would re-serialize replicas for nothing).
                    self._s3_eager_calls += 1
                    h, code_len = tok.encoder(mels, mels_lens)
                    code = tok.quantizer.encode(h)
                    return code, code_len
                # Events only exist on NPU; on other devices the lock alone
                # is enough (single stream per process).
                # Capture needs graph-capturable ACLNN kernels; self-configure
                # here instead of relying on the Code2Wav stage having done it.
                _prepare_s3_quantize_graph_runtime()
                stream = torch.npu.current_stream(self.device) if self.device.type == "npu" else None
                if stream is not None and self._s3_graph_done is not None:
                    stream.wait_event(self._s3_graph_done)
                try:
                    enc = tok.encoder
                    # freqs_cis ships as a CPU buffer and the forward does
                    # ``self.freqs_cis.to(x.device)`` every call — that H2D copy
                    # blocks graph capture, so hoist it once.
                    if enc.freqs_cis.device != self.device:
                        enc.freqs_cis = enc.freqs_cis.to(self.device)
                    # The captured graph owns its inputs, so they cannot alias
                    # the caller's per-request tensors.
                    static_mels = mels.clone()
                    static_lens = mels_lens.clone()
                    for _ in range(2):
                        h, code_len = enc(static_mels, static_lens)
                        tok.quantizer.encode(h)
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        h, code_len = enc(static_mels, static_lens)
                        static_code = tok.quantizer.encode(h)
                    torch.npu.synchronize()
                    entry = (graph, static_mels, static_lens, static_code, code_len)
                    # Captured kernels are recorded, not executed, so
                    # static_code and code_len still hold uninitialized
                    # pool memory here (the warmup results above went to
                    # temporary tensors). Replay once so this request
                    # gets the same real result the replay path produces.
                    # The entry is published only after the replay+clone
                    # succeeded: a failed first replay must not leave the
                    # bucket registered, or later same-length requests
                    # would replay a broken graph instead of falling back.
                    graph.replay()
                    code = static_code.clone()
                    code_len = entry[4].clone()
                    self._s3_graphs[frames] = entry
                    self._s3_replay_calls += 1
                    self._s3_graph_done = stream.record_event() if stream is not None else None
                    logger.info(
                        "S3 quantize graph captured for %d frames (replays so far: %d, eager calls: %d)",
                        frames,
                        self._s3_replay_calls,
                        self._s3_eager_calls,
                    )
                    return code, code_len
                except Exception as exc:  # noqa: BLE001
                    self._s3_failed.add(frames)
                    logger.warning(
                        "S3 quantize graph capture failed for %d frames (%s); falling back to eager for this bucket",
                        frames,
                        exc,
                    )
                    h, code_len = tok.encoder(mels, mels_lens)
                    code = tok.quantizer.encode(h)
                    return code, code_len
            stream = torch.npu.current_stream(self.device) if self.device.type == "npu" else None
            if stream is not None and self._s3_graph_done is not None:
                stream.wait_event(self._s3_graph_done)
            graph, static_mels, static_lens, static_code, static_len = entry
            static_mels.copy_(mels)
            static_lens.copy_(mels_lens)
            graph.replay()
            code = static_code.clone()
            code_len = static_len.clone()
            self._s3_graph_done = stream.record_event() if stream is not None else None
            self._s3_replay_calls += 1
            if self._s3_replay_calls in (1, 10, 100, 1000):
                logger.info(
                    "S3 quantize graph replay #%d (eager calls so far: %d)",
                    self._s3_replay_calls,
                    self._s3_eager_calls,
                )
            return code, code_len

    @staticmethod
    def _prepare_prompt_cpu(prompt_wav: str):
        """Extract reference features without accessing the accelerator."""
        audio_np_raw, sample_rate = load_audio(prompt_wav, sr=None, mono=True)
        sample_rate = int(sample_rate)

        # 16 kHz branch: s3tokenizer + speaker embedding
        audio_16k = AudioResampler(target_sr=16000).resample(audio_np_raw.astype(np.float32), orig_sr=sample_rate)
        audio = torch.from_numpy(audio_16k)
        mels = s3tokenizer.log_mel_spectrogram(audio)
        mels, mels_lens = s3tokenizer.padding([mels])
        spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        # 24 kHz branch: mel spectrogram for flow model conditioning
        # Must resample from the ORIGINAL audio, not the 16 kHz version.
        audio_24k = AudioResampler(target_sr=24000).resample(audio_np_raw.astype(np.float32), orig_sr=sample_rate)
        audio_24k_t = torch.from_numpy(audio_24k).unsqueeze(0)  # [1, T]
        prompt_mel = (
            mel_spectrogram(
                audio_24k_t,
                n_fft=1920,
                num_mels=80,
                sampling_rate=24000,
                hop_size=480,
                win_size=1920,
                fmin=0,
                fmax=8000,
                center=False,
            )
            .transpose(1, 2)
            .squeeze(0)
        )  # [T, num_mels]
        return mels, mels_lens, spk_feat, prompt_mel

    def _prepare_prompt(self, prompt_wav: str, *, cpu_features=None):
        """Prepare prompt audio for conditioning"""
        if cpu_features is None:
            audio_np_raw, sample_rate = load_audio(prompt_wav, sr=None, mono=True)
            sample_rate = int(sample_rate)

            # 16 kHz branch: s3tokenizer + speaker embedding
            audio_16k = AudioResampler(target_sr=16000).resample(audio_np_raw.astype(np.float32), orig_sr=sample_rate)
            audio = torch.from_numpy(audio_16k)
            mels = s3tokenizer.log_mel_spectrogram(audio)
            mels, mels_lens = s3tokenizer.padding([mels])
        else:
            mels, mels_lens, spk_feat, prompt_mel = cpu_features
        mels_d = _h2d_async(mels, self.device)
        mels_lens_d = _h2d_async(mels_lens, self.device)
        if mels.shape[-1] > _S3_MAX_FRAMES:
            # Long audio (>30 s) keeps the library call: it routes to the
            # sliding-window mixed-batch path.
            prompt_speech_tokens, prompt_speech_tokens_lens = self.audio_tokenizer.quantize(mels_d, mels_lens_d)
        else:
            # Short audio: NPU graph replay of the library's else-branch
            # (encoder + quantizer.encode). The device-side
            # `mel_len > max_frames` check is skipped — its `.any()` forces an
            # implicit D2H sync; the frame count is static on host
            # (mels.shape[-1]), so the branch decision needs no sync.
            prompt_speech_tokens, prompt_speech_tokens_lens = self._quantize_s3_graph(mels_d, mels_lens_d)

        if cpu_features is None:
            spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
        spk_emb = self._forward_spk_embedding(spk_feat)
        if self._spk_event is not None:
            # Device-side join for the private campplus stream (never blocks the
            # host); it lands here so the host-side 24 kHz mel work overlaps.
            torch.npu.current_stream(self.device).wait_event(self._spk_event)
            self._spk_event = None

        if cpu_features is None:
            # 24 kHz branch: mel spectrogram for flow model conditioning
            # Must resample from the ORIGINAL audio, not the 16 kHz version.
            audio_24k = AudioResampler(target_sr=24000).resample(audio_np_raw.astype(np.float32), orig_sr=sample_rate)
            audio_24k_t = torch.from_numpy(audio_24k).unsqueeze(0)  # [1, T]
            prompt_mel = (
                mel_spectrogram(
                    audio_24k_t,
                    n_fft=1920,
                    num_mels=80,
                    sampling_rate=24000,
                    hop_size=480,
                    win_size=1920,
                    fmin=0,
                    fmax=8000,
                    center=False,
                )
                .transpose(1, 2)
                .squeeze(0)
            )  # [T, num_mels]
        prompt_mels = _h2d_async(prompt_mel.unsqueeze(0), self.device)
        # Build on host then async-copy: torch.tensor(..., device=...) is itself
        # a synchronous H2D that blocks behind queued decode work.
        prompt_mels_lens = _h2d_async(torch.tensor([prompt_mel.shape[0]], dtype=torch.int32), self.device)
        prompt_mels = torch.nn.functional.pad(
            prompt_mels,
            (0, 0, 0, prompt_speech_tokens.shape[1] * self.flow.up_rate - prompt_mels.shape[1]),
            mode="replicate",
        )

        return prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens

    def forward(
        self, generated_speech_tokens: list, prompt_wav: str, return_bytes: bool = True
    ) -> torch.Tensor | bytes:
        """
        Convert audio tokens to waveform

        Args:
            generated_speech_tokens: List of audio token IDs (0-6560)
            prompt_wav: Path to prompt wav file for speaker conditioning
            return_bytes: If True, return WAV bytes; else return tensor

        Returns:
            Audio waveform as bytes or tensor
        """
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, prompt_speech_tokens_lens, spk_emb, prompt_mels, prompt_mels_lens = self.cache[prompt_wav]

        generated_speech_tokens = torch.tensor([generated_speech_tokens], dtype=torch.int32, device=self.device)
        generated_speech_tokens_lens = torch.tensor(
            [generated_speech_tokens.shape[1]], dtype=torch.int32, device=self.device
        )

        with torch.amp.autocast(str(self.device.type), dtype=torch.float16 if self.float16 else torch.float32):
            mel = self.flow.inference(
                generated_speech_tokens,
                generated_speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                prompt_mels_lens,
                spk_emb,
                self.n_timesteps,
            )

        wav, _ = self.hift.inference(speech_feat=mel)

        if return_bytes:
            output = io.BytesIO()
            torchaudio.save(output, wav.cpu(), sample_rate=DEFAULT_TOKEN2WAV_CONFIG.sample_rate, format="wav")
            return output.getvalue()
        else:
            return wav

    # ------------------------------------------------------------------
    # Per-request streaming (does NOT mutate self – uses external state)
    # ------------------------------------------------------------------

    def setup_stream_for(self, prompt_wav: str, state: _StreamState) -> None:
        """Initialise flow + HiFT caches into *state* (no self mutation)."""
        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        prompt_speech_tokens, _, spk_emb, prompt_mels, _ = self.cache[prompt_wav]

        if self.speech_window is None or self.speech_window.device != self.device:
            self.speech_window = torch.from_numpy(np.hamming(2 * self.source_cache_len)).to(
                device=self.device, dtype=torch.float32
            )

        pre_lookahead = DEFAULT_STREAM_CONFIG.pre_lookahead_len
        state.stream_cache = self.flow.setup_cache(
            torch.cat(
                [prompt_speech_tokens, prompt_speech_tokens[:, :pre_lookahead]],
                dim=1,
            ),
            prompt_mels,
            spk_emb,
            n_timesteps=self.n_timesteps,
        )
        state.hift_cache_dict = {
            "mel": torch.zeros(1, prompt_mels.shape[2], 0, device=self.device),
            "source": torch.zeros(1, 1, 0, device=self.device),
            "speech": torch.zeros(1, 0, device=self.device),
        }
        state.setup_done = True

    def stream_chunk_for(
        self,
        audio_tokens: list[int] | torch.Tensor,
        prompt_wav: str,
        last_chunk: bool,
        state: _StreamState,
    ) -> torch.Tensor:
        """Process one chunk using *state* (no self mutation except speech_window)."""
        if state.stream_cache is None:
            raise ValueError("stream_cache not initialised – call setup_stream_for() first")

        if prompt_wav not in self.cache:
            self.cache[prompt_wav] = self._prepare_prompt(prompt_wav)
        _, _, spk_emb, prompt_mels, _ = self.cache[prompt_wav]

        if isinstance(audio_tokens, torch.Tensor):
            # Already on device; avoids the D2H->H2D round-trip of rebuilding from a list.
            # clone() so token_tensor never aliases the caller's input_ids: when the ids
            # already arrive int32-on-device the .to() is a no-op and would share storage,
            # so an in-place write inside inference_chunk could corrupt input_ids.
            token_tensor = audio_tokens.reshape(1, -1).to(device=self.device, dtype=torch.int32).clone()
        else:
            token_tensor = torch.tensor([audio_tokens], dtype=torch.int32, device=self.device)

        with torch.amp.autocast(
            str(self.device.type),
            dtype=torch.float16 if self.float16 else torch.float32,
        ):
            chunk_mel, state.stream_cache = self.flow.inference_chunk(
                token=token_tensor,
                spk=spk_emb,
                cache=state.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=self.n_timesteps,
            )

        # Trim estimator attention cache to avoid unbounded growth
        keep = DEFAULT_STREAM_CONFIG.estimator_cache_keep
        est_att = state.stream_cache.get("estimator_att_cache")
        if est_att is not None and est_att.shape[4] > (prompt_mels.shape[1] + keep):
            state.stream_cache["estimator_att_cache"] = torch.cat(
                [
                    est_att[:, :, :, :, : prompt_mels.shape[1]],
                    est_att[:, :, :, :, -keep:],
                ],
                dim=4,
            )

        # ---- HiFT vocoder with overlap-add ----
        hift_cache_mel = state.hift_cache_dict["mel"]
        hift_cache_source = state.hift_cache_dict["source"]
        hift_cache_speech = state.hift_cache_dict["speech"]

        mel = torch.cat([hift_cache_mel, chunk_mel], dim=2)
        speech, source = self.hift.inference(mel, hift_cache_source)

        if hift_cache_speech.shape[-1] > 0:
            speech = fade_in_out(speech, hift_cache_speech, self.speech_window)

        state.hift_cache_dict = {
            "mel": mel[..., -self.mel_cache_len :].clone().detach(),
            "source": source[:, :, -self.source_cache_len :].clone().detach(),
            "speech": speech[:, -self.source_cache_len :].clone().detach(),
        }

        if not last_chunk:
            speech = speech[:, : -self.source_cache_len]

        return speech.squeeze(0)

    @staticmethod
    def reset_stream_for(state: _StreamState) -> None:
        """Clear streaming caches for the given state."""
        state.stream_cache = None
        state.hift_cache_dict = {}
        state.setup_done = False
        state.finished = True


class StepAudio2Token2WavForConditionalGeneration(nn.Module, SupportsPP):
    """vLLM-compatible wrapper for Step-Audio2 Token2Wav"""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "token2wav.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        model_path = getattr(self.config, "token2wav_path", None)
        model_name_or_path = vllm_config.model_config.model
        if model_path is None:
            # Resolve HF repo names to local cache path
            if not os.path.isdir(model_name_or_path):
                from vllm_omni.transformers_utils.repo_utils import hf_api

                model_name_or_path = hf_api().snapshot_download(model_name_or_path)
            model_path = f"{model_name_or_path}/token2wav"

        float16 = getattr(self.config, "token2wav_float16", False)

        device = "cuda"
        if hasattr(vllm_config, "device_config") and vllm_config.device_config:
            device = str(vllm_config.device_config.device)

        self.token2wav = StepAudio2Token2WavCore(model_path=model_path, float16=float16, device=device)

        # Match the MiniCPM-o wiring: on NPU run the campplus speaker embedding
        # on device (torch, weights from the same campplus.onnx) instead of the
        # per-request CPU onnxruntime session (~120-155ms host time each).
        from vllm_omni.platforms import current_omni_platform

        if current_omni_platform.is_npu():
            self.token2wav.enable_device_spk_embedding()
            logger.info("Step-Audio2 Token2Wav: campplus speaker embedding running on device (torch)")

        self.have_multimodal_outputs = True
        # Required for the runner to pass async_chunk info via
        # runtime_additional_information on every forward step.
        self.enable_update_additional_information = True

        # Per-request streaming states (ordered list matching batch order).
        self._stream_states: list[_StreamState] = []

        # Resolve default prompt wav path from model directory.
        # Priority: env var > {model_dir}/assets/default_female.wav > bare filename (cwd)
        self._default_prompt_wav = self._resolve_prompt_wav(model_name_or_path)

        # Opt-in batched code2wav backend: the in-tree MiniCPM-o engine
        # (BatchedToken2Wav, PR #8443) constructed over this stage's core.
        # The Step-Audio-2-mini token2wav assets are byte-identical to
        # MiniCPM-o's (flow/hift/campplus/speech-tokenizer sha256-verified),
        # so the same engine serves this stage. Default OFF: the pip
        # cosyvoice2 streaming path stays the shipped behavior.
        batched_extra = self._batched_extra()
        self._batched_enabled = read_bool_flag(batched_extra, "token2wav_batched_backend", False)
        self._batched_tf32 = self._batched_enabled and read_bool_flag(batched_extra, "token2wav_allow_tf32", True)
        self._batched_backend: Any = None
        self._batched_precapture_pending = self._batched_enabled and read_bool_flag(
            batched_extra, "token2wav_graph_precapture", True
        )
        # Per-request backend streaming states, keyed by request id. Finished
        # requests drop their entry; a size cap evicts the least-recently-used
        # entry so abandoned (preempted-and-gone) requests cannot leak state.
        self._batched_states: OrderedDict[str, Any] = OrderedDict()
        scheduler = getattr(vllm_config, "scheduler_config", None)
        max_num_seqs = getattr(scheduler, "max_num_seqs", None)
        self._batched_states_cap = max(16, 2 * int(max_num_seqs)) if max_num_seqs else 16
        if self._batched_enabled:
            # The batched path splits the runner's flat ``input_ids`` by
            # per-request token counts, so it needs the padding trimmed (#6712)
            # and the per-request ids the runner stamps in kwargs.
            self.requires_exact_input_shape = True
            self.requires_request_ids = True

        self.make_empty_intermediate_tensors = lambda: None

    @staticmethod
    def _resolve_prompt_wav(model_root: str) -> str:
        """Find the default prompt wav, searching model assets directory."""
        env_val = os.environ.get("STEP_AUDIO2_DEFAULT_PROMPT_WAV")
        if env_val and os.path.exists(env_val):
            return env_val
        # Search common locations: model dir, then bundled assets in this package
        pkg_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(model_root, "assets", STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            os.path.join(model_root, STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            os.path.join(pkg_dir, "assets", STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            os.path.join(pkg_dir, STEP_AUDIO2_DEFAULT_PROMPT_WAV),
            STEP_AUDIO2_DEFAULT_PROMPT_WAV,
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        # Return the env var (even if not found) or the bare filename as fallback
        return env_val or STEP_AUDIO2_DEFAULT_PROMPT_WAV

    def get_language_model(self) -> torch.nn.Module:
        # Return the wrapper so vLLM interface helpers resolve
        # embed_input_ids() on this adapter instead of the bare core model.
        return self

    @property
    def sampler(self):
        return Sampler()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional: dict | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | bytes:
        """
        Forward pass for Token2Wav

        Args:
            input_ids: Audio token IDs (from stage input processor)
            positions: Position IDs (unused)
            intermediate_tensors: Intermediate tensors (unused)
            inputs_embeds: Input embeddings (unused)
            additional: Additional information dict (deprecated)
            runtime_additional_information: Per-request dicts from the
                async_chunk processor.  Each dict contains ``audio_tokens``
                (list[int]) and ``meta.left_context_size`` as the
                last-chunk marker (0 = non-final, 1 = final).
            **kwargs: Other kwargs

        Returns:
            OmniOutput with waveform tensor
        """
        # ----- async_chunk streaming path -----
        # Only enter when at least one entry carries ``left_context_size``
        # (the field the framework forwards from the async_chunk payload).
        # The runner always passes runtime_additional_information (possibly
        # [{}]), so a bare truthiness check would incorrectly enter this
        # branch in synchronous mode.
        if runtime_additional_information and any(
            _get_left_context_size(info) is not None for info in runtime_additional_information
        ):
            return self._forward_async_chunk(input_ids, runtime_additional_information, **kwargs)

        # ----- profiling / warmup guard -----
        # Profile/warmup passes use all-zero token IDs. Real audio tokens
        # are non-zero, so check sum == 0 rather than a length threshold
        # (short utterances may have fewer than 50 tokens).
        if isinstance(input_ids, torch.Tensor):
            is_profile_mode = input_ids.numel() == 0 or input_ids.sum().item() == 0
        else:
            is_profile_mode = not input_ids if not hasattr(input_ids, "__len__") else len(input_ids) == 0

        if is_profile_mode:
            device = torch.device("cpu")
            if isinstance(inputs_embeds, torch.Tensor) and inputs_embeds.numel() > 0:
                device = inputs_embeds.device
            elif isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0:
                device = input_ids.device

            dummy_audio = torch.zeros(1000, device=device, dtype=torch.float32)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [dummy_audio]},
            )

        # ----- synchronous (full-sequence) path -----
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids_list = input_ids.cpu().tolist()
            else:
                input_ids_list = input_ids[0].cpu().tolist()
        else:
            input_ids_list = input_ids if isinstance(input_ids, list) else [input_ids]

        prompt_wav = self._default_prompt_wav

        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"Token2Wav: prompt_wav file not found: {prompt_wav}")

        waveform_tensor = self.token2wav(input_ids_list, prompt_wav=prompt_wav, return_bytes=False)

        if waveform_tensor.dim() == 2 and waveform_tensor.shape[0] == 1:
            waveform_tensor = waveform_tensor.squeeze(0)

        waveform_tensor_cpu = waveform_tensor.detach().cpu().contiguous()

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [waveform_tensor_cpu]},
        )

    # ------------------------------------------------------------------
    # Async-chunk streaming forward
    # ------------------------------------------------------------------

    def _forward_async_chunk(
        self,
        input_ids: torch.Tensor,
        runtime_additional_information: list[dict[str, Any]],
        **kwargs: Any,
    ) -> OmniOutput:
        """Handle one async_chunk forward step (batch=1 on the legacy path).

        The end-of-stream flag is encoded as ``left_context_size == 1``
        (the only field the framework transparently forwards from the
        async_chunk payload to ``request.additional_information``).
        """
        if self._batched_enabled:
            return self._forward_async_chunk_batched(input_ids, runtime_additional_information, **kwargs)

        # --- batch=1 guard ---
        batch_size = sum(1 for info in runtime_additional_information if _get_left_context_size(info) is not None)
        if batch_size != 1:
            raise RuntimeError(
                f"Token2Wav async_chunk only supports batch=1, got {batch_size}. "
                "Batch>1 requires framework support for batch_req_ids."
            )

        info = next(info for info in runtime_additional_information if _get_left_context_size(info) is not None)
        last_chunk = _get_left_context_size(info) == 1

        # --- Manage single stream state ---
        # If the previous request was preempted (setup_done but not finished),
        # its stale caches must be discarded before we start a new session.
        if self._stream_states and not self._stream_states[0].finished:
            state = self._stream_states[0]
            if state.setup_done and not last_chunk:
                # Heuristic: a non-last chunk arriving into an already-active
                # state that we did NOT set up this call means a new request
                # reused the slot.  For a genuinely continuing request the
                # state is fine; we cannot distinguish without request IDs,
                # so we only forcibly reset on an explicit new-session signal
                # (setup_done=True but stream_cache already consumed all
                # previous data -- detected below when stream_chunk fails).
                pass
        if not self._stream_states or self._stream_states[0].finished:
            self._stream_states = [_StreamState()]
        state = self._stream_states[0]

        # --- Extract audio tokens ---
        # Kept on device; stream_chunk_for takes the tensor directly.
        audio_tokens = input_ids.flatten()

        # Empty chunk (e.g. EOF with no audio tokens)
        if audio_tokens.numel() == 0:
            if state.setup_done:
                self.token2wav.reset_stream_for(state)
            else:
                state.finished = True
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [torch.zeros(1, dtype=torch.float32)]},
            )

        prompt_wav = self._default_prompt_wav
        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"Token2Wav: prompt_wav file not found: {prompt_wav}")

        try:
            if not state.setup_done:
                logger.info("Token2Wav: initialising streaming caches")
                self.token2wav.setup_stream_for(prompt_wav, state)

            waveform = self.token2wav.stream_chunk_for(
                audio_tokens,
                prompt_wav=prompt_wav,
                last_chunk=last_chunk,
                state=state,
            )
        except Exception:
            logger.exception("Token2Wav: stream error, resetting state")
            self.token2wav.reset_stream_for(state)
            raise

        if last_chunk:
            self.token2wav.reset_stream_for(state)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": [waveform.detach().cpu().contiguous()]},
        )

    @staticmethod
    def _split_flat_tokens(flat: torch.Tensor, counts: Any, key: str) -> list[torch.Tensor]:
        """Split a flat batch ``input_ids`` into per-request token segments.

        Same contract as the MiniCPM-o facade's segment splitting: the runner's
        ``seq_token_counts`` (or the payload meta's per-entry counts) must sum
        to the flat token count, or the batch cannot be mapped to requests.
        """
        if counts is None or isinstance(counts, (str, bytes, bytearray)):
            raise RuntimeError(f"Token2Wav batched path cannot split input_ids: invalid {key}")
        if not isinstance(counts, Sequence):
            raise RuntimeError(f"Token2Wav batched path cannot split input_ids: {key} is not a sequence")
        normalized = [int(value) for value in counts]
        if any(value < 0 for value in normalized):
            raise RuntimeError(f"Token2Wav batched path got negative {key}: {normalized}")
        if sum(normalized) != int(flat.numel()):
            raise RuntimeError(
                f"Token2Wav batched path {key} sum {sum(normalized)} != {int(flat.numel())} input tokens"
            )
        return list(torch.split(flat, normalized))

    def _forward_async_chunk_batched(
        self,
        input_ids: torch.Tensor,
        runtime_additional_information: list[dict[str, Any]],
        **kwargs: Any,
    ) -> OmniOutput:
        """Async-chunk streaming through the opt-in BatchedToken2Wav backend.

        One ragged decode per forward step covers every scheduled request —
        including mixed steady/final chunks of different widths — with
        per-request states keyed by request id, so batch>1 works without the
        legacy path's batch=1 guard. Audio returns device-resident (the same
        contract the MiniCPM-o code2wav stage uses).
        """
        if self._batched_backend is None:
            self._build_batched_backend()
        backend = self._batched_backend
        assert backend is not None
        if self._batched_precapture_pending:
            # Runner without the warmup hook: sweep here (first chunk) rather
            # than capturing lazily mid-stream.
            self._precapture_backend_graphs()

        prompt_wav = self._default_prompt_wav
        if not os.path.exists(prompt_wav):
            raise FileNotFoundError(f"Token2Wav: prompt_wav file not found: {prompt_wav}")

        # --- parse per-request payloads (before segmenting: the fallback
        # needs to know which entries actually carry tokens) ---
        if isinstance(input_ids, torch.Tensor):
            flat = input_ids.reshape(-1)
        else:
            flat = torch.empty(0, dtype=torch.long)
        total = len(runtime_additional_information)
        request_ids = kwargs.get("request_ids")
        device = flat.device if isinstance(flat, torch.Tensor) else torch.device("cpu")
        placeholder = torch.zeros(1, dtype=torch.float32, device=device)
        outputs: list[torch.Tensor] = [placeholder] * total
        parsed: list[tuple[int, str, bool] | None] = [None] * total
        for index, info in enumerate(runtime_additional_information):
            meta = info.get("meta") if isinstance(info, dict) else None
            if not isinstance(meta, dict):
                meta = info if isinstance(info, dict) else {}
            marker = meta.get("left_context_size")
            if marker is None and meta.get("last_chunk") is None:
                continue  # bookkeeping-only step: no payload to decode
            last_chunk = bool(meta["last_chunk"]) if meta.get("last_chunk") is not None else marker == 1
            request_id = str(
                meta.get("request_id")
                or (request_ids[index] if request_ids is not None and index < len(request_ids) else "")
                or index
            )
            parsed[index] = (index, request_id, last_chunk)

        # --- split the flat input_ids into per-request segments ---
        payload_indexes = [index for index, entry in enumerate(parsed) if entry is not None]
        runner_counts = kwargs.get("seq_token_counts")
        if runner_counts is not None:
            segments = self._split_flat_tokens(flat, runner_counts, "seq_token_counts")
        else:
            # Fallback: per-entry payload sizes stamped by the stage input
            # processor (valid when every scheduled entry carries a payload).
            meta_counts = [
                (
                    int(info["meta"]["code_flat_numel"])
                    if isinstance(info, dict)
                    and isinstance(info.get("meta"), dict)
                    and info["meta"].get("code_flat_numel") is not None
                    else None
                )
                for info in runtime_additional_information
            ]
            if all(count is not None for count in meta_counts) and sum(meta_counts) == int(flat.numel()):
                segments = list(torch.split(flat, meta_counts))
            elif len(payload_indexes) == 1:
                # No usable per-request counts and only one entry could own
                # the tokens: give them all to that entry.
                only = payload_indexes[0]
                segments = [flat[:0]] * total
                segments[only] = flat
            else:
                raise RuntimeError(
                    "Token2Wav batched path cannot split input_ids across "
                    f"{len(payload_indexes)} payload requests without seq_token_counts "
                    "or per-payload code_flat_numel"
                )
        if len(segments) != total:
            raise RuntimeError(f"Token2Wav batched path: {len(segments)} token segments != {total} scheduled requests")

        rows: list[tuple[int, str, torch.Tensor, bool]] = []
        for index, entry in enumerate(parsed):
            if entry is None:
                continue
            _, request_id, last_chunk = entry
            segment = segments[index]
            if segment.numel() == 0:
                # Empty payload (EOF marker with no audio tokens).
                self._batched_states.pop(request_id, None)
                continue
            rows.append((index, request_id, segment, last_chunk))

        if rows:
            # The same context ``capture_auxiliary_graphs`` swept under, so a
            # graph captured there replays the kernels recorded here.
            with torch.inference_mode(), _stage_matmul_policy(self._batched_tf32):
                try:
                    features = backend.prepare_prompt(_BATCHED_PROMPT_CACHE_ID, prompt_wav)
                    tokens = [row[2].to(device=self.token2wav.device) for row in rows]
                    states = []
                    for _, request_id, _, _ in rows:
                        state = self._batched_states.pop(request_id, None)
                        if state is None:
                            state = backend.setup_batch(features, 1)[0]
                        states.append(state)
                    last_chunks = [row[3] for row in rows]
                    # Uniform rows (same cache signature, token width and
                    # final flag) decode through one stacked decode_batch —
                    # the Whole-Euler graph path; ragged only for genuinely
                    # mixed rows, whose per-request attention solve is slower
                    # (measured 2.2x at batch 1 on H100). Same dispatch rule
                    # as the MiniCPM-o facade's _decode_rows.
                    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
                        state_shape_signature,
                    )

                    if (
                        len({state_shape_signature(state) for state in states}) > 1
                        or len({int(t.numel()) for t in tokens}) > 1
                        or len(set(last_chunks)) > 1
                    ):
                        audios, next_states = backend.decode_ragged_batch(
                            tokens, features, states, last_chunks=last_chunks
                        )
                    else:
                        audios, next_states = backend.decode_batch(
                            torch.stack(tokens, dim=0), features, states, last_chunk=last_chunks[0]
                        )
                except Exception:
                    # Drop this step's states so a retry starts from a fresh
                    # setup instead of continuing from a half-consumed cache.
                    for _, request_id, _, _ in rows:
                        self._batched_states.pop(request_id, None)
                    raise
                for (index, request_id, _, last_chunk), audio, next_state in zip(rows, audios, next_states):
                    outputs[index] = audio.reshape(-1).to(dtype=torch.float32)
                    if last_chunk:
                        self._batched_states.pop(request_id, None)
                    else:
                        self._batched_states[request_id] = next_state
                        self._batched_states.move_to_end(request_id)
                while len(self._batched_states) > self._batched_states_cap:
                    self._batched_states.popitem(last=False)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": outputs},
        )

    # ------------------------------------------------------------------
    # Opt-in BatchedToken2Wav backend (MiniCPM-o engine over this core)
    # ------------------------------------------------------------------

    _BATCHED_KNOBS = (
        "token2wav_batched_backend",
        "token2wav_allow_tf32",
        "token2wav_device_spk",
        "token2wav_setup_cache_size",
        "token2wav_graph_precapture",
        "token2wav_hift_capture_batch_sizes",
        "enable_hift_graph",
        "enable_cfm_graph",
        "enable_whole_euler",
        "cfm_fused_body",
    )

    def _batched_extra(self) -> dict[str, Any]:
        """Collect the batched-backend knobs from the HF config.

        The knobs arrive through the stage's ``hf_overrides`` (the same route
        as ``architectures`` and ``token2wav_float16``), so they surface as
        attributes on the HF config. Unset keys read as ``None`` so
        ``read_bool_flag`` falls back to its default.
        """
        return {key: getattr(self.config, key, None) for key in self._BATCHED_KNOBS}

    def _build_batched_backend(self) -> None:
        """Construct the opt-in ``BatchedToken2Wav`` engine over the core.

        Mirrors the MiniCPM-o facade's ``_build_backend`` contract: the build
        must run outside inference mode (parameters keep version counters —
        ``dit_fused``'s packed-QKV cache reads ``t._version``, which inference
        tensors lack), while prompt setup, graph capture and decode run under
        the caller's inference mode (graph replay does in-place copies into
        static buffers). The wrapper wraps itself in the right contexts so
        callers don't have to.
        """
        if self._batched_backend is not None or not self._batched_enabled:
            return
        from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
        from vllm_omni.platforms import current_omni_platform

        extra = self._batched_extra()
        is_cuda = current_omni_platform.is_cuda()
        with torch.inference_mode(False), torch.no_grad():
            self.token2wav._ensure_models_loaded()
            # The external stepaudio2.Token2wav builds speech_window eagerly
            # in its __init__; the core builds it lazily inside
            # setup_stream_for, but the backend clones it at construction
            # time (MiniCPM's facade materializes it for the same reason).
            if self.token2wav.speech_window is None:
                self.token2wav.speech_window = torch.from_numpy(np.hamming(2 * self.token2wav.source_cache_len)).to(
                    device=self.token2wav.device, dtype=torch.float32
                )
            # Match the MiniCPM-o CUDA wiring: campplus speaker embedding on
            # device (torch, weights from the same campplus.onnx) instead of
            # the per-request CPU onnxruntime session. NPU already enabled it
            # in __init__.
            if is_cuda and read_bool_flag(extra, "token2wav_device_spk", True):
                self.token2wav.enable_device_spk_embedding(use_trt=False)
                logger.info("Step-Audio2 Token2Wav: campplus speaker embedding running on device (torch)")

            # HiFT capture batches: powers of two below the scheduler's batch
            # (capped at 32) and the cap itself — same sizing rule as the
            # MiniCPM-o facade.
            scheduler = getattr(self.vllm_config, "scheduler_config", None)
            max_num_seqs = getattr(scheduler, "max_num_seqs", None)
            raw_capture_sizes = extra.get("token2wav_hift_capture_batch_sizes")
            batch_cap = min(int(max_num_seqs), 32) if max_num_seqs else 32
            capture_batch_sizes = (
                [1 << i for i in range(batch_cap.bit_length()) if 1 << i < batch_cap] + [batch_cap]
                if raw_capture_sizes is None
                else list(raw_capture_sizes)
            )
            setup_cache_size = int(extra.get("token2wav_setup_cache_size") or 1)
            if setup_cache_size < 0:
                raise ValueError("Step-Audio2 token2wav_setup_cache_size must be >= 0")

            self._batched_backend = BatchedToken2Wav(
                self.token2wav,
                connector_config={
                    # Chunk geometry from the streaming SSOT — the stage input
                    # processor slices payloads with the same constants.
                    "codec_chunk_frames": DEFAULT_STREAM_CONFIG.chunk_size,
                    "initial_codec_chunk_frames": 0,
                    "codec_left_context_frames": DEFAULT_STREAM_CONFIG.pre_lookahead_len,
                    "hift_max_lazy_graphs": 8,
                    "hift_graph_exact_batch_sizes": (1,),
                },
                hift_graph_config={
                    "enabled": read_bool_flag(extra, "enable_hift_graph", False),
                    "capture_batch_sizes": capture_batch_sizes,
                    "max_serial_batch": 4,
                },
                cfm_graph_config={
                    "enabled": read_bool_flag(extra, "enable_cfm_graph", False),
                    "enable_whole_euler": read_bool_flag(extra, "enable_whole_euler", True),
                    "max_serial_batch": 4,
                    "max_graphs": 32,
                    # Default on for CUDA (with TF32) and NPU, matching the
                    # MiniCPM-o facade; an explicit knob overrides either way.
                    "fused_body": read_bool_flag(
                        extra,
                        "cfm_fused_body",
                        (is_cuda and self._batched_tf32) or current_omni_platform.is_npu(),
                    ),
                },
                setup_cache_size=setup_cache_size,
                cfm_tf32=is_cuda and self._batched_tf32,
            )
        logger.info(
            "Step-Audio2 Token2Wav: BatchedToken2Wav backend enabled (tf32=%s, hift_graph=%s, cfm_graph=%s)",
            self._batched_tf32,
            read_bool_flag(extra, "enable_hift_graph", False),
            read_bool_flag(extra, "enable_cfm_graph", False),
        )

    def capture_auxiliary_graphs(self) -> None:
        """Warm the opt-in backend's device graphs from the worker warmup hook.

        The worker calls this after its own init work (weights loaded,
        profiling done): the MiniCPM-o facade measured that capturing during
        weight load raises the reserved footprint by ~4 GiB, enough to OOM at
        concurrency, so the sweep must not run in ``__init__``/``load_weights``.
        Runners without this hook defer the sweep to the first request.

        Per-family failures fall back to lazy capture, but a poisoned graph
        wrapper fails startup here rather than the first user request (same
        policy as the MiniCPM-o facade).
        """
        if self._batched_backend is None:
            if not self._batched_enabled:
                return
            self._build_batched_backend()
        self._precapture_backend_graphs()

    def _precapture_backend_graphs(self) -> None:
        """Run the pending graph sweep once, under the context forward uses.

        Inference mode plus this stage's matmul policy, so a graph captured
        here replays the kernels the decode path records. No-op when the
        sweep already ran or the knob is off.
        """
        if not self._batched_precapture_pending or self._batched_backend is None:
            return
        self._batched_precapture_pending = False
        backend = self._batched_backend
        started = time.perf_counter()

        def precapture(name: str, capture: Callable[[], int]) -> None:
            try:
                captured = capture()
            except Exception:
                # Encoder capture failures poison their CUDA/NPU wrapper.
                # Fail startup rather than promise lazy capture and fail the
                # first user request with a restart-required error.
                if any(
                    getattr(getattr(backend, attr, None), "_failed", False)
                    for attr in (
                        "_encoder_graphs",
                        "_chunk_encoder_graph",
                        "hift_graph_wrapper",
                        "_cfm_graph_wrapper",
                        "_whole_euler_graph_wrapper",
                    )
                ):
                    raise
                logger.warning(
                    "Step-Audio2 Token2Wav: %s precapture failed; graphs capture lazily",
                    name,
                    exc_info=True,
                )
                return
            if captured:
                logger.info(
                    "Step-Audio2 Token2Wav: precaptured %d %s graphs in %.1f s",
                    captured,
                    name,
                    time.perf_counter() - started,
                )

        # The sweep runs under the same context ``forward`` will use, so a
        # graph captured here replays the kernels the lazy path records.
        with torch.inference_mode(), _stage_matmul_policy(self._batched_tf32):
            precapture("HiFT", backend.precapture_hift)
            try:
                features = backend.prepare_prompt(_BATCHED_PROMPT_CACHE_ID, self._default_prompt_wav)
            except Exception:
                logger.warning(
                    "Step-Audio2 Token2Wav: default prompt failed; graphs capture lazily",
                    exc_info=True,
                )
                return
            precapture("Whole-Euler", lambda: backend.precapture_whole_euler(features))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        for name, _ in self.token2wav.named_buffers():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params

    def load_weights_without_buffers(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        for name, _ in self.token2wav.named_parameters():
            loaded_params.add(f"token2wav.{name}")
        return loaded_params
