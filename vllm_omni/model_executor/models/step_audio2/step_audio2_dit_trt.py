# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TensorRT acceleration for the Step-Audio2 / CosyVoice2 flow DiT estimator.

Ports stepfun-ai/Step-Audio2#70 into vLLM-Omni. The upstream PR patches
``cosyvoice2.flow.flow_matching`` to dispatch between a torch estimator and a
raw TRT execution context; here the swap happens one level up instead — the
batched Code2Wav driver (``BatchedToken2Wav._estimator_step``) already owns
the per-timestep estimator call, so a :class:`TrtDiTStepper` replaces only
that call and the pip-installed ``cosyvoice2`` package stays untouched.

Differences from the upstream PR:

* ONNX export is a function (:func:`export_dit_chunk_onnx`) that wraps the
  already-loaded torch ``DiT`` module and runs at engine-build time — no
  separate ``tools/export_onnx_*.py`` step.
* The engine executes on the caller's current CUDA stream, so no context
  pool, no private stream, and no ``synchronize()`` per call (a private
  stream without synchronization produces silent garbage audio; with
  synchronization it costs two round-trips per denoise step).
* Cache tensors are exchanged as stacked ``(depth, 2B, ...)`` tensors, the
  same layout ``BatchedToken2Wav`` already uses.
"""

from __future__ import annotations

import contextlib
import os
import stat
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRT_CACHE_ENV = "MINICPMO_TOKEN2WAV_TRT_CACHE"


def _publish_atomically(
    path: Path,
    writer: Callable[[Path, BinaryIO], None],
    *,
    requires_path_write: bool = False,
) -> None:
    """Publish an artifact through an exclusively owned same-directory temp."""
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    stream = temporary.open("xb")
    published = False
    try:
        original_mode: int | None = None
        mode_adjusted = False
        write_completed = False
        try:
            if requires_path_write:
                original_mode = stat.S_IMODE(os.fstat(stream.fileno()).st_mode)
                mode_adjusted = not original_mode & stat.S_IWUSR
                if mode_adjusted:
                    os.fchmod(stream.fileno(), original_mode | stat.S_IWUSR)
            writer(temporary, stream)
            stream.flush()
            if mode_adjusted:
                os.fchmod(stream.fileno(), original_mode)
            write_completed = True
        finally:
            if not stream.closed:
                if not write_completed and mode_adjusted and original_mode is not None:
                    with contextlib.suppress(OSError):
                        os.fchmod(stream.fileno(), original_mode)
                if write_completed:
                    stream.close()
                else:
                    with contextlib.suppress(OSError):
                        stream.close()
        os.replace(temporary, path)
        published = True
    finally:
        if not published:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                with contextlib.suppress(Exception):
                    logger.warning("Failed to remove temporary StepAudio2 build artifact %s", temporary, exc_info=True)


class _DiTChunkWrapper(nn.Module):
    """ONNX-exportable streaming ``forward_chunk`` with explicit stacked caches.

    Mirrors ``DiT.forward_chunk`` (cosyvoice2 pip package) but takes and
    returns caches as plain tensors instead of registered buffers, matching
    the I/O contract of the TRT engine and of ``BatchedToken2Wav``.
    """

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit

    def forward(
        self,
        x: torch.Tensor,  # (2B, mel, T)
        mu: torch.Tensor,  # (2B, mel, T)
        t: torch.Tensor,  # (2B,)
        spks: torch.Tensor,  # (2B, spk)
        cond: torch.Tensor,  # (2B, mel, T)
        cnn_cache: torch.Tensor,  # (depth, 2B, c1+c2, kernel-1)
        att_cache: torch.Tensor,  # (depth, 2B, heads, L, head_dim*2)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_emb = self.dit.t_embedder(t).unsqueeze(1)
        spks_f = spks.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        h = torch.cat((x, mu, spks_f, cond), dim=1).transpose(1, 2)
        h = self.dit.in_proj(h)
        new_cnn = []
        new_att = []
        for i, block in enumerate(self.dit.blocks):
            h, block_cnn, block_att = block.forward_chunk(h, t_emb, cnn_cache[i], att_cache[i], None)
            new_cnn.append(block_cnn)
            new_att.append(block_att)
        h = self.dit.final_layer(h, t_emb)
        return h.transpose(1, 2), torch.stack(new_cnn), torch.stack(new_att)


def _dit_dims(dit: nn.Module) -> dict[str, int]:
    block0 = dit.blocks[0]
    return {
        "depth": len(dit.blocks),
        "mel": int(dit.out_channels),
        "spk": int(dit.in_channels) - 3 * int(dit.out_channels),
        "heads": int(block0.attn.num_heads),
        "att_dim": int(block0.attn.head_dim) * 2,
        "cnn_ch": int(block0.conv.in_channels + block0.conv.out_channels),
        "cnn_w": int(block0.conv.block[1].causal_padding[0]),
    }


def export_dit_chunk_onnx(
    dit: nn.Module,
    onnx_path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Export the streaming DiT chunk graph to ONNX in ``dtype``.

    Fused-in replacement for the upstream ``tools/export_onnx_streaming_token2wav.py``.
    TensorRT >= 10 deprecates weakly-typed networks (TRT 11 removes
    ``BuilderFlag.FP16`` outright), so the engine precision comes from the
    ONNX graph itself. The DiT is deep-copied for the cast — the live torch
    estimator keeps its original dtype.
    """
    import copy

    dims = _dit_dims(dit)
    export_dit = copy.deepcopy(dit).to(device=device, dtype=dtype).eval()
    wrapper = _DiTChunkWrapper(export_dit).eval()
    batch, chunk, cache_len = 2, 48, 16
    sample = (
        torch.randn(batch, dims["mel"], chunk, device=device, dtype=dtype),
        torch.randn(batch, dims["mel"], chunk, device=device, dtype=dtype),
        torch.rand(batch, device=device, dtype=dtype),
        torch.randn(batch, dims["spk"], device=device, dtype=dtype),
        torch.randn(batch, dims["mel"], chunk, device=device, dtype=dtype),
        torch.zeros(dims["depth"], batch, dims["cnn_ch"], dims["cnn_w"], device=device, dtype=dtype),
        torch.randn(dims["depth"], batch, dims["heads"], cache_len, dims["att_dim"], device=device, dtype=dtype),
    )
    input_names = ["x", "mu", "t", "spks", "cond", "cnn_cache", "att_cache"]
    output_names = ["out", "new_cnn_cache", "new_att_cache"]
    dynamic_axes = {
        "x": {0: "b", 2: "t"},
        "mu": {0: "b", 2: "t"},
        "t": {0: "b"},
        "spks": {0: "b"},
        "cond": {0: "b", 2: "t"},
        "cnn_cache": {1: "b"},
        "att_cache": {1: "b", 3: "l"},
        "out": {0: "b", 2: "t"},
        "new_cnn_cache": {1: "b"},
        "new_att_cache": {1: "b", 3: "l_out"},
    }
    logger.info("Exporting DiT chunk ONNX to %s", onnx_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    def write_onnx(temporary: Path, stream: BinaryIO) -> None:
        del stream
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample,
                str(temporary),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                dynamo=False,
            )

    _publish_atomically(onnx_path, write_onnx, requires_path_write=True)


def convert_onnx_to_trt(
    onnx_path: str | Path,
    plan_path: str | Path,
    dims: dict[str, int],
    *,
    max_batch: int = 16,
    max_len: int = 3000,
    max_cache_len: int = 1000,
) -> None:
    """Build a TRT engine from the exported ONNX (fused-in from upstream tools).

    Engine precision is strongly typed from the ONNX graph itself — export
    the graph in the desired dtype (TRT 11 removed ``BuilderFlag.FP16``).
    """
    import tensorrt as trt

    logger.info("Building DiT TRT engine at %s — this can take a few minutes", plan_path)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    # TRT >= 10 removed EXPLICIT_BATCH (networks are always explicit-batch).
    flag = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    network = builder.create_network(1 << int(flag) if flag is not None else 0)
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise ValueError(f"failed to parse {onnx_path}: {errors}")

    config = builder.create_builder_config()

    depth, mel, spk = dims["depth"], dims["mel"], dims["spk"]
    heads, att_dim = dims["heads"], dims["att_dim"]
    cnn_ch, cnn_w = dims["cnn_ch"], dims["cnn_w"]
    opt_b = min(4, max_batch)
    shapes = {
        "x": [(2, mel, 2), (opt_b, mel, 500), (max_batch, mel, max_len)],
        "mu": [(2, mel, 2), (opt_b, mel, 500), (max_batch, mel, max_len)],
        "t": [(2,), (opt_b,), (max_batch,)],
        "spks": [(2, spk), (opt_b, spk), (max_batch, spk)],
        "cond": [(2, mel, 2), (opt_b, mel, 500), (max_batch, mel, max_len)],
        "cnn_cache": [
            (depth, 2, cnn_ch, cnn_w),
            (depth, opt_b, cnn_ch, cnn_w),
            (depth, max_batch, cnn_ch, cnn_w),
        ],
        "att_cache": [
            (depth, 2, heads, 0, att_dim),
            (depth, opt_b, heads, 500, att_dim),
            (depth, max_batch, heads, max_cache_len, att_dim),
        ],
    }
    profile = builder.create_optimization_profile()
    for name, (mn, op, mx) in shapes.items():
        profile.set_shape(name, mn, op, mx)
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT engine build failed")
    plan_path = Path(plan_path)
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    def write_plan(temporary: Path, stream: BinaryIO) -> None:
        del temporary
        stream.write(engine_bytes)

    _publish_atomically(plan_path, write_plan)
    logger.info("DiT TRT engine written to %s", plan_path)


class TrtDiTStepper:
    """Drop-in TRT replacement for one DiT denoise step.

    ``step`` matches the tensors ``BatchedToken2Wav._decode_cfm`` has in hand
    per timestep. Executes on the caller's current CUDA stream — same-stream
    ordering makes explicit synchronization unnecessary.
    """

    def __init__(
        self,
        plan_path: str | Path,
        dims: dict[str, int],
        device: torch.device,
        dtype: torch.dtype,
        *,
        max_batch: int = 16,
        max_len: int = 3000,
        max_cache_len: int = 1000,
    ):
        import tensorrt as trt

        self.dims = dims
        self.device = device
        self.dtype = dtype
        # Optimization-profile bounds the engine was built with; step() checks
        # them so an out-of-profile call fails with a readable error instead
        # of an opaque execute_async_v3 failure.
        self.max_batch = int(max_batch)
        self.max_len = int(max_len)
        self.max_cache_len = int(max_cache_len)
        with open(plan_path, "rb") as f:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TRT engine {plan_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TRT execution context")
        # TRT rejects null bindings; zero-length cache tensors share this address.
        self._dummy = torch.zeros(1, device=device, dtype=dtype)

    def _addr(self, tensor: torch.Tensor) -> int:
        return self._dummy.data_ptr() if tensor.numel() == 0 else tensor.data_ptr()

    def step(
        self,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dims = self.dims
        batch = int(x.shape[0])
        chunk = int(x.shape[2])
        cache_frames = int(att_cache.shape[3]) if att_cache is not None else 0
        if batch > self.max_batch or chunk > self.max_len or cache_frames > self.max_cache_len:
            raise ValueError(
                f"DiT TRT engine profile exceeded: batch={batch} (max {self.max_batch}, "
                f"= token2wav_trt_max_batch, CFG included), chunk={chunk} frames "
                f"(max {self.max_len}), att cache={cache_frames} frames (max {self.max_cache_len}). "
                "Rebuild the engine with a larger token2wav_trt_max_batch or disable token2wav_trt."
            )
        if cnn_cache is None:
            cnn_cache = torch.zeros(
                (dims["depth"], batch, dims["cnn_ch"], dims["cnn_w"]), device=self.device, dtype=self.dtype
            )
        if att_cache is None:
            att_cache = torch.zeros(
                (dims["depth"], batch, dims["heads"], 0, dims["att_dim"]), device=self.device, dtype=self.dtype
            )
        cache_len = int(att_cache.shape[3])

        inputs = {
            "x": x.to(self.dtype).contiguous(),
            "mu": mu.to(self.dtype).contiguous(),
            "t": t.to(self.dtype).contiguous(),
            "spks": spks.to(self.dtype).contiguous(),
            "cond": cond.to(self.dtype).contiguous(),
            "cnn_cache": cnn_cache.to(self.dtype).contiguous(),
            "att_cache": att_cache.to(self.dtype).contiguous(),
        }
        out = torch.empty((batch, dims["mel"], chunk), device=self.device, dtype=self.dtype)
        new_cnn = torch.empty_like(inputs["cnn_cache"])
        new_att = torch.empty(
            (dims["depth"], batch, dims["heads"], cache_len + chunk, dims["att_dim"]),
            device=self.device,
            dtype=self.dtype,
        )

        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, self._addr(tensor))
        self.context.set_tensor_address("out", out.data_ptr())
        self.context.set_tensor_address("new_cnn_cache", new_cnn.data_ptr())
        self.context.set_tensor_address("new_att_cache", self._addr(new_att))

        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TRT DiT execution failed")
        return out, new_cnn, new_att


def _default_cache_dir() -> Path:
    env = os.environ.get(_TRT_CACHE_ENV, "").strip()
    if env:
        return Path(env)
    hf_home = os.environ.get("HF_HOME", "").strip()
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "vllm_omni" / "token2wav_trt"


def build_dit_trt_stepper(
    dit: nn.Module,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    max_batch: int = 16,
    cache_dir: str | Path | None = None,
) -> TrtDiTStepper:
    """Export (if needed), build (if needed) and load the DiT TRT engine.

    Artifacts are cached under ``cache_dir`` (default: ``$MINICPMO_TOKEN2WAV_TRT_CACHE``
    or ``$HF_HOME/vllm_omni/token2wav_trt``), keyed by dtype/batch/TRT
    version/GPU architecture — serialized plans are SM-specific, so a cache
    shared across heterogeneous machines must not reuse another arch's plan.
    """
    import tensorrt as trt

    device = torch.device(device)
    dims = _dit_dims(dit)
    cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    dtype_tag = str(dtype).replace("torch.", "")
    major, minor = torch.cuda.get_device_capability(device)
    sm_tag = f"sm{major}{minor}"
    onnx_path = cache / f"dit_chunk.d{dims['depth']}.{dtype_tag}.onnx"
    plan_path = cache / (f"dit_chunk.d{dims['depth']}.{dtype_tag}.b{max_batch}.{sm_tag}.trt{trt.__version__}.plan")

    if not plan_path.is_file() or plan_path.stat().st_size == 0:
        if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
            export_dit_chunk_onnx(dit, onnx_path, device, dtype)
        convert_onnx_to_trt(onnx_path, plan_path, dims, max_batch=max_batch)
    else:
        logger.info("Reusing cached DiT TRT engine %s", plan_path)
    return TrtDiTStepper(plan_path, dims, device, dtype, max_batch=max_batch)
