# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TensorRT speaker-embedding (campplus) extractor for CosyVoice3.

Drop-in accelerator for the ONNX-Runtime campplus session used in the
multimodal processor. ``campplus.onnx`` maps an fbank feature
``[B, T, 80]`` to a 192-d speaker embedding ``[B, 192]``; on this stack the
ONNX path only has a CPU execution provider, so it dominates per-request
prompt-processing latency. Building a TensorRT engine once and running it on
GPU removes that bottleneck.

The engine is built lazily from ``campplus.onnx`` on first use and cached as
a ``.plan`` file (engines are device/TRT-version specific, so the cache key
includes both). Set ``COSYVOICE3_TRT_CACHE`` to override the cache dir.

Written against TensorRT >= 10/11: ``EXPLICIT_BATCH`` is the implicit default
(create the network with no flags) and ``ITensor.dtype`` is read-only (I/O
dtypes come from the ONNX graph), so neither is set here.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from uuid import uuid4

import torch
from filelock import FileLock
from vllm.logger import init_logger

logger = init_logger(__name__)

# campplus fbank sequence length bounds for the optimization profile.
_MIN_T = 4
_OPT_T = 500
_MAX_T = 3000
_EMBED_DIM = 192


def _cpu_reference_onnx(onnx_path: str) -> str:
    """Materialize portable ORT optimizations before handing the graph to TRT.

    The CPU reference uses ORT_ENABLE_ALL. Its basic PadFusion can change a
    ceil-mode AveragePool's count_include_pad, including when the explicit
    Pad is zero. Parsing the original graph in TRT would then compute a
    different speaker embedding. Export BASIC (standard ONNX operators), not
    provider-specific ALL fusions, to retain the CPU reference's pool semantics.
    """
    import onnxruntime

    root = Path(os.environ.get("COSYVOICE3_TRT_CACHE") or Path.home() / ".cache/vllm_omni/cosyvoice3_trt")
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(onnx_path).read_bytes())
    digest.update(f"|ort{onnxruntime.__version__}|basic-v1".encode())
    target = root / f"campplus_ort_{digest.hexdigest()}.onnx"
    if target.is_file() and target.stat().st_size:
        return str(target)
    with FileLock(str(target) + ".lock"):
        if target.is_file() and target.stat().st_size:
            return str(target)
        fd, temporary = _create_cache_temporary(str(target))
        os.close(fd)
        try:
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 1
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
            options.optimized_model_filepath = temporary
            session = onnxruntime.InferenceSession(onnx_path, sess_options=options, providers=["CPUExecutionProvider"])
            del session
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return str(target)


def _trt_logger():
    import tensorrt as trt

    return trt.Logger(trt.Logger.WARNING)


def _convert_onnx_to_trt(onnx_path: str, plan_path: str, *, allow_tf32: bool = True) -> None:
    import tensorrt as trt

    logger.info("Building campplus TensorRT engine from %s ...", onnx_path)
    trt_logger = _trt_logger()
    builder = trt.Builder(trt_logger)
    # EXPLICIT_BATCH is implicit in TRT>=10; create the network with no flags.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    if not allow_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 31)  # 2 GiB

    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (1, _MIN_T, 80),
        (1, _OPT_T, 80),
        (1, _MAX_T, 80),
    )
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build campplus engine from {onnx_path}")
    _write_plan_atomic(plan_path, engine_bytes)
    logger.info("Wrote campplus TensorRT engine to %s", plan_path)


def _create_cache_temporary(target: str) -> tuple[int, str]:
    """Create an exclusive sibling using normal file permissions and umask.

    Unlike mkstemp's fixed 0600 mode, O_CREAT with 0666 retains group/read
    access allowed by the caller's umask without changing process-wide state.
    """
    temporary = f"{target}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    return descriptor, temporary


def _write_plan_atomic(plan_path: str, payload: bytes) -> None:
    """Publish one complete engine.

    A fixed ``plan_path + '.tmp'`` is shared by every builder. Two processes
    can interleave writes there and ``os.replace`` a torn plan into place.
    """
    descriptor, temporary = _create_cache_temporary(plan_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, plan_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _ensure_plan(onnx_path: str, plan_path: str, *, allow_tf32: bool) -> None:
    """Build a missing engine once for every process sharing the plan cache."""
    if os.path.isfile(plan_path) and os.path.getsize(plan_path):
        return
    with FileLock(plan_path + ".lock"):
        if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
            _convert_onnx_to_trt(onnx_path, plan_path, allow_tf32=allow_tf32)


def _resolve_plan_path(onnx_path: str, prefix: str = "campplus") -> str:
    cache_dir = os.environ.get("COSYVOICE3_TRT_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "vllm_omni", "cosyvoice3_trt"
    )
    os.makedirs(cache_dir, exist_ok=True)
    try:
        # No-arg form targets the current device (avoids the banned
        # torch.cuda.current_device per ruff TID251).
        dev_name = torch.cuda.get_device_name()
    except Exception:
        dev_name = "unknown"
    import tensorrt as trt

    st = os.stat(onnx_path)
    key = f"{os.path.abspath(onnx_path)}|{st.st_size}|{int(st.st_mtime)}|{dev_name}|trt{trt.__version__}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"{prefix}_{digest}.plan")


class CampplusTRT:
    """Lazily-built TensorRT campplus speaker-embedding extractor."""

    def __init__(self, onnx_path: str, device: str | torch.device, *, allow_tf32: bool = True):
        import tensorrt as trt

        self.device = torch.device(device)
        plan_path = _resolve_plan_path(onnx_path, prefix="campplus" if allow_tf32 else "campplus_ieee")
        _ensure_plan(onnx_path, plan_path, allow_tf32=allow_tf32)

        runtime = trt.Runtime(_trt_logger())
        with open(plan_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize campplus TensorRT engine {plan_path}")
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        # TRT execution contexts are not safe for concurrent enqueue; serialize.
        self._lock = threading.Lock()
        logger.info("Loaded campplus TensorRT engine (%s)", plan_path)

    @torch.inference_mode()
    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: ``[T, 80]`` fbank features. Returns ``[1, 192]`` embedding."""
        frames = int(feat.shape[0]) if feat.ndim == 2 else -1
        if feat.ndim != 2 or int(feat.shape[1]) != 80 or frames < _MIN_T or frames > _MAX_T:
            raise ValueError(
                f"campplus TensorRT accepts fbank [T, 80] with T in {_MIN_T}..{_MAX_T}, got {tuple(feat.shape)}"
            )
        x = feat.unsqueeze(0).to(self.device, dtype=torch.float32).contiguous()
        out = torch.empty((x.shape[0], _EMBED_DIM), device=self.device, dtype=torch.float32)
        stream = torch.cuda.current_stream(self.device)
        with self._lock:
            self.context.set_input_shape(self.input_name, tuple(x.shape))
            self.context.set_tensor_address(self.input_name, x.data_ptr())
            self.context.set_tensor_address(self.output_name, out.data_ptr())
            ok = self.context.execute_async_v3(stream.cuda_stream)
            if not ok:
                raise RuntimeError("campplus TensorRT execute_async_v3 failed")
            stream.synchronize()
        return out


# Process-wide cache: the mm processor is re-created per request (the deploy
# config disables the mm-processor object cache), so building a fresh
# CampplusTRT each time would re-deserialize the ~33 MB engine and allocate a
# new execution context on every request — inflating TTFP. Cache by
# (ONNX path, device, TF32 policy) and reuse; the context serializes concurrent enqueues
# via its own lock, so a single shared instance is safe across requests.
_CAMPPLUS_CACHE: dict[tuple[str, str, bool], CampplusTRT] = {}


def get_campplus_trt(onnx_path: str, device: str | torch.device, *, cpu_reference: bool = False) -> CampplusTRT:
    """Return a process-wide cached :class:`CampplusTRT` (engine + context)."""
    if cpu_reference:
        onnx_path = _cpu_reference_onnx(onnx_path)
    # Preserve the CPU reference's FP32 arithmetic as well as its graph.
    allow_tf32 = not cpu_reference
    key = (os.path.abspath(onnx_path), str(device), allow_tf32)
    inst = _CAMPPLUS_CACHE.get(key)
    if inst is None:
        inst = CampplusTRT(onnx_path, device, allow_tf32=allow_tf32)
        _CAMPPLUS_CACHE[key] = inst
    return inst
