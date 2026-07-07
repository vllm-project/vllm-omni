# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# campplus fbank sequence length bounds for the optimization profile.
_MIN_T = 4
_OPT_T = 500
_MAX_T = 3000
_EMBED_DIM = 192


def _trt_logger():
    import tensorrt as trt

    return trt.Logger(trt.Logger.WARNING)


def _convert_onnx_to_trt(onnx_path: str, plan_path: str) -> None:
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
    tmp = plan_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(engine_bytes)
    os.replace(tmp, plan_path)
    logger.info("Wrote campplus TensorRT engine to %s", plan_path)


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

    def __init__(self, onnx_path: str, device: str | torch.device):
        import tensorrt as trt

        self.device = torch.device(device)
        plan_path = _resolve_plan_path(onnx_path)
        if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
            _convert_onnx_to_trt(onnx_path, plan_path)

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
# (engine path, device) and reuse; the context serializes concurrent enqueues
# via its own lock, so a single shared instance is safe across requests.
_CAMPPLUS_CACHE: dict[tuple[str, str], CampplusTRT] = {}


def get_campplus_trt(onnx_path: str, device: str | torch.device) -> CampplusTRT:
    """Return a process-wide cached :class:`CampplusTRT` (engine + context)."""
    key = (os.path.abspath(onnx_path), str(device))
    inst = _CAMPPLUS_CACHE.get(key)
    if inst is None:
        inst = CampplusTRT(onnx_path, device)
        _CAMPPLUS_CACHE[key] = inst
    return inst
