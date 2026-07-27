# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TensorRT engine for the CosyVoice3 flow-decoder (CFM) DiT estimator.

The estimator is the per-step network the conditional flow-matching ODE solver
calls during code2wav (token -> mel). It dominates code2wav latency; the
upstream ``CausalConditionalCFM.forward_estimator`` already supports running it
through a TensorRT engine (it switches on ``self.estimator`` not being an
``nn.Module`` and drives it via ``acquire_estimator`` / ``execute_async_v3``).
This module builds that engine from the bundled ``flow.decoder.estimator*.onnx``
and wraps it so it can be dropped in for the torch estimator.

The estimator engine has 6 inputs ``x, mask, mu, t, spks, cond`` and one output.
``x/mask/mu/cond`` carry a dynamic time dim; ``t``/``spks`` are fixed, so only
the former need an optimization profile (TRT infers the latter from the ONNX).
Shapes mirror the CosyVoice runtime.

Precision: TensorRT >= 11 dropped the weakly-typed FP16/INT8 builder flags, so
fp16 only comes from a STRONGLY_TYPED network built from an fp16 ONNX
(``*autocast_fp16*``, fp16 I/O). An fp32 ONNX is built fp32 + the TF32 matmul
flag. ``EXPLICIT_BATCH`` is implicit (no flag) and ``ITensor.dtype`` is
read-only, so neither is set here.
"""

from __future__ import annotations

import os
import queue

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import (
    _resolve_plan_path,
    _trt_logger,
)

logger = init_logger(__name__)

# Optimization-profile shapes for the dynamic-length inputs (CFG batch dim = 2,
# 80 mel channels, time dim min/opt/max). Matches CosyVoice's token2wav.
_DYNAMIC_INPUTS = ("x", "mask", "mu", "cond")
_MIN_SHAPES = ((2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4))
_OPT_SHAPES = ((2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500))
_MAX_SHAPES = ((2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000))


def _is_fp16_onnx(onnx_path: str) -> bool:
    """Heuristic: the project's fp16 estimator ONNX is exported strongly-typed
    and named ``*autocast_fp16*`` / ``*fp16*`` (vs ``*fp32*``)."""
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def _convert_onnx_to_trt(onnx_path: str, plan_path: str, strongly_typed: bool) -> None:
    import tensorrt as trt

    logger.info(
        "Building flow-estimator TensorRT engine from %s (%s) ...",
        onnx_path,
        "strongly-typed/fp16" if strongly_typed else "fp32+TF32",
    )
    trt_logger = _trt_logger()
    builder = trt.Builder(trt_logger)
    # STRONGLY_TYPED takes precision from the ONNX graph (fp16 engine from an
    # fp16 ONNX) — this is the only way to get fp16 in TRT>=11, which dropped
    # the weakly-typed FP16 BuilderFlag. Otherwise EXPLICIT_BATCH is implicit,
    # so create the network with no flags.
    if strongly_typed:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    else:
        network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4 GiB
    if not strongly_typed:
        # fp32 ONNX: enable the best available reduced-precision matmul flag
        # (FP16 on older TRT, else TF32 on Ampere+/Hopper). For a strongly-typed
        # network the precision is fixed by the graph, so no flag is set.
        for _flag_name in ("FP16", "TF32"):
            _flag = getattr(trt.BuilderFlag, _flag_name, None)
            if _flag is not None:
                config.set_flag(_flag)
                break

    profile = builder.create_optimization_profile()
    for name, mn, op, mx in zip(_DYNAMIC_INPUTS, _MIN_SHAPES, _OPT_SHAPES, _MAX_SHAPES):
        profile.set_shape(name, mn, op, mx)
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build flow-estimator engine from {onnx_path}")
    tmp = plan_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(engine_bytes)
    os.replace(tmp, plan_path)
    logger.info("Wrote flow-estimator TensorRT engine to %s", plan_path)


class TrtContextWrapper:
    """Pool of TensorRT execution contexts for the flow estimator.

    Exposes the ``acquire_estimator`` / ``release_estimator`` contract that
    ``CausalConditionalCFM.forward_estimator`` expects.
    """

    def __init__(
        self, engine, device: str | torch.device, io_dtype: torch.dtype = torch.float32, trt_concurrent: int = 1
    ):
        self.trt_engine = engine
        # Engine I/O dtype (fp16 for a strongly-typed fp16 engine). The flow runs
        # in fp32, so forward_estimator casts to/from this at the boundary.
        self.io_dtype = io_dtype
        self._pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
        for _ in range(trt_concurrent):
            ctx = engine.create_execution_context()
            assert ctx is not None, "failed to create TRT execution context (out of memory?)"
            stream = torch.cuda.stream(torch.cuda.Stream(torch.device(device)))
            self._pool.put([ctx, stream])

    def acquire_estimator(self):
        return self._pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self._pool.put([context, stream])


def build_flow_estimator_trt(onnx_path: str, device: str | torch.device) -> TrtContextWrapper:
    """Build/load the flow-estimator TRT engine and return a context-pool wrapper.

    An fp16 ONNX (``*autocast_fp16*``) is built as a strongly-typed network (the
    only way to get fp16 in TRT>=11); an fp32 ONNX is built fp32 + TF32.
    """
    import tensorrt as trt

    strongly_typed = _is_fp16_onnx(onnx_path)
    plan_path = _resolve_plan_path(onnx_path, prefix="flow_estimator")
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(onnx_path, plan_path, strongly_typed=strongly_typed)

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize flow-estimator TensorRT engine {plan_path}")
    logger.info("Loaded flow-estimator TensorRT engine (%s)", plan_path)
    io_dtype = torch.float16 if strongly_typed else torch.float32
    return TrtContextWrapper(engine, device=device, io_dtype=io_dtype)
