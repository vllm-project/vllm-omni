# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

The bundled ONNX was traced with full attention, so upstream's streaming
chunk-causal mask (``DiT.forward(streaming=True)``) never reaches such an
engine. ``build_chunk_mask_flow_estimator_trt`` exports the repo's own DiT with
a seventh input, ``attn_mask`` (``(2, 1, T, T)`` bool), so the host builds the
same mask upstream would and the engine honours it; ``supports_attn_mask`` on
the wrapper tells the caller which kind of engine it holds.

Precision: TensorRT >= 11 dropped the weakly-typed FP16/INT8 builder flags, so
fp16 only comes from a STRONGLY_TYPED network built from an fp16 ONNX
(``*autocast_fp16*``, fp16 I/O). An fp32 ONNX is built fp32 + the TF32 matmul
flag. ``EXPLICIT_BATCH`` is implicit (no flag) and ``ITensor.dtype`` is
read-only, so neither is set here.
"""

from __future__ import annotations

import contextlib
import os
import queue
import uuid

import torch
import torch.nn.functional as F
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
# The chunk-mask engine adds the query-key map, dynamic on both time dims.
ATTN_MASK_INPUT = "attn_mask"
_MASK_MIN_SHAPE, _MASK_OPT_SHAPE, _MASK_MAX_SHAPE = (2, 1, 4, 4), (2, 1, 500, 500), (2, 1, 3000, 3000)


def _is_fp16_onnx(onnx_path: str) -> bool:
    """Heuristic: the project's fp16 estimator ONNX is exported strongly-typed
    and named ``*autocast_fp16*`` / ``*fp16*`` (vs ``*fp32*``)."""
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def _write_plan_atomically(engine_bytes, plan_path: str) -> None:
    tmp = f"{plan_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    tmp_created = False
    try:
        with open(tmp, "xb") as f:
            tmp_created = True
            f.write(engine_bytes)
        os.replace(tmp, plan_path)
        tmp_created = False
    except BaseException:
        if tmp_created:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Failed to remove temporary TensorRT plan %s", tmp, exc_info=True)
        raise


def _convert_onnx_to_trt(onnx_path: str, plan_path: str, strongly_typed: bool, with_attn_mask: bool = False) -> None:
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
    if with_attn_mask:
        profile.set_shape(ATTN_MASK_INPUT, _MASK_MIN_SHAPE, _MASK_OPT_SHAPE, _MASK_MAX_SHAPE)
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build flow-estimator engine from {onnx_path}")
    _write_plan_atomically(engine_bytes, plan_path)
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
        # Whether the engine takes the chunk-causal query-key map as an input
        # (see ``build_chunk_mask_flow_estimator_trt``). A legacy engine runs
        # full attention whatever the caller's ``streaming`` flag says.
        self.supports_attn_mask = _engine_has_input(engine, ATTN_MASK_INPUT)
        self.input_names = _engine_input_names(engine)
        # The output buffer must match the engine's output dtype, which an
        # autocast-traced graph can leave different from its inputs.
        self.out_dtype = _engine_tensor_dtype(engine, "estimator_out", io_dtype)
        # Filled in by the model when it swaps the estimator, so the host can
        # build the mask with the DiT's block size.
        self.static_chunk_size = 0
        self._pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
        for _ in range(trt_concurrent):
            ctx = engine.create_execution_context()
            assert ctx is not None, "failed to create TRT execution context (out of memory?)"
            stream = torch.cuda.Stream(torch.device(device))
            self._pool.put([ctx, stream])

    def acquire_estimator(self):
        return self._pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self._pool.put([context, stream])


def _engine_input_names(engine) -> frozenset[str]:
    try:
        import tensorrt as trt

        return frozenset(
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        )
    except Exception:
        return frozenset()


def _engine_has_input(engine, name: str) -> bool:
    return name in _engine_input_names(engine)


def _engine_tensor_dtype(engine, name: str, fallback: torch.dtype) -> torch.dtype:
    try:
        import tensorrt as trt

        dtype = engine.get_tensor_dtype(name)
        return {trt.float16: torch.float16, trt.float32: torch.float32, trt.bfloat16: torch.bfloat16}.get(
            dtype, fallback
        )
    except Exception:
        return fallback


def _engine_io_dtype(engine, fallback: torch.dtype) -> torch.dtype:
    """The dtype of the engine's ``x`` input, which the caller casts to."""
    return _engine_tensor_dtype(engine, "x", fallback)


class _EstimatorWithMaskInput(torch.nn.Module):
    """Export view of the DiT: the attention map is an input, not derived."""

    def __init__(self, estimator: torch.nn.Module):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, mask, mu, t, spks, cond, attn_mask):
        out = self.estimator(x, mask, mu, t, spks, cond, attn_mask=attn_mask)
        # With an explicit map the DiT never reads ``mask``, and the exporter
        # would drop an unused input. Zeroing padded frames keeps the
        # six-input calling convention of the bundled ONNX (the caller drops
        # those frames anyway). The cast keeps the output in the input dtype
        # under autocast, so the engine's I/O is uniform.
        return (out * mask.to(out.dtype)).to(x.dtype)


@contextlib.contextmanager
def _fp32_attention_for_export():
    """Trace scaled-dot-product attention in fp32 under fp16 autocast.

    The softmax over a masked ``T x T`` score map loses precision in fp16 as
    ``T`` grows (the bundled engine keeps it in fp32 too: its error against
    the torch DiT is about 4x lower than an all-fp16 trace). Casting q/k/v up
    and the output back down is recorded into the graph, so the engine runs
    that one block in fp32 and everything else in fp16.
    """
    original = F.scaled_dot_product_attention

    def fp32_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        with torch.autocast(device_type=query.device.type, enabled=False):
            out = original(
                query.float(),
                key.float(),
                value.float(),
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )
        return out.to(query.dtype)

    F.scaled_dot_product_attention = fp32_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def export_chunk_mask_estimator_onnx(estimator: torch.nn.Module, onnx_path: str, *, fp16: bool = True) -> str:
    """Export the repo's DiT to ONNX with ``attn_mask`` as a seventh input.

    Traced under fp16 autocast when ``fp16`` (the layout of the project's
    ``*autocast_fp16*`` ONNX, which TensorRT >= 11 builds strongly typed),
    with attention kept in fp32; plain fp32 otherwise. The result is written
    next to the model's other estimator ONNX files so the plan cache keys off
    it like any other.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Exporting the chunk-mask flow estimator needs the 'onnx' package (pip install onnx)"
        ) from exc

    device = next(estimator.parameters()).device
    was_training = estimator.training
    estimator.eval()
    wrapper = _EstimatorWithMaskInput(estimator).to(device).eval()
    frames = 64
    x = torch.randn(2, 80, frames, device=device)
    mask = torch.ones(2, 1, frames, device=device)
    mu = torch.randn(2, 80, frames, device=device)
    t = torch.rand(2, device=device)
    spks = torch.randn(2, 80, device=device)
    cond = torch.randn(2, 80, frames, device=device)
    attn_mask = torch.ones(2, 1, frames, frames, dtype=torch.bool, device=device)
    dynamic_axes = {
        "x": {2: "seq_len"},
        "mask": {2: "seq_len"},
        "mu": {2: "seq_len"},
        "cond": {2: "seq_len"},
        ATTN_MASK_INPUT: {2: "seq_len", 3: "seq_len"},
        "estimator_out": {2: "seq_len"},
    }
    tmp = f"{onnx_path}.tmp.{os.getpid()}"
    logger.info("Exporting chunk-mask flow estimator ONNX to %s (fp16=%s) ...", onnx_path, fp16)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16),
        _fp32_attention_for_export() if fp16 else contextlib.nullcontext(),
    ):
        torch.onnx.export(
            wrapper,
            (x, mask, mu, t, spks, cond, attn_mask),
            tmp,
            input_names=["x", "mask", "mu", "t", "spks", "cond", ATTN_MASK_INPUT],
            output_names=["estimator_out"],
            dynamic_axes=dynamic_axes,
            opset_version=18,
            dynamo=False,
        )
    os.replace(tmp, onnx_path)
    if was_training:
        estimator.train()
    return onnx_path


def build_chunk_mask_flow_estimator_trt(
    estimator: torch.nn.Module,
    onnx_dir: str,
    device: str | torch.device,
    *,
    fp16: bool = True,
) -> TrtContextWrapper:
    """Build/load a flow-estimator engine that takes the chunk-causal mask.

    The ONNX is exported from ``estimator`` (the loaded torch DiT) into
    ``onnx_dir`` on first use and cached there; the plan is cached like the
    legacy engine's. ``estimator.static_chunk_size`` is copied onto the
    wrapper so ``forward_estimator`` can build the same mask upstream would.
    """
    import tensorrt as trt

    tag = "autocast_fp16" if fp16 else "fp32"
    onnx_path = os.path.join(onnx_dir, f"flow.decoder.estimator.chunk_mask.{tag}.onnx")
    if not os.path.exists(onnx_path) or os.path.getsize(onnx_path) == 0:
        os.makedirs(onnx_dir, exist_ok=True)
        export_chunk_mask_estimator_onnx(estimator, onnx_path, fp16=fp16)
    plan_path = _resolve_plan_path(onnx_path, prefix="flow_estimator_chunk_mask")
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(onnx_path, plan_path, strongly_typed=fp16, with_attn_mask=True)

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize chunk-mask flow-estimator TensorRT engine {plan_path}")
    logger.info("Loaded chunk-mask flow-estimator TensorRT engine (%s)", plan_path)
    wrapper = TrtContextWrapper(engine, device=device, io_dtype=_engine_io_dtype(engine, torch.float32))
    if not wrapper.supports_attn_mask:
        raise RuntimeError(f"chunk-mask engine {plan_path} has no '{ATTN_MASK_INPUT}' input")
    wrapper.static_chunk_size = int(getattr(estimator, "static_chunk_size", 0))
    return wrapper


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
    io_dtype = _engine_io_dtype(engine, torch.float16 if strongly_typed else torch.float32)
    return TrtContextWrapper(engine, device=device, io_dtype=io_dtype)
