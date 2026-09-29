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
The upstream CosyVoice3 fp16 ONNX fixes classifier-free-guidance (CFG) batch to
2, while cross-request flow batching produces CFG batch ``2N``. The default
single-request path therefore keeps the original fixed-batch engine unchanged.
When cross-request batching is enabled, the ONNX graph I/O batch metadata is
made symbolic before TensorRT parses it, and one engine is built with two
optimization profiles: profile 0 keeps CFG batch 2 fixed, while profile 1 covers
CFG batches 4 through the stage scheduler limit. One context switches profiles
on demand, avoiding a second resident engine/context while preserving the
single-request specialization.

Precision: TensorRT >= 11 dropped the weakly-typed FP16/INT8 builder flags, so
fp16 only comes from a STRONGLY_TYPED network built from an fp16 ONNX
(``*autocast_fp16*``, fp16 I/O). An fp32 ONNX is built fp32 + the TF32 matmul
flag. ``EXPLICIT_BATCH`` is implicit (no flag) and ``ITensor.dtype`` is
read-only, so neither is set here.
"""

from __future__ import annotations

import os
import queue
import uuid

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import (
    _resolve_plan_path,
    _trt_logger,
)

logger = init_logger(__name__)

# Optimization-profile shapes for the original fixed-CFG-batch engine.
_DYNAMIC_INPUTS = ("x", "mask", "mu", "cond")
_MIN_SHAPES = ((2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4))
_OPT_SHAPES = ((2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500))
_MAX_SHAPES = ((2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000))

# Bump when the dynamic-batch rewrite/profile contract changes so cached plans
# from older implementations cannot be loaded accidentally.
_DYNAMIC_BATCH_PLAN_VERSION = 4
_DYNAMIC_BATCH_IO_NAMES = ("x", "mask", "mu", "t", "spks", "cond", "estimator_out")
_MIN_T = 4
_OPT_T = 500
_SINGLE_MAX_T = 3000
_BATCH_MAX_T = 1024


def _is_fp16_onnx(onnx_path: str) -> bool:
    """Heuristic: the project's fp16 estimator ONNX is exported strongly-typed
    and named ``*autocast_fp16*`` / ``*fp16*`` (vs ``*fp32*``)."""
    name = os.path.basename(onnx_path).lower()
    return "fp16" in name or "autocast" in name


def _set_onnx_cfg_batch_dynamic(model):
    """Make only the estimator graph I/O batch dimensions symbolic.

    CosyVoice3's current estimator export fixes CFG batch to 2 even though the
    graph operations are batch-generic.  TensorRT must see the symbolic batch
    before parsing; changing ``INetworkDefinition`` input shapes afterwards is
    too late for the time-embedding path, whose inferred shape is already fixed.
    """
    expected = set(_DYNAMIC_BATCH_IO_NAMES)
    found: set[str] = set()
    for value in (*model.graph.input, *model.graph.output):
        if value.name not in expected:
            continue
        dims = value.type.tensor_type.shape.dim
        if len(dims) == 0:
            raise ValueError(f"ONNX tensor {value.name!r} has no batch dimension")
        dims[0].ClearField("dim_value")
        dims[0].dim_param = "cfg_batch"
        found.add(value.name)
    missing = expected - found
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"CosyVoice3 estimator ONNX is missing expected graph I/O: {names}")
    return model


def _dynamic_batch_onnx_bytes(onnx_path: str) -> bytes:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - packaging error on CUDA installs
        raise RuntimeError(
            "Dynamic CosyVoice3 TensorRT batching requires the 'onnx' package; "
            "install the CUDA requirements before enabling COSYVOICE3_BATCH_FLOW"
        ) from exc

    model = onnx.load(onnx_path)
    _set_onnx_cfg_batch_dynamic(model)
    return model.SerializeToString()


def _cfg_batch_profile(min_batch: int, opt_batch: int, max_batch: int, *, max_t: int):
    if not (2 <= min_batch <= opt_batch <= max_batch):
        raise ValueError(f"invalid CFG batch profile: min={min_batch}, opt={opt_batch}, max={max_batch}")
    if max_t < _OPT_T:
        raise ValueError(f"CFG batch profile max_t must be at least {_OPT_T}, got {max_t}")
    return {
        "x": ((min_batch, 80, _MIN_T), (opt_batch, 80, _OPT_T), (max_batch, 80, max_t)),
        "mask": ((min_batch, 1, _MIN_T), (opt_batch, 1, _OPT_T), (max_batch, 1, max_t)),
        "mu": ((min_batch, 80, _MIN_T), (opt_batch, 80, _OPT_T), (max_batch, 80, max_t)),
        "t": ((min_batch,), (opt_batch,), (max_batch,)),
        "spks": ((min_batch, 80), (opt_batch, 80), (max_batch, 80)),
        "cond": ((min_batch, 80, _MIN_T), (opt_batch, 80, _OPT_T), (max_batch, 80, max_t)),
    }


def _fixed_cfg_batch_profile():
    return _cfg_batch_profile(2, 2, 2, max_t=_SINGLE_MAX_T)


def _dynamic_batch_profile(max_cfg_batch: int):
    if max_cfg_batch < 4:
        raise ValueError(f"dynamic CFG batch must be at least 4, got {max_cfg_batch}")
    return _cfg_batch_profile(
        4,
        min(8, max_cfg_batch),
        max_cfg_batch,
        max_t=_BATCH_MAX_T,
    )


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


def _convert_onnx_to_trt(
    onnx_path: str,
    plan_path: str,
    strongly_typed: bool,
    *,
    max_cfg_batch: int | None = None,
) -> None:
    import tensorrt as trt

    dynamic_batch = max_cfg_batch is not None
    logger.info(
        "Building flow-estimator TensorRT engine from %s (%s%s) ...",
        onnx_path,
        "strongly-typed/fp16" if strongly_typed else "fp32+TF32",
        f", dynamic CFG batch <= {max_cfg_batch}" if dynamic_batch else "",
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
    if dynamic_batch:
        model_bytes = _dynamic_batch_onnx_bytes(onnx_path)
    else:
        with open(onnx_path, "rb") as f:
            model_bytes = f.read()
    if not parser.parse(model_bytes):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise ValueError(f"Failed to parse {onnx_path}: {errs}")

    config = builder.create_builder_config()
    workspace = 1 << 32
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    if dynamic_batch:
        runtime_resize = getattr(
            trt.PreviewFeature,
            "RUNTIME_ACTIVATION_RESIZE_10_10",
            None,
        )
        if runtime_resize is not None:
            config.set_preview_feature(runtime_resize, True)
    if not strongly_typed:
        # fp32 ONNX: enable the best available reduced-precision matmul flag
        # (FP16 on older TRT, else TF32 on Ampere+/Hopper). For a strongly-typed
        # network the precision is fixed by the graph, so no flag is set.
        for _flag_name in ("FP16", "TF32"):
            _flag = getattr(trt.BuilderFlag, _flag_name, None)
            if _flag is not None:
                config.set_flag(_flag)
                break

    if dynamic_batch:
        assert max_cfg_batch is not None
        # Profile 0 preserves the single-request specialization while profile 1
        # covers cross-request CFG batches. Keeping both in one engine avoids a
        # second resident copy of the weights and a second execution context.
        for shape_spec in (_fixed_cfg_batch_profile(), _dynamic_batch_profile(max_cfg_batch)):
            profile = builder.create_optimization_profile()
            for name, (mn, op, mx) in shape_spec.items():
                profile.set_shape(name, mn, op, mx)
            if not profile:
                raise RuntimeError("TensorRT produced an invalid flow-estimator optimization profile")
            config.add_optimization_profile(profile)
    else:
        profile = builder.create_optimization_profile()
        for name, mn, op, mx in zip(_DYNAMIC_INPUTS, _MIN_SHAPES, _OPT_SHAPES, _MAX_SHAPES):
            profile.set_shape(name, mn, op, mx)
        if not profile:
            raise RuntimeError("TensorRT produced an invalid flow-estimator optimization profile")
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build flow-estimator engine from {onnx_path}")
    _write_plan_atomically(engine_bytes, plan_path)
    logger.info("Wrote flow-estimator TensorRT engine to %s", plan_path)


class TrtContextWrapper:
    """Pool of TensorRT execution contexts for the flow estimator.

    A normal engine has one fixed-CFG-batch profile. A cross-request engine has
    profile 0 for CFG batch 2 and profile 1 for CFG batches 4..2N. The same
    context switches profiles on its CUDA stream before input shapes are rebound.
    """

    def __init__(
        self,
        engine,
        device: str | torch.device,
        io_dtype: torch.dtype = torch.float32,
        trt_concurrent: int = 1,
    ):
        self.trt_engine = engine
        self.io_dtype = io_dtype
        self._device = torch.device(device)
        self._dynamic_profiles = int(getattr(engine, "num_optimization_profiles", 1)) > 1
        if self._dynamic_profiles and trt_concurrent != 1:
            raise ValueError("dynamic TensorRT profile switching requires trt_concurrent=1")
        self._pool: queue.Queue = queue.Queue(maxsize=trt_concurrent)
        self._active_profile_by_context: dict[int, int] = {}
        for _ in range(trt_concurrent):
            ctx = engine.create_execution_context()
            assert ctx is not None, "failed to create TRT execution context (out of memory?)"
            stream = torch.cuda.Stream(self._device)
            self._active_profile_by_context[id(ctx)] = 0
            self._pool.put([ctx, stream])

    def _profile_for_shape(self, batch_size: int, sequence_length: int | None = None) -> int | None:
        if batch_size == 2:
            profile_index = 0
        elif self._dynamic_profiles and batch_size >= 4 and batch_size % 2 == 0:
            profile_index = 1
        else:
            return None

        try:
            min_shape, _, max_shape = self.trt_engine.get_tensor_profile_shape("x", profile_index)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0 if batch_size == 2 else None

        if not (int(min_shape[0]) <= batch_size <= int(max_shape[0])):
            return None
        if sequence_length is not None and not (int(min_shape[2]) <= sequence_length <= int(max_shape[2])):
            return None
        return profile_index

    def supports_estimator_shape(self, batch_size: int, sequence_length: int) -> bool:
        """Return whether an optimization profile accepts this CFG batch/length."""
        return self._profile_for_shape(batch_size, sequence_length) is not None

    def acquire_estimator(self, batch_size: int = 2, sequence_length: int | None = None):
        profile_index = self._profile_for_shape(batch_size, sequence_length)
        if profile_index is None:
            suffix = "" if sequence_length is None else f" at sequence length {sequence_length}"
            raise RuntimeError(f"TensorRT flow estimator does not support CFG batch {batch_size}{suffix}")

        [context, stream] = self._pool.get()
        context_id = id(context)
        prepared = False
        try:
            if self._active_profile_by_context[context_id] != profile_index:
                # Queue the profile switch on the same stream used by the
                # subsequent execute_async_v3(). CUDA stream ordering provides
                # the synchronization TensorRT requires between the profile
                # switch and enqueue without a host-side stream.synchronize().
                switched = context.set_optimization_profile_async(
                    profile_index,
                    stream.cuda_stream,
                )
                if not switched:
                    raise RuntimeError(
                        f"failed to select TensorRT optimization profile {profile_index} for CFG batch {batch_size}"
                    )
                self._active_profile_by_context[context_id] = profile_index
            prepared = True
            return [context, stream], self.trt_engine
        finally:
            if not prepared:
                self._pool.put([context, stream])

    def release_estimator(self, context, stream):
        if id(context) not in self._active_profile_by_context:
            raise RuntimeError("attempted to release an unknown TensorRT execution context")
        self._pool.put([context, stream])


def build_flow_estimator_trt(
    onnx_path: str,
    device: str | torch.device,
    *,
    max_cfg_batch: int | None = None,
) -> TrtContextWrapper:
    """Build/load the flow-estimator TRT engine and return a context-pool wrapper.

    With cross-request batching disabled, this is the original fixed-CFG-batch
    engine. With max_cfg_batch > 2, one dynamic engine contains profile 0
    for CFG batch 2 and profile 1 for CFG batches 4 through max_cfg_batch.
    """
    import tensorrt as trt

    strongly_typed = _is_fp16_onnx(onnx_path)
    dynamic_batch = max_cfg_batch is not None and max_cfg_batch > 2
    if max_cfg_batch is not None and max_cfg_batch % 2 != 0:
        raise ValueError(f"max_cfg_batch must be even, got {max_cfg_batch}")

    if dynamic_batch:
        prefix = f"flow_estimator_dynamic_v{_DYNAMIC_BATCH_PLAN_VERSION}_b{max_cfg_batch}_t{_BATCH_MAX_T}"
    else:
        prefix = "flow_estimator"
    plan_path = _resolve_plan_path(onnx_path, prefix=prefix)

    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        _convert_onnx_to_trt(
            onnx_path,
            plan_path,
            strongly_typed=strongly_typed,
            max_cfg_batch=max_cfg_batch if dynamic_batch else None,
        )

    runtime = trt.Runtime(_trt_logger())
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize flow-estimator TensorRT engine {plan_path}")

    if dynamic_batch:
        logger.info(
            "Loaded flow-estimator TensorRT engine (CFG batch 2/T<=%d and CFG batch 4..%d/T<=%d, %s)",
            _SINGLE_MAX_T,
            max_cfg_batch,
            _BATCH_MAX_T,
            plan_path,
        )
    else:
        logger.info("Loaded flow-estimator TensorRT engine (%s)", plan_path)

    io_dtype = torch.float16 if strongly_typed else torch.float32
    return TrtContextWrapper(
        engine,
        device=device,
        io_dtype=io_dtype,
    )
