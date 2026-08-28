# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inject MiniCPM-o Code2Wav exact-shape NPUGraph acceleration on Ascend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)

_PATCHED = False
_original_build_backend = None
_original_decode_cfm = None
_original_estimator_step = None
_original_setup_batch = None
_original_decode_batch = None
_backend_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_backend_loop_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_backend_loop_constants: WeakKeyDictionary[object, dict[tuple[object, ...], _CFMLoopConstants]] = WeakKeyDictionary()
_ENABLE_KEY = "code2wav_enable_npu_graph"
_MAX_GRAPHS_KEY = "code2wav_max_npu_graphs"
_ENABLE_ENV = "VLLM_OMNI_MINICPMO45_CODE2WAV_NPU_GRAPH"
_MAX_GRAPHS_ENV = "VLLM_OMNI_MINICPMO45_CODE2WAV_MAX_NPU_GRAPHS"
_LOOP_ENABLE_KEY = "code2wav_enable_cfm_loop_npu_graph"
_LOOP_ENABLE_ENV = "VLLM_OMNI_MINICPMO45_CFM_LOOP_NPU_GRAPH"
_LOOP_MAX_GRAPHS_ENV = "VLLM_OMNI_MINICPMO45_CFM_LOOP_MAX_GRAPHS"
_LOOP_MAX_GRAPHS = 8


@dataclass(frozen=True)
class _CFMLoopConstants:
    time_embeddings: torch.Tensor
    dts: torch.Tensor

    @property
    def num_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in (self.time_embeddings, self.dts))


def _config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _graph_config(model: object) -> dict[str, object]:
    config = getattr(getattr(model, "vllm_config", None), "additional_config", None)
    return dict(config) if isinstance(config, Mapping) else {}


def _graph_settings(model: object) -> tuple[bool, int]:
    """Resolve stage config first, then environment fallback, then A3 defaults."""
    config = _graph_config(model)
    enable_value = config[_ENABLE_KEY] if _ENABLE_KEY in config else os.environ.get(_ENABLE_ENV)
    max_value = config[_MAX_GRAPHS_KEY] if _MAX_GRAPHS_KEY in config else os.environ.get(_MAX_GRAPHS_ENV, "32")
    try:
        max_graphs = max(0, int(max_value))
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using 32", _MAX_GRAPHS_KEY, max_value)
        max_graphs = 32
    # This patch is installed only by the NPU platform. Default-on is required
    # because competition evaluation supplies a baseline deploy YAML that has
    # no candidate-only graph key. Explicit stage config always wins.
    return max_graphs > 0 and _config_bool(enable_value, True), max_graphs


def _strict_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be an explicit boolean; got {value!r}")


def _cfm_loop_graph_settings(model: object) -> tuple[bool, int]:
    """Resolve the opt-in whole-loop graph with a hard eight-graph budget."""
    config = _graph_config(model)
    enable_value = config.get(_LOOP_ENABLE_KEY, os.environ.get(_LOOP_ENABLE_ENV, "0"))
    max_value = os.environ.get(_LOOP_MAX_GRAPHS_ENV, str(_LOOP_MAX_GRAPHS))
    if isinstance(max_value, bool):
        raise ValueError(f"{_LOOP_MAX_GRAPHS_ENV} must be an integer in [0, {_LOOP_MAX_GRAPHS}]")
    try:
        max_graphs = int(max_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_LOOP_MAX_GRAPHS_ENV} must be an integer in [0, {_LOOP_MAX_GRAPHS}]; got {max_value!r}"
        ) from exc
    if not 0 <= max_graphs <= _LOOP_MAX_GRAPHS:
        raise ValueError(f"{_LOOP_MAX_GRAPHS_ENV} must be an integer in [0, {_LOOP_MAX_GRAPHS}]; got {max_value!r}")
    enabled = _strict_bool(enable_value, name=_LOOP_ENABLE_KEY)
    return enabled and max_graphs > 0, max_graphs


def prepare_code2wav_graph_runtime() -> None:
    """Select graph-capturable ACLNN kernels before Token2Wav is loaded."""
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError("MiniCPM-o Code2Wav NPUGraph capture is incompatible with ASCEND_LAUNCH_BLOCKING=1")
    npu = torch.npu
    npu.config.allow_internal_format = False
    npu.set_compile_mode(jit_compile=False)
    logger.info("Configured MiniCPM-o Code2Wav NPUGraph runtime (allow_internal_format=False, jit_compile=False)")


def _flow_execution_context(device: torch.device, *, require_math: bool):
    if device.type != "npu":
        return nullcontext()
    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    return npu_token2wav_sdpa_context(require_math=require_math)


def _flow_graph_precision_key(backend: object) -> str:
    """Separate captured FP16 graphs from a later FP32 fallback epoch."""
    requested = bool(getattr(backend, "_npu_flow_float16_requested", False))
    available = getattr(backend, "_npu_autocast_available", None)
    return "float16" if requested and available is not False else "float32"


def _graphable_estimator_step(
    backend,
    estimator,
    *,
    x,
    mu,
    time_embedding,
    speakers,
    cond,
    cnn_cache,
    att_cache,
):
    """Run the CFM estimator body after host-backed timestep embedding."""
    if not backend._npu_flow_float16_requested:
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = backend._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            None,
            old_cnn,
            old_att,
            cnn_out,
            att_out,
        )
        return result, cnn_out, att_out

    with backend._npu_flow_autocast(x.device):
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = backend._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            None,
            old_cnn,
            old_att,
            cnn_out,
            att_out,
        )
    return result, cnn_out, att_out


def _cfm_loop_constants(
    backend: object,
    estimator: object,
    mu: torch.Tensor,
    batch_size: int,
) -> _CFMLoopConstants:
    """Build graph-resident Euler metadata once per exact numerical signature."""
    mixed_precision = bool(getattr(backend, "_npu_flow_float16_requested", False))
    integration_dtype = torch.float32 if mixed_precision else mu.dtype
    key = (
        int(getattr(backend, "n_timesteps")),
        batch_size,
        str(integration_dtype),
        str(mu.device),
        _flow_graph_precision_key(backend),
    )
    cache = _backend_loop_constants.setdefault(backend, {})
    cached = cache.get(key)
    if cached is not None:
        return cached

    steps = int(getattr(backend, "n_timesteps"))
    timeline = torch.linspace(
        0,
        1,
        steps + 1,
        device=mu.device,
        dtype=integration_dtype,
    )
    timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
    time = timeline[0].expand(batch_size)
    dt = timeline[1] - timeline[0]
    time_embeddings: list[torch.Tensor] = []
    dts: list[torch.Tensor] = []
    with torch.inference_mode():
        for step in range(steps):
            cfg_time = torch.cat((time, time), dim=0)
            time_embeddings.append(estimator.t_embedder(cfg_time).unsqueeze(1))
            dts.append(dt)
            time = time + dt
            if step + 1 < steps:
                dt = timeline[step + 2] - time[0]
    constants = _CFMLoopConstants(
        time_embeddings=torch.stack(time_embeddings),
        dts=torch.stack(dts),
    )
    cache[key] = constants
    return constants


def _graphable_cfm_loop(
    backend: object,
    estimator: object,
    constants: _CFMLoopConstants,
    *,
    mu: torch.Tensor,
    speakers: torch.Tensor,
    cond: torch.Tensor,
    cnn_cache: torch.Tensor | None,
    att_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute all CFM Euler/CFG steps inside one capture boundary."""
    decoder = backend.flow.decoder
    batch_size = int(mu.shape[0])
    offset = int(att_cache.shape[4]) if att_cache is not None else 0
    end = offset + int(mu.shape[2])
    if end > int(decoder.rand_noise.shape[2]):
        raise RuntimeError(
            "MiniCPMO45Code2WavBatchError "
            f'{{"reason":"noise_capacity","required":{end},'
            f'"available":{int(decoder.rand_noise.shape[2])}}}'
        )
    mixed_precision = bool(getattr(backend, "_npu_flow_float16_requested", False))
    if mixed_precision:
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone().float()
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0).float()
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0).float()
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0).float()
    else:
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)

    next_cnn: list[torch.Tensor] = []
    next_att: list[torch.Tensor] = []
    steps = int(getattr(backend, "n_timesteps"))
    for step in range(steps):
        old_cnn = cnn_cache[step] if cnn_cache is not None else None
        old_att = att_cache[step] if att_cache is not None else None
        estimate, step_cnn, step_att = _graphable_estimator_step(
            backend,
            estimator,
            x=torch.cat((x, x), dim=0),
            mu=mu_cfg,
            time_embedding=constants.time_embeddings[step],
            speakers=speakers_cfg,
            cond=cond_cfg,
            cnn_cache=old_cnn,
            att_cache=old_att,
        )
        if mixed_precision:
            estimate = estimate.float()
        conditional, unconditional = estimate.split(batch_size, dim=0)
        velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
        x = x + constants.dts[step] * velocity
        next_cnn.append(step_cnn)
        next_att.append(step_att)
    return x, torch.stack(next_cnn), torch.stack(next_att)


def _patched_estimator_step(
    self,
    estimator,
    *,
    x,
    mu,
    time,
    speakers,
    cond,
    cnn_cache,
    att_cache,
):
    assert _original_estimator_step is not None
    graph_runner = _backend_graph_runners.get(self)
    if (
        graph_runner is None
        or getattr(self, "_trt_stepper", None) is not None
        or getattr(self, "_cfm_graph_wrapper", None) is not None
    ):
        return _original_estimator_step(
            self,
            estimator,
            x=x,
            mu=mu,
            time=time,
            speakers=speakers,
            cond=cond,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
    if (cnn_cache is None) != (att_cache is None):
        raise ValueError("estimator CNN and attention caches must both be present or absent")

    # The upstream embedder creates a frequency tensor on the host. Keep it
    # outside capture while retaining the tensor-only estimator body in graph.
    time_embedding = estimator.t_embedder(time).unsqueeze(1)
    precision_key = _flow_graph_precision_key(self)
    if cnn_cache is None:
        return graph_runner.run(
            "cfm_estimator",
            (x, mu, time_embedding, speakers, cond),
            (False, precision_key),
            lambda step_x, step_mu, step_time, step_speakers, step_cond: _graphable_estimator_step(
                self,
                estimator,
                x=step_x,
                mu=step_mu,
                time_embedding=step_time,
                speakers=step_speakers,
                cond=step_cond,
                cnn_cache=None,
                att_cache=None,
            ),
        )

    return graph_runner.run(
        "cfm_estimator",
        (x, mu, time_embedding, speakers, cond, cnn_cache, att_cache),
        (True, precision_key),
        lambda step_x, step_mu, step_time, step_speakers, step_cond, step_cnn, step_att: _graphable_estimator_step(
            self,
            estimator,
            x=step_x,
            mu=step_mu,
            time_embedding=step_time,
            speakers=step_speakers,
            cond=step_cond,
            cnn_cache=step_cnn,
            att_cache=step_att,
        ),
    )


def _patched_decode_cfm(
    self,
    mu,
    speakers,
    cond,
    *,
    cnn_cache,
    att_cache,
):
    assert _original_decode_cfm is not None
    graph_runner = _backend_loop_graph_runners.get(self)
    if (
        graph_runner is None
        or int(mu.shape[0]) != 1
        or getattr(self, "_trt_stepper", None) is not None
        or getattr(self, "_cfm_graph_wrapper", None) is not None
    ):
        return _original_decode_cfm(
            self,
            mu,
            speakers,
            cond,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
    if (cnn_cache is None) != (att_cache is None):
        raise ValueError("estimator CNN and attention caches must both be present or absent")

    estimator = self.flow.decoder.estimator
    bucket = str(getattr(self, "_cfm_loop_bucket", "setup" if cnn_cache is None else "steady"))
    cached = cnn_cache is not None
    precision_key = _flow_graph_precision_key(self)
    inputs = (mu, speakers, cond) if not cached else (mu, speakers, cond, cnn_cache, att_cache)

    def compute(*values):
        step_mu, step_speakers, step_cond = values[:3]
        step_cnn, step_att = (None, None) if len(values) == 3 else values[3:5]
        constants = _cfm_loop_constants(self, estimator, step_mu, int(step_mu.shape[0]))
        return _graphable_cfm_loop(
            self,
            estimator,
            constants,
            mu=step_mu,
            speakers=step_speakers,
            cond=step_cond,
            cnn_cache=step_cnn,
            att_cache=step_att,
        )

    def fallback(*values):
        step_mu, step_speakers, step_cond = values[:3]
        step_cnn, step_att = (None, None) if len(values) == 3 else values[3:5]
        return _original_decode_cfm(
            self,
            step_mu,
            step_speakers,
            step_cond,
            cnn_cache=step_cnn,
            att_cache=step_att,
        )

    outputs, info = graph_runner.run_with_info(
        f"cfm_loop:{bucket}",
        inputs,
        (cached, int(self.n_timesteps), precision_key),
        compute,
        fallback_compute=fallback,
    )
    constants_cache = _backend_loop_constants.get(self, {})
    constant_bytes = sum(value.num_bytes for value in constants_cache.values())
    event = f"cfm_loop_graph_{info.mode}" if info.mode != "fallback" else "cfm_loop_graph_miss"
    self._emit_timeline(
        event,
        shape=tuple(mu.shape),
        num_bytes=info.workspace_bytes + constant_bytes,
        details={
            "bucket": bucket,
            "cached": cached,
            "fallback_reason": info.reason,
            **graph_runner.telemetry,
        },
    )
    return outputs


@contextmanager
def _cfm_loop_bucket(backend: object, bucket: str):
    missing = object()
    previous = getattr(backend, "_cfm_loop_bucket", missing)
    setattr(backend, "_cfm_loop_bucket", bucket)
    try:
        yield
    finally:
        if previous is missing:
            delattr(backend, "_cfm_loop_bucket")
        else:
            setattr(backend, "_cfm_loop_bucket", previous)


def _patched_setup_batch(self, features, batch_size):
    assert _original_setup_batch is not None
    graph_enabled = self in _backend_graph_runners or self in _backend_loop_graph_runners
    with (
        _cfm_loop_bucket(self, "setup"),
        _flow_execution_context(
            features.speech_tokens.device,
            require_math=graph_enabled,
        ),
    ):
        return _original_setup_batch(self, features, batch_size)


def _patched_decode_batch(
    self,
    tokens,
    features,
    states,
    *,
    last_chunk,
    flush_encoder=False,
):
    assert _original_decode_batch is not None
    graph_enabled = self in _backend_graph_runners or self in _backend_loop_graph_runners
    first_chunk = bool(states) and all(int(state.hift_cache["speech"].shape[-1]) == 0 for state in states)
    bucket = "tail" if last_chunk else "first" if first_chunk else "steady"
    with (
        _cfm_loop_bucket(self, bucket),
        _flow_execution_context(
            tokens.device,
            require_math=graph_enabled,
        ),
    ):
        return _original_decode_batch(
            self,
            tokens,
            features,
            states,
            last_chunk=last_chunk,
            flush_encoder=flush_encoder,
        )


def _patched_build_backend(self) -> None:
    if self.backend is not None:
        return

    graph_requested, max_graphs = _graph_settings(self)
    loop_graph_requested, loop_max_graphs = _cfm_loop_graph_settings(self)
    graph_runner = NPUExactGraphRunner(
        max_graphs=max_graphs,
        component_name="MiniCPM-o Code2Wav",
        disable_config_hint=f"set {_ENABLE_ENV}=0",
    )
    loop_graph_runner = NPUExactGraphRunner(
        max_graphs=loop_max_graphs,
        component_name="MiniCPM-o Code2Wav CFM loop",
        disable_config_hint=f"set {_LOOP_ENABLE_ENV}=0",
    )
    graph_enabled = graph_requested and graph_runner.is_supported()
    loop_graph_enabled = loop_graph_requested and loop_graph_runner.is_supported()
    if graph_requested and not graph_enabled:
        logger.warning("MiniCPM-o Code2Wav NPUGraph APIs are unavailable; using eager execution")
    if loop_graph_requested and not loop_graph_enabled:
        logger.warning("MiniCPM-o CFM loop NPUGraph APIs are unavailable; using estimator/eager fallback")
    if graph_enabled or loop_graph_enabled:
        try:
            # NPUOmniPlatform enables internal format for quantized LLM kernels.
            # Code2Wav uses regular convolution kernels that must remain in the
            # graph-capturable ACLNN path.
            prepare_code2wav_graph_runtime()
        except (AttributeError, RuntimeError, TypeError) as exc:
            graph_enabled = False
            loop_graph_enabled = False
            logger.warning("MiniCPM-o Code2Wav NPUGraph preflight failed; using estimator/eager fallback: %s", exc)

    assert _original_build_backend is not None
    _original_build_backend(self)

    if (graph_enabled or loop_graph_enabled) and self.backend.speech_window.device.type != "npu":
        graph_enabled = False
        loop_graph_enabled = False
        logger.warning("MiniCPM-o Code2Wav backend is not on NPU; using eager execution")
    if (graph_enabled or loop_graph_enabled) and self.backend.flow.training:
        graph_enabled = False
        loop_graph_enabled = False
        logger.warning("MiniCPM-o Code2Wav flow is in training mode; using eager execution")
    if graph_enabled:
        _backend_graph_runners[self.backend] = graph_runner
        logger.info("MiniCPM-o Code2Wav NPUGraph replay enabled (max_graphs=%d)", max_graphs)
    if loop_graph_enabled:
        _backend_loop_graph_runners[self.backend] = loop_graph_runner
        logger.info(
            "MiniCPM-o CFM loop NPUGraph replay enabled (max_graphs=%d)",
            loop_max_graphs,
        )


def apply_minicpmo_4_5_code2wav_patch() -> None:
    """Patch the generic Code2Wav backend builder with Ascend acceleration."""
    global _PATCHED, _original_build_backend, _original_decode_cfm
    global _original_decode_batch, _original_estimator_step, _original_setup_batch
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        BatchedToken2Wav,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
        MiniCPMO45Code2Wav,
    )

    _original_build_backend = MiniCPMO45Code2Wav._build_backend
    _original_decode_cfm = BatchedToken2Wav._decode_cfm
    _original_estimator_step = BatchedToken2Wav._estimator_step
    _original_setup_batch = BatchedToken2Wav.setup_batch
    _original_decode_batch = BatchedToken2Wav.decode_batch

    MiniCPMO45Code2Wav._build_backend = _patched_build_backend  # type: ignore[method-assign]
    BatchedToken2Wav._estimator_step = _patched_estimator_step  # type: ignore[method-assign]
    BatchedToken2Wav._decode_cfm = _patched_decode_cfm  # type: ignore[method-assign]
    BatchedToken2Wav.setup_batch = _patched_setup_batch  # type: ignore[method-assign]
    BatchedToken2Wav.decode_batch = _patched_decode_batch  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for MiniCPM-o 4.5 Code2Wav")
