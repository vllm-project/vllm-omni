# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inject MiniCPM-o Code2Wav NPUGraph acceleration on Ascend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from typing import cast
from weakref import WeakKeyDictionary

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)

_PATCHED = False
_original_build_backend = None
_original_estimator_step = None
_original_setup_batch = None
_original_decode_batch = None
_backend_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_ENABLE_KEY = "code2wav_enable_npu_graph"
_MAX_GRAPHS_KEY = "code2wav_max_npu_graphs"
_BATCH_BUCKETS_KEY = "cfm_graph_batch_buckets"
_BF16_ATTENTION_CACHE_KEY = "code2wav_bfloat16_attention_cache"


def _batch_bucket(batch: int, buckets: tuple[int, ...]) -> int:
    """Smallest configured bucket that fits ``batch`` (``batch`` if none does).

    The capture key covers the full tensor shape
    (``NPUExactGraphRunner._tensor_signature``), so every distinct batch size
    multiplies the captured shape space. Rounding the batch up to a small set
    of buckets collapses that product: with mel bucketing already applied, an
    8-way concurrent stage-2 run otherwise needs tens of exact shapes and both
    exhausts the graph pool and grows graph memory past the card budget.
    """
    for bucket in buckets:
        if bucket >= batch:
            return bucket
    return batch


def _pad_rows(value: torch.Tensor, target_batch: int) -> torch.Tensor:
    """Append all-zero rows along the batch dimension (dim 0)."""
    missing = target_batch - int(value.shape[0])
    if missing <= 0:
        return value
    return torch.cat((value, value.new_zeros((missing, *value.shape[1:]))), dim=0)


def _pad_cache_rows(value: torch.Tensor, target_batch: int) -> torch.Tensor:
    """Append all-zero request slots to a packed cache (batch is dim 1)."""
    missing = target_batch - int(value.shape[1])
    if missing <= 0:
        return value
    return torch.cat((value, value.new_zeros((value.shape[0], missing, *value.shape[2:]))), dim=1)


def _parse_batch_buckets(raw: object) -> tuple[int, ...]:
    """Read the ``cfm_graph_batch_buckets`` list; empty disables batch padding."""
    if raw in (None, "", [], ()):
        return ()
    if isinstance(raw, str):
        raw = [part for part in raw.replace(",", " ").split() if part]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{_BATCH_BUCKETS_KEY} must be a list of positive ints, got {raw!r}")
    buckets: list[int] = []
    for item in raw:
        value = int(item)
        if value <= 0 or value != float(item):
            raise ValueError(f"{_BATCH_BUCKETS_KEY} must be positive integers, got {item!r}")
        if value not in buckets:
            buckets.append(value)
    return tuple(sorted(buckets))


def _config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _graph_config(model: object) -> dict[str, object]:
    config = getattr(getattr(model, "vllm_config", None), "additional_config", None)
    return dict(config) if isinstance(config, Mapping) else {}


def prepare_code2wav_graph_runtime() -> None:
    """Select graph-capturable ACLNN kernels before Token2Wav is loaded."""
    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError(
            "MiniCPM-o Code2Wav NPUGraph capture is incompatible with "
            "ASCEND_LAUNCH_BLOCKING=1; unset it or set it to 0 before startup."
        )
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
    attn_mask=None,
    valid_frames=None,
):
    """Run the CFM estimator body after host-backed timestep embedding."""
    width = int(x.shape[-1])
    speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
    if valid_frames is not None and valid_frames < width:
        # Match the CUDA path: padded columns must not carry the speaker vector.
        speaker_features = speaker_features.clone()
        speaker_features[:, :, valid_frames:] = 0.0
    estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
    cnn_out, att_out = backend._estimator_buffers(estimator, estimator_input, att_cache)
    old_cnn = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
    old_att = att_cache if att_cache is not None else [None] * len(estimator.blocks)
    result = estimator.blocks_forward_chunk(
        estimator_input,
        time_embedding,
        attn_mask,
        old_cnn,
        old_att,
        cnn_out,
        att_out,
    )
    return result, cnn_out, att_out


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
    attn_mask=None,
    valid_lengths=None,
    valid_frames=None,
    time_embedding=None,
):
    assert _original_estimator_step is not None
    graph_runner = _backend_graph_runners.get(self)
    if (
        graph_runner is None
        or self._trt_stepper is not None
        or self._cfm_graph_wrapper is not None
        # The ragged (per-request length) path stays eager, but the padding
        # mask produced by bucketing has a shape fixed by the bucket, so it
        # replays as a graph input like any other tensor.
        or valid_lengths is not None
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
            attn_mask=attn_mask,
            valid_lengths=valid_lengths,
            valid_frames=valid_frames,
            time_embedding=time_embedding,
        )
    if (cnn_cache is None) != (att_cache is None):
        raise ValueError("estimator CNN and attention caches must both be present or absent")

    # valid_frames only matters for padded inputs; skip forwarding it for
    # unpadded calls so platform-owned graphable bodies that predate the
    # padding feature keep working with the old signature.
    graphable_kwargs = {"valid_frames": valid_frames} if valid_frames is not None else {}

    # The upstream embedder creates a frequency tensor on the host. Keep it
    # outside capture while retaining the tensor-only estimator body in graph.
    if time_embedding is None:
        time_embedding = estimator.t_embedder(time).unsqueeze(1)
    # A bucketing mask travels as a graph input: the capture key only covers
    # shapes, so replaying an existing graph must copy the current mask in.
    # Like `valid_frames`, it is forwarded only when present, so graphable
    # bodies written before this feature keep working with the old signature.
    has_mask = attn_mask is not None
    mask_inputs = (attn_mask,) if has_mask else ()

    def _step_kwargs(rest):
        kwargs = dict(graphable_kwargs)
        if has_mask:
            kwargs["attn_mask"] = rest[0]
        return kwargs

    # Batch-bucket alignment. Mel frames are already aligned by
    # cfm_graph_bucket_frames, so padding the batch up to a bucket leaves every
    # live slot's math untouched: slots are independent along the batch
    # dimension and padded slots (inputs and packed caches) are all-zero, their
    # outputs are dropped below. This only collapses the captured shape space,
    # which the exact-shape capture key would otherwise multiply across batch
    # sizes. A mask whose batch axis is not the leading one is not recognized
    # here, so alignment is skipped rather than guessed at.
    live_batch = int(x.shape[0])
    padded_batch = live_batch
    batch_buckets = getattr(self, "_cfm_graph_batch_buckets", ())
    if batch_buckets and padded_batch < _batch_bucket(live_batch, batch_buckets):
        mask_batch_ok = not has_mask or int(attn_mask.shape[0]) == live_batch
        if mask_batch_ok:
            padded_batch = _batch_bucket(live_batch, batch_buckets)
            x = _pad_rows(x, padded_batch)
            mu = _pad_rows(mu, padded_batch)
            speakers = _pad_rows(speakers, padded_batch)
            cond = _pad_rows(cond, padded_batch)
            time_embedding = _pad_rows(time_embedding, padded_batch)
            if has_mask:
                attn_mask = _pad_rows(attn_mask, padded_batch)
                mask_inputs = (attn_mask,)
            if cnn_cache is not None:
                cnn_cache = _pad_cache_rows(cnn_cache, padded_batch)
                att_cache = _pad_cache_rows(att_cache, padded_batch)

    def _unpad(outputs):
        if padded_batch == live_batch:
            return outputs
        result, new_cnn, new_att = outputs
        return (result[:live_batch], new_cnn[:, :live_batch], new_att[:, :live_batch])

    if cnn_cache is None:
        return _unpad(
            graph_runner.run(
                "cfm_estimator",
                (x, mu, time_embedding, speakers, cond, *mask_inputs),
                ((False, has_mask) if has_mask else (False,)),
                lambda step_x, step_mu, step_time, step_speakers, step_cond, *rest: _graphable_estimator_step(
                    self,
                    estimator,
                    x=step_x,
                    mu=step_mu,
                    time_embedding=step_time,
                    speakers=step_speakers,
                    cond=step_cond,
                    cnn_cache=None,
                    att_cache=None,
                    **_step_kwargs(rest),
                ),
            )
        )

    return _unpad(
        graph_runner.run(
            "cfm_estimator",
            (x, mu, time_embedding, speakers, cond, cnn_cache, att_cache, *mask_inputs),
            ((True, has_mask) if has_mask else (True,)),
            lambda step_x,
            step_mu,
            step_time,
            step_speakers,
            step_cond,
            step_cnn,
            step_att,
            *rest: _graphable_estimator_step(
                self,
                estimator,
                x=step_x,
                mu=step_mu,
                time_embedding=step_time,
                speakers=step_speakers,
                cond=step_cond,
                cnn_cache=step_cnn,
                att_cache=step_att,
                **_step_kwargs(rest),
            ),
        )
    )


def _patched_setup_batch(self, features, batch_size):
    assert _original_setup_batch is not None
    with _flow_execution_context(
        features.speech_tokens.device,
        require_math=self in _backend_graph_runners,
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
    with _flow_execution_context(
        tokens.device,
        require_math=self in _backend_graph_runners,
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

    extra = self._extra_config()
    if bool(extra.get(_BF16_ATTENTION_CACHE_KEY, False)):
        raise ValueError(
            "MiniCPM-o Code2Wav code2wav_bfloat16_attention_cache is "
            "currently supported only on CUDA; leave it unset or false on NPU."
        )

    config = _graph_config(self)
    max_graphs = max(0, int(cast(int | str, config.get(_MAX_GRAPHS_KEY, 32))))
    graph_enabled = max_graphs > 0 and _config_bool(config.get(_ENABLE_KEY), False)
    batch_buckets = _parse_batch_buckets(config.get(_BATCH_BUCKETS_KEY))
    if graph_enabled:
        # NPUOmniPlatform enables internal format for quantized LLM kernels.
        # Code2Wav uses regular convolution kernels that must remain in the
        # graph-capturable ACLNN path.
        prepare_code2wav_graph_runtime()
        if batch_buckets:
            logger.info(
                "MiniCPM-o Code2Wav NPUGraph batch buckets %s "
                "(batch sizes round up to the next bucket; unset keeps exact-batch capture)",
                list(batch_buckets),
            )

    assert _original_build_backend is not None
    _original_build_backend(self)
    # The estimator-step patch runs on the backend, so the bucket set lives
    # there. Unset (or graphs off) means every distinct batch size keeps its
    # own captured shape. Plain-object test doubles cannot carry attributes;
    # they keep exact-batch capture.
    try:
        self.backend._cfm_graph_batch_buckets = batch_buckets if graph_enabled else ()
    except AttributeError:
        pass

    graph_runner = None
    if graph_enabled:
        graph_runner = NPUExactGraphRunner(
            max_graphs=max_graphs,
            component_name="MiniCPM-o Code2Wav",
            disable_config_hint=(
                "set platforms.npu.stages[stage_id=2].additional_config.code2wav_enable_npu_graph=false"
            ),
        )
        if self.backend.speech_window.device.type == "npu" and not graph_runner.is_supported():
            raise RuntimeError(
                "MiniCPM-o Code2Wav NPUGraph capture requires torch.npu "
                "NPUGraph, graph, is_current_stream_capturing, and synchronize APIs."
            )
        if self.backend.flow.training:
            raise ValueError("MiniCPM-o Code2Wav NPUGraph capture requires flow.eval()")
        _backend_graph_runners[self.backend] = graph_runner

    if graph_enabled:
        logger.info(
            "MiniCPM-o Code2Wav NPUGraph replay enabled (max_graphs=%d)",
            max_graphs,
        )


def apply_minicpmo_4_5_code2wav_patch() -> None:
    """Patch the generic Code2Wav backend builder with Ascend acceleration."""
    global _PATCHED, _original_build_backend
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
    _original_estimator_step = BatchedToken2Wav._estimator_step
    _original_setup_batch = BatchedToken2Wav.setup_batch
    _original_decode_batch = BatchedToken2Wav.decode_batch

    MiniCPMO45Code2Wav._build_backend = _patched_build_backend  # type: ignore[method-assign]
    BatchedToken2Wav._estimator_step = _patched_estimator_step  # type: ignore[method-assign]
    BatchedToken2Wav.setup_batch = _patched_setup_batch  # type: ignore[method-assign]
    BatchedToken2Wav.decode_batch = _patched_decode_batch  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for MiniCPM-o 4.5 Code2Wav")
