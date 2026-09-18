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
_original_encode_chunk = None
_original_decode_cfm = None
_original_hift_inference = None
_original_setup_batch = None
_original_decode_batch = None
_backend_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_encoder_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_cfm_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_hift_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()
_HIFT_ENABLE_KEY = "code2wav_enable_hift_npu_graph"
_HIFT_MAX_GRAPHS_KEY = "code2wav_max_hift_npu_graphs"
_CFM_ENABLE_KEY = "code2wav_enable_full_cfm_npu_graph"
_CFM_MAX_GRAPHS_KEY = "code2wav_max_full_cfm_npu_graphs"
_ENCODER_ENABLE_KEY = "code2wav_enable_encoder_npu_graph"
_ENCODER_MAX_GRAPHS_KEY = "code2wav_max_encoder_npu_graphs"
_ENABLE_KEY = "code2wav_enable_npu_graph"
_MAX_GRAPHS_KEY = "code2wav_max_npu_graphs"
_BF16_ATTENTION_CACHE_KEY = "code2wav_bfloat16_attention_cache"


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
        None,
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
):
    assert _original_estimator_step is not None
    graph_runner = _backend_graph_runners.get(self)
    if (
        graph_runner is None
        or self._trt_stepper is not None
        or self._cfm_graph_wrapper is not None
        or attn_mask is not None
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
        )
    if (cnn_cache is None) != (att_cache is None):
        raise ValueError("estimator CNN and attention caches must both be present or absent")

    # valid_frames only matters for padded inputs; skip forwarding it for
    # unpadded calls so platform-owned graphable bodies that predate the
    # padding feature keep working with the old signature.
    graphable_kwargs = {"valid_frames": valid_frames} if valid_frames is not None else {}

    # The upstream embedder creates a frequency tensor on the host. Keep it
    # outside capture while retaining the tensor-only estimator body in graph.
    time_embedding = estimator.t_embedder(time).unsqueeze(1)
    if cnn_cache is None:
        return graph_runner.run(
            "cfm_estimator",
            (x, mu, time_embedding, speakers, cond),
            (False,),
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
                **graphable_kwargs,
            ),
        )

    return graph_runner.run(
        "cfm_estimator",
        (x, mu, time_embedding, speakers, cond, cnn_cache, att_cache),
        (True,),
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
            **graphable_kwargs,
        ),
    )


def _patched_decode_cfm(self, mu, speakers, cond, *, cnn_cache, att_cache, valid_lengths=None):
    assert _original_decode_cfm is not None
    runner = _cfm_graph_runners.get(self)
    if (
        runner is None
        or valid_lengths is not None
        or self._trt_stepper is not None
        or self._cfm_graph_wrapper is not None
    ):
        return _original_decode_cfm(
            self, mu, speakers, cond, cnn_cache=cnn_cache, att_cache=att_cache, valid_lengths=valid_lengths
        )
    if (cnn_cache is None) != (att_cache is None):
        return _original_decode_cfm(
            self, mu, speakers, cond, cnn_cache=cnn_cache, att_cache=att_cache, valid_lengths=valid_lengths
        )

    decoder = self.flow.decoder
    batch_size = int(mu.shape[0])
    offset = int(att_cache.shape[4]) if att_cache is not None else 0
    end = offset + int(mu.shape[2])
    if end > int(decoder.rand_noise.shape[2]):
        raise RuntimeError(
            "MiniCPMO45Code2WavBatchError "
            f'{{"reason":"noise_capacity","required":{end},'
            f'"available":{int(decoder.rand_noise.shape[2])}}}'
        )
    # Preserve the original time + dt recurrence, including its rounding.
    # The upstream timestep embedder creates host tensors, so keep it outside
    # capture. Pass noise as an input rather than binding a request's offset.
    timeline = torch.linspace(0, 1, self.n_timesteps + 1, device=mu.device, dtype=mu.dtype)
    timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
    time = timeline[0].expand(batch_size)
    dt = timeline[1] - timeline[0]
    embeddings, deltas = [], []
    for step in range(self.n_timesteps):
        embeddings.append(decoder.estimator.t_embedder(torch.cat((time, time))).unsqueeze(1))
        deltas.append(dt)
        time = time + dt
        if step + 1 < self.n_timesteps:
            dt = timeline[step + 2] - time[0]
    noise = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1)
    has_cache = cnn_cache is not None
    inputs = (mu, speakers, cond, noise, torch.stack(embeddings), torch.stack(deltas))
    if has_cache:
        inputs += (cnn_cache, att_cache)

    def compute(step_mu, step_speakers, step_cond, step_noise, step_embeddings, step_deltas, *caches):
        x = step_noise.clone()
        mu_cfg = torch.cat((step_mu, torch.zeros_like(step_mu)))
        speakers_cfg = torch.cat((step_speakers, torch.zeros_like(step_speakers)))
        cond_cfg = torch.cat((step_cond, torch.zeros_like(step_cond)))
        next_cnn, next_att = [], []
        for step in range(self.n_timesteps):
            estimate, new_cnn, new_att = _graphable_estimator_step(
                self,
                decoder.estimator,
                x=torch.cat((x, x)),
                mu=mu_cfg,
                time_embedding=step_embeddings[step],
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=caches[0][step] if has_cache else None,
                att_cache=caches[1][step] if has_cache else None,
            )
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + step_deltas[step] * velocity
            next_cnn.append(new_cnn)
            next_att.append(new_att.to(dtype=self._estimator_att_cache_dtype))
        return x, torch.stack(next_cnn), torch.stack(next_att)

    with _flow_execution_context(mu.device, require_math=True):
        return runner.run(
            "full_cfm",
            inputs,
            (has_cache, self.n_timesteps, decoder.inference_cfg_rate, self._estimator_att_cache_dtype),
            compute,
        )


@torch.inference_mode()
def _patched_hift_inference(self, mel, source_cache):
    assert _original_hift_inference is not None
    runner = _hift_graph_runners.get(self)
    if runner is None or self.hift_graph_wrapper is not None:
        return _original_hift_inference(self, mel, source_cache)

    hift = self.hift
    # Generate fresh excitation exactly once per chunk, outside warmup/capture.
    # Keep FFT and its complex-valued intermediates outside the NPU graph too.
    f0 = hift.f0_predictor(mel)
    source = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
    source, _, _ = hift.m_source(source)
    source = source.transpose(1, 2)
    if source_cache.shape[2] != 0:
        source[:, :, : source_cache.shape[2]] = source_cache
    real, imag = hift._stft(source.squeeze(1))
    source_stft = torch.cat((real, imag), dim=1)
    magnitude, phase = runner.run("hift_decoder", (mel, source_stft), (), hift._decode_from_source_stft)
    return hift._finalize_decode(magnitude, phase), source


def _patched_encode_chunk(self, tokens, *, last_chunk, cnn_cache, att_cache):
    assert _original_encode_chunk is not None
    runner = _encoder_graph_runners.get(self)
    if runner is None:
        return _original_encode_chunk(self, tokens, last_chunk=last_chunk, cnn_cache=cnn_cache, att_cache=att_cache)

    # Growing positional encoding may allocate host tensors; do it before
    # capture. A replacement PE buffer must not reuse a graph holding the old one.
    self._ensure_relpos_pe(tokens, att_cache)
    pe = getattr(getattr(self.flow.encoder, "embed", None), "pe", None)
    pe_key = (pe.data_ptr(), tuple(pe.shape), str(pe.dtype)) if isinstance(pe, torch.Tensor) else None
    has_cnn, has_att = cnn_cache is not None, att_cache is not None
    inputs = (tokens,) + ((cnn_cache,) if has_cnn else ()) + ((att_cache,) if has_att else ())

    def compute(step_tokens, *caches):
        step_cnn = caches[0] if has_cnn else None
        step_att = caches[int(has_cnn)] if has_att else None
        embedded = self.flow.input_embedding(step_tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded, last_chunk=last_chunk, cnn_cache=step_cnn, att_cache=step_att
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

    return runner.run("flow_encoder", inputs, (last_chunk, has_cnn, has_att, pe_key), compute)


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
    if graph_enabled:
        # NPUOmniPlatform enables internal format for quantized LLM kernels.
        # Code2Wav uses regular convolution kernels that must remain in the
        # graph-capturable ACLNN path.
        prepare_code2wav_graph_runtime()

    assert _original_build_backend is not None
    _original_build_backend(self)

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

    # Encoder capture is opt-in under the existing Code2Wav graph switch.
    # Keep its budget independent of the CFM estimator's shape cache.
    encoder_max_graphs = max(0, int(cast(int | str, config.get(_ENCODER_MAX_GRAPHS_KEY, 32))))
    if graph_enabled and encoder_max_graphs > 0 and _config_bool(config.get(_ENCODER_ENABLE_KEY), False):
        _encoder_graph_runners[self.backend] = NPUExactGraphRunner(
            max_graphs=encoder_max_graphs,
            component_name="MiniCPM-o Code2Wav encoder",
            disable_config_hint="set code2wav_enable_encoder_npu_graph=false in stage 2 additional_config",
        )
        logger.info("MiniCPM-o Code2Wav encoder NPUGraph enabled (max_graphs=%d)", encoder_max_graphs)

    cfm_max_graphs = max(0, int(cast(int | str, config.get(_CFM_MAX_GRAPHS_KEY, 32))))
    if graph_enabled and cfm_max_graphs > 0 and _config_bool(config.get(_CFM_ENABLE_KEY), False):
        _cfm_graph_runners[self.backend] = NPUExactGraphRunner(
            max_graphs=cfm_max_graphs,
            component_name="MiniCPM-o Code2Wav full CFM",
            disable_config_hint="set code2wav_enable_full_cfm_npu_graph=false in stage 2 additional_config",
        )
        logger.info("MiniCPM-o Code2Wav full CFM NPUGraph enabled (max_graphs=%d)", cfm_max_graphs)

    hift_max_graphs = max(0, int(cast(int | str, config.get(_HIFT_MAX_GRAPHS_KEY, 32))))
    if graph_enabled and hift_max_graphs > 0 and _config_bool(config.get(_HIFT_ENABLE_KEY), False):
        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hifigan import HiFTGenerator

        # Causal/custom generators can override source and decoder semantics.
        # Only route the concrete implementation whose inference we preserve.
        if type(self.backend.hift) is not HiFTGenerator:
            logger.warning("HiFT NPUGraph skipped for unsupported generator %s", type(self.backend.hift).__name__)
        elif self.backend.hift.training:
            raise ValueError("MiniCPM-o HiFT NPUGraph capture requires hift.eval()")
        else:
            _hift_graph_runners[self.backend] = NPUExactGraphRunner(
                max_graphs=hift_max_graphs,
                component_name="MiniCPM-o Code2Wav HiFT",
                disable_config_hint="set code2wav_enable_hift_npu_graph=false in stage 2 additional_config",
            )
            logger.info("MiniCPM-o Code2Wav HiFT NPUGraph enabled (max_graphs=%d)", hift_max_graphs)

    if graph_enabled:
        logger.info(
            "MiniCPM-o Code2Wav NPUGraph replay enabled (max_graphs=%d)",
            max_graphs,
        )


def apply_minicpmo_4_5_code2wav_patch() -> None:
    """Patch the generic Code2Wav backend builder with Ascend acceleration."""
    global _PATCHED, _original_build_backend, _original_encode_chunk, _original_decode_cfm
    global _original_decode_batch, _original_estimator_step, _original_setup_batch, _original_hift_inference
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        BatchedToken2Wav,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
        MiniCPMO45Code2Wav,
    )

    _original_build_backend = MiniCPMO45Code2Wav._build_backend
    _original_hift_inference = BatchedToken2Wav._hift_inference
    _original_decode_cfm = BatchedToken2Wav._decode_cfm
    _original_encode_chunk = BatchedToken2Wav._encode_chunk
    _original_estimator_step = BatchedToken2Wav._estimator_step
    _original_setup_batch = BatchedToken2Wav.setup_batch
    _original_decode_batch = BatchedToken2Wav.decode_batch

    MiniCPMO45Code2Wav._build_backend = _patched_build_backend  # type: ignore[method-assign]
    BatchedToken2Wav._hift_inference = _patched_hift_inference  # type: ignore[method-assign]
    BatchedToken2Wav._decode_cfm = _patched_decode_cfm  # type: ignore[method-assign]
    BatchedToken2Wav._encode_chunk = _patched_encode_chunk  # type: ignore[method-assign]
    BatchedToken2Wav._estimator_step = _patched_estimator_step  # type: ignore[method-assign]
    BatchedToken2Wav.setup_batch = _patched_setup_batch  # type: ignore[method-assign]
    BatchedToken2Wav.decode_batch = _patched_decode_batch  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for MiniCPM-o 4.5 Code2Wav")
