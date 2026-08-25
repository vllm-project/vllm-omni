# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.benchmarks.ultra_timeline import emit_ultra_timeline_event

logger = init_logger(__name__)

_SILENCE_TOKEN = 4218


def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators (e.g. Ascend
    NPU) that never registered autocast support. Degrade to a no-op there: an
    enclosing region can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(self, token2wav: Any, *, npu_flow_float16: bool = False):
        super().__init__()
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        # The upstream streaming path preallocates fixed-size CFM and DiT
        # caches. This adapter never calls that path and supplies dynamically
        # sized request-owned buffers to ``blocks_forward_chunk`` instead.
        decoder = self.flow.decoder
        for module in (decoder, decoder.estimator):
            for buffer_name in ("att_cache_buffer", "cnn_cache_buffer"):
                if buffer_name in module._buffers:
                    setattr(module, buffer_name, None)
        hift_parameter = next(self.hift.parameters(), None)
        if hift_parameter is not None and hift_parameter.device.type == "cuda":
            # Prime the CUDA state used by HiFT during backend construction.
            # Otherwise, the first live audio chunk can fail when async stages
            # share one GPU.
            device = hift_parameter.device
            dtype = hift_parameter.dtype
            mel_channels = int(self.hift.conv_pre.in_channels)
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[device]),
                _autocast_disabled(device),
            ):
                # 50 mel frames match the default first streamed vocoder chunk.
                speech, source = self.hift(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self._npu_flow_float16_requested = bool(npu_flow_float16)
        self._npu_autocast_available: bool | None = None
        self._npu_autocast_fallback_count = 0
        self._npu_autocast_fallback_reason: str | None = None
        self._npu_autocast_fallback_error_type: str | None = None
        self.n_timesteps = int(token2wav.n_timesteps)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        self._timeline_request_ids: tuple[str, ...] = ()

    @contextmanager
    def timeline_context(self, request_ids: list[str]):
        """Attach host request ids to one synchronous backend invocation."""
        previous = self._timeline_request_ids
        self._timeline_request_ids = tuple(str(request_id) for request_id in request_ids)
        try:
            yield
        finally:
            self._timeline_request_ids = previous

    def _emit_timeline(
        self,
        event: str,
        *,
        shape: tuple[int, ...] | None = None,
        num_bytes: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if not self._timeline_request_ids:
            return
        event_details = dict(details or {})
        precision = self.precision_telemetry()
        event_details.update(
            cfm_steps=self.n_timesteps,
            flow_requested_dtype=precision["requested_dtype"],
            flow_effective_dtype=precision["effective_dtype"],
            flow_precision_fallback_count=precision["fallback_count"],
        )
        for request_id in self._timeline_request_ids:
            emit_ultra_timeline_event(
                event,
                request_id=request_id,
                stage=2,
                stream="compute",
                shape=shape,
                num_bytes=num_bytes,
                details=event_details,
            )

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            cached = PromptFeatures(
                speech_tokens=values[0],
                speaker_embedding=values[2],
                mels=values[3],
            )
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release request-owned prompt features after stream completion."""
        self._prompt_features.pop((prompt_cache_id, prompt_wav), None)

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.speaker_embedding.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    @staticmethod
    def _is_npu_device(device: torch.device) -> bool:
        return device.type == "npu"

    @staticmethod
    def _make_npu_autocast() -> AbstractContextManager[Any]:
        return torch.autocast("npu", dtype=torch.float16)

    def precision_telemetry(self) -> dict[str, int | str | None]:
        """Return Host-only NPU Flow precision state without device sync."""
        requested_dtype = "float16" if self._npu_flow_float16_requested else "float32"
        if not self._npu_flow_float16_requested or self._npu_autocast_available is False:
            effective_dtype = "float32"
        elif self._npu_autocast_available is True:
            effective_dtype = "float16"
        else:
            effective_dtype = "pending"
        return {
            "requested_dtype": requested_dtype,
            "effective_dtype": effective_dtype,
            "fallback_count": self._npu_autocast_fallback_count,
            "fallback_reason": self._npu_autocast_fallback_reason,
            "fallback_error_type": self._npu_autocast_fallback_error_type,
        }

    def _record_npu_autocast_fallback(self, error: BaseException) -> None:
        if self._npu_autocast_available is False:
            return
        self._npu_autocast_available = False
        self._npu_autocast_fallback_count += 1
        self._npu_autocast_fallback_reason = "npu_autocast_unavailable"
        self._npu_autocast_fallback_error_type = type(error).__name__
        logger.warning(
            "MiniCPM-o NPU Flow FP16 requested but torch.autocast('npu') is unavailable; "
            "falling back to FP32 (%s: %s)",
            type(error).__name__,
            error,
        )

    @contextmanager
    def _npu_flow_autocast(self, device: torch.device) -> Iterator[None]:
        """Enter NPU FP16 autocast; only context-entry failures fall back."""
        if not self._npu_flow_float16_requested or not self._is_npu_device(device):
            yield
            return
        if self._npu_autocast_available is False:
            with _autocast_disabled(device):
                yield
            return

        stack = ExitStack()
        try:
            stack.enter_context(self._make_npu_autocast())
        except (AttributeError, AssertionError, NotImplementedError, RuntimeError, TypeError, ValueError) as error:
            stack.close()
            if self._npu_autocast_available is True:
                raise RuntimeError(
                    "MiniCPM-o NPU Flow autocast became unavailable after FP16 execution; "
                    "restart the Stage 2 process to avoid mixed-precision cache reuse"
                ) from error
            self._record_npu_autocast_fallback(error)
            with _autocast_disabled(device):
                yield
            return

        self._npu_autocast_available = True
        with stack:
            yield

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._npu_flow_float16_requested or not self._is_npu_device(tokens.device):
            embedded = self.flow.input_embedding(tokens)
            hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
                xs=embedded,
                last_chunk=last_chunk,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )
            return self.flow.encoder_proj(hidden), new_cnn, new_att

        with self._npu_flow_autocast(tokens.device):
            embedded = self.flow.input_embedding(tokens)
            hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
                xs=embedded,
                last_chunk=last_chunk,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )
            projected = self.flow.encoder_proj(hidden)
        return projected, new_cnn, new_att

    @staticmethod
    def _estimator_buffers(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn = x.new_empty((depth, batch_size, cnn_channels, cnn_width))
        att = x.new_empty((depth, batch_size, heads, old_att_len + chunk_size, att_width))
        return cnn, att

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._npu_flow_float16_requested or not self._is_npu_device(x.device):
            time_embedding = estimator.t_embedder(time).unsqueeze(1)
            width = int(x.shape[-1])
            speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
            estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
            cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
            old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
            old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
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

        # Keep the host-backed timestep embedding in FP32, matching the NPU
        # Graph boundary. Only the DiT estimator body enters mixed precision.
        time_embedding = estimator.t_embedder(time).unsqueeze(1)
        with self._npu_flow_autocast(x.device):
            width = int(x.shape[-1])
            speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
            estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
            cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
            old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
            old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
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

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        npu_mixed_precision = self._npu_flow_float16_requested and self._is_npu_device(mu.device)
        if npu_mixed_precision:
            integration_dtype = torch.float32
            x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone().float()
            mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0).float()
            speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0).float()
            cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0).float()
        else:
            integration_dtype = mu.dtype
            x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
            mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
            speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
            cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        timeline = torch.linspace(
            0,
            1,
            self.n_timesteps + 1,
            device=mu.device,
            dtype=integration_dtype,
        )
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        time = timeline[0].expand(batch_size)
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        dt = timeline[1] - timeline[0]
        for step in range(self.n_timesteps):
            old_cnn = cnn_cache[step] if cnn_cache is not None else None
            old_att = att_cache[step] if att_cache is not None else None
            estimate, step_cnn, step_att = self._estimator_step(
                estimator,
                x=torch.cat((x, x), dim=0),
                mu=mu_cfg,
                time=torch.cat((time, time), dim=0),
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=old_cnn,
                att_cache=old_att,
            )
            if npu_mixed_precision:
                estimate = estimate.float()
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + dt * velocity
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time[0]
            next_cnn.append(step_cnn)
            next_att.append(step_att)
        return x, torch.stack(next_cnn), torch.stack(next_att)

    @staticmethod
    def _split_flow_cache(cache: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, torch.Tensor]]:
        if batch_size == 1:
            # Every tensor in ``cache`` was produced for this one request by
            # the current setup/decode call. Transfer tensor ownership to the
            # request state without singleton slices, cats, or clones. A new
            # dict keeps container ownership isolated from the local batch.
            return [{name: value.detach() for name, value in cache.items()}]

        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            cache["estimator_cnn_cache"][:, :, row : row + 1],
                            cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": torch.cat(
                        (
                            cache["estimator_att_cache"][:, :, row : row + 1],
                            cache["estimator_att_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                }
            )
        return result

    @staticmethod
    def _stack_flow_cache(states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        if len(states) == 1:
            # Flow/DiT treat incoming caches as read-only and return new cache
            # tensors. Preserve their storage for the scored batch=1 path
            # while isolating the temporary batch container.
            return dict(states[0].flow_cache)

        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": torch.cat((*conditional_att, *unconditional_att), dim=2),
        }

    def setup_batch(
        self,
        features: PromptFeatures,
        batch_size: int,
    ) -> list[BatchedToken2WavState]:
        prompt_tokens, speakers, prompt_mels = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            self._emit_timeline(
                "cfm_setup_begin",
                shape=tuple(hidden.shape),
                num_bytes=hidden.numel() * hidden.element_size(),
                details={"dtype": str(hidden.dtype)},
            )
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
            )
            self._emit_timeline(
                "cfm_setup_end",
                shape=tuple(estimator_att.shape),
                num_bytes=estimator_att.numel() * estimator_att.element_size(),
                details={"dtype": str(estimator_att.dtype)},
            )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        return [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
            )
            for row in split
        ]

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        if lookahead is not None and not last_chunk:
            num_frames = int(tokens.shape[1])
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        flow_cache = self._stack_flow_cache(states)
        speakers = features.speaker_embedding.expand(batch_size, -1)
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            self._emit_timeline(
                "cfm_begin",
                shape=tuple(hidden.shape),
                num_bytes=hidden.numel() * hidden.element_size(),
                details={"dtype": str(hidden.dtype)},
            )
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
            )
            self._emit_timeline(
                "cfm_end",
                shape=tuple(chunk_mel.shape),
                num_bytes=chunk_mel.numel() * chunk_mel.element_size(),
                details={"dtype": str(chunk_mel.dtype)},
            )

        prompt_len = int(features.mels.shape[1])
        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
        )
        if batch_size == 1:
            old_mel = states[0].hift_cache["mel"]
            old_source = states[0].hift_cache["source"]
            old_speech = states[0].hift_cache["speech"]
        else:
            old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
            old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
            old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        if self._npu_flow_float16_requested and self._is_npu_device(chunk_mel.device):
            chunk_mel = chunk_mel.float()
            old_mel = old_mel.float()
            old_source = old_source.float()
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        self._emit_timeline(
            "hift_begin",
            shape=tuple(mel.shape),
            num_bytes=mel.numel() * mel.element_size(),
            details={"dtype": str(mel.dtype)},
        )
        if self._npu_flow_float16_requested and self._is_npu_device(mel.device):
            with _autocast_disabled(mel.device):
                speech, source = self.hift(mel, old_source)
        else:
            speech, source = self.hift(mel, old_source)
        self._emit_timeline(
            "hift_end",
            shape=tuple(speech.shape),
            num_bytes=speech.numel() * speech.element_size(),
            details={"dtype": str(speech.dtype)},
        )
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
        next_states = [
            BatchedToken2WavState(
                flow_cache=new_flow[row],
                hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states
