# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from .cuda_graph_wrapper import CFMGraphWrapper, HiFTGraphWrapper

logger = init_logger(__name__)

_SILENCE_TOKEN = 4218
# CosyVoice2 RelPos PE is built for max_len=5000 and `forward_chunk` never
# calls ``extend_pe``. After the 2x upsample a single Code2Wav prefill longer
# than this many codec tokens overflows the PE slice (matrix_ac vs matrix_bd).
_DEFAULT_RELPOS_MAX_POS = 5000
_MAX_ENCODE_TOKEN_CAP = 1024


def relpos_encode_token_budget(
    *,
    max_pos: int,
    stride: int,
    cache_offset: int,
    lookahead: int,
    cap: int = _MAX_ENCODE_TOKEN_CAP,
) -> int:
    """How many codec tokens ``forward_chunk`` can take before RelPos PE wraps.

    ``position_encoding(size)`` needs ``size <= max_pos``. After upsample,
    ``size = cache_offset * stride + ~stride * token_frames`` (plus last-chunk
    lookahead pad).
    """
    stride = max(1, int(stride))
    lookahead = max(0, int(lookahead))
    room = int(max_pos) // stride - max(0, int(cache_offset)) - lookahead - 1
    return max(lookahead + 1, min(int(cap), room))


def plan_token2wav_encode_slices(
    num_frames: int,
    *,
    max_frames: int,
    min_nonfinal: int,
    last_chunk: bool,
) -> list[tuple[int, int]]:
    """Split a Code2Wav token span so every non-final piece has a lookahead window."""
    if num_frames <= 0:
        return []
    min_nonfinal = max(1, int(min_nonfinal))
    max_frames = max(int(max_frames), min_nonfinal)
    last_min = 1 if last_chunk else min_nonfinal
    if num_frames <= max_frames:
        return [(0, num_frames)]

    slices: list[tuple[int, int]] = []
    start = 0
    while start < num_frames:
        remaining = num_frames - start
        if remaining <= max_frames:
            slices.append((start, num_frames))
            break
        take = max_frames
        tail = remaining - take
        if tail <= max_frames and tail < last_min:
            take = remaining - last_min
        if take < min_nonfinal:
            take = remaining
        slices.append((start, start + take))
        start += take
    return slices


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


def _token2wav_sdpa_context(device: torch.device):
    if device.type != "npu":
        return nullcontext()

    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    return npu_token2wav_sdpa_context()


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


def _undecorate_dynamo(module: nn.Module, method: str) -> None:
    """Restore ``method`` on ``module`` if TorchDynamo wrapped it.

    ``cosyvoice2`` decorates ``UpsampleConformerEncoderV2.forward_chunk`` with
    ``torch.compile(backend="eager")``. That backend performs no Inductor
    optimisation, so the wrapper only adds tracing and guard construction, and
    duplex pays it again on every unseen chunk shape -- seconds inside a live
    response. Dropping the wrapper leaves the original implementation, which is
    what the eager backend was executing anyway.
    """
    bound = getattr(module, method, None)
    original = getattr(bound, "_torchdynamo_orig_callable", None) or getattr(bound, "__wrapped__", None)
    if original is None:
        return
    function = getattr(original, "__func__", original)
    module.__dict__[method] = function.__get__(module, type(module))
    logger.info("Bypassed TorchDynamo wrapper on %s.%s", type(module).__name__, method)


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(
        self,
        token2wav: Any,
        trt_stepper: Any | None = None,
        *,
        connector_config: Mapping[str, int] | None = None,
        hift_graph_config: Mapping[str, Any] | None = None,
        cfm_graph_config: Mapping[str, Any] | None = None,
        bfloat16_attention_cache: bool = False,
    ):
        super().__init__()
        self._token2wav = token2wav
        # Optional TrtDiTStepper (step_audio2_dit_trt): replaces only the
        # per-timestep DiT estimator call; encoder and HiFT stay on torch.
        self._trt_stepper = trt_stepper
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        encoder = getattr(self.flow, "encoder", None)
        if encoder is not None:
            _undecorate_dynamo(encoder, "forward_chunk")
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
                speech, source = self.hift.inference(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self._estimator_att_compute_dtype = torch.float16 if self.float16 else torch.float32
        self._estimator_att_cache_dtype = (
            torch.bfloat16 if bfloat16_attention_cache else self._estimator_att_compute_dtype
        )
        self.n_timesteps = int(token2wav.n_timesteps)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self.hift_graph_wrapper: HiFTGraphWrapper | None = None
        graph_config = dict(hift_graph_config or {})
        if bool(graph_config.get("enabled", False)):
            if hift_parameter is None:
                raise ValueError("MiniCPM-o HiFT Graph requires a parameterized HiFT module")
            if hift_parameter.device.type != "cuda":
                logger.info("HiFT CUDA Graph is disabled on device type %s", hift_parameter.device.type)
            else:
                if connector_config is None:
                    raise ValueError("MiniCPM-o HiFT CUDA Graph requires connector chunk configuration")
                if self.mel_cache_len <= 0 or self.source_cache_len % self.mel_cache_len != 0:
                    raise ValueError(
                        "MiniCPM-o HiFT CUDA Graph requires source_cache_len to be divisible by mel_cache_len"
                    )
                capture_batch_sizes = graph_config.get("capture_batch_sizes", [1])
                logger.info("Enabling HiFT CUDA Graph with batch sizes %s", capture_batch_sizes)
                self.hift_graph_wrapper = HiFTGraphWrapper(
                    token2wav=token2wav,
                    connector_config=dict(connector_config),
                    capture_batch_sizes=capture_batch_sizes,
                )
                with torch.inference_mode(), _autocast_disabled(hift_parameter.device):
                    self.hift_graph_wrapper.capture()
                logger.info("HiFT CUDA Graph captured successfully")
        self._cfm_graph_wrapper: CFMGraphWrapper | None = None
        cfm_graph_cfg = dict(cfm_graph_config or {})
        if bool(cfm_graph_cfg.get("enabled", False)):
            flow_parameter = next(self.flow.parameters(), None)
            if flow_parameter is not None and flow_parameter.device.type == "cuda":
                estimator = self.flow.decoder.estimator
                self._cfm_graph_wrapper = CFMGraphWrapper(
                    graph_fn=estimator.blocks_forward_chunk,
                    max_graphs=int(cfm_graph_cfg.get("max_graphs", 32)),
                )
                logger.info("CFM CUDA Graph enabled (max_graphs=%d)", int(cfm_graph_cfg.get("max_graphs", 32)))
            else:
                logger.info(
                    "CFM CUDA Graph is disabled on device type %s",
                    flow_parameter.device.type if flow_parameter is not None else "unknown",
                )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}

    def _hift_inference(
        self,
        mel: torch.Tensor,
        source_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hift_graph_wrapper is None:
            return self.hift.inference(mel, source_cache)
        return self.hift_graph_wrapper.replay(mel, source_cache)

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

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _relpos_max_pos(self) -> int:
        embed = getattr(self.flow.encoder, "embed", None)
        pe = getattr(embed, "pe", None)
        if isinstance(pe, torch.Tensor) and pe.numel() > 0:
            return max(1, int(pe.size(1) // 2))
        return _DEFAULT_RELPOS_MAX_POS

    def _upsample_stride(self) -> int:
        stride = getattr(getattr(self.flow.encoder, "up_layer", None), "stride", None)
        return max(1, int(stride)) if stride is not None else 2

    def _max_encode_token_frames(self, states: list[BatchedToken2WavState]) -> int:
        att = states[0].flow_cache.get("conformer_att_cache") if states else None
        offset1 = int(att.shape[3] // 2) if att is not None else 0
        lookahead = self._pre_lookahead_len() or 0
        return relpos_encode_token_budget(
            max_pos=self._relpos_max_pos(),
            stride=self._upsample_stride(),
            cache_offset=offset1,
            lookahead=lookahead,
        )

    def _ensure_relpos_pe(self, tokens: torch.Tensor, att_cache: torch.Tensor | None) -> None:
        """Grow CosyVoice RelPos PE before ``forward_chunk``, which never calls extend_pe."""
        embed = getattr(self.flow.encoder, "embed", None)
        extend_pe = getattr(embed, "extend_pe", None)
        if not callable(extend_pe):
            return
        offset1 = int(att_cache.shape[3] // 2) if att_cache is not None else 0
        lookahead = self._pre_lookahead_len() or 0
        needed = offset1 * self._upsample_stride() + self._upsample_stride() * (int(tokens.shape[1]) + lookahead + 1)
        extend_pe(tokens.new_zeros((1, max(needed, 1))))

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_relpos_pe(tokens, att_cache)
        embedded = self.flow.input_embedding(tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

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
        attn_mask: torch.Tensor | None = None,
        valid_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._trt_stepper is not None and valid_lengths is None:
            out, new_cnn, new_att = self._trt_stepper.step(
                x=x,
                mu=mu,
                t=time,
                spks=speakers,
                cond=cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )
            return out.to(mu.dtype), new_cnn, new_att
        time_embedding = estimator.t_embedder(time).unsqueeze(1)
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        if isinstance(old_att, torch.Tensor) and old_att.dtype != estimator_input.dtype:
            old_att = old_att.to(dtype=estimator_input.dtype)
        if self._cfm_graph_wrapper is not None and valid_lengths is None:
            graph_cnn = torch.zeros_like(cnn_out) if cnn_cache is None else cnn_cache
            graph_att = (
                estimator_input.new_zeros(att_out.shape[:3] + (0,) + att_out.shape[4:])
                if att_cache is None
                else old_att
            )
            return self._cfm_graph_wrapper.replay(
                estimator_input,
                time_embedding,
                graph_cnn,
                graph_att,
                cnn_out,
                att_out,
            )
        if valid_lengths is not None:
            if not hasattr(estimator, "in_proj"):
                raise RuntimeError('MiniCPMO45Code2WavBatchError {"reason":"ragged_kernel_unavailable"}')
            result = self._blocks_forward_chunk_ragged(
                estimator,
                estimator_input,
                time_embedding,
                attn_mask,
                old_cnn,
                old_att,
                cnn_out,
                att_out,
                valid_lengths,
            )
        else:
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

    @staticmethod
    def _gather_causal_cache(
        history: torch.Tensor,
        valid_lengths: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        indices = valid_lengths[:, None] + torch.arange(width, device=history.device)[None, :]
        return history.gather(
            2,
            indices[:, None, :].expand(-1, int(history.shape[1]), -1),
        )

    def _blocks_forward_chunk_ragged(
        self,
        estimator: nn.Module,
        estimator_input: torch.Tensor,
        time_embedding: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cnn_cache: Any,
        att_cache: Any,
        cnn_cache_buffer: torch.Tensor,
        att_cache_buffer: torch.Tensor,
        valid_lengths: list[int],
    ) -> torch.Tensor:
        """Run one padded DiT batch while capturing exact per-row CNN state."""
        lengths = torch.tensor(
            (*valid_lengths, *valid_lengths),
            device=estimator_input.device,
            dtype=torch.long,
        )
        x = estimator.in_proj(estimator_input.transpose(1, 2))
        for block_index, block in enumerate(estimator.blocks):
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                shift_conv,
                scale_conv,
                gate_conv,
            ) = block.adaLN_modulation(time_embedding).chunk(9, dim=-1)

            normalized = block.norm1(x) * (1 + scale_msa) + shift_msa
            x_att, new_att = block.attn.forward_chunk(
                normalized,
                att_cache[block_index],
                attn_mask,
            )
            x = x + gate_msa * x_att

            conv_input = block.norm3(x) * (1 + scale_conv) + shift_conv
            old_cnn = cnn_cache[block_index]
            if old_cnn is None:
                width = int(block.conv.block[1].causal_padding[0])
                old_cnn = conv_input.new_zeros(
                    (
                        int(conv_input.shape[0]),
                        int(block.conv.in_channels + block.conv.out_channels),
                        width,
                    )
                )
            old_cnn1, old_cnn2 = old_cnn.split((block.conv.in_channels, block.conv.out_channels), dim=1)

            conv1_input = block.conv.block[0](conv_input)
            conv1_output, _ = block.conv.block[1].forward_chunk(conv1_input, old_cnn1)
            new_cnn1 = self._gather_causal_cache(
                torch.cat((old_cnn1, conv1_input), dim=2),
                lengths,
                int(old_cnn1.shape[2]),
            )

            conv2_input = block.conv.block[2:6](conv1_output)
            conv2_output, _ = block.conv.block[6].forward_chunk(conv2_input, old_cnn2)
            new_cnn2 = self._gather_causal_cache(
                torch.cat((old_cnn2, conv2_input), dim=2),
                lengths,
                int(old_cnn2.shape[2]),
            )
            x = x + gate_conv * block.conv.block[7](conv2_output)
            x = x + gate_mlp * block.mlp(block.norm2(x) * (1 + scale_mlp) + shift_mlp)

            cnn_cache_buffer[block_index].copy_(torch.cat((new_cnn1, new_cnn2), dim=1))
            att_cache_buffer[block_index][:, :, : int(new_att.shape[2]), :].copy_(new_att)

        return estimator.final_layer(x, time_embedding).transpose(1, 2)

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        valid_lengths: list[int] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | list[torch.Tensor],
    ]:
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
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        timeline = torch.linspace(
            0,
            1,
            self.n_timesteps + 1,
            device=mu.device,
            dtype=mu.dtype,
        )
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        time = timeline[0].expand(batch_size)
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        attn_mask = None
        if valid_lengths is not None:
            if len(valid_lengths) != batch_size:
                raise ValueError(f"valid length count {len(valid_lengths)} != batch {batch_size}")
            cfg_lengths = torch.tensor(
                (*valid_lengths, *valid_lengths),
                device=mu.device,
            )
            positions = torch.arange(int(mu.shape[2]), device=mu.device)
            valid_queries = positions.unsqueeze(0) < cfg_lengths.unsqueeze(1)
            current_keys = valid_queries.unsqueeze(1).expand(-1, int(mu.shape[2]), -1)
            old_keys = torch.ones(
                (2 * batch_size, int(mu.shape[2]), offset),
                dtype=torch.bool,
                device=mu.device,
            )
            attn_mask = valid_queries.unsqueeze(2) & torch.cat((current_keys, old_keys), dim=2)
        next_cnn: list[torch.Tensor] = []
        next_att_cache: torch.Tensor | None = None
        ragged_att_cache: list[torch.Tensor] | None = None
        dt = timeline[1] - timeline[0]
        with _token2wav_sdpa_context(mu.device):
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
                    attn_mask=attn_mask,
                    valid_lengths=valid_lengths,
                )
                conditional, unconditional = estimate.split(batch_size, dim=0)
                velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
                x = x + dt * velocity
                time = time + dt
                if step + 1 < self.n_timesteps:
                    dt = timeline[step + 2] - time[0]
                next_cnn.append(step_cnn)
                if valid_lengths is not None:
                    if ragged_att_cache is None:
                        ragged_att_cache = [
                            torch.empty(
                                (
                                    self.n_timesteps,
                                    int(step_att.shape[0]),
                                    2,
                                    int(step_att.shape[2]),
                                    valid_length + offset,
                                    int(step_att.shape[4]),
                                ),
                                device=step_att.device,
                                dtype=self._estimator_att_cache_dtype,
                            )
                            for valid_length in valid_lengths
                        ]
                    current_width = int(mu.shape[2])
                    for row, (valid_length, row_cache) in enumerate(zip(valid_lengths, ragged_att_cache, strict=True)):
                        for cfg_row, source_row in enumerate((row, batch_size + row)):
                            row_cache[step, :, cfg_row, :, :valid_length].copy_(
                                step_att[:, source_row, :, :valid_length]
                            )
                            if offset:
                                row_cache[step, :, cfg_row, :, valid_length:].copy_(
                                    step_att[:, source_row, :, current_width:]
                                )
                    del step_att
                    continue
                if next_att_cache is None:
                    next_att_cache = torch.empty(
                        (self.n_timesteps, *step_att.shape),
                        device=step_att.device,
                        dtype=self._estimator_att_cache_dtype,
                    )
                next_att_cache[step].copy_(step_att)
                del step_att
        if ragged_att_cache is not None:
            return x, torch.stack(next_cnn), ragged_att_cache
        assert next_att_cache is not None
        return x, torch.stack(next_cnn), next_att_cache

    def _split_flow_cache(self, cache: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, torch.Tensor]]:
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            estimator_att = cache["estimator_att_cache"]
            if estimator_att.dtype == self._estimator_att_cache_dtype:
                request_att = torch.cat(
                    (
                        estimator_att[:, :, row : row + 1],
                        estimator_att[:, :, batch_size + row : batch_size + row + 1],
                    ),
                    dim=2,
                ).detach()
            else:
                request_shape = list(estimator_att.shape)
                request_shape[2] = 2
                request_att = torch.empty(
                    request_shape,
                    device=estimator_att.device,
                    dtype=self._estimator_att_cache_dtype,
                )
                request_att[:, :, 0:1].copy_(estimator_att[:, :, row : row + 1])
                request_att[:, :, 1:2].copy_(estimator_att[:, :, batch_size + row : batch_size + row + 1])
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
                    "estimator_att_cache": request_att,
                }
            )
        return result

    def _stack_flow_cache(self, states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        estimator_att = torch.cat((*conditional_att, *unconditional_att), dim=2)
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": estimator_att,
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
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
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
        num_frames = int(tokens.shape[1])
        if lookahead is not None and not last_chunk:
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        # A non-async Talker dump can land thousands of codec tokens in one
        # Code2Wav prefill. CosyVoice RelPos PE (max_len=5000) plus 2x upsample
        # cannot score that in one ``forward_chunk`` (6968 vs 985 on NPU).
        max_frames = self._max_encode_token_frames(states)
        slices = plan_token2wav_encode_slices(
            num_frames,
            max_frames=max_frames,
            min_nonfinal=(lookahead + 1) if lookahead is not None else 1,
            last_chunk=last_chunk,
        )
        if len(slices) > 1:
            logger.info(
                "MiniCPM-o Code2Wav splitting %d codec tokens into %d encoder windows "
                "(max_frames=%d) to stay inside RelPos PE.",
                num_frames,
                len(slices),
                max_frames,
            )
            parts: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
            current = states
            for index, (start, end) in enumerate(slices):
                is_last_piece = index == len(slices) - 1
                audios, current = self._decode_batch_once(
                    tokens[:, start:end],
                    features,
                    current,
                    last_chunk=last_chunk and is_last_piece,
                    flush_encoder=flush_encoder and is_last_piece,
                )
                for row, audio in enumerate(audios):
                    parts[row].append(audio)
            merged = [
                torch.cat(row_parts) if row_parts else tokens.new_zeros((0,), dtype=torch.float32)
                for row_parts in parts
            ]
            return merged, current
        return self._decode_batch_once(
            tokens,
            features,
            states,
            last_chunk=last_chunk,
            flush_encoder=flush_encoder,
        )

    def _decode_batch_once(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
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
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
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
        old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
        old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
        old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        speech, source = self._hift_inference(mel, old_source)
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

    @staticmethod
    def _require_complete_ragged_outputs(
        audios: list[torch.Tensor | None],
        next_states: list[BatchedToken2WavState | None],
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        missing_audio_rows = [row for row, audio in enumerate(audios) if audio is None]
        missing_state_rows = [row for row, state in enumerate(next_states) if state is None]
        if missing_audio_rows or missing_state_rows:
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"incomplete_ragged_output","audio_rows":{missing_audio_rows},'
                f'"state_rows":{missing_state_rows}}}'
            )
        return (
            cast(list[torch.Tensor], audios),
            cast(list[BatchedToken2WavState], next_states),
        )

    def decode_ragged_batch(
        self,
        tokens: list[torch.Tensor],
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunks: list[bool],
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = len(tokens)
        if batch_size != len(states) or batch_size != len(last_chunks):
            raise ValueError("ragged token, state, and final-flag batches must have the same size")
        if batch_size == 0:
            return [], []

        lookahead = self._pre_lookahead_len()
        for row, (row_tokens, last_chunk) in enumerate(zip(tokens, last_chunks, strict=True)):
            num_frames = int(row_tokens.numel())
            if lookahead is not None and not last_chunk and num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","row":{row},'
                    f'"frames":{num_frames},"minimum":{lookahead + 1}}}'
                )

        max_frames = self._max_encode_token_frames(states)
        if any(int(row_tokens.numel()) > max_frames for row_tokens in tokens):
            # decode_batch owns the RelPos-safe encoder slicing used by
            # non-async Talker dumps. Preserve that path for overlong rows;
            # ordinary short duplex chunks continue through one padded DiT.
            exact_groups: dict[tuple[int, bool], list[int]] = {}
            for row, (row_tokens, last_chunk) in enumerate(zip(tokens, last_chunks, strict=True)):
                exact_groups.setdefault((int(row_tokens.numel()), last_chunk), []).append(row)
            audios: list[torch.Tensor | None] = [None] * batch_size
            next_states: list[BatchedToken2WavState | None] = [None] * batch_size
            for (_, last_chunk), rows in exact_groups.items():
                group_audio, group_states = self.decode_batch(
                    torch.stack([tokens[row] for row in rows], dim=0),
                    features,
                    [states[row] for row in rows],
                    last_chunk=last_chunk,
                )
                for group_row, row in enumerate(rows):
                    audios[row] = group_audio[group_row]
                    next_states[row] = group_states[group_row]
            return self._require_complete_ragged_outputs(audios, next_states)
        encoder_groups: dict[tuple[int, bool], list[int]] = {}
        for row, (row_tokens, last_chunk) in enumerate(zip(tokens, last_chunks, strict=True)):
            encoder_groups.setdefault((int(row_tokens.numel()), last_chunk), []).append(row)

        hidden_rows: list[torch.Tensor | None] = [None] * batch_size
        conformer_cnn_rows: list[torch.Tensor | None] = [None] * batch_size
        conformer_att_rows: list[torch.Tensor | None] = [None] * batch_size
        for (_, last_chunk), rows in encoder_groups.items():
            group_states = [states[row] for row in rows]
            group_cache = self._stack_flow_cache(group_states)
            group_tokens = torch.stack([tokens[row] for row in rows], dim=0)
            with self._autocast(group_tokens.device):
                hidden, conformer_cnn, conformer_att = self._encode_chunk(
                    group_tokens,
                    last_chunk=last_chunk,
                    cnn_cache=group_cache["conformer_cnn_cache"],
                    att_cache=group_cache["conformer_att_cache"],
                )
            for group_row, row in enumerate(rows):
                hidden_rows[row] = hidden[group_row : group_row + 1]
                conformer_cnn_rows[row] = conformer_cnn[group_row : group_row + 1]
                conformer_att_rows[row] = conformer_att[:, group_row : group_row + 1]

        missing_hidden_rows = [row for row, value in enumerate(hidden_rows) if value is None]
        if missing_hidden_rows:
            raise RuntimeError(
                f'MiniCPMO45Code2WavBatchError {{"reason":"incomplete_encoder_output","rows":{missing_hidden_rows}}}'
            )
        resolved_hidden = cast(list[torch.Tensor], hidden_rows)
        hidden_lengths = [int(value.shape[1]) for value in resolved_hidden]
        max_hidden_length = max(hidden_lengths)
        padded_hidden = resolved_hidden[0].new_zeros((batch_size, max_hidden_length, int(resolved_hidden[0].shape[2])))
        for row, hidden in enumerate(resolved_hidden):
            padded_hidden[row, : int(hidden.shape[1])].copy_(hidden[0])

        flow_cache = self._stack_flow_cache(states)
        speakers = features.speaker_embedding.expand(batch_size, -1)
        with self._autocast(padded_hidden.device):
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            cond = torch.zeros_like(padded_hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                padded_hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
                valid_lengths=hidden_lengths,
            )

        prompt_len = int(features.mels.shape[1])
        assert isinstance(estimator_att, list)
        new_flow: list[dict[str, torch.Tensor]] = []
        for row, row_estimator_att in enumerate(estimator_att):
            conformer_cnn = conformer_cnn_rows[row]
            conformer_att = conformer_att_rows[row]
            assert conformer_cnn is not None and conformer_att is not None
            if row_estimator_att.shape[4] > prompt_len + 100:
                row_estimator_att = torch.cat(
                    (
                        row_estimator_att[..., :prompt_len, :],
                        row_estimator_att[..., -100:, :],
                    ),
                    dim=4,
                )
            if conformer_att.shape[3] > prompt_len + 100:
                conformer_att = torch.cat(
                    (
                        conformer_att[..., :prompt_len, :],
                        conformer_att[..., -100:, :],
                    ),
                    dim=3,
                )
            new_flow.append(
                {
                    "conformer_cnn_cache": conformer_cnn.detach().clone(),
                    "conformer_att_cache": conformer_att.detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            estimator_cnn[:, :, row : row + 1],
                            estimator_cnn[:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": row_estimator_att,
                }
            )

        audios: list[torch.Tensor | None] = [None] * batch_size
        next_states: list[BatchedToken2WavState | None] = [None] * batch_size
        vocoder_groups: dict[tuple[int, bool], list[int]] = {}
        for row, last_chunk in enumerate(last_chunks):
            vocoder_groups.setdefault((hidden_lengths[row], last_chunk), []).append(row)
        for (hidden_length, last_chunk), rows in vocoder_groups.items():
            old_mel = torch.cat([states[row].hift_cache["mel"] for row in rows], dim=0)
            old_source = torch.cat([states[row].hift_cache["source"] for row in rows], dim=0)
            old_speech = torch.cat([states[row].hift_cache["speech"] for row in rows], dim=0)
            group_mel = torch.cat(
                [chunk_mel[row : row + 1, :, :hidden_length] for row in rows],
                dim=0,
            )
            mel = torch.cat((old_mel, group_mel), dim=2)
            speech, source = self._hift_inference(mel, old_source)
            if old_speech.shape[-1] > 0:
                window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
                speech = self._fade_in_out(speech, old_speech, window)
            next_hift = {
                "mel": mel[..., -self.mel_cache_len :].detach(),
                "source": source[..., -self.source_cache_len :].detach(),
                "speech": speech[..., -self.source_cache_len :].detach(),
            }
            emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
            for group_row, row in enumerate(rows):
                audios[row] = emitted[group_row].reshape(-1).to(dtype=torch.float32)
                next_states[row] = BatchedToken2WavState(
                    flow_cache=new_flow[row],
                    hift_cache={
                        name: value[group_row : group_row + 1].detach().clone() for name, value in next_hift.items()
                    },
                )

        return self._require_complete_ragged_outputs(audios, next_states)
