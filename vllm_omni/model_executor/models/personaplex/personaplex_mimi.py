# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming Mimi codec for PersonaPlex (moshi-free).

Frame-clocked duplex needs a codec that encodes/decodes exactly one 80 ms frame
per call with state carried across the conversation. The ``moshi`` package
provided that; this module removes the dependency by combining:

- **transformers ``MimiModel``** (``kyutai/mimi`` — the same checkpoint family
  PersonaPlex ships) for the quantizer and every SEANet conv weight. Its
  streaming support only covers the encoder convs, so conv streaming is done
  here instead, uniformly.
- **Our own streaming wrappers** mirroring the Moshi reference semantics (MIT),
  verified against recorded reference outputs:
  * ``Conv1d``: left-context carry of ``effective_kernel - stride`` samples,
    zero-initialized at stream start (``pad_mode="constant"``).
  * ``ConvTranspose1d``: overlap-add tail carry of ``kernel - stride`` output
    samples, with the double-counted bias subtracted on merge.
  * transformer: the encoder/decoder transformers use a 250-position sliding
    context over a ring KV with absolute-offset RoPE. Hugging Face's cache path
    diverges once the window engages (position 250), so the transformers are
    reimplemented here on the same ring-KV design as
    ``personaplex_temporal.py`` and loaded directly from the PersonaPlex
    checkpoint's fused layout (LayerNorm + per-layer LayerScale + GELU FFN).

All per-stream state is ``[B, ...]`` with per-row reset (``reset_slot``), so the
codec composes with elastic slot recycling in batched duplex serving.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    _apply_rope,
    _RingKV,
)

logger = init_logger(__name__)

DEFAULT_HF_REPO = "kyutai/mimi"
FRAME_SIZE = 1920
CODEBOOKS = 8
_GRAPH_WARMUP_ITERS = 2
_GRAPH_STREAMS: dict[int, torch.cuda.Stream] = {}


def graph_stream() -> torch.cuda.Stream:
    """The one side stream every PersonaPlex CUDA graph of the current device warms up and is captured on.

    The caching allocator reuses a freed block only for the stream that allocated it, and cuBLAS keeps a
    workspace (32 MiB on sm90) for every stream that ran a matmul, for the life of the process. A new
    stream per capture left each warmup's temporaries in segments that no later allocation could reuse,
    and a workspace allocated among them kept ``empty_cache`` from releasing them. On one stream the
    warmups reuse the same segments, and its workspace is created here, before any temporary, in a
    segment of its own; the captured graphs replay with it.
    """
    device = torch.accelerator.current_device_index()
    stream = _GRAPH_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda.current_blas_handle()  # allocates the stream's cuBLAS workspace
        _GRAPH_STREAMS[device] = stream
    return stream


def _normalize_active(active: torch.Tensor | None, all_active: torch.Tensor) -> torch.Tensor:
    # None treats all rows as active. The shared Stage 0 and Code2Wav codecs
    # pass bool[B]: True advances that row's streaming state; False keeps an
    # absent or padded row's offsets and convolution carries unchanged.
    if active is None:
        return all_active
    if active.shape != all_active.shape:
        raise ValueError(f"active must have shape {tuple(all_active.shape)}, got {tuple(active.shape)}")
    return active.to(device=all_active.device, dtype=torch.bool)


def _map_moshi_codec_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map bundled Moshi codec keys to ``transformers.MimiModel`` keys."""

    mapped: dict[str, torch.Tensor] = {}
    transformer_prefixes = ("encoder_transformer.", "decoder_transformer.")
    for name, tensor in state_dict.items():
        if name.startswith(transformer_prefixes):
            continue

        if name.startswith("encoder.model."):
            target = name.replace("encoder.model.", "encoder.layers.", 1)
            target = target.replace(".conv.conv.", ".conv.")
        elif name.startswith("decoder.model."):
            target = name.replace("decoder.model.", "decoder.layers.", 1)
            target = target.replace(".convtr.convtr.", ".conv.")
            target = target.replace(".conv.conv.", ".conv.")
        elif name.startswith("downsample.conv.conv.conv."):
            target = name.replace("downsample.conv.conv.conv.", "downsample.conv.", 1)
        elif name.startswith("upsample.convtr.convtr.convtr."):
            target = name.replace("upsample.convtr.convtr.convtr.", "upsample.conv.", 1)
        elif name.startswith("quantizer.rvq_first."):
            target = name.replace(
                "quantizer.rvq_first.",
                "quantizer.semantic_residual_vector_quantizer.",
                1,
            )
        elif name.startswith("quantizer.rvq_rest."):
            target = name.replace(
                "quantizer.rvq_rest.",
                "quantizer.acoustic_residual_vector_quantizer.",
                1,
            )
        else:
            raise KeyError(f"unrecognized PersonaPlex Mimi checkpoint key: {name}")

        target = target.replace(".vq.layers.", ".layers.")
        target = target.replace("._codebook._initialized", ".codebook.initialized")
        target = target.replace("._codebook.cluster_usage", ".codebook.cluster_usage")
        target = target.replace("._codebook.embedding_sum", ".codebook.embed_sum")
        mapped[target] = tensor
    return mapped


class _StreamConv1d:
    """Moshi ``RawStreamingConv1d``: causal left-context carry per call.

    ``pad_mode`` controls the stream-start left padding: SEANet convs use zeros
    (``constant``); the down/upsample resamplers use ``replicate`` (the first
    real sample), marked per row so elastic slot recycling re-primes correctly.

    The carry and the fresh-row flags are updated in place, so CUDA graph replays advance them.
    """

    def __init__(self, conv: nn.Conv1d, pad_mode: str = "constant") -> None:
        self.conv = conv
        self.kernel = (conv.kernel_size[0] - 1) * conv.dilation[0] + 1
        self.stride = conv.stride[0]
        self.pad_mode = pad_mode
        self.prev: torch.Tensor | None = None
        self._fresh: torch.Tensor | None = None

    def reset(self, batch_size: int, device, dtype) -> None:
        pad = self.kernel - self.stride
        self.prev = torch.zeros(batch_size, self.conv.in_channels, pad, device=device, dtype=dtype)
        self._fresh = torch.ones(batch_size, dtype=torch.bool, device=device)

    def reset_slot(self, b: int) -> None:
        # In-place fills: assigning a Python scalar would sync the host.
        self.prev[b].zero_()
        self._fresh[b].fill_(True)

    def reset_all_slots(self) -> None:
        self.prev.zero_()
        self._fresh.fill_(True)

    def __call__(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if self.pad_mode == "replicate":
            pad = self.prev.shape[-1]
            edge = x[..., 0:1].expand(-1, -1, pad)
            fresh = (self._fresh & active).view(-1, 1, 1)
            self.prev.copy_(torch.where(fresh, edge.to(self.prev.dtype), self.prev))
        self._fresh.logical_and_(~active)
        x = torch.cat([self.prev, x], dim=-1)
        t = x.shape[-1]
        num_frames = max(0, (t - self.kernel) // self.stride + 1)
        prev = x[..., num_frames * self.stride :]
        self.prev.copy_(torch.where(active.view(-1, 1, 1), prev, self.prev))
        if num_frames == 0:
            return x.new_zeros(x.shape[0], self.conv.out_channels, 0)
        return self.conv(x[..., : (num_frames - 1) * self.stride + self.kernel])


class _StreamConvTr1d:
    """Moshi ``RawStreamingConvTranspose1d``: overlap-add tail carry per call."""

    def __init__(self, conv: nn.ConvTranspose1d) -> None:
        self.conv = conv
        self.kernel = conv.kernel_size[0]
        self.stride = conv.stride[0]
        self.partial: torch.Tensor | None = None
        self._fresh: torch.Tensor | None = None

    def reset(self, batch_size: int, device, dtype) -> None:
        self.partial = torch.zeros(
            batch_size, self.conv.out_channels, self.kernel - self.stride, device=device, dtype=dtype
        )
        self._fresh = torch.ones(batch_size, dtype=torch.bool, device=device)

    def reset_slot(self, b: int) -> None:
        self.partial[b].zero_()
        self._fresh[b].fill_(True)

    def reset_all_slots(self) -> None:
        self.partial.zero_()
        self._fresh.fill_(True)

    def __call__(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        length = out.shape[-1]
        tail = self.kernel - self.stride
        pt = self.partial.shape[-1]
        merge = self.partial
        if self.conv.bias is not None:
            # The carried tail already includes the bias; the fresh output adds
            # it again, so subtract one copy -- except on a row's very first
            # frame, where the carry is zeros by construction.
            fresh = (self._fresh & active).view(-1, 1, 1)
            merge = torch.where(fresh, 0.0, merge - self.conv.bias[:, None])
            self._fresh.logical_and_(~active)
        out[..., :pt] += merge
        self.partial.copy_(torch.where(active.view(-1, 1, 1), out[..., length - tail :], self.partial))
        return out[..., : length - tail]


class _MimiTransformerLayer(nn.Module):
    """Moshi mimi transformer layer: LayerNorm, LayerScale, GELU FFN, no biases."""

    def __init__(self, dim: int = 512, num_heads: int = 8, ffn: int = 2048) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.out_proj_weight = nn.Parameter(torch.empty(dim, dim))
        self.linear1 = nn.Parameter(torch.empty(ffn, dim))
        self.linear2 = nn.Parameter(torch.empty(dim, ffn))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(
        self,
        x: torch.Tensor,
        kv: _RingKV,
        offset: torch.Tensor,
        context: int,
        active: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm1(x)
        qkv = F.linear(h, self.in_proj_weight)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, offset)
        keys, values, pos_k = kv.complete(k, v, active=active)
        pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < context)
        attn = F.scaled_dot_product_attention(q, keys, values, attn_bias.unsqueeze(1), dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, T, self.dim)
        x = x + self.scale1 * F.linear(attn, self.out_proj_weight)
        h = self.norm2(x)
        h = F.linear(F.gelu(F.linear(h, self.linear1)), self.linear2)
        return x + self.scale2 * h


class _MimiStreamingTransformer(nn.Module):
    """The 8-layer mimi encoder/decoder transformer as a stateful stepper."""

    def __init__(self, num_layers: int = 8, dim: int = 512, num_heads: int = 8, context: int = 250) -> None:
        super().__init__()
        self.context = context
        self.layers = nn.ModuleList([_MimiTransformerLayer(dim, num_heads) for _ in range(num_layers)])
        self._kv: list[_RingKV] | None = None
        self._offset: torch.Tensor | None = None

    def streaming_init(self, batch_size: int) -> None:
        p = next(self.parameters())
        heads = self.layers[0].num_heads
        hd = self.layers[0].head_dim
        self._kv = [_RingKV(batch_size, heads, hd, self.context, p.device, p.dtype) for _ in self.layers]
        self._offset = torch.zeros(batch_size, device=p.device, dtype=torch.long)

    def reset_streaming(self) -> None:
        for kv in self._kv:
            kv.reset()
        self._offset.zero_()

    def reset_slot(self, b: int) -> None:
        # A recycled row restarts at position 0, exactly like a fresh stream,
        # instead of carrying its predecessor's absolute RoPE positions.
        for kv in self._kv:
            kv.reset_row(b)
        self._offset[b].zero_()

    def step(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T, dim]`` (T = positions this frame, typically 2)."""
        for layer, kv in zip(self.layers, self._kv):
            x = layer(x, kv, self._offset, self.context, active)
        self._offset.add_(x.shape[1] * active.to(self._offset.dtype))
        return x

    def load_weights(self, state_dict: dict[str, torch.Tensor], prefix: str) -> int:
        loaded = 0
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                base = f"{prefix}.transformer.layers.{i}"
                pairs = [
                    (layer.in_proj_weight, f"{base}.self_attn.in_proj_weight"),
                    (layer.out_proj_weight, f"{base}.self_attn.out_proj.weight"),
                    (layer.linear1, f"{base}.linear1.weight"),
                    (layer.linear2, f"{base}.linear2.weight"),
                    (layer.norm1.weight, f"{base}.norm1.weight"),
                    (layer.norm1.bias, f"{base}.norm1.bias"),
                    (layer.norm2.weight, f"{base}.norm2.weight"),
                    (layer.norm2.bias, f"{base}.norm2.bias"),
                    (layer.scale1, f"{base}.layer_scale_1.scale"),
                    (layer.scale2, f"{base}.layer_scale_2.scale"),
                ]
                for param, name in pairs:
                    param.data.copy_(state_dict[name].reshape(param.shape).to(param.dtype))
                    loaded += 1
        return loaded


def _walk_seanet(layers) -> list[tuple[str, object]]:
    """Wrap a Hugging Face Mimi SEANet layer list with streaming conv state."""
    stages: list[tuple[str, object]] = []
    for layer in layers:
        kind = type(layer).__name__
        if kind == "MimiConv1d":
            stages.append(("conv", _StreamConv1d(layer.conv)))
        elif kind == "MimiConvTranspose1d":
            stages.append(("convtr", _StreamConvTr1d(layer.conv)))
        elif kind == "ELU":
            stages.append(("act", layer))
        elif kind == "MimiResnetBlock":
            block = (
                layer.block[0],
                _StreamConv1d(layer.block[1].conv),
                layer.block[2],
                _StreamConv1d(layer.block[3].conv),
            )
            stages.append(("res", block))
        else:  # pragma: no cover - unexpected layer type
            raise ValueError(f"unhandled Mimi SEANet layer: {kind}")
    return stages


def _seanet_conv_states(stages) -> list[_StreamConv1d | _StreamConvTr1d]:
    """The streaming conv states of a ``_walk_seanet`` stage list."""
    states = []
    for _, stage in stages:
        if isinstance(stage, (_StreamConv1d, _StreamConvTr1d)):
            states.append(stage)
        elif isinstance(stage, tuple):
            states += [stage[1], stage[3]]
    return states


def _residual_decode(rvq: nn.Module, codes: torch.Tensor) -> torch.Tensor:
    """``MimiResidualVectorQuantizer.decode`` of ``[B, K, T]`` codes, summed from the first codebook.

    Transformers seeds the sum with ``torch.tensor(0.0, device=...)``, a copy a CUDA graph cannot capture.
    """
    per_codebook = codes.transpose(0, 1)
    out = rvq.layers[0].decode(per_codebook[0])
    for layer, indices in zip(rvq.layers[1:], per_codebook[1:]):
        out = out + layer.decode(indices)
    if rvq.output_proj is not None:
        out = rvq.output_proj(out)
    return out


@dataclass
class _CodecGraph:
    """One codec call over every streaming row, captured into a CUDA graph."""

    graph: torch.cuda.CUDAGraph
    inputs: torch.Tensor
    active: torch.Tensor
    output: torch.Tensor

    def replay(self, inputs: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        self.inputs.copy_(inputs.reshape(self.inputs.shape))
        self.active.copy_(active)
        self.graph.replay()
        return self.output.clone()


class PersonaPlexMimiCodec(nn.Module):
    """Streaming Mimi encode/decode at one 80 ms frame per call (moshi-free)."""

    def __init__(self, hf_repo: str = DEFAULT_HF_REPO, checkpoint: str | None = None, device: str = "cuda") -> None:
        super().__init__()
        from safetensors.torch import load_file
        from transformers import MimiConfig, MimiModel

        from vllm_omni.transformers_utils.repo_utils import hf_api

        self.device = torch.device(device)

        # The PersonaPlex repo ships the reference mimi checkpoint in the moshi
        # fused layout. Build the matching Transformers graph locally, then load
        # every codec tensor from that bundled checkpoint. Only the two HF
        # transformer stacks remain absent because the streaming implementations
        # below replace them with the checkpoint's fused QKV layout.
        if checkpoint is None:
            checkpoint = hf_api().hf_hub_download(
                "nvidia/personaplex-7b-v1",
                "tokenizer-e351c8d8-checkpoint125.safetensors",
            )
        sd = load_file(checkpoint, device=str(self.device))
        self.model = MimiModel(MimiConfig())
        codec_state = _map_moshi_codec_weights(sd)
        incompatible = self.model.load_state_dict(codec_state, strict=False)
        expected_missing = {
            name
            for name in self.model.state_dict()
            if name.startswith(("encoder_transformer.", "decoder_transformer."))
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "PersonaPlex Mimi checkpoint did not exactly cover the local codec graph: "
                f"missing={sorted(set(incompatible.missing_keys) - expected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
        # The fused streaming transformers below replace these unloaded HF
        # stacks. Drop their randomly initialized weights before moving Mimi to
        # the target device.
        del self.model.encoder_transformer, self.model.decoder_transformer
        self.model = self.model.to(self.device).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.encoder_transformer = _MimiStreamingTransformer().to(self.device, self.dtype)
        self.decoder_transformer = _MimiStreamingTransformer().to(self.device, self.dtype)
        n_enc = self.encoder_transformer.load_weights(sd, "encoder_transformer")
        n_dec = self.decoder_transformer.load_weights(sd, "decoder_transformer")
        assert n_enc == n_dec == 80, (n_enc, n_dec)
        del sd

        m = self.model
        self._enc_stages = _walk_seanet(m.encoder.layers)
        self._downsample = _StreamConv1d(m.downsample.conv, pad_mode="replicate")
        self._upsample = _StreamConvTr1d(m.upsample.conv)
        self._dec_stages = _walk_seanet(m.decoder.layers)
        self._batch_size: int | None = None
        # The halves streaming_init allocated rows for: Stage 0 only encodes, Stage 1 only decodes.
        self._halves = {"encode": False, "decode": False}
        self._all_active: torch.Tensor
        self._encode_graph: _CodecGraph | None = None
        self._decode_graph: _CodecGraph | None = None

    # -- streaming state ------------------------------------------------------

    def _half_state(self, half: str) -> tuple[list, _MimiStreamingTransformer]:
        """The conv states and the transformer that carry the ``"encode"`` or ``"decode"`` half's stream."""
        if half == "encode":
            return [*_seanet_conv_states(self._enc_stages), self._downsample], self.encoder_transformer
        return [self._upsample, *_seanet_conv_states(self._dec_stages)], self.decoder_transformer

    def _allocated_halves(self) -> list[tuple[list, _MimiStreamingTransformer]]:
        return [self._half_state(half) for half, allocated in self._halves.items() if allocated]

    def _conv_states(self):
        for convs, _ in self._allocated_halves():
            yield from convs

    def _require(self, half: str) -> None:
        if not self._halves[half]:
            raise RuntimeError(
                f"PersonaPlex Mimi has no {half} streaming state; call streaming_init(batch_size, {half}=True) first"
            )

    def streaming_init(self, batch_size: int, *, encode: bool = True, decode: bool = True) -> None:
        """Allocate ``batch_size`` streaming rows of the encoder half, the decoder half, or both.

        A half left out (Stage 0 never decodes, Stage 1 never encodes) gets no conv carries and no
        transformer ring KV, and calling into it raises.
        """
        if not (encode or decode):
            raise ValueError("PersonaPlex Mimi streaming_init needs encode=True, decode=True or both")
        # New state buffers invalidate a graph captured over the old ones.
        self._encode_graph = None
        self._decode_graph = None
        self._batch_size = batch_size
        self._halves = {"encode": encode, "decode": decode}
        self._all_active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        for convs, transformer in self._allocated_halves():
            for state in convs:
                state.reset(batch_size, self.device, self.dtype)
            transformer.streaming_init(batch_size)

    def reset_streaming(self) -> None:
        assert self._batch_size is not None
        for convs, transformer in self._allocated_halves():
            for state in convs:
                state.reset_all_slots()
            transformer.reset_streaming()

    def reset_slot(self, b: int) -> None:
        for convs, transformer in self._allocated_halves():
            for state in convs:
                state.reset_slot(b)
            transformer.reset_slot(b)

    # -- per-frame codec -------------------------------------------------------

    @staticmethod
    def _run_stages(x: torch.Tensor, stages, active: torch.Tensor) -> torch.Tensor:
        for kind, stage in stages:
            if kind == "res":
                act0, conv1, act2, conv3 = stage
                x = x + conv3(act2(conv1(act0(x), active)), active)
            else:
                x = stage(x, active) if kind in {"conv", "convtr"} else stage(x)
        return x

    def _dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, 8, T]`` codes -> latents, the same sum as ``quantizer.decode`` but capturable."""
        quantizer = self.model.quantizer
        semantic = quantizer.num_semantic_quantizers
        out = _residual_decode(quantizer.semantic_residual_vector_quantizer, codes[:, :semantic])
        if codes.shape[1] > semantic:
            out = out + _residual_decode(quantizer.acoustic_residual_vector_quantizer, codes[:, semantic:])
        return out

    # -- CUDA graphs ------------------------------------------------------------

    @torch.no_grad()
    def _capture(
        self,
        kind: str,
        run: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> _CodecGraph | None:
        """Capture ``run(inputs, active)`` over every streaming row; None (run eagerly) off CUDA or on failure.

        Warmup advances every row's streaming state, so all rows are reset
        afterwards; the graph keeps the addresses of the state buffers, which
        are only ever updated in place.
        """
        if self._batch_size is None:
            raise RuntimeError("PersonaPlex Mimi streaming_init must run before capturing a graph")
        if self.device.type != "cuda":
            return None
        inputs = torch.zeros(self._batch_size, *shape, device=self.device, dtype=dtype)
        active = torch.ones_like(self._all_active)
        try:
            stream = graph_stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(_GRAPH_WARMUP_ITERS):
                    run(inputs, active)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # A private pool: the codec replays between the stage's other
            # graphs in an order unrelated to their capture order.
            with torch.cuda.graph(
                graph,
                pool=torch.cuda.graph_pool_handle(),
                stream=stream,
                capture_error_mode="thread_local",
            ):
                output = run(inputs, active)
        except Exception:
            logger.warning("PersonaPlex Mimi CUDA graph capture failed; the codec runs eagerly", exc_info=True)
            return None
        finally:
            self.reset_streaming()
        logger.info("Captured the PersonaPlex Mimi %s CUDA graph for %d streaming rows", kind, self._batch_size)
        return _CodecGraph(graph=graph, inputs=inputs, active=active, output=output)

    def capture_encode_graph(self) -> bool:
        """Replay full-batch ``encode_frame`` calls from a CUDA graph; returns whether one is in use."""
        self._require("encode")
        if self._encode_graph is None:
            self._encode_graph = self._capture("encoder", self._encode_frame, (FRAME_SIZE,), self.dtype)
        return self._encode_graph is not None

    def capture_decode_graph(self) -> bool:
        """Replay full-batch ``decode_frame`` calls (not ``decode_frames``) from a CUDA graph."""
        self._require("decode")
        if self._decode_graph is None:
            self._decode_graph = self._capture("decoder", self._decode_frame, (CODEBOOKS,), torch.long)
        return self._decode_graph is not None

    # -- per-frame codec -------------------------------------------------------

    @torch.no_grad()
    def encode_frame(self, pcm: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, frame_size]`` float PCM -> ``[B, 8]`` codes."""
        self._require("encode")
        active = _normalize_active(active, self._all_active)
        graph = self._encode_graph
        if graph is not None and pcm.shape[0] == self._batch_size and not torch.cuda.is_current_stream_capturing():
            return graph.replay(pcm, active)
        return self._encode_frame(pcm, active)

    def _encode_frame(self, pcm: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        x = pcm.to(self.device, self.dtype).view(-1, 1, FRAME_SIZE)
        x = self._run_stages(x, self._enc_stages, active)
        x = self.encoder_transformer.step(x.transpose(1, 2), active).transpose(1, 2)
        x = self._downsample(x, active)
        codes = self.model.quantizer.encode(x, num_quantizers=CODEBOOKS)  # [Q, B, T]
        return codes[:CODEBOOKS, :, 0].transpose(0, 1).contiguous()

    @torch.no_grad()
    def decode_frame(self, codes: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, 8]`` codes -> ``[B, frame_size]`` float PCM."""
        self._require("decode")
        active = _normalize_active(active, self._all_active)
        graph = self._decode_graph
        if graph is not None and codes.shape[0] == self._batch_size and not torch.cuda.is_current_stream_capturing():
            return graph.replay(codes, active)
        return self._decode_frame(codes, active)

    def _decode_frame(self, codes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        emb = self._dequantize(codes.to(self.device).view(-1, CODEBOOKS, 1))
        emb = self._upsample(emb, active)
        emb = self.decoder_transformer.step(emb.transpose(1, 2), active).transpose(1, 2)
        x = self._run_stages(emb, self._dec_stages, active)
        return x[:, 0, :]

    def decode_frames(self, codes: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, 8, F]`` codes -> ``[B, F * frame_size]`` float PCM."""
        self._require("decode")
        emb = self._dequantize(codes.to(self.device))
        active = _normalize_active(active, self._all_active)
        emb = self._upsample(emb, active)
        emb = self.decoder_transformer.step(emb.transpose(1, 2), active).transpose(1, 2)
        x = self._run_stages(emb, self._dec_stages, active)
        return x[:, 0, :]
