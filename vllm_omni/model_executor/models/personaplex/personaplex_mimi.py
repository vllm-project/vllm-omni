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
    reimplemented here on the ring KV from ``personaplex_temporal.py``
    and loaded directly from the PersonaPlex
    checkpoint's fused layout (LayerNorm + per-layer LayerScale + GELU FFN).

All per-stream state is ``[B, ...]`` with per-row reset (``reset_slot``), so the
codec composes with elastic slot recycling in batched duplex serving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    _apply_rope,
    _RingKV,
    _ringkv_positions,
    _rope_tables,
)

DEFAULT_HF_REPO = "kyutai/mimi"
FRAME_SIZE = 1920
CODEBOOKS = 8

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.personaplex.personaplex_mimi_cudagraph import MimiFrameGraph


def _normalize_active(active: torch.Tensor | None, all_active: torch.Tensor) -> torch.Tensor:
    # Stage 0 shares one codec with a row per session slot and passes bool[B]:
    # True advances that row's streaming state; False keeps an absent or padded
    # row's offsets and convolution carries unchanged. Code2Wav leases one B=1
    # codec per request and omits `active`, which means all rows are active.
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

    The carry and the fresh-row flags are allocated once in ``reset`` and only
    ever updated in place, so a CUDA graph recorded over ``__call__`` keeps
    reading and writing the live state across frames, row resets and sessions.
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

    def reset_all(self) -> None:
        self.prev.zero_()
        self._fresh.fill_(True)

    def reset_slot(self, b: int) -> None:
        self.prev[b].zero_()
        self._fresh[b] = True

    def __call__(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        active_view = active.view(-1, 1, 1)
        if self.pad_mode == "replicate":
            pad = self.prev.shape[-1]
            edge = x[..., 0:1].expand(-1, -1, pad)
            fresh = (self._fresh & active).view(-1, 1, 1)
            self.prev.copy_(torch.where(fresh, edge.to(self.prev.dtype), self.prev))
        self._fresh.logical_and_(~active)
        x = torch.cat([self.prev, x], dim=-1)
        t = x.shape[-1]
        num_frames = max(0, (t - self.kernel) // self.stride + 1)
        self.prev.copy_(torch.where(active_view, x[..., num_frames * self.stride :], self.prev))
        if num_frames == 0:
            return x.new_zeros(x.shape[0], self.conv.out_channels, 0)
        return self.conv(x[..., : (num_frames - 1) * self.stride + self.kernel])


class _StreamConvTr1d:
    """Moshi ``RawStreamingConvTranspose1d``: overlap-add tail carry per call.

    Like ``_StreamConv1d``, the tail carry is updated in place only.
    """

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

    def reset_all(self) -> None:
        self.partial.zero_()
        self._fresh.fill_(True)

    def reset_slot(self, b: int) -> None:
        self.partial[b].zero_()
        self._fresh[b] = True

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
            merge = merge - self.conv.bias[:, None]
            first = (self._fresh & active).view(-1, 1, 1)
            merge = torch.where(first, torch.zeros_like(merge), merge)
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
        rope: tuple[torch.Tensor, torch.Tensor],
        ring: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm1(x)
        qkv = F.linear(h, self.in_proj_weight)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, *rope)
        keys, values, pos_k = kv.complete(k, v, active, *ring)
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
        self._offset[b] = 0

    def step(self, x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T, dim]`` (T = positions this frame, typically 2)."""
        T = x.shape[1]
        # Offset-pure tables, identical for every layer: build once per step.
        rope = _rope_tables(self._offset, T, self.layers[0].head_dim)
        ring = _ringkv_positions(self._offset, T, self.context, active)
        for layer, kv in zip(self.layers, self._kv):
            x = layer(x, kv, self._offset, self.context, active, rope, ring)
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
        self._init_streaming_stages()

    def _init_streaming_stages(self) -> None:
        m = self.model
        self._enc_stages = _walk_seanet(m.encoder.layers)
        self._downsample = _StreamConv1d(m.downsample.conv, pad_mode="replicate")
        self._upsample = _StreamConvTr1d(m.upsample.conv)
        self._dec_stages = _walk_seanet(m.decoder.layers)
        self._batch_size: int | None = None
        self._cuda_graphs: dict[str, MimiFrameGraph] = {}
        self._all_active: torch.Tensor

    # -- streaming state ------------------------------------------------------

    def _conv_states(self):
        for _, stage in (*self._enc_stages, *self._dec_stages):
            if isinstance(stage, (_StreamConv1d, _StreamConvTr1d)):
                yield stage
            elif isinstance(stage, tuple):
                yield stage[1]
                yield stage[3]
        yield self._downsample
        yield self._upsample

    def streaming_init(self, batch_size: int) -> None:
        # Re-initializing at the same batch size resets in place: the state
        # tensors keep their storage, so captured frame graphs stay valid.
        if getattr(self, "_batch_size", None) == batch_size:
            self.reset_streaming()
            return
        if getattr(self, "_cuda_graphs", None):
            # The state is reallocated below, so graphs recorded at the old
            # batch size would read stale tensors; drop them and say so.
            logger.warning(
                "PersonaPlex Mimi stream resized from %s to %d rows; dropping its CUDA graphs (%s)",
                self._batch_size,
                batch_size,
                "/".join(sorted(self._cuda_graphs)),
            )
        self._batch_size = batch_size
        self._cuda_graphs = {}
        self._all_active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        for s in self._conv_states():
            s.reset(batch_size, self.device, self.dtype)
        self.encoder_transformer.streaming_init(batch_size)
        self.decoder_transformer.streaming_init(batch_size)

    def reset_streaming(self) -> None:
        assert self._batch_size is not None
        for state in self._conv_states():
            state.reset_all()
        self.encoder_transformer.reset_streaming()
        self.decoder_transformer.reset_streaming()

    def reset_slot(self, b: int) -> None:
        for s in self._conv_states():
            s.reset_slot(b)
        self.encoder_transformer.reset_slot(b)
        self.decoder_transformer.reset_slot(b)

    def capture_cuda_graphs(
        self,
        *,
        encode: bool = True,
        decode_frame_counts: tuple[int, ...] = (1,),
        warmup_iters: int = 2,
        pool: tuple[int, int] | None = None,
    ) -> list[str]:
        """Record the per-frame codec steps at the streaming batch size.

        ``encode`` records ``encode_frame``; each ``F`` in ``decode_frame_counts``
        records ``decode_frames`` over ``F`` frames (``decode_frame`` is ``F=1``).
        Every row is recorded active; the ``active`` mask is a graph input, so
        one graph serves any subset of live rows. Warmup frames run through the
        real streaming state, which is then reset in place, so the codec is
        fresh afterwards. ``pool`` is a CUDA graph memory pool to share with
        other codecs replayed on the same stream; by default the codec gets a
        private one. Returns the captured graph names; on a non-CUDA device the
        codec stays eager, and so it does after a recoverable capture failure,
        which is logged with its traceback. Warmup, device execution and reset
        failures propagate instead of leaving an apparently usable codec.
        """
        from vllm_omni.model_executor.models.personaplex.personaplex_mimi_cudagraph import (
            capture_mimi_frame_graphs,
        )

        assert self._batch_size is not None, "call streaming_init first"
        self._cuda_graphs = capture_mimi_frame_graphs(
            self,
            encode=encode,
            decode_frame_counts=decode_frame_counts,
            warmup_iters=warmup_iters,
            pool=pool,
        )
        return sorted(self._cuda_graphs)

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

    def _frame_graph(self, name: str) -> MimiFrameGraph | None:
        graph = self._cuda_graphs.get(name)
        if graph is None or torch.cuda.is_current_stream_capturing():
            return None
        return graph

    @torch.no_grad()
    def encode_frame(self, pcm: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, frame_size]`` float PCM -> ``[B, 8]`` codes."""
        active = _normalize_active(active, self._all_active)
        graph = self._frame_graph("encode")
        if graph is not None:
            return graph.replay(pcm.reshape(-1, FRAME_SIZE), active)
        return self._encode_frame_eager(pcm, active)

    def _encode_frame_eager(self, pcm: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        x = pcm.to(self.device, self.dtype).view(-1, 1, FRAME_SIZE)
        x = self._run_stages(x, self._enc_stages, active)
        x = self.encoder_transformer.step(x.transpose(1, 2), active).transpose(1, 2)
        x = self._downsample(x, active)
        codes = self.model.quantizer.encode(x, num_quantizers=CODEBOOKS)  # [Q, B, T]
        return codes[:CODEBOOKS, :, 0].transpose(0, 1).contiguous()

    def _quantizer_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """``MimiSplitResidualVectorQuantizer.decode`` without its host-side scalar.

        The Hugging Face RVQ decode seeds its sum with ``torch.tensor(0.0)``, a
        host-to-device copy that cannot be recorded in a CUDA graph. This
        seeds it with an on-device zero instead; the arithmetic is unchanged.
        """
        quantizer = self.model.quantizer
        split = quantizer.num_semantic_quantizers

        def rvq_decode(rvq, rvq_codes: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((), device=rvq_codes.device)
            for layer, indices in zip(rvq.layers, rvq_codes.transpose(0, 1)):
                out = out + layer.decode(indices)
            if rvq.output_proj is not None:
                out = rvq.output_proj(out)
            return out

        out = rvq_decode(quantizer.semantic_residual_vector_quantizer, codes[:, :split])
        if codes.shape[1] > split:
            out += rvq_decode(quantizer.acoustic_residual_vector_quantizer, codes[:, split:])
        return out

    @torch.no_grad()
    def decode_frame(self, codes: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, 8]`` codes -> ``[B, frame_size]`` float PCM."""
        return self.decode_frames(codes.reshape(-1, CODEBOOKS, 1), active)

    @torch.no_grad()
    def decode_frames(self, codes: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, 8, F]`` codes -> ``[B, F * frame_size]`` float PCM."""
        active = _normalize_active(active, self._all_active)
        graph = self._frame_graph(f"decode_f{codes.shape[-1]}")
        if graph is not None:
            return graph.replay(codes, active)
        return self._decode_frames_eager(codes, active)

    def _decode_frames_eager(self, codes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        emb = self._quantizer_decode(codes.to(self.device))
        emb = self._upsample(emb, active)
        emb = self.decoder_transformer.step(emb.transpose(1, 2), active).transpose(1, 2)
        x = self._run_stages(emb, self._dec_stages, active)
        return x[:, 0, :]
