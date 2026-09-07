# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    _apply_rope,
    _RingKV,
)

DEFAULT_HF_REPO = "kyutai/mimi"
FRAME_SIZE = 1920
CODEBOOKS = 8


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
        self._fresh = torch.ones(batch_size, dtype=torch.bool)

    def reset_slot(self, b: int) -> None:
        self.prev[b].zero_()
        self._fresh[b] = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_mode == "replicate" and bool(self._fresh.any()):
            pad = self.prev.shape[-1]
            edge = x[..., 0:1].expand(-1, -1, pad)
            fresh = self._fresh.to(x.device).view(-1, 1, 1)
            self.prev = torch.where(fresh, edge.to(self.prev.dtype), self.prev)
        self._fresh[:] = False
        x = torch.cat([self.prev, x], dim=-1)
        t = x.shape[-1]
        num_frames = max(0, (t - self.kernel) // self.stride + 1)
        self.prev = x[..., num_frames * self.stride :]
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

    def reset(self, batch_size: int, device, dtype) -> None:
        self.partial = torch.zeros(
            batch_size, self.conv.out_channels, self.kernel - self.stride, device=device, dtype=dtype
        )
        self._fresh = torch.ones(batch_size, dtype=torch.bool)

    def reset_slot(self, b: int) -> None:
        self.partial[b].zero_()
        self._fresh[b] = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        length = out.shape[-1]
        tail = self.kernel - self.stride
        pt = self.partial.shape[-1]
        merge = self.partial.clone()
        if self.conv.bias is not None:
            # The carried tail already includes the bias; the fresh output adds
            # it again, so subtract one copy -- except on a row's very first
            # frame, where the carry is zeros by construction.
            merge = merge - self.conv.bias[:, None]
            merge[self._fresh] = 0.0
            self._fresh[:] = False
        out[..., :pt] += merge
        self.partial = out[..., length - tail :].clone()
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

    def forward(self, x: torch.Tensor, kv: _RingKV, offset: torch.Tensor, context: int) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm1(x)
        qkv = F.linear(h, self.in_proj_weight)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, offset)
        keys, values, pos_k = kv.complete(k, v)
        pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
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
        self._offset = torch.zeros(1, device=p.device, dtype=torch.long)

    def reset_streaming(self) -> None:
        for kv in self._kv:
            kv.reset()
        self._offset.zero_()

    def reset_slot(self, b: int) -> None:
        for kv in self._kv:
            kv.reset_slot(b)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T, dim]`` (T = positions this frame, typically 2)."""
        for layer, kv in zip(self.layers, self._kv):
            x = layer(x, kv, self._offset, self.context)
        self._offset.add_(x.shape[1])
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
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import MimiConfig, MimiModel

        self.device = torch.device(device)

        # The PersonaPlex repo ships the reference mimi checkpoint in the moshi
        # fused layout. Build the matching Transformers graph locally, then load
        # every codec tensor from that bundled checkpoint. Only the two HF
        # transformer stacks remain absent because the streaming implementations
        # below replace them with the checkpoint's fused QKV layout.
        if checkpoint is None:
            checkpoint = hf_hub_download(
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
        self._batch_size = batch_size
        for s in self._conv_states():
            s.reset(batch_size, self.device, self.dtype)
        self.encoder_transformer.streaming_init(batch_size)
        self.decoder_transformer.streaming_init(batch_size)

    def reset_streaming(self) -> None:
        assert self._batch_size is not None
        for state in self._conv_states():
            for b in range(self._batch_size):
                state.reset_slot(b)
        self.encoder_transformer.reset_streaming()
        self.decoder_transformer.reset_streaming()

    def reset_slot(self, b: int) -> None:
        for s in self._conv_states():
            s.reset_slot(b)
        self.encoder_transformer.reset_slot(b)
        self.decoder_transformer.reset_slot(b)

    # -- per-frame codec -------------------------------------------------------

    @staticmethod
    def _run_stages(x: torch.Tensor, stages) -> torch.Tensor:
        for kind, stage in stages:
            if kind == "res":
                act0, conv1, act2, conv3 = stage
                x = x + conv3(act2(conv1(act0(x))))
            else:
                x = stage(x)
        return x

    @torch.no_grad()
    def encode_frame(self, pcm: torch.Tensor) -> torch.Tensor:
        """``[B, frame_size]`` float PCM -> ``[B, 8]`` codes."""
        x = pcm.to(self.device, self.dtype).view(-1, 1, FRAME_SIZE)
        x = self._run_stages(x, self._enc_stages)
        x = self.encoder_transformer.step(x.transpose(1, 2)).transpose(1, 2)
        x = self._downsample(x)
        codes = self.model.quantizer.encode(x)  # [Q, B, T]
        return codes[:CODEBOOKS, :, 0].transpose(0, 1).contiguous()

    @torch.no_grad()
    def decode_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, 8]`` codes -> ``[B, frame_size]`` float PCM."""
        emb = self.model.quantizer.decode(codes.to(self.device).view(-1, CODEBOOKS, 1))
        emb = self._upsample(emb)
        emb = self.decoder_transformer.step(emb.transpose(1, 2)).transpose(1, 2)
        x = self._run_stages(emb, self._dec_stages)
        return x[:, 0, :]

    @torch.no_grad()
    def decode_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """``[B, 8, F]`` codes -> ``[B, F * frame_size]`` float PCM."""
        emb = self.model.quantizer.decode(codes.to(self.device))
        emb = self._upsample(emb)
        emb = self.decoder_transformer.step(emb.transpose(1, 2)).transpose(1, 2)
        x = self._run_stages(emb, self._dec_stages)
        return x[:, 0, :]
