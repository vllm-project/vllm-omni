# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete incremental ZAC-1 Codec and native Stage-1 entry.

Neural modules and persistent state precede release checkpoint loading,
synthesis, slot management and the final Zonos3CodecModel entry class.
CUDA capture/replay remains in cuda_graph_wrapper.py. Adapted from the
Zyphra/ZONOS3-Mini release; see LICENSE.Zyphra.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import Resample
from vllm.config import VllmConfig

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking
from vllm_omni.worker_v2.warmup import is_warmup_request


class ZACTransformerSpec(TypedDict):
    d_model: int
    num_heads: int
    context_frames: int
    dim_feedforward: int
    layer_scale: float
    input_dimension: int
    output_dimension: int
    num_layers: int


@dataclass
class ZACPackedRingState:
    """Tensor-ring storage, with distinct non-leaseable padding rows.

    ``max_chunk`` is expressed in this layer's tokens. The extra rows are not
    slots: each padding token has its own destination, avoiding scatter races.
    """

    keys: torch.Tensor
    values: torch.Tensor
    capacity: int
    ring_capacity: int
    max_chunk: int


def zac_packed_ring_attention(
    layer: ZACMossLayer,
    x: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rvq_lengths: torch.Tensor,
    state: ZACPackedRingState,
    *,
    scale: int = 1,
) -> torch.Tensor:
    """Packed-Q sliding-window attention; no per-request attention execution.

    Metadata and cumulative RVQ lengths stay on device. Only trailing capacity
    rows are padding. The caller validates each CPU chunk against max_chunk
    before mutation. A layer's offsets and absolute positions derive from its
    fixed RVQ upsampling scale. This does not advance the shared RVQ cursor.

    Writing the complete chunk first is safe: R >= W-1+M, hence the last
    new key cannot alias the earliest query's oldest required key.
    """
    rows, width = x.shape
    if state.ring_capacity < layer.context - 1 + state.max_chunk:
        raise ValueError("Packed ring cannot retain a full multi-Q causal window")
    if state.keys.shape[0] < state.capacity * state.ring_capacity + rows:
        raise ValueError("Packed ring needs a distinct scratch row per capacity token")
    row = torch.arange(rows, device=x.device)
    offsets = cu_seqlens * scale
    torch._assert_async(
        ((offsets[1:] - offsets[:-1]) <= state.max_chunk).all(),
        "Packed ring chunk exceeds the layer max_chunk; split before decoding",
    )
    if x.device.type == "cuda":
        from .packed_kernels import segment_layout

        request, local, valid = segment_layout(offsets, rows)
    else:
        request = torch.searchsorted(offsets[1:], row, right=True).clamp_max(slot_ids.numel() - 1)
        valid = row < offsets[-1]
        local = row - offsets[request]
    slot = slot_ids[request]
    origin = rvq_lengths[slot] * scale
    position = origin + local
    q, k, v = layer.qkv(layer.norm1(x)).reshape(rows, 3, layer.heads, layer.head_dim).unbind(1)
    frequency = torch.exp(
        torch.arange(layer.head_dim // 2, device=x.device, dtype=torch.float32)
        * (-math.log(10000.0) * 2 / layer.head_dim)
    )
    phase = position.float()[:, None] * frequency
    cos, sin = phase.cos()[:, None], phase.sin()[:, None]

    def rope(z: torch.Tensor) -> torch.Tensor:
        real, imag = z.float().reshape(rows, layer.heads, layer.head_dim // 2, 2).unbind(-1)
        return torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).flatten(-2).to(z.dtype)

    q, k = rope(q), rope(k)
    address = torch.where(
        valid,
        slot * state.ring_capacity + position.remainder(state.ring_capacity),
        state.capacity * state.ring_capacity + row,
    )
    if x.device.type == "cuda":
        from .packed_kernels import packed_ring_attention

        attention = packed_ring_attention(
            q,
            k,
            v,
            address,
            state.keys,
            state.values,
            slot,
            position,
            origin,
            valid,
            state.ring_capacity,
            layer.context,
        )
    else:
        attention = _zac_packed_ring_reference(layer, q, k, v, slot, position, valid, address, state)
    x = x + layer.attention_scale * layer.attention_out(attention.reshape(rows, width))
    result = x + layer.ffn_scale * layer.ffn(layer.norm2(x))
    return torch.where(valid[:, None], result, 0.0)


def _zac_packed_ring_reference(layer, q, k, v, slot, position, valid, address, state):
    """CPU reference; materializes one fixed window per packed Q."""
    state.keys.index_copy_(0, address, k.to(state.keys.dtype))
    state.values.index_copy_(0, address, v.to(state.values.dtype))
    age = torch.arange(layer.context - 1, -1, -1, device=q.device)
    key_position = position[:, None] - age
    read = slot[:, None] * state.ring_capacity + key_position.remainder(state.ring_capacity)
    keys = state.keys[read].to(q.dtype).permute(0, 2, 1, 3)
    values = state.values[read].to(q.dtype).permute(0, 2, 1, 3)
    mask = (key_position >= 0) & valid[:, None]
    attention = F.scaled_dot_product_attention(q[:, :, None], keys, values, attn_mask=mask[:, None, None])
    return attention


class ZACMossLayer(nn.Module):
    """Moss projections and feed-forward modules for packed ring attention."""

    def __init__(self, spec: ZACTransformerSpec) -> None:
        super().__init__()
        d = spec["d_model"]
        self.heads = spec["num_heads"]
        if d % self.heads or (d // self.heads) % 2 or spec["context_frames"] < 1:
            raise ValueError("ZAC attention needs even head width and positive context")
        self.head_dim = d // self.heads
        self.context = spec["context_frames"]
        self.norm1 = nn.LayerNorm(d, eps=1e-5)
        self.norm2 = nn.LayerNorm(d, eps=1e-5)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.attention_out = nn.Linear(d, d, bias=False)
        # Dropout has no effect during inference; Identity preserves FFN keys.
        self.ffn = nn.Sequential(
            nn.Linear(d, spec["dim_feedforward"], bias=False),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(spec["dim_feedforward"], d, bias=False),
        )
        self.attention_scale = nn.Parameter(torch.full((d,), spec["layer_scale"]))
        self.ffn_scale = nn.Parameter(torch.full((d,), spec["layer_scale"]))


class ZACMossStage(nn.Module):
    def __init__(self, spec: ZACTransformerSpec, *, omit_output_projection: bool = False) -> None:
        super().__init__()
        self.input_proj = nn.Linear(spec["input_dimension"], spec["d_model"], bias=False)
        self.layers = nn.ModuleList([ZACMossLayer(spec) for _ in range(spec["num_layers"])])
        self.output_proj = (
            nn.Identity()
            if omit_output_projection
            else nn.Linear(spec["d_model"], spec["output_dimension"], bias=False)
        )


class ZACPatch(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 1:
            raise ValueError("ZAC upsampling patch size must be positive")
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        return x.reshape(b, c // self.size, self.size, t).permute(0, 1, 3, 2).reshape(b, c // self.size, t * self.size)


class ZACConvNeXt(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(width, width, 7, groups=width)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, hidden)
        self.pwconv2 = nn.Linear(hidden, width)

    def allocate_state(self, capacity: int, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        return torch.zeros((capacity, self.dwconv.in_channels, 6), dtype=dtype, device=device)


@dataclass(frozen=True)
class ZACDecoderConfig:
    stages: tuple[ZACTransformerSpec | int, ...]
    stage_layers: tuple[tuple[int, ...] | None, ...]
    semantic_width: int
    semantic_hidden: int
    semantic_depth: int
    codebook_dim: int
    latent_width: int
    head_depth: int
    head_hidden: int
    n_ffts: tuple[int, int]
    hop: int
    fir_kernel: int = 127
    codebook_size: int = 4096
    trainable_stages: tuple[int, ...] = ()

    @classmethod
    def from_release(cls, config: Mapping[str, Any]) -> ZACDecoderConfig:
        model, head = config["model"], config["head"]
        if (
            model["sample_rate"],
            model["internal_sample_rate"],
            model["frame_rate"],
            model["internal_hop_length"],
            model["external_samples_per_frame"],
            model["residual_codebooks"],
        ) != (44100, 44160, 15, 2944, 2940, 9):
            raise ValueError("Unsupported ZAC-1 rate, hop or codebook layout")
        if model["semantic_codebook_size"] != 4096 or model["residual_codebook_size"] != 4096:
            raise ValueError("ZAC-1 needs ten 4096-entry codebooks")
        if head["type"] != "mr_istft" or len(head["n_ffts"]) != 2:
            raise ValueError("ZAC-1 requires the multi-resolution iSTFT release head")
        specs = model["decoder_stages"]
        if len(specs) < 10 or specs[8]["frame_rate"] != 240 or specs[9].get("patch_size") != 23:
            raise ValueError("Unexpected ZAC-1 retained decoder layout")
        stages: list[ZACTransformerSpec | int] = []
        stage_layers: list[tuple[int, ...] | None] = []
        scale = 1
        for index, spec in enumerate(specs[:9]):
            if spec["type"] == "PatchedPretransform":
                if spec["direction"] != "up":
                    raise ValueError("ZAC decoder patch must upsample")
                stages.append(int(spec["patch_size"]))
                stage_layers.append(None)
                scale *= int(spec["patch_size"])
            elif spec["type"] == "MossProjectedTransformer":
                keep = tuple(head.get("stage_layers", {}).get(str(index), range(spec["num_layers"])))
                if not keep or tuple(sorted(set(keep))) != keep or keep[-1] >= spec["num_layers"] or keep[0] < 0:
                    raise ValueError("Invalid ZAC retained transformer layers")
                stages.append(cast(ZACTransformerSpec, {**spec, "num_layers": len(keep)}))
                stage_layers.append(keep)
            else:
                raise ValueError(f"Unsupported retained ZAC stage: {spec['type']}")
        if scale != 16:
            raise ValueError("ZAC-1 must upsample code frames by sixteen")
        trainable = tuple(int(i) for i in head.get("trainable_stages", ()))
        if trainable != tuple(range(9 - len(trainable), 9)):
            raise ValueError("Trainable ZAC stages must be a contiguous suffix of decoder.0..8")
        if int(head.get("fir_kernel", 127)) < 1 or int(head.get("fir_kernel", 127)) % 2 != 1:
            raise ValueError("ZAC FIR kernel must be positive and odd")
        dual = model["dual_encoding"]
        return cls(
            stages=tuple(stages),
            stage_layers=tuple(stage_layers),
            semantic_width=int(dual["width"]),
            semantic_hidden=int(dual["hidden"]),
            semantic_depth=int(dual["depth"]),
            codebook_dim=int(model["codebook_dim"]),
            latent_width=int(specs[0]["input_dimension"]),
            head_depth=int(head["depth"]),
            head_hidden=int(head["hidden"]),
            n_ffts=tuple(head["n_ffts"]),
            hop=184,
            fir_kernel=int(head.get("fir_kernel", 127)),
            trainable_stages=trainable,
        )


def _wn_conv(input_width: int, output_width: int) -> nn.Conv1d:
    # DAC release checkpoints use legacy weight_g/weight_v parameter names.
    return nn.utils.weight_norm(nn.Conv1d(input_width, output_width, 1))


class _VectorQuantizer(nn.Module):
    def __init__(self, output_width: int, codebook_dim: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.out_proj = _wn_conv(codebook_dim, output_width)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.embedding(codes, self.codebook.weight).transpose(1, 2))


class _RVQ(nn.Module):
    def __init__(self, count: int, width: int, codebook_dim: int, codebook_size: int) -> None:
        super().__init__()
        self.quantizers = nn.ModuleList([_VectorQuantizer(width, codebook_dim, codebook_size) for _ in range(count)])


class _DualQuantizer(nn.Module):
    def __init__(self, config: ZACDecoderConfig) -> None:
        super().__init__()
        self.semantic_quantizer = _RVQ(1, config.semantic_width, config.codebook_dim, config.codebook_size)
        self.quantizer = _RVQ(9, config.latent_width, config.codebook_dim, config.codebook_size)


class ZACSpectralHead(nn.Module):
    def __init__(self, width: int, config: ZACDecoderConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ZACConvNeXt(width, config.head_hidden) for _ in range(config.head_depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.outs = nn.ModuleList([nn.Linear(width, n + 2) for n in config.n_ffts])
        self.mixer = nn.Conv1d(2, 1, config.fir_kernel, padding=config.fir_kernel // 2, bias=False)
        self.max_log_magnitude = math.log(100.0)
        for i, n in enumerate(config.n_ffts):
            self.register_buffer(f"window_{i}", torch.hann_window(n), persistent=False)


@dataclass
class ZACPackedDecoderState:
    frames: torch.Tensor
    semantic_history: list[torch.Tensor]
    stage_history: list[list[ZACPackedRingState]]
    head_history: list[torch.Tensor]


def _packed_convnext(
    block: ZACConvNeXt, x: torch.Tensor, slots: torch.Tensor, offsets: torch.Tensor, history: torch.Tensor
) -> torch.Tensor:
    """Segmented depthwise convolution: only finite boundary context is gathered.

    The [tokens, 7, channels] stencil never introduces a batch time dimension.
    History is committed after every stencil has read the previous snapshot.
    """
    rows = x.shape[0]
    if x.device.type == "cuda":
        from .packed_kernels import segment_layout

        request, local, valid = segment_layout(offsets, rows)
    else:
        row = torch.arange(rows, device=x.device)
        request = torch.searchsorted(offsets[1:], row, right=True).clamp_max(slots.numel() - 1)
        valid = row < offsets[-1]
        local = row - offsets[request]
    old = history.transpose(1, 2)
    if x.device.type == "cuda":
        from .packed_kernels import segmented_depthwise

        h = segmented_depthwise(x, history, block.dwconv.weight, block.dwconv.bias, slots, request, local, valid)
    else:
        taps = torch.arange(7, device=x.device)
        source = row[:, None] + taps - 6
        from_history = local[:, None] + taps < 6
        boundary = old[slots[request, None], (local[:, None] + taps).clamp(0, 5)]
        fresh = x[source.clamp(0, rows - 1)]
        stencil = torch.where(from_history[:, :, None], boundary, fresh)
        h = (stencil * block.dwconv.weight[:, 0].T[None]).sum(1) + block.dwconv.bias
    result = x + block.pwconv2(F.gelu(block.pwconv1(block.norm(h))))
    if x.device.type == "cuda":
        from .packed_kernels import commit_conv_history

        commit_conv_history(x, history, slots, offsets)
    else:
        lengths = offsets[1:] - offsets[:-1]
        u = lengths[:, None] + torch.arange(6, device=x.device)
        previous = old[slots[:, None], u.clamp(0, 5)]
        current = x[(offsets[:-1, None] + u - 6).clamp(0, rows - 1)]
        next_history = torch.where((u < 6)[:, :, None], previous, current)
        history.index_copy_(0, slots, next_history.transpose(1, 2))
    return torch.where(valid[:, None], result, 0.0)


class ZACDecoderNetwork(nn.Module):
    def __init__(self, config: ZACDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.quantizer = _DualQuantizer(config)
        self.semantic_head = nn.ModuleList(
            [ZACConvNeXt(config.semantic_width, config.semantic_hidden) for _ in range(config.semantic_depth)]
            + [_wn_conv(config.semantic_width, config.latent_width)]
        )
        self.stages = nn.ModuleList(
            [
                ZACPatch(spec)
                if isinstance(spec, int)
                else ZACMossStage(spec, omit_output_projection=i == len(config.stages) - 1)
                for i, spec in enumerate(config.stages)
            ]
        )
        final_spec = config.stages[-1]
        if isinstance(final_spec, int):
            raise ValueError("Last retained ZAC stage must be a transformer")
        self.head = ZACSpectralHead(final_spec["d_model"], config)
        # Torchaudio builds its resampling kernel on CPU even when the network
        # is constructed inside vLLM's CUDA device context. Align the registered
        # buffer before either eager synthesis or graph capture uses it.
        self.to_external = Resample(44160, 44100, lowpass_filter_width=16, rolloff=0.99).to(
            device=next(self.parameters()).device
        )

    def allocate_packed_state(self, capacity: int, max_chunk: int, token_capacity: int) -> ZACPackedDecoderState:
        parameter = next(self.parameters())
        device = parameter.device
        cache_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        stages = []
        scale = 1
        for stage in self.stages:
            layers = []
            if isinstance(stage, ZACPatch):
                scale *= stage.size
            else:
                for layer in stage.layers:
                    ring = layer.context - 1 + max_chunk * scale
                    shape = (capacity * ring + token_capacity * scale, layer.heads, layer.head_dim)
                    layers.append(
                        ZACPackedRingState(
                            torch.zeros(shape, dtype=cache_dtype, device=device),
                            torch.zeros(shape, dtype=cache_dtype, device=device),
                            capacity,
                            ring,
                            max_chunk * scale,
                        )
                    )
            stages.append(layers)
        return ZACPackedDecoderState(
            torch.zeros(capacity, dtype=torch.long, device=device),
            [block.allocate_state(capacity, torch.float32, device) for block in self.semantic_head[:-1]],
            stages,
            [block.allocate_state(capacity, torch.float32, device) for block in self.head.blocks],
        )

    def forward_packed(
        self, codes: torch.Tensor, slots: torch.Tensor, cu_seqlens: torch.Tensor, state: ZACPackedDecoderState
    ) -> list[torch.Tensor]:
        """Continuous RVQ → continuous spectra, with padding only at capacity tail.

        All sequences share every projection, convolution stencil and attention
        launch. Each attention layer reads the original cumulative RVQ cursor;
        the cursor is committed once, after the complete neural forward.
        """
        device_type = codes.device.type
        with torch.autocast(device_type, enabled=False):
            # The pointwise checkpoint projections retain weight-norm prehooks.
            x = self.quantizer.semantic_quantizer.quantizers[0](codes[None, :, 0])[0].T
            for block, history in zip(self.semantic_head[:-1], state.semantic_history, strict=True):
                x = _packed_convnext(block, x, slots, cu_seqlens, history)
            z = self.semantic_head[-1](x.T[None])[0].T.float()
            for i, quantizer in enumerate(self.quantizer.quantizer.quantizers):
                code = codes[:, i + 1]
                contribution = quantizer(code.clamp_min(0)[None])[0].T
                z = z + torch.where(code[:, None] >= 0, contribution, 0.0)
        x, scale = z, 1
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            for stage, history in zip(self.stages, state.stage_history, strict=True):
                if isinstance(stage, ZACPatch):
                    x = x.reshape(x.shape[0], -1, stage.size).transpose(1, 2).reshape(x.shape[0] * stage.size, -1)
                    scale *= stage.size
                else:
                    # Legacy pointwise/stage projections consume column-major
                    # token matrices. Preserve that GEMM layout across packed
                    # patch stages: identical-value release experiments show
                    # row-major selection changes BF16 reduction substantially.
                    # This transposes one packed slab, never pads per request.
                    x = x.T.contiguous().T
                    x = stage.input_proj(x)
                    for layer, ring in zip(stage.layers, history, strict=True):
                        x = zac_packed_ring_attention(layer, x, slots, cu_seqlens, state.frames, ring, scale=scale)
                    x = stage.output_proj(x)
        with torch.autocast(device_type, enabled=False):
            x = x.float()
            for block, history in zip(self.head.blocks, state.head_history, strict=True):
                x = _packed_convnext(block, x, slots, cu_seqlens * scale, history)
            x = self.head.norm(x)
            output = [linear(x).float() for linear in self.head.outs]
        state.frames.index_add_(0, slots, cu_seqlens[1:] - cu_seqlens[:-1])
        return output

    def load_release_weights(
        self, generator: Mapping[str, torch.Tensor], trainable: Mapping[str, torch.Tensor]
    ) -> None:
        """Map generator stage pruning, then strictly apply the released head override."""
        own = self.state_dict()
        mapped = {}
        for key, value in generator.items():
            target = key
            if key.startswith("decoder."):
                parts = key.split(".")
                index = int(parts[1])
                if index >= len(self.stages) or key.startswith(f"decoder.{len(self.stages) - 1}.output_proj."):
                    continue
                keep = self.config.stage_layers[index]
                if len(parts) > 3 and parts[2] == "layers" and keep is not None:
                    old = int(parts[3])
                    if old not in keep:
                        continue
                    parts[3] = str(keep.index(old))
                parts[0] = "stages"
                target = ".".join(parts)
            if target in own and not target.startswith("head."):
                mapped[target] = value
        expected = {key for key in own if not key.startswith("head.")}
        if set(mapped) != expected:
            raise ValueError(f"Incomplete ZAC generator weights: {sorted(expected - set(mapped))[:8]}")
        prefixes = ("head.",) + tuple(f"stages.{i}." for i in self.config.trainable_stages)
        expected_head = {key for key in own if key.startswith(prefixes)}
        if set(trainable) != expected_head:
            raise ValueError("ZAC head checkpoint keys do not match the selected decoder variant")
        mapped.update(trainable)
        self.load_state_dict(mapped, strict=True)
        self.eval().requires_grad_(False)


GENERATOR_SHA256 = "75a587a3eda9ca3d345f0c305756a7ff8619b01685c261b08ae243c69c2caf9a"


@dataclass(frozen=True)
class Zonos3CodecFiles:
    config: str
    checkpoint: str
    generator: str = "autoencoder/generator.pt"

    @classmethod
    def for_variant(cls, variant: str) -> Zonos3CodecFiles:
        configs = {
            "keep2_plain": "config_full_unfreezeall_mr_drop30_4x10.json",
            "full_mr": "config_full_unfreezeall_mr.json",
        }
        if variant not in configs:
            raise ValueError(f"Unknown ZAC-1 decoder variant: {variant}")
        root = f"autoencoder/heads/{variant}"
        return cls(config=f"{root}/{configs[variant]}", checkpoint=f"{root}/step_0050000.pt")

    def resolve(self, model: str, cache_dir: str | None, revision: str | None, *, with_weights: bool) -> Path:
        if Path(model).is_dir():
            return Path(model)
        patterns = [self.config]
        if with_weights:
            patterns.extend((self.generator, self.checkpoint))
        return Path(download_weights_from_hf_specific(model, cache_dir, patterns, revision=revision, require_all=True))

    def read_config(self, root: Path) -> dict[str, Any]:
        with (root / self.config).open() as handle:
            config = json.load(handle)
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("model"), dict)
            or not isinstance(config.get("head"), dict)
        ):
            raise ValueError("Invalid ZAC decoder release configuration")
        return config

    def load(self, root: Path, network: ZACDecoderNetwork, config: dict[str, Any]) -> None:
        generator_path = root / self.generator
        with generator_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != GENERATOR_SHA256:
            raise ValueError(f"ZAC generator checksum mismatch: {digest}")
        generator = torch.load(generator_path, map_location="cpu", mmap=True, weights_only=True)
        if generator.get("step") != 600000 or generator.get("config", {}).get("model") != config["model"]:
            raise ValueError("ZAC generator step/config does not match the selected decoder release")
        head = torch.load(root / self.checkpoint, map_location="cpu", mmap=True, weights_only=True)
        if head.get("step") != 50000:
            raise ValueError("ZAC head checkpoint must be the released step 50000")
        if "trainable" not in head:
            raise ValueError("ZAC release head must contain its complete trainable state")
        network.load_release_weights(generator["generator"], head["trainable"])


class ZACPackedOLA:
    """Incremental packed iFFT/OLA with finite *time-domain* history.

    Each new spectrum is transformed once. The normalized branch outputs have
    shared packed offsets, retaining the larger FFT's lookahead until final.
    """

    def __init__(self, network: ZACDecoderNetwork, capacity: int, max_columns: int, token_capacity: int) -> None:
        self.device = next(network.parameters()).device
        self.capacity, self.max_columns = capacity, max_columns
        self.n_ffts, self.hop = network.config.n_ffts, network.config.hop
        self.lookahead = max(self.n_ffts) // 2
        self.ring = max_columns * self.hop + 2 * max(self.n_ffts)
        self.windows = [getattr(network.head, f"window_{i}").double() for i in range(2)]
        self.maximum = network.head.max_log_magnitude
        self.columns = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.emitted = torch.zeros_like(self.columns)
        self.ola = [
            torch.zeros(capacity * self.ring + token_capacity * n, dtype=torch.float64, device=self.device)
            for n in self.n_ffts
        ]
        self.envelope = [torch.zeros_like(x) for x in self.ola]

    def __call__(self, spectra, slots, offsets, final):
        count = offsets[1:] - offsets[:-1]
        torch._assert_async((count <= self.max_columns).all(), "Packed OLA chunk exceeds capacity")
        old = self.columns[slots]
        new = old + count
        start = self.emitted[slots]
        end = torch.where(final, new * self.hop, (new * self.hop - self.lookahead).clamp_min(0))
        lengths = end - start
        pcm_offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        # Only one trailing capacity pad, including bounded per-sequence flush.
        output_capacity = spectra[0].shape[0] * self.hop + slots.numel() * self.lookahead
        output_row = torch.arange(output_capacity, device=self.device)
        output_request = torch.searchsorted(pcm_offsets[1:], output_row, right=True).clamp_max(slots.numel() - 1)
        output_valid = output_row < pcm_offsets[-1]
        wave_position = start[output_request] + output_row - pcm_offsets[output_request]
        rows = spectra[0].shape[0]
        row = torch.arange(rows, device=self.device)
        request = torch.searchsorted(offsets[1:], row, right=True).clamp_max(slots.numel() - 1)
        valid = row < offsets[-1]
        column = old[request] + row - offsets[request]
        ring_position = torch.arange(self.ring, device=self.device)
        outputs = []
        for branch, (y, n, window) in enumerate(zip(spectra, self.n_ffts, self.windows, strict=True)):
            # Clear only newly exposed coordinates, before adding new frames.
            # Previous finalized samples remain available until shared lookahead
            # permits their emission. No historical spectrum is transformed.
            clear_lo = torch.where(old > 0, old * self.hop + n - self.hop, 0)
            clear_hi = torch.where(new > 0, new * self.hop + n - self.hop, 0)
            delta = (ring_position[None] - clear_lo[:, None]).remainder(self.ring)
            clear = delta < (clear_hi - clear_lo)[:, None]
            addresses = slots[:, None] * self.ring + ring_position
            for history in (self.ola[branch], self.envelope[branch]):
                previous = history[addresses]
                history[addresses] = torch.where(clear, 0.0, previous)
            log_mag, phase = y.double().split(n // 2 + 1, -1)
            magnitude = log_mag.clamp_max(self.maximum).exp()
            imaginary = magnitude * phase.sin()
            imaginary[:, 0] = imaginary[:, -1] = 0
            frames = torch.fft.irfft(torch.complex(magnitude * phase.cos(), imaginary), n, dim=-1) * window
            sample = torch.arange(n, device=self.device)
            destination = slots[request, None] * self.ring + (column[:, None] * self.hop + sample).remainder(self.ring)
            scratch = self.capacity * self.ring + row[:, None] * n + sample
            destination = torch.where(valid[:, None], destination, scratch)
            self.ola[branch].scatter_add_(0, destination.flatten(), torch.where(valid[:, None], frames, 0.0).flatten())
            envelope = window.square()[None].expand(rows, -1)
            self.envelope[branch].scatter_add_(
                0, destination.flatten(), torch.where(valid[:, None], envelope, 0.0).flatten()
            )
            source = slots[output_request] * self.ring + (wave_position + n // 2).remainder(self.ring)
            denominator = self.envelope[branch][source]
            waveform = self.ola[branch][source] / denominator.clamp_min(torch.finfo(torch.float64).tiny)
            outputs.append(torch.where(output_valid, waveform, 0.0))
        self.columns.index_copy_(0, slots, new)
        self.emitted.index_copy_(0, slots, end)
        return outputs, pcm_offsets, lengths


class ZACPackedSynthesis:
    """Complete packed synthesis, preserving FP64/FP32 boundaries.

    OLA retains raw time-domain contributions, waveform rings retain normalized
    samples and FIR output. CUDA FIR/resampling reduce taps inside segmented
    kernels, with linear global workspace. CPU uses the tensor reference.
    """

    def __init__(self, network: ZACDecoderNetwork, capacity: int, max_columns: int, token_capacity: int) -> None:
        self.ola = ZACPackedOLA(network, capacity, max_columns, token_capacity)
        self.device, self.capacity = self.ola.device, capacity
        rs = network.to_external
        self.orig, self.new, self.width = rs.orig_freq // rs.gcd, rs.new_freq // rs.gcd, rs.width
        self.kernel = rs.kernel[:, 0].detach().double()
        self.mixer = network.head.mixer.weight[0].detach().double()
        self.half_fir = self.mixer.shape[-1] // 2
        self.ring = self.ola.ring + 2 * (self.orig + self.width + self.mixer.shape[-1])
        self.wave_capacity = token_capacity * self.ola.hop + capacity * self.ola.lookahead
        self.mix_capacity = self.wave_capacity + capacity * self.half_fir
        self.wave = [
            torch.zeros(capacity * self.ring + self.wave_capacity, dtype=torch.float64, device=self.device)
            for _ in range(2)
        ]
        self.mixed = torch.zeros(capacity * self.ring + self.mix_capacity, dtype=torch.float64, device=self.device)
        self.mixed_end = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.emitted = torch.zeros_like(self.mixed_end)

    def _layout(self, starts, ends, capacity):
        lengths = ends - starts
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        row = torch.arange(capacity, device=self.device)
        if self.device.type == "cuda":
            from .packed_kernels import segment_layout

            request, local, valid = segment_layout(offsets, capacity)
            position = starts[request] + local
        else:
            request = torch.searchsorted(offsets[1:], row, right=True).clamp_max(starts.numel() - 1)
            valid = row < offsets[-1]
            position = starts[request] + row - offsets[request]
        return offsets, row, request, valid, position

    def _store(self, target, values, slots, layout):
        _, row, request, valid, position = layout
        address = torch.where(
            valid, slots[request] * self.ring + position.remainder(self.ring), self.capacity * self.ring + row
        )
        target.index_copy_(0, address, values)

    def __call__(self, spectra, slots, offsets, final):
        wave_start = self.ola.emitted[slots].clone()
        waves, _, _ = self.ola(spectra, slots, offsets, final)
        wave_end = self.ola.emitted[slots]
        total = self.ola.columns[slots] * self.ola.hop
        wave_capacity = spectra[0].shape[0] * self.ola.hop + slots.numel() * self.ola.lookahead
        mix_capacity = wave_capacity + slots.numel() * self.half_fir
        wave_layout = self._layout(wave_start, wave_end, wave_capacity)
        for target, values in zip(self.wave, waves, strict=True):
            self._store(target, values, slots, wave_layout)
        mix_start = self.mixed_end[slots]
        mix_end = torch.where(final, total, (wave_end - self.half_fir).clamp_min(0))
        mix_layout = self._layout(mix_start, mix_end, mix_capacity)
        _, _, request, valid, position = mix_layout
        if self.device.type == "cuda":
            from .packed_kernels import segmented_fir

            mixed = segmented_fir(self.wave, self.mixer, slots, request, valid, position, total, self.ring)
        else:
            taps = torch.arange(self.mixer.shape[-1], device=self.device)
            source = position[:, None] + taps - self.half_fir
            mask = valid[:, None] & (source >= 0) & (source < total[request, None])
            address = slots[request, None] * self.ring + source.remainder(self.ring)
            mixed = torch.zeros(mix_capacity, dtype=torch.float64, device=self.device)
            for branch in range(2):
                mixed = mixed + (torch.where(mask, self.wave[branch][address], 0.0) * self.mixer[branch]).sum(-1)
            # CPU reference retains the official FIR rounding boundary.
            mixed = mixed.float().double()
        self._store(self.mixed, mixed, slots, mix_layout)
        pcm_start = self.emitted[slots]
        stable_end = ((mix_end - self.width - self.orig) // self.orig + 1).clamp_min(0) * self.new
        final_end = (total * self.new + self.orig - 1) // self.orig
        pcm_end = torch.where(final, final_end, stable_end)
        pcm_capacity = math.ceil(mix_capacity * self.new / self.orig) + slots.numel() * self.new
        pcm_layout = self._layout(pcm_start, pcm_end, pcm_capacity)
        pcm_offsets, _, request, valid, position = pcm_layout
        if self.device.type == "cuda":
            from .packed_kernels import segmented_resample

            audio = segmented_resample(
                self.mixed,
                self.kernel,
                slots,
                request,
                valid,
                position,
                total,
                self.ring,
                self.orig,
                self.new,
                self.width,
            )
        else:
            taps = torch.arange(self.kernel.shape[-1], device=self.device)
            source = (position // self.new)[:, None] * self.orig - self.width + taps
            mask = valid[:, None] & (source >= 0) & (source < total[request, None])
            address = slots[request, None] * self.ring + source.remainder(self.ring)
            values = torch.where(mask, self.mixed[address], 0.0)
            audio = (values * self.kernel[position.remainder(self.new)]).sum(-1).float()
        self.mixed_end.index_copy_(0, slots, mix_end)
        self.emitted.index_copy_(0, slots, pcm_end)
        return audio, pcm_offsets, pcm_end - pcm_start


class ZACPackedCodec(nn.Module):
    """Complete pack→pack Codec shared by eager execution and graph capture.

    The manager owns slot initialization and output copies.
    """

    def __init__(self, network: ZACDecoderNetwork, capacity: int, max_chunk: int, token_capacity: int) -> None:
        super().__init__()
        self.network = network
        self.state = network.allocate_packed_state(capacity, max_chunk, token_capacity)
        self.scale = math.prod(stage.size for stage in network.stages if isinstance(stage, ZACPatch))
        self.synthesis = ZACPackedSynthesis(network, capacity, max_chunk * self.scale, token_capacity * self.scale)

    def forward(self, codes, slots, cu_seqlens, final):
        spectra = self.network.forward_packed(codes, slots, cu_seqlens, self.state)
        return self.synthesis(spectra, slots, cu_seqlens * self.scale, final)

    def reset(self, slots: torch.Tensor) -> None:
        """Recycle slots. KV is lazy: position zero only reads newly written KV."""
        self.state.frames.index_fill_(0, slots, 0)
        for history in self.state.semantic_history + self.state.head_history:
            history.index_fill_(0, slots, 0)
        synthesis, ola = self.synthesis, self.synthesis.ola
        for counter in (ola.columns, ola.emitted, synthesis.mixed_end, synthesis.emitted):
            counter.index_fill_(0, slots, 0)
        for tensor in ola.ola + ola.envelope:
            tensor[: ola.capacity * ola.ring].view(ola.capacity, ola.ring).index_fill_(0, slots, 0)
        for tensor in synthesis.wave + [synthesis.mixed]:
            tensor[: synthesis.capacity * synthesis.ring].view(synthesis.capacity, synthesis.ring).index_fill_(
                0, slots, 0
            )

    def output_end(self, frames: int, final: bool) -> int:
        """CPU-known PCM frontier; avoids reading device offsets to split output."""
        s = self.synthesis
        samples = frames * self.scale * s.ola.hop
        if final:
            return (samples * s.new + s.orig - 1) // s.orig
        mixed = max(0, samples - s.ola.lookahead - s.half_fir)
        return max(0, (mixed - s.width - s.orig) // s.orig + 1) * s.new


@dataclass(frozen=True)
class ZACCodecChunk:
    request_id: str
    chunk_seq: int
    frame_offset: int
    codes: torch.Tensor
    finished: bool = False
    skip_samples: int = 0


@dataclass(frozen=True)
class ZACPCMOutput:
    request_id: str
    chunk_seq: int
    audio: torch.Tensor
    finished: bool


@dataclass
class _Stream:
    slot: int
    chunk_seq: int = 0
    frames: int = 0
    skip_samples: int = 0
    emitted_samples: int = 0


class CodecManager:
    def __init__(
        self,
        network: ZACDecoderNetwork,
        capacity: int,
        *,
        graph_batch_sizes: tuple[int, ...] = (),
        graph_frame_sizes: tuple[int, ...] = (),
        max_chunk: int = 31,
    ) -> None:
        from .cuda_graph_wrapper import ZACCodecGraphWrapper

        if capacity < 1:
            raise ValueError("Codec request capacity must be positive")
        self.network = network
        self.device = next(network.parameters()).device
        self.max_chunk = max_chunk
        self.graph = ZACCodecGraphWrapper(
            network,
            capacity,
            max_chunk,
            [(b, b * t) for b in graph_batch_sizes for t in graph_frame_sizes],
        )
        self.state = self.graph.codec.state
        if graph_batch_sizes and self.device.type == "cuda":
            self.graph.capture()
        self._free = list(range(capacity - 1, -1, -1))
        self._streams: dict[str, _Stream] = {}
        self._finished: set[str] = set()

    def _allocate(self, chunk: ZACCodecChunk) -> _Stream:
        slot = self._free[-1]
        self.graph.codec.reset(index_to_device([slot], self.device))
        stream = _Stream(slot=slot, skip_samples=chunk.skip_samples)
        self._free.pop()
        self._streams[chunk.request_id] = stream
        return stream

    def release(self, request_id: str) -> None:
        """Cancellation/completion hook: no stale partial audio is flushed."""
        stream = self._streams.pop(request_id, None)
        if stream is not None:
            self._free.append(stream.slot)
        self._finished.discard(request_id)

    def _validate(self, chunks: Sequence[ZACCodecChunk]) -> None:
        ids = [chunk.request_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("A codec runner step must contain at most one chunk per request")
        new = 0
        for chunk in chunks:
            if chunk.request_id in self._finished:
                raise ValueError("Codec request already finished; completion cleanup is required before reuse")
            stream = self._streams.get(chunk.request_id)
            expected_seq, expected_frames = (0, 0) if stream is None else (stream.chunk_seq, stream.frames)
            if chunk.chunk_seq != expected_seq or chunk.frame_offset != expected_frames:
                raise ValueError("Out-of-order or missing ZAC codec chunk")
            if chunk.skip_samples < 0 or (stream is not None and chunk.skip_samples):
                raise ValueError("Codec context skip belongs only on the first chunk")
            codes = chunk.codes
            if codes.device.type != "cpu" or codes.dtype not in (torch.int32, torch.int64):
                raise ValueError("Codec input must be CPU-materialized integer RVQ codes")
            if codes.ndim != 2 or codes.shape[1] != 10:
                raise ValueError("Expected ZAC codes [frames, 10]")
            # Only CPU data is inspected. Device execution has no sync to read
            # code values, EOS, chunk position or valid lengths.
            if bool(((codes < -1) | (codes >= self.network.config.codebook_size)).any()) or bool(
                (codes[:, 0] < 0).any()
            ):
                raise ValueError("Invalid ZAC code index (only residual codebooks may be absent)")
            if codes.shape[0] == 0 and not chunk.finished:
                raise ValueError("Empty ZAC input must carry the final marker")
            new += stream is None
        if new > len(self._free):
            raise RuntimeError("No free ZAC decoder slots for the scheduled cohort")

    @torch.inference_mode()
    def decode(self, chunks: Sequence[ZACCodecChunk]) -> list[ZACPCMOutput]:
        self._validate(chunks)
        if any(len(chunk.codes) > self.max_chunk for chunk in chunks):
            raise ValueError(f"Packed Codec input exceeds configured {self.max_chunk} RVQ frames per chunk")
        if not chunks:
            return []
        succeeded = False
        try:
            for chunk in chunks:
                if chunk.request_id not in self._streams:
                    self._allocate(chunk)
            lengths = [len(chunk.codes) for chunk in chunks]
            codes = torch.cat([chunk.codes.to(torch.long) for chunk in chunks])
            if not codes.numel():
                codes = torch.zeros(1, 10, dtype=torch.long, device="cpu")
            slots = index_to_device([self._streams[c.request_id].slot for c in chunks], self.device)
            audio, _, _ = self.graph(
                to_device_nonblocking(codes, self.device),
                slots,
                index_to_device(lengths, self.device),
                to_device_nonblocking(torch.tensor([c.finished for c in chunks], device="cpu"), self.device),
            )
            counts = [
                self.graph.codec.output_end(self._streams[c.request_id].frames + n, c.finished)
                - self._streams[c.request_id].emitted_samples
                for c, n in zip(chunks, lengths, strict=True)
            ]
            # Graph outputs alias bucket storage. One owning copy survives later
            # replays and asynchronous PCM materialization for the whole cohort.
            owned = audio[: sum(counts)].clone()
            outputs = []
            offset = 0
            for chunk, n, count in zip(chunks, lengths, counts, strict=True):
                stream = self._streams[chunk.request_id]
                skip = min(count, stream.skip_samples)
                outputs.append(
                    ZACPCMOutput(
                        chunk.request_id, chunk.chunk_seq, owned[offset + skip : offset + count], chunk.finished
                    )
                )
                offset += count
                stream.skip_samples -= skip
                stream.frames += n
                stream.emitted_samples += count
                stream.chunk_seq += 1
                if chunk.finished:
                    self.release(chunk.request_id)
                    self._finished.add(chunk.request_id)
            succeeded = True
            return outputs
        finally:
            if not succeeded:
                # Reclaim every slot in a partially mutated cohort before retry.
                for chunk in chunks:
                    self.release(chunk.request_id)


if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    from .model_state import Zonos3CodecModelState


def _host_integer(value: object, field: str) -> int:
    if isinstance(value, torch.Tensor):
        if (
            value.device.type != "cpu"
            or value.numel() != 1
            or value.dtype not in (torch.int32, torch.int64, torch.bool)
        ):
            raise ValueError(f"Zonos3 {field} must be a CPU integer scalar")
        return int(value)
    if not isinstance(value, int):
        raise ValueError(f"Zonos3 {field} must be an integer")
    return value


class Zonos3CodecModel(nn.Module):
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    requires_request_ids = True
    requires_raw_input_tokens = True
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        variant = getattr(model_config.hf_config, "zonos3_decoder", "keep2_plain")
        self.files = Zonos3CodecFiles.for_variant(variant)
        self.root = self.files.resolve(
            model_config.model, vllm_config.load_config.download_dir, model_config.revision, with_weights=False
        )
        self.release_config = self.files.read_config(self.root)
        # The released codec stores FP32 weights; only transformer GEMMs use
        # BF16 autocast. Never inherit the AR stage's default parameter dtype.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            self.network = ZACDecoderNetwork(ZACDecoderConfig.from_release(self.release_config))
        finally:
            torch.set_default_dtype(previous_dtype)
        self.allow_patterns_overrides = [self.files.generator]
        self._manager: CodecManager | None = None
        self._weights_loaded = False
        self._internal_to_external: dict[str, str] = {}
        self._external_to_internal: dict[str, str] = {}

    def create_mrv2_model_state(
        self, vllm_config: VllmConfig, encoder_cache: EncoderCache | None, device: torch.device
    ) -> Zonos3CodecModelState:
        from .model_state import Zonos3CodecModelState

        return Zonos3CodecModelState(vllm_config, self, encoder_cache, device)

    def _codec_manager(self) -> CodecManager:
        if self._manager is None:
            capacity = self.vllm_config.scheduler_config.max_num_seqs
            connector = self.vllm_config.model_config.stage_connector_config
            extra = connector.get("extra", connector) if isinstance(connector, dict) else connector.extra
            chunk_size = extra.get("codec_chunk_frames", 15)
            ramp = extra.get("codec_chunk_ramp", (1, 2, 4, 8, 15))
            frames = tuple(sorted({1, chunk_size, *ramp}))
            if any(type(size) is not int or size < 1 for size in frames):
                raise ValueError("Zonos3 Codec chunk sizes must be positive integers")
            # The first transport chunk can include up to 30 silence-primer
            # frames. Ring capacity must cover them even without graph capture.
            max_chunk = max(chunk_size, *ramp, (ramp[0] if ramp else chunk_size) + 30)
            batches = tuple(1 << i for i in range((capacity - 1).bit_length())) + (capacity,)
            if self.vllm_config.model_config.enforce_eager:
                batches, frames = (), ()
            self._manager = CodecManager(
                self.network,
                capacity,
                graph_batch_sizes=batches,
                graph_frame_sizes=frames,
                max_chunk=max_chunk,
            )
        return self._manager

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: object = None) -> None:
        """Declare the native generation task, as other forward-only codecs do.

        OmniGenerationModelRunner consumes OmniOutput directly and never samples
        logits. The method exists for upstream architecture/task inspection.
        """
        return None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Native task inspection requires the embedding interface even for a
        # forward-only generation stage. Chunk placeholders have no learned
        # meaning; actual neural inputs come exclusively from codes.audio.
        width = self.vllm_config.model_config.get_inputs_embeds_size()
        return torch.zeros((input_ids.shape[0], width), dtype=torch.float32, device=input_ids.device)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        request_ids: list[str] | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        seq_token_counts: list[int] | None = None,
        sampling_metadata: object = None,
        logits_index: torch.Tensor | None = None,
        sampler: object = None,
    ) -> OmniOutput:
        del (
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            seq_token_counts,
            sampling_metadata,
            logits_index,
            sampler,
        )
        if not self._weights_loaded:
            raise RuntimeError("Zonos3 Codec release weights have not been loaded")
        infos = model_intermediate_buffer if model_intermediate_buffer is not None else runtime_additional_information
        if request_ids is None or infos is None or len(request_ids) != len(infos):
            raise ValueError("Zonos3 Codec needs one native request ID and payload per scheduled row")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Duplicate native codec request IDs")
        pending: list[ZACCodecChunk] = []
        indices: list[int] = []
        bindings: list[tuple[str, str]] = []
        for i, (internal_id, info) in enumerate(zip(request_ids, infos, strict=True)):
            if is_warmup_request(internal_id):
                continue
            meta = info.get("meta")
            if not isinstance(meta, Mapping) or not isinstance(meta.get("request_id"), str):
                raise ValueError("Zonos3 Codec payload is missing its external request identity")
            external_id = meta["request_id"]
            if (
                self._internal_to_external.get(internal_id, external_id) != external_id
                or self._external_to_internal.get(external_id, internal_id) != internal_id
            ):
                raise ValueError("Zonos3 Codec request identity changed without lifecycle cleanup")
            namespace = info.get("codes")
            codes = namespace.get("audio") if isinstance(namespace, Mapping) else None
            if not isinstance(codes, torch.Tensor):
                raise ValueError("Zonos3 Codec payload must include CPU codes.audio, including on final flush")
            finished = _host_integer(meta.get("finished"), "finished")
            if finished not in (0, 1):
                raise ValueError("Zonos3 Codec finished must be boolean")
            pending.append(
                ZACCodecChunk(
                    request_id=external_id,
                    chunk_seq=_host_integer(meta.get("chunk_seq"), "chunk_seq"),
                    frame_offset=_host_integer(meta.get("num_processed_tokens"), "num_processed_tokens"),
                    codes=codes,
                    finished=bool(finished),
                    skip_samples=_host_integer(meta.get("ref_context_size", 0), "ref_context_size") * 2940,
                )
            )
            indices.append(i)
            bindings.append((internal_id, external_id))
        device = next(self.network.parameters()).device
        audio = [torch.empty(0, device=device, dtype=torch.float32) for _ in request_ids]
        if pending:
            decoded = self._codec_manager().decode(pending)
            for i, output in zip(indices, decoded, strict=True):
                audio[i] = output.audio
            for internal_id, external_id in bindings:
                self._internal_to_external[internal_id] = external_id
                self._external_to_internal[external_id] = internal_id
        payload = cast(
            OmniPayload, {"model_outputs": audio, "sr": [torch.tensor(44100, dtype=torch.int32)] * len(audio)}
        )
        return OmniOutput(text_hidden_states=torch.empty((len(audio), 0), device=device), multimodal_outputs=payload)

    def on_requests_finished(self, finished_req_ids: set[str]) -> None:
        for internal_id in finished_req_ids:
            external_id = self._internal_to_external.pop(internal_id, None)
            if external_id is not None:
                self._external_to_internal.pop(external_id, None)
                if self._manager is not None:
                    self._manager.release(external_id)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        # Release .pt files contain nested state dictionaries rather than flat
        # HF tensors. Resolve their exact paths and load strictly once; do not
        # iterate the default loader's unrelated/nested tensor stream.
        del weights
        if self._weights_loaded:
            raise RuntimeError("Zonos3 Codec release weights are already loaded")
        config = self.vllm_config
        self.root = self.files.resolve(
            config.model_config.model, config.load_config.download_dir, config.model_config.revision, with_weights=True
        )
        self.files.load(self.root, self.network, self.release_config)
        if next(self.network.parameters()).device.type == "cuda":
            # Stage 1 has its own worker process. Set the reference's FP32
            # arithmetic policy once at startup, rather than racing global
            # backend switches across requests in the decode hot path.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self._weights_loaded = True
        self._codec_manager()
