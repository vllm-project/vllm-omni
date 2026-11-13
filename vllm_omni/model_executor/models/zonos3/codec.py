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
from collections import defaultdict
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
class ZACAttentionState:
    keys: torch.Tensor
    values: torch.Tensor

    def reset(self, slots: torch.Tensor) -> None:
        self.keys.index_fill_(0, slots, 0)
        self.values.index_fill_(0, slots, 0)


class ZACMossLayer(nn.Module):
    """One real B>1 neural forward over equal-length chunks at different positions.

    Past keys are right-aligned in a fixed ``context-1`` window. A row's
    absolute position masks uninitialized entries, including after slot reuse.
    Eager groups use exact lengths. Graph groups pass device valid lengths so
    padded frames never advance persistent history.
    """

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

    def allocate_state(self, capacity: int, dtype: torch.dtype, device: torch.device | str) -> ZACAttentionState:
        shape = (capacity, self.heads, self.context - 1, self.head_dim)
        return ZACAttentionState(
            torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device)
        )

    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
        state: ZACAttentionState,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        steps = torch.arange(t, device=x.device)
        frequency = torch.exp(
            torch.arange(self.head_dim // 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) * 2 / self.head_dim)
        )
        phase = (positions[:, None] + steps).float()[:, :, None] * frequency[None, None, :]
        cos, sin = phase.cos()[:, :, None], phase.sin()[:, :, None]

        def rope(z: torch.Tensor) -> torch.Tensor:
            real, imag = z.float().reshape(b, t, self.heads, self.head_dim // 2, 2).unbind(-1)
            return torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).flatten(-2).to(z.dtype)

        q = rope(q).transpose(1, 2)
        k = torch.cat((state.keys.index_select(0, slots).to(k.dtype), rope(k).transpose(1, 2)), dim=2)
        v = torch.cat((state.values.index_select(0, slots).to(v.dtype), v.transpose(1, 2)), dim=2)
        past = self.context - 1
        key_index = torch.arange(past + t, device=x.device)
        delta = (steps + past)[:, None] - key_index[None, :]
        valid = (delta >= 0) & (delta < self.context)
        initialized = key_index[None, :] >= (past - positions.clamp(max=past))[:, None]
        mask = valid[None, None, :, :] & initialized[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # The cache is bounded. Rows at different chunk indices share the GEMMs
        # but never their keys or conv histories. No padded token advances state.
        if past:
            if lengths is None:
                next_k, next_v = k[:, :, -past:], v[:, :, -past:]
            else:
                indices = lengths[:, None] + torch.arange(past, device=x.device)
                indices = indices[:, None, :, None].expand(b, self.heads, past, self.head_dim)
                next_k, next_v = k.gather(2, indices), v.gather(2, indices)
            state.keys.index_copy_(0, slots, next_k.to(state.keys.dtype))
            state.values.index_copy_(0, slots, next_v.to(state.values.dtype))
        x = x + self.attention_scale * self.attention_out(out.transpose(1, 2).reshape(b, t, d))
        return x + self.ffn_scale * self.ffn(self.norm2(x))


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

    def forward(
        self,
        x: torch.Tensor,
        slots: torch.Tensor,
        positions: torch.Tensor,
        states: list[ZACAttentionState],
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x.transpose(1, 2))
        for layer, state in zip(self.layers, states, strict=True):
            x = layer(x, slots, positions, state, lengths)
        return self.output_proj(x).transpose(1, 2)


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

    def forward(
        self, x: torch.Tensor, slots: torch.Tensor, history: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        joined = torch.cat((history.index_select(0, slots).to(x.dtype), x), -1)
        h = self.dwconv(joined).transpose(1, 2)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h))))
        if lengths is None:
            next_history = joined[..., -6:]
        else:
            indices = lengths[:, None, None] + torch.arange(6, device=x.device)
            next_history = joined.gather(2, indices.expand(-1, x.shape[1], -1))
        history.index_copy_(0, slots, next_history.to(history.dtype))
        return x + h.transpose(1, 2)


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
class ZACDecoderState:
    frames: torch.Tensor
    semantic_history: list[torch.Tensor]
    stage_history: list[list[ZACAttentionState]]
    head_history: list[torch.Tensor]

    def reset(self, slots: torch.Tensor) -> None:
        self.frames.index_fill_(0, slots, 0)
        for history in self.semantic_history + self.head_history:
            history.index_fill_(0, slots, 0)
        for stage in self.stage_history:
            for layer in stage:
                layer.reset(slots)


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

    def allocate_state(self, capacity: int) -> ZACDecoderState:
        parameter = next(self.parameters())
        device = parameter.device
        cache_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        return ZACDecoderState(
            frames=torch.zeros(capacity, dtype=torch.int64, device=device),
            semantic_history=[
                block.allocate_state(capacity, torch.float32, device) for block in self.semantic_head[:-1]
            ],
            stage_history=[
                [layer.allocate_state(capacity, cache_dtype, device) for layer in stage.layers]
                if isinstance(stage, ZACMossStage)
                else []
                for stage in self.stages
            ],
            head_history=[block.allocate_state(capacity, torch.float32, device) for block in self.head.blocks],
        )

    def forward(
        self, codes: torch.Tensor, slots: torch.Tensor, state: ZACDecoderState, lengths: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """One neural batch, [B, new_frames, 10] → two [B, head_frames, bins] branches.

        The manager validates codes on CPU and stages the batch once. Optional
        device lengths exclude RVQ padding from all KV/conv and position commits.
        """
        positions = state.frames.index_select(0, slots)
        device_type = codes.device.type
        with torch.autocast(device_type, enabled=False):
            x = self.quantizer.semantic_quantizer.quantizers[0](codes[:, :, 0])
            for block, history in zip(self.semantic_head[:-1], state.semantic_history, strict=True):
                x = block(x, slots, history, lengths)
            z = self.semantic_head[-1](x).float()
            for i, quantizer in enumerate(self.quantizer.quantizer.quantizers):
                code = codes[:, :, i + 1]
                contribution = quantizer(code.clamp(min=0))
                z = z + torch.where(code[:, None, :] >= 0, contribution, 0.0)
        x, scale = z, 1
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            for stage, history in zip(self.stages, state.stage_history, strict=True):
                if isinstance(stage, ZACPatch):
                    x = stage(x)
                    scale *= stage.size
                else:
                    x = stage(x, slots, positions * scale, history, None if lengths is None else lengths * scale)
        with torch.autocast(device_type, enabled=False):
            x = x.float()
            for block, history in zip(self.head.blocks, state.head_history, strict=True):
                x = block(x, slots, history, None if lengths is None else lengths * scale)
            x = self.head.norm(x.transpose(1, 2))
            output = [linear(x).float() for linear in self.head.outs]
        state.frames.index_copy_(0, slots, positions + (codes.shape[1] if lengths is None else lengths))
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


class ZACISTFTStream:
    def __init__(
        self,
        n_ffts: Sequence[int],
        hop: int,
        windows: Sequence[torch.Tensor],
        mixer: torch.Tensor,
        max_log_magnitude: float,
    ) -> None:
        if len(n_ffts) != 2 or len(windows) != 2:
            raise ValueError("ZAC-1 requires two iSTFT branches")
        if hop <= 0 or any(n % hop or n < 2 * hop for n in n_ffts):
            raise ValueError("iSTFT window sizes must be multiples of hop")
        if mixer.ndim != 2 or mixer.shape[0] != 2 or mixer.shape[1] % 2 != 1:
            raise ValueError("ZAC mixer must have two branches and an odd FIR kernel")
        self.n_ffts, self.hop = tuple(n_ffts), hop
        self.device = mixer.device
        self.windows = [w.to(device=self.device, dtype=torch.float64) for w in windows]
        if any(w.shape != (n,) for w, n in zip(self.windows, n_ffts, strict=True)):
            raise ValueError("iSTFT window shape mismatch")
        self.mixer = mixer.to(torch.float64)
        self.max_log_magnitude = max_log_magnitude
        self.kernel = mixer.shape[1]
        self.half_kernel = self.kernel // 2
        self.ola = [self._zeros(0), self._zeros(0)]
        self.envelope = [self._zeros(0), self._zeros(0)]
        self.ola_base = [0, 0]
        self.mix_input = [self._zeros(self.half_kernel), self._zeros(self.half_kernel)]
        self.mix_base = 0
        self.frames = 0
        self.done = False

    def _zeros(self, n: int) -> torch.Tensor:
        return torch.zeros(n, device=self.device, dtype=torch.float64)

    def _overlap_add(self, frames: torch.Tensor, n_fft: int) -> torch.Tensor:
        span = (frames.shape[0] - 1) * self.hop + n_fft
        return F.fold(
            frames.T.unsqueeze(0), output_size=(1, span), kernel_size=(1, n_fft), stride=(1, self.hop)
        ).reshape(span)

    def push(self, spectra: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.done:
            raise RuntimeError("iSTFT stream already flushed")
        if len(spectra) != 2:
            raise ValueError("Expected two spectral branches")
        cols = spectra[0].shape[0]
        for y, n in zip(spectra, self.n_ffts, strict=True):
            if y.ndim != 2 or y.shape != (cols, n + 2):
                raise ValueError("Expected aligned [frames, n_fft+2] spectral branches")
        if cols == 0:
            return self._zeros(0).float()
        for i, (y, n_fft) in enumerate(zip(spectra, self.n_ffts, strict=True)):
            y = y.to(device=self.device, dtype=torch.float64)
            bins = n_fft // 2 + 1
            magnitude = y[:, :bins].clamp(max=self.max_log_magnitude).exp()
            phase = y[:, bins:]
            imaginary = magnitude * phase.sin()
            imaginary[:, 0] = imaginary[:, -1] = 0
            window = self.windows[i]
            frames = torch.fft.irfft(torch.complex(magnitude * phase.cos(), imaginary), n_fft, dim=-1) * window
            add = self._overlap_add(frames, n_fft)
            envelope = self._overlap_add(window.square().expand(cols, n_fft), n_fft)
            offset = self.frames * self.hop - self.ola_base[i]
            needed = offset + add.shape[0]
            grow = max(0, needed - self.ola[i].shape[0])
            if grow:
                self.ola[i] = torch.cat((self.ola[i], self._zeros(grow)))
                self.envelope[i] = torch.cat((self.envelope[i], self._zeros(grow)))
            self.ola[i][offset:needed] += add
            self.envelope[i][offset:needed] += envelope
        self.frames += cols
        for i in range(2):
            self._finalize(i, self.frames * self.hop)
        return self._mix_emit()

    def _finalize(self, branch: int, end: int) -> None:
        n = end - self.ola_base[branch]
        if n <= 0:
            return
        skip = min(n, max(0, self.n_ffts[branch] // 2 - self.ola_base[branch]))
        samples = self.ola[branch][skip:n] / self.envelope[branch][skip:n]
        self.mix_input[branch] = torch.cat((self.mix_input[branch], samples))
        self.ola[branch] = self.ola[branch][n:]
        self.envelope[branch] = self.envelope[branch][n:]
        self.ola_base[branch] = end

    def _mix_emit(self) -> torch.Tensor:
        end = min(x.shape[0] for x in self.mix_input) + self.mix_base - self.kernel + 1
        if end <= self.mix_base:
            return self._zeros(0).float()
        take = end - self.mix_base + self.kernel - 1
        x = torch.stack([y[:take] for y in self.mix_input]).unsqueeze(0)
        output = F.conv1d(x, self.mixer.unsqueeze(0)).reshape(-1)
        drop = end - self.mix_base
        self.mix_input = [x[drop:] for x in self.mix_input]
        self.mix_base = end
        return output.float()

    def flush(self) -> torch.Tensor:
        if self.done:
            return self._zeros(0).float()
        self.done = True
        if self.frames == 0:
            return self._zeros(0).float()
        length = self.frames * self.hop
        for i in range(2):
            self._finalize(i, self.ola_base[i] + self.ola[i].shape[0])
            keep = self.half_kernel + length - self.mix_base
            x = self.mix_input[i]
            x = x[:keep] if x.shape[0] >= keep else torch.cat((x, self._zeros(keep - x.shape[0])))
            self.mix_input[i] = torch.cat((x, self._zeros(self.half_kernel)))
        return self._mix_emit()


class ZACResamplerStream:
    """Finite-lookahead stream using torchaudio's stored 44160→44100 kernel."""

    def __init__(self, kernel: torch.Tensor, orig: int, new: int, width: int) -> None:
        if orig <= 0 or new <= 0 or width < 0 or kernel.shape != (new, 2 * width + orig):
            raise ValueError("Invalid ZAC resampling kernel")
        self.orig, self.new, self.width = orig, new, width
        self.kernel_length = 2 * width + orig
        self.kernel = kernel.to(torch.float64).unsqueeze(1)
        self.buffer = kernel.new_zeros(0, dtype=torch.float64)
        self.base = 0
        self.n_input = 0
        self.next_block = 0
        self.done = False

    def push(self, x: torch.Tensor) -> torch.Tensor:
        if self.done:
            raise RuntimeError("Resampler already flushed")
        self.buffer = torch.cat((self.buffer, x.to(self.kernel).reshape(-1)))
        self.n_input += x.numel()
        return self._run(False)

    def flush(self) -> torch.Tensor:
        if self.done:
            return self.buffer.new_zeros(0, dtype=torch.float32)
        self.done = True
        return self._run(True)

    def _run(self, flush: bool) -> torch.Tensor:
        total = -(-self.new * self.n_input // self.orig)
        reach = self.n_input + self.width - self.kernel_length
        end = -(-total // self.new) if flush else (reach // self.orig + 1 if reach >= 0 else 0)
        output = self.buffer.new_zeros(0, dtype=torch.float32)
        if end > self.next_block:
            lo = self.next_block * self.orig - self.width
            hi = (end - 1) * self.orig - self.width + self.kernel_length
            a, b = max(lo, 0), min(hi, self.n_input)
            if b > a:
                segment = F.pad(self.buffer[a - self.base : b - self.base], (a - lo, hi - b))
            else:
                segment = self.buffer.new_zeros(hi - lo)
            y = F.conv1d(segment.view(1, 1, -1), self.kernel, stride=self.orig)[0].T.reshape(-1)
            if flush:
                y = y[: total - self.next_block * self.new]
            output = y.float()
            self.next_block = end
        keep_from = min(self.n_input, max(0, self.next_block * self.orig - self.width))
        if keep_from > self.base:
            self.buffer = self.buffer[keep_from - self.base :]
            self.base = keep_from
        return output


class ZACBatchedSynthesis:
    """Fixed per-slot spectral history and GPU sample cursors for graph replay.

    Reconstruct only the finite overlap needed by the next resampling blocks.
    Centered iSTFT, FIR zero padding and resampler final boundaries are expressed
    in absolute sample coordinates. Padded spectra never contribute an envelope
    or waveform sample. Returned audio has static width and a device valid count.
    """

    def __init__(self, network: ZACDecoderNetwork, capacity: int) -> None:
        cfg, head, resampler = network.config, network.head, network.to_external
        self.device = next(network.parameters()).device
        self.n_ffts, self.hop = cfg.n_ffts, cfg.hop
        self.scale = math.prod(stage.size for stage in network.stages if isinstance(stage, ZACPatch))
        self.orig, self.new = resampler.orig_freq // resampler.gcd, resampler.new_freq // resampler.gcd
        self.width = resampler.width
        self.windows = [getattr(head, f"window_{i}").double() for i in range(2)]
        self.mixer = head.mixer.weight.detach().double()
        self.kernel = resampler.kernel.detach().double()
        self.half_fir = self.mixer.shape[-1] // 2
        self.half_fft = max(self.n_ffts) // 2
        self.max_log_magnitude = head.max_log_magnitude
        # Include iSTFT overlap, FIR context, resampler lookahead and the
        # un-emitted polyphase block. This bound is independent of audio length.
        span = max(self.n_ffts) + self.half_fft + 2 * self.half_fir + self.orig + 2 * self.width
        self.history_columns = math.ceil(span / self.hop) + 2
        self.columns = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.emitted = torch.zeros_like(self.columns)
        self.history = [
            torch.zeros(capacity, self.history_columns, n + 2, device=self.device, dtype=torch.float32)
            for n in self.n_ffts
        ]

    def reset(self, slots: torch.Tensor) -> None:
        """Clear recycled or capture-only slots without touching other streams."""
        self.columns.index_fill_(0, slots, 0)
        self.emitted.index_fill_(0, slots, 0)
        for history in self.history:
            history.index_fill_(0, slots, 0)

    def output_end(self, code_frames: int, final: bool) -> int:
        """Host-side coverage calculation, using only CPU request progress."""
        samples = code_frames * self.scale * self.hop
        if final:
            return (samples * self.new + self.orig - 1) // self.orig
        mixed = max(0, samples - self.half_fft - self.half_fir)
        return max(0, (mixed - self.width - self.orig) // self.orig + 1) * self.new

    def _fold(self, frames: torch.Tensor) -> torch.Tensor:
        batch, cols, n_fft = frames.shape
        span = (cols - 1) * self.hop + n_fft
        return F.fold(frames.transpose(1, 2), (1, span), (1, n_fft), stride=(1, self.hop)).reshape(batch, span)

    def __call__(
        self, spectra: list[torch.Tensor], slots: torch.Tensor, lengths: torch.Tensor, final: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, padded_cols, _ = spectra[0].shape
        old = self.columns.index_select(0, slots)
        count = lengths * self.scale
        total = (old + count) * self.hop
        emitted = self.emitted.index_select(0, slots)
        stable = (total - self.half_fft - self.half_fir).clamp_min(0)
        regular_end = ((stable - self.width - self.orig) // self.orig + 1).clamp_min(0) * self.new
        final_end = (total * self.new + self.orig - 1) // self.orig
        end = torch.where(final, final_end, regular_end)
        # Static upper bound includes a final tail, even for zero valid frames.
        blocks = (
            math.ceil((padded_cols * self.hop + self.half_fft + self.half_fir + 2 * self.orig + self.width) / self.orig)
            + 1
        )
        mixed_length = (blocks - 1) * self.orig + self.kernel.shape[-1]
        wave = (emitted // self.new)[:, None] * self.orig - self.width - self.half_fir
        wave = wave + torch.arange(mixed_length + 2 * self.half_fir, device=self.device)
        raw_base = (old - self.history_columns) * self.hop
        frame_indices = torch.arange(self.history_columns + padded_cols, device=self.device)
        valid_frames = (frame_indices[None] >= (self.history_columns - old)[:, None]) & (
            frame_indices[None] < (self.history_columns + count)[:, None]
        )
        aligned = []
        for i, (y, n_fft, window) in enumerate(zip(spectra, self.n_ffts, self.windows, strict=True)):
            joined = torch.cat((self.history[i].index_select(0, slots), y), 1)
            bins = n_fft // 2 + 1
            log_mag, phase = joined.double().split(bins, -1)
            magnitude = log_mag.clamp(max=self.max_log_magnitude).exp()
            imaginary = magnitude * phase.sin()
            imaginary[..., 0] = imaginary[..., -1] = 0
            frames = torch.fft.irfft(torch.complex(magnitude * phase.cos(), imaginary), n_fft, dim=-1) * window
            ola = self._fold(torch.where(valid_frames[..., None], frames, 0.0))
            envelope = self._fold(window.square()[None, None] * valid_frames[..., None])
            index = wave + n_fft // 2 - raw_base[:, None]
            safe = index.clamp(0, ola.shape[-1] - 1)
            denominator = envelope.gather(1, safe)
            values = ola.gather(1, safe) / denominator.clamp_min(torch.finfo(torch.float64).tiny)
            valid = (wave >= 0) & (wave < total[:, None]) & (index >= 0) & (index < ola.shape[-1])
            aligned.append(torch.where(valid & (denominator > 0), values, 0.0))
            keep = count[:, None, None] + torch.arange(self.history_columns, device=self.device)[None, :, None]
            self.history[i].index_copy_(0, slots, joined.gather(1, keep.expand(batch, -1, n_fft + 2)))
        # Match the reference rounding boundary between FIR and resampling.
        mixed = F.conv1d(torch.stack(aligned, 1), self.mixer).float().double()
        mixed_positions = wave[:, :mixed_length] + self.half_fir
        mixed = torch.where(((mixed_positions >= 0) & (mixed_positions < total[:, None]))[:, None], mixed, 0.0)
        audio = F.conv1d(mixed, self.kernel, stride=self.orig).transpose(1, 2).reshape(batch, -1).float()
        indices = (emitted // self.new)[:, None] * self.new + torch.arange(audio.shape[1], device=self.device)
        audio = torch.where((indices >= emitted[:, None]) & (indices < end[:, None]), audio, 0.0)
        self.columns.index_copy_(0, slots, old + count)
        self.emitted.index_copy_(0, slots, end)
        return audio, end - emitted


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
    head: ZACISTFTStream
    resampler: ZACResamplerStream
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
    ) -> None:
        from .cuda_graph_wrapper import ZACCodecGraphWrapper

        if capacity < 1:
            raise ValueError("Codec request capacity must be positive")
        self.network = network
        scratch = max(graph_batch_sizes, default=0)
        self.state = network.allocate_state(capacity + scratch)
        self.device = next(network.parameters()).device
        self.graph = (
            ZACCodecGraphWrapper(network, self.state, capacity, graph_batch_sizes, graph_frame_sizes)
            if graph_batch_sizes or graph_frame_sizes
            else None
        )
        if self.graph is not None and self.device.type == "cuda":
            self.graph.capture()
        self._free = list(range(capacity - 1, -1, -1))
        self._streams: dict[str, _Stream] = {}
        self._finished: set[str] = set()

    def _allocate(self, chunk: ZACCodecChunk) -> _Stream:
        slot = self._free[-1]
        cfg = self.network.config
        head = self.network.head
        resampler = self.network.to_external
        slots = index_to_device([slot], self.device)
        self.state.reset(slots)
        if self.graph is not None:
            self.graph.reset(slots)
        stream = _Stream(
            slot=slot,
            head=ZACISTFTStream(
                cfg.n_ffts,
                cfg.hop,
                [getattr(head, f"window_{i}") for i in range(2)],
                head.mixer.weight[0].detach(),
                head.max_log_magnitude,
            ),
            resampler=ZACResamplerStream(
                resampler.kernel[:, 0].detach(),
                resampler.orig_freq // resampler.gcd,
                resampler.new_freq // resampler.gcd,
                resampler.width,
            ),
            skip_samples=chunk.skip_samples,
        )
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
        outputs: list[ZACPCMOutput | None] = [None] * len(chunks)
        groups: dict[int, list[int]] = defaultdict(list)
        succeeded = False
        try:
            for i, chunk in enumerate(chunks):
                if chunk.request_id not in self._streams:
                    self._allocate(chunk)
                frames = chunk.codes.shape[0]
                groups[self.graph.frame_bucket(frames) if self.graph is not None else frames].append(i)
            for length, indices in groups.items():
                streams = [self._streams[chunks[i].request_id] for i in indices]
                if self.graph is not None:
                    batch = torch.zeros(len(indices), max(1, length), 10, dtype=torch.long, device="cpu")
                    actual_lengths = [chunks[i].codes.shape[0] for i in indices]
                    for row, i in enumerate(indices):
                        batch[row, : actual_lengths[row]] = chunks[i].codes
                    slots = index_to_device([stream.slot for stream in streams], self.device)
                    lengths = index_to_device(actual_lengths, self.device)
                    final = to_device_nonblocking(
                        torch.tensor([chunks[i].finished for i in indices], dtype=torch.bool, device="cpu"), self.device
                    )
                    audio, _ = self.graph(to_device_nonblocking(batch, self.device), slots, lengths, final)
                    for row, (i, stream) in enumerate(zip(indices, streams, strict=True)):
                        chunk = chunks[i]
                        valid = actual_lengths[row]
                        end = self.graph.synthesis.output_end(stream.frames + valid, chunk.finished)
                        take = end - stream.emitted_samples
                        skip = min(take, stream.skip_samples)
                        stream.skip_samples -= skip
                        stream.frames += valid
                        stream.chunk_seq += 1
                        stream.emitted_samples = end
                        # A later replay overwrites bucket storage; published
                        # audio must own its bytes through async materialization.
                        outputs[i] = ZACPCMOutput(
                            chunk.request_id, chunk.chunk_seq, audio[row, skip:take].clone(), chunk.finished
                        )
                        if chunk.finished:
                            self.release(chunk.request_id)
                            self._finished.add(chunk.request_id)
                    continue
                spectra = None
                if length:
                    batch = torch.stack([chunks[i].codes.to(torch.int64) for i in indices])
                    slots = index_to_device([stream.slot for stream in streams], self.device)
                    # One real batched neural network forward per compatible
                    # chunk shape. Different cache positions remain per slot.
                    spectra = self.network(to_device_nonblocking(batch, self.device), slots, self.state)
                for row, (i, stream) in enumerate(zip(indices, streams, strict=True)):
                    chunk = chunks[i]
                    parts = []
                    if spectra is not None:
                        parts.append(stream.resampler.push(stream.head.push([branch[row] for branch in spectra])))
                    if chunk.finished:
                        parts.extend((stream.resampler.push(stream.head.flush()), stream.resampler.flush()))
                    audio = torch.cat(parts) if parts else torch.empty(0, device=self.device, dtype=torch.float32)
                    skip = min(audio.numel(), stream.skip_samples)
                    stream.skip_samples -= skip
                    stream.frames += length
                    stream.chunk_seq += 1
                    outputs[i] = ZACPCMOutput(chunk.request_id, chunk.chunk_seq, audio[skip:], chunk.finished)
                    if chunk.finished:
                        self.release(chunk.request_id)
                        self._finished.add(chunk.request_id)
            succeeded = True
        finally:
            if not succeeded:
                # A partially executed cohort cannot be retried against mutated
                # recurrent state. Propagate the error and reclaim its slots.
                for chunk in chunks:
                    self.release(chunk.request_id)
        result = []
        for output in outputs:
            assert output is not None
            result.append(output)
        return result


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
            extra = self.vllm_config.additional_config or {}
            enabled = extra.get("zonos3_codec_cuda_graph", False)
            if not isinstance(enabled, bool):
                raise ValueError("zonos3_codec_cuda_graph must be boolean")
            capacity = self.vllm_config.scheduler_config.max_num_seqs
            batches = tuple(extra.get("zonos3_codec_graph_batch_sizes", (1, 2, 4, 8, 16, 32))) if enabled else ()
            frames = tuple(extra.get("zonos3_codec_graph_frame_sizes", (1, 2, 4, 8, 15))) if enabled else ()
            if enabled:
                if any(type(size) is not int or size < 1 for size in (*batches, *frames)):
                    raise ValueError("Zonos3 Codec graph bucket sizes must be positive integers")
                batches = tuple(size for size in batches if size <= capacity)
                if not batches or max(batches) != capacity:
                    batches = (*batches, capacity)
            self._manager = CodecManager(self.network, capacity, graph_batch_sizes=batches, graph_frame_sizes=frames)
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
        terminal_indices: list[int] = []
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
            if meta.get("aborted") is True:
                if self._manager is not None:
                    self._manager.release(external_id)
                self._internal_to_external.pop(internal_id, None)
                self._external_to_internal.pop(external_id, None)
                continue
            if meta.get("transport_terminal") is True and not _host_integer(meta.get("finished", False), "finished"):
                if self._manager is not None:
                    self._manager.release(external_id)
                self._internal_to_external.pop(internal_id, None)
                self._external_to_internal.pop(external_id, None)
                terminal_indices.append(i)
                continue
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
        for i in terminal_indices:
            audio[i] = torch.empty(0, device=device, dtype=torch.float32)
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
