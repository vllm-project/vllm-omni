# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VDN-H3: MiniMax-H3 with Video DeltaNet hybrid attention.

OpenVDN (https://github.com/OpenVDN/vdn-minimax-h3) converts the dense
self-attention of all 50 DiT blocks into

    out = out_proj(softmax_gate(x) * window_softmax(q, k, v))
        + to_out_linear(linear_branch(x, q_raw, k_raw, v))      # video rows only

``window_softmax`` is the ``VDNH3_ATTN`` attention backend. This module holds
the learned rest -- the per-head softmax gate, the linear branch, its output
projection -- and the checkpoint that carries them.

The linear branch summarizes, for every video frame ``t``, everything outside
its softmax window ``[lo, hi]``:

    features   k, v: depthwise 5x5 spatial + 5-tap temporal conv; SiLU;
               L2 norm on q, k; no RoPE (it reads the raw projections)
    statistics A_t = K^T diag(beta) K, B_t = V^T diag(beta) K        (fp32)
    scans      S_t = (S_{t-1} diag(alpha_t) + B_t)(I + A_t)^-1, forward and
               reverse over frames, both starting from half the prompt's state
    gather     prefix[lo-1] + suffix[hi+1], each decayed to t by prod alpha
    readout    RMSNorm(q_t S^T) * sigmoid(output_gate(x))

A checkpoint is an exploded directory over the base H3 FL2VA transformer
(``model_spec.json``, ``metadata.json``, ``linear_branch/``, ``adapters/*/``).
The served checkpoint is ``stage-dmd-step-250``, an 8-step DMD student. As
with FastH3 the LoRAs are fused into the native weights while they stream in,
and the branch tensors follow the base stream.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader

from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNLayout
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy
from vllm_omni.errors import OmniClientError
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.repo_utils import hf_api

from .fasth3 import _resolve_dit_attention_backend, _resolve_native_target

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from .minimax_h3_transformer import MiniMaxH3DiTModel

logger = init_logger(__name__)

_BF16 = torch.bfloat16
# Each directional scan starts from this fraction of the prompt state; baked
# into the trained checkpoints.
TEXT_STATE_SCALE = 0.5
_CONV_TAPS = 5
_BRANCH_KEY = re.compile(r"^transformer_blocks\.(\d+)\.attn\.((?:linear_attention|softmax_gate|to_out_linear)\..+)$")
_LORA_KEY = re.compile(r"^(.+)\.lora_([AB])\.[^.]+\.weight$")
VDN_TASKS = frozenset({"t2va", "fl2va"})
# The served checkpoint's directory inside the OpenVDN release.
VDN_CHECKPOINT = "stage-dmd-step-250"


@dataclass(frozen=True)
class VDNConfig:
    chunk: int
    radius: int
    anchor_frames: str
    enable_softmax_gate: bool
    enable_text_state: bool
    bridge: str
    short_conv: tuple[str, ...]

    @classmethod
    def from_transform(cls, config: Mapping[str, Any], *, head_dim: int) -> VDNConfig:
        """Read the ``hybrid_attention`` transform config (OpenVDN spec v2)."""
        soft, lin = config["softmax_attention"], config["linear_attention"]
        # The released checkpoints use the exact solve over an fp32 A at the
        # attention head size.
        for key, expected in (("delta_rule", "vdn_solve"), ("a_fp32", True), ("linear_head_dim", head_dim)):
            if lin.get(key, expected) != expected:
                raise ValueError(f"unsupported VDN linear_attention.{key}={lin.get(key)!r}, expected {expected!r}")
        short_conv = tuple((lin.get("short_conv") or {}).get("targets", ()))
        if not set(short_conv) <= {"q", "k", "v"} or len(set(short_conv)) != len(short_conv):
            raise ValueError(f"invalid VDN short_conv targets {short_conv!r}")
        bridge = lin.get("bridge", "alpha")
        if bridge not in ("alpha", "none"):
            raise ValueError(f"unsupported VDN bridge {bridge!r}")
        return cls(
            chunk=int(soft.get("chunk", 0)),
            radius=int(soft["radius"]),
            anchor_frames=str(config.get("anchor_frames", "none")),
            enable_softmax_gate=bool(config.get("enable_softmax_gate", True)),
            enable_text_state=bool(lin.get("enable_text_state", False)),
            bridge=bridge,
            short_conv=short_conv,
        )

    def window(self, *, used: int, text_len: int, video_start: int, grid: tuple[int, int, int]) -> VDNLayout:
        return VDNLayout(
            used=used,
            text_len=text_len,
            video_start=video_start,
            num_frames=grid[0],
            frame_height=grid[1],
            frame_width=grid[2],
            chunk=self.chunk,
            radius=self.radius,
            anchor_frames=self.anchor_frames,
        )


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------


def _lora_scale(config: Mapping[str, Any], module: str, rank: int) -> float:
    alpha = (config.get("alpha_pattern") or {}).get(module, config["alpha"])
    return float(alpha) / rank


def _checkpoint_dir(path: str, *, from_hub: bool) -> Path | None:
    """The checkpoint directory ``--lora-path`` names.

    ``path`` is that directory, a local copy of the OpenVDN release, or (with
    ``from_hub``) the release's Hub repository, of which only the checkpoint
    directory is downloaded.
    """
    local = Path(path).expanduser()
    if not local.is_dir():
        if not from_hub:
            return None
        prefetch_subfolders(path, [VDN_CHECKPOINT], include_root_metadata=False)
        patterns = [f"{VDN_CHECKPOINT}/*", f"{VDN_CHECKPOINT}/**"]
        local = Path(hf_api().snapshot_download(repo_id=path, allow_patterns=patterns, local_files_only=True))
    nested = local / VDN_CHECKPOINT
    return nested if (nested / "model_spec.json").is_file() else local


class VDNCheckpoint:
    """An exploded OpenVDN ``weights`` artifact applied over base H3 FL2VA."""

    def __init__(self, root: Path, spec: Mapping[str, Any], metadata: Mapping[str, Any], *, head_dim: int) -> None:
        self.root = root
        self.metadata = dict(metadata.get("metadata") or {})
        if self.metadata.get("turbo_num_steps") is None:
            raise ValueError(f"{root}: VDN-H3 serving takes the 8-step DMD checkpoint ({VDN_CHECKPOINT})")
        self.num_inference_steps = int(self.metadata["turbo_num_steps"])
        self.config = VDNConfig.from_transform(spec["transforms"][0]["config"], head_dim=head_dim)
        self._head_dim = head_dim
        # native parameter -> [(layout, lora_A, lora_B, scale)], in merge order.
        self._loras: dict[str, list[tuple[str, torch.Tensor, torch.Tensor, float]]] = {}
        adapters = root / "adapters"
        for adapter in sorted(p for p in adapters.iterdir() if p.is_dir()) if adapters.is_dir() else ():
            # Current releases name the file adapter_spec.json, earlier ones adapter_config.json.
            spec_file = adapter / "adapter_spec.json"
            if not spec_file.is_file():
                spec_file = adapter / "adapter_config.json"
            config = json.loads(spec_file.read_text(encoding="utf-8"))["config"]
            factors: dict[str, dict[str, torch.Tensor]] = {}
            with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as f:
                for name in f.keys():
                    match = _LORA_KEY.match(name)
                    if match is None:
                        raise ValueError(f"{adapter}: not a LoRA tensor: {name}")
                    factors.setdefault(match.group(1), {})[match.group(2)] = f.get_tensor(name)
            for module, pair in factors.items():
                # A DiT block's softmax projections sit one level down (attn.orig).
                target = _resolve_native_target(module.replace(".attn.orig.", ".attn."))
                if target is None or set(pair) != {"A", "B"}:
                    raise ValueError(f"{adapter}: cannot place LoRA {module} on native H3")
                scale = _lora_scale(config, module, pair["A"].shape[0])
                self._loras.setdefault(f"{target[0]}.weight", []).append((target[1], pair["A"], pair["B"], scale))
        self._branch_file = root / "linear_branch" / "model.safetensors"
        self._fused: set[str] = set()
        self._branch_names: set[str] = set()

    @classmethod
    def from_path(cls, path: str | Path, *, head_dim: int) -> VDNCheckpoint | None:
        """The checkpoint at ``path``, or None when it is not a VDN artifact."""
        root = Path(path)
        spec_path = root / "model_spec.json"
        if not spec_path.is_file():
            return None
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        transforms = spec.get("transforms") or []
        if not any(t.get("type") == "hybrid_attention" for t in transforms):
            return None
        if [(t.get("type"), t.get("version")) for t in transforms] != [("hybrid_attention", 2)]:
            raise ValueError(f"{root}: unsupported VDN transforms {transforms!r}")
        if spec.get("base", {}).get("class_name") != "MiniMaxH3Transformer3DModel":
            raise ValueError(f"{root}: VDN base is not MiniMax-H3: {spec.get('base')!r}")
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("kind") != "weights":
            raise ValueError(f"{root}: not a VDN weights artifact")
        return cls(root, spec, metadata, head_dim=head_dim)

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig, transformer: MiniMaxH3DiTModel) -> VDNCheckpoint | None:
        """Claim ``--lora-path`` when it names a VDN checkpoint.

        Only a ``VDNH3_ATTN`` deployment resolves a Hub repository id, so another
        adapter's id is never downloaded here.
        """
        path = getattr(od_config, "lora_path", None)
        if isinstance(path, (list, tuple)):
            path = path[0] if len(path) == 1 else None
        vdn_backend = _resolve_dit_attention_backend(od_config) == "VDNH3_ATTN"
        root = _checkpoint_dir(path, from_hub=vdn_backend) if path else None
        checkpoint = cls.from_path(root, head_dim=transformer.arch.attention_head_dim) if root is not None else None
        if checkpoint is None and vdn_backend:
            raise ValueError(
                f"VDNH3_ATTN runs VDN-H3 checkpoints: pass OpenVDN/vdn-minimax-h3 or its {VDN_CHECKPOINT} "
                "directory with --lora-path"
            )
        return checkpoint

    def _fuse(self, name: str, weight: torch.Tensor) -> torch.Tensor:
        loras = self._loras.get(name)
        if loras is None:
            return weight
        self._fused.add(name)
        device = weight.device if weight.device.type != "cpu" else current_omni_platform.get_torch_device()
        weight = weight.to(device, copy=True)
        for layout, a, b, scale in loras:
            delta = (b.to(device).float() @ a.to(device).float()) * scale
            if layout in ("q", "k", "v"):
                # H3 stores QKV grouped per head: [heads, (q, k, v), head_dim, in].
                grouped = weight.view(-1, 3, self._head_dim, weight.shape[-1])
                target, delta = grouped[:, "qkv".index(layout)], delta.view(-1, self._head_dim, weight.shape[-1])
            elif layout == "swap_halves":
                # Native fc1 is [gate; up], the diffusers projection [up; gate].
                target, delta = weight, torch.cat(delta.chunk(2)[::-1])
            else:
                target = weight
            # One bf16 rounding per adapter, as OpenVDN merges them.
            target.add_(delta.to(weight.dtype))
        return weight

    def apply(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        """Fuse the LoRAs into the native H3 stream, then append the branch."""
        for name, weight in weights:
            yield name, self._fuse(name, weight)
        with safe_open(self._branch_file, framework="pt", device="cpu") as f:
            for name in f.keys():
                match = _BRANCH_KEY.match(name)
                if match is None:
                    raise ValueError(f"{self._branch_file}: unexpected tensor {name}")
                native = f"blocks.{match.group(1)}.attn.vdn.{match.group(2)}"
                self._branch_names.add(native)
                yield native, f.get_tensor(name)

    def validate(self, loaded: set[str], expected_branch: set[str]) -> None:
        """Every LoRA met its parameter and every branch tensor has a home."""
        if missing := sorted(set(self._loras) - self._fused):
            raise ValueError(f"VDN LoRAs target parameters the checkpoint never provided: {missing[:5]}")
        if self._branch_names != expected_branch or not expected_branch <= loaded:
            raise ValueError(
                "VDN linear-branch tensors do not match the model: "
                f"missing={sorted(expected_branch - self._branch_names)[:5]}, "
                f"unexpected={sorted(self._branch_names - expected_branch)[:5]}"
            )
        logger.info(
            "VDN-H3 %s: fused %d LoRA targets, loaded %d branch tensors",
            self.root.name,
            len(self._fused),
            len(self._branch_names),
        )
        self._loras.clear()

    def check_serving_contract(self, *, partition: str, od_config: OmniDiffusionConfig) -> None:
        if partition != "fl2va":
            raise ValueError("VDN-H3 is trained on the FL2VA partition; serve it with --task-type fl2va")
        backend = _resolve_dit_attention_backend(od_config)
        if backend != "VDNH3_ATTN":
            raise ValueError(
                "VDN-H3 needs --diffusion-attention-backend VDNH3_ATTN; any other backend would skip the "
                f"frame window and run the hybrid weights as dense attention (got {backend or 'default'})"
            )
        parallel = od_config.parallel_config
        if any(int(getattr(parallel, key, 1) or 1) != 1 for key in ("ulysses_degree", "ring_degree")):
            raise ValueError("VDN-H3 does not support sequence parallelism; use --tensor-parallel-size")
        if resolve_offload_strategy(od_config) is OffloadStrategy.DISTRIBUTED_LAYER_WISE:
            # Its host-weight plan installs the DiT without load_weights(), where the fusion lives.
            raise ValueError("VDN-H3 is fused while the weights stream in; use a non-distributed offload mode")

    def check_request(self, sampling: OmniDiffusionSamplingParams, task: str) -> None:
        if task not in VDN_TASKS:
            raise OmniClientError(f"VDN-H3 serves {sorted(VDN_TASKS)}, got task={task!r}")
        if sampling.lora_request is not None:
            raise OmniClientError("VDN-H3 fuses its adapters at startup; per-request LoRA is unavailable")
        steps = self.num_inference_steps
        if int(sampling.num_inference_steps or 0) != steps:
            raise OmniClientError(f"this VDN-H3 checkpoint is a {steps}-step student; set num_inference_steps={steps}")
        extra = sampling.extra_args or {}
        for key, meta_key in (("flow_shift", "video_shift"), ("audio_flow_shift", "audio_shift")):
            expected = self.metadata.get(meta_key)
            if expected is not None and key in extra and float(extra[key]) != float(expected):
                raise OmniClientError(f"this VDN-H3 checkpoint was distilled at {key}={expected:g}")


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------


def _column(in_features: int, out_features: int, *, bias: bool, prefix: str) -> ColumnParallelLinear:
    """Head-major outputs: tensor parallelism takes this rank's heads."""
    return ColumnParallelLinear(
        in_features, out_features, bias=bias, gather_output=False, params_dtype=_BF16, quant_config=None, prefix=prefix
    )


def _replicated(in_features: int, out_features: int, *, prefix: str) -> ReplicatedLinear:
    return ReplicatedLinear(in_features, out_features, bias=False, params_dtype=_BF16, quant_config=None, prefix=prefix)


class _Weight(nn.Module):
    """A plain weight; a ``sharded`` one is head-major and TP keeps this rank's heads."""

    def __init__(self, *shape: int, sharded: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=_BF16), requires_grad=False)
        if sharded:
            self.weight.weight_loader = sharded_weight_loader(0)


class _FrameDecay(nn.Module):
    """alpha_t = exp(-exp(A_log) * softplus(up(down(frame_mean_t)) + dt_bias)), in fp32."""

    def __init__(self, hidden: int, heads: int, local_heads: int, head_dim: int, *, prefix: str) -> None:
        super().__init__()
        self.down = _replicated(hidden, head_dim, prefix=f"{prefix}.down")
        self.up = _column(head_dim, heads * head_dim, bias=False, prefix=f"{prefix}.up")
        self.A_log = nn.Parameter(torch.empty(local_heads, dtype=_BF16), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.empty(local_heads * head_dim, dtype=_BF16), requires_grad=False)
        for param in (self.A_log, self.dt_bias):
            param.weight_loader = sharded_weight_loader(0)

    def forward(self, frame_mean: torch.Tensor) -> torch.Tensor:
        """frame_mean [F, hidden] fp32 -> alpha [F, heads, head_dim] fp32."""
        # fp32 weights too: the scans multiply alpha across ~100 frames.
        delta = F.linear(F.linear(frame_mean, self.down.weight.float()), self.up.weight.float())
        delta = (delta + self.dt_bias.float()).view(frame_mean.shape[0], self.A_log.shape[0], -1)
        return torch.exp(-torch.exp(self.A_log.float()).unsqueeze(-1) * F.softplus(delta))


class _OutputGate(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, *, prefix: str) -> None:
        super().__init__()
        self.down = _replicated(hidden, head_dim, prefix=f"{prefix}.down")
        self.up = _column(head_dim, heads * head_dim, bias=True, prefix=f"{prefix}.up")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.up(self.down(x)[0])[0])


class _SoftmaxGate(nn.Module):
    def __init__(self, hidden: int, heads: int, *, prefix: str) -> None:
        super().__init__()
        self.up = _column(hidden, heads, bias=True, prefix=f"{prefix}.up")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.up(x)[0])


class _ShortConv(nn.Module):
    """Separable depthwise conv on head-major channels: 5x5 within each frame,
    then 5 zero-padded taps across frames."""

    def __init__(self, channels: int, targets: tuple[str, ...]) -> None:
        super().__init__()
        self.targets = targets
        for name in targets:
            setattr(self, f"{name}_sp", _Weight(channels, 1, _CONV_TAPS, _CONV_TAPS))
            setattr(self, f"{name}_tm", _Weight(channels, 1, _CONV_TAPS))

    def forward(self, proj: str, tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """tokens [F*S, heads, head_dim] -> the same shape in fp32."""
        frames, height, width = grid
        channels = tokens.shape[1] * tokens.shape[2]
        # [F*S, C] read as [F, H, W, C] is the channels_last layout of [F, C, H, W].
        volume = tokens.reshape(frames, height, width, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(volume, getattr(self, f"{proj}_sp").weight, padding=_CONV_TAPS // 2, groups=channels)
        x = F.pad(volume.permute(0, 2, 3, 1).float(), (0, 0, 0, 0, 0, 0, _CONV_TAPS // 2, _CONV_TAPS // 2))
        taps = getattr(self, f"{proj}_tm").weight.float().squeeze(1)  # [C, 5]
        out = sum(x[dt : dt + frames] * taps[:, dt] for dt in range(_CONV_TAPS))
        return out.reshape(tokens.shape)


def _activate(x: torch.Tensor, *, l2norm: bool, dtype: torch.dtype = _BF16) -> torch.Tensor:
    """SiLU [+ L2 norm over head_dim] in fp32, rounded once."""
    x = F.silu(x.float())
    return (F.normalize(x, dim=-1, eps=1e-6) if l2norm else x).to(dtype)


def _frame_statistics(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """k, v [F, heads, S, d], beta [F, heads, S] -> A, B [F, heads, d, d] fp32.

    A is inverted downstream, so it is accumulated in fp32 and symmetrized; B
    enters the state linearly and uses bf16 operands.
    """
    k32 = k.float()
    A = (k32 * beta.float().unsqueeze(-1)).transpose(-1, -2) @ k32
    A = 0.5 * (A + A.transpose(-1, -2))
    B = ((v * beta.unsqueeze(-1).to(v.dtype)).transpose(-1, -2) @ k).float()
    return A, B


def _delta_factors(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """S' = (S diag(alpha) + B)(I + A)^-1 as S' = S @ transition + injection (a, b = A, B)."""
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype).expand_as(a)
    # I + A is SPD by construction; the _ex form skips the error check's host sync.
    lower = torch.linalg.cholesky_ex(a + eye).L
    # (I + A)^-1 = L^-T L^-1: one triangular solve and a matmul.
    lower_inv = torch.linalg.solve_triangular(lower, eye, upper=False)
    inverse = lower_inv.transpose(-1, -2) @ lower_inv
    return alpha.unsqueeze(-1) * inverse, b @ inverse


def _scan(transitions: torch.Tensor, injections: torch.Tensor, start: torch.Tensor, *, reverse: bool) -> torch.Tensor:
    states = torch.empty_like(injections)
    state = start
    for frame in reversed(range(len(states))) if reverse else range(len(states)):
        torch.baddbmm(injections[frame], state, transitions[frame], out=states[frame])
        state = states[frame]
    return states


@functools.lru_cache(maxsize=16)
def _gather_indices(bounds: tuple[tuple[int, int], ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    lo = torch.tensor([b[0] for b in bounds], device=device)
    hi = torch.tensor([b[1] for b in bounds], device=device)
    return lo, hi, torch.arange(len(bounds), device=device)


def _outside_window_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    *,
    text_state: torch.Tensor | None,
    bridge: str,
) -> torch.Tensor:
    """prefix[lo-1] + suffix[hi+1] for every frame, decayed to that frame.

    A side past the clip reads the scans' start (the text state, or nothing).
    """
    frames = prefix.shape[0]
    lo, hi, t = _gather_indices(tuple(bounds), prefix.device)
    fill = torch.zeros_like(prefix[0]) if text_state is None else text_state
    left = torch.where((lo > 0).view(-1, 1, 1, 1), prefix[(lo - 1).clamp(min=0)], fill)
    right = torch.where((hi < frames - 1).view(-1, 1, 1, 1), suffix[(hi + 1).clamp(max=frames - 1)], fill)
    if bridge == "alpha":
        # prod_{u=a..b} alpha_u as a difference of exclusive log-prefix sums.
        log_alpha = torch.log(alpha.clamp_min(1e-12)).cumsum(0)
        cumulative = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha])
        # alpha is per key channel: broadcast over the value rows.
        left = left * torch.exp(cumulative[t + 1] - cumulative[lo.clamp(min=0)]).unsqueeze(2)
        right = right * torch.exp(cumulative[(hi + 1).clamp(max=frames)] - cumulative[t]).unsqueeze(2)
    return left + right


class VDNLinearBranch(nn.Module):
    """OpenVDN's BidirectionalLinearBranch on this rank's heads."""

    def __init__(self, hidden: int, heads: int, head_dim: int, config: VDNConfig, *, prefix: str) -> None:
        super().__init__()
        local_heads = heads // get_tensor_model_parallel_world_size()
        self.config = config
        self.short_conv = _ShortConv(local_heads * head_dim, config.short_conv) if config.short_conv else None
        self.alpha = _FrameDecay(hidden, heads, local_heads, head_dim, prefix=f"{prefix}.alpha")
        self.beta_proj = _column(hidden, heads, bias=False, prefix=f"{prefix}.beta_proj")
        self.output_gate = _OutputGate(hidden, heads, head_dim, prefix=f"{prefix}.output_gate")
        self.norm = _Weight(head_dim, sharded=False)  # RMSNorm over head_dim, shared by all heads

    def _features(self, proj: str, tokens: torch.Tensor, grid: tuple[int, int, int] | None) -> torch.Tensor:
        if grid is not None and self.short_conv is not None and proj in self.short_conv.targets:
            tokens = self.short_conv(proj, tokens, grid)
        return _activate(tokens, l2norm=proj != "v")

    def _text_state(self, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """The prompt written into a zero state as one delta-rule chunk, halved."""
        k = self._features("k", k, None).transpose(0, 1).unsqueeze(0)
        v = self._features("v", v, None).transpose(0, 1).unsqueeze(0)
        A, B = _frame_statistics(k.contiguous(), v.contiguous(), beta.transpose(0, 1).unsqueeze(0))
        _, injection = _delta_factors(A, B, torch.ones(A.shape[:-1], device=A.device))
        return TEXT_STATE_SCALE * injection[0]

    @torch.compiler.disable
    def _state(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        frame_mean: torch.Tensor,
        text: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        bounds: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Features by frame -> the state outside each frame's window [F, heads, d, d]."""
        text_state = self._text_state(*text) if text is not None else None
        A, B = _frame_statistics(k, v, beta)
        alpha = self.alpha(frame_mean)
        transitions, injections = _delta_factors(A, B, alpha)
        del A, B
        start = torch.zeros_like(injections[0]) if text_state is None else text_state
        prefix = _scan(transitions, injections, start, reverse=False)
        suffix = _scan(transitions, injections, start, reverse=True)
        del transitions, injections
        state = _outside_window_state(prefix, suffix, alpha, bounds, text_state=text_state, bridge=self.config.bridge)
        return state.to(k.dtype)

    def forward(
        self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: VDNLayout
    ) -> tuple[slice, torch.Tensor] | None:
        """Readout of the video rows the branch covers: ``(rows, [rows, heads*d])``.

        x is the attention input and q/k/v its raw (pre-norm, pre-RoPE)
        projections on this rank's heads. Under ``anchor_frames == "both"`` the
        first and last frame are exact softmax both ways, so the branch drops
        them from its input and they read zero.
        """
        if window.full_cover:
            return None
        skip = 1 if window.anchor_frames == "both" else 0
        frames, per_frame = window.num_frames - 2 * skip, window.tokens_per_frame
        if frames <= 0:
            return None
        rows = slice(window.video_start + skip * per_frame, window.video_end - skip * per_frame)
        bounds = [(lo - skip, hi - skip) for lo, hi in window.window_bounds()[skip : skip + frames]]
        grid = (frames, window.frame_height, window.frame_width)
        heads, head_dim = q.shape[1], q.shape[2]

        beta = torch.sigmoid(self.beta_proj(x)[0])
        text = None
        if self.config.enable_text_state and window.text_len:
            text = (k[: window.text_len], v[: window.text_len], beta[: window.text_len])

        def by_frame(tokens: torch.Tensor) -> torch.Tensor:  # [F*S, h, d] -> [F, h, S, d]
            return tokens.view(frames, per_frame, heads, -1).transpose(1, 2).contiguous()

        state = self._state(
            by_frame(self._features("k", k[rows], grid)),
            by_frame(self._features("v", v[rows], grid)),
            beta[rows].view(frames, per_frame, heads).transpose(1, 2),
            x[rows].view(frames, per_frame, -1).mean(dim=1, dtype=torch.float32),
            text,
            bounds,
        )
        readout = by_frame(self._features("q", q[rows], None)) @ state.transpose(-1, -2)  # [F, h, S, d]
        readout = readout.float()
        readout = readout * torch.rsqrt(readout.pow(2).mean(-1, keepdim=True) + 1e-6) * self.norm.weight.float()
        gate = self.output_gate(x[rows]).float().view(frames, per_frame, heads, head_dim)
        out = (readout.transpose(1, 2) * gate).to(q.dtype)
        return rows, out.reshape(frames * per_frame, heads * head_dim)


class VDNH3HybridAttention(nn.Module):
    """The learned half of one DiT block's hybrid attention (``blocks.N.attn.vdn``)."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        head_dim: int,
        config: VDNConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.softmax_gate = (
            _SoftmaxGate(hidden, heads, prefix=f"{prefix}.softmax_gate") if config.enable_softmax_gate else None
        )
        self.linear_attention = VDNLinearBranch(hidden, heads, head_dim, config, prefix=f"{prefix}.linear_attention")
        # Quantized like out_proj, its softmax twin; the branch's narrow
        # gate/decay projections stay BF16.
        self.to_out_linear = RowParallelLinear(
            heads * head_dim,
            hidden,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out_linear",
        )

    def gate_softmax(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """out [T, heads, d] window softmax -> scaled per (row, head)."""
        if self.softmax_gate is None:
            return out
        return out * self.softmax_gate(x).to(out.dtype).unsqueeze(-1)

    def add_linear(
        self, out: torch.Tensor, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: VDNLayout
    ) -> torch.Tensor:
        """out [T, hidden] += to_out_linear(linear branch) on the covered rows."""
        branch = self.linear_attention(x, q, k, v, window)
        if branch is not None:
            rows, readout = branch
            out[rows] += self.to_out_linear(readout)[0]
        return out


__all__ = [
    "VDNCheckpoint",
    "VDNConfig",
    "VDNH3HybridAttention",
    "VDNLinearBranch",
    "VDN_TASKS",
]
