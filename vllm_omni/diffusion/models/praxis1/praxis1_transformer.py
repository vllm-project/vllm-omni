# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Architecture adapted from https://github.com/runwayml/praxis-1-oss (Praxis-1).
"""Praxis-1 video/action transformer and its joint Euler window sampler.

Noising is ``x_t = (1 - t) * x0 + t * eps`` and the model predicts the velocity
``v = eps - x0``; one Euler step is ``x + (t_next - t_cur) * v`` with a negative
step. The window of the chunk at context depth ``d`` holds ``d + 1`` chunks: the
clean copies of the ``d`` context chunks and ``d + 1`` noised chunks at one
shared time ``t``. Each step makes one backbone call on the whole window and one
expert call on its taps. Only the last chunk is returned.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import WeightsMapper

from vllm_omni.diffusion.models.praxis1.action_expert import ActionExpert, run_expert_on_taps
from vllm_omni.diffusion.models.praxis1.backbone import VideoBackbone
from vllm_omni.model_extras.praxis1 import SamplerSettings

VIDEO_SALT = 0
ACTION_SALT = 1


def guidance_combine(v_cond: Tensor, v_uncond: Tensor, scale: float) -> Tensor:
    """Return the guided velocity ``v_u + scale * (v_c - v_u)`` in float32."""
    if float(scale) == 1.0:
        return v_cond
    return v_uncond.float() + float(scale) * (v_cond.float() - v_uncond.float())


def window_time_grid(num_steps: int, shift: float = 1.0) -> list[float]:
    """Return the descending float64 grid ``1 -> 0`` warped by the flow shift."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    ts = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64, device="cpu")
    if shift != 1.0:
        ts = shift * ts / (1.0 + (shift - 1.0) * ts)
    return [float(t) for t in ts]


def seeded_normal(shape: Sequence[int], seed: int, salt: int) -> Tensor:
    """Return float32 standard normal noise drawn on the CPU from ``seed`` and ``salt``."""
    s = (int(seed) * 1000003 + int(salt)) & 0x7FFFFFFFFFFFFFFF
    gen = torch.Generator().manual_seed(s)
    return torch.randn(tuple(shape), generator=gen, dtype=torch.float32, device="cpu")


# Checkpoint tensors follow the reference module tree. Omni packs Q/K/V and the
# SwiGLU gate/up projections into one parallel linear each, routes the experts
# through FusedMoE and stores the shared expert as linear layers.
_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_prefix={"transformer.": ""},
    orig_to_new_substr={
        ".input_projections.TextEncodingTensor.proj.": ".input_projections.TextEncodingTensor.",
        ".output_projections.VideoTensor.proj.": ".output_projections.VideoTensor.",
    },
    orig_to_new_stacked={
        ".in_proj.q_weight": (".in_proj.weight", "q"),
        ".in_proj.k_weight": (".in_proj.weight", "k"),
        ".in_proj.v_weight": (".in_proj.weight", "v"),
        ".linear.a_weight": (".linear.weight", 0),
        ".linear.b_weight": (".linear.weight", 1),
    },
    orig_to_new_suffix={
        ".routed.gate_up_weight": ".experts.routed_experts.w13_weight",
        ".routed.down_weight": ".experts.routed_experts.w2_weight",
        ".shared.gate_up": ".shared.gate_up.weight",
        ".shared.down": ".shared.down.weight",
    },
)
_EXPERT_SUFFIXES = (".experts.routed_experts.w13_weight", ".experts.routed_experts.w2_weight")
_SHARED_SUFFIXES = (".shared.gate_up.weight", ".shared.down.weight")


class Praxis1Transformer(nn.Module):
    """Video backbone plus action expert, built from a Praxis-1 checkpoint config."""

    def __init__(self, backbone: VideoBackbone, expert: ActionExpert, config: dict) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_expert = expert
        self.config = config

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load checkpoint tensors strictly; return the loaded local names."""
        state = self.state_dict(keep_vars=True)
        loaded: set[str] = set()
        packed_shards: dict[str, set[str | int]] = {}
        for local, tensor in _WEIGHTS_MAPPER.apply(weights):
            packed_shard = getattr(tensor, "shard_id", None)
            if local.endswith(_EXPERT_SUFFIXES):
                tensor = tensor.transpose(1, 2)
            elif local.endswith(_SHARED_SUFFIXES):
                tensor = tensor.T
            if local not in state:
                raise ValueError(f"Unexpected Praxis-1 checkpoint tensor: {local}")
            if packed_shard is not None:
                packed_shards.setdefault(local, set()).add(packed_shard)
                state[local].weight_loader(state[local], tensor, packed_shard)
            else:
                default_weight_loader(state[local], tensor)
            loaded.add(local)
        if missing := state.keys() - loaded:
            raise ValueError(f"Missing Praxis-1 checkpoint tensors: {sorted(missing)}")
        incomplete = {
            key: shards
            for key, shards in packed_shards.items()
            if shards != ({"q", "k", "v"} if key.endswith(".in_proj.weight") else {0, 1})
        }
        if incomplete:
            raise ValueError(f"Incomplete Praxis-1 QKV/SwiGLU tensors: {incomplete}")
        return loaded

    @classmethod
    def from_config(cls, config: dict, attention_impl: str = "flex") -> Praxis1Transformer:
        """Build the model with uninitialized weights."""
        return cls(
            VideoBackbone(config["backbone"], attention_impl),
            ActionExpert(config["action_expert"], attention_impl),
            config,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.backbone.encoder.registers.dtype

    @property
    def device(self) -> torch.device:
        return self.backbone.encoder.registers.device

    @torch.no_grad()
    def sample_window_joint(
        self,
        text: Tensor,
        text_uncond: Tensor | None,
        anchor: Tensor,
        cleans: list[Tensor],
        proprio_all: Tensor,
        seed: int,
        settings: SamplerSettings,
    ) -> tuple[Tensor, Tensor]:
        """Denoise the window of the chunk at depth ``d = len(cleans)`` jointly.

        ``text`` is ``(1, S, C_text)``; ``text_uncond`` is the encoded blank
        caption, required when guidance is on. ``anchor`` is ``(1, C, 1, h, w)``
        and each clean chunk ``(1, C, L, h, w)``. ``proprio_all`` holds the
        normalized boundary pose of each chunk ``(1, d + 1, P)``, current last.
        Returns the last chunk's video latent ``(1, C, L, h, w)`` and its
        normalized action tokens ``(1, L, A)``, both float32.
        """
        guided = settings.guided
        if guided and text_uncond is None:
            raise ValueError("text guidance needs the encoded blank caption")
        dev, md = self.device, self.dtype
        lat = int(self.config["chunking"]["latents_per_chunk"])
        max_chunks = int(self.config["chunking"]["n_chunks"])
        d = len(cleans)
        kk = d + 1
        if kk > max_chunks:
            raise ValueError(f"depth {d} is out of range for {max_chunks} chunks")
        pro = proprio_all if proprio_all.ndim == 3 else proprio_all[None]
        if pro.shape[:2] != (1, kk):
            raise ValueError(f"proprio_all {tuple(pro.shape)} for {kk} chunks")
        a_dim = int(self.config["action_expert"]["action_patch_dim"])
        c, h, w = anchor.shape[1], anchor.shape[3], anchor.shape[4]

        xv = seeded_normal((1, c, kk * lat, h, w), seed, VIDEO_SALT).to(dev)
        xa = seeded_normal((kk * lat, a_dim), seed, ACTION_SALT).to(dev)
        ts = window_time_grid(settings.num_steps, settings.t_shift)
        anchor_m = anchor.to(dev, md)
        clean = torch.cat([z.to(dev, md) for z in cleans], dim=2) if d else None
        pro = pro.to(dev, torch.float32)
        text_c = text.to(dev, md)
        text_u = None if text_uncond is None else text_uncond.to(dev, md)

        def velocity(
            text_in: Tensor, x_v: Tensor, x_a: Tensor, t_cur: float, decode: bool, predict_actions: bool = True
        ) -> tuple[Tensor | None, Tensor | None]:
            t_trunk = torch.full((1, kk), t_cur, dtype=md, device=dev)
            out = self.backbone.forward_chunk_causal(text_in, anchor_m, clean, x_v.to(md), t_trunk, decode=decode)
            v = None if out.velocity is None else out.velocity.float()
            if not predict_actions:
                return v, None
            taps = out.taps
            a = run_expert_on_taps(
                self.action_expert,
                taps.taps,
                taps.embeddings,
                taps.rope_ids,
                noised_span=taps.noised_span,
                clean_span=taps.clean_span,
                n_chunks=kk,
                frames_per_chunk=taps.frames_per_chunk,
                tokens_per_frame=taps.tokens_per_frame,
                noised_actions=x_a[None],
                action_t_per_chunk=torch.full((1, kk), t_cur, device=dev),
                proprio_per_chunk=pro,
            ).float()[0]
            return v, a

        # Skip the unguided passes whose outputs guidance would not use.
        decode_uncond = float(settings.cfg_video_scale) != 1.0
        actions_uncond = float(settings.cfg_scale) != 1.0
        for step in range(settings.num_steps):
            t_cur, t_next = ts[step], ts[step + 1]
            v, a = velocity(text_c, xv, xa, t_cur, True)
            assert v is not None and a is not None
            if guided:
                assert text_u is not None
                v_u, a_u = velocity(text_u, xv, xa, t_cur, decode_uncond, actions_uncond)
                if v_u is not None:
                    v = guidance_combine(v, v_u, settings.cfg_video_scale)
                if a_u is not None:
                    a = guidance_combine(a, a_u, settings.cfg_scale)
            dt = t_next - t_cur
            xv = xv + dt * v
            xa = xa + dt * a
        return xv[:, :, d * lat :].contiguous(), xa[d * lat :].contiguous()[None]
