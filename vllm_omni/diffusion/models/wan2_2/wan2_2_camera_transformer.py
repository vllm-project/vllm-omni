# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Camera-controlled (PRoPE) variant of ``WanTransformer3DModel`` for DreamX-World-5B-Cam.

Adds a per-block PRoPE camera self-attention branch (``CameraSelfAttention``) summed
into the main self-attention before the ``gate_msa`` modulation, matching upstream
DreamX (``models/wan_transformer3d.py``). The camera (PRoPE) path runs single-GPU only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.utils import PPMissingLayer, make_layers

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import get_forward_context
from vllm_omni.diffusion.layers.norm import RMSNorm

from .prope_utils import prope_qkv
from .wan2_2_transformer import WanTransformer3DModel, WanTransformerBlock


class CameraSelfAttention(nn.Module):
    """Parallel PRoPE self-attention branch (ported from DreamX PropeSelfAttention).

    Plain (TP-replicated) Linear projections + RMSNorm on Q/K; PRoPE transforms
    via :func:`prope_qkv`; unified :class:`Attention` layer; ``out_proj`` zero-init
    so the branch is a no-op until trained weights load.
    """

    def __init__(
        self, dim: int, attn_dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6, prefix: str = ""
    ):
        super().__init__()
        assert attn_dim % num_heads == 0, f"attn_dim={attn_dim} not divisible by num_heads={num_heads}"
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        assert self.head_dim % 4 == 0, f"PRoPE requires head_dim % 4 == 0, got {self.head_dim}"

        # Independent q/k/v projections (dim -> attn_dim); names match the checkpoint.
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)

        self.norm_q = RMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(attn_dim, eps=eps) if qk_norm else nn.Identity()

        # Zero-init out_proj: parallel branch contributes nothing until trained.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Unified attention layer; the camera (PRoPE) path is single-GPU only, so SP is skipped.
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            role="self",
            skip_sequence_parallel=True,
            prefix=f"{prefix}.attn" if prefix else "attn",
        )

    def forward(self, x: torch.Tensor, cam_emb: dict) -> torch.Tensor:
        """x: ``[B, L, dim]`` (modulated+normed tokens). Returns ``[B, L, dim]``."""
        batch_size, seq_len, _ = x.shape
        cameras = cam_emb["viewmats"].shape[1]
        assert seq_len % cameras == 0, (
            f"camera-attention seqlen ({seq_len}) must be divisible by cameras "
            f"({cameras} = latent frames); check num_frames/camera-condition alignment"
        )

        act_dtype = x.dtype

        # [B, L, attn_dim] -> [B, L, N, D] -> [B, N, L, D] for prope_qkv.
        q = self.norm_q(self.q_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.norm_k(self.k_proj(x)).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # PRoPE projective math in float32 — upstream keeps the camera condition in
        # fp32, and the SE(3)/intrinsics inversions + projection compose are
        # precision-sensitive (bf16 would compound error across all blocks). The
        # attention itself then runs in the activation dtype.
        q, k, v, apply_fn_o = prope_qkv(
            q.float(), k.float(), v.float(), viewmats=cam_emb["viewmats"].float(), Ks=cam_emb["K"].float()
        )

        # Attention in the activation dtype; the layer expects [B, L, N, D].
        out = self.attn(
            q.to(act_dtype).transpose(1, 2),
            k.to(act_dtype).transpose(1, 2),
            v.to(act_dtype).transpose(1, 2),
        ).transpose(1, 2)

        # Inverse PRoPE transform (fp32) on the attention output, then project back.
        out = apply_fn_o(out.float()).to(act_dtype)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.attn_dim)
        out = self.out_proj(out)
        return out


class WanCameraTransformerBlock(WanTransformerBlock):
    """``WanTransformerBlock`` + optional parallel PRoPE camera self-attention.

    The active ``cam_emb`` for the current forward is stashed on the instance as
    ``_cam_emb`` by :meth:`WanCameraTransformer3DModel.forward` (the base block
    forward signature is fixed, so the camera condition is passed out-of-band;
    diffusion forwards are sequential, so this is safe).
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        cross_attn_norm: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        enable_cam: bool = True,
        attn_compress: int = 1,
        cam_qk_norm: bool = True,
    ):
        super().__init__(
            dim,
            ffn_dim,
            num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attn_norm=cross_attn_norm,
            quant_config=quant_config,
            prefix=prefix,
        )
        if enable_cam:
            self.cam_self_attn = CameraSelfAttention(
                dim=dim,
                attn_dim=dim // attn_compress,
                num_heads=num_heads // attn_compress,
                qk_norm=cam_qk_norm,
                eps=eps,
                prefix=f"{prefix}.cam_self_attn" if prefix else "cam_self_attn",
            )
        else:
            self.cam_self_attn = None
        self._cam_emb: dict | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        hidden_states_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

        if temb.ndim == 4:
            # temb: [B, seq, 6, inner_dim] (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb
            ).chunk(6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: [B, 6, inner_dim] (wan2.1 / wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb
            ).chunk(6, dim=1)

        # 1. Self-attention (+ parallel PRoPE camera self-attention).
        norm_hidden_states = self.norm1(hidden_states, scale_msa, shift_msa).type_as(hidden_states)
        self_attn_metadata = AttentionMetadata(attn_mask=hidden_states_mask)
        attn_output = self.attn1(norm_hidden_states, rotary_emb, self_attn_metadata)
        if self.cam_self_attn is not None and self._cam_emb is not None:
            # Same modulated-normed activation feeds both branches; summed before gate.
            attn_output = attn_output + self.cam_self_attn(norm_hidden_states, self._cam_emb)
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention.
        norm_hidden_states = self.norm2(hidden_states).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward.
        norm_hidden_states = self.norm3(hidden_states, c_scale_msa, c_shift_msa).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        return hidden_states


# --- native (DreamX) -> diffusers (Wan2.2-TI2V-5B-Diffusers) weight key rename ----
# After this rename the base WanTransformer3DModel.load_weights handles the
# remaining diffusers -> vLLM remap (to_qkv fusion, ffn.net.0->net_0, to_out.0->
# to_out, scale_shift_table). cam_self_attn.* keys have no diffusers analog and
# pass through unchanged to default_weight_loader.
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.*)$")
_TOP_RENAME = {
    "patch_embedding.weight": "patch_embedding.weight",
    "patch_embedding.bias": "patch_embedding.bias",
    "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "time_projection.1.bias": "condition_embedder.time_proj.bias",
    "head.head.weight": "proj_out.weight",
    "head.head.bias": "proj_out.bias",
    "head.modulation": "scale_shift_table",
}
_BLOCK_PREFIX_RENAME = {
    "self_attn.q.": "attn1.to_q.",
    "self_attn.k.": "attn1.to_k.",
    "self_attn.v.": "attn1.to_v.",
    "self_attn.o.": "attn1.to_out.0.",
    "self_attn.norm_q.": "attn1.norm_q.",
    "self_attn.norm_k.": "attn1.norm_k.",
    "cross_attn.q.": "attn2.to_q.",
    "cross_attn.k.": "attn2.to_k.",
    "cross_attn.v.": "attn2.to_v.",
    "cross_attn.o.": "attn2.to_out.0.",
    "cross_attn.norm_q.": "attn2.norm_q.",
    "cross_attn.norm_k.": "attn2.norm_k.",
    "ffn.0.": "ffn.net.0.proj.",
    "ffn.2.": "ffn.net.2.",
    "norm3.": "norm2.",
}


def native_to_diffusers_key(name: str) -> str:
    """Map a DreamX native-Wan weight key to its diffusers-named equivalent.

    Names that are already diffusers-named (or unknown) pass through unchanged,
    so a pre-converted checkpoint also loads.
    """
    if name in _TOP_RENAME:
        return _TOP_RENAME[name]
    m = _BLOCK_RE.match(name)
    if not m:
        return name
    i, rest = m.group(1), m.group(2)
    if rest.startswith("cam_self_attn."):
        return name  # DreamX-specific; no diffusers analog.
    if rest == "modulation":
        return f"blocks.{i}.scale_shift_table"
    for nat, dif in _BLOCK_PREFIX_RENAME.items():
        if rest.startswith(nat):
            return f"blocks.{i}.{dif}{rest[len(nat) :]}"
    return name


class WanCameraTransformer3DModel(WanTransformer3DModel):
    """Wan2.2 TI2V-5B transformer with a per-block PRoPE camera self-attention branch."""

    _repeated_blocks = ["WanCameraTransformerBlock"]

    def __init__(
        self,
        *,
        cam_method: str = "prope",
        attn_compress: int = 1,
        cam_self_attn_layers: list[int] | None = None,
        add_control_adapter: bool = True,
        cam_qk_norm: bool = True,
        quant_config: QuantizationConfig | None = None,
        **kwargs,
    ):
        super().__init__(quant_config=quant_config, **kwargs)
        if cam_method != "prope":
            raise NotImplementedError(f"cam_method={cam_method!r} not supported (only 'prope')")
        self.cam_method = cam_method
        self.attn_compress = attn_compress
        self.cam_self_attn_layers = cam_self_attn_layers
        self.add_control_adapter = add_control_adapter
        self.cam_qk_norm = cam_qk_norm

        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        num_heads = self.config.num_attention_heads
        num_layers = self.config.num_layers
        cam_layers = set(range(num_layers)) if cam_self_attn_layers is None else set(cam_self_attn_layers)

        def _build_block(prefix: str) -> WanCameraTransformerBlock:
            idx = int(prefix.split(".")[-1])
            return WanCameraTransformerBlock(
                inner_dim,
                self.config.ffn_dim,
                num_heads,
                self.config.eps,
                self.config.added_kv_proj_dim,
                self.config.cross_attn_norm,
                quant_config=quant_config,
                prefix=prefix,
                enable_cam=add_control_adapter and (idx in cam_layers),
                attn_compress=attn_compress,
                cam_qk_norm=cam_qk_norm,
            )

        # Rebuild blocks with the camera-aware block. The base __init__ already
        # built plain WanTransformerBlocks; free them first (drop the reference so
        # they are reclaimed before the camera blocks are built — avoids a 2x
        # transient block-memory peak at init for the 5B model) then rebuild.
        # Keeps blocks.{i}.cam_self_attn.* keys so the base load_weights loads them.
        self.blocks = None
        self.start_layer, self.end_layer, self.blocks = make_layers(num_layers, _build_block, prefix="blocks")

    def _validate_camera_condition(self, cam_emb: dict, hidden_states: torch.Tensor) -> None:
        """Fail loud on a misaligned camera condition or unsupported parallelism.

        The camera branch runs single-GPU: SP/PP shard/pad the token sequence, but
        the camera attention runs full (no ``hidden_states_mask``), so SP/PP would
        silently corrupt it. Also enforces ``cameras == latent frames`` —
        ``CameraSelfAttention`` only checks ``seqlen % cameras == 0``, which a wrong
        count (e.g. ``cameras == 1``) can pass while silently tiling one camera
        across all frames.
        """
        ctx = get_forward_context()
        pc = getattr(getattr(ctx, "omni_diffusion_config", None), "parallel_config", None)
        if pc is not None and (
            getattr(pc, "sequence_parallel_size", 1) > 1 or getattr(pc, "pipeline_parallel_size", 1) > 1
        ):
            raise NotImplementedError(
                "DreamX camera path requires sequence_parallel_size == 1 and "
                "pipeline_parallel_size == 1; the camera (PRoPE) branch attends over "
                "the full, unsharded sequence."
            )

        if not isinstance(cam_emb, dict) or "viewmats" not in cam_emb or "K" not in cam_emb:
            raise ValueError("cam_emb must be a dict with keys 'viewmats' and 'K'")
        vm, ks = cam_emb["viewmats"], cam_emb["K"]
        if vm.ndim != 4 or tuple(vm.shape[-2:]) != (4, 4):
            raise ValueError(f"cam_emb['viewmats'] must be [B, cameras, 4, 4], got {tuple(vm.shape)}")
        if ks.ndim != 4 or tuple(ks.shape[-2:]) != (3, 3):
            raise ValueError(f"cam_emb['K'] must be [B, cameras, 3, 3], got {tuple(ks.shape)}")
        if vm.shape[1] != ks.shape[1]:
            raise ValueError(f"viewmats cameras ({vm.shape[1]}) != K cameras ({ks.shape[1]})")

        # hidden_states enters as the latent [B, C, F_lat, H, W]; with PP == 1
        # (enforced above) cameras must equal the post-patch latent frame count.
        post_patch_num_frames = hidden_states.shape[2] // self.config.patch_size[0]
        if vm.shape[1] != post_patch_num_frames:
            raise ValueError(
                f"camera count ({vm.shape[1]}) must equal latent frames ({post_patch_num_frames} = "
                f"num_frames // patch_t); camera condition is misaligned with num_frames "
                "(a wrong count can silently divide the token sequence)"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        intermediate_tensors=None,
        return_dict: bool = True,
        attention_kwargs: dict | None = None,
        cam_emb: dict | None = None,
    ):
        # Validate, then move the (CFG-invariant) camera condition to the activation
        # device and stash it on each block for the base forward's block loop. Keep
        # its dtype (fp32) — CameraSelfAttention does the PRoPE projective math in
        # fp32 (matching upstream), casting to the activation dtype only for attention.
        if cam_emb is not None:
            self._validate_camera_condition(cam_emb, hidden_states)
            batch = hidden_states.shape[0]
            cam_emb = {k: v.to(device=hidden_states.device) for k, v in cam_emb.items()}
            # Broadcast a batch-1 condition to the latent batch (e.g. when
            # num_outputs_per_prompt > 1 repeats the latents/prompt); prope_qkv
            # requires viewmats.shape[0] == hidden_states batch.
            cam_emb = {
                k: (v.expand(batch, *v.shape[1:]) if v.shape[0] == 1 and batch > 1 else v) for k, v in cam_emb.items()
            }
        local_blocks = [b for b in self.blocks[self.start_layer : self.end_layer] if not isinstance(b, PPMissingLayer)]
        for block in local_blocks:
            block._cam_emb = cam_emb
        try:
            return super().forward(
                hidden_states,
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                intermediate_tensors,
                return_dict,
                attention_kwargs,
            )
        finally:
            for block in local_blocks:
                block._cam_emb = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Rename native-Wan keys to diffusers, then delegate to the base loader."""
        return super().load_weights((native_to_diffusers_key(name), w) for name, w in weights)


def create_camera_transformer_from_config(
    config: dict, quant_config: QuantizationConfig | None = None
) -> WanCameraTransformer3DModel:
    """Build ``WanCameraTransformer3DModel`` from a config dict.

    Accepts both native DreamX field names (``dim``/``num_heads``/``in_dim``...)
    and diffusers names (``num_attention_heads``/``attention_head_dim``/
    ``in_channels``...), plus the camera extras.
    """
    cfg = dict(config)
    kwargs: dict = {}

    num_heads = cfg.get("num_attention_heads", cfg.get("num_heads"))
    dim = cfg.get("hidden_size", cfg.get("dim"))
    head_dim = cfg.get("attention_head_dim")
    if head_dim is None and dim is not None and num_heads:
        head_dim = dim // num_heads
    if num_heads is not None:
        kwargs["num_attention_heads"] = num_heads
    if head_dim is not None:
        kwargs["attention_head_dim"] = head_dim

    in_ch = cfg.get("in_channels", cfg.get("in_dim"))
    out_ch = cfg.get("out_channels", cfg.get("out_dim"))
    if in_ch is not None:
        kwargs["in_channels"] = in_ch
    if out_ch is not None:
        kwargs["out_channels"] = out_ch

    for key in (
        "text_dim",
        "freq_dim",
        "ffn_dim",
        "num_layers",
        "cross_attn_norm",
        "eps",
        "image_dim",
        "added_kv_proj_dim",
        "rope_max_seq_len",
        "pos_embed_seq_len",
    ):
        if key in cfg:
            kwargs[key] = cfg[key]
    if "patch_size" in cfg:
        kwargs["patch_size"] = tuple(cfg["patch_size"])

    # Camera extras.
    kwargs["cam_method"] = cfg.get("cam_method", "prope")
    kwargs["attn_compress"] = cfg.get("attn_compress", 1)
    kwargs["cam_self_attn_layers"] = cfg.get("cam_self_attn_layers", None)
    kwargs["add_control_adapter"] = cfg.get("add_control_adapter", True)
    # qk_norm gates the camera branch's Q/K RMSNorm (config-driven, matching
    # upstream). Native config uses a bool; diffusers uses a string / None.
    kwargs["cam_qk_norm"] = bool(cfg.get("qk_norm", True))

    if "quantization_config" in cfg:
        from vllm_omni.quantization.factory import resolve_quant_config_from_disk

        quant_config = resolve_quant_config_from_disk(quant_config, cfg["quantization_config"])
    if quant_config is not None:
        kwargs["quant_config"] = quant_config

    return WanCameraTransformer3DModel(**kwargs)
