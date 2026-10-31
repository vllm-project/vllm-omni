# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSSImage DiT transformer for vLLM-Omni (TP / USP / batched).

Ported from the training repository's finalized standalone modeling file
``pipeline_moss_image.py`` with vLLM-Omni parallel support:

  * **Tensor parallelism**: DiT/QKV and FFN projections use vLLM
    ``QKVParallelLinear`` / ``MergedColumnParallelLinear`` / ``RowParallelLinear``;
    small replicated layers (embedders, modulation, norms, head) stay dense.
  * **Sequence parallelism (Ulysses)**: the joint [video | text] sequence is
    sharded across SP ranks via the declared ``_sp_plan`` hooks; RoPE is
    computed globally in fp32 and sharded together with the sequence.
  * **Attention**: runs through the vLLM-Omni ``Attention`` layer with an
    optional key-padding mask (video tokens attend everything; padded text
    tokens are masked out), matching the reference per-sample true-length
    semantics under batched execution.
  * **Precision contract** (unchanged): bf16 learned weights, fp32 RoPE
    frequencies/phases (cast to activation dtype at the end), fp32 CFG and
    Euler updates (handled by the pipeline), float64 sinusoid time embedding
    fed to the MLP as fp32.

Checkpoint compatibility: state-dict names are identical to the diffusers
checkpoint (463 tensors verified against
``transformer/diffusion_pytorch_model.safetensors.index.json``); QKV/FFN
sharding is handled by ``load_weights`` mappings.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch import nn
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.distributed.sp_sharding import get_sequence_parallel_world_size
from vllm_omni.diffusion.models.mossimage.ops import (
    gated_residual,
    try_qk_norm_rope,
    try_rms_norm,
    try_rotate_qk,
    try_situ,
)

# (fused_param_stem, checkpoint_stem, shard_id) mappings for stacked loads.
_MOSSIMAGE_STACKED_PARAMS_MAPPING = (
    # Text refinement blocks: checkpoint stores fused qkv as one tensor and
    # separate ffn_gate / ffn_up; the model merges ffn into [up | gate].
    (".ffn_up_gate.", ".ffn_up.", 0),
    (".ffn_up_gate.", ".ffn_gate.", 1),
    # DiT blocks: checkpoint stores per-attention-branch q/k/v separately.
    (".proj.qkv.", ".proj.q.", "q"),
    (".proj.qkv.", ".proj.k.", "k"),
    (".proj.qkv.", ".proj.v.", "v"),
)


class MOSSImageRMSNorm(nn.Module):
    """RMSNorm computed in fp32, matching the training reference."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor | None = None, shift: torch.Tensor | None = None
    ) -> torch.Tensor:
        fused = try_rms_norm(x, self.weight, scale, shift)
        if fused is not None:
            return fused
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        result = y.to(x.dtype) * self.weight
        if scale is not None:
            result = result * (1 + scale[:, None]) + shift[:, None]
        return result


def _situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    fused = try_situ(hidden, gate)
    if fused is not None:
        return fused
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply 4D RoPE: ``x`` is [B, S, H, D], ``rope`` is [1 or B, S, D]."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


class MOSSImageAttention(nn.Module):
    """Per-head QK RMSNorm + 4D-RoPE + vLLM-Omni attention (parameter-free glue).

    ``q_norm``/``k_norm`` stay as direct attributes of the owning block so
    checkpoint state-dict names (e.g. ``layers.0.q_norm.weight``) keep
    matching. Q/K/V arrive as [B, S, heads*head_dim] (TP-sharded in the last
    dim); the vLLM-Omni attention layer consumes and returns
    [B, S, heads, head_dim].
    """

    def __init__(self, num_heads: int, head_dim: int, prefix: str = "", skip_sequence_parallel: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=False,
            num_kv_heads=num_heads,
            role="self",
            prefix=prefix,
            skip_sequence_parallel=skip_sequence_parallel,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rope: torch.Tensor,
        q_norm: MOSSImageRMSNorm,
        k_norm: MOSSImageRMSNorm,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = q.unflatten(-1, (self.num_heads, -1))
        k = k.unflatten(-1, (self.num_heads, -1))
        rotated = try_qk_norm_rope(q, k, q_norm.weight, k_norm.weight, rope)
        if rotated is None:
            q, k = q_norm(q), k_norm(k)
            rotated = try_rotate_qk(q, k, rope)
            if rotated is None:
                rotated = _rotate(q, rope), _rotate(k, rope)
        q, k = rotated
        v = v.unflatten(-1, (self.num_heads, -1))
        # Match the reference SDPA contract: Q/K follow V's dtype.
        metadata = AttentionMetadata(attn_mask=key_mask) if key_mask is not None else None
        out = self.attn(q.to(v.dtype), k.to(v.dtype), v, metadata)
        return out.flatten(2)


class MOSSImageTextBlock(nn.Module):
    """Text-refinement block (biased qkv/proj, gated SiTU FFN). TP-sharded."""

    def __init__(self, dim: int, ffn_dim: int, heads: int, prefix: str = ""):
        super().__init__()
        head_dim = dim // heads
        self.norm1 = MOSSImageRMSNorm(dim)
        self.norm2 = MOSSImageRMSNorm(dim)
        self.q_norm = MOSSImageRMSNorm(head_dim)
        self.k_norm = MOSSImageRMSNorm(head_dim)
        # Checkpoint names: qkv/proj/ffn_up/ffn_down live directly on the block.
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=heads,
            bias=True,
            prefix=f"{prefix}.qkv" if prefix else "",
        )
        self.heads = self.qkv.num_heads  # heads on this TP rank
        # The text-refinement blocks run on the full (replicated) text
        # sequence: it lives outside the _sp_plan sharding boundary, so this
        # attention site must opt out of sequence-parallel redistribution.
        self.attention = MOSSImageAttention(
            self.heads,
            head_dim,
            prefix=f"{prefix}.attn" if prefix else "",
            skip_sequence_parallel=True,
        )
        self.proj = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            prefix=f"{prefix}.proj" if prefix else "",
        )
        self.ffn_up_gate = MergedColumnParallelLinear(
            dim, [ffn_dim, ffn_dim], bias=True, prefix=f"{prefix}.ffn_up_gate" if prefix else ""
        )
        self.ffn_down = RowParallelLinear(
            ffn_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            prefix=f"{prefix}.ffn_down" if prefix else "",
        )

    def forward(self, x: torch.Tensor, rope: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        qkv, _ = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, -1)
        x = x + self.proj(self.attention(q, k, v, rope, self.q_norm, self.k_norm, key_mask))
        y = self.norm2(x)
        up_gate, _ = self.ffn_up_gate(y)
        hidden, gate = up_gate.chunk(2, -1)  # [up | gate]
        return x + self.ffn_down(_situ(hidden, gate))


class MOSSImageTextProjection(nn.Module):
    """Qwen3.5 hidden-state projection with two refinement blocks."""

    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int, prefix: str = ""):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList(
            [MOSSImageTextBlock(dim, ffn_dim, heads, prefix=f"{prefix}.blocks.{i}") for i in range(2)]
        )

    def forward(self, x: torch.Tensor, rope: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, key_mask)
        return x


class MOSSImageBlock(nn.Module):
    """Main DiT block: shared AdaLN modulation, self-attention + gated SiTU FFN. TP-sharded."""

    def __init__(self, dim: int, ffn_dim: int, heads: int, prefix: str = ""):
        super().__init__()
        head_dim = dim // heads
        self.norm1 = MOSSImageRMSNorm(dim)
        self.norm2 = MOSSImageRMSNorm(dim)
        self.q_norm = MOSSImageRMSNorm(head_dim)
        self.k_norm = MOSSImageRMSNorm(head_dim)
        # Checkpoint names: proj.{q,k,v,o,ffn_up_gate,ffn_down} (the qkv entry
        # is fused from three checkpoint tensors at load time).
        self.proj = nn.ModuleDict(
            {
                "qkv": QKVParallelLinear(
                    hidden_size=dim,
                    head_size=head_dim,
                    total_num_heads=heads,
                    bias=False,
                    prefix=f"{prefix}.proj.qkv" if prefix else "",
                ),
                "o": RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    return_bias=False,
                    prefix=f"{prefix}.proj.o" if prefix else "",
                ),
                "ffn_up_gate": MergedColumnParallelLinear(
                    dim,
                    [ffn_dim, ffn_dim],
                    bias=False,
                    prefix=f"{prefix}.proj.ffn_up_gate" if prefix else "",
                ),
                "ffn_down": RowParallelLinear(
                    ffn_dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    return_bias=False,
                    prefix=f"{prefix}.proj.ffn_down" if prefix else "",
                ),
            }
        )
        self.heads = self.proj["qkv"].num_heads
        self.attention = MOSSImageAttention(self.heads, head_dim, prefix=f"{prefix}.attn" if prefix else "")

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x, scale_a, shift_a)
        dtype = self.proj["qkv"].weight.dtype
        y = y.to(dtype)
        qkv, _ = self.proj["qkv"](y)
        q, k, v = qkv.chunk(3, -1)
        branch = self.proj["o"](self.attention(q, k, v, rope, self.q_norm, self.k_norm, key_mask))
        x = gated_residual(x, branch, gate_a)
        y = self.norm2(x, scale_f, shift_f)
        up_gate, _ = self.proj["ffn_up_gate"](y.to(dtype))
        hidden, gate = up_gate.chunk(2, -1)  # [up | gate]
        return gated_residual(x, self.proj["ffn_down"](_situ(hidden, gate)), gate_f)


class MOSSImageRoPE(nn.Module):
    """4D (r, f, h, w) rotary embedding with canonical fp32 frequencies.

    Frequencies are derived constants; they are rebuilt in fp32 after any
    dtype conversion or state-dict load so bf16 model conversion can never
    silently round them (this reproduces the standalone release's fix).
    """

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        # Buffers follow the active default device: vLLM-Omni initializes the
        # model inside a ``with <device>:`` context and only moves parameters
        # after loading, so an explicit CPU buffer would stay on CPU.
        self.register_buffer("bands", self._default_bands(torch.get_default_device()))

    def _default_bands(self, device) -> torch.Tensor:
        # Values are constructed on CPU in fp32 (training-compatible), then
        # moved to the target device.
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Retain the key for checkpoint compatibility; rebuild canonical fp32.
        self.bands = self._default_bands(self.bands.device)

    def forward(
        self,
        height: int,
        width: int,
        text_length: int,
        device: torch.device,
        *,
        text_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build coordinates using semantic lengths, independent of padded text width."""
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        batch = 1 if text_lengths is None else text_lengths.shape[0]
        num_video = height * width
        positions = torch.zeros(batch, num_video + text_length, 4, device=device, dtype=torch.float32)
        positions[:, :num_video, 1] = text_length if text_lengths is None else text_lengths[:, None]
        positions[:, :num_video, 2] = h.flatten().float() - (height - 1) / 2
        positions[:, :num_video, 3] = w.flatten().float() - (width - 1) / 2
        # Valid right-padded text tokens retain positions 0..L-1 for each sample.
        positions[:, num_video:, 1] = torch.arange(text_length, device=device)
        phase = positions.unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)


class MOSSImageDenoiseBody(nn.Module):
    """Sequence-sharded DiT body (all blocks). ``layers`` is a shared reference
    to the transformer's root ModuleList, so parameter names stay rooted at
    ``layers.{i}`` for checkpoint compatibility.

    Note: the SP hook metadata unwraps modules that hold exactly one child
    (``_unwrap_module``), which would collapse this shell into the bare
    ModuleList and break the ``hidden_states`` signature lookup. The extra
    identity anchor (no parameters) keeps the shell intact.
    """

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self._sp_anchor = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        modulation: torch.Tensor,
        key_mask: torch.Tensor | None = None,
        skip_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        x = hidden_states
        for index, layer in enumerate(self.layers):
            if index not in skip_layers:
                x = layer(x, rope, modulation, key_mask)
        return x


class MOSSImageFinalOut(nn.Module):
    """Per-token output head (norm + AdaLN scale/shift + projection).

    Holds shared references to the root transformer's ``final_norm_video`` /
    ``final_linear_video``; because it is per-token it can run directly on the
    SP-sharded sequence, and the ``_sp_plan`` gathers its output.
    """

    def __init__(self, norm: MOSSImageRMSNorm, proj: nn.Linear):
        super().__init__()
        self.final_norm_video = norm
        self.final_linear_video = proj

    def forward(self, hidden_states: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        # head: [B, 2, dim] -> (shift, scale)
        x = hidden_states.to(self.final_norm_video.weight.dtype)
        x = self.final_norm_video(x)
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        return self.final_linear_video(x.to(self.final_linear_video.weight.dtype))


class MOSSImageTransformer(nn.Module):
    """MOSSImage DiT: batched forward with TP and Ulysses sequence parallelism.

    Input contract is unchanged from the reference implementation:
    ``hidden_states`` [B, latent_channels, H, W], ``encoder_hidden_states``
    [B, S, text_dim] (right-padded), ``encoder_attention_mask`` binary [B, S].
    Padded text tokens are masked out of all attention sites, reproducing the
    reference per-sample true-length semantics.
    """

    supports_skip_layer_guidance = True
    _repeated_blocks = ["MOSSImageBlock"]
    _layerwise_offload_blocks_attrs = ["layers"]
    # Ulysses: shard the joint [video | text] sequence and the RoPE table at
    # the denoise body, then gather the per-token output projection. The
    # modulation vectors are per-batch (no sequence axis) and stay replicated.
    _sp_plan = {
        "denoise_body": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
            "rope": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True),
        },
        "final_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig | None = None,
        dim: int = 3072,
        ffn_dim: int = 8192,
        num_heads: int = 24,
        num_layers: int = 42,
        text_dim: int = 4096,
        freq_dim: int = 256,
        latent_channels: int = 128,
        rope_base: float = 10000.0,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.dim = dim
        self.num_heads = num_heads
        self.freq_dim = freq_dim
        self.latent_channels = latent_channels
        self.quant_config = getattr(od_config, "quantization_config", None) if od_config else None
        self.stacked_params_mapping = list(_MOSSIMAGE_STACKED_PARAMS_MAPPING)

        # Replicated (non-TP) small layers: embedders, time/modulation, head.
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MOSSImageTextProjection(text_dim, dim, ffn_dim, num_heads, prefix="text_embedder")
        self.rope = MOSSImageRoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [MOSSImageBlock(dim, ffn_dim, num_heads, prefix=f"layers.{i}") for i in range(num_layers)]
        )
        self.final_norm_video = MOSSImageRMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

        # Plan hook boundaries (shared references; parameter names unchanged).
        self.denoise_body = MOSSImageDenoiseBody(self.layers)
        self.final_out = MOSSImageFinalOut(self.final_norm_video, self.final_linear_video)

    def _time_modulation(self, timestep, batch: int, device: torch.device):
        times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        return mod, head

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | list[float] | float,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        _conditioning_cache: dict | None = None,
        skip_layers: tuple[int, ...] = (),
        **kwargs,
    ):
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        device = hidden_states.device
        text_len = encoder_hidden_states.shape[1]

        if encoder_attention_mask is None:
            text_mask = torch.ones(batch, text_len, dtype=torch.bool, device=device)
        else:
            text_mask = encoder_attention_mask.bool()
            if text_mask.shape != (batch, text_len):
                raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        text_lengths = text_mask.sum(dim=1)
        if bool((text_lengths == 0).any()):
            raise ValueError("Each sample needs at least one valid text token.")

        # Align storage with the Ulysses shard count. Semantic text lengths
        # above exclude both batch padding and the extra SP slots below.
        sp_size = max(1, get_sequence_parallel_world_size())
        num_video = height * width
        joint_len = num_video + text_len
        if joint_len % sp_size != 0:
            pad = sp_size - (joint_len % sp_size)
            encoder_hidden_states = torch.nn.functional.pad(encoder_hidden_states, (0, 0, 0, pad))
            text_mask = torch.nn.functional.pad(text_mask, (0, pad), value=False)
            text_len += pad

        # An all-valid mask is equivalent to no mask. Resolve this once here
        # rather than letting every attention layer synchronize to test it.
        # Keep the existing mask contract for sequence-parallel hooks.
        attention_text_mask = text_mask
        if sp_size == 1 and bool(text_mask.all()):
            attention_text_mask = None

        video = hidden_states.flatten(2).transpose(1, 2).contiguous()
        video = self.video_embedder(video.to(self.video_embedder.weight.dtype))
        # RoPE is computed for the full joint sequence in fp32, then cast to
        # the activation dtype (matching the reference activation contract).
        # Only the full-forward pipeline supplies a fresh per-branch cache for
        # an immutable request batch. Never cache across requests, stepwise
        # batch updates, training, compilation, or SP hook transformations.
        cache = _conditioning_cache
        if sp_size != 1 or torch.is_grad_enabled() or torch.compiler.is_compiling():
            cache = None
        if cache is not None and "conditioning" in cache:
            rope, text = cache["conditioning"]
        else:
            rope = self.rope(
                height,
                width,
                text_len,
                device,
                text_lengths=text_lengths if attention_text_mask is not None else None,
            ).to(video.dtype)
            text = self.text_embedder(encoder_hidden_states, rope[:, num_video:], attention_text_mask)
            if cache is not None:
                cache["conditioning"] = (rope, text)
        x = torch.cat((video, text), 1)
        joint_key_mask = None
        if attention_text_mask is not None:
            joint_key_mask = torch.cat(
                (torch.ones(batch, num_video, dtype=torch.bool, device=device), text_mask), dim=1
            )

        mod, head = self._time_modulation(timestep, batch, device)
        if skip_layers:
            self.validate_skip_layers(skip_layers)
            x = self.denoise_body(x, rope, mod, joint_key_mask, skip_layers=skip_layers)
        else:
            x = self.denoise_body(x, rope, mod, joint_key_mask)
        x = self.final_out(x, head)  # _sp_plan gathers along the sequence dim
        x = x[:, :num_video]  # drop text (and any SP pad) tokens
        sample = x.transpose(1, 2).reshape(batch, self.latent_channels, height, width)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)

    def validate_skip_layers(self, skip_layers: tuple[int, ...]) -> None:
        if any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.layers)
            for index in skip_layers
        ):
            raise ValueError(f"MOSSImage SLG layer IDs must be integers in [0, {len(self.layers) - 1}].")

    # ------------------------------------------------------------------
    # Weight loading: checkpoint names are unchanged (dense naming); fused
    # QKV/FFN shards are mapped here.
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        # rope.bands is a persistent buffer (fp32, replicated).
        params.setdefault("rope.bands", self.rope.bands)
        loaded_params: set[str] = set()

        for orig_name, loaded_weight in weights:
            name = orig_name

            # 1) Stacked shards: text FFN gate/up and DiT q/k/v.
            mapped = False
            for fused_stem, ckpt_stem, shard_id in self.stacked_params_mapping:
                if ckpt_stem not in name:
                    continue
                name = name.replace(ckpt_stem, fused_stem)
                param = params.get(name)
                if param is None:
                    raise KeyError(f"MOSSImage: mapped param {name!r} (from {orig_name!r}) not found")
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                mapped = True
                break
            if mapped:
                continue

            # 2) DiT fused FFN: checkpoint stores one [up | gate] tensor.
            if name.endswith("proj.ffn_up_gate.weight"):
                param = params.get(name)
                if param is None:
                    raise KeyError(f"MOSSImage: param {name!r} not found")
                up, gate = loaded_weight.chunk(2, dim=0)
                param.weight_loader(param, up.contiguous(), 0)
                param.weight_loader(param, gate.contiguous(), 1)
                loaded_params.add(name)
                continue

            # 3) Text refinement QKV: checkpoint stores one [q | k | v] tensor.
            if name.startswith("text_embedder.") and (name.endswith("qkv.weight") or name.endswith("qkv.bias")):
                param = params.get(name)
                if param is None:
                    raise KeyError(f"MOSSImage: param {name!r} not found")
                q, k, v = loaded_weight.chunk(3, dim=0)
                param.weight_loader(param, q.contiguous(), "q")
                param.weight_loader(param, k.contiguous(), "k")
                param.weight_loader(param, v.contiguous(), "v")
                loaded_params.add(name)
                continue

            # 4) Everything else: dense or row/column-parallel layers whose
            #    registered weight_loader handles TP narrowing.
            param = params.get(name)
            if param is None:
                raise KeyError(f"MOSSImage: unexpected checkpoint weight {orig_name!r}")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params
