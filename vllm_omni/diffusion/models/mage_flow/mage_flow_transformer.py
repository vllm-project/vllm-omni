# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

"""Native vLLM-Omni implementation of the Mage-Flow NR-MMDiT."""

from collections.abc import Iterable

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)

from .mage_flow_layers import (
    AdaLayerNormContinuous,
    MageFlowDoubleStreamBlock,
    MageFlowEmbedRope,
    MageFlowImageRopePrepare,
    MageFlowTimestepProjEmbeddings,
    _request_isolated_forward,
    _sequence_parallel_active,
)

logger = init_logger(__name__)

# The official checkpoint ships unfused ``to_q``/``to_k``/``to_v`` (and the text
# stream's ``add_*_proj``) tensors, while the model fuses each triple into a
# single ``QKVParallelLinear`` so tensor parallelism can shard whole heads.
MAGE_FLOW_STACKED_PARAMS_MAPPING: list[tuple[str, str, str]] = [
    # (fused param name, checkpoint weight name, shard id)
    (".to_qkv", ".to_q", "q"),
    (".to_qkv", ".to_k", "k"),
    (".to_qkv", ".to_v", "v"),
    (".add_qkv_proj", ".add_q_proj", "q"),
    (".add_qkv_proj", ".add_k_proj", "k"),
    (".add_qkv_proj", ".add_v_proj", "v"),
]


def _warn_on_unmasked_sp_padding() -> None:
    """Report the numerical caveat when SP auto-padding is in play.

    ``_sp_plan`` declares ``auto_pad=True``, so a token count that is not
    divisible by the sequence-parallel degree is zero-padded at split time and
    trimmed again at gather time. Those filler tokens still take part in
    attention, exactly as they do for Wan under ``mask_sp_padding=False``.
    Masking them would need a mask that survives the Ulysses all-to-all, which
    this model's joint attention does not build, so ``mask_sp_padding`` is not
    honoured here and says so rather than implying strict masking.
    """
    if not is_forward_context_available():
        return
    ctx = get_forward_context()
    if ctx.sp_original_seq_len is None or ctx.sp_padding_size <= 0:
        return
    logger.warning_once(
        "SP auto-padding applied %d token(s) (seq_len=%d). Padding tokens are "
        "not masked from attention, which avoids the varlen attention path but "
        "may produce minor numerical differences. Mage-Flow does not honour "
        "parallel_config.mask_sp_padding; choose a token count divisible by the "
        "sequence-parallel degree if you need every token masked exactly.",
        ctx.sp_padding_size,
        ctx.sp_original_seq_len,
    )


def resolve_mage_flow_stacked_name(name: str) -> tuple[str, str | None]:
    """Rewrite a checkpoint weight name onto its fused QKV parameter.

    Returns ``(param_name, shard_id)``; ``shard_id`` is ``None`` for weights
    that are not part of a fused projection and load unchanged.
    """
    for param_name, weight_name, shard_id in MAGE_FLOW_STACKED_PARAMS_MAPPING:
        if weight_name in name and param_name not in name:
            return name.replace(weight_name, param_name), shard_id
    return name, None


class MageFlowTransformer2DModel(nn.Module):
    """The official Mage-Flow transformer with padded request-batch attention."""

    _repeated_blocks = ["MageFlowDoubleStreamBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    # Sequence parallelism shards the image stream only. The text stream stays
    # replicated and is spliced back in as joint tensors inside attention, the
    # same split Qwen-Image uses for its dual-stream blocks. hidden_states and
    # the RoPE table are emitted together so they shard in lockstep.
    _sp_plan = {
        "image_rope_prepare": {
            0: SequenceParallelInput(
                split_dim=1,
                expected_dims=3,
                split_output=True,
                auto_pad=True,
            ),
            1: SequenceParallelInput(
                split_dim=1,
                expected_dims=3,
                split_output=True,
                auto_pad=True,
            ),
        },
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig | None = None,
        *,
        in_channels: int = 128,
        out_channels: int = 128,
        context_in_dim: int = 2560,
        hidden_size: int = 3072,
        num_heads: int = 24,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        depth_single_blocks: int = 0,
        axes_dim: list[int] | None = None,
        theta: int = 10000,
        patch_size: int = 1,
        qkv_bias: bool = True,
        guidance_embed: bool = False,
        rope_type: str = "msrope",
        time_type: str = "qwen_proj",
        double_block_type: str = "double_stream",
        apply_text_rotary_emb: bool = False,
        packing: bool = True,
        schedule_mode: str = "z-image",
        static_shift: float = 6.0,
        use_time_shift: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        # The pipeline expands transformer/config.json into these keyword
        # arguments, so no separate lookup against od_config is needed.
        axes_dim = list(axes_dim) if axes_dim is not None else [16, 56, 56]

        if hidden_size % num_heads:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        head_dim = hidden_size // num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size:
            raise ValueError(
                f"Mage-Flow tensor parallelism shards whole attention heads, so num_heads "
                f"({num_heads}) must be divisible by tensor_parallel_size ({tp_size})"
            )
        if sum(axes_dim) != head_dim:
            raise ValueError(f"sum(axes_dim) ({sum(axes_dim)}) must equal head_dim ({head_dim})")
        if in_channels != out_channels:
            raise ValueError(
                f"Mage-Flow requires matching in_channels and out_channels, got {in_channels} and {out_channels}"
            )
        if depth_single_blocks != 0:
            raise ValueError(
                f"Mage-Flow single-stream blocks are not supported; checkpoint declares {depth_single_blocks}"
            )
        if patch_size != 1:
            raise ValueError(f"Mage-Flow currently requires patch_size=1, got {patch_size}")
        unsupported_config = {
            "mlp_ratio": (mlp_ratio, 4.0),
            "qkv_bias": (qkv_bias, True),
            "guidance_embed": (guidance_embed, False),
            "rope_type": (rope_type, "msrope"),
            "time_type": (time_type, "qwen_proj"),
            "double_block_type": (double_block_type, "double_stream"),
            "apply_text_rotary_emb": (apply_text_rotary_emb, False),
            "use_time_shift": (use_time_shift, False),
        }
        mismatches = {
            name: (actual, expected) for name, (actual, expected) in unsupported_config.items() if actual != expected
        }
        if mismatches:
            raise ValueError(f"Unsupported Mage-Flow transformer config values: {mismatches}")
        if not packing:
            raise ValueError("Mage-Flow checkpoint must declare packing=true")
        if schedule_mode != "z-image":
            raise ValueError(f"Unsupported Mage-Flow schedule_mode {schedule_mode!r}; expected 'z-image'")
        if static_shift != 6.0:
            raise ValueError(f"Unsupported Mage-Flow static_shift={static_shift}; expected 6.0")

        self.od_config = od_config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_dim = head_dim
        self.axes_dim = axes_dim
        self.patch_size = patch_size
        self.schedule_mode = schedule_mode
        self.static_shift = static_shift

        self.pos_embed = MageFlowEmbedRope(
            theta=theta,
            axes_dim=axes_dim,
            scale_rope=True,
        )
        self.img_in = nn.Linear(in_channels, hidden_size)
        self.txt_norm = RMSNorm(context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(context_in_dim, hidden_size)
        self.time_text_embed = MageFlowTimestepProjEmbeddings(hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                MageFlowDoubleStreamBlock(
                    dim=hidden_size,
                    num_attention_heads=num_heads,
                    attention_head_dim=head_dim,
                    prefix=f"transformer_blocks.{index}",
                )
                for index in range(depth)
            ]
        )
        # Registered after img_in/pos_embed so named_parameters() keeps
        # reporting the checkpoint's own names for the shared submodules.
        self.image_rope_prepare = MageFlowImageRopePrepare(self.img_in, self.pos_embed)
        self.norm_out = AdaLayerNormContinuous(
            hidden_size,
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )

    @staticmethod
    def _normalize_image_grids(
        image_grid_hw: list[tuple[int, int]] | list[tuple[int, int, int]] | tuple[int, int],
    ) -> list[tuple[int, int, int]]:
        grids = [image_grid_hw] if isinstance(image_grid_hw, tuple) else image_grid_hw
        normalized: list[tuple[int, int, int]] = []
        for grid in grids:
            if len(grid) == 2:
                height, width = grid
                normalized.append((1, int(height), int(width)))
            elif len(grid) == 3:
                frame, height, width = grid
                normalized.append((int(frame), int(height), int(width)))
            else:
                raise ValueError(f"invalid Mage-Flow image grid: {grid}")
        return normalized

    def _normalize_batched_image_grids(
        self,
        image_grid_hw: (
            list[tuple[int, int]]
            | list[tuple[int, int, int]]
            | list[list[tuple[int, int]]]
            | list[list[tuple[int, int, int]]]
            | tuple[int, int]
        ),
        batch_size: int,
    ) -> list[list[tuple[int, int, int]]]:
        if batch_size == 1 and (
            isinstance(image_grid_hw, tuple) or not image_grid_hw or isinstance(image_grid_hw[0], tuple)
        ):
            return [self._normalize_image_grids(image_grid_hw)]
        if not isinstance(image_grid_hw, list) or len(image_grid_hw) != batch_size:
            raise ValueError(
                f"batched Mage-Flow image_grid_hw must provide one grid list per request, got batch_size={batch_size}"
            )
        return [self._normalize_image_grids(grids) for grids in image_grid_hw]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_grid_hw: (
            list[tuple[int, int]]
            | list[tuple[int, int, int]]
            | list[list[tuple[int, int]]]
            | list[list[tuple[int, int, int]]]
            | tuple[int, int]
        ),
        image_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        if hidden_states.ndim != 3 or encoder_hidden_states.ndim != 3:
            raise ValueError("Mage-Flow hidden_states and encoder_hidden_states must be 3-D")
        if hidden_states.shape[0] != encoder_hidden_states.shape[0]:
            raise ValueError("Mage-Flow image and text batch sizes must match")
        batch_size = hidden_states.shape[0]
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if image_attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("image_attention_mask must match image token dimensions")
        if encoder_attention_mask is not None:
            if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError("encoder_attention_mask must match encoder token dimensions")
        else:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool,
                device=encoder_hidden_states.device,
            )

        grids_per_sample = self._normalize_batched_image_grids(image_grid_hw, batch_size)
        expected_tokens = [sum(frame * height * width for frame, height, width in grids) for grids in grids_per_sample]
        actual_tokens = image_attention_mask.to(torch.int64).sum(dim=1).tolist()
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"valid image token counts do not match image grids: got {actual_tokens}, expected {expected_tokens}"
            )

        hidden_states, image_rotary_emb = self.image_rope_prepare(
            hidden_states,
            image_attention_mask,
            grids_per_sample,
            hidden_states.shape[1],
        )
        # The _sp_plan hook fires on image_rope_prepare's outputs, so the image
        # stream is sharded from here on while the text stream stays whole.
        sequence_parallel = _sequence_parallel_active()
        if sequence_parallel and not (bool(image_attention_mask.all()) and bool(encoder_attention_mask.all())):
            # Batch padding, not shard padding: sharding splits the padded
            # sequence blindly, so a padded request would put real tokens and
            # filler on different ranks with no mask to tell them apart.
            # Requests of equal length shard cleanly.
            raise ValueError(
                "Mage-Flow sequence parallelism does not support padded token "
                "sequences: sharding would put real tokens and filler on "
                "different ranks with no mask to tell them apart. The pipeline "
                "already refuses --max-num-seqs > 1 under SP at startup and "
                "routes guided requests through sequential CFG rather than the "
                "packed path, so reaching this point means a caller built a "
                "padded batch directly."
            )
        if sequence_parallel:
            _warn_on_unmasked_sp_padding()

        encoder_hidden_states = _request_isolated_forward(
            self.txt_in,
            self.txt_norm(encoder_hidden_states),
            encoder_attention_mask,
        )
        if not sequence_parallel:
            # Under SP the mask spans the full sequence while hidden_states holds
            # only this rank's shard. The masks are all-valid there (checked
            # above), so skipping these multiplies is a no-op, not a shortcut.
            hidden_states = hidden_states * image_attention_mask[..., None]
        encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None]
        timestep = timestep.to(hidden_states.dtype)
        if timestep.shape != (batch_size,):
            raise ValueError(f"Mage-Flow timestep must have shape ({batch_size},), got {tuple(timestep.shape)}")
        temb = self.time_text_embed(timestep, hidden_states)

        # Every mask below is indexed against the full sequence, which no longer
        # matches the sharded image stream. They are all-valid under SP, so the
        # blocks run unmasked instead.
        block_image_mask = None if sequence_parallel else image_attention_mask

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                image_attention_mask=block_image_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        # proj_out carries the _sp_plan gather hook, so its output is whole
        # again and the trailing mask applies to the full sequence.
        output = _request_isolated_forward(
            self.proj_out,
            self.norm_out(hidden_states, temb),
            block_image_mask,
        )
        output = output * image_attention_mask[..., None]
        return (output,)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the transformer, fusing the checkpoint's split Q/K/V projections.

        ``AutoWeightsLoader`` delegates here for everything under ``transformer.``
        and re-adds that prefix to the names returned, so the loader's strict
        coverage check sees the pipeline-qualified parameter names.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            lookup_name, shard_id = resolve_mage_flow_stacked_name(name)
            param = params_dict.get(lookup_name)
            if param is None:
                raise KeyError(f"Unexpected Mage-Flow transformer weight: {name}")
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(lookup_name)
        return loaded_params
