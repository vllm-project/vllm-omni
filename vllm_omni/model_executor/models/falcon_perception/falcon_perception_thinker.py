# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception thinker: the autoregressive image+text stage.

Architecture notes that differ from a stock decoder-only LLM:

* **No vision tower.** Raw RGB is cut into ``spatial_patch_size`` squares and
  pushed through a single ``img_projector`` linear into the token stream, so
  image and text tokens share one transformer stack (early fusion).
* **Weightless norms.** The pre-attention, pre-FFN and QK norms are plain
  ``F.rms_norm`` — the checkpoint carries no per-layer norm parameters. Only
  the final ``norm`` is learned.
* **Attention sinks.** The reference scales the attention output by
  ``sigmoid(lse - sink)``, which is algebraically identical to adding
  ``exp(sink)`` to the softmax denominator — i.e. exactly vLLM's ``sinks=``.
* **Interleaved gated FFN.** ``w13`` packs gate/up as ``[g0, u0, g1, u1, ...]``
  with a ``sqrt(2)`` folded into the gate rows; the activation is ``relu()**2``.
* **Split RoPE.** See ``rope.py``. The learned 2-D frequencies are per
  *attention* head, so K cannot be shared across a GQA group and is expanded to
  ``num_heads`` before the rotation.
* **Geometry feedback loop.** ``<coord>`` / ``<size>`` tokens carry continuous
  values decoded from the *previous* step's hidden state and re-injected as
  input embeddings through Fourier encoders.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.falcon_perception.processing_falcon_perception import (
    FalconPerceptionDummyInputsBuilder,
    FalconPerceptionMultiModalProcessor,
    FalconPerceptionProcessingInfo,
)
from vllm_omni.model_executor.models.falcon_perception.profiling import mark, nvtx_range
from vllm_omni.model_executor.models.falcon_perception.rope import (
    apply_3d_rotary_emb,
    apply_golden_freqs_cis_to_visual_pos,
    compute_pos_hw,
    pack_image_grid_positions,
    precompute_freqs_cis,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _grid_hws_from_mm_features(mm_features: list | None) -> list[tuple[int, int]]:
    """Read ``(grid_h, grid_w)`` per image out of the runner's mm features.

    ``mm_features`` carries the multimodal processor's own fields, so this sees
    ``image_grid_hw`` for every image in the request whether or not the encoder
    cache will later short-circuit ``embed_multimodal``.
    """
    grids: list[tuple[int, int]] = []
    for feature in mm_features or ():
        item = getattr(feature, "data", None)
        if item is None:
            continue
        value = item.get_data().get("image_grid_hw")
        if value is None:
            continue
        for grid_h, grid_w in torch.as_tensor(value).reshape(-1, 2).tolist():
            grids.append((int(grid_h), int(grid_w)))
    return grids


class FalconPerceptionMLP(nn.Module):
    """Squared-ReLU gated FFN with a weightless pre-norm."""

    def __init__(self, hidden_size: int, intermediate_size: int, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(x, (x.size(-1),))
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        # relu(gate)**2 * up — the sqrt(2) scale is pre-baked into the gate rows.
        out, _ = self.down_proj(F.relu(gate).square() * up)
        return out


class FalconPerceptionAttention(nn.Module):
    """GQA with weightless QK-norm, split RoPE and attention sinks.

    ``Attention`` is configured with ``num_kv_heads == num_heads`` on purpose.
    The learned 2-D rotation is indexed per *query* head and its rows differ
    within a GQA group, so the two K heads of a group receive different
    rotations and cannot share a cache entry. The projection stays GQA-shaped
    (``total_num_kv_heads`` KV heads, so TP sharding and the checkpoint layout
    are unchanged); K/V are expanded right before the rotation.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        assert num_heads % tp_size == 0, f"num_heads={num_heads} must be divisible by TP size {tp_size}"
        self.num_heads = num_heads // tp_size
        # ``QKVParallelLinear`` replicates KV heads when num_kv_heads < tp_size,
        # so mirror its arithmetic rather than assuming clean division.
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # One learned sink logit per (local) attention head.
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        set_weight_attrs(self.sinks, {"weight_loader": self._load_sinks})

        self.attn = Attention(
            self.num_heads,
            head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

    def _load_sinks(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        shard = loaded_weight.narrow(0, tp_rank * self.num_heads, self.num_heads)
        param.data.copy_(shard)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor,
        spatial_enabled: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        x = F.rms_norm(hidden_states, (hidden_states.size(-1),))

        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))

        # Expand K/V to the query head count *before* RoPE — see class docstring.
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        q, k = apply_3d_rotary_emb(q, k, freqs_cis, freqs_cis_2d, spatial_enabled)

        attn_out = self.attn(
            q.reshape(num_tokens, -1),
            k.reshape(num_tokens, -1),
            v.reshape(num_tokens, -1),
        )
        out, _ = self.o_proj(attn_out)
        return out


class FalconPerceptionDecoderLayer(nn.Module):
    def __init__(self, *, config, cache_config=None, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = FalconPerceptionAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = FalconPerceptionMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor,
        spatial_enabled: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(hidden_states, freqs_cis, freqs_cis_2d, spatial_enabled)
        return hidden_states + self.mlp(hidden_states)


@support_torch_compile(
    dynamic_arg_dims={
        "positions": {1: "num_tokens"},
        "inputs_embeds": {0: "num_tokens"},
        "pos_hw": {0: "num_tokens"},
    }
)
class FalconPerceptionBackbone(nn.Module):
    """Token embeddings, the transformer stack, the final norm and the patch projector."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                FalconPerceptionDecoderLayer(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{i}"),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        patch_dim = config.temporal_patch_size * (config.spatial_patch_size**2) * config.channel_size
        self.img_projector = ReplicatedLinear(
            patch_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "img_projector"),
        )

        # Temporal RoPE table (first half of head_dim), rebuilt at init.
        rope_dim = config.head_dim // 2
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(rope_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )
        # Learned 2-D frequencies, loaded from the checkpoint and sharded by head.
        tp_size = get_tensor_model_parallel_world_size()
        num_local_heads = config.num_attention_heads // tp_size
        self.register_buffer(
            "freqs_cis_golden",
            torch.empty((num_local_heads, rope_dim // 2, 2), dtype=torch.float32),
            persistent=True,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pos_hw: torch.Tensor,
        spatial_enabled: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds

        # ``positions`` is the [3, N] M-RoPE buffer; all three rows hold the same
        # collapsed temporal position (see get_mrope_input_positions).
        pos_t = positions[0] if positions.dim() == 2 else positions
        freqs_cis = self.freqs_cis[pos_t]

        freqs_cis_2d = apply_golden_freqs_cis_to_visual_pos(self.freqs_cis_golden, pos_hw)

        for layer in self.layers:
            hidden_states = layer(hidden_states, freqs_cis, freqs_cis_2d, spatial_enabled)

        return self.norm(hidden_states)


class FourierEncoder(nn.Module):
    """Fourier-feature encoder used to turn continuous (x, y) / (h, w) into embeddings."""

    def __init__(self, in_dim: int, feat_dim: int, out_dim: int) -> None:
        super().__init__()
        self.embed = ReplicatedLinear(in_dim, feat_dim // 2, bias=False)
        self.transform = ReplicatedLinear(feat_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The decoders return normalised float32 coordinates while the weights
        # follow the model dtype (bf16), so align before the GEMM.
        x = x.to(self.embed.weight.dtype)
        f, _ = self.embed(x)
        f = 2 * math.pi * f
        f = torch.cat([f.cos(), f.sin()], dim=-1)
        out, _ = self.transform(f)
        return out


class BboxDecoder(nn.Module):
    """Two-layer squared-ReLU MLP producing per-axis bin logits."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w1 = ReplicatedLinear(in_dim, hidden_dim, bias=False)
        self.w2 = ReplicatedLinear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.w1(x.to(self.w1.weight.dtype))
        out, _ = self.w2(F.relu(h).square())
        return out


_COORD_STATE_KEY = "falcon_perception_geometry"
_COORD_REPEAT_THRESHOLD = 0.01
_MAX_COORD_ATTEMPTS = 100


def _select_coordinate_from_logits(
    logits: torch.Tensor,
    existing_coords: torch.Tensor | None,
    *,
    repeat_threshold: float = _COORD_REPEAT_THRESHOLD,
    max_attempts: int = _MAX_COORD_ATTEMPTS,
) -> torch.Tensor:
    """Select one non-repeated ``(x, y)`` coordinate from ``(2, bins)`` logits.

    Falcon Perception feeds the selected coordinate back into the next decode
    step. Re-emitting a previous centre therefore changes the future token
    trajectory, rather than merely duplicating a box in post-processing. Match
    the reference by masking both selected axis bins and retrying when a centre
    is within one percent of an earlier centre for this request.
    """
    if logits.ndim != 2 or logits.shape[0] != 2:
        raise ValueError(f"coordinate logits must have shape (2, bins), got {tuple(logits.shape)}")

    num_bins = logits.shape[-1]
    if num_bins <= 1:
        raise ValueError(f"coordinate decoder must expose at least two bins, got {num_bins}")

    history = None
    if isinstance(existing_coords, torch.Tensor) and existing_coords.numel() > 0:
        history = existing_coords.reshape(-1, 2).to(device=logits.device, dtype=torch.float32)

    working = logits.clone()
    selected = torch.zeros(2, device=logits.device, dtype=torch.float32)
    for _ in range(max_attempts):
        selected_bins = torch.argmax(working, dim=-1)
        selected = selected_bins.float() / (num_bins - 1)
        if history is None:
            break
        repeated = torch.all(torch.abs(history - selected) < repeat_threshold, dim=-1).any()
        if not bool(repeated.item()):
            break
        working[0, selected_bins[0]] = float("-inf")
        working[1, selected_bins[1]] = float("-inf")
    return selected.unsqueeze(0)


@MULTIMODAL_REGISTRY.register_processor(
    FalconPerceptionMultiModalProcessor,
    info=FalconPerceptionProcessingInfo,
    dummy_inputs=FalconPerceptionDummyInputsBuilder,
)
class FalconPerceptionThinker(nn.Module, CustomProcessMixin, SupportsMultiModal, SupportsMRoPE):
    """Image + text -> tokens, bounding boxes and per-instance segmentation features."""

    merge_by_field_config = True

    # The runner probes these to decide which hooks to call. ``preprocess``
    # re-injects the previous step's decoded geometry; ``postprocess`` stashes
    # the last hidden state each step. Both are per-request via the runner's
    # intermediate buffer — the model itself holds no per-request state.
    has_preprocess = True
    has_postprocess = True
    postprocess_uses_multimodal_outputs = False
    have_multimodal_outputs = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|image|>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = FalconPerceptionBackbone(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        if config.perception_heads:
            self.coord_encoder = FourierEncoder(2, config.coord_enc_dim, config.hidden_size)
            self.coord_decoder = BboxDecoder(config.hidden_size, config.coord_dec_dim, config.coord_out_dim)
            self.size_encoder = FourierEncoder(2, config.size_enc_dim, config.hidden_size)
            self.size_decoder = BboxDecoder(config.hidden_size, config.size_dec_dim, config.size_out_dim)

        # Lazily captured CUDA graph for the per-step geometry block; see
        # ``_geometry_embed_cached``.
        self._geom_graph: torch.cuda.CUDAGraph | None = None
        self._geom_sig: tuple | None = None
        self._geom_static: dict[str, torch.Tensor] = {}
        self._geom_graph_disabled = False

        self._patch_dim = config.temporal_patch_size * (config.spatial_patch_size**2) * config.channel_size
        self._structural_ids = config.image_structural_token_ids
        self._end_of_image_token_id = config.img_end_id

    def get_language_model(self) -> nn.Module:
        return self.model

    # ---------------------------------------------------------------- M-RoPE

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        """Collapse each image block to a single temporal position.

        The reference builds ``pos_t`` as a cumulative sum that does not
        increment on the 4 register tokens, the image patch tokens or
        ``<end_of_image>`` — only ``<image_cls>`` consumes an increment. The
        net effect is that ``[cls, reg1..4, patches..., eoi]`` all share the
        position assigned to ``cls``, and the following text resumes at +1.

        Returns ``[3, N]``. Row 0 is ``pos_t``, the only row the rotation reads.
        Rows 1 and 2 are dead weight for this model (``mrope_section`` is
        ``[head_dim // 2, 0, 0]``), so they carry the packed image patch grid —
        that is how ``forward`` learns the grid without depending on
        ``embed_multimodal`` having run. The runner rebuilds this buffer from
        ``mm_features`` for every request, including the ones whose image the
        multimodal encoder cache will serve from an earlier request.
        """
        start_id = self.config.image_cls_token_id
        end_id = self._end_of_image_token_id

        pos_t: list[int] = []
        in_image = False
        next_pos = 0

        for tok in input_tokens:
            if tok == start_id and not in_image:
                in_image = True
            pos_t.append(next_pos)
            if not in_image:
                next_pos += 1
            if tok == end_id and in_image:
                in_image = False
                next_pos += 1

        positions = torch.zeros((3, len(input_tokens)), dtype=torch.long)
        positions[0] = torch.tensor(pos_t, dtype=torch.long)
        # Row 0 only: rows 1/2 hold packed grid values, not positions.
        mrope_position_delta = int(positions[0].max().item()) + 1 - len(input_tokens)

        grids = _grid_hws_from_mm_features(mm_features)
        if grids:
            image_index = torch.tensor(
                [i for i, tok in enumerate(input_tokens) if tok == self.config.img_id],
                dtype=torch.long,
            )
            packed = torch.cat([pack_image_grid_positions(grid_h, grid_w) for grid_h, grid_w in grids])
            # A prompt update that leaves the literal ``<|image|>`` in place, or
            # drops one, would misalign every patch row against its grid. Fail
            # loudly here rather than rotate the image by a fraction of a patch.
            if packed.shape[0] != image_index.numel():
                raise ValueError(
                    f"Falcon Perception: prompt has {image_index.numel()} image patch tokens but grids "
                    f"{grids} describe {packed.shape[0]} patches. The placeholder run and the processor "
                    "grid must agree exactly."
                )
            positions[1:3, image_index] = packed.t()

        return positions, mrope_position_delta

    # --------------------------------------------------------- multimodal in

    def embed_multimodal(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor] | None = None,
        image_grid_hw: torch.Tensor | list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Patchify, project, and wrap each image with its structural tokens.

        Returns one tensor per image of shape ``(5 + n_patches + 1, hidden)``,
        matching the placeholder run inserted by the processor.
        """
        assert pixel_values is not None, "Falcon Perception requires pixel_values"
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = [pixel_values] if pixel_values.ndim == 3 else list(pixel_values)
        if isinstance(image_grid_hw, torch.Tensor):
            image_grid_hw = image_grid_hw.tolist()

        proj_weight = self.model.img_projector.weight
        device, dtype = proj_weight.device, proj_weight.dtype
        patch = self.config.spatial_patch_size

        # Annotating here is safe: the runner calls ``embed_multimodal`` from the
        # multimodal encoder path, outside the region vLLM compiles. It is also
        # the range that shows whether the encoder cache hit — on a hit this
        # method is not called at all, so the range is simply absent.
        with nvtx_range("fp/s0/embed_multimodal"):
            all_patches: list[torch.Tensor] = []
            patch_counts: list[int] = []
            for idx, pv in enumerate(pixel_values):
                pv = pv.to(device=device, dtype=dtype)
                h, w = int(pv.shape[0]), int(pv.shape[1])
                gh, gw = h // patch, w // patch
                if image_grid_hw is not None and idx < len(image_grid_hw):
                    gh, gw = int(image_grid_hw[idx][0]), int(image_grid_hw[idx][1])
                pv = pv[: gh * patch, : gw * patch]
                # (gh*ps, gw*ps, C) -> (gh*gw, ps*ps*C), h-major to match the
                # spatial position grid built in rope.compute_pos_hw.
                patches = pv.reshape(gh, patch, gw, patch, -1).permute(0, 2, 1, 3, 4).reshape(gh * gw, self._patch_dim)
                all_patches.append(patches)
                patch_counts.append(gh * gw)

            projected, _ = self.model.img_projector(torch.cat(all_patches, dim=0))
            per_image = projected.split(patch_counts)

            structural_ids = torch.tensor(
                [*self._structural_ids, self._end_of_image_token_id],
                device=device,
                dtype=torch.long,
            )
            structural = self.model.embed_input_ids(structural_ids).to(dtype=dtype)
            n_prefix = len(self._structural_ids)

            return tuple(torch.cat([structural[:n_prefix], img, structural[n_prefix:]], dim=0) for img in per_image)

    # -------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # The patch grid is read back per token from the M-RoPE positions
        # buffer, which the runner rebuilds from mm_features on every step.
        # Nothing here may depend on embed_multimodal having run: on a
        # multimodal encoder cache hit (same image, different query — the
        # workload this model is served for) vLLM returns the stored embedding
        # and skips that call entirely, so a grid derived there is unavailable
        # by construction. Losing it leaves pos_hw None, which silently drops
        # the golden 2-D RoPE for every image token and degenerates the output.
        pos_hw = None
        if input_ids is not None:
            pos_hw = compute_pos_hw(input_ids, positions, image_token_id=self.config.img_id)

        # Keep multimodal position reconstruction eager, but give the compiled
        # backbone one stable tensor signature in both prefill and decode. A
        # tensor gate preserves the reference's no-spatial-RoPE decode path;
        # merely rotating by identity would still round-trip bf16 values
        # through float32 and can move mask boundaries.
        if pos_hw is None:
            pos_hw = torch.zeros((positions.shape[-1], 2), device=positions.device, dtype=torch.float32)
            spatial_enabled = torch.zeros((1,), device=positions.device, dtype=torch.bool)
        else:
            spatial_enabled = torch.ones((1,), device=positions.device, dtype=torch.bool)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Falcon Perception requires input_ids or inputs_embeds")
            inputs_embeds = self.model.embed_input_ids(input_ids)

        return self.model(positions, inputs_embeds, pos_hw, spatial_enabled)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def make_omni_output(self, model_output: torch.Tensor, **kwargs: Any) -> OmniOutput:
        """Emit per-row geometry alongside the hidden states.

        The decoders run on every row rather than only on ``<coord>`` /
        ``<size>`` rows: which rows matter is decided by the *sampled* token,
        which is not known until after this call. Emitting unconditionally and
        letting the consumer mask by token id keeps this stateless and
        batch-safe, and mirrors the reference, which also runs both heads every
        step to stay sync-free. The rows accumulate across steps, so the
        consumer sees one ``(xy, hw)`` pair per generated token, index-aligned
        with the token stream.
        """
        if not self.config.perception_heads or model_output.numel() == 0:
            return OmniOutput(text_hidden_states=model_output)

        hidden = model_output.reshape(-1, model_output.shape[-1])
        # Under ``hidden_states`` rather than a "geometry" section of its own:
        # the stage payload only transports the schema's sections
        # (data_entry_keys.py), so a custom top-level section is dropped.
        geometry = {
            "box_xy": self._decode_coords(hidden),
            "box_hw": self._decode_sizes(hidden),
        }

        # ``box_xy`` above is hidden-row aligned and therefore emitted before
        # sampling reveals whether a row produced ``<coord>``. The request's
        # preprocess hook selects and de-duplicates that coordinate one step
        # later. Ship those accepted coordinates as per-request deltas so the
        # stage bridge can use exactly the values that were fed back into the
        # model, rather than the pre-dedup argmax rows.
        runtime_infos = kwargs.get("runtime_additional_information") or []
        if isinstance(runtime_infos, list):
            selected_xy: list[torch.Tensor] = []
            for info in runtime_infos:
                state = info.get(_COORD_STATE_KEY, {}) if isinstance(info, dict) else {}
                active = state.get("selected_xy_active") if isinstance(state, dict) else None
                xy = state.get("selected_xy") if isinstance(state, dict) else None
                is_active = isinstance(active, torch.Tensor) and active.numel() > 0 and bool(active.reshape(-1)[-1])
                if is_active and isinstance(xy, torch.Tensor) and xy.numel() == 2:
                    selected_xy.append(xy.reshape(1, 2))
                else:
                    selected_xy.append(torch.zeros((0, 2), dtype=torch.float32))
            geometry["selected_box_xy"] = selected_xy
        return OmniOutput(text_hidden_states=model_output, multimodal_outputs={"hidden_states": geometry})

    # ------------------------------------------------- geometry feedback loop

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        """Stash this step's last hidden state for the next step's preprocess.

        Same contract as the in-tree talker stages: the runner writes the
        returned dict into its per-request intermediate buffer, so nothing is
        kept on the module and continuous batching stays safe.
        """
        # Runner-called, once per decode step, outside the compiled model.
        with nvtx_range("fp/s0/postprocess"):
            if hidden_states.numel() == 0:
                return {}
            return {"hidden_states": {"last": hidden_states[-1].detach().contiguous()}}

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Re-inject geometry decoded from the previous step's hidden state.

        The reference decodes ``(x, y)`` / ``(h, w)`` from the hidden state that
        *produced* the ``<coord>`` / ``<size>`` token, then feeds those values
        back as that token's input embedding one step later. Here the previous
        hidden state arrives through ``info_dict['hidden_states']['last']``.
        """
        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional.items():
                merged.setdefault(k, v)
            info_dict = merged

        if input_embeds is None:
            input_embeds = self.model.embed_input_ids(input_ids)

        if not self.config.perception_heads:
            return input_ids, input_embeds, {}

        last_hidden = (info_dict.get("hidden_states") or {}).get("last")
        is_decode = isinstance(last_hidden, torch.Tensor) and input_ids.numel() > 0

        # Step boundary marker. ``preprocess`` is called by the runner exactly
        # once per model step, before the forward, and outside every compiled
        # region — so this is the one safe place to delimit steps from inside the
        # model. It is a *mark*, not a range: a range would have to be popped in
        # ``postprocess``, and a push/pop split across two callbacks corrupts the
        # whole NVTX stack the first time one of them is skipped.
        # Measure a step as the gap between consecutive marks; the phase label
        # tells you which side of prefill you are on.
        mark(f"fp/s0/step[{'decode' if is_decode else 'prefill'}]")

        if not is_decode:
            # Prefill (or the very first decode step): the prompt carries no
            # <coord>/<size> tokens, so the embeddings pass through unchanged.
            return input_ids, input_embeds, {}

        # Only the newest token can be a geometry token during decode.
        # Runner-called per decode step and outside the compiled model, so the
        # range is safe. This is the loop the phase profile put at 33 ms/step,
        # i.e. most of a dense request — the reason it is annotated separately
        # from the backbone forward it precedes.
        with nvtx_range("fp/s0/preprocess/geometry"):
            token = input_ids[-1]
            h = last_hidden.to(device=input_embeds.device, dtype=input_embeds.dtype).reshape(1, -1)

            is_coord = token == self.config.coord_token_id
            is_size = token == self.config.size_token_id
            token_id = int(token.item())
            state = info_dict.get(_COORD_STATE_KEY)
            if not isinstance(state, dict):
                state = {}

            update: dict[str, Any]
            if token_id == self.config.coord_token_id:
                selected_xy = self._select_coord(h, state.get("coord_history"))
                new_embed = self.coord_encoder(selected_xy).to(input_embeds.dtype)
                previous = state.get("coord_history")
                if isinstance(previous, torch.Tensor) and previous.numel() > 0:
                    previous = previous.reshape(-1, 2).to(device=selected_xy.device, dtype=selected_xy.dtype)
                    coord_history = torch.cat((previous, selected_xy), dim=0)
                else:
                    coord_history = selected_xy
                update = {
                    _COORD_STATE_KEY: {
                        "coord_history": coord_history,
                        "selected_xy": selected_xy,
                        "selected_xy_active": torch.ones(1, device=selected_xy.device, dtype=torch.bool),
                    }
                }
            else:
                new_embed = self._geometry_embed_cached(h, is_coord, is_size, input_embeds[-1:])
                # Clear the delta marker every non-coordinate step so a prior
                # accepted centre is emitted exactly once by make_omni_output.
                update = {
                    _COORD_STATE_KEY: {
                        "selected_xy_active": torch.zeros(1, device=input_embeds.device, dtype=torch.bool),
                    }
                }
            input_embeds = torch.cat([input_embeds[:-1], new_embed], dim=0)

        # The update contains only request-scoped coordinate feedback state.
        # The mask head still takes its row-aligned geometry from
        # ``make_omni_output`` and its ``<seg>`` rows from the latent trajectory;
        # keeping geometry out of this dict avoids the five CPU transfers an
        # earlier implementation performed on every decode step.
        return input_ids, input_embeds, update

    def _geometry_embed(
        self,
        h: torch.Tensor,
        is_coord: torch.Tensor,
        is_size: torch.Tensor,
        last_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Decode geometry from the previous hidden state, re-encode it as an embedding.

        Both heads run unconditionally and the result is selected with
        ``torch.where``: branching on a GPU token id would force a device sync
        on every decode step.
        """
        xy = self._decode_coords(h)
        hw = self._decode_sizes(h)
        # ``torch.where`` is out-of-place, so the first call already allocates a
        # fresh tensor and leaves ``last_embed`` untouched: no defensive clone.
        new_embed = torch.where(is_coord, self.coord_encoder(xy).to(last_embed.dtype), last_embed)
        new_embed = torch.where(is_size, self.size_encoder(hw).to(new_embed.dtype), new_embed)
        return new_embed

    def _geometry_embed_cached(
        self,
        h: torch.Tensor,
        is_coord: torch.Tensor,
        is_size: torch.Tensor,
        last_embed: torch.Tensor,
    ) -> torch.Tensor:
        """``_geometry_embed`` replayed from a CUDA graph, when that is possible.
        This is because the block is launch-bound, not compute-bound
        A graph is the right tool rather than ``torch.compile``
        """
        eager_ok = (
            self._geom_graph_disabled
            or h.device.type != "cuda"
            # Nested capture is illegal, and attempting one inside vLLM's own
            # capture would corrupt *its* graph, not just fail ours.
            or torch.cuda.is_current_stream_capturing()
        )
        if eager_ok:
            return self._geometry_embed(h, is_coord, is_size, last_embed)

        sig = (tuple(h.shape), h.dtype, tuple(last_embed.shape), last_embed.dtype)
        if self._geom_graph is None or self._geom_sig != sig:
            if not self._capture_geometry_graph(h, is_coord, is_size, last_embed, sig):
                return self._geometry_embed(h, is_coord, is_size, last_embed)

        static = self._geom_static
        static["h"].copy_(h)
        static["last"].copy_(last_embed)
        static["is_coord"].copy_(is_coord)
        static["is_size"].copy_(is_size)
        self._geom_graph.replay()
        # Cloned because the graph writes into this buffer again next step.
        return static["out"].clone()

    def _capture_geometry_graph(
        self,
        h: torch.Tensor,
        is_coord: torch.Tensor,
        is_size: torch.Tensor,
        last_embed: torch.Tensor,
        sig: tuple,
    ) -> bool:
        """Capture ``_geometry_embed`` into a CUDA graph. ``False`` if unavailable."""
        try:
            static = {
                "h": h.clone(),
                "last": last_embed.clone(),
                "is_coord": is_coord.clone(),
                "is_size": is_size.clone(),
            }
            # Warm up on a side stream first: the first call allocates
            # workspaces and picks cuBLAS kernels, none of which may happen
            # during capture.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._geometry_embed(static["h"], static["is_coord"], static["is_size"], static["last"])
            torch.cuda.current_stream().wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static["out"] = self._geometry_embed(static["h"], static["is_coord"], static["is_size"], static["last"])
        except (RuntimeError, torch.cuda.CudaError) as exc:
            logger.warning(
                "falcon_perception: could not capture the geometry CUDA graph (%s); "
                "falling back to eager for the rest of this process",
                exc,
            )
            self._geom_graph_disabled = True
            return False

        self._geom_graph = graph
        self._geom_static = static
        self._geom_sig = sig
        return True

    def _decode_coords(self, hidden: torch.Tensor) -> torch.Tensor:
        """(N, D) hidden -> (N, 2) normalised centre coordinates in [0, 1]."""
        logits = self.coord_decoder(hidden).view(hidden.shape[0], 2, -1)
        num_bins = logits.shape[-1]
        return torch.argmax(logits, dim=-1).float() / (num_bins - 1)

    def _select_coord(self, hidden: torch.Tensor, existing_coords: torch.Tensor | None) -> torch.Tensor:
        logits = self.coord_decoder(hidden).view(hidden.shape[0], 2, -1)
        if logits.shape[0] != 1:
            raise ValueError(f"coordinate feedback expects one request row, got {logits.shape[0]}")
        return _select_coordinate_from_logits(logits[0], existing_coords)

    def _decode_sizes(self, hidden: torch.Tensor) -> torch.Tensor:
        """(N, D) hidden -> (N, 2) normalised (h, w) sizes, de-bucketed in log2 space."""
        logits = self.size_decoder(hidden).view(hidden.shape[0], 2, -1)
        num_bins = logits.shape[-1]
        pred = torch.argmax(logits, dim=-1).float() / (num_bins - 1)
        min_size = math.log2(1.0 / num_bins)
        pred = pred * (0.0 - min_size) + min_size
        return torch.pow(2.0, pred)

    # -------------------------------------------------------- weight loading

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        head_dim = self.config.head_dim
        n_heads = self.config.num_attention_heads
        n_kv = self.config.num_key_value_heads
        q_dim, kv_dim = n_heads * head_dim, n_kv * head_dim

        for name, weight in weights:
            if name.startswith("model."):
                name = name[len("model.") :]

            if name == "freqs_cis_golden":
                rows = n_heads // tp_size
                shard = weight[:n_heads].narrow(0, tp_rank * rows, rows)
                self.model.freqs_cis_golden.copy_(shard.to(self.model.freqs_cis_golden.dtype))
                loaded.add("model.freqs_cis_golden")
                continue

            if name.endswith("attention.wqkv.weight"):
                layer = name.split(".")[1]
                target = f"model.layers.{layer}.self_attn.qkv_proj.weight"
                param = params_dict[target]
                loader = param.weight_loader
                q, k, v = weight.split([q_dim, kv_dim, kv_dim], dim=0)
                loader(param, q, "q")
                loader(param, k, "k")
                loader(param, v, "v")
                loaded.add(target)
                continue

            if name.endswith("feed_forward.w13.weight"):
                layer = name.split(".")[1]
                target = f"model.layers.{layer}.mlp.gate_up_proj.weight"
                param = params_dict[target]
                loader = param.weight_loader
                # Checkpoint packs gate/up interleaved as [g0, u0, g1, u1, ...];
                # MergedColumnParallelLinear wants them as two contiguous shards.
                pairs = weight.view(-1, 2, weight.shape[-1])
                loader(param, pairs[:, 0, :].contiguous(), 0)
                loader(param, pairs[:, 1, :].contiguous(), 1)
                loaded.add(target)
                continue

            target = self._map_weight_name(name)
            if target is None or target not in params_dict:
                continue
            param = params_dict[target]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded.add(target)

        # ``freqs_cis_golden`` is a *buffer*, and vLLM's missing-weight guard
        # only covers ``named_parameters()``
        # (model_loader/default_loader.py: ``weights_to_load = {name for name, _
        # in model.named_parameters()}``). So a checkpoint that lacks this key,
        # or an upstream rename, would leave ``torch.empty`` memory serving as
        # the learned 2-D rotation: no exception, and masks that look plausible
        # but are wrong. That is the same silent-corruption shape as the
        # encoder-cache bug (18 masks vs 121), so fail loudly instead.
        if "model.freqs_cis_golden" not in loaded:
            raise ValueError(
                "Falcon Perception: checkpoint has no 'freqs_cis_golden'. It is a buffer, "
                "not a parameter, so vLLM's missing-weight check cannot catch it and the "
                "golden 2-D RoPE would silently run on uninitialized memory."
            )

        return loaded

    def _map_weight_name(self, name: str) -> str | None:
        """Map a reference checkpoint key onto this module's parameter names."""
        direct = {
            "tok_embeddings.weight": "model.embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "norm.weight": "model.norm.weight",
            "img_projector.weight": "model.img_projector.weight",
        }
        if name in direct:
            return direct[name]

        if name.startswith("layers."):
            parts = name.split(".")
            layer = parts[1]
            rest = ".".join(parts[2:])
            rest = rest.replace("attention.wo.", "self_attn.o_proj.")
            rest = rest.replace("attention.sinks", "self_attn.sinks")
            rest = rest.replace("feed_forward.w2.", "mlp.down_proj.")
            return f"model.layers.{layer}.{rest}"

        # Perception heads live on this module under their reference names.
        if name.startswith(("coord_encoder.", "coord_decoder.", "size_encoder.", "size_decoder.")):
            return name

        return None


__all__: Sequence[str] = ["FalconPerceptionThinker"]
