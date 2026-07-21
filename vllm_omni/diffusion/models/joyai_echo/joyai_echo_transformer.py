"""JoyAI-Echo transformer.

Subclass of :class:`LTX2VideoTransformer3DModel` adding the JoyAI-Echo deltas:

- Top-level caption projections (188160 -> 4096 / 2048) replacing the
  ``LTX2TextConnectors`` external module used by stock LTX-2.3.
- Two integrated 8-layer ``LTX2ConnectorTransformer1d`` modules
  (``video_embeddings_connector`` / ``audio_embeddings_connector``) running
  after the caption projection, with 128 learnable registers each.
- A custom ``load_weights`` translating JoyAI's checkpoint key naming
  (``patchify_proj``, ``adaln_single``, ``q_norm``, ``transformer_1d_blocks``,
  ...) into the parent's diffusers naming so the parent's QKV-fusion + TP
  weight-loading logic at ``ltx2_transformer.py:2008-2074`` can be reused
  verbatim.

JoyAI's transformer block-level architecture is identical to LTX-2.3 (gated
attention + cross-attn AdaLN modulation + audio-to-video / video-to-audio
attention all already supported by the parent through the
``gated_attn`` / ``audio_gated_attn`` / ``cross_attn_mod`` / ``audio_cross_attn_mod``
constructor flags).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
from diffusers.pipelines.ltx2.connectors import LTX2ConnectorTransformer1d
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    AudioVisualModelOutput,
    LTX2VideoTransformer3DModel,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)


# Mapping from the JoyAI checkpoint config (`transformer/config.json`) keys to
# the parent ``LTX2VideoTransformer3DModel.__init__`` keyword names. JoyAI keys
# not listed here are routed by name match (or dropped via the ``_KNOWN_DROP``
# set if they are config-only flags consumed by the JoyAI subclass itself).
_JOYAI_TO_LTX2_CONFIG = {
    # Renamed
    "av_cross_ada_norm": "cross_attn_mod",
    "audio_av_cross_ada_norm": "audio_cross_attn_mod",  # not present in JoyAI; reserved
    "apply_gated_attention": "gated_attn",
    "audio_apply_gated_attention": "audio_gated_attn",  # not present in JoyAI; reserved
    "positional_embedding_theta": "rope_theta",
    "av_ca_timestep_scale_multiplier": "cross_attn_timestep_scale_multiplier",
    # ``positional_embedding_max_pos`` is split across (frames, height, width)
    # in JoyAI but the parent stores them as separate scalar args. We unpack
    # this in ``_translate_joyai_config`` instead of via this map.
}


# JoyAI config keys that are JoyAI-specific (consumed by ``JoyAIEchoTransformer``
# itself) or are equivalent / superseded by the parent model. They must NOT be
# forwarded to ``super().__init__``.
_KNOWN_DROP = frozenset(
    {
        "_class_name",
        # Per-modality cross-attn-adaLN. JoyAI only ships the single
        # ``cross_attention_adaln`` flag (see translator below).
        "cross_attention_adaln",
        # Drop diffusers/PixArt-style flags that have no parent counterpart and
        # are baked into the parent's architecture.
        "attention_type",
        "double_self_attention",
        "dropout",
        "norm_num_groups",
        "num_embeds_ada_norm",
        "num_vector_embeds",
        "only_cross_attention",
        "cross_attention_norm",
        "upcast_attention",
        "use_linear_projection",
        "standardization_norm",  # parent uses qk_norm only
        "positional_embedding_type",  # always rope
        "positional_embedding_max_pos",  # unpacked into base_num_frames/h/w
        "audio_positional_embedding_max_pos",  # unpacked into audio_pos_embed_max_pos
        "use_audio_video_cross_attention",  # always on for LTX-2.3
        "share_ff",
        "use_middle_indices_grid",
        "frequencies_precision",
        "caption_projection_first_linear",
        "caption_projection_second_linear",
        "caption_proj_input_norm",
        "caption_proj_before_connector",
        "causal_temporal_positioning",
        # Connector params -- consumed by the JoyAI subclass directly.
        "use_embeddings_connector",
        "connector_attention_head_dim",
        "connector_num_attention_heads",
        "connector_num_layers",
        "connector_positional_embedding_max_pos",
        "connector_num_learnable_registers",
        "connector_norm_output",
        "connector_apply_gated_attention",
        "connector_learnable_registers_std",
        "audio_connector_attention_head_dim",
        "audio_connector_num_attention_heads",
        "text_encoder_norm_type",
    }
)


def _translate_joyai_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a raw JoyAI ``transformer/config.json`` dict into:

    1. ``ltx2_kwargs`` -- forwarded to ``LTX2VideoTransformer3DModel.__init__``.
    2. ``joyai_kwargs`` -- consumed by ``JoyAIEchoTransformer.__init__`` itself.
    """
    ltx2_kwargs: dict[str, Any] = {}
    joyai_kwargs: dict[str, Any] = {}

    # 1. Connector params
    joyai_kwargs["use_embeddings_connector"] = cfg.get("use_embeddings_connector", True)
    joyai_kwargs["connector_num_attention_heads"] = cfg.get("connector_num_attention_heads", 32)
    joyai_kwargs["connector_attention_head_dim"] = cfg.get("connector_attention_head_dim", 128)
    joyai_kwargs["audio_connector_num_attention_heads"] = cfg.get("audio_connector_num_attention_heads", 32)
    joyai_kwargs["audio_connector_attention_head_dim"] = cfg.get("audio_connector_attention_head_dim", 64)
    joyai_kwargs["connector_num_layers"] = cfg.get("connector_num_layers", 8)
    joyai_kwargs["connector_num_learnable_registers"] = cfg.get("connector_num_learnable_registers", 128)
    joyai_kwargs["connector_apply_gated_attention"] = cfg.get("connector_apply_gated_attention", True)
    pos_embed = cfg.get("connector_positional_embedding_max_pos", [4096])
    joyai_kwargs["connector_pos_embed_max_pos"] = int(pos_embed[0]) if pos_embed else 4096

    # 2. Positional embedding split (JoyAI ships one combined list)
    pos = cfg.get("positional_embedding_max_pos", [20, 2048, 2048])
    if pos and len(pos) >= 3:
        ltx2_kwargs["pos_embed_max_pos"] = int(pos[0])
        ltx2_kwargs["base_height"] = int(pos[1])
        ltx2_kwargs["base_width"] = int(pos[2])

    audio_pos = cfg.get("audio_positional_embedding_max_pos", [20])
    if audio_pos:
        ltx2_kwargs["audio_pos_embed_max_pos"] = int(audio_pos[0])

    # 3. Renamed scalars
    if "av_cross_ada_norm" in cfg:
        ltx2_kwargs["cross_attn_mod"] = bool(cfg["av_cross_ada_norm"])
        ltx2_kwargs["audio_cross_attn_mod"] = bool(cfg["av_cross_ada_norm"])
    if "apply_gated_attention" in cfg:
        ltx2_kwargs["gated_attn"] = bool(cfg["apply_gated_attention"])
        ltx2_kwargs["audio_gated_attn"] = bool(cfg["apply_gated_attention"])
    if "positional_embedding_theta" in cfg:
        ltx2_kwargs["rope_theta"] = float(cfg["positional_embedding_theta"])
    if "av_ca_timestep_scale_multiplier" in cfg:
        ltx2_kwargs["cross_attn_timestep_scale_multiplier"] = int(cfg["av_ca_timestep_scale_multiplier"])

    # 4. Direct forward of remaining keys whose names match the parent
    for k, v in cfg.items():
        if k in _KNOWN_DROP:
            continue
        if k in {
            "av_cross_ada_norm",
            "apply_gated_attention",
            "positional_embedding_theta",
            "av_ca_timestep_scale_multiplier",
        }:
            continue
        ltx2_kwargs[k] = v

    # 5. Hardcoded LTX-2.3 settings JoyAI requires:
    ltx2_kwargs["use_prompt_embeddings"] = False  # JoyAI uses external caption projection

    return ltx2_kwargs, joyai_kwargs


# Parameter-name remap rules applied to incoming JoyAI checkpoint keys
# *before* delegating to ``super().load_weights``.
#
# ORDERING CONSTRAINT (enforced at module import below): longer / more
# specific source patterns MUST come first so they are not shadowed by
# shorter prefixes. For example, ``audio_adaln_single.`` must precede
# ``adaln_single.``, otherwise the latter would consume the former.
# A defensive ``sorted(..., key=len, reverse=True)`` check below catches
# accidental re-ordering at import time.
_KEY_RENAME_PREFIXES: tuple[tuple[str, str], ...] = (
    ("text_embedding_projection.video_aggregate_embed.", "text_embedding_projection_video."),
    ("text_embedding_projection.audio_aggregate_embed.", "text_embedding_projection_audio."),
    # Top-level module renames
    ("audio_patchify_proj.", "audio_proj_in."),
    ("patchify_proj.", "proj_in."),
    ("audio_adaln_single.", "audio_time_embed."),
    ("adaln_single.", "time_embed."),
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
    ("av_ca_video_scale_shift_adaln_single.", "av_cross_attn_video_scale_shift."),
    ("av_ca_audio_scale_shift_adaln_single.", "av_cross_attn_audio_scale_shift."),
    ("av_ca_a2v_gate_adaln_single.", "av_cross_attn_video_a2v_gate."),
    ("av_ca_v2a_gate_adaln_single.", "av_cross_attn_audio_v2a_gate."),
)


# Defensive check: any two src patterns where one is a strict prefix of the
# other must be ordered longer-first, otherwise the shorter pattern would
# consume keys the longer one was meant to handle.
def _validate_key_rename_ordering(prefixes: tuple[tuple[str, str], ...]) -> None:
    for i, (src_i, _) in enumerate(prefixes):
        for src_j, _ in prefixes[i + 1 :]:
            if src_j.startswith(src_i) and src_j != src_i:
                raise RuntimeError(
                    f"_KEY_RENAME_PREFIXES ordering violation: '{src_j}' must precede its prefix '{src_i}'"
                )


_validate_key_rename_ordering(_KEY_RENAME_PREFIXES)


def _rename_joyai_key(name: str) -> str:
    # Connector internals: transformer_1d_blocks -> transformer_blocks
    if "_embeddings_connector." in name:
        name = name.replace(".transformer_1d_blocks.", ".transformer_blocks.")

    # Q/K norm naming: ``q_norm`` / ``k_norm`` -> ``norm_q`` / ``norm_k``
    name = name.replace(".q_norm.", ".norm_q.").replace(".k_norm.", ".norm_k.")

    # Per-block scale_shift_table renames (a2v cross-attention modulation params)
    if name.endswith(".scale_shift_table_a2v_ca_video"):
        name = name[: -len(".scale_shift_table_a2v_ca_video")] + ".video_a2v_cross_attn_scale_shift_table"
    elif name.endswith(".scale_shift_table_a2v_ca_audio"):
        name = name[: -len(".scale_shift_table_a2v_ca_audio")] + ".audio_a2v_cross_attn_scale_shift_table"

    # Top-level prefix renames
    for src, dst in _KEY_RENAME_PREFIXES:
        if name.startswith(src):
            return dst + name[len(src) :]

    return name


class JoyAIEchoTransformer(LTX2VideoTransformer3DModel):
    """LTX-2.3-derived audio+video transformer for JoyAI-Echo.

    The forward pass differs from :class:`LTX2VideoTransformer3DModel` only
    in step (4): prompt embeddings are projected (188160 -> inner) and run
    through an integrated connector before being handed off to the standard
    LTX-2.3 transformer block stack.
    """

    def __init__(
        self,
        *,
        # Connector params (consumed here)
        use_embeddings_connector: bool = True,
        connector_num_attention_heads: int = 32,
        connector_attention_head_dim: int = 128,
        audio_connector_num_attention_heads: int = 32,
        audio_connector_attention_head_dim: int = 64,
        connector_num_layers: int = 8,
        connector_num_learnable_registers: int = 128,
        connector_apply_gated_attention: bool = True,
        connector_pos_embed_max_pos: int = 4096,
        # Other JoyAI-only flags (silently accepted)
        text_encoder_hidden_size: int = 188160,  # 3840 * 49
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **ltx2_kwargs: Any,
    ) -> None:
        # Force LTX-2.3 settings JoyAI inherits
        ltx2_kwargs.setdefault("use_prompt_embeddings", False)
        ltx2_kwargs.setdefault("gated_attn", True)
        ltx2_kwargs.setdefault("audio_gated_attn", True)
        ltx2_kwargs.setdefault("cross_attn_mod", True)
        ltx2_kwargs.setdefault("audio_cross_attn_mod", True)
        ltx2_kwargs.setdefault("qk_norm", "rms_norm")
        ltx2_kwargs.setdefault("rope_type", "split")

        super().__init__(
            quant_config=quant_config,
            **ltx2_kwargs,
        )

        if not use_embeddings_connector:
            raise NotImplementedError("JoyAIEchoTransformer requires use_embeddings_connector=True.")

        cross_attention_dim = ltx2_kwargs.get("cross_attention_dim", 4096)
        audio_cross_attention_dim = ltx2_kwargs.get("audio_cross_attention_dim", 2048)
        rope_type = ltx2_kwargs.get("rope_type", "split")

        # 1. Top-level caption projections (replace the parent's caption_projection)
        self.text_embedding_projection_video = nn.Linear(text_encoder_hidden_size, cross_attention_dim, bias=True)
        self.text_embedding_projection_audio = nn.Linear(text_encoder_hidden_size, audio_cross_attention_dim, bias=True)

        # 2. Integrated connectors (8-layer Transformer1D each, with gated attn)
        self.video_embeddings_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=connector_num_attention_heads,
            attention_head_dim=connector_attention_head_dim,
            num_layers=connector_num_layers,
            num_learnable_registers=connector_num_learnable_registers,
            rope_base_seq_len=connector_pos_embed_max_pos,
            rope_type=rope_type,
            gated_attention=connector_apply_gated_attention,
        )
        self.audio_embeddings_connector = LTX2ConnectorTransformer1d(
            num_attention_heads=audio_connector_num_attention_heads,
            attention_head_dim=audio_connector_attention_head_dim,
            num_layers=connector_num_layers,
            num_learnable_registers=connector_num_learnable_registers,
            rope_base_seq_len=connector_pos_embed_max_pos,
            rope_type=rope_type,
            gated_attention=connector_apply_gated_attention,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AudioVisualModelOutput | tuple:
        # 1. Project raw 188160-dim Gemma stack to the per-modality dims
        video_text = self.text_embedding_projection_video(encoder_hidden_states)
        audio_text = self.text_embedding_projection_audio(audio_encoder_hidden_states)

        # 2. Convert the 2D padding masks to 4D additive bias before feeding
        # them to the integrated connectors. This mirrors diffusers'
        # ``LTX2TextConnectors.forward`` (``diffusers/pipelines/ltx2/connectors.py:462-465``):
        # the ``LTX2ConnectorTransformer1d`` blocks pass the mask directly to
        # ``LTX2AttnProcessor2_0``, which forwards it to SDPA without dtype
        # conversion -- so a raw 2D long mask would crash.
        def _to_additive_bias(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
            if mask is None:
                return None
            m = (mask.to(torch.int64) - 1).to(dtype)
            m = m.reshape(m.shape[0], 1, -1, m.shape[-1])
            return m * torch.finfo(dtype).max

        add_video_mask = _to_additive_bias(encoder_attention_mask, video_text.dtype)
        add_audio_mask = _to_additive_bias(audio_encoder_attention_mask, audio_text.dtype)

        # 3. Run the integrated connectors. ``LTX2ConnectorTransformer1d``
        # uses learnable registers (``num_learnable_registers=128`` for JoyAI)
        # which **replace** padded positions with trainable embeddings
        # *inside* the connector (see
        # ``diffusers/pipelines/ltx2/connectors.py:298-316``). After the
        # connector, every output position carries valid content (registers
        # are valid trainable embeddings, not padding) and the returned
        # additive-bias mask is overwritten to all-zeros.
        video_text, _ = self.video_embeddings_connector(video_text, add_video_mask)
        audio_text, _ = self.audio_embeddings_connector(audio_text, add_audio_mask)

        # 4. Because every output position is valid after the registers swap,
        # we must NOT zero out any positions here and we must signal "no
        # padding" to the parent transformer body. We do this by forwarding
        # an all-ones 2D mask so the parent's bias conversion at
        # ``ltx2_transformer.py:1424-1431`` produces a zero additive bias
        # (i.e. attend everywhere).
        #
        # An earlier iteration of this code naively tried to derive the
        # binary mask from the (post-connector) additive-bias mask via
        # ``(add_mask < 1e-6)`` and/or zero out the embedding using the
        # *original* 2D mask -- both destroyed the learnable registers and
        # produced garbage outputs. The diffusers reference
        # ``LTX2TextConnectors.forward`` (``connectors.py:470-476``) does
        # the same ``< 1e-6`` check but it is a true no-op in the
        # registers-True case (additive bias is all-zeros so the binary
        # mask is all-ones, and the multiplication is identity). We
        # replicate the *intent* rather than the literal code, keeping the
        # registers path safe even if downstream consumers change.
        def _all_ones_like(mask: torch.Tensor | None) -> torch.Tensor | None:
            return None if mask is None else torch.ones_like(mask)

        # 5. Delegate to the LTX-2.3 transformer body. With
        # ``use_prompt_embeddings=False`` the parent does NOT re-project the
        # already-projected embeddings (see ``ltx2_transformer.py:1942-1949``).
        return super().forward(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=video_text,
            audio_encoder_hidden_states=audio_text,
            timestep=timestep,
            audio_timestep=audio_timestep,
            sigma=sigma,
            audio_sigma=audio_sigma,
            encoder_attention_mask=_all_ones_like(encoder_attention_mask),
            audio_encoder_attention_mask=_all_ones_like(audio_encoder_attention_mask),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Apply JoyAI -> diffusers key remapping then defer to the parent.

        Connector weights are routed through the default loader because they
        use separate Q/K/V (LTX2ConnectorTransformer1d) and would otherwise be
        falsely matched by the parent's ``.attn1.to_q``-style fusion rule at
        ``ltx2_transformer.py:2046``. Everything else is delegated to the
        parent so that QKV fusion + TP-shard fall-back paths run unchanged.
        """
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        deferred: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            renamed = _rename_joyai_key(name)
            if "_embeddings_connector." in renamed:
                if renamed not in params_dict:
                    logger.warning("Skipping connector weight %s -- not found.", renamed)
                    continue
                param = params_dict[renamed]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, tensor)
                else:
                    default_weight_loader(param, tensor)
                loaded.add(renamed)
            else:
                deferred.append((renamed, tensor))

        loaded |= super().load_weights(iter(deferred))
        return loaded
