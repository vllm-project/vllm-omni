# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for Gepard-1.0, a single-stage autoregressive TTS.

Text tokens -> one 32-code FSQ audio frame per step -> NeMo NanoCodec ->
waveform. The backbone is a vLLM-native ``Qwen3_5ForCausalLM``; the codebook
heads, binary stop head and voice-clone ref_compressor are Gepard additions.

Parses the model's ``gepard_config.json`` sidecar, which nests the LM
parameters under ``backbone_config`` and carries audio-head cardinalities,
special tokens, codec settings and the short-text repetition layout.

The backbone uses standard 1D RoPE. The sidecar's ``backbone_config`` carries
vestigial mrope keys from a template, so they are stripped when building the
text config to keep ``uses_mrope()`` False — same idiom as ``qwen3_tts``.
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

# The backbone ships as a standard HF ``qwen3_5_text`` config; the sidecar
# nests the same fields under ``backbone_config``.
_BACKBONE_MODEL_TYPE = "qwen3_5_text"

# rope_parameters keys that would (incorrectly) flip vLLM's uses_mrope() to
# True.  Stripped from the backbone config — see module docstring / RoPE note.
_MROPE_KEYS = ("mrope_section", "mrope_interleaved")


class GepardConfig(PretrainedConfig):
    """Configuration for the Gepard-1.0 native-AR TTS model.

    Args mirror ``gepard_config.json``.  Defaults match the trained
    ``nineninesix/gepard-1.0`` checkpoint so an instance built with no
    arguments (e.g. dummy/profiling loads) is still self-consistent.
    """

    model_type = "gepard"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # -- LM backbone (Qwen3.5 text config), nested in the sidecar --
        backbone_config: dict | None = None,
        # -- Audio frame: 32 FSQ channels sampled per step --
        # gepard_config.json ``audio_heads`` = {level_audio_0: 8, ...}; the
        # per-channel cardinalities cycle [8, 7, 6, 6] across 8 groups.
        audio_heads: dict | None = None,
        audio_embed_dim: int = 32,
        # -- Special tokens (gepard_config.json ``special_tokens``) --
        special_tokens: dict | None = None,
        # -- Short-text repetition (gepard_config.json ``text_repetition``) --
        text_repetition: dict | None = None,
        # -- NeMo NanoCodec (gepard_config.json ``codec``) --
        codec: dict | None = None,
        # -- Voice cloning (parsed + carried; PR1 uses null_prefix only) --
        voice_cloning: dict | None = None,
        # -- Stop head --
        stop_threshold: float = 0.5,
        stop_loss_weight: float = 2.0,  # training-only, kept for fidelity
        stop_pos_weight: float = 25.0,  # training-only, kept for fidelity
        # -- Sampling temperature. Applies to head0 through vLLM's
        #    SamplingParams and to the 31 side-channel heads in-model. The
        #    sidecar's ``top_p`` is deliberately not surfaced: the 32 heads are
        #    sampled by Gumbel-max, which has no nucleus step, so a top_p
        #    attribute here would read as a knob that does nothing. --
        temperature: float = 0.3,
        **kwargs,
    ):
        # WARNING: every attribute below is assigned before super().__init__()
        # at the bottom — transformers 5.x runs validators inside
        # PretrainedConfig.__init__ that call get_text_config().
        self.backbone_config = self._normalize_backbone(backbone_config)

        # ---- Audio heads -------------------------------------------------
        # Order level_audio_0..N into a flat list of per-channel cardinalities.
        self.audio_head_levels = self._parse_audio_heads(audio_heads)
        self.num_audio_heads = len(self.audio_head_levels)  # 32
        self.audio_embed_dim = audio_embed_dim  # 32
        # head0 is "the token" vLLM samples through its sampler; its vocab is
        # the first channel's cardinality. STOP is a synthetic sentinel one
        # past head0's valid range (0..head0_vocab-1).
        self.head0_vocab_size = self.audio_head_levels[0] if self.audio_head_levels else 8
        self.stop_token = self.head0_vocab_size  # 8

        # ---- Special tokens ---------------------------------------------
        st = special_tokens or {}
        self.start_of_text = st.get("start_of_text", 248073)
        self.end_of_text = st.get("end_of_text", 248074)
        self.start_of_speech = st.get("start_of_speech", 248070)
        self.end_of_speech = st.get("end_of_speech", 248071)
        self.tts_pad = st.get("tts_pad", 248076)
        # SPEAKER_TOKEN_BASE placeholder slots begin at tokeniser_length.
        self.tokeniser_length = st.get("tokeniser_length", 248077)
        self.speaker_token_base = self.tokeniser_length

        # ---- Short-text repetition (prompt layout) -----------------------
        # A short text is repeated so its text region carries enough token
        # mass; ``prompt.py`` reads these. They must match the training
        # layout, so they come from the checkpoint rather than a literal.
        # ``mixed_keep_prob`` / ``seed`` in the same block are training-only
        # (the inference repeat count is deterministic) and not parsed.
        tr = text_repetition or {}
        self.text_repetition_enabled = tr.get("enabled", True)
        self.text_repetition_target_tokens = tr.get("target_text_tokens", 16)
        self.text_repetition_apply_below = tr.get("apply_below", 13)
        self.text_repetition_max_repeats = tr.get("max_repeats", 8)

        # ---- Codec (NeMo NanoCodec, runs OUTSIDE vLLM) -------------------
        cc = codec or {}
        self.codec_id = cc.get("codec_id", "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps")
        self.codec_sample_rate = cc.get("sample_rate", 22050)
        self.codec_frame_rate_hz = cc.get("frame_rate_hz", 21.5)
        # Per-GROUP FSQ levels (the [8,7,6,6] the 32 channels cycle through);
        # num_codec_groups * len(fsq_levels) == num_audio_heads (8 * 4 == 32).
        self.fsq_levels = cc.get("fsq_levels", [8, 7, 6, 6])
        self.num_codec_groups = cc.get("num_layers", 8)
        self.codec_do_unfold = cc.get("do_unfold", True)

        # ---- Stop head ---------------------------------------------------
        self.stop_threshold = stop_threshold
        self.stop_loss_weight = stop_loss_weight
        self.stop_pos_weight = stop_pos_weight

        # ---- Sampling / generation --------------------------------------
        self.temperature = temperature

        # ---- Voice cloning (carried for the follow-up cloning PR) --------
        vc = voice_cloning or {}
        self.voice_cloning_enabled = vc.get("enabled", True)
        comp = vc.get("compressor", {}) or {}
        self.num_speaker_prefix = comp.get("num_queries", 8)
        self.ref_compressor_num_blocks = comp.get("num_layers", 2)
        self.ref_compressor_num_heads = comp.get("num_heads", 8)
        self.ref_compressor_d_model = comp.get("d_model", 1024)
        self.ref_compressor_ffn_mult = comp.get("ffn_hidden_size_multiplier", 4)

        # Last on purpose — triggers transformers-5.x validators that call
        # get_text_config(); every attribute they touch must already exist.
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    #  Rebuild from a checkpoint under hf_overrides routing
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(
        cls,
        model: str,
        backbone_config: dict | None = None,
        revision: str | None = None,
    ) -> GepardConfig:
        """Build the full config for a checkpoint whose ``config.json``
        self-identifies as the bare backbone (``qwen3_5_text``).

        Deploy-yaml ``hf_overrides`` patches ``architectures``/``model_type``
        onto the *loaded* ``Qwen3_5TextConfig`` instance — it cannot change
        its class — so the talker receives a backbone config, not a
        ``GepardConfig``.  This rebuilds the real thing: audio/special/codec
        fields from the ``gepard_config.json`` sidecar, backbone fields from
        the loaded checkpoint config (authoritative — it is what vLLM's
        ``hf_text_config`` already runs the backbone on).

        ``revision`` must be the one the weights were loaded from: a revision
        that moves the audio-head cardinalities or the special tokens moves the
        prompt layout and the STOP sentinel with them.
        """
        sidecar: dict = {}
        try:
            # vLLM util: resolves both local snapshot dirs and hub ids.
            from vllm.transformers_utils.config import get_hf_file_to_dict

            sidecar = dict(get_hf_file_to_dict("gepard_config.json", model, revision=revision) or {})
        except (OSError, ValueError) as e:
            # A missing/unreadable/malformed sidecar is expected — the defaults
            # match the trained checkpoint. Anything else (an import error, a
            # signature change in the vLLM util) is a real bug and must surface.
            logger.warning(
                "GepardConfig: could not read gepard_config.json from %s (%s: %s); "
                "audio fields fall back to trained-checkpoint defaults.",
                model,
                type(e).__name__,
                e,
            )
        if backbone_config:
            bb = dict(backbone_config)
            # The loaded config carries the hf_overrides-patched identity
            # (model_type="gepard"); the backbone must stay qwen3_5_text or
            # get_text_config() would recurse into GepardConfig.
            bb["model_type"] = _BACKBONE_MODEL_TYPE
            bb.pop("architectures", None)
            sidecar["backbone_config"] = bb
        return cls(**sidecar)

    # ------------------------------------------------------------------ #
    #  Parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_audio_heads(audio_heads: dict | None) -> list[int]:
        """``{level_audio_0: 8, level_audio_1: 7, ...}`` -> ``[8, 7, ...]``.

        Sorted by the numeric suffix so channel order is deterministic and
        independent of dict insertion order.  Falls back to the trained
        [8,7,6,6]*8 layout when no dict is provided (dummy loads).
        """
        if not audio_heads:
            return [8, 7, 6, 6] * 8
        return [audio_heads[k] for k in sorted(audio_heads, key=lambda s: int(str(s).rsplit("_", 1)[-1]))]

    @classmethod
    def _normalize_backbone(cls, backbone_config: dict | None) -> dict:
        """Return the backbone dict with vestigial mRoPE fields removed.

        The sidecar's ``backbone_config.rope_parameters`` carries
        ``mrope_interleaved``/``mrope_section`` template leftovers that do
        NOT reflect the trained checkpoint (its ``config.json`` is plain
        ``rope_type: default``).  Leaving them in would flip vLLM's
        ``uses_mrope()`` to True and wire ``positions`` wrongly.  Strip them
        so the backbone runs standard 1D RoPE.
        """
        bb = dict(backbone_config or {})
        rope = bb.get("rope_parameters")
        if isinstance(rope, dict) and any(k in rope for k in _MROPE_KEYS):
            rope = {k: v for k, v in rope.items() if k not in _MROPE_KEYS}
            bb["rope_parameters"] = rope
            logger.info(
                "GepardConfig: stripped vestigial mRoPE keys from backbone rope_parameters "
                "(model uses standard 1D default RoPE)."
            )
        return bb

    # ------------------------------------------------------------------ #
    #  Text config for vLLM
    # ------------------------------------------------------------------ #

    def get_text_config(self, **kwargs) -> PretrainedConfig:
        """Return the Qwen3.5 backbone config as the text config.

        vLLM's ``Qwen3_5ForCausalLM`` reads ``hidden_size`` /
        ``num_attention_heads`` / ``rope_parameters`` / etc. off this.
        Built (and cached) from the mRoPE-stripped ``backbone_config``.
        """
        cached = getattr(self, "_text_config", None)
        if cached is not None:
            return cached

        bb = dict(self.backbone_config)
        # pop, don't setdefault: for_model()'s first positional is literally
        # named ``model_type`` — leaving the key in **bb passes it twice
        # (TypeError: got multiple values for argument 'model_type').
        model_type = bb.pop("model_type", _BACKBONE_MODEL_TYPE)
        try:
            text_config = AutoConfig.for_model(model_type, **bb)
        except (KeyError, ValueError) as e:
            # Only "this transformers build does not know qwen3_5_text" is a
            # legitimate fallback. A TypeError here means the kwargs are wrong
            # (see the model_type pop above) — that must NOT degrade silently
            # into an empty config, so it is deliberately not caught.
            text_config = PretrainedConfig(**bb)
            logger.warning(
                "GepardConfig: AutoConfig.for_model(%s) failed (%s: %s); using a plain "
                "PretrainedConfig for the backbone. Verify vLLM still resolves "
                "the Qwen3.5 model.",
                model_type,
                type(e).__name__,
                e,
            )
        self._text_config = text_config
        return text_config


AutoConfig.register("gepard", GepardConfig)

__all__ = ["GepardConfig"]
