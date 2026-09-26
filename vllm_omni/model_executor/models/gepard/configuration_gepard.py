# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for Gepard-1.0, a single-stage autoregressive TTS.

Text tokens -> one 32-code FSQ audio frame per step -> NeMo NanoCodec ->
waveform. The backbone is a vLLM-native ``Qwen3_5ForCausalLM``; the codebook
heads, binary stop head and voice-clone ref_compressor are Gepard additions.

Parses the model's ``gepard_config.json`` sidecar, which nests the LM
parameters under ``backbone_config`` and carries audio-head cardinalities,
special tokens, codec settings and the short-text repetition layout.
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

_BACKBONE_MODEL_TYPE = "qwen3_5_text"

# Stripped from the backbone config: the model uses standard 1D RoPE, and these
# would flip vLLM's uses_mrope() to True.
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
        backbone_config: dict | None = None,
        audio_heads: dict | None = None,
        audio_embed_dim: int = 32,
        special_tokens: dict | None = None,
        text_repetition: dict | None = None,
        codec: dict | None = None,
        voice_cloning: dict | None = None,
        stop_threshold: float = 0.5,
        stop_loss_weight: float = 2.0,
        stop_pos_weight: float = 25.0,
        temperature: float = 0.3,
        **kwargs,
    ):
        # Everything is assigned before super().__init__() at the bottom:
        # transformers 5.x runs validators there that call get_text_config().
        self.backbone_config = self._normalize_backbone(backbone_config)

        self.audio_head_levels = self._parse_audio_heads(audio_heads)
        self.num_audio_heads = len(self.audio_head_levels)
        self.audio_embed_dim = audio_embed_dim
        # head0 is the token vLLM samples; STOP is a synthetic sentinel one past
        # its valid range.
        self.head0_vocab_size = self.audio_head_levels[0] if self.audio_head_levels else 8
        self.stop_token = self.head0_vocab_size

        st = special_tokens or {}
        self.start_of_text = st.get("start_of_text", 248073)
        self.end_of_text = st.get("end_of_text", 248074)
        self.start_of_speech = st.get("start_of_speech", 248070)
        self.end_of_speech = st.get("end_of_speech", 248071)
        self.tts_pad = st.get("tts_pad", 248076)
        # The speaker placeholder slots begin at tokeniser_length.
        self.tokeniser_length = st.get("tokeniser_length", 248077)
        self.speaker_token_base = self.tokeniser_length

        # Prompt layout, read by prompt.py. These must match the training
        # layout, so they come from the checkpoint rather than a literal.
        tr = text_repetition or {}
        self.text_repetition_enabled = tr.get("enabled", True)
        self.text_repetition_target_tokens = tr.get("target_text_tokens", 16)
        self.text_repetition_apply_below = tr.get("apply_below", 13)
        self.text_repetition_max_repeats = tr.get("max_repeats", 8)

        # NeMo NanoCodec, which runs outside vLLM.
        cc = codec or {}
        self.codec_id = cc.get("codec_id", "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps")
        self.codec_sample_rate = cc.get("sample_rate", 22050)
        self.codec_frame_rate_hz = cc.get("frame_rate_hz", 21.5)
        # Per-group FSQ levels; num_codec_groups * len(fsq_levels) == 32.
        self.fsq_levels = cc.get("fsq_levels", [8, 7, 6, 6])
        self.num_codec_groups = cc.get("num_layers", 8)
        self.codec_do_unfold = cc.get("do_unfold", True)

        self.stop_threshold = stop_threshold
        self.stop_loss_weight = stop_loss_weight
        self.stop_pos_weight = stop_pos_weight
        # head0 goes through vLLM's SamplingParams; the other 31 heads use this
        # in-model. There is no top_p: Gumbel-max has no nucleus step.
        self.temperature = temperature

        # Carried for the cloning follow-up; zero-shot uses null_prefix only.
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

    @classmethod
    def from_checkpoint(
        cls,
        model: str,
        backbone_config: dict | None = None,
        revision: str | None = None,
    ) -> GepardConfig:
        """Build the full config for a checkpoint that self-identifies as the
        bare backbone: audio fields from the sidecar, backbone fields from the
        loaded config.

        ``revision`` must be the one the weights came from — a revision that
        moves the audio-head cardinalities or the special tokens moves the
        prompt layout and the STOP sentinel with them.
        """
        sidecar: dict = {}
        try:
            from vllm.transformers_utils.config import get_hf_file_to_dict

            sidecar = dict(get_hf_file_to_dict("gepard_config.json", model, revision=revision) or {})
        except (OSError, ValueError) as e:
            # A missing or malformed sidecar is expected; the defaults match the
            # trained checkpoint. Anything else must surface.
            logger.warning(
                "GepardConfig: could not read gepard_config.json from %s (%s: %s); "
                "audio fields fall back to trained-checkpoint defaults.",
                model,
                type(e).__name__,
                e,
            )
        if backbone_config:
            bb = dict(backbone_config)
            # The loaded config carries the hf_overrides-patched identity; the
            # backbone must stay qwen3_5_text or get_text_config() recurses.
            bb["model_type"] = _BACKBONE_MODEL_TYPE
            bb.pop("architectures", None)
            sidecar["backbone_config"] = bb
        return cls(**sidecar)

    @staticmethod
    def _parse_audio_heads(audio_heads: dict | None) -> list[int]:
        """``{level_audio_0: 8, level_audio_1: 7, ...}`` -> ``[8, 7, ...]``.

        Sorted by the numeric suffix, so channel order does not depend on dict
        insertion order.
        """
        if not audio_heads:
            return [8, 7, 6, 6] * 8
        return [audio_heads[k] for k in sorted(audio_heads, key=lambda s: int(str(s).rsplit("_", 1)[-1]))]

    @classmethod
    def _normalize_backbone(cls, backbone_config: dict | None) -> dict:
        """Return the backbone dict with vestigial mRoPE fields removed.

        They are template leftovers that the trained checkpoint's own
        ``config.json`` does not carry. Left in, they flip vLLM's
        ``uses_mrope()`` to True and wire ``positions`` wrongly.
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

    def get_text_config(self, **kwargs) -> PretrainedConfig:
        """Return the Qwen3.5 backbone config, which vLLM runs the backbone on."""
        cached = getattr(self, "_text_config", None)
        if cached is not None:
            return cached

        bb = dict(self.backbone_config)
        # pop, not setdefault: for_model()'s first positional is named
        # model_type, so leaving the key in **bb passes it twice.
        model_type = bb.pop("model_type", _BACKBONE_MODEL_TYPE)
        try:
            text_config = AutoConfig.for_model(model_type, **bb)
        except (KeyError, ValueError) as e:
            # Only an unknown model_type is a legitimate fallback. A TypeError
            # means the kwargs are wrong and must not degrade silently.
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
