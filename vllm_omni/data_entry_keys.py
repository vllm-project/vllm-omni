"""Canonical data-entry key names for inter-stage communication.

Adding a new model?
~~~~~~~~~~~~~~~~~~~
Every key you put into the inter-stage payload (``additional_information``,
``multimodal_output``, ``pooling_output``) **must** follow this format:

    {type}.{qualifier}          e.g.  "hidden_states.output"

Allowed types
    hidden_states  – intermediate / output hidden-state tensors
    embed          – embedding tensors (prefill, decode, special tokens)
    ids            – token-ID sequences
    codes          – codec / audio code tensors
    meta           – scalar metadata, control flags, shapes

Rules
    1. Pick from the existing keys below whenever your concept matches.
       Two models that transfer the same semantic concept MUST use the
       same key (e.g. all models use ``"codes.audio"`` for audio codes,
       not a model-specific name).
    2. If no existing key fits, create a new ``{type}.{qualifier}`` key.
       Keep the qualifier short and descriptive (snake_case).
    3. Never use bare names like ``"hidden"`` or ``"finished"`` — always
       use the dotted form.  ``normalize_keys()`` will catch old-style
       names at runtime and log a deprecation warning.

Existing keys (reference)
    hidden_states.layer_0          hidden_states.layer_{N}
    hidden_states.output           hidden_states.trailing_text
    hidden_states.last
    embed.prefill                  embed.decode
    embed.cached_decode            embed.tts_bos / tts_eos / tts_pad
    embed.voice                    embed.speech_feat
    ids.all        ids.prompt      ids.output
    ids.speech_token               ids.prior_image
    codes.audio                    codes.ref
    meta.finished                  meta.left_context_size
    meta.override_keys             meta.num_processed_tokens
    meta.next_stage_prompt_len     meta.ar_width
    meta.eol_token_id              meta.visual_token_start_id
    meta.visual_token_end_id       meta.gen_token_mask
    meta.generated_len             meta.omni_task
    meta.height                    meta.width

This module provides:
- ``_LEGACY_KEY_MAP``: old key name → canonical key name
- ``normalize_keys(payload)``: translate any legacy keys in a dict
- ``validate_key(key)``: check a key follows the naming convention
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Legacy key mapping (old name → canonical name) ──
_LEGACY_KEY_MAP: dict[str, str] = {
    # hidden states
    "0": "hidden_states.layer_0",
    "hidden": "hidden_states.layer_0",
    "24": "hidden_states.layer_24",
    "thinker_hidden_states": "hidden_states.output",
    "thinker_result": "hidden_states.output",
    "latent": "hidden_states.output",
    "trailing_text_hidden": "hidden_states.trailing_text",
    "tailing_text_hidden": "hidden_states.trailing_text",
    "last_talker_hidden": "hidden_states.last",
    # embeddings
    "thinker_prefill_embeddings": "embed.prefill",
    "prompt_embeds": "embed.prefill",
    "thinker_decode_embeddings": "embed.decode",
    "decode_output_prompt_embeds": "embed.decode",
    "cached_thinker_decode_embeddings": "embed.cached_decode",
    "tts_bos_embed": "embed.tts_bos",
    "tts_eos_embed": "embed.tts_eos",
    "tts_pad_embed": "embed.tts_pad",
    "embedding": "embed.voice",
    "speech_feat": "embed.speech_feat",
    # token ids
    "thinker_sequences": "ids.all",
    "thinker_input_ids": "ids.prompt",
    "prompt_token_ids": "ids.prompt",
    "prefix_ids": "ids.prompt",
    "thinker_output_token_ids": "ids.output",
    "speech_token": "ids.speech_token",
    "prior_token_image_ids": "ids.prior_image",
    # codes
    "code_predictor_codes": "codes.audio",
    "audio_codes": "codes.audio",
    "ref_code": "codes.ref",
    # metadata
    "finished": "meta.finished",
    "finished_flag": "meta.finished",
    "left_context_size": "meta.left_context_size",
    "override_keys": "meta.override_keys",
    "num_processed_tokens": "meta.num_processed_tokens",
    "next_stage_prompt_len": "meta.next_stage_prompt_len",
    "ar_width": "meta.ar_width",
    "eol_token_id": "meta.eol_token_id",
    "visual_token_start_id": "meta.visual_token_start_id",
    "visual_token_end_id": "meta.visual_token_end_id",
    "gen_token_mask": "meta.gen_token_mask",
    "generated_len": "meta.generated_len",
    "omni_task": "meta.omni_task",
    "height": "meta.height",
    "width": "meta.width",
}


_VALID_TYPES = frozenset({"hidden_states", "embed", "ids", "codes", "meta"})


def validate_key(key: str) -> bool:
    """Return *True* if *key* follows the ``{type}.{qualifier}`` convention.

    >>> validate_key("codes.audio")
    True
    >>> validate_key("finished")
    False
    """
    parts = key.split(".", 1)
    return len(parts) == 2 and parts[0] in _VALID_TYPES and len(parts[1]) > 0


def normalize_keys(payload: dict[str, Any], *, warn: bool = True) -> dict[str, Any]:
    """Return a new dict with legacy key names replaced by canonical names.

    Keys already using canonical names are passed through unchanged.
    Unknown keys (not legacy and not canonical) are warned about if they
    don't follow the ``{type}.{qualifier}`` naming convention.
    """
    out: dict[str, Any] = {}
    for key, value in payload.items():
        canonical = _LEGACY_KEY_MAP.get(key)
        if canonical is not None:
            if warn:
                logger.warning(
                    "Deprecated data-entry key %r – use %r instead",
                    key,
                    canonical,
                )
            out[canonical] = value
        else:
            if warn and not validate_key(key):
                logger.warning(
                    "Data-entry key %r does not follow the "
                    "'{type}.{qualifier}' convention – see "
                    "vllm_omni/data_entry_keys.py for guidance",
                    key,
                )
            out[key] = value
    return out
