# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for building TADA TTS prompts offline.

TADA walks the input text one token per step, emitting one acoustic frame each, so a request
is split into a short prefilled prompt plus a forced ``tada_walk_ids`` token stream. Voice
cloning additionally encodes a reference waveform + transcript into prompt acoustic features.
These helpers run outside the serving worker (e.g. an example script or a test) and produce the
prompt dict consumed by the AR stage's ``preprocess``.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import torch


def get_tokenizer(model: str, _cache: dict = {}):
    """Load (and cache) the model's tokenizer."""
    if model not in _cache:
        from transformers import AutoTokenizer

        _cache[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return _cache[model]


def get_shift_acoustic(model: str, _cache: dict = {}) -> int:
    """Read ``shift_acoustic`` from the model config (default 5)."""
    if model not in _cache:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        _cache[model] = int(getattr(cfg, "shift_acoustic", 5))
    return _cache[model]


def chat_prefix(system_prompt: str = "", user_turn: str | None = None) -> str:
    """Build the chat-template prefix (system/user/assistant headers)."""
    prefix = f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"
    if user_turn:
        prefix += f"<|start_header_id|>user<|end_header_id|>{user_turn}<|eot_id|>"
    prefix += "<|start_header_id|>assistant<|end_header_id|>"
    return prefix


def _resolve_codec_path(model: str) -> str:
    """Resolve the codec directory from ``TADA_CODEC_PATH`` or a sibling of the model path."""
    return os.environ.get("TADA_CODEC_PATH") or os.path.join(os.path.dirname(os.path.abspath(model)), "tada-codec")


def _get_encoder(model: str, _cache: dict = {}):
    """Load (and cache) the encoder + aligner from the local codec weights."""
    if "enc" not in _cache:
        from vllm_omni.model_executor.models.tada_tts.codec.encoder import Encoder

        _cache["enc"] = Encoder.from_local(_resolve_codec_path(model), model, device="cpu", dtype=torch.float32)
    return _cache["enc"]


def build_zeroshot_prompt(text: str, model: str) -> tuple[dict, int]:
    """Build a zero-shot prompt, returning ``(prompt_dict, walk_len)``.

    ``prompt_token_ids`` holds ``[BOS] + chat-template headers`` (prefilled); ``tada_walk_ids``
    holds ``tokenize(text) + [<|eot_id|>] * shift`` (forced one per decode step). ``walk_len``
    sets ``max_tokens`` so generation stops on consuming the last walk token.
    """
    tok = get_tokenizer(model)
    shift = get_shift_acoustic(model)
    bos_id = tok.bos_token_id
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")

    prefix_ids = tok.encode(chat_prefix(), add_special_tokens=False)
    walk_ids = tok.encode(text, add_special_tokens=False) + [eot_id] * shift

    prompt = {
        "prompt_token_ids": [bos_id] + prefix_ids,
        "additional_information": {
            "tada_walk_ids": walk_ids,
            # The acoustic stream lags the text by ``shift``, so the first ``shift`` decode
            # frames carry the header tail rather than the text — drop them.
            "tada_trim_lead": shift,
        },
    }
    return prompt, len(walk_ids)


def build_voice_clone_prompt(
    text: str, ref_audio: str, ref_text: str, model: str, num_transition_steps: int = 5
) -> tuple[dict, int]:
    """Build a voice-cloning prompt from a reference wav and its transcript.

    Encodes the reference audio to per-token acoustic features + a token→frame alignment, then:
      * ``prompt_token_ids`` = ``[BOS] + headers + transcript[:-N]`` — prefilled with the
        reference acoustic substituted over the transcript region so the voice enters the KV cache.
      * ``tada_walk_ids``    = ``transcript[-N:] + tokenize(text) + [<|eot_id|>] * shift`` —
        walked in decode. The first ``N`` walked tokens (transcript tail) form a transition that
        smooths the prompt→synthesis boundary; their frames are fed back but dropped from output.
    """
    import numpy as np
    import soundfile as sf
    import torch.nn.functional as Fnn
    from transformers import AutoConfig

    tok = get_tokenizer(model)
    shift = get_shift_acoustic(model)
    ntc = int(getattr(AutoConfig.from_pretrained(model, trust_remote_code=True), "num_time_classes", 256))
    enc = _get_encoder(model)

    wav, sr = sf.read(ref_audio)
    wav_t = torch.tensor(np.asarray(wav), dtype=torch.float32).reshape(1, -1)
    out = enc(wav_t, text=ref_text, sample_rate=sr)
    token_values = out.token_values[0]  # [Tp, feat_dim] (normalised)
    token_positions = out.token_positions[0].long()  # [Tp]

    # Per-token durations from the alignment positions.
    sel = token_positions.float()
    prev = Fnn.pad(sel, (1, 0), value=1)[:-1]
    time_gaps = Fnn.pad((sel - prev).clamp(0, ntc - 1), (1, 0), value=0)  # [Tp+1]
    tb = time_gaps[:-1].long()
    ta = time_gaps[1:].long()

    bos_id = tok.bos_token_id
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    prefix_ids = tok.encode(chat_prefix(), add_special_tokens=False)
    transcript_ids = tok.encode(ref_text, add_special_tokens=False)
    synth_ids = tok.encode(text, add_special_tokens=False)

    n = min(token_values.shape[0], len(transcript_ids))
    token_values, tb, ta, transcript_ids = token_values[:n], tb[:n], ta[:n], transcript_ids[:n]

    # Split off the transition tail (keep >= 1 prompt token).
    n_trans = max(0, min(num_transition_steps, n - 1))
    n_prompt = n - n_trans
    prompt_ids = transcript_ids[:n_prompt]
    transition_ids = transcript_ids[n_prompt:]

    prefill_ids = [bos_id] + prefix_ids + prompt_ids
    walk_ids = transition_ids + synth_ids + [eot_id] * shift
    prefix_len = len(prefix_ids)

    prompt = {
        "prompt_token_ids": prefill_ids,
        "additional_information": {
            "tada_walk_ids": walk_ids,
            # The acoustic stream lags the text by ``shift``, so the first ``shift`` decode
            # frames carry the prompt-transcript tail and the next ``n_trans`` carry the
            # transition tokens — drop all of them.
            "tada_trim_lead": n_trans + shift,
            # Prefix-padded prompt arrays (without BOS).
            "tada_prompt_acoustic": Fnn.pad(token_values[:n_prompt], (0, 0, prefix_len, 0)).contiguous(),
            "tada_prompt_masks": Fnn.pad(torch.ones(n_prompt, dtype=torch.long), (prefix_len, 0)).contiguous(),
            "tada_prompt_tb": Fnn.pad(tb[:n_prompt], (prefix_len, 0)).contiguous(),
            "tada_prompt_ta": Fnn.pad(ta[:n_prompt], (prefix_len, 0)).contiguous(),
        },
    }
    return prompt, len(walk_ids)


def apply_walk_sampling_params(sampling_params_list: list[Any], walk_len: int) -> list[Any]:
    """Return a copy of the per-stage sampling params with stage 0 set to walk exactly
    ``walk_len`` tokens: ``max_tokens = walk_len`` enforces the fixed length, ``ignore_eos``
    keeps the unused sampled token from ending generation early, and ``temperature = 0`` makes
    that sample deterministic. Later-stage params are unchanged.
    """
    sp_list = [copy.deepcopy(sp) for sp in sampling_params_list]
    sp0 = sp_list[0]
    sp0.max_tokens = walk_len
    sp0.ignore_eos = True
    sp0.temperature = 0.0
    return sp_list
