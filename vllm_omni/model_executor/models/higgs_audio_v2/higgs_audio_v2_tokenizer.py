# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TTS prompt builder + scope validators for higgs-audio v2.

vllm-omni's higgs path supports two request shapes:

* **Plain text -> 24 kHz speech.** :func:`build_plain_text_prompt` runs the
  upstream HF processor with a bare ``"Generate audio following instruction."``
  system prompt; the emitted ``input_ids`` are byte-identical to upstream.
* **Voice clone (shallow).** :func:`build_voice_clone_prompt` runs the
  upstream HF processor with both a target text turn and a reference
  ``(ref_audio, ref_text)`` ICL turn; the processor returns ``input_ids``
  pre-expanded with audio placeholders plus an ``audio_input_ids`` tensor
  carrying the encoded reference codes (HF tokenizer encodes audio on the
  spot — no encoder vendored in vllm-omni).

Still out of scope and rejected with explicit 4xx: multi-speaker
``[SPEAKERn]`` dialogue, ``profile:`` text-only speaker descriptions, the
``ref_audio_in_system_message`` system-block variant, and chunked long-form
generation.
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import torch

__all__ = [
    "UnsupportedInputError",
    "MULTI_SPEAKER_TAG_PATTERN",
    "validate_plain_text_input",
    "build_plain_text_conversation",
    "build_plain_text_prompt",
    "build_voice_clone_conversation",
    "build_voice_clone_prompt",
    "input_ids_to_python_list",
]


class UnsupportedInputError(ValueError):
    """Raised when a request asks for an out-of-scope higgs_audio_v2 feature."""


# Matches the upstream multi-speaker SPEAKERn tag, e.g. [SPEAKER0], [SPEAKER12].
MULTI_SPEAKER_TAG_PATTERN = re.compile(r"\[SPEAKER\d+\]", re.IGNORECASE)


def validate_plain_text_input(text: str) -> None:
    """Reject multi-speaker tags inside the user text body.

    Phase-1 explicitly does NOT support multi-speaker dialogue. Catching the
    pattern here means the rejection happens at the tokenizer boundary and is
    visible to both offline (`pipeline.py`) and online (`serving_speech.py`)
    code paths.
    """
    if not isinstance(text, str):
        raise UnsupportedInputError(f"higgs_audio_v2 expects plain text input; got {type(text).__name__}")
    if MULTI_SPEAKER_TAG_PATTERN.search(text):
        raise UnsupportedInputError(
            "higgs_audio_v2 v1 does not support multi-speaker [SPEAKERn] tags; received text contains a speaker tag"
        )


def build_plain_text_conversation(text: str) -> list[dict[str, Any]]:
    """Build the canonical single-speaker plain-text conversation.

    Uses the bare system prompt ``"Generate audio following instruction."``
    that matches the upstream HF reference's input formatting; this exact
    wording is required for input-token parity with the upstream processor.
    """
    validate_plain_text_input(text)
    system_prompt = "Generate audio following instruction."
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    ]


def build_plain_text_prompt(
    processor: Any,
    text: str,
    *,
    sampling_rate: int = 24000,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    """Run the upstream processor's chat template on a plain-text input.

    Returns the processor output dict (``input_ids`` plus any auxiliary tensors
    such as ``attention_mask``). The serving layer passes ``input_ids`` to
    Stage 0 as ``prompt_token_ids`` after a ``.tolist()``.

    Using the upstream processor verbatim (no system-prompt rewriting) is
    what preserves input-token parity with the HF reference.
    """
    conversation = build_plain_text_conversation(text)
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        sampling_rate=sampling_rate,
        return_tensors=return_tensors,
    )
    if "input_ids" not in inputs:
        raise RuntimeError(f"HiggsAudioV2 processor returned no input_ids; got keys {list(inputs.keys())!r}")
    return inputs


def input_ids_to_python_list(inputs: dict[str, Any]) -> list[int]:
    """Convenience: pull a flat ``list[int]`` of token IDs from a processor output."""
    ids = inputs["input_ids"]
    if isinstance(ids, torch.Tensor):
        if ids.ndim == 2 and int(ids.shape[0]) != 1:
            raise ValueError(f"expected batch=1 prompt; got input_ids shape {tuple(ids.shape)}")
        return ids.reshape(-1).tolist()
    return list(ids)


_AUDIO_OUT_TOKEN = "<|AUDIO_OUT|>"
_AUDIO_OUT_BOS_TOKEN = "<|audio_out_bos|>"
_AUDIO_EOS_TOKEN = "<|audio_eos|>"
_AUDIO_DELAY_TOKEN = "<|reserved_special_token_6|>"
_AUDIO_STREAM_BOS_ID = 1024
_AUDIO_STREAM_EOS_ID = 1025


def _build_delay_pattern(codes: torch.Tensor) -> torch.Tensor:
    """Apply the upstream delay-pattern wrap to a ``[num_codebooks, T]`` code tensor.

    Mirrors ``HiggsAudioV2Processor.build_delay_pattern``: prepend BOS, append
    EOS, then arrange in a triangular pattern that stretches the sequence by
    ``num_codebooks - 1`` frames so codebook k starts emitting real codes at
    frame k.
    """
    num_codebooks, seq_len = codes.shape
    bos = codes.new_full((num_codebooks, 1), _AUDIO_STREAM_BOS_ID)
    eos = codes.new_full((num_codebooks, 1), _AUDIO_STREAM_EOS_ID)
    wrapped = torch.cat([bos, codes, eos], dim=1)
    wrapped_len = wrapped.shape[1]
    new_seq_len = wrapped_len + num_codebooks - 1

    output = torch.ones((1, num_codebooks, new_seq_len), dtype=codes.dtype, device=codes.device)
    bos_mask = torch.tril(output, -1) > 0
    eos_mask = torch.triu(output, wrapped_len) > 0
    data_mask = ~(bos_mask | eos_mask)
    output[bos_mask] = _AUDIO_STREAM_BOS_ID
    output[data_mask] = wrapped.reshape(-1)
    output[eos_mask] = _AUDIO_STREAM_EOS_ID
    return output[0]


_ENCODER_CACHE: Any | None = None
_K2_OMNIVOICE_REPO = "k2-fsa/OmniVoice"
_K2_OMNIVOICE_SUBDIR = "audio_tokenizer"


def _load_audio_tokenizer():
    """Load the higgs-audio v2 codec via HF ``HiggsAudioV2TokenizerModel``.

    The boson-ai standalone tokenizer repo (``bosonai/higgs-audio-v2-tokenizer``)
    ships a ``model.safetensors`` that is actually a copy of the 3B talker LM,
    not the codec — pointing HF's ``from_pretrained`` at it yields an
    encoder/decoder with randomly initialized structural weights, producing
    noise codes at encode time. The k2-fsa OmniVoice repo bundles the same
    codec weights repackaged under
    ``audio_tokenizer/{config.json,model.safetensors,preprocessor_config.json}``
    with key naming aligned to ``HiggsAudioV2TokenizerModel``'s class layout
    (``acoustic_encoder.*`` / ``acoustic_decoder.*`` / ``quantizer.quantizers.*``
    / ``fc/fc2`` / ``semantic_model.*``), so HF can load it directly.

    Only the ``audio_tokenizer/`` subdirectory (~806 MB) is downloaded.
    """
    from huggingface_hub import snapshot_download
    from transformers import HiggsAudioV2TokenizerModel

    repo_path = snapshot_download(
        _K2_OMNIVOICE_REPO,
        allow_patterns=[f"{_K2_OMNIVOICE_SUBDIR}/*"],
    )
    audio_tokenizer_dir = os.path.join(repo_path, _K2_OMNIVOICE_SUBDIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HiggsAudioV2TokenizerModel.from_pretrained(audio_tokenizer_dir).to(device)
    return model.eval()


def _encode_ref_audio_codes(
    wav: np.ndarray,
    sr: int,
) -> torch.Tensor:
    """Encode a single ref clip to codec codes via HF ``HiggsAudioV2TokenizerModel``.

    Returns shape ``[num_codebooks, T_raw]`` (before BOS/EOS + delay-pattern wrap).
    """
    import torchaudio

    global _ENCODER_CACHE
    if _ENCODER_CACHE is None:
        try:
            _ENCODER_CACHE = _load_audio_tokenizer()
        except ImportError as exc:
            raise RuntimeError(
                "higgs_audio_v2 voice clone needs `transformers>=5.3.0` "
                "(which ships the HiggsAudioV2TokenizerModel class). "
                "Install via `pip install -U 'transformers>=5.3.0'`."
            ) from exc

    target_sr = int(getattr(_ENCODER_CACHE.config, "sample_rate", 24000))
    wav_t = torch.as_tensor(wav, dtype=torch.float32).reshape(-1)
    if sr and sr != target_sr:
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
    # HF encode() expects (batch=1, channels=1, num_samples).
    input_values = wav_t.unsqueeze(0).unsqueeze(0).to(_ENCODER_CACHE.device)
    with torch.inference_mode():
        codes = _ENCODER_CACHE.encode(input_values, return_dict=False)
    # codes shape: (batch=1, num_quantizers, codes_length) -> (num_quantizers, T).
    if codes.dim() == 3:
        codes = codes.squeeze(0)
    return codes.detach().to("cpu").long()


def build_voice_clone_conversation(
    text: str,
    ref_text: str,
) -> list[dict[str, Any]]:
    """ChatML conversation for voice clone (HF jinja-template-compatible).

    The assistant turn is a list with a single ``{"type": "audio"}`` content
    block — no ``"audio"``/``"url"``/``"path"`` key, so ``apply_chat_template``
    won't try to extract audio data (that path would collide with our explicit
    audio encoder). The template renders the assistant block as the literal
    ``<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>``; we expand the single
    ``<|AUDIO_OUT|>`` token to ``N × audio_token + (num_codebooks-1) × delay_token``
    in :func:`build_voice_clone_prompt` to match the reference clip's frame count.
    """
    validate_plain_text_input(text)
    validate_plain_text_input(ref_text)
    return [
        {"role": "system", "content": "Generate audio following instruction."},
        {"role": "user", "content": ref_text},
        {"role": "assistant", "content": [{"type": "audio"}]},
        {"role": "user", "content": text},
    ]


def build_voice_clone_prompt(
    processor: Any,
    text: str,
    ref_audio_wav: np.ndarray | torch.Tensor,
    ref_audio_sr: int,
    ref_text: str,
    *,
    return_tensors: str | None = "pt",
) -> dict[str, Any]:
    """Build a voice-clone prompt using the upstream boson audio tokenizer.

    Returns a dict carrying:
      - ``prompt_token_ids``: ``list[int]`` — input_ids with the ref-audio
        ``<|AUDIO_OUT|>`` + ``<|reserved_special_token_6|>`` placeholders
        already expanded to match the encoded reference clip's frame count.
      - ``audio_input_ids``: ``Tensor[T_frames, num_codebooks]`` — encoded
        reference codes with BOS/EOS + delay pattern, transposed to ``[T, Q]``
        for parity with the HF processor output shape.
      - ``audio_input_ids_mask``: ``Tensor[T_frames]`` — all-True bool mask.
    """
    if isinstance(ref_audio_wav, torch.Tensor):
        wav = ref_audio_wav.detach().to("cpu").float().reshape(-1).numpy()
    else:
        wav = np.asarray(ref_audio_wav, dtype=np.float32).reshape(-1)

    # 1. Encode via the upstream boson tokenizer (real weights from model.pth).
    codes_qt = _encode_ref_audio_codes(wav, ref_audio_sr)  # [num_codebooks, T_raw]
    if codes_qt.ndim == 3:
        codes_qt = codes_qt[0]
    num_codebooks = int(codes_qt.shape[0])

    # 2. Apply BOS/EOS + delay-pattern wrap.
    audio_input_ids = _build_delay_pattern(codes_qt)  # [num_codebooks, T_full]
    T_full = int(audio_input_ids.shape[1])

    # 3. Render the chat template, then expand the single <|AUDIO_OUT|> marker
    #    in the assistant turn into the full placeholder block. After expansion
    #    the text contains exactly T_full audio-mask positions, matching the
    #    audio_input_ids time axis.
    conversation = build_voice_clone_conversation(text, ref_text)
    rendered = processor.tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    n_delay = num_codebooks - 1
    n_audio = T_full - n_delay
    if n_audio < 0:
        raise RuntimeError(f"ref clip too short ({T_full} frames) for delay pattern with num_codebooks={num_codebooks}")
    placeholders = _AUDIO_OUT_TOKEN * n_audio + _AUDIO_DELAY_TOKEN * n_delay
    expanded_assistant = f"{_AUDIO_OUT_BOS_TOKEN}{placeholders}{_AUDIO_EOS_TOKEN}"
    if _AUDIO_OUT_TOKEN not in rendered:
        raise RuntimeError(
            f"Voice-clone chat-template render is missing the assistant "
            f"audio placeholder marker. conversation={conversation!r}"
        )
    rendered = rendered.replace(_AUDIO_OUT_TOKEN, expanded_assistant, 1)

    # 4. Tokenize the rendered prompt. ``apply_chat_template`` already emits
    #    ``<|begin_of_text|>``, so disable add_special_tokens.
    encoded = processor.tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors=return_tensors,
    )
    prompt_token_ids = encoded["input_ids"]
    if isinstance(prompt_token_ids, torch.Tensor):
        prompt_token_ids = prompt_token_ids.reshape(-1).tolist()

    # 5. Transpose codes to [T_full, num_codebooks] for HF parity.
    audio_input_ids_t = audio_input_ids.transpose(0, 1).contiguous().to(torch.long)
    audio_input_ids_mask = torch.ones(T_full, dtype=torch.bool)

    return {
        "prompt_token_ids": prompt_token_ids,
        "audio_input_ids": audio_input_ids_t,
        "audio_input_ids_mask": audio_input_ids_mask,
    }
