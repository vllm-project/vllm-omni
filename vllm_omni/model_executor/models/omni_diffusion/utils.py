import random
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

# Omni-Diffusion's audio understanding path uses 16 kHz features.
OMNI_DIFFUSION_INPUT_SAMPLE_RATE = 16000

# Its GLM-4-Voice decoder returns generated speech at 22.05 kHz.
OMNI_DIFFUSION_OUTPUT_SAMPLE_RATE = 22050

# The default RoPE theta value for Omni-Diffusion models, used if not
# specified in the model config (config.json).
OMNI_DIFFUSION_DEFAULT_ROPE_THETA = 1000000.0

# The default RoPE type for Omni-Diffusion models on transformers v5.
OMNI_DIFFUSION_DEFAULT_ROPE_TYPE = "default"

# The default partial rotary factor for Omni-Diffusion with transformers v5.
OMNI_DIFFUSION_DEFAULT_PARTIAL_ROTARY_FACTOR = 1.0

OMNI_DIFFUSION_OUTPUT_TEXT_ONLY_TASKS = ("ASR", "VQA", "SVQA")

OMNI_DIFFUSION_TEXT_ADAPTER_TOKEN_IDS_KEY = "omni_diffusion_text_token_ids"

OMNI_DIFFUSION_AUDIO_START_TOKEN = "<|audio_0|>"

OMNI_DIFFUSION_IMAGE_START_TOKEN = "<|image_0|>"

OMNI_DIFFUSION_IM_START_TOKEN = "<|im_start|>"

OMNI_DIFFUSION_IM_END_TOKEN = "<|im_end|>"

OMNI_DIFFUSION_END_OF_TEXT_TOKEN = "<|endoftext|>"

# Token ID of ``<|endoftext|>`` in the Omni-Diffusion Dream tokenizer. The
# stage-1 text adapter emits it after the target text so vLLM can stop AR
# decoding without running until max_tokens.
OMNI_DIFFUSION_END_OF_TEXT_TOKEN_ID = 151643

# Number of discrete audio code tokens reserved by Omni-Diffusion's tokenizer.
OMNI_DIFFUSION_AUDIO_CODEBOOK_SIZE = 16384

# Number of discrete image code tokens reserved by Omni-Diffusion's tokenizer.
OMNI_DIFFUSION_IMAGE_CODEBOOK_SIZE = 8192

logger = init_logger(__name__)


def normalize_token_id_sequence(value: Any, *, source: str) -> list[int]:
    """Normalize one token-ID sequence from a tensor or Python sequence.

    Inter-stage payloads may be serialized as ``[T]`` or as a single-item
    batch ``[1, T]``. Omni-Diffusion's text adapter handles one sequence per
    request, so reject multi-sequence batches instead of silently selecting
    the first one.
    """
    if value is None:
        return []

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected {source} to be a token-ID sequence, got {type(value)!r}.")

    token_ids = list(value)
    if token_ids and isinstance(token_ids[0], Sequence) and not isinstance(token_ids[0], (str, bytes)):
        if len(token_ids) != 1:
            raise ValueError(f"Expected one {source} token sequence, got {len(token_ids)}.")
        token_ids = list(token_ids[0])

    return [int(token_id) for token_id in token_ids]


def set_generation_seed(seed: int | None) -> None:
    """Seed the random generators used by Omni-Diffusion generation."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class OmniDiffusionModelSpecialTokens(str, Enum):
    """Special tokens used in Omni-Diffusion models."""

    AUD_TAG = "<|audio|>"
    AUD_CONTEXT = "<|context_of_audio|>"
    AUD_START = "<|begin_of_audio|>"
    AUD_END = "<|end_of_audio|>"

    IMG_TAG = "<|image|>"
    IMG_START = "<|begin_of_image|>"
    IMG_END = "<|end_of_image|>"


class OmniDiffusionTokenizerBaseData:
    """Holds the token IDs for Omni-Diffusion special tokens."""

    def __init__(self, tokenizer: Any) -> None:
        self.omni_diffusion_special_tokens_2_token_ids: dict[OmniDiffusionModelSpecialTokens, int] = {}
        for token in OmniDiffusionModelSpecialTokens:
            token_id = get_single_token_id(tokenizer, token.value)
            self.omni_diffusion_special_tokens_2_token_ids[token] = token_id
        logger.info(
            "Initialized OmniDiffusionTokenizerBaseData with special token IDs: %s",
            self.omni_diffusion_special_tokens_2_token_ids,
        )

    def get_token_id(self, token: OmniDiffusionModelSpecialTokens) -> int:
        """Get the token ID for a given Omni-Diffusion special token."""
        return self.omni_diffusion_special_tokens_2_token_ids[token]


def get_single_token_id(tokenizer: Any, token: str) -> int:
    """Get the token ID for a single token.

    Args:
        tokenizer (Any): The tokenizer to use.
        token (str): The token for which to get the ID.

    Returns:
        int: The token ID.
    """
    return get_single_token_ids(tokenizer, [token])[0]


def get_single_token_ids(tokenizer: Any, tokens: Sequence[str]) -> list[int]:
    """Get the token IDs for a list of tokens.

    Args:
        tokenizer (Any): The tokenizer to use.
        tokens (Sequence[str]): The tokens for which to get the IDs.

    Returns:
        list[int]: The token IDs.
    """
    batch_token_ids = tokenizer(list(tokens), add_special_tokens=False).input_ids
    if len(batch_token_ids) != len(tokens):
        raise ValueError(f"Expected {len(tokens)} encoded token sequences, got {len(batch_token_ids)}.")

    token_ids: list[int] = []
    for token, encoded_ids in zip(tokens, batch_token_ids):
        if len(encoded_ids) != 1:
            raise ValueError(f"Expected {token!r} to map to exactly one token ID, got {encoded_ids}.")
        token_ids.append(int(encoded_ids[0]))
    return token_ids
