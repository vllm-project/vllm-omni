# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Prepare hash-relevant input before installing it on the scheduler thread."""

import json
from copy import copy
from hashlib import sha256

import regex as re
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.request import Request
from vllm.v1.utils import ConstantList

_INPUT_FIELDS = (
    "prompt_token_ids",
    "prompt_embeds",
    "prompt_is_token_ids",
    "num_prompt_tokens",
    "_all_token_ids",
    "_output_token_ids",
    "all_token_ids",
    "output_token_ids",
    "num_computed_tokens",
    "mm_features",
    "cache_salt",
    "_omni_original_cache_salt",
    "_omni_conditioning_digest",
    "sampling_params",
    "block_hashes",
    "_prompt_embeds_per_block_hashes",
    "skip_reading_prefix_cache",
)


def compose_conditioning_cache_salt(caller_salt: str | None, conditioning_digest: str) -> str:
    """Combine a validated fixed-input digest with the opaque ORIGINAL salt.

    Never infer whether a user salt is already composed from its contents.
    The caller's value is kept separately on each finalized input snapshot.
    """
    if not isinstance(conditioning_digest, str) or re.fullmatch(r"[0-9a-f]{64}", conditioning_digest) is None:
        raise ValueError("conditioning digest must be a lowercase SHA-256 hex string")
    if caller_salt is not None and not isinstance(caller_salt, str):
        raise ValueError("caller cache salt must be a string or None")
    encoded = json.dumps(
        ["vllm-omni.conditioning.v1", caller_salt, conditioning_digest],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def prepare_request_input(
    request: Request,
    *,
    prompt_token_ids: list[int] | None,
    mm_features: list[MultiModalFeatureSpec],
    cache_salt: str | None,
    sampling_params: SamplingParams | None,
    prompt_embeds: torch.Tensor | None = None,
    prompt_is_token_ids: list[bool] | None = None,
    conditioning_digest: str | None = None,
) -> Request:
    """Validate and hash a candidate without mutating the admitted request.

    The shallow copy is only a hasher input, never a second scheduled request.
    Tensor payloads retain their existing ownership; token lists and hash memos
    get fresh storage so dispatched steps can keep their old snapshots.
    """
    token_ids = list(prompt_token_ids) if prompt_token_ids is not None else None
    if token_ids is not None and any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("finalized prompt token IDs must be non-negative integers")
    if prompt_embeds is not None and prompt_embeds.ndim != 2:
        raise ValueError("finalized prompt embeddings must have shape [tokens, hidden]")
    length = length_from_prompt_token_ids_or_embeds(token_ids, prompt_embeds)
    mask = list(prompt_is_token_ids) if prompt_is_token_ids is not None else None
    if mask is not None:
        if token_ids is None or len(mask) != length or any(type(value) is not bool for value in mask):
            raise ValueError("finalized prompt token mask must align with token IDs")
        if prompt_embeds is None and not all(mask):
            raise ValueError("finalized prompt embeddings must cover the masked rows")

    effective_salt = (
        cache_salt if conditioning_digest is None else compose_conditioning_cache_salt(cache_salt, conditioning_digest)
    )
    candidate = copy(request)
    candidate.prompt_token_ids = token_ids
    candidate.prompt_embeds = prompt_embeds
    candidate.prompt_is_token_ids = mask
    candidate.num_prompt_tokens = length
    if token_ids is None:
        candidate._all_token_ids = [0] * length
    elif mask is None:
        candidate._all_token_ids = list(token_ids)
    else:
        candidate._all_token_ids = [token if is_token else 0 for token, is_token in zip(token_ids, mask)]
    candidate._output_token_ids = []
    candidate.all_token_ids = ConstantList(candidate._all_token_ids)
    candidate.output_token_ids = ConstantList(candidate._output_token_ids)
    candidate.num_computed_tokens = 0
    candidate.mm_features = list(mm_features)
    candidate.cache_salt = effective_salt
    setattr(candidate, "_omni_original_cache_salt", cache_salt)
    setattr(candidate, "_omni_conditioning_digest", conditioning_digest)
    candidate.sampling_params = sampling_params
    candidate.block_hashes = []
    candidate._prompt_embeds_per_block_hashes = {}
    candidate.update_block_hashes()
    candidate.skip_reading_prefix_cache = candidate.get_skip_reading_prefix_cache()
    return candidate


def install_request_input(request: Request, candidate: Request) -> None:
    """Install validated input, retaining lifetime and in-flight accounting."""
    request.__dict__.update({name: getattr(candidate, name) for name in _INPUT_FIELDS})
    setattr(request, "_omni_input_finalized", True)
