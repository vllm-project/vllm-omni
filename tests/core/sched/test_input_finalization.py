# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Hash identity and snapshot ownership at scheduler input acceptance."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request

from vllm_omni.core.sched.input_finalization import install_request_input, prepare_request_input

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(embeds, tokens, mask):
    init_none_hash(sha256)
    return Request(
        request_id="input",
        prompt_token_ids=tokens,
        prompt_embeds=embeds,
        prompt_is_token_ids=mask,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=get_request_block_hasher(2, sha256),
        cache_salt="caller",
    )


@pytest.mark.parametrize(
    ("tokens", "mask", "embedding_rows"),
    [(None, None, 4), ([1, 2, 3, 4], None, 4), ([1, 0, 3, 0], [True, False, True, False], 4)],
)
def test_new_embeddings_rebuild_memo_and_match_fresh_request(tokens, mask, embedding_rows):
    embeddings = torch.arange(embedding_rows * 3, dtype=torch.float32).reshape(embedding_rows, 3)
    request = _request(torch.zeros_like(embeddings), tokens, mask)
    old_hashes = request.block_hashes
    old_memo = request._prompt_embeds_per_block_hashes
    memo_snapshot = dict(old_memo)
    request.num_in_flight_tokens = 3

    candidate = prepare_request_input(
        request,
        prompt_token_ids=tokens,
        prompt_embeds=embeddings,
        prompt_is_token_ids=mask,
        mm_features=[],
        cache_salt="caller",
        sampling_params=request.sampling_params,
    )
    # Preparation has no visible effects, including when a hasher can fail.
    assert request.block_hashes is old_hashes
    assert request._prompt_embeds_per_block_hashes is old_memo
    install_request_input(request, candidate)

    fresh = _request(embeddings, tokens, mask)
    assert request.block_hashes == fresh.block_hashes
    assert request.block_hashes != old_hashes
    assert request._prompt_embeds_per_block_hashes == fresh._prompt_embeds_per_block_hashes
    assert old_memo == memo_snapshot
    assert request.num_in_flight_tokens == 3
    assert request.prompt_embeds is embeddings


def test_hash_failure_does_not_partially_install_new_input():
    request = _request(None, [1, 2, 3, 4], None)
    old_hashes = list(request.block_hashes)
    old_tokens = list(request.all_token_ids)

    def fail_hash(_request):
        raise RuntimeError("hash failure")

    request._block_hasher = fail_hash
    with pytest.raises(RuntimeError, match="hash failure"):
        prepare_request_input(
            request,
            prompt_token_ids=[11, 12],
            mm_features=[],
            cache_salt="different-caller",
            conditioning_digest="ab" * 32,
            sampling_params=request.sampling_params,
        )
    assert request.block_hashes == old_hashes
    assert list(request.all_token_ids) == old_tokens
    assert request.cache_salt == "caller"
    assert not getattr(request, "_omni_input_finalized", False)
    assert not hasattr(request, "_omni_original_cache_salt")
    assert not hasattr(request, "_omni_conditioning_digest")


def test_append_after_finalization_preserves_the_valid_hash_prefix():
    request = _request(None, [1, 2, 3, 4], None)
    candidate = prepare_request_input(
        request,
        prompt_token_ids=[11, 12, 13, 14],
        mm_features=[],
        cache_salt="caller",
        sampling_params=request.sampling_params,
    )
    install_request_input(request, candidate)
    prefix = list(request.block_hashes)
    request.append_output_token_ids([15, 16])
    fresh = _request(None, [11, 12, 13, 14, 15, 16], None)
    assert request.block_hashes == fresh.block_hashes
    assert request.block_hashes[: len(prefix)] == prefix
    assert request.cache_salt == "caller"


@pytest.mark.parametrize("caller_salt", [None, "", "caller", "omni-conditioning-v1:opaque-caller-input"])
def test_conditioning_salt_is_installed_with_ids_and_keeps_original_caller(caller_salt):
    from vllm_omni.core.sched.input_finalization import compose_conditioning_cache_salt

    request = _request(None, [0, 0, 0, 0], None)
    request.cache_salt = caller_salt
    digest = "ab" * 32
    old_ids, old_hashes = request.all_token_ids, request.block_hashes
    candidate = prepare_request_input(
        request,
        prompt_token_ids=[11, 12, 13, 14],
        mm_features=[],
        cache_salt=caller_salt,
        sampling_params=request.sampling_params,
        conditioning_digest=digest,
    )
    assert request.all_token_ids is old_ids
    assert request.block_hashes is old_hashes
    assert request.cache_salt == caller_salt
    install_request_input(request, candidate)
    fresh = _request(None, [11, 12, 13, 14], None)
    fresh.cache_salt = compose_conditioning_cache_salt(caller_salt, digest)
    fresh.block_hashes = []
    fresh.update_block_hashes()
    assert request.block_hashes == fresh.block_hashes
    assert request.cache_salt == fresh.cache_salt != caller_salt
    assert request._omni_original_cache_salt == caller_salt
    assert request._omni_conditioning_digest == digest
    assert list(old_ids) == [0, 0, 0, 0]

    # New accepted content has a new caller salt, not the prior composed salt.
    replacement = prepare_request_input(
        request,
        prompt_token_ids=[21, 22],
        mm_features=[],
        cache_salt="next-caller",
        sampling_params=request.sampling_params,
    )
    install_request_input(request, replacement)
    assert request.cache_salt == request._omni_original_cache_salt == "next-caller"
    assert request._omni_conditioning_digest is None


@pytest.mark.parametrize("invalid_digest", ["", "caller-label", "aa" * 31, "AA" * 32, "gg" * 32, 1, True])
def test_invalid_conditioning_digest_fails_before_input_mutation(invalid_digest):
    request = _request(None, [1, 2, 3, 4], None)
    old_state = dict(request.__dict__)
    with pytest.raises(ValueError, match="conditioning digest"):
        prepare_request_input(
            request,
            prompt_token_ids=[11, 12],
            mm_features=[],
            cache_salt="new-caller",
            sampling_params=request.sampling_params,
            conditioning_digest=invalid_digest,
        )
    assert request.__dict__ == old_state


def test_salt_encoding_is_versioned_unambiguous_and_preserves_empty_vs_absent():
    from vllm_omni.core.sched.input_finalization import compose_conditioning_cache_salt

    assert compose_conditioning_cache_salt("caller", "ab" * 32) == (
        "665819e6b1fd0843bd8a173865a6848f8bb719fe959028e0cb9decd60513a190"
    )
    salts = [None, "", "caller", "caller:ab", '"caller"', "不同用户"]
    assert len({compose_conditioning_cache_salt(salt, "ab" * 32) for salt in salts}) == len(salts)
