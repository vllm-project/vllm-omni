# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    EuclideanCodebook,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_book():
    book = EuclideanCodebook(8, 16).eval()
    with torch.no_grad():
        book.embedding_sum.normal_()
        book.cluster_usage.copy_(torch.linspace(0, 2, 16))
    return book


def reference(book, codes):
    return F.embedding(codes, book.embedding_sum / book.cluster_usage.clamp(min=book.epsilon)[:, None])


def test_cache_matches_reference_and_is_not_serialized():
    book = make_book()
    codes = torch.tensor([[0, 1, 15], [3, 7, 3]])
    book.precompute_embedding()
    cached = book._normalized_embedding
    with torch.inference_mode():
        torch.testing.assert_close(book.decode(codes), reference(book, codes), rtol=0, atol=0)
        book.decode(codes)
    assert book._normalized_embedding is cached
    assert set(book.state_dict()) == {"cluster_usage", "embedding_sum"}


def test_conversion_invalidates_cache_and_recomputes_from_weights():
    book = make_book()
    book.precompute_embedding()
    book.to(torch.bfloat16)
    assert book._normalized_embedding is None
    book.precompute_embedding()
    with torch.inference_mode():
        codes = torch.tensor([1, 3, 8])
        torch.testing.assert_close(book.decode(codes), reference(book, codes), rtol=0, atol=0)


def test_reload_and_inplace_loader_refresh():
    book = make_book()
    book.precompute_embedding()
    new = make_book()
    book.load_state_dict(new.state_dict())
    assert book._normalized_embedding is None
    book.precompute_embedding()
    with torch.no_grad():
        book.embedding_sum.add_(1)
    # vLLM's in-place loader explicitly calls this again after loading.
    book.precompute_embedding()
    with torch.inference_mode():
        codes = torch.tensor([2, 4])
        torch.testing.assert_close(book.decode(codes), reference(book, codes), rtol=0, atol=0)


def test_gradients_bypass_cache_and_training_invalidates():
    book = make_book()
    book.precompute_embedding()
    book.decode(torch.tensor([1, 2])).sum().backward()
    assert book.embedding_sum.grad is not None
    assert book.cluster_usage.grad is not None
    book.train()
    assert book._normalized_embedding is None
