# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU unit tests for QwenImage21Pipeline helpers: prepare_latents duplication
order, KV-cache assembly across step phases, and the post-process func."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    QwenImage21Pipeline,
    get_qwen_image_21_post_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _toy_pipeline() -> QwenImage21Pipeline:
    """A pipeline shell with the attributes prepare_latents reads, no weights."""
    pipeline = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 16
    pipeline.latent_channels = 4
    return pipeline


# ---------------------------------------------------------------------------
# prepare_latents
# ---------------------------------------------------------------------------


def test_prepare_latents_duplicates_condition_latents_request_major():
    """batch=2 prompts x n=2 outputs must order condition latents A,A,B,B.

    ``encode_prompt`` expands prompt embeddings request-major (A,A,B,B); the
    condition latents must follow the same order or generations 2 and 3 swap
    condition images. ``repeat`` would produce the wrong (A,B,A,B) tile-major
    order; ``repeat_interleave`` is correct.
    """
    pipeline = _toy_pipeline()
    cond_a = torch.full((1, 4, 1, 8, 8), 1.0)
    cond_b = torch.full((1, 4, 1, 8, 8), 2.0)
    stacked = torch.cat([cond_a, cond_b], dim=0)  # one row per request

    latents, image_latents = pipeline.prepare_latents(
        [stacked],
        batch_size=4,
        num_channels_latents=4,
        height=64,
        width=64,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )

    assert latents.shape == (4, 16, 4)  # packed: 4x4 latent tokens, 4 channels
    assert image_latents.shape == (4, 64, 4)  # packed: 8x8 condition tokens
    row_means = image_latents.mean(dim=(1, 2))
    torch.testing.assert_close(row_means, torch.tensor([1.0, 1.0, 2.0, 2.0]))


def test_prepare_latents_without_duplication_when_rows_match_batch():
    pipeline = _toy_pipeline()
    stacked = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1, 1).expand(4, 4, 1, 2, 2).contiguous()

    _, image_latents = pipeline.prepare_latents(
        [stacked],
        batch_size=4,
        num_channels_latents=4,
        height=32,
        width=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(image_latents.mean(dim=(1, 2)), torch.arange(4, dtype=torch.float32))


def test_prepare_latents_rejects_non_divisible_duplication():
    pipeline = _toy_pipeline()
    stacked = torch.zeros(2, 4, 1, 8, 8)

    with pytest.raises(ValueError, match="Cannot duplicate"):
        pipeline.prepare_latents(
            [stacked],
            batch_size=3,
            num_channels_latents=4,
            height=64,
            width=64,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(0),
        )


def test_prepare_latents_rejects_generator_list_length_mismatch():
    pipeline = _toy_pipeline()

    with pytest.raises(ValueError, match="list of generators"):
        pipeline.prepare_latents(
            None,
            batch_size=2,
            num_channels_latents=4,
            height=32,
            width=32,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=[torch.Generator().manual_seed(0)],
        )


def test_prepare_latents_concatenates_multiple_condition_images():
    pipeline = _toy_pipeline()
    first = torch.full((2, 4, 1, 4, 4), 1.0)
    second = torch.full((2, 4, 1, 2, 2), 3.0)

    _, image_latents = pipeline.prepare_latents(
        [first, second],
        batch_size=2,
        num_channels_latents=4,
        height=32,
        width=32,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )

    # Token dim concatenates the packed condition images: 16 + 4 per row.
    assert image_latents.shape == (2, 20, 4)
    torch.testing.assert_close(image_latents[:, :16].mean(), torch.tensor(1.0))
    torch.testing.assert_close(image_latents[:, 16:].mean(), torch.tensor(3.0))


# ---------------------------------------------------------------------------
# KV-cache assembly across step phases
# ---------------------------------------------------------------------------


def _state(cache) -> SimpleNamespace:
    return SimpleNamespace(extra={"kv_cache": cache})


def _decode_cache(rows: int, value: float) -> list[dict]:
    return [
        {
            "cond": {
                "key": torch.full((rows, 2), value),
                "value": torch.full((rows, 2), value + 0.5),
            }
        }
    ]


def test_kv_cache_phase_classification():
    assert QwenImage21Pipeline._kv_cache_phase(None) == "none"
    assert QwenImage21Pipeline._kv_cache_phase([{}]) == "prefill"
    assert QwenImage21Pipeline._kv_cache_phase(_decode_cache(1, 0.0)) == "decode"


def test_assemble_kv_cache_all_none():
    assert QwenImage21Pipeline._assemble_kv_cache([_state(None), _state(None)]) == (None, False)


def test_assemble_kv_cache_prefill_returns_fresh_owned_cache():
    states = [_state([{} for _ in range(3)]), _state([{} for _ in range(3)])]

    kv_cache, take_ownership = QwenImage21Pipeline._assemble_kv_cache(states)

    assert take_ownership is True
    assert kv_cache == [{}, {}, {}]
    # Fresh blocks, not aliases of either request's cache: the batched prefill
    # writes into these and _scatter_kv_cache splits them back per request.
    for block in kv_cache:
        assert all(block is not request_block for state in states for request_block in state.extra["kv_cache"])


def test_assemble_kv_cache_decode_concatenates_in_request_order():
    first, second = _decode_cache(2, 1.0), _decode_cache(1, 3.0)
    states = [_state(first), _state(second)]

    kv_cache, take_ownership = QwenImage21Pipeline._assemble_kv_cache(states)

    assert take_ownership is False
    assert len(kv_cache) == 1
    cond = kv_cache[0]["cond"]
    torch.testing.assert_close(cond["key"], torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0]]))
    torch.testing.assert_close(cond["value"], torch.tensor([[1.5, 1.5], [1.5, 1.5], [3.5, 3.5]]))
    # Decode only reads the cache: per-request tensors must stay untouched.
    assert first[0]["cond"]["key"].shape == (2, 2)
    assert second[0]["cond"]["key"].shape == (1, 2)


def test_assemble_kv_cache_decode_merges_every_part():
    first = [{"cond": {"key": torch.ones(1, 1), "value": torch.ones(1, 1), "key_scale": torch.ones(1)}}]
    second = [{"cond": {"key": torch.zeros(1, 1), "value": torch.zeros(1, 1), "key_scale": torch.zeros(1)}}]

    kv_cache, take_ownership = QwenImage21Pipeline._assemble_kv_cache([_state(first), _state(second)])

    assert take_ownership is False
    assert set(kv_cache[0]["cond"]) == {"key", "value", "key_scale"}
    torch.testing.assert_close(kv_cache[0]["cond"]["key_scale"], torch.tensor([1.0, 0.0]))


def test_assemble_kv_cache_mixed_phases_raises():
    """Defensive backstop: a late prefill must never join a decoding batch.

    The preprocessor sets ``allow_mixed_step_phases = False`` so the
    StepScheduler defers late requests until the batch finishes decoding; this
    raise is the last line of defense if a mixed batch is ever assembled.
    """
    states = [_state([{}]), _state(_decode_cache(1, 0.0))]

    with pytest.raises(ValueError, match="mixed KV-cache phases"):
        QwenImage21Pipeline._assemble_kv_cache(states)

    with pytest.raises(ValueError, match="mixed KV-cache phases"):
        QwenImage21Pipeline._assemble_kv_cache([_state(None), _state(_decode_cache(1, 0.0))])


def test_scatter_kv_cache_splits_rows_back_per_request():
    states = [_state([{}]), _state([{}])]
    merged = [
        {
            "cond": {
                "key": torch.tensor([[1.0], [2.0], [3.0]]),
                "value": torch.tensor([[4.0], [5.0], [6.0]]),
            }
        }
    ]

    QwenImage21Pipeline._scatter_kv_cache(states, [1, 2], merged)

    torch.testing.assert_close(states[0].extra["kv_cache"][0]["cond"]["key"], torch.tensor([[1.0]]))
    torch.testing.assert_close(states[0].extra["kv_cache"][0]["cond"]["value"], torch.tensor([[4.0]]))
    torch.testing.assert_close(states[1].extra["kv_cache"][0]["cond"]["key"], torch.tensor([[2.0], [3.0]]))
    assert states[1].extra["kv_cache"][0]["cond"]["key"].is_contiguous()


# ---------------------------------------------------------------------------
# post-process func
# ---------------------------------------------------------------------------


@pytest.fixture
def postprocess(tmp_path):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
    return get_qwen_image_21_post_process_func(SimpleNamespace(model=str(tmp_path)))


def test_postprocess_tensor_returns_pil_images(postprocess):
    from PIL import Image

    images = torch.rand(2, 4, 16, 16) * 2 - 1

    result = postprocess(images)

    assert len(result) == 2
    assert all(isinstance(image, Image.Image) for image in result)


def test_postprocess_envelope_unwraps_payload_and_keeps_metadata(postprocess):
    from PIL import Image

    envelope = {
        "payload": {"image": torch.rand(1, 4, 16, 16), "seed": 42},
        "metadata": {"request_id": "abc"},
    }

    result = postprocess(envelope)

    assert result["metadata"] == {"request_id": "abc"}
    assert result["payload"]["seed"] == 42
    assert len(result["payload"]["image"]) == 1
    assert isinstance(result["payload"]["image"][0], Image.Image)


def test_postprocess_envelope_requires_image_payload(postprocess):
    with pytest.raises(ValueError, match=r"payload\['image'\]"):
        postprocess({"payload": {"seed": 42}, "metadata": {}})


def test_postprocess_envelope_normalizes_non_dict_metadata(postprocess):
    result = postprocess({"payload": {"image": torch.rand(1, 4, 16, 16)}, "metadata": "bogus"})

    assert result["metadata"] == {}
