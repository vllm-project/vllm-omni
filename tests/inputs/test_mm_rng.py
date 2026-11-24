# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Hunyuan RNG identity through the upstream input and cache paths."""

from dataclasses import replace
from uuid import uuid4

import pytest
import torch
from PIL import Image
from transformers import BatchFeature, PretrainedConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.cache.lru import LruKeyReplicatedReceiverCache, LruKeyReplicatedSenderCache
from vllm.multimodal.cache.shm import ShmObjectStoreSenderCache
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import InputProcessingContext, ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3DummyInputsBuilder,
    HunyuanImage3MultiModalProcessor,
    HunyuanImage3ProcessingInfo,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def real_processor(mocker):
    # Keep production hashing, caching and field splitting; only pixel
    # preprocessing and prompt expansion are lightweight substitutes.
    model_config = mocker.Mock(spec=ModelConfig, model="tiny-hunyuan", max_model_len=20)
    mm_config = MultiModalConfig(mm_processor_cache_gb=0.01)
    model_config.multimodal_config = mm_config
    model_config.get_multimodal_config.return_value = mm_config
    info = HunyuanImage3ProcessingInfo(InputProcessingContext(model_config, tokenizer=None))
    info.get_data_parser = mocker.Mock(return_value=MultiModalDataParser())
    info.get_hf_config = mocker.Mock(return_value=PretrainedConfig(image_base_size=4))
    image_counts = []

    def process_pixels(prompt, images, **kwargs):
        image_counts.append(len(images))
        return BatchFeature({"vae_token_grid_hw": torch.tensor([[2, 3]] * len(images))} if images else {})

    def placeholders(mm_items, **_):
        count = mm_items.get_count("image", strict=False)
        return [99] * (2 * count), {
            "image": [PlaceholderFeaturesInfo("image", i, 2 * i, [99, 99], None) for i in range(count)]
        }

    info.get_image_processor = mocker.Mock(return_value=process_pixels)
    processor = object.__new__(HunyuanImage3MultiModalProcessor)
    processor.info = info
    processor.dummy_inputs = HunyuanImage3DummyInputsBuilder(info)
    processor._get_mm_prompt_updates = lambda items, *_: {
        "image": [[] for _ in range(items.get_count("image", strict=False))]
    }
    processor._maybe_apply_prompt_updates = placeholders
    return processor, image_counts


def make_inputs(processor, images, uuids, seed, cache):
    return ProcessorInputs(
        [99],
        processor.info.get_data_parser().parse_mm_data({"image": images}),
        {"image": uuids},
        hf_processor_mm_kwargs={} if seed is None else {"vae_generator_seed": seed},
        cache=cache,
    )


@pytest.fixture
def cached_request(real_processor):
    processor, _ = real_processor
    sender = LruKeyReplicatedSenderCache(processor.info.ctx.model_config)
    receiver = LruKeyReplicatedReceiverCache(processor.info.ctx.model_config)

    def request(images, uuids, seed=0):
        inputs = make_inputs(processor, images, uuids, seed, sender)
        result = processor.apply(inputs, TimingContext(enabled=False))
        hashes = result["mm_hashes"]["image"]
        items = [
            receiver.get_and_update_item(item, key)
            for item, key in zip(result["mm_kwargs"]["image"], hashes, strict=True)
        ]
        return hashes, items

    return request


@pytest.mark.parametrize("seed", [0, 42])
def test_seeded_reference_prefixes_and_lru_hits(cached_request, real_processor, seed):
    _, image_counts = real_processor
    a, b = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    first, items = cached_request([a, a, b], ["A", "A", "B"], seed)
    assert len(set(first)) == 3
    assert all(item["vae_generator_seed"].data.item() == seed for item in items)
    repeated, hits = cached_request([None, None, None], ["A", "A", "B"], seed)
    assert repeated == first
    assert all(hit is item for hit, item in zip(hits, items, strict=True))
    moved, _ = cached_request([b, a, a], ["B", "A", "A"], seed)
    assert len(set(moved)) == 3 and not set(first) & set(moved)
    assert image_counts == [3, 0, 3]
    prefix, _ = cached_request([None, None], ["A", "A"], seed)
    assert prefix == first[:2]
    mixed, items = cached_request([None, None, a], ["A", "A", "A"], seed)
    assert mixed[:2] == first[:2] and len(set(mixed)) == 3
    assert all(item["vae_generator_seed"].data.item() == seed for item in items)
    assert image_counts[-1] == 1  # Validation sees only the uncached occurrence.
    other_seed, _ = cached_request([a], ["A"], 42 if seed == 0 else 0)
    assert other_seed[0] != first[0]
    with pytest.raises(ValueError, match="Cache miss for image at index 2 but data is not provided"):
        cached_request([a, b, None], ["A", "B", "A"], seed)


def test_seedless_references_reuse_images_across_prefixes(cached_request):
    a, b = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    first, items = cached_request([a, a], ["A", "A"], seed=None)
    moved, hits = cached_request([b, None], ["B", "A"], seed=None)
    assert first[0] == first[1] == moved[1]
    assert hits[1] is items[0] and "vae_generator_seed" not in hits[1]


@pytest.mark.parametrize("use_v2", [False, True])
def test_profile_dummy_inputs_do_not_inherit_configured_request_seed(real_processor, use_v2):
    processor, image_counts = real_processor
    processor.info.ctx.model_config.use_v2_model_runner = use_v2
    processor.info.ctx.model_config.multimodal_config.mm_processor_kwargs = {"vae_generator_seed": 42}
    result = processor.get_dummy_mm_inputs({"image": 1})
    assert image_counts == [1]
    assert "vae_generator_seed" not in result["mm_kwargs"]["image"][0]


@pytest.mark.parametrize("seed", [0, 42])
def test_mrv2_accepts_seeded_images_and_uuid_cache_hits(real_processor, processor_cache, seed):
    processor, image_counts = real_processor
    processor.info.ctx.model_config.use_v2_model_runner = True
    inputs = make_inputs(processor, [Image.new("RGB", (4, 4), "red")], ["image-uuid"], seed, processor_cache)
    first = processor.apply(inputs, TimingContext(enabled=False))
    assert first["mm_kwargs"]["image"][0] is not None
    inputs = replace(inputs, mm_data_items=processor.info.get_data_parser().parse_mm_data({"image": [None]}))
    cached = processor.apply(inputs, TimingContext(enabled=False))
    assert cached["mm_hashes"] == first["mm_hashes"] and image_counts == [1, 0]
    item = cached["mm_kwargs"]["image"][0]
    assert item is None or "address" in item


def test_mrv2_allows_seed_without_images(real_processor):
    processor, _ = real_processor
    processor.info.ctx.model_config.use_v2_model_runner = True
    inputs = make_inputs(processor, [], [], 42, None)
    result = processor.apply(inputs, TimingContext(enabled=False))
    assert not result["mm_kwargs"].get("image")


@pytest.mark.parametrize("seed", [-(2**63) - 1, 2**63, 1.5, "42", True])
def test_invalid_seed_is_rejected_before_processor_cache_mutation(seed, real_processor, mocker):
    processor, image_counts = real_processor
    sender = LruKeyReplicatedSenderCache(processor.info.ctx.model_config)
    inputs = make_inputs(processor, [Image.new("RGB", (4, 4), "red")], ["image-uuid"], seed, sender)
    cache_write = mocker.spy(sender, "get_and_update_item")
    with pytest.raises(VLLMValidationError, match="vae_generator_seed"):
        processor.apply(inputs, TimingContext(enabled=False))
    assert image_counts == []
    cache_write.assert_not_called()
    valid = processor.apply(
        replace(inputs, hf_processor_mm_kwargs={"vae_generator_seed": 0}), TimingContext(enabled=False)
    )
    assert valid["mm_kwargs"]["image"][0] is not None


@pytest.fixture(params=["lru", "shm"])
def processor_cache(request, real_processor, mocker, monkeypatch):
    processor, _ = real_processor
    model_config = processor.info.ctx.model_config
    if request.param == "lru":
        yield LruKeyReplicatedSenderCache(model_config)
        return
    monkeypatch.setattr("vllm.envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME", f"hunyuan-input-{uuid4().hex}")
    model_config.multimodal_config.mm_shm_cache_max_object_size_mb = 1
    config = mocker.Mock(spec=VllmConfig, model_config=model_config)
    config.parallel_config = mocker.Mock(world_size=1)
    cache = ShmObjectStoreSenderCache(config)
    try:
        yield cache
    finally:
        cache.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vae_generator_seed", None),
        ("vae_generator_seed", [0]),
        ("vae_generator_seed", torch.tensor([0.0])),
        ("vae_generator_seed", torch.tensor(0)),
        ("vae_generator_seed", torch.tensor([1])),
        ("vae_generator_seed", torch.tensor([0, 0])),
        ("vae_generator_seed", torch.empty((1,), dtype=torch.long, device="meta")),
        ("vae_token_grid_hw", None),
        ("vae_token_grid_hw", [[2, 3]]),
        ("vae_token_grid_hw", torch.tensor([[2.0, 3.0]])),
        ("vae_token_grid_hw", torch.tensor([2, 3])),
        ("vae_token_grid_hw", torch.tensor([[0, 3]])),
        ("vae_token_grid_hw", torch.tensor([[-1, 3]])),
        ("vae_token_grid_hw", torch.tensor([[2, 3], [2, 3]])),
        ("vae_token_grid_hw", torch.empty((1, 2), dtype=torch.long, device="meta")),
    ],
)
def test_invalid_rng_metadata_is_rejected_before_cache_write(field, value, processor_cache, real_processor, mocker):
    processor, _ = real_processor
    original = processor._call_hf_processor

    def malformed_output(*args, **kwargs):
        result = original(*args, **kwargs)
        if value is None:
            result.pop(field)
        else:
            result[field] = value
        return result

    processor._call_hf_processor = malformed_output
    inputs = make_inputs(processor, [Image.new("RGB", (4, 4), "red")], ["image-uuid"], 0, processor_cache)
    cache_write = mocker.spy(processor_cache, "get_and_update_item")
    with pytest.raises(VLLMValidationError, match=field):
        processor.apply(inputs, TimingContext(enabled=False))
    cache_write.assert_not_called()

    # The same cache key must still accept valid pixels after rejection. Its
    # subsequent UUID-only hit contains None (LRU) or an address (SHM), not RNG fields.
    processor._call_hf_processor = original
    valid = processor.apply(inputs, TimingContext(enabled=False))
    assert valid["mm_kwargs"]["image"][0] is not None
    cached = processor.apply(
        replace(inputs, mm_data_items=processor.info.get_data_parser().parse_mm_data({"image": [None]})),
        TimingContext(enabled=False),
    )
    item = cached["mm_kwargs"]["image"][0]
    assert item is None or "address" in item
