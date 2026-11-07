# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Hunyuan RNG identity through the upstream input and cache paths."""

from dataclasses import replace
from itertools import count

import pytest
import torch
from PIL import Image
from transformers import BatchFeature, PretrainedConfig
from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.cache.lru import LruKeyReplicatedReceiverCache, LruKeyReplicatedSenderCache
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import InputProcessingContext, ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo
from vllm.renderers import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3DummyInputsBuilder,
    HunyuanImage3MultiModalProcessor,
    HunyuanImage3ProcessingInfo,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _input_processor(engine_input, mocker):
    processor = object.__new__(InputProcessor)
    processor.model_config = mocker.Mock(spec=ModelConfig, is_encoder_decoder=False, enable_prompt_embeds=True)
    parallel = mocker.Mock(
        spec=ParallelConfig, data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
    )
    processor.vllm_config = mocker.Mock(spec=VllmConfig, parallel_config=parallel)
    processor.lora_config = None
    processor.generation_config_fields = {}
    renderer = mocker.Mock(spec=BaseRenderer, tokenizer=None)
    renderer.default_cmpl_tok_params = TokenizeParams(max_total_tokens=128)
    renderer.render_cmpl.return_value = (engine_input,)
    renderer.get_eos_token_id.return_value = None
    processor.renderer = renderer
    processor._validate_params = mocker.Mock(return_value=False)
    processor._validate_lora = mocker.Mock()
    processor._validate_model_inputs = mocker.Mock()
    processor.apply_watermarking = mocker.Mock()
    return processor


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
        return (
            BatchFeature(
                {
                    "vae_token_grid_hw": torch.tensor([[2, 3]] * len(images)),
                    "vae_pixel_size": torch.tensor([18] * len(images)),
                    "vae_pixel_values": torch.ones(18 * len(images)),
                }
            )
            if images
            else BatchFeature()
        )

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


@pytest.fixture
def cached_request(real_processor, mocker):
    processor, _ = real_processor
    sender = LruKeyReplicatedSenderCache(processor.info.ctx.model_config)
    receiver = LruKeyReplicatedReceiverCache(processor.info.ctx.model_config)
    request_ids = count()

    def request(images, uuids, seed=0, raw=False):
        inputs = ProcessorInputs(
            [99],
            processor.info.get_data_parser().parse_mm_data({"image": images}),
            {"image": uuids},
            hf_processor_mm_kwargs={} if seed is None else {"vae_generator_seed": seed},
            cache=sender,
        )
        rendered = processor.apply(inputs, TimingContext(enabled=False))
        admission = _input_processor(rendered, mocker)
        result = admission.process_inputs(
            f"request-{next(request_ids)}",
            {"prompt": "image"} if raw else rendered,
            SamplingParams(seed=999),
            ("generate",),
        )
        assert admission.renderer.render_cmpl.call_count == int(raw)
        receiver.get_and_update_features(result.mm_features)
        return result

    return request


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("seed", [0, 42])
def test_seeded_reference_prefixes_and_lru_hits(cached_request, real_processor, raw, seed):
    _, image_counts = real_processor
    a, b = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    first = cached_request([a, a, b], ["A", "A", "B"], seed, raw).mm_features
    assert len({f.identifier for f in first}) == 3
    assert all(f.identifier == f.mm_hash for f in first)
    assert all(f.data["vae_generator_seed"].data.item() == seed for f in first)

    repeated = cached_request([None, None, None], ["A", "A", "B"], seed, raw).mm_features
    assert [f.identifier for f in repeated] == [f.identifier for f in first]
    assert all(hit.data is original.data for hit, original in zip(repeated, first))

    moved = cached_request([b, a, a], ["B", "A", "A"], seed, raw).mm_features
    assert len({f.identifier for f in moved}) == 3
    assert not {f.identifier for f in first} & {f.identifier for f in moved}
    assert image_counts == [3, 0, 3]

    prefix = cached_request([None, None], ["A", "A"], seed, raw).mm_features
    assert [f.identifier for f in prefix] == [f.identifier for f in first[:2]]
    other_seed = cached_request([a], ["A"], 42 if seed == 0 else 0, raw).mm_features
    assert other_seed[0].identifier != first[0].identifier


@pytest.mark.parametrize("seed", [0, 42])
def test_uuid_only_reference_requires_warm_prefix(cached_request, seed):
    a, b = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    cached_request([a], ["A"], seed)
    with pytest.raises(ValueError, match="Cache miss for image at index 1 but data is not provided"):
        cached_request([b, None], ["B", "A"], seed)


def test_seedless_references_reuse_images_across_prefixes(cached_request):
    a, b = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    first = cached_request([a, a], ["A", "A"], seed=None).mm_features
    assert first[0].identifier == first[1].identifier
    moved = cached_request([b, None], ["B", "A"], seed=None).mm_features
    assert moved[1].identifier == first[0].identifier
    assert moved[1].data is first[0].data
    assert "vae_generator_seed" not in moved[1].data


def test_mrope_prefix_stripping_retains_seed_and_grid(cached_request):
    images = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    request = Request.from_engine_core_request(cached_request(images, ["A", "B"]), block_hasher=None)
    request.num_computed_tokens = 2
    features = request.mm_features
    result = NewRequestData.from_request(request, ([0],), uses_mrope=True)
    assert set(result.mm_features[0].data) == {"vae_generator_seed", "vae_token_grid_hw"}
    assert result.mm_features[0].data["vae_token_grid_hw"] is features[0].data["vae_token_grid_hw"]
    assert result.mm_features[0].data["vae_generator_seed"].data.item() == 0
    assert result.mm_features[1].data is features[1].data
    assert "vae_pixel_values" in features[0].data


def test_profile_dummy_inputs_do_not_inherit_configured_request_seed(real_processor):
    processor, image_counts = real_processor
    processor.info.ctx.model_config.multimodal_config.mm_processor_kwargs = {"vae_generator_seed": 42}
    result = processor.get_dummy_mm_inputs({"image": 1})
    assert image_counts == [1]
    assert "vae_generator_seed" not in result["mm_kwargs"]["image"][0]


@pytest.mark.parametrize("seed", [-(2**63) - 1, 2**63, 1.5, "42", True])
def test_invalid_seed_is_rejected_before_processor_cache_mutation(seed, real_processor, mocker):
    processor, image_counts = real_processor
    sender = LruKeyReplicatedSenderCache(processor.info.ctx.model_config)
    inputs = ProcessorInputs(
        [99],
        processor.info.get_data_parser().parse_mm_data({"image": Image.new("RGB", (4, 4), "red")}),
        {"image": ["image-uuid"]},
        hf_processor_mm_kwargs={"vae_generator_seed": seed},
        cache=sender,
    )
    cache_write = mocker.spy(sender, "get_and_update_item")
    with pytest.raises(VLLMValidationError, match="vae_generator_seed"):
        processor.apply(inputs, TimingContext(enabled=False))
    assert image_counts == []
    cache_write.assert_not_called()
    valid = processor.apply(
        replace(inputs, hf_processor_mm_kwargs={"vae_generator_seed": 0}), TimingContext(enabled=False)
    )
    assert valid["mm_kwargs"]["image"][0] is not None
