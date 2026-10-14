# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real processor/cache plumbing with small, deterministic CPU preprocessing."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature
from vllm.multimodal.cache import LruKeyReplicatedSenderCache, MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataItems, ProcessorBatchItems
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import BaseMultiModalProcessor, PromptReplacement

from vllm_omni.model_executor.models.qwen3_omni.processed_media import processor_value_digest
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerMultiModalProcessor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _items(data, validate=True):
    return MultiModalDataItems({key: ProcessorBatchItems(value, key) for key, value in data.items()})


class _Processor(Qwen3OmniMoeThinkerMultiModalProcessor):
    # Keep real Qwen apply/get_mm_prompt_updates and upstream cache/placeholder
    # code. Only the HF call/config are toy CPU preprocessing, without weights.
    def __init__(self, cache_cls):
        mm_config = SimpleNamespace(mm_processor_cache_gb=0.001, mm_hasher_algorithm="sha256")
        self.cache = None if cache_cls is None else cache_cls(SimpleNamespace(get_multimodal_config=lambda: mm_config))
        self.info = SimpleNamespace(
            model_id="processor-model",
            ctx=SimpleNamespace(get_mm_config=lambda: mm_config),
            parse_mm_data=_items,
            get_hf_processor=lambda: SimpleNamespace(to_dict=lambda: {"resize": 2}),
        )
        self.processed_items = 0

    def _get_hf_mm_data(self, mm_items):
        return BaseMultiModalProcessor._get_hf_mm_data(self, mm_items)

    def _apply_hf_processor_main(self, mm_items, hf_kwargs):
        values = mm_items.get("image", [])
        self.processed_items += len(values)
        return BatchFeature({"pixel_values": torch.tensor([[value * 2] for value in values], dtype=torch.float32)})

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return {"pixel_values": MultiModalFieldConfig.batched("image")}

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return [PromptReplacement(modality="image", target=[9], replacement=[90, 90])]


def _apply(processor, values, *, uuids=None, options=None, prompt=None):
    return processor.apply(
        ProcessorInputs(
            prompt if prompt is not None else [1] + [9] * len(values) + [2],
            _items({"image": values}),
            None if uuids is None else {"image": uuids},
            hf_processor_mm_kwargs=options or {},
            cache=processor.cache,
        ),
        TimingContext(enabled=False),
    )


@pytest.mark.parametrize("cache_cls", [None, MultiModalProcessorOnlyCache, LruKeyReplicatedSenderCache])
def test_proof_tracks_actual_processed_values_not_caller_uuid(cache_cls):
    processor = _Processor(cache_cls)
    first = _apply(processor, [3], uuids=["caller-label"])
    first_proof = first["_omni_processed_media"]
    assert first_proof.media[0].content_digest == processor_value_digest({"pixel_values": torch.tensor([6.0])})
    assert first_proof.routing_hashes == ("caller-label",)

    # A reused UUID returns the old cached CPU value by upstream's contract.
    # Its provenance must describe that value, never the newly supplied raw 7.
    second = _apply(processor, [7], uuids=["caller-label"])
    second_proof = second["_omni_processed_media"]
    if cache_cls is None:
        assert second_proof.media[0].content_digest != first_proof.media[0].content_digest
    else:
        assert second_proof == first_proof
        assert processor.processed_items == 1
    if cache_cls is LruKeyReplicatedSenderCache:
        assert second["mm_kwargs"]["image"][0] is None
        third = _apply(processor, [None], uuids=["caller-label"])
        assert third["_omni_processed_media"] == first_proof


def test_sender_hit_neither_rehashes_nor_keeps_a_second_tensor_cache(monkeypatch):
    import vllm_omni.model_executor.models.qwen3_omni.processed_media as module

    processor = _Processor(LruKeyReplicatedSenderCache)
    first = _apply(processor, [4])

    def no_hash(_):
        raise AssertionError("cache hit rehashed media")

    monkeypatch.setattr(module, "processor_value_digest", no_hash)
    second = _apply(processor, [4])
    assert second["_omni_processed_media"] == first["_omni_processed_media"]
    assert second["mm_kwargs"]["image"][0] is None
    assert processor.cache is not None
    processor.cache.clear_cache()
    with pytest.raises(AssertionError, match="rehashed"):
        _apply(processor, [4])


def test_cached_proof_rebinds_to_current_item_order_and_positions():
    processor = _Processor(LruKeyReplicatedSenderCache)
    first = _apply(processor, [3, 4])
    second = _apply(processor, [4, 3], prompt=[1, 8, 9, 8, 9, 2])
    original = first["_omni_processed_media"].media
    current = second["_omni_processed_media"].media
    assert [item.offset for item in current] == [2, 5]
    assert current[0].content_digest == original[1].content_digest
    assert current[1].content_digest == original[0].content_digest
    assert processor.processed_items == 2


def test_different_options_change_preprocessing_even_with_same_processed_values():
    processor = _Processor(MultiModalProcessorOnlyCache)
    first = _apply(processor, [3], options={"resize": 1})["_omni_processed_media"].media[0]
    second = _apply(processor, [3], options={"resize": 2})["_omni_processed_media"].media[0]
    assert first.content_digest == second.content_digest
    assert first.preprocessing_digest != second.preprocessing_digest


def test_provenance_survives_upstream_update_replacement():
    processor = _Processor(MultiModalProcessorOnlyCache)
    _apply(processor, [3])
    assert processor.cache is not None
    cached = next(iter(processor.cache._cache.values())).prompt_updates[0]
    shifted = replace(cached, item_idx=3)
    assert shifted.content_digest == cached.content_digest
    assert shifted.item_idx == 3


def test_passthrough_embeddings_do_not_gain_trusted_processor_identity():
    class _Passthrough(ProcessorBatchItems):
        def get_passthrough_data(self):
            return {"caller_embeds": torch.ones(1, 2)}

    processor = _Processor(LruKeyReplicatedSenderCache)
    inputs = ProcessorInputs([1, 9, 2], MultiModalDataItems({"image": _Passthrough([3], "image")}))
    result = processor.apply(inputs, TimingContext(enabled=False))
    assert result["_omni_processed_media"] is None
    assert result["prompt_token_ids"] == [1, 90, 90, 2]


@pytest.mark.parametrize("value", [torch.empty(2, device="meta"), object(), {1: 2}, float("nan")])
def test_digest_rejects_unverifiable_values_without_device_copy(value):
    with pytest.raises(ValueError):
        processor_value_digest(value)


def test_digest_is_order_independent_but_sensitive_to_dtype_shape_and_values():
    base = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    digest = processor_value_digest({"data": base, "grid": [2, 3]})
    assert digest == processor_value_digest({"grid": [2, 3], "data": base.clone()})
    assert digest == processor_value_digest({"data": base.T.contiguous().T, "grid": [2, 3]})
    for changed in (base.float(), base.reshape(3, 2), base + 1):
        assert digest != processor_value_digest({"data": changed, "grid": [2, 3]})
    assert processor_value_digest([1, 23]) != processor_value_digest([12, 3])


def test_namespace_accepts_real_hf_processor_configuration_without_weights():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
    from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

    special = {
        key: f"<{key}>"
        for key in (
            "image_token",
            "audio_token",
            "video_token",
            "vision_bos_token",
            "vision_eos_token",
            "audio_bos_token",
            "audio_eos_token",
        )
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        extra_special_tokens=special,
    )
    hf = Qwen3OmniMoeProcessor(
        image_processor=Qwen2VLImageProcessor(),
        video_processor=Qwen2VLVideoProcessor(),
        feature_extractor=WhisperFeatureExtractor(),
        tokenizer=tokenizer,
    )
    processor = _Processor(LruKeyReplicatedSenderCache)
    processor.info.get_hf_processor = lambda: hf
    assert len(processor._media_preprocessing_namespace) == 64
    result = _apply(processor, [3])
    assert result["_omni_processed_media"] is not None
    assert "_media_preprocessing_namespace" in processor.__dict__
