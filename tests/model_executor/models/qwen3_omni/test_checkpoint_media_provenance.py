# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in CPU preprocessing qualification using a real local Qwen checkpoint.

Set VLLM_OMNI_TEST_QWEN3_MODEL to an immutable local snapshot. No weights are
loaded and no files are downloaded by these tests. These checks are not model
inference, GPU/NPU qualification, WER or performance evidence.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vllm.config import DeviceConfig, LoadConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import LruKeyReplicatedSenderCache, MultiModalProcessorOnlyCache
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.sampling_params import SamplingParams

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.inputs.input_processor import OmniInputProcessor
from vllm_omni.inputs.preprocess import build_omni_renderer
from vllm_omni.inputs.processed_media import ProcessedMediaProvenance

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def checkpoint_config():
    checkpoint = os.environ.get("VLLM_OMNI_TEST_QWEN3_MODEL")
    if not checkpoint:
        pytest.skip("set VLLM_OMNI_TEST_QWEN3_MODEL to qualify checkpoint preprocessing")
    assert checkpoint is not None
    assert Path(checkpoint, "config.json").is_file(), "qualification requires a local checkpoint snapshot"
    return OmniModelConfig(
        model=checkpoint,
        tokenizer=checkpoint,
        model_stage="thinker",
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        hf_config_name="thinker_config",
        dtype="bfloat16",
        max_model_len=4096,
        enforce_eager=True,
    )


def _media(modality, changed=False):
    if modality == "image":
        return Image.new("RGB", (56, 56), color=(80 if changed else 12, 40, 80))
    if modality == "video":
        return np.full((4, 56, 56, 3), 80 if changed else 12, dtype=np.uint8)
    frequency = 440 if changed else 200
    return np.sin(np.arange(16000, dtype=np.float32) * (2 * np.pi * frequency / 16000)) * 0.2


def _prompt(processor, modality):
    hf = processor.info.get_hf_processor()
    if modality == "audio":
        content = hf.audio_bos_token + hf.audio_token + hf.audio_eos_token
    else:
        content = hf.vision_bos_token + getattr(hf, f"{modality}_token") + hf.vision_eos_token
    return f"<|im_start|>user\n{content}Describe.<|im_end|>\n<|im_start|>assistant\n"


@pytest.mark.parametrize("cache_cls", [MultiModalProcessorOnlyCache, LruKeyReplicatedSenderCache])
@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_checkpoint_processing_and_cache_hits_keep_actual_media_proof(
    checkpoint_config, cache_cls, modality, monkeypatch
):
    import vllm_omni.model_executor.models.qwen3_omni.processed_media as digest_module

    cache = cache_cls(checkpoint_config)
    processor = MULTIMODAL_REGISTRY.create_processor(checkpoint_config)
    tokens = processor.info.get_tokenizer().encode(_prompt(processor, modality))

    def apply(value, label):
        return processor.apply(
            ProcessorInputs(tokens, processor.info.parse_mm_data({modality: value}), {modality: [label]}, cache=cache),
            TimingContext(enabled=False),
        )

    first = apply(_media(modality), "caller-label")
    proof = first["_omni_processed_media"]
    assert isinstance(proof, ProcessedMediaProvenance)
    assert len(proof.media) == 1 and proof.media[0].modality == modality
    assert proof.prompt_token_ids == tuple(first["prompt_token_ids"])
    assert proof.media[0].length > 0
    assert len(proof.media[0].content_digest) == len(proof.media[0].preprocessing_digest) == 64

    def forbidden_hash(_):
        raise AssertionError("checkpoint cache hit rehashed media")

    with monkeypatch.context() as context:
        context.setattr(digest_module, "processor_value_digest", forbidden_hash)
        hit = apply(_media(modality, changed=True), "caller-label")
    # Upstream UUID semantics reuse the old processed value; its old provenance
    # must follow that value, not describe the newly supplied raw media.
    assert hit["_omni_processed_media"] == proof
    if cache_cls is LruKeyReplicatedSenderCache:
        assert hit["mm_kwargs"][modality][0] is None
    changed = apply(_media(modality, changed=True), "different-label")
    assert changed["_omni_processed_media"].media[0].content_digest != proof.media[0].content_digest


def test_checkpoint_proof_survives_real_request_construction(checkpoint_config):
    config = VllmConfig(
        model_config=checkpoint_config,
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
    )
    renderer = build_omni_renderer(config)
    processor = OmniInputProcessor(config, renderer)
    request = processor.process_inputs(
        request_id="checkpoint-image",
        prompt={
            "prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n"
            "<|im_start|>assistant\n",
            "multi_modal_data": {"image": _media("image")},
        },
        params=SamplingParams(max_tokens=4, temperature=0),
        supported_tasks=("generate",),
    )
    proof = request.processed_media_provenance
    assert isinstance(proof, ProcessedMediaProvenance)
    proof.validate(request.prompt_token_ids, request.mm_features)
    assert len(proof.media) == 1
