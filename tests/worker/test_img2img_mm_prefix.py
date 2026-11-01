# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bagel's real prompt mask retains both image-prefix attention spans."""

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.multimodal.parse import MultiModalDataItems
from vllm.v1.attention.backends import utils

from vllm_omni.model_executor.models.bagel.bagel import Img2ImgProcessorItems, OmniBagelMultiModalProcessor
from vllm_omni.utils.mm_prefix import apply_img2img_mm_prefix_patch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def bagel_feature():
    processor = object.__new__(OmniBagelMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(
            vit_config=SimpleNamespace(image_size=980, patch_size=14),
            latent_patch_size=2,
            vae_config={"downsample": 8},
            max_latent_size=64,
        ),
        get_tokenizer=lambda: SimpleNamespace(get_vocab=lambda: {"<|fim_middle|>": 42}),
    )
    mm_items = MultiModalDataItems({"img2img": Img2ImgProcessorItems([Image.new("RGB", (1024, 672))])})
    update = processor._get_prompt_updates(mm_items, {}, {})[0].resolve(0)
    assert update.modality == "img2img"
    mask = update.content.is_embed(update.content.full)
    assert mask.device.type == "cpu"
    position = PlaceholderRange(offset=0, length=len(update.content.full), is_embed=mask)
    # Includes the VAE and ViT markers, with one non-embedding separator.
    assert position.extract_embeds_range() == [(0, 2689), (2691, 7592)]
    return MultiModalFeatureSpec(
        data=object(), modality="img2img", identifier="encoder-id", mm_position=position, mm_hash="processor-id"
    )


def test_real_bagel_img2img_ranges(bagel_feature):
    features = [[bagel_feature]]
    assert utils.compute_mm_prefix_ranges(features) == {0: [(0, 2689), (2691, 7592)]}
    assert features[0][0] is bagel_feature
    assert bagel_feature.modality == "img2img"


@pytest.mark.parametrize(
    "sliding_window,clamp_sliding_window,span_pad",
    [(None, False, 0), (1024, False, 0), (1024, True, 0), (None, False, 16)],
)
def test_delegates_native_range_options(bagel_feature, sliding_window, clamp_sliding_window, span_pad):
    original = utils.compute_mm_prefix_ranges._omni_mm_prefix_original
    expected = original([[replace(bagel_feature, modality="image")]], sliding_window, clamp_sliding_window, span_pad)
    assert utils.compute_mm_prefix_ranges([[bagel_feature]], sliding_window, clamp_sliding_window, span_pad) == expected


def test_other_modalities_and_metadata_unchanged(bagel_feature):
    original = utils.compute_mm_prefix_ranges._omni_mm_prefix_original
    others = [replace(bagel_feature, modality=modality) for modality in ("image", "video", "audio", "prompt_embeds")]
    assert utils.compute_mm_prefix_ranges([others, []]) == original([others, []])
    identity = (bagel_feature.data, bagel_feature.mm_position, bagel_feature.identifier, bagel_feature.mm_hash)
    assert utils.compute_mm_prefix_ranges([[bagel_feature], others, []]) == {
        0: [(0, 2689), (2691, 7592)],
        1: original([others])[0],
        2: [],
    }
    assert bagel_feature.data is identity[0]
    assert bagel_feature.mm_position is identity[1]
    assert bagel_feature.identifier is identity[2]
    assert bagel_feature.mm_hash is identity[3]
    assert bagel_feature.modality == "img2img"


def test_installation_handles_both_import_orders_and_is_idempotent(monkeypatch):
    original = utils.compute_mm_prefix_ranges._omni_mm_prefix_original
    monkeypatch.setattr(utils, "compute_mm_prefix_ranges", original)
    runners = []
    for name in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_states.default"):
        module = ModuleType(name)
        setattr(module, "compute_mm_prefix_ranges", original)
        monkeypatch.setitem(sys.modules, name, module)
        runners.append(module)
    apply_img2img_mm_prefix_patch()
    wrapped = utils.compute_mm_prefix_ranges
    assert wrapped is not original
    assert all(module.compute_mm_prefix_ranges is wrapped for module in runners)
    namespace: dict[str, object] = {}
    exec("from vllm.v1.attention.backends.utils import compute_mm_prefix_ranges", namespace)
    assert namespace["compute_mm_prefix_ranges"] is wrapped
    apply_img2img_mm_prefix_patch()
    assert utils.compute_mm_prefix_ranges is wrapped


def test_conflicting_runner_alias_fails_before_any_rebinding(monkeypatch):
    original = utils.compute_mm_prefix_ranges._omni_mm_prefix_original
    monkeypatch.setattr(utils, "compute_mm_prefix_ranges", original)
    module = ModuleType("vllm.v1.worker.gpu_model_runner")
    setattr(module, "compute_mm_prefix_ranges", lambda: None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="Conflicting"):
        apply_img2img_mm_prefix_patch()
    assert utils.compute_mm_prefix_ranges is original


@pytest.mark.parametrize("version,raises", [((0, 32, 0), False), ((0, 33, 0), True)])
def test_missing_helper_old_support_guard_and_new_failure(monkeypatch, version, raises):
    import vllm.version

    monkeypatch.delattr(utils, "compute_mm_prefix_ranges")
    monkeypatch.setattr(vllm.version, "__version_tuple__", version)
    if raises:
        with pytest.raises(RuntimeError, match="missing compute_mm_prefix_ranges"):
            apply_img2img_mm_prefix_patch()
    else:
        apply_img2img_mm_prefix_patch()
