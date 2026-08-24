# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the SenseNova-Vision end2end example formatters.

Guards the #2/#3 divergence bug class permanently: every formatter's prompt
must carry exactly the scaffold markers its ``multi_modal_data`` requires, so

* N ``<|image_pad|>`` markers  <-> N items under ``multi_modal_data["image"]``
  (understanding path, ViT-only),
* N ``<|fim_middle|>`` markers <-> N items under ``multi_modal_data["img2img"]``
  (generation-conditioning path, VAE+ViT),

and feeding each formatted prompt through the registered AR multimodal
processor leaves zero unmatched placeholders (vLLM raises ``RuntimeError`` on a
mm item with no placeholder at request time).

All tests are CPU-only; the tokenizer comes from the locally cached
SenseNova-Vision-7B-MoT checkpoint with ``local_files_only=True``. No model
weights are loaded.
"""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import pytest
from PIL import Image
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems

from examples.offline_inference.sensenova_vision.end2end import (
    _format_dense_detection_prompts,
    _format_dense_ocr_prompts,
    _format_img2dense_prompts,
    _format_img2img_prompts,
    _format_img2text_prompts,
    _format_mixed_prompts,
    _format_multi_img2text_prompts,
    _format_recon3d_prompts,
    _format_text2img_prompts,
    _format_text2text_prompts,
    _format_think_img2img_prompts,
    _format_think_text2img_prompts,
    _format_think_text2text_prompts,
)
from vllm_omni.model_executor.models.bagel.bagel import Img2ImgProcessorItems

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_FIM_MIDDLE = "<|fim_middle|>"
_IMAGE_PAD = "<|image_pad|>"


def _cached_checkpoint() -> str | None:
    """The checkpoint root if cached locally (same lookup as the tokenizer tests)."""
    env_path = os.environ.get("SENSENOVA_VISION_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--sensenova--SenseNova-Vision-7B-MoT/snapshots/*")
    matches = sorted(glob.glob(snapshot))
    return matches[-1] if matches else None


@pytest.fixture(scope="module")
def tokenizer():
    snap = _cached_checkpoint()
    if snap is None:
        pytest.skip("SenseNova-Vision-7B-MoT not cached and SENSENOVA_VISION_MODEL_PATH is unset")
    from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
        VLLMSenseNovaVisionTokenizer,
    )

    return VLLMSenseNovaVisionTokenizer.from_pretrained(
        snap,
        local_files_only=True,
        trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def hf_config():
    """Faithful stand-in for the merged SenseNovaVision BagelConfig."""
    return SimpleNamespace(
        vit_max_num_patch_per_side=70,
        latent_patch_size=2,
        max_latent_size=64,
        vae_config={"downsample": 8, "z_channels": 16},
        vit_config=SimpleNamespace(image_size=980, patch_size=14),
    )


class _StubCtx(SimpleNamespace):
    """Duck-typed ``InputProcessingContext``: only what the processing-info /
    processor paths under test actually touch."""

    def get_tokenizer(self):
        return self.tokenizer

    def get_hf_config(self):
        return self.hf_config


@pytest.fixture()
def info_ctx(tokenizer, hf_config):
    checkpoint = _cached_checkpoint()
    assert checkpoint is not None
    return _StubCtx(
        tokenizer=tokenizer,
        hf_config=hf_config,
        model_config=SimpleNamespace(
            model=checkpoint,
            get_multimodal_config=lambda: SimpleNamespace(enable_mm_embeds=False),
        ),
    )


def _make_processor(info):
    """A real OmniBagelMultiModalProcessor whose info is injected."""
    from vllm_omni.model_executor.models.bagel import bagel as bagel_module

    proc = object.__new__(bagel_module.OmniBagelMultiModalProcessor)
    proc.info = info
    proc.dummy_inputs = None
    proc.cache = None
    proc.data_parser = info.get_data_parser()
    return proc


def _assert_prompt_matches_mm_data(prompt: dict) -> tuple[int, int]:
    """Scaffold-vs-mm-data consistency checks shared by every formatter test.

    Returns ``(num_pads, num_fims)`` found in the prompt string.
    """
    text = prompt["prompt"]
    num_pads = text.count(_IMAGE_PAD)
    num_fims = text.count(_FIM_MIDDLE)
    mm_data = prompt.get("multi_modal_data") or {}
    modalities = prompt["modalities"]

    n_images = len(mm_data.get("image", [])) if isinstance(mm_data.get("image"), list) else int("image" in mm_data)
    n_img2img = (
        len(mm_data.get("img2img", [])) if isinstance(mm_data.get("img2img"), list) else int("img2img" in mm_data)
    )

    assert num_pads == n_images, f"{num_pads} <|image_pad|> markers must bind {n_images} image items"
    assert num_fims == n_img2img, f"{num_fims} <|fim_middle|> markers must bind {n_img2img} img2img items"

    if "image" in mm_data:
        assert modalities == ["text"], "the <|image_pad|> understanding path runs under modalities=['text']"
    if "img2img" in mm_data:
        assert modalities == ["img2img"], "the <|fim_middle|> conditioning path must declare modalities=['img2img']"
    if not mm_data:
        assert "image" not in modalities or modalities == ["image"], "generation scaffolds carry no mm data"
    return num_pads, num_fims


def _placeholders_bind_all_items(prompt: dict, tokenizer, info_ctx) -> None:
    """Feed the formatted prompt through the registered processor expansion.

    Asserts every mm item binds to exactly one placeholder range (no unmatched
    placeholders -> no RuntimeError at request time).
    """
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionProcessingInfo,
    )

    text = prompt["prompt"]
    prompt_ids = tokenizer.encode(text)
    mm_data = prompt.get("multi_modal_data") or {}

    image_items = mm_data.get("image") or []
    img2img_items = mm_data.get("img2img") or []
    if not image_items and not img2img_items:
        # Text-only prompts have nothing to bind; just require the chat
        # markers survive tokenization as single special tokens.
        assert tokenizer.convert_tokens_to_ids(_IM_START) in prompt_ids
        return

    images = [image_items] if not isinstance(image_items, list) else list(image_items)
    img2imgs = [img2img_items] if not isinstance(img2img_items, list) else list(img2img_items)

    info = OmniSenseNovaVisionProcessingInfo(info_ctx)
    proc = _make_processor(info)
    parsed: dict[str, object] = {}
    if images:
        parsed["image"] = ImageProcessorItems(images)
    if img2imgs:
        parsed["img2img"] = Img2ImgProcessorItems(img2imgs)
    mm_items = MultiModalDataItems(parsed)

    updates = proc._get_prompt_updates(mm_items, {}, MultiModalKwargsItems())
    mm_prompt_updates = proc._bind_and_group_updates(updates, mm_items.get_all_counts())
    new_ids, placeholders = proc._apply_prompt_updates(prompt_ids, mm_prompt_updates)

    expected = {"image": len(images), "img2img": len(img2imgs)}
    for modality, count in expected.items():
        phs = placeholders.get(modality, [])
        assert len(phs) == count, (
            f"{count} {modality} item(s) must bind to {count} placeholder ranges, got {len(phs)} (prompt={text!r})"
        )
        for i, ph in enumerate(phs):
            assert ph.item_idx == i
    # Expansion must produce non-empty placeholder ranges for each modality.
    for modality in ("image", "img2img"):
        if modality not in placeholders:
            continue
        covered = set()
        for ph in placeholders[modality]:
            covered.update(range(ph.start_idx, ph.start_idx + len(ph.tokens)))
        assert covered, f"{modality}: placeholder ranges must be non-empty"


# ---------------------------------------------------------------------------
# Scaffold shape assertions per formatter (cheap, always run)
# ---------------------------------------------------------------------------


def test_text2text_scaffold():
    (p,) = _format_text2text_prompts(["What is the capital of France?"])
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 0)
    assert p["prompt"].startswith(f"{_IM_START}user\n")
    assert p["prompt"].endswith(f"{_IM_END}\n{_IM_START}assistant\n")


def test_img2text_scaffold():
    img = Image.new("RGB", (32, 32))
    (p,) = _format_img2text_prompts(["describe"], img)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (1, 0)


@pytest.mark.parametrize(
    "formatter",
    [_format_dense_detection_prompts, _format_dense_ocr_prompts],
)
def test_dense_understanding_scaffolds(formatter):
    img = Image.new("RGB", (32, 32))
    (p,) = formatter(["detect things"], img)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (1, 0)


def test_text2img_scaffold():
    (p,) = _format_text2img_prompts(["a corgi astronaut"])
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 0)
    # Generation scaffold has NO user/assistant turns.
    assert p["prompt"] == f"{_IM_START}a corgi astronaut{_IM_END}"


def test_img2img_scaffold():
    img = Image.new("RGB", (32, 32))
    (p,) = _format_img2img_prompts(["edit this"], img)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 1)
    assert p["prompt"].startswith(_FIM_MIDDLE)


def test_img2dense_scaffold():
    img = Image.new("RGB", (32, 32))
    (p,) = _format_img2dense_prompts(["depth map"], img)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 1)


def test_multi_img2text_one_pad_per_view():
    imgs = [Image.new("RGB", (32, 32)) for _ in range(3)]
    (p,) = _format_multi_img2text_prompts(["camera pose"], imgs)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (3, 0)


def test_recon3d_one_fim_per_view():
    imgs = [Image.new("RGB", (32, 32)) for _ in range(4)]
    (p,) = _format_recon3d_prompts(["reconstruct"], imgs)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 4)
    # All fim markers precede the chat-wrapped prompt (upstream interleaves
    # [*images, prompt]).
    assert p["prompt"].startswith(_FIM_MIDDLE * 4)
    assert p["modalities"] == ["img2img"]


def test_mixed_scaffold_conditions_via_img2img():
    """mixed keeps today's image-only behavior via the img2img key (#1 folding out of scope)."""
    img = Image.new("RGB", (32, 32))
    (p,) = _format_mixed_prompts(["caption it"], img)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 1)
    assert p["mode"] == "caption_generate"


def test_think_text2text_wraps_with_vlm_system_prompt():
    (p,) = _format_think_text2text_prompts(["what is here?"])
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 0)
    assert p["mode"] == "think_understanding"
    # system turn + user turn + opened assistant continuation.
    assert f"{_IM_START}system\n" in p["prompt"]
    assert p["prompt"].endswith(f"{_IM_START}user\nwhat is here?{_IM_END}\n{_IM_START}assistant\n")


def test_think_text2img_wraps_with_gen_system_prompt():
    (p,) = _format_think_text2img_prompts(["a corgi astronaut"])
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 0)
    assert p["mode"] == "think_generate"
    assert p["prompt"].startswith(f"{_IM_START}system\n")


def test_think_img2img_keeps_conditioning_marker():
    img = Image.new("RGB", (32, 32))
    (p,) = _format_think_img2img_prompts(["make it cartoon"], img)
    num_pads, num_fims = _assert_prompt_matches_mm_data(p)
    assert (num_pads, num_fims) == (0, 1)
    assert p["mode"] == "think_edit"
    assert p["prompt"].startswith(_FIM_MIDDLE)


# ---------------------------------------------------------------------------
# Processor-binding regression (real tokenizer + real expansion rules)
# ---------------------------------------------------------------------------


def test_processor_binds_single_image_understanding(tokenizer, info_ctx):
    img = Image.new("RGB", (64, 64))
    (p,) = _format_img2text_prompts(["describe this image"], img)
    _placeholders_bind_all_items(p, tokenizer, info_ctx)


def test_processor_binds_single_image_generation_conditioning(tokenizer, info_ctx):
    img = Image.new("RGB", (64, 64))
    (p,) = _format_img2img_prompts(["turn into a cartoon"], img)
    _placeholders_bind_all_items(p, tokenizer, info_ctx)


def test_processor_binds_multi_view_camera_pose(tokenizer, info_ctx):
    imgs = [Image.new("RGB", (64, 64)) for _ in range(3)]
    (p,) = _format_multi_img2text_prompts(["estimate camera pose"], imgs)
    _placeholders_bind_all_items(p, tokenizer, info_ctx)


def test_processor_binds_multi_view_recon3d(tokenizer, info_ctx):
    imgs = [Image.new("RGB", (512, 512)) for _ in range(4)]
    (p,) = _format_recon3d_prompts(["reconstruct the scene"], imgs)
    _placeholders_bind_all_items(p, tokenizer, info_ctx)
