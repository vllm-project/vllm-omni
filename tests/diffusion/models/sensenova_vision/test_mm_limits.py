# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for SenseNova-Vision multi-image support in the AR stage.

Covers four things:

1. ``OmniSenseNovaVisionProcessingInfo`` raises the supported mm limits to
   ``{"image": 10, "img2img": 10}`` (explicit cap matching the upstream
   recon3d ``max_images=10``, kept bounded for mm memory profiling), while
   the shared BAGEL base keeps its conservative ``{...: 1}`` limits.
2. N ``<|image_pad|>`` / ``<|fim_middle|>`` placeholders bind N mm items via
   ``_get_prompt_updates`` expansion + placeholder-range extraction.
3. ``_adjust_positions_for_img2img`` + MoT mask routing is correct for a
   single request containing **two** img2img blocks (this path previously
   consumed only the first block per request).
4. ``embed_multimodal`` returns N embeddings for batched N-item inputs on
   both the ``image`` and ``img2img`` keys.

All tests are CPU-only.  The tokenizer comes from the locally cached
SenseNova-Vision-7B-MoT checkpoint with ``local_files_only=True``; no model
weights are loaded.

Worst-case token budget (stage 0, ``deploy/sensenova_vision.yaml`` has
``max_num_batched_tokens: 32768``)::

    per img2img block (recon3d-size 512x512 input):
        VAE section  = (512/16)^2 + 2            =   1026 tokens
        separator    =                              1 token
        ViT section  = (980/14)^2 + 2            =   4902 tokens  (fixed)
        block total  =                             5929 tokens
    10-image request (limit cap): 10 x 5929       =  59290 prompt tokens

A single block (5929) fits comfortably inside one 32768-token prefill step;
a full 10-image request exceeds one step and therefore progresses via vLLM's
chunked prefill.  The limit stays a finite 10 (never ``None``) so mm memory
profiling remains bounded.
"""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing.processor import PromptReplacement

from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
    VLLMSenseNovaVisionTokenizer,
)
from vllm_omni.model_executor.models.bagel import bagel as bagel_module

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]

# --- Checkpoint-derived constants (SenseNova-Vision-7B-MoT) -----------------
VIT_MAX_NUM_PATCH_PER_SIDE = 70  # -> 70^2 = 4900 image_pad placeholders/item
VIT_PATCH_TOTAL = VIT_MAX_NUM_PATCH_PER_SIDE**2 + 2  # + start/end markers
LATENT_DOWNSAMPLE = 16  # vae downsample 8 * latent_patch_size 2
MAX_LATENT_SIZE = 64


def _cached_checkpoint() -> str | None:
    """The checkpoint root if cached locally (same lookup as the tokenizer tests)."""
    env_path = os.environ.get("SENSENOVA_VISION_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--sensenova--SenseNova-Vision-7B-MoT/snapshots/*")
    matches = sorted(glob.glob(snapshot))
    return matches[-1] if matches else None


@pytest.fixture(scope="module")
def checkpoint() -> str:
    snap = _cached_checkpoint()
    if snap is None:
        pytest.skip("SenseNova-Vision-7B-MoT not cached and SENSENOVA_VISION_MODEL_PATH is unset")
    return snap


@pytest.fixture(scope="module")
def tokenizer(checkpoint: str) -> VLLMSenseNovaVisionTokenizer:
    """The tokenizer exactly as ``SenseNovaVisionPipeline.__init__`` builds it."""
    return VLLMSenseNovaVisionTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def hf_config():
    """Faithful stand-in for the merged SenseNovaVision BagelConfig."""
    return SimpleNamespace(
        vit_max_num_patch_per_side=VIT_MAX_NUM_PATCH_PER_SIDE,
        latent_patch_size=2,
        max_latent_size=MAX_LATENT_SIZE,
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
def info_ctx(tokenizer, hf_config, checkpoint):
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
    proc = object.__new__(bagel_module.OmniBagelMultiModalProcessor)
    proc.info = info
    proc.dummy_inputs = None
    proc.cache = None
    proc.data_parser = info.get_data_parser()
    return proc


def _expected_img2img_block_len(h: int, w: int) -> tuple[int, int, int]:
    """(vae_total, vit_total, block_total) for an HxW img2img item.

    Mirrors the resize arithmetic in ``_get_prompt_updates`` /
    ``_resize_to_stride``.
    """
    stride = LATENT_DOWNSAMPLE
    max_img_size = MAX_LATENT_SIZE * stride
    scale = min(max_img_size / max(h, w), 1.0)
    min_img_size = min(256, max_img_size)
    scale = max(scale, min_img_size / min(h, w))
    new_h = min(max(stride, int(round(h * scale / stride)) * stride), max_img_size)
    new_w = min(max(stride, int(round(w * scale / stride)) * stride), max_img_size)
    num_vae_patches = (new_h // stride) * (new_w // stride)
    num_vae_total = num_vae_patches + 2
    return num_vae_total, VIT_PATCH_TOTAL, num_vae_total + 1 + VIT_PATCH_TOTAL


# ---------------------------------------------------------------------------
# 1. mm limits override
# ---------------------------------------------------------------------------


def test_sensenova_mm_limits_raise_to_ten(info_ctx):
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionProcessingInfo,
    )

    info = OmniSenseNovaVisionProcessingInfo(info_ctx)
    assert info.get_supported_mm_limits() == {"image": 10, "img2img": 10}


def test_shared_bagel_base_limits_unchanged(info_ctx):
    """The override must live in the SenseNova subclass only."""
    assert bagel_module.OmniBagelProcessingInfo(info_ctx).get_supported_mm_limits() == {
        "image": 1,
        "img2img": 1,
    }


def test_model_class_registered_with_sensenova_info():
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
        OmniSenseNovaVisionProcessingInfo,
    )

    factories = OmniSenseNovaVisionForConditionalGeneration._processor_factory
    assert factories.info is OmniSenseNovaVisionProcessingInfo


# ---------------------------------------------------------------------------
# 2a. N <|image_pad|> placeholders bind N image items
# ---------------------------------------------------------------------------


def test_n_image_placeholders_bind_n_items(tokenizer, hf_config, info_ctx):
    n = 3
    pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    images = [Image.new("RGB", (64, 64)) for _ in range(n)]
    mm_items = MultiModalDataItems({"image": ImageProcessorItems(images)})

    info = bagel_module.OmniBagelProcessingInfo(info_ctx)
    proc = _make_processor(info)
    updates = proc._get_prompt_updates(mm_items, {}, MultiModalKwargsItems())
    image_updates = [u for u in updates if isinstance(u, PromptReplacement) and u.modality == "image"]
    assert len(image_updates) == 1

    mm_prompt_updates = proc._bind_and_group_updates(updates, mm_items.get_all_counts())
    # Both modalities are always registered (their placeholder tokens exist in
    # the vocab); only "image" has items here.
    assert "image" in mm_prompt_updates
    assert len(mm_prompt_updates["image"]) == n, "each image item must get its own resolved update"

    prompt_ids = [pad_id] * n
    new_ids, placeholders = proc._apply_prompt_updates(prompt_ids, mm_prompt_updates)

    img_ph = placeholders["image"]
    assert len(img_ph) == n, "N <|image_pad|> placeholders must bind N image items"
    assert [ph.item_idx for ph in img_ph] == list(range(n))
    expected_len = VIT_MAX_NUM_PATCH_PER_SIDE**2
    starts = []
    for ph in img_ph:
        assert ph.tokens == [pad_id] * expected_len
        starts.append(ph.start_idx)
    assert starts == sorted(starts) and len(set(starts)) == n, "placeholder ranges must not overlap"
    assert sum(len(ph.tokens) for ph in img_ph) == len(new_ids)


# ---------------------------------------------------------------------------
# 2b. N <|fim_middle|> placeholders bind N img2img items -> N blocks
# ---------------------------------------------------------------------------


def test_n_fim_middle_placeholders_produce_n_blocks(tokenizer, hf_config, info_ctx):
    n = 2
    fim_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
    sizes = [(512, 512), (256, 384)]  # (H, W)
    images = [Image.new("RGB", (w, h)) for h, w in sizes]
    mm_items = MultiModalDataItems({"img2img": bagel_module.Img2ImgProcessorItems(images)})

    info = bagel_module.OmniBagelProcessingInfo(info_ctx)
    proc = _make_processor(info)
    updates = proc._get_prompt_updates(mm_items, {}, MultiModalKwargsItems())
    mm_prompt_updates = proc._bind_and_group_updates(updates, mm_items.get_all_counts())
    # Both modalities are always registered (see 2a); only "img2img" has items.
    assert len(mm_prompt_updates["img2img"]) == n

    prompt_ids = [fim_id] * n
    _new_ids, placeholders = proc._apply_prompt_updates(prompt_ids, mm_prompt_updates)

    blocks = placeholders["img2img"]
    assert len(blocks) == n, "N <|fim_middle|> placeholders must produce N img2img blocks"
    expected_lens = [_expected_img2img_block_len(h, w)[2] for h, w in sizes]
    for i, (ph, exp_len) in enumerate(zip(blocks, expected_lens)):
        assert ph.item_idx == i
        assert ph.length == exp_len, f"block {i}: got {ph.length}, expected {exp_len}"
        # is_embed mask: VAE markers+patches True, separator False, ViT True
        vae_total, vit_total, total = _expected_img2img_block_len(*sizes[i])
        mask = ph.is_embed
        assert mask.shape[0] == total
        assert mask[:vae_total].all(), "VAE section (markers + patches) must be embedded"
        assert not mask[vae_total], "separator must not be embedded"
        assert mask[vae_total + 1 :].all(), "ViT section must be embedded"


# ---------------------------------------------------------------------------
# 2c. _adjust_positions_for_img2img + MoT routing with TWO img2img blocks
# ---------------------------------------------------------------------------


class _PositionAdjustStub:
    """Carries exactly the state ``_adjust_positions_for_img2img`` touches."""

    def __init__(self, pending_infos, img2img_token_id):
        self._pending_img2img_info = list(pending_infos)
        self._last_img2img_info = None
        self._ropes_pending = []
        self._img2img_token_id = img2img_token_id
        self._vae_token_mask = None
        self._has_vae_tokens = False
        self._has_non_vae_tokens = True


def _two_block_ids(fim_id: int) -> tuple[list[int], int, int]:
    """One request: pre-text(2) + block1 + gap-text(2) + block2 + post-text(2).

    Each block carries (num_vae=6, num_vit=8): a 6-token VAE section
    (start marker + 4 latent patches + end marker), 1 separator, and an
    8-token ViT section (markers + 6 patches) -- matching the embed-side
    layout ``[se, vae..., ee, se, vit..., ee]`` with every block token
    rendered as <|fim_middle|>.
    """
    num_vae, num_vit = 6, 8
    block = [fim_id] * (num_vae + 1 + num_vit)
    ids = [11, 22] + block + [33, 44] + block + [55, 66]
    return ids, num_vae, num_vit


def test_adjust_positions_handles_two_img2img_blocks_in_one_request(tokenizer):
    fim_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
    ids, num_vae, num_vit = _two_block_ids(fim_id)
    infos = [(num_vae, num_vit, 512, 512), (num_vae, num_vit, 512, 512)]

    stub = _PositionAdjustStub(infos, fim_id)
    adjust = bagel_module.OmniBagelForConditionalGeneration._adjust_positions_for_img2img
    out = adjust(stub, torch.arange(len(ids)), torch.tensor(ids))
    got = out.tolist()

    # Layout: pre(2) | blk1 @ [2,17) | gap(2) @ [17,19) | blk2 @ [19,34) | post(2)
    # Each block: VAE+separator share the current text slot M, ViT shares
    # M+1, and the following text resumes sequentially at M+2.
    m1, m2 = 2, 6  # text counters when each block starts
    expected = [0, 1]
    expected += [m1] * (num_vae + 1) + [m1 + 1] * num_vit  # block 1
    expected += [m1 + 2, m1 + 3]  # inter-block text continues sequentially
    expected += [m2] * (num_vae + 1) + [m2 + 1] * num_vit  # block 2
    expected += [m2 + 2, m2 + 3]  # trailing text
    assert got == expected, (
        "both img2img blocks must get shared VAE/ViT positions; "
        f"second block was left with raw sequential positions: {got}"
    )

    # MoT routing: latent patches of BOTH blocks route through moe_gen.
    mask = stub._vae_token_mask
    assert mask is not None and stub._has_vae_tokens
    b1_latent = list(range(2 + 1, 2 + num_vae - 1))  # strip start/end markers
    b2_latent = list(range(19 + 1, 19 + num_vae - 1))
    assert all(mask[i] for i in b1_latent + b2_latent), mask.int().tolist()
    assert not mask[0] and not mask[-1], "text tokens must not be VAE-masked"
    assert not mask[2] and not mask[8], "block 1 marker/separator must not be VAE-masked"
    assert not mask[19] and not mask[25], "block 2 marker/separator must not be VAE-masked"
    assert stub._has_non_vae_tokens

    # Exactly one ropes entry per request (flush_pending_metadata maps batch
    # order -> req_ids), carrying the FINAL continuation rope and the last
    # block's image shape.
    assert len(stub._ropes_pending) == 1
    meta = stub._ropes_pending[0]
    assert meta["ropes"] == [m2 + 4]
    assert meta["image_shape"] == [512, 512]
    assert meta["prefill_position_count"] == len(ids)
    assert stub._pending_img2img_info == []


def test_adjust_positions_single_block_unchanged(tokenizer):
    """Guard: the multi-block fix must not alter single-block results."""
    fim_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
    num_vae, num_vit = 6, 8
    block = [fim_id] * (num_vae + 1 + num_vit)
    ids = [7, 8, 9] + block + [10, 11]

    stub = _PositionAdjustStub([(num_vae, num_vit, 512, 512)], fim_id)
    adjust = bagel_module.OmniBagelForConditionalGeneration._adjust_positions_for_img2img
    out = adjust(stub, torch.arange(len(ids)), torch.tensor(ids))

    m = 3
    expected = [0, 1, 2] + [m] * (num_vae + 1) + [m + 1] * num_vit + [m + 2, m + 3]
    assert out.tolist() == expected
    assert stub._ropes_pending == [{"ropes": [m + 4], "image_shape": [512, 512], "prefill_position_count": len(ids)}]
    mask = stub._vae_token_mask
    assert all(mask[i] for i in range(m + 1, m + num_vae - 1))


# ---------------------------------------------------------------------------
# 2d. embed_multimodal returns N embeddings for batched N-item inputs
# ---------------------------------------------------------------------------


def test_parse_and_validate_splits_image_and_img2img():
    inst = object.__new__(bagel_module.OmniBagelForConditionalGeneration)
    pv_image = torch.zeros(2, 3, 8, 8)
    pv_img2img = torch.zeros(3, 3, 8, 8)

    mm = inst._parse_and_validate_multimodal_inputs(pixel_values=pv_image, pixel_values_img2img=pv_img2img)

    assert set(mm) == {"img2text", "img2img"}
    assert mm["img2text"]["pixel_values"] is pv_image
    assert mm["img2img"]["pixel_values"] is pv_img2img


def test_embed_multimodal_returns_n_embeddings_per_modality():
    n_images, n_img2img = 3, 2
    inst = object.__new__(bagel_module.OmniBagelForConditionalGeneration)
    calls = []

    def fake_img2text(mm_input):
        calls.append(("img2text", mm_input["pixel_values"].shape[0]))
        return tuple(torch.full((1, 4), float(i)) for i in range(n_images))

    def fake_img2img(mm_input):
        calls.append(("img2img", mm_input["pixel_values"].shape[0]))
        return tuple(torch.full((1, 4), 100.0 + i) for i in range(n_img2img))

    inst._parse_and_validate_multimodal_inputs = lambda **kw: {
        "img2text": {"pixel_values": torch.zeros(n_images, 3, 8, 8)},
        "img2img": {"pixel_values": torch.zeros(n_img2img, 3, 8, 8)},
    }
    inst._process_img2text_input = fake_img2text
    inst._process_img2img_input = fake_img2img

    out = inst.embed_multimodal()

    assert ("img2text", n_images) in calls and ("img2img", n_img2img) in calls
    assert len(out) == n_images + n_img2img, "one embedding per mm item"
    assert [t[0, 0].item() for t in out] == [0, 1, 2, 100, 101]


def test_img2img_batch_flattens_leading_batch_dim():
    """A (B, N, C, H, W) img2img tensor must yield one info tuple per image."""
    inst = object.__new__(bagel_module.OmniBagelForConditionalGeneration)
    infos = []
    inst.latent_downsample = LATENT_DOWNSAMPLE
    inst.max_latent_size = MAX_LATENT_SIZE
    inst.latent_channel = 16
    inst.latent_patch_size = 2
    inst.config = SimpleNamespace(vit_config=SimpleNamespace(image_size=64))
    inst.device = torch.device("cpu")

    captured = {}

    def fake_process_image_input(mm_input):
        captured["pv"] = mm_input["pixel_values"]
        return tuple(torch.zeros(1, 4) for _ in range(captured["pv"].shape[0]))

    class _FakeVAE:
        def encode(self, x):
            # Bare latent tensor, 16 channels, /8 spatial (DiagonalGaussian output).
            return torch.zeros(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    inst._process_image_input = fake_process_image_input
    inst.vae = _FakeVAE()
    inst._resize_to_stride = lambda pv: pv
    inst.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), 4)))
    inst.vae2llm = lambda z: z[:, :4]
    inst.latent_pos_embed = lambda pos: torch.zeros(1, 4)
    inst.time_embedder = lambda t: torch.zeros(1, 4)
    inst._start_of_image_id = 151652
    inst._end_of_image_id = 151653
    inst._ropes_pending = []
    inst._pending_img2img_info = infos
    inst._last_img2img_info = None

    batched = torch.zeros(1, 2, 3, 32, 32)  # (batch=1, num_images=2, ...)
    inst._process_img2img_input({"pixel_values": batched})

    assert captured["pv"].shape[0] == 2, "leading batch dim must be flattened"
    assert len(infos) == 2, "one (num_vae, num_vit, H, W) info tuple per image"


# ---------------------------------------------------------------------------
# 3. Worst-case token budget arithmetic
# ---------------------------------------------------------------------------


def test_worst_case_token_budget_arithmetic():
    """Documented budget for the limit=10 cap (see module docstring).

    A single recon3d-size block (5929 tokens) must fit inside one stage-0
    prefill step (``max_num_batched_tokens: 32768``); a full 10-image request
    (59290 tokens) exceeds one step and progresses via chunked prefill.
    """
    vae_total, vit_total, block_total = _expected_img2img_block_len(512, 512)
    assert (vae_total, vit_total) == (1026, 4902)
    assert block_total == 5929

    stage0_step_budget = 32768  # deploy/sensenova_vision.yaml stages[0]
    assert block_total < stage0_step_budget, "one img2img block must fit in a single prefill step"

    limit = 10
    worst_case_prompt_tokens = limit * block_total
    assert worst_case_prompt_tokens == 59290
    # The cap must stay finite so mm profiling allocates a bounded dummy
    # batch (never unbounded/None).
    assert isinstance(limit, int) and limit > 0
