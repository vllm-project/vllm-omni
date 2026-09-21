# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

    per img2img block (recon3d-size 512x512 input, SenseNova VAE->ViT):
        VAE section  = (512/16)^2 + 2            =   1026 tokens
        separator    =                              1 token
        ViT section  = aspect grid + 2             ~=  1371 tokens
        block total  =                             2398 tokens
    10-image request (limit cap): 10 x 2398       =  23980 prompt tokens

(BAGEL separator layout is kept so extract_embeds_range() yields two mm
ranges for M-RoPE; sizes come from ``_sensenova_*_resize_dims``.)

A single block and a full 10-image request both fit inside one 32768-token
prefill step.  The limit stays a finite 10 (never ``None``) so mm memory
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
    assert snap is not None
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

    Mirrors the BAGEL-BASE resize arithmetic; only used by the 2b test,
    which exercises the base ``OmniBagelMultiModalProcessor`` expansion.
    """
    from vllm_omni.diffusion.models.bagel.pipeline_bagel import bagel_image_size

    stride = LATENT_DOWNSAMPLE
    max_img_size = MAX_LATENT_SIZE * stride
    scale = min(max_img_size / max(h, w), 1.0)
    min_img_size = min(256, max_img_size)
    scale = max(scale, min_img_size / min(h, w))
    new_h = min(max(stride, int(round(h * scale / stride)) * stride), max_img_size)
    new_w = min(max(stride, int(round(w * scale / stride)) * stride), max_img_size)
    num_vae_patches = (new_h // stride) * (new_w // stride)
    num_vae_total = num_vae_patches + 2
    vit_w, vit_h = bagel_image_size(w, h, 980, 224, 14)
    num_vit_total = (vit_h // 14) * (vit_w // 14) + 2
    return num_vae_total, num_vit_total, num_vae_total + 1 + num_vit_total


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
    # BAGEL base leaves understanding unbounded (None) and caps img2img at 1.
    assert bagel_module.OmniBagelProcessingInfo(info_ctx).get_supported_mm_limits() == {
        "image": None,
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
    from vllm_omni.diffusion.models.bagel.pipeline_bagel import bagel_image_size

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
    vit_w, vit_h = bagel_image_size(64, 64, 980, 224, 14)
    expected_len = (vit_h // 14) * (vit_w // 14) + 2
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
    # (H, W)
    sizes: list[tuple[int, int]] = [(512, 512), (256, 384)]  # (H, W)
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


def test_sensenova_img2img_expansion_is_upstream_exact(tokenizer, hf_config, info_ctx):
    """SenseNova placeholder counts must lockstep with ``_sensenova_*_resize_dims``.

    Layout keeps BAGEL's separator so extract_embeds_range() yields two mm
    ranges for M-RoPE; sizes use the official VAE then ViT chain.
    """
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionMultiModalProcessor,
        OmniSenseNovaVisionProcessingInfo,
        _sensenova_img2img_token_counts,
        _sensenova_vae_resize_dims,
        _sensenova_vit_resize_dims,
    )

    h, w = 375, 500  # non-square -> resize arithmetic actually exercised
    image = Image.new("RGB", (w, h))
    mm_items = MultiModalDataItems({"img2img": bagel_module.Img2ImgProcessorItems([image])})

    info = OmniSenseNovaVisionProcessingInfo(info_ctx)
    proc = object.__new__(OmniSenseNovaVisionMultiModalProcessor)
    proc.info = info
    proc.dummy_inputs = None
    proc.cache = None
    proc.data_parser = info.get_data_parser()

    updates = proc._get_prompt_updates(mm_items, {}, MultiModalKwargsItems())
    mm_prompt_updates = proc._bind_and_group_updates(updates, mm_items.get_all_counts())
    prompt_ids = [tokenizer.convert_tokens_to_ids("<|fim_middle|>")]
    _new_ids, placeholders = proc._apply_prompt_updates(prompt_ids, mm_prompt_updates)

    (ph,) = placeholders["img2img"]
    fim_id = tokenizer.get_vocab()["<|fim_middle|>"]

    # Official two-stage transform: VAE resize, then ViT resize OF THE VAE-
    # RESIZED image.  The ViT count follows aspect ratio (upstream
    # ImageTransform(980, 224, 14)), NOT the fixed 70x70 square.
    new_h, new_w = _sensenova_vae_resize_dims(h, w)
    vit_h, vit_w = _sensenova_vit_resize_dims(new_h, new_w)
    num_vae_total, num_vit_total, vae_h, vae_w = _sensenova_img2img_token_counts(h, w)
    assert (vae_h, vae_w) == (new_h, new_w)
    num_vit_patches = (vit_h // 14) * (vit_w // 14)
    assert num_vit_patches <= VIT_MAX_NUM_PATCH_PER_SIDE**2, "aspect grid must stay within the 70x70 cap"
    assert num_vit_total == num_vit_patches + 2

    # BAGEL separator between VAE and ViT sections for M-RoPE ranges.
    total = num_vae_total + 1 + num_vit_total
    assert ph.length == total
    assert all(t == fim_id for t in ph.tokens)

    mask = ph.is_embed
    assert mask is not None and mask.shape[0] == total
    assert mask[:num_vae_total].all(), "VAE section must be embedded"
    assert not mask[num_vae_total], "separator must not be embedded"
    assert mask[num_vae_total + 1 :].all(), "ViT section must be embedded"


# ---------------------------------------------------------------------------
# 2c. _adjust_positions_for_img2img + MoT routing with TWO img2img blocks
# ---------------------------------------------------------------------------


class _PositionAdjustStub:
    """Carries exactly the state ``_adjust_positions_for_img2img`` touches."""

    def __init__(self, pending_infos, start_of_image_id, end_of_image_id, img2img_token_id):
        self._pending_img2img_info = list(pending_infos)
        self._last_img2img_info = None
        self._ropes_pending = []
        self._start_of_image_id = start_of_image_id
        self._end_of_image_id = end_of_image_id
        self._img2img_token_id = img2img_token_id
        self._vae_token_mask = None
        self._has_vae_tokens = False
        self._has_non_vae_tokens = True
        self._img2img_layouts = {}
        self._step_req_schedule = []
        # Bind the per-request helper methods so the state machine calls on
        # ``self`` resolve against the stub (the production model carries
        # them as bound methods on the class).
        from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
            OmniSenseNovaVisionForConditionalGeneration,
        )

        for _name in (
            "_match_leading_blocks",
            "_enter_layout_from_span",
            "_resolve_block_geometry",
            "_collapse_chunk_into_layout",
            "_emit_layout_rope",
        ):
            fn = getattr(OmniSenseNovaVisionForConditionalGeneration, _name)
            setattr(self, _name, fn.__get__(self))


def _two_block_ids(soi_id: int, fim_id: int, eoi_id: int) -> tuple[list[int], int, int]:
    """One request: pre-text(2) + block1 + gap-text(2) + block2 + post-text(2).

    Each block carries (num_vae=6, num_vit=8) in the upstream-exact layout:
    ``[SOI] 4 latent patches [EOI] [SOI] 6 patches [EOI]`` -- ADJACENT
    sections, NO separator -- with <|fim_middle|> placeholders standing in
    for every patch slot, matching the embed-side layout
    ``[se, vae..., ee, se, vit..., ee]``.
    """
    num_vae, num_vit = 6, 8
    block = [soi_id] + [fim_id] * (num_vae - 2) + [eoi_id] + [soi_id] + [fim_id] * (num_vit - 2) + [eoi_id]
    ids = [11, 22] + block + [33, 44] + block + [55, 66]
    return ids, num_vae, num_vit


@pytest.mark.skip(reason="outdated: SenseNova now reuses BAGEL M-RoPE layout; custom position rewrite removed")
def test_adjust_positions_handles_two_img2img_blocks_in_one_request(tokenizer):
    vocab = tokenizer.get_vocab()
    soi_id = vocab["<|vision_start|>"]
    eoi_id = vocab["<|vision_end|>"]
    fim_id = vocab["<|fim_middle|>"]
    ids, num_vae, num_vit = _two_block_ids(soi_id, fim_id, eoi_id)
    infos = [(num_vae, num_vit, 512, 512), (num_vae, num_vit, 512, 512)]

    stub = _PositionAdjustStub(infos, soi_id, eoi_id, fim_id)
    stub._step_req_schedule = [("r1", 0, len(ids))]
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
    )

    adjust = OmniSenseNovaVisionForConditionalGeneration._adjust_positions_for_img2img
    out = adjust(stub, torch.arange(len(ids)), torch.tensor(ids))
    got = out.tolist()

    # Layout: pre(2) | blk1 @ [2,16) | gap(2) @ [16,18) | blk2 @ [18,32) | post(2)
    # Each 14-token block: VAE section (SOI marker incl.) shares the current
    # text slot M, ViT section shares M+1, and the following text resumes
    # sequentially at M+2.
    m1, m2 = 2, 6  # text counters when each block starts
    expected = [0, 1]
    expected += [m1] * num_vae + [m1 + 1] * num_vit  # block 1
    expected += [m1 + 2, m1 + 3]  # inter-block text continues sequentially
    expected += [m2] * num_vae + [m2 + 1] * num_vit  # block 2
    expected += [m2 + 2, m2 + 3]  # trailing text
    assert got == expected, (
        "both img2img blocks must get shared VAE/ViT positions; "
        f"second block was left with raw sequential positions: {got}"
    )

    # MoT routing: latent patches of BOTH blocks route through moe_gen.
    mask = stub._vae_token_mask
    assert mask is not None and stub._has_vae_tokens
    b1_latent = list(range(2 + 1, 2 + num_vae - 1))  # between block 1's markers
    b2_latent = list(range(18 + 1, 18 + num_vae - 1))  # between block 2's markers
    assert all(mask[i] for i in b1_latent + b2_latent), mask.int().tolist()
    assert not mask[0] and not mask[-1], "text tokens must not be VAE-masked"
    assert not any(mask[i] for i in (2, 7, 8, 13)), "block 1 SOI/EOI markers must not be VAE-masked"
    assert not any(mask[i] for i in (18, 23, 24, 29)), "block 2 SOI/EOI markers must not be VAE-masked"
    assert stub._has_non_vae_tokens

    # The PREFILL chunk (whole prompt in one step) emits only the continuation
    # rope (flush_pending_metadata last-wins keeps the later decode entry).
    assert stub._ropes_pending == [{"ropes": [m2 + 4]}]
    assert stub._pending_img2img_info == []
    # The layout survives prefill so the FIRST DECODE step can emit the full
    # metadata (prefill_position_count == num_computed == prompt_len there).
    assert "r1" in stub._img2img_layouts

    stub._step_req_schedule = [("r1", len(ids), 1)]
    adjust(stub, torch.tensor([len(ids)]), torch.tensor([77]))
    assert len(stub._ropes_pending) == 2
    meta = stub._ropes_pending[-1]
    assert meta["ropes"] == [m2 + 4]
    assert meta["image_shape"] == [512, 512]
    assert meta["prefill_position_count"] == len(ids)
    assert stub._img2img_layouts == {}, "layout must be pruned at first decode"


@pytest.mark.skip(reason="outdated: SenseNova now reuses BAGEL M-RoPE layout; custom position rewrite removed")
def test_adjust_positions_single_block_unchanged(tokenizer):
    """Guard: the upstream-exact relayout must not alter single-block results."""
    vocab = tokenizer.get_vocab()
    soi_id = vocab["<|vision_start|>"]
    eoi_id = vocab["<|vision_end|>"]
    fim_id = vocab["<|fim_middle|>"]
    num_vae, num_vit = 6, 8
    block = [soi_id] + [fim_id] * (num_vae - 2) + [eoi_id] + [soi_id] + [fim_id] * (num_vit - 2) + [eoi_id]
    ids = [7, 8, 9] + block + [10, 11]

    stub = _PositionAdjustStub([(num_vae, num_vit, 512, 512)], soi_id, eoi_id, fim_id)
    stub._step_req_schedule = [("r1", 0, len(ids))]
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
    )

    adjust = OmniSenseNovaVisionForConditionalGeneration._adjust_positions_for_img2img
    out = adjust(stub, torch.arange(len(ids)), torch.tensor(ids))

    m = 3
    expected = [0, 1, 2] + [m] * num_vae + [m + 1] * num_vit + [m + 2, m + 3]
    assert out.tolist() == expected
    # Prefill emits the plain continuation rope; the decode step carries the
    # full metadata (prefill_position_count == num_computed == prompt_len).
    assert stub._ropes_pending == [{"ropes": [m + 4]}]
    mask = stub._vae_token_mask
    assert mask is not None
    assert all(mask[i] for i in range(m + 1, m + num_vae - 1))

    stub._step_req_schedule = [("r1", len(ids), 1)]
    adjust(stub, torch.tensor([len(ids)]), torch.tensor([77]))
    assert stub._ropes_pending[-1] == {
        "ropes": [m + 4],
        "image_shape": [512, 512],
        "prefill_position_count": len(ids),
    }
    assert stub._img2img_layouts == {}


def _block_ids(soi_id: int, fim_id: int, eoi_id: int, num_vae: int = 6, num_vit: int = 8) -> list[int]:
    """One full img2img block (default 6+8 geometry), upstream-exact layout."""
    return [soi_id] + [fim_id] * (num_vae - 2) + [eoi_id] + [soi_id] + [fim_id] * (num_vit - 2) + [eoi_id]


def _split_chunks(ids: list[int], cut: int) -> tuple[list[int], list[int]]:
    """Split token ids into (pre, post) at a token boundary (offset cut)."""
    return ids[:cut], ids[cut:]


@pytest.mark.skip(reason="outdated: SenseNova now reuses BAGEL M-RoPE layout; custom position rewrite removed")
def test_adjust_positions_split_block_spans_two_chunks(tokenizer):
    """Bug C: a block split by chunked prefill must be collapsed correctly.

    Chunk 1 ends inside the VAE section; chunk 2 begins inside it.  The
    encoder re-encodes the split block, so its info is refilled at the
    continuation chunk's FIFO head.  The layout persists in one stub across
    both calls (as it does in the model across forward() steps)."""
    vocab = tokenizer.get_vocab()
    soi_id = vocab["<|vision_start|>"]
    eoi_id = vocab["<|vision_end|>"]
    fim_id = vocab["<|fim_middle|>"]
    num_vae, num_vit = 6, 8
    block = _block_ids(soi_id, fim_id, eoi_id, num_vae, num_vit)
    ids = [7, 8] + block + [9, 10]
    # Cut inside the VAE section: pre-text(2) + SOI + 2 fim patches.
    cut = 2 + 1 + 2  # 5 tokens, ends inside the VAE interior
    pre, post = _split_chunks(ids, cut)
    assert len(pre) == 5 and pre[:2] == [7, 8] and pre[2] == soi_id
    assert pre[3] == fim_id and pre[4] == fim_id and post[0] == fim_id

    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
    )

    adjust = OmniSenseNovaVisionForConditionalGeneration._adjust_positions_for_img2img

    stub = _PositionAdjustStub([(num_vae, num_vit, 512, 512)], soi_id, eoi_id, fim_id)

    # Chunk 1: window ends mid-VAE (partial block).
    stub._step_req_schedule = [("r1", 0, len(pre))]
    out1 = adjust(stub, torch.arange(len(ids)), torch.tensor(pre)).tolist()[: len(pre)]
    assert out1[:2] == [0, 1], out1
    assert all(v == 2 for v in out1[2:]), out1  # partial VAE (SOI+2 patches) share M=2

    # Chunk 2: continuation.  The encoder re-encodes the split block: stale
    # info at the FIFO head; the layout keeps the in-progress block geometry.
    stub._pending_img2img_info = [(num_vae, num_vit, 512, 512)]
    stub._step_req_schedule = [("r1", len(pre), len(post))]
    out2 = adjust(stub, torch.arange(len(pre), len(ids)), torch.tensor(post)).tolist()

    m = 2
    # Chunk 2 consumes: 3 remaining VAE tokens (fim, fim, eoi) -> M, then the
    # 8-token ViT section -> M+1, then trailing text -> M+2, M+3.
    vae_consumed_1 = 3  # SOI + 2 fim patches in chunk 1
    expected2 = [m] * (num_vae - vae_consumed_1)
    expected2 += [m + 1] * num_vit
    expected2 += [m + 2, m + 3]
    assert len(expected2) == len(post), (len(expected2), len(post))
    assert out2 == expected2, f"chunk2: {out2}"

    combined = out1 + out2
    expected_full = [0, 1] + [m] * num_vae + [m + 1] * num_vit + [m + 2, m + 3]
    assert combined == expected_full, f"combined: {combined}"

    # The stale re-encode is consumed by the harness via ``_pending
    # _img2img_info`` hand-in; correctness follows from the layout phase
    # machine, not from FIFO accounting, so nothing else needs to remain.


@pytest.mark.skip(reason="outdated: SenseNova now reuses BAGEL M-RoPE layout; custom position rewrite removed")
def test_adjust_positions_chunked_multi_block_continuation(tokenizer):
    """A two-block request split across three chunks must match single-chunk.

    Chunk1: pre-text + block1 (complete) + partial VAE of block2.
    Chunk2: rest of block2's VAE (stale re-encode at FIFO head).
    Chunk3: rest of block2's ViT + trailing text (prefill done)."""
    vocab = tokenizer.get_vocab()
    soi_id = vocab["<|vision_start|>"]
    eoi_id = vocab["<|vision_end|>"]
    fim_id = vocab["<|fim_middle|>"]
    num_vae, num_vit = 6, 8
    block = _block_ids(soi_id, fim_id, eoi_id, num_vae, num_vit)
    ids = [11, 22] + block + block + [33, 44]

    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
    )

    adjust = OmniSenseNovaVisionForConditionalGeneration._adjust_positions_for_img2img

    # Single-chunk reference.
    ref = _PositionAdjustStub([(num_vae, num_vit, 512, 512)] * 2, soi_id, eoi_id, fim_id)
    ref._step_req_schedule = [("r1", 0, len(ids))]
    ref_out = adjust(ref, torch.arange(len(ids)), torch.tensor(ids)).tolist()

    # Three chunks: cut1 ends inside VAE of block2; cut2 ends block2's VAE.
    cut1 = 2 + (num_vae + num_vit) + (1 + 3)  # pre + blk1 + 4 tokens of blk2 VAE
    cut2 = 2 + (num_vae + num_vit) + num_vae  # pre + blk1 + full blk2 VAE
    c1, rest = _split_chunks(ids, cut1)
    c2, c3 = _split_chunks(rest, cut2 - cut1)

    stub = _PositionAdjustStub([], soi_id, eoi_id, fim_id)
    combined = []
    comp = 0
    for i, chunk in enumerate((c1, c2, c3)):
        if i == 0:
            stub._pending_img2img_info = [(num_vae, num_vit, 512, 512)] * 2
        else:
            # The encoder re-encodes block2 in each continuation chunk.
            stub._pending_img2img_info = [(num_vae, num_vit, 512, 512)]
        stub._step_req_schedule = [("r1", comp, len(chunk))]
        seg_out = adjust(stub, torch.arange(comp, comp + len(chunk)), torch.tensor(chunk)).tolist()
        combined += seg_out
        comp += len(chunk)

    assert combined == ref_out, f"chunked {combined} != single-chunk {ref_out}"

    # Every prefill chunk emits a plain rope; the layout persists until the
    # FIRST DECODE step, whose entry carries the full metadata.
    assert stub._ropes_pending, "must have rope entries"
    assert all("image_shape" not in m for m in stub._ropes_pending), stub._ropes_pending
    assert "r1" in stub._img2img_layouts

    stub._step_req_schedule = [("r1", len(ids), 1)]
    adjust(stub, torch.tensor([len(ids)]), torch.tensor([77]))
    final_meta = stub._ropes_pending[-1]
    assert final_meta["image_shape"] == [512, 512]
    assert final_meta["prefill_position_count"] == len(ids)
    assert stub._img2img_layouts == {}


@pytest.mark.skip(reason="outdated: SenseNova now reuses BAGEL M-RoPE layout; custom position rewrite removed")
def test_adjust_positions_mixed_img2img_text_batch(tokenizer):
    """Batch of (img2img, text-only, img2img) must map ropes 1:1 in order."""
    vocab = tokenizer.get_vocab()
    soi_id = vocab["<|vision_start|>"]
    eoi_id = vocab["<|vision_end|>"]
    fim_id = vocab["<|fim_middle|>"]
    num_vae, num_vit = 6, 8
    block = _block_ids(soi_id, fim_id, eoi_id, num_vae, num_vit)

    a_ids = [1, 2] + block + [3]  # img2img request
    b_ids = [10, 20, 30]  # text-only sibling
    c_ids = [5] + block + [6, 7]  # second img2img (different shape)

    ab = a_ids + b_ids + c_ids
    infos = [(num_vae, num_vit, 512, 512), (num_vae, num_vit, 256, 384)]
    # Each request's positions are its OWN sequence (0-based), concatenated.
    positions = list(range(len(a_ids))) + list(range(len(b_ids))) + list(range(len(c_ids)))

    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
    )

    adjust = OmniSenseNovaVisionForConditionalGeneration._adjust_positions_for_img2img

    stub = _PositionAdjustStub(infos, soi_id, eoi_id, fim_id)
    stub._step_req_schedule = [
        ("a", 0, len(a_ids)),
        ("b", 0, len(b_ids)),
        ("c", 0, len(c_ids)),
    ]
    out = adjust(stub, torch.tensor(positions), torch.tensor(ab)).tolist()

    a_expected = [0, 1] + [2] * num_vae + [3] * num_vit + [4]
    b_expected = [0, 1, 2]
    c_expected = [0] + [1] * num_vae + [2] * num_vit + [3, 4]
    assert out == a_expected + b_expected + c_expected, out

    # Prefill: ONE plain rope per request in batch order.
    assert len(stub._ropes_pending) == 3, stub._ropes_pending
    assert stub._ropes_pending[0] == {"ropes": [5]}
    assert stub._ropes_pending[1] == {"ropes": [3]}, "text-only sibling plain rope"
    assert stub._ropes_pending[2] == {"ropes": [5]}
    assert set(stub._img2img_layouts) == {"a", "c"}

    # First decode step: img2img requests emit the full metadata (rope +
    # image_shape + prefill_position_count == num_computed == prompt_len);
    # the text-only sibling stays plain; layouts are pruned.
    stub._step_req_schedule = [
        ("a", len(a_ids), 1),
        ("b", len(b_ids), 1),
        ("c", len(c_ids), 1),
    ]
    decode_positions = [len(a_ids), len(b_ids), len(c_ids)]
    adjust(stub, torch.tensor(decode_positions), torch.tensor([77, 78, 79]))
    assert len(stub._ropes_pending) == 6, stub._ropes_pending
    assert stub._ropes_pending[3] == {
        "ropes": [5],
        "image_shape": [512, 512],
        "prefill_position_count": len(a_ids),
    }
    assert stub._ropes_pending[4] == {"ropes": [4]}, "text-only sibling decode plain rope"
    assert stub._ropes_pending[5]["image_shape"] == [256, 384]
    assert stub._ropes_pending[5]["prefill_position_count"] == len(c_ids)
    assert stub._img2img_layouts == {}


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
    infos: list[tuple[int, int, int, int]] = []
    inst.latent_downsample = LATENT_DOWNSAMPLE
    inst.max_latent_size = MAX_LATENT_SIZE
    inst.latent_channel = 16
    inst.latent_patch_size = 2
    inst.config = SimpleNamespace(vit_config=SimpleNamespace(image_size=64, patch_size=14))
    inst.device = torch.device("cpu")

    captured = {}

    def fake_vit_embeddings(images):
        captured["n"] = len(images)
        return [torch.zeros(1, 4) for _ in images]

    class _FakeVAE:
        def encode(self, x):
            # Bare latent tensor, 16 channels, /8 spatial (DiagonalGaussian output).
            return torch.zeros(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    inst._vit_embeddings = fake_vit_embeddings
    inst.vae = _FakeVAE()
    inst._resize_to_stride = lambda pv: pv
    inst.get_flattened_position_ids = lambda *a, **k: torch.zeros(1, dtype=torch.long)
    inst.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), 4)))
    inst.vae2llm = lambda z: torch.zeros(z.shape[0], 4)
    inst.latent_pos_embed = lambda pos: torch.zeros(1, 4)
    inst.time_embedder = lambda t: torch.zeros(1, 4)
    inst._start_of_image_id = 151652
    inst._end_of_image_id = 151653
    inst._ropes_pending = []
    inst._pending_img2img_info = infos
    inst._img2img_info_by_size = {}
    inst._img2img_by_req = {}
    inst._last_img2img_info = None

    batched = torch.zeros(1, 2, 3, 32, 32)  # (batch=1, num_images=2, ...)
    inst._process_img2img_input({"pixel_values": batched})

    assert captured["n"] == 2, "leading batch dim must be flattened"
    assert len(infos) == 2, "one (num_vae, num_vit, H, W) info tuple per image"


def test_sensenova_img2img_seeds_size_cache_for_cache_served_request():
    """A SenseNova img2img embed must seed the cross-request size cache.

    Regression for the aspect-ratio bug: ``seg`` (2.jpg) then ``normal``
    (2.jpg) in one process lost the second request's ``image_shape``, so the
    DiT fell back to a square 1024x1024 output.  ``_process_img2img_input``
    appends to ``_pending_img2img_info`` but the size lookup for a later
    request whose image the encoder/prefix cache serves (no embed run) is
    ``_img2img_info_by_size`` — that cache is only seeded by
    ``_register_img2img_info``.
    """
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
    )

    inst = object.__new__(OmniSenseNovaVisionForConditionalGeneration)
    inst.latent_downsample = LATENT_DOWNSAMPLE
    inst.max_latent_size = MAX_LATENT_SIZE
    inst.latent_channel = 16
    inst.latent_patch_size = 2
    inst.config = SimpleNamespace(vit_config=SimpleNamespace(image_size=64, patch_size=14))
    inst.device = torch.device("cpu")

    captured = {}

    def fake_vit_embeddings(images):
        captured["n"] = len(images)
        return [torch.zeros(1, 4) for _ in images]

    class _FakeVAE:
        def encode(self, x):
            return torch.zeros(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    inst._vit_embeddings = fake_vit_embeddings
    inst._resize_to_stride = lambda pv: pv
    inst._resize_for_vit = lambda pv: pv
    inst.vae = _FakeVAE()
    inst.get_flattened_position_ids = lambda *a, **k: torch.zeros(1, dtype=torch.long)
    inst.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), 4)))
    inst.vae2llm = lambda z: torch.zeros(z.shape[0], 4)
    inst.latent_pos_embed = lambda pos: torch.zeros(1, 4)
    inst.time_embedder = lambda t: torch.zeros(1, 4)
    inst._start_of_image_id = 151652
    inst._end_of_image_id = 151653
    inst._ropes_pending = []
    inst._pending_img2img_info = []
    inst._img2img_info_by_size = {}
    inst._img2img_by_req = {}
    inst._last_img2img_info = None

    img = torch.zeros(1, 1, 3, 32, 32)  # (batch, num_images, C, H, W)
    inst._process_img2img_input({"pixel_values": img})

    # Pending metadata consumed by this step's routing...
    assert len(inst._pending_img2img_info) == 1
    # ...and the size cache must ALSO be seeded so a cache-served follow-up
    # request (no embed run) can resolve its (H, W).
    key = tuple(inst._pending_img2img_info[0][:2])
    assert key in inst._img2img_info_by_size, "size cache must be seeded by the embed run"
    assert inst._img2img_info_by_size[key][2:] == (32, 32), inst._img2img_info_by_size[key]


# ---------------------------------------------------------------------------
# 2e. SigLIP pos rows for aspect-preserving ViT grids (navit-exact lookup)
# ---------------------------------------------------------------------------


class _CallableTable:
    """Stand-in for ``nn.Embedding``: indexable-callable around a raw table
    (the real module is called as ``position_embedding(position_ids)`` AND
    read as ``position_embedding.weight``)."""

    def __init__(self, table: torch.Tensor):
        self.weight = table

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]


def _make_siglip_embeddings_stub(grid: int, dim: int, patch_size: int):
    """A minimal stand-in for vLLM's SiglipVisionEmbeddings with the SAME
    attribute surface the real module exposes (call-and-weight position
    embedding table, position_ids buffer, patch_size)."""
    from types import SimpleNamespace

    table = torch.randn(grid * grid, dim)
    emb = SimpleNamespace(
        position_embedding=_CallableTable(table),
        position_ids=torch.arange(grid * grid).unsqueeze(0),
        patch_size=patch_size,
    )

    def _broken_interp(self, embeddings, height, width):
        # Faithful copy of the BUGGY line in vllm's siglip.py:322:
        # sqrt of weight.shape[1] (hidden size) instead of shape[0].
        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[1]  # BUG
        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)
        raise RuntimeError("shape mismatch should have happened before this")

    emb.interpolate_pos_encoding = _broken_interp.__get__(emb)
    return emb


@pytest.mark.skip(reason="outdated: custom SigLIP pos-encoding patch removed; uses bagel _vit_embeddings")
def test_fix_siglip_pos_encoding_aspect_grid_exact_rows():
    """_fix_siglip_pos_encoding must bind the navit-EXACT lookup: for any
    aspect grid <= 70x70, position rows come straight from the trained table
    at ids h*70 + w -- NEVER bicubic resampling (siglip_navit never
    interpolates; blended rows are OOD poison, see out_13).  The pre-fix
    vLLM implementation additionally crashed on non-square feeds (sqrt of
    the HIDDEN size ~= 33 reshape)."""
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        _fix_siglip_pos_encoding,
    )

    emb = _make_siglip_embeddings_stub(70, 1152, patch_size=14)
    assert _fix_siglip_pos_encoding(emb), "fixture must be recognized and patched"

    # Non-square ASPECT feed: 37 x 37 in this symmetric case; the ROW ORDER
    # is what matters and is asserted below with distinct row content.
    gh = gw = 37
    feed = torch.randn(1, gh * gw, 1152)
    out = emb.interpolate_pos_encoding(feed, gh * 14, gw * 14)
    assert out.shape == (1, gh * gw, 1152)
    # Every returned row must BE a raw trained row (no blending): the packed
    # sequence of ids h*70 + w.
    table = emb.position_embedding.weight
    ids = (torch.arange(gh)[:, None] * 70 + torch.arange(gw)).reshape(-1)
    assert torch.equal(out[0], table[ids]), "pos rows must be exact trained-table lookups"


@pytest.mark.skip(reason="outdated: custom SigLIP pos-encoding patch removed; uses bagel _vit_embeddings")
def test_fix_siglip_pos_encoding_aspect_rect_rows():
    """True rectangular grid (gcg_seg profile: taller than wide): rows follow
    h-major order into the square trained table."""
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        _fix_siglip_pos_encoding,
    )

    emb = _make_siglip_embeddings_stub(70, 1152, patch_size=14)
    assert _fix_siglip_pos_encoding(emb)
    gh, gw = 37, 49
    feed = torch.randn(1, gh * gw, 1152)
    out = emb.interpolate_pos_encoding(feed, gh * 14, gw * 14)
    assert out.shape == (1, gh * gw, 1152)
    table = emb.position_embedding.weight
    ids = (torch.arange(gh)[:, None] * 70 + torch.arange(gw)).reshape(-1)
    assert torch.equal(out[0], table[ids])


@pytest.mark.skip(reason="outdated: custom SigLIP pos-encoding patch removed; uses bagel _vit_embeddings")
def test_fix_siglip_pos_encoding_rejects_oversized_grid():
    """No rows exist beyond 70x70; an oversized grid must fail loudly
    instead of silently indexing garbage."""
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        _fix_siglip_pos_encoding,
    )

    emb = _make_siglip_embeddings_stub(70, 1152, patch_size=14)
    assert _fix_siglip_pos_encoding(emb)
    with pytest.raises(RuntimeError, match="exceeds the learned position table"):
        emb.interpolate_pos_encoding(torch.randn(1, 71 * 71, 1152), 71 * 14, 71 * 14)


@pytest.mark.skip(reason="outdated: custom SigLIP pos-encoding patch removed; uses bagel _vit_embeddings")
def test_fix_siglip_pos_encoding_square_feed_uses_table_directly():
    """The early-return branch must remain: square + matching grid returns
    the raw position embedding rows unchanged."""
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        _fix_siglip_pos_encoding,
    )

    emb = _make_siglip_embeddings_stub(70, 1152, patch_size=14)
    assert _fix_siglip_pos_encoding(emb)
    feed = torch.randn(1, 4900, 1152)
    out = emb.interpolate_pos_encoding(feed, 980, 980)
    assert torch.equal(out, emb.position_embedding.weight.unsqueeze(0))


@pytest.mark.skip(reason="outdated: custom SigLIP pos-encoding patch removed; uses bagel _vit_embeddings")
def test_fix_siglip_pos_encoding_idempotent():
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        _fix_siglip_pos_encoding,
    )

    emb = _make_siglip_embeddings_stub(70, 1152, patch_size=14)
    assert _fix_siglip_pos_encoding(emb)
    bound_method = emb.interpolate_pos_encoding
    assert _fix_siglip_pos_encoding(emb)
    assert emb.interpolate_pos_encoding is bound_method, "must not rebind on a second call"


@pytest.mark.skip(reason="outdated: custom SigLIP pos-encoding patch removed; uses bagel _vit_embeddings")
def test_stock_vllm_interpolation_crashes_on_aspect_grid():
    """Documentation-by-test: the STOCK vllm SiglipVisionEmbeddings raises
    exactly the profile-run error on any non-square feed.  This pins WHY we
    need _fix_siglip_pos_encoding bound before the first img2img request."""
    try:
        from vllm.model_executor.models.siglip import SiglipVisionEmbeddings
    except ImportError:  # pragma: no cover
        pytest.skip("vllm siglip module unavailable")

    cfg = SimpleNamespace(
        hidden_size=1152,
        image_size=980,
        patch_size=14,
        num_channels=3,
    )
    # A bare-shell SiglipVisionEmbeddings: init ONLY enough nn.Module state
    # (no weights loaded) to attach the position-embedding submodule.
    emb = object.__new__(SiglipVisionEmbeddings)
    torch.nn.Module.__init__(emb)
    emb.config = cfg
    emb.embed_dim = cfg.hidden_size
    emb.image_size = cfg.image_size
    emb.patch_size = cfg.patch_size
    emb.num_patches = (cfg.image_size // cfg.patch_size) ** 2
    emb.num_positions = emb.num_patches
    emb.position_embedding = torch.nn.Embedding(emb.num_positions, cfg.hidden_size)
    emb.register_buffer("position_ids", torch.arange(emb.num_positions).unsqueeze(0))

    # 37x37 non-square feed -> stock code reshapes to sqrt(hidden)=33 -> boom.
    feed = torch.randn(1, 37 * 37, 1152)
    with pytest.raises(RuntimeError, match="invalid for input of size"):
        emb.interpolate_pos_encoding(feed, 37 * 14, 37 * 14)


# ---------------------------------------------------------------------------
# 3. Worst-case token budget arithmetic
# ---------------------------------------------------------------------------


def test_worst_case_token_budget_arithmetic():
    """Budget for the limit=10 cap with SenseNova VAE/ViT lockstep sizing.

    A 512x512 recon3d-size block uses VAE+sep+ViT placeholders and must fit
    inside one stage-0 prefill step (``max_num_batched_tokens: 32768``).
    With aspect-aware ViT the 10-image budget also fits in one step.
    """
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        _sensenova_img2img_token_counts,
    )

    num_vae, num_vit, _, _ = _sensenova_img2img_token_counts(512, 512)
    per_block = num_vae + 1 + num_vit
    assert per_block == 2398
    assert per_block <= 32768
    assert 10 * per_block <= 32768
