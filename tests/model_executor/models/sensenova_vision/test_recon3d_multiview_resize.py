# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the recon3d multi-view VAE resize in the AR stage.

When a request conditions on more than one img2img image (multi-view
recon3d), ``_process_img2img_input`` must apply the upstream per-task
``recon3d_vae_transform`` (``ImageTransform(512, 256, 16)``) instead of the
checkpoint default ``ImageTransform(1024, 512, 16)`` -- mirroring upstream
``reconstruct_3d`` swapping in the recon3d transforms for every conditioned
view.  ``OmniSenseNovaVisionMultiModalProcessor`` must apply the same gate
when sizing the ``<|fim_middle|>`` placeholder blocks so the prompt token
counts keep matching the runtime VAE embeds.

The resized (h_px, w_px) flows through ``_register_img2img_info`` into
``kv_metadata["image_shape"]``, so the recon3d DiT output grid follows the
recon3d VAE transform too (e.g. 384x512 for 4:3 views, 512x512 for square
benchmark views).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems

from vllm_omni.model_executor.models.bagel import bagel as bagel_module
from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
    SENSENOVA_RECON3D_VIT_MAX_SIZE,
    OmniSenseNovaVisionForConditionalGeneration,
    OmniSenseNovaVisionMultiModalProcessor,
    OmniSenseNovaVisionProcessingInfo,
    _sensenova_img2img_token_counts,
    _sensenova_vae_resize_dims,
    _sensenova_vit_patch_count,
    _sensenova_vit_resize_dims,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Resize arithmetic
# ---------------------------------------------------------------------------


def test_default_vae_resize_unchanged_for_43_views() -> None:
    """Single-image default stays ImageTransform(1024, 512, 16): 1920x1440 -> 1024x768."""
    assert _sensenova_vae_resize_dims(1440, 1920) == (768, 1024)


def test_recon3d_vae_resize_43_views() -> None:
    """ImageTransform(512, 256, 16) on the 4:3 benchmark views -> 512x384 (W x H)."""
    assert _sensenova_vae_resize_dims(1440, 1920, max_size=512, min_size=256) == (384, 512)


def test_recon3d_vae_resize_square_views() -> None:
    """Square views at/below the max are kept (no upscale, stride aligned)."""
    assert _sensenova_vae_resize_dims(512, 512, max_size=512, min_size=256) == (512, 512)
    assert _sensenova_vae_resize_dims(700, 700, max_size=512, min_size=256) == (512, 512)


def test_recon3d_vae_resize_floors_short_edge() -> None:
    """A small input is upscaled so the short edge reaches 256 (200x300 -> 256x384)."""
    assert _sensenova_vae_resize_dims(200, 300, max_size=512, min_size=256) == (256, 384)


def test_default_vit_resize_unchanged() -> None:
    """Default ViT chain stays ImageTransform(980, 224, 14): 384x512 -> 378x518."""
    assert _sensenova_vit_resize_dims(384, 512) == (378, 518)


def test_recon3d_vit_resize_dims() -> None:
    """ImageTransform(448, 224, 14) on the recon3d VAE grids.

    4:3 VAE grid 384x512 -> 336x448 (768 patches); square VAE grid 512x512
    -> 448x448 (1024 patches), the upstream benchmark geometry.
    """
    assert _sensenova_vit_resize_dims(384, 512, max_size=SENSENOVA_RECON3D_VIT_MAX_SIZE) == (336, 448)
    assert _sensenova_vit_resize_dims(512, 512, max_size=SENSENOVA_RECON3D_VIT_MAX_SIZE) == (448, 448)
    assert _sensenova_vit_patch_count(384, 512, vit_max_size=SENSENOVA_RECON3D_VIT_MAX_SIZE) == 768
    assert _sensenova_vit_patch_count(512, 512, vit_max_size=SENSENOVA_RECON3D_VIT_MAX_SIZE) == 1024


# ---------------------------------------------------------------------------
# Token counts (placeholder sizing input)
# ---------------------------------------------------------------------------


def test_token_counts_single_image_keeps_default() -> None:
    """num_images=1 (the default) must reproduce the pre-change behavior."""
    num_vae_total, num_vit_total, vae_h, vae_w = _sensenova_img2img_token_counts(1440, 1920)
    assert (vae_h, vae_w) == (768, 1024)
    assert num_vae_total == (768 // 16) * (1024 // 16) + 2
    assert num_vit_total == _sensenova_vit_patch_count(768, 1024) + 2


def test_token_counts_multi_image_switch_to_recon3d() -> None:
    """num_images>1 sizes VAE+ViT blocks at the recon3d transform targets."""
    num_vae_total, num_vit_total, vae_h, vae_w = _sensenova_img2img_token_counts(1440, 1920, num_images=3)
    assert (vae_h, vae_w) == (384, 512)
    assert num_vae_total == (384 // 16) * (512 // 16) + 2
    assert num_vit_total == _sensenova_vit_patch_count(384, 512, vit_max_size=SENSENOVA_RECON3D_VIT_MAX_SIZE) + 2
    # And it must differ from the single-image sizing.
    assert num_vae_total != _sensenova_img2img_token_counts(1440, 1920)[0]
    assert num_vit_total != _sensenova_img2img_token_counts(1440, 1920)[1]


def test_token_counts_square_multiview_matches_benchmark() -> None:
    """Square recon3d views (upstream benchmark) -> 512x512, 32x32 latent grid."""
    num_vae_total, _, vae_h, vae_w = _sensenova_img2img_token_counts(512, 512, num_images=2)
    assert (vae_h, vae_w) == (512, 512)
    assert num_vae_total == 32 * 32 + 2


# ---------------------------------------------------------------------------
# Model-side resize methods
# ---------------------------------------------------------------------------


def _model_stub() -> OmniSenseNovaVisionForConditionalGeneration:
    inst = object.__new__(OmniSenseNovaVisionForConditionalGeneration)
    inst.latent_downsample = 16
    inst.max_latent_size = 64
    inst.latent_channel = 16
    inst.latent_patch_size = 2
    inst.config = SimpleNamespace(vit_config=SimpleNamespace(image_size=64, patch_size=14))
    inst.device = torch.device("cpu")
    return inst


def test_resize_methods_select_the_right_grid() -> None:
    inst = _model_stub()
    pv = torch.zeros(1, 3, 200, 300)
    assert inst._resize_to_stride(pv).shape[2:] == (512, 768)
    assert inst._resize_to_recon3d_vae(pv).shape[2:] == (256, 384)
    # ViT transforms operate on the already-VAE-resized image.
    assert inst._resize_for_vit(torch.zeros(1, 3, 384, 512)).shape[2:] == (378, 518)
    assert inst._resize_to_recon3d_vit(torch.zeros(1, 3, 384, 512)).shape[2:] == (336, 448)


# ---------------------------------------------------------------------------
# Embed gate: 1 image -> default VAE grid, >1 images -> recon3d VAE grid
# ---------------------------------------------------------------------------


def _wire_embed_fakes(inst, calls: dict) -> None:
    """Attach the fakes ``_process_img2img_input`` needs; record resize calls."""

    def fake_vit_embeddings(images):
        calls["vit_sizes"] = [tuple(img.shape[-2:]) for img in images]
        return [torch.zeros(1, 4) for _ in images]

    class _FakeVAE:
        def encode(self, x):
            return torch.zeros(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    orig_stride = OmniSenseNovaVisionForConditionalGeneration._resize_to_stride
    orig_recon3d = OmniSenseNovaVisionForConditionalGeneration._resize_to_recon3d_vae
    orig_vit_default = OmniSenseNovaVisionForConditionalGeneration._resize_for_vit
    orig_vit_recon3d = OmniSenseNovaVisionForConditionalGeneration._resize_to_recon3d_vit

    def stride_resize(pv):
        calls.setdefault("stride", []).append(tuple(pv.shape[2:]))
        return orig_stride(inst, pv)

    def recon3d_resize(pv):
        calls.setdefault("recon3d", []).append(tuple(pv.shape[2:]))
        return orig_recon3d(inst, pv)

    def vit_default(pv):
        calls.setdefault("vit_default", []).append(tuple(pv.shape[2:]))
        return orig_vit_default(inst, pv)

    def vit_recon3d(pv):
        calls.setdefault("vit_recon3d", []).append(tuple(pv.shape[2:]))
        return orig_vit_recon3d(inst, pv)

    inst._vit_embeddings = fake_vit_embeddings
    inst._resize_to_stride = stride_resize
    inst._resize_to_recon3d_vae = recon3d_resize
    inst._resize_for_vit = vit_default
    inst._resize_to_recon3d_vit = vit_recon3d
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


def test_embed_single_image_uses_default_vae_grid() -> None:
    inst = _model_stub()
    calls: dict = {}
    _wire_embed_fakes(inst, calls)

    inst._process_img2img_input({"pixel_values": torch.zeros(1, 1, 3, 200, 300)})

    assert calls.get("recon3d") is None, "single image must not take the recon3d transform"
    assert calls.get("vit_recon3d") is None, "single image must not take the recon3d ViT transform"
    assert calls["stride"] == [(200, 300)]
    # ViT sees the default-chain dims of the VAE-resized image.
    assert calls["vit_default"] == [(512, 768)]
    # info (h_px, w_px) follows the default grid and feeds kv_metadata["image_shape"].
    infos = list(inst._img2img_info_by_size.values())
    assert len(infos) == 1 and infos[0][2:] == (512, 768)
    assert len(inst._pending_img2img_info) == 1


def test_embed_multi_image_uses_recon3d_vae_grid() -> None:
    inst = _model_stub()
    calls: dict = {}
    _wire_embed_fakes(inst, calls)

    inst._process_img2img_input({"pixel_values": torch.zeros(1, 2, 3, 200, 300)})

    assert calls.get("stride") is None, "multi-view must not take the default transform"
    assert calls.get("vit_default") is None, "multi-view must not take the default ViT transform"
    assert calls["recon3d"] == [(200, 300)] * 2
    # ViT sees the recon3d-chain dims of the VAE-resized image.
    assert calls["vit_recon3d"] == [(256, 384)] * 2
    assert calls["vit_sizes"] == [(252, 378)] * 2
    # Every view's info carries the recon3d VAE dims -> DiT image_shape.  The
    # size cache is keyed by (num_vae, num_vit), so two identical views
    # collapse to one entry (base-class dedup); the pending list stays 1/image.
    infos = list(inst._img2img_info_by_size.values())
    assert len(infos) == 1 and infos[0][2:] == (256, 384)
    assert len(inst._pending_img2img_info) == 2
    assert all(info[2:] == (256, 384) for info in inst._pending_img2img_info)


# ---------------------------------------------------------------------------
# Processor parity: placeholder counts must match the embed-side gate
# ---------------------------------------------------------------------------


def _cached_checkpoint() -> str | None:
    import glob
    import os

    env_path = os.environ.get("SENSENOVA_VISION_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--sensenova--SenseNova-Vision-7B-MoT/snapshots/*")
    matches = sorted(glob.glob(snapshot))
    return matches[-1] if matches else None


@pytest.fixture(scope="module")
def _sensenova_processor():
    """A real OmniSenseNovaVisionMultiModalProcessor with stubbed ctx."""
    checkpoint = _cached_checkpoint()
    if checkpoint is None:
        pytest.skip("SenseNova-Vision-7B-MoT not cached and SENSENOVA_VISION_MODEL_PATH is unset")

    from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
        VLLMSenseNovaVisionTokenizer,
    )

    tokenizer = VLLMSenseNovaVisionTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )
    hf_config = SimpleNamespace(
        vit_max_num_patch_per_side=70,
        latent_patch_size=2,
        max_latent_size=64,
        vae_config={"downsample": 8, "z_channels": 16},
        vit_config=SimpleNamespace(image_size=980, patch_size=14),
    )
    ctx = SimpleNamespace(
        tokenizer=tokenizer,
        hf_config=hf_config,
        model_config=SimpleNamespace(
            model=checkpoint,
            get_multimodal_config=lambda: SimpleNamespace(enable_mm_embeds=False),
        ),
        get_tokenizer=lambda: tokenizer,
        get_hf_config=lambda: hf_config,
    )
    info = OmniSenseNovaVisionProcessingInfo(ctx)
    proc = object.__new__(OmniSenseNovaVisionMultiModalProcessor)
    proc.info = info
    proc.dummy_inputs = None
    proc.cache = None
    proc.data_parser = info.get_data_parser()
    return proc, tokenizer


def _img2img_placeholder_lengths(proc, tokenizer, sizes_w_h: list[tuple[int, int]]) -> list[int]:
    images = [Image.new("RGB", size) for size in sizes_w_h]
    mm_items = MultiModalDataItems({"img2img": bagel_module.Img2ImgProcessorItems(images)})
    updates = proc._get_prompt_updates(mm_items, {}, MultiModalKwargsItems())
    mm_prompt_updates = proc._bind_and_group_updates(updates, mm_items.get_all_counts())
    prompt_ids = [tokenizer.convert_tokens_to_ids("<|fim_middle|>")] * len(sizes_w_h)
    _new_ids, placeholders = proc._apply_prompt_updates(prompt_ids, mm_prompt_updates)
    blocks = placeholders["img2img"]
    assert [ph.item_idx for ph in blocks] == list(range(len(sizes_w_h)))
    return [ph.length for ph in blocks]


def test_processor_placeholder_matches_multiview_embeds(_sensenova_processor) -> None:
    """3 img2img items -> recon3d-sized placeholder blocks (VAE 384x512 grid)."""
    proc, tokenizer = _sensenova_processor
    lengths = _img2img_placeholder_lengths(proc, tokenizer, [(1920, 1440)] * 3)

    num_vae_total, num_vit_total, _, _ = _sensenova_img2img_token_counts(1440, 1920, num_images=3)
    # block = vae_total + 1 separator + vit_total
    assert lengths == [num_vae_total + 1 + num_vit_total] * 3


def test_processor_placeholder_single_image_keeps_default(_sensenova_processor) -> None:
    """A lone img2img item keeps the default-sized placeholder block."""
    proc, tokenizer = _sensenova_processor
    (length,) = _img2img_placeholder_lengths(proc, tokenizer, [(1920, 1440)])

    num_vae_total, num_vit_total, _, _ = _sensenova_img2img_token_counts(1440, 1920)
    assert length == num_vae_total + 1 + num_vit_total


def test_processor_placeholder_gate_follows_item_count(_sensenova_processor) -> None:
    """Same image size: 1 item -> default sizing, 2 items -> recon3d sizing.

    For 4:3 views the recon3d VAE grid (512x384) is smaller than the default
    (1024x768), so the multi-view placeholder blocks shrink.
    """
    proc, tokenizer = _sensenova_processor
    single = _img2img_placeholder_lengths(proc, tokenizer, [(1920, 1440)])
    multi = _img2img_placeholder_lengths(proc, tokenizer, [(1920, 1440)] * 2)
    assert single[0] > multi[0], "the recon3d VAE grid must shrink the multi-view placeholder blocks"
