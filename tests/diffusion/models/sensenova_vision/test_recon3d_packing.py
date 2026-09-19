# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for SenseNova-Vision recon3d multi-view packing and per-task transforms.

``SenseNovaVisionPipeline._forward_recon3d`` decodes ``num_views`` square views
from a single injected AR KV context.  The packing arithmetic is ported verbatim
from upstream ``inference/inferencer.py::gen_image``:

    curr_kvlens = [kv_len] + [0] * (num_views - 1)
    curr_rope   = [rope0 + x for x in range(num_views)]
    image_sizes = [image_shape] * num_views

The per-view entries only build the packed indexes / RoPE / position ids: the
resulting ``packed_seqlens`` and the gen cache's ``key_values_lens`` are then
**collapsed into a single packed sequence** (upstream ``inferencer.py:164-166``),
so all views co-attend to the full AR KV cache and to each other inside one
non-causal sequence per CFG branch.

The per-task transform table distinguishes the VAE and ViT target sides
(e.g. recon3d -> VAE 512 / ViT 448; camera-pose -> ViT 560).  Everything here is
pure Python + PIL plus tensor stubs (no GPU, no checkpoint download).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.models.sensenova_vision.pipeline_sensenova_vision import (
    SenseNovaVisionPipeline,
)
from vllm_omni.diffusion.models.sensenova_vision.transforms_sensenova_vision import (
    PER_TASK_VAE_SIDE,
    PER_TASK_VIT_SIDE,
    ResizeSpec,
    max_long_edge_resize,
    packed_seqlens,
    recon3d_packing,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_recon3d_packing_n2() -> None:
    """Two views: first continues from the AR KV, second starts empty."""
    num_views = 2
    kv_len = 137
    base_rope = 137
    kv_lens, ropes = recon3d_packing(num_views, kv_len, base_rope)
    assert kv_lens == [137, 0]
    assert ropes == [137, 138]


def test_recon3d_packing_n4() -> None:
    """Four views share the same KV prefix, ropes increment per view."""
    kv_len = 300
    kv_lens, ropes = recon3d_packing(4, kv_len, 300)
    assert kv_lens == [300, 0, 0, 0]
    assert ropes == [300, 301, 302, 303]


def test_packed_seqlens_n2() -> None:
    """Per-branch packed_seqlens = (h*w + 2) markers; latent 32x32 -> 1026."""
    seqlens = packed_seqlens(2, 32, 32)
    assert seqlens == [1026, 1026]


def test_per_task_vae_side_contract() -> None:
    """recon3d selects VAE 512; camera-pose has no VAE prefill."""
    assert PER_TASK_VAE_SIDE["recon3d"] == 512
    assert PER_TASK_VAE_SIDE["camera_pose"] is None


def test_per_task_vit_side_contract() -> None:
    """recon3d ViT 448 / camera-pose ViT 560."""
    assert PER_TASK_VIT_SIDE["recon3d"] == 448
    assert PER_TASK_VIT_SIDE["camera_pose"] == 560


def test_resize_spec_target_side() -> None:
    """Stride-aligned square target: largest stride multiple <= max_size."""
    # ImageTransform(512, 256, 16) -> 512; (448, 224, 14) -> 448; (560, 378, 14) -> 560.
    assert ResizeSpec(512, 256, 16).target_side == 512
    assert ResizeSpec(448, 224, 14).target_side == 448
    assert ResizeSpec(560, 378, 14).target_side == 560


def test_resize_spec_vae_grid() -> None:
    """Latent grid for the recon3d VAE side (downsample 8, patch 2 -> 16)."""
    grid = ResizeSpec(512, 256, 16).vae_grid(latent_downsample=16)
    assert grid == (32, 32)


def test_max_long_edge_resize_downscales_to_target() -> None:
    """A square input above the max downscales to the stride-aligned target."""
    img = Image.new("RGB", (700, 700))
    fn = max_long_edge_resize(512, 256, 16)
    out = fn(img)
    assert out.size == (512, 512)


def test_resize_does_not_upscale_below_target() -> None:
    """Inputs already within max_size are left at their native size (no upscale)."""
    img = Image.new("RGB", (256, 256))
    out = max_long_edge_resize(512, 256, 16)(img)
    assert out.size == (256, 256)


def test_max_long_edge_resize_clamps_stride() -> None:
    """Output side is a multiple of stride and never below stride."""
    img = Image.new("RGB", (1024, 1024))
    out = max_long_edge_resize(560, 378, 14)(img)
    assert out.size[0] % 14 == 0
    assert out.size[0] <= 560


def _make_naive_cache(seq_len: int) -> NaiveCache:
    """A real NaiveCache with one layer whose KV rows are on the given seq."""
    cache = NaiveCache(num_layers=1)
    cache.key_cache[0] = torch.zeros(seq_len, 4)
    cache.value_cache[0] = torch.zeros(seq_len, 4)
    return cache


def test_naive_cache_merge_cfg_single_sequence_per_branch() -> None:
    """Merged CFG caches expose exactly one (collapsed) sequence per branch.

    Regression for the per-view KV split that made recon3d point maps
    meaningless: ``_forward_recon3d`` used to hand ``_forward_gen`` a
    ``key_values_lens`` of ``[kv_len, 0, ..., 0]`` per branch, so only view 0
    owned the AR context and the remaining views denoised from pure noise.
    The collapsed upstream topology (``inferencer.py:164-166``) keeps one
    packed sequence per CFG branch — the sequence that holds *all* views —
    so each branch cache contributes a single ``seq_len`` entry and
    ``Bagel.forward``'s ``batched_seqlens = packed_seqlens.repeat(num_branches)``
    stays index-aligned with the merged cache split.
    """
    gen = _make_naive_cache(100)
    gen.key_values_lens = [100]  # collapsed: all views in one sequence
    cfg_text = _make_naive_cache(200)
    cfg_text.key_values_lens = [200]
    cfg_img = _make_naive_cache(300)
    cfg_img.key_values_lens = [300]

    merged = NaiveCache.merge([gen, cfg_text, cfg_img])
    # One entry per CFG branch (matches batched_seqlens = packed_seqlens.repeat(3)).
    assert merged.key_values_lens == [100, 200, 300]
    # Rows are concatenated in the same branch order.
    assert merged.key_cache[0].shape[0] == 100 + 200 + 300
    # split_with_zeros must produce exactly len(merged.key_values_lens) slices.
    slices = NaiveCache.split_with_zeros(merged.key_cache[0], merged.key_values_lens)
    assert len(slices) == 3
    assert slices[0].shape[0] == 100
    assert slices[1].shape[0] == 200  # cfg_text branch
    assert slices[2].shape[0] == 300  # cfg_img branch


def _cache(seq_len: int) -> SimpleNamespace:
    """A minimal duck-typed naive KV cache with ``key_cache[0]`` shaped rows.

    ``NaiveCache.from_object`` iterates the cache to rebuild layer-indexed
    tensors, so the mock exposes per-layer tensors as an indexable iterable.
    """
    return SimpleNamespace(
        key_cache=[torch.zeros(seq_len, 4)],
        value_cache=[torch.zeros(seq_len, 4)],
    )


def _recon3d_request(*, num_views: int = 2, mode: str = "recon3d") -> DiffusionRequestBatch:
    params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        extra_args={"sensenova_vision_mode": mode, "num_views": num_views},
        past_key_values=_cache(16),
        kv_metadata={"ropes": [16], "image_shape": [16, 16]},
    )
    req = OmniDiffusionRequest(prompt="recon3d", request_id="req-recon3d", sampling_params=params)
    return DiffusionRequestBatch(requests=[req])


def _recon3d_request_from_prompt(*, num_markers: int, num_views: int | None = None) -> DiffusionRequestBatch:
    """A recon3d request whose prompt carries ``num_markers`` <|fim_middle|> views.

    Mirrors ``_format_recon3d_prompts``: each conditioned input view contributes
    exactly one marker at the head of the prompt.  ``num_views=None`` leaves the
    per-request knob unset so ``_forward_recon3d`` must derive the default.
    """
    extra_args: dict[str, Any] = {"sensenova_vision_mode": "recon3d"}
    if num_views is not None:
        extra_args["num_views"] = num_views
    params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        extra_args=extra_args,
        past_key_values=_cache(16),
        kv_metadata={"ropes": [16], "image_shape": [16, 16]},
    )
    prompt = {"prompt": "<|fim_middle|>" * num_markers + "<|im_start|>recon3d<|im_end|>", "modalities": ["img2img"]}
    req = OmniDiffusionRequest(prompt=prompt, request_id="req-recon3d-views", sampling_params=params)
    return DiffusionRequestBatch(requests=[req])


def _fake_prepare_vae_latent(**kw: Any) -> dict[str, Any]:
    """``prepare_vae_latent`` stub returning *per-view* ``packed_seqlens``.

    Mirrors ``Bagel.prepare_input``: one ``h*w + 2`` entry per view, where
    ``h*w`` is the latent token count for the requested ``image_shape``.  The
    pipeline collapses this to a single summed entry before denoising and
    re-splits the denoised latents with the saved per-view values, so the stub
    must stay per-view and tensor-typed.
    """
    image_sizes = kw.get("image_sizes", [])
    downsample = 8
    per_view = [(h // downsample) * (w // downsample) + 2 for h, w in image_sizes]
    return {
        "packed_seqlens": torch.tensor(per_view, dtype=torch.int),
        "packed_init_noises": torch.zeros(sum(s - 2 for s in per_view), 1),
        "image_sizes": image_sizes,
    }


def _fake_generate_image(**kw: Any) -> tuple[list[Any], None, None, None]:
    """``generate_image`` stub unpacking ``x_t`` by the unpack lengths.

    Mirrors the real tail of ``Bagel.generate_image``: the denoised ``x_t``
    (whose token count matches the collapsed ``packed_seqlens``) is split by
    ``unpack_seqlens - 2``, i.e. one chunk of ``h*w`` latent tokens per view.
    """
    unpack_seqlens = kw["unpack_seqlens"]
    return [torch.zeros(max(s - 2, 0), 1) for s in unpack_seqlens.tolist()], None, None, None


def _recon3d_pipeline() -> SenseNovaVisionPipeline:
    """Build a SenseNovaVisionPipeline instance without loading weights."""
    pipeline = object.__new__(SenseNovaVisionPipeline)
    pipeline.bagel = SimpleNamespace(
        latent_downsample=8,
        max_latent_size=64,
        prepare_vae_latent=_fake_prepare_vae_latent,
        generate_image=_fake_generate_image,
    )
    pipeline.new_token_ids = {}
    pipeline.device = torch.device("cpu")
    pipeline.scheduler = None
    pipeline.scheduler_kwargs = None
    pipeline.od_config = SimpleNamespace(dtype=torch.bfloat16)
    pipeline.vae = SimpleNamespace()
    pipeline._stage_durations = None
    pipeline._decode_image_from_latent = lambda *a: Image.new("RGB", (4, 4))
    return pipeline


def test_is_recon3d_selects_mode() -> None:
    """Only the recon3d mode routes to the multi-view decode."""
    pipeline = _recon3d_pipeline()
    assert pipeline._is_recon3d(_recon3d_request(mode="recon3d")) is True
    assert pipeline._is_recon3d(_recon3d_request(mode="generate")) is False


def test_single_stage_context_resize_uses_sensenova_transforms() -> None:
    """Local prefill uses the model-executor VAE then ViT resize chains."""
    pipeline = object.__new__(SenseNovaVisionPipeline)
    image = Image.new("RGB", (1600, 800))

    default_vae = pipeline._resize_context_image(image, num_images=1)
    default_vit = pipeline._context_vit_transform(default_vae, num_images=1)
    assert default_vae.size == (1024, 512)
    assert tuple(default_vit.shape) == (3, 490, 980)

    recon_vae = pipeline._resize_context_image(image, num_images=2)
    recon_vit = pipeline._context_vit_transform(recon_vae, num_images=2)
    assert recon_vae.size == (512, 256)
    assert tuple(recon_vit.shape) == (3, 224, 448)


def test_forward_recon3d_decodes_num_views_images() -> None:
    """``_forward_recon3d`` decodes one PIL image per view and packs them as a list."""
    pipeline = _recon3d_pipeline()
    out = pipeline._forward_recon3d(_recon3d_request(num_views=3))
    payload = out.output["payload"]
    assert isinstance(payload["image"], list)
    assert len(payload["image"]) == 3
    assert all(isinstance(img, Image.Image) for img in payload["image"])


def test_forward_recon3d_single_stage_prefills_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-KV recon3d reuses locally-prefilled CFG contexts and emits every view."""
    pipeline = _recon3d_pipeline()
    contexts = []
    for _ in range(3):
        cache = _make_naive_cache(16)
        contexts.append({"kv_lens": [16], "ropes": [16], "past_key_values": cache})

    monkeypatch.setattr(
        pipeline,
        "_prepare_single_stage_contexts",
        lambda first_prompt, sampling: (contexts[0], contexts[1], contexts[2], (16, 16)),
    )
    params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        extra_args={"sensenova_vision_mode": "recon3d"},
    )
    prompt = {
        "prompt": "<|image_pad|><|image_pad|>recon3d",
        "modalities": ["img2img"],
        "multi_modal_data": {"img2img": [Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16))]},
    }
    req = OmniDiffusionRequest(prompt=prompt, request_id="req-single-stage-recon3d", sampling_params=params)

    out = pipeline._forward_recon3d(DiffusionRequestBatch(requests=[req]))
    assert len(out.output["payload"]["image"]) == 2
    assert contexts[0]["past_key_values"].key_values_lens == [16]


def test_single_stage_omits_empty_cfg_companion_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty text-CFG cache must use BAGEL's native injected-KV fallback."""
    pipeline = _recon3d_pipeline()
    gen_cache = _make_naive_cache(8)
    empty_cache = NaiveCache(num_layers=1)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        pipeline,
        "_prepare_single_stage_contexts",
        lambda first_prompt, sampling: (
            {"kv_lens": [8], "ropes": [8], "past_key_values": gen_cache},
            {"kv_lens": [0], "ropes": [0], "past_key_values": empty_cache},
            {"kv_lens": [8], "ropes": [8], "past_key_values": gen_cache},
            (16, 16),
        ),
    )

    def capture_base_forward(self, first_prompt, sampling, *, prepare_only=False):
        captured["sampling"] = sampling
        return SimpleNamespace()

    monkeypatch.setattr(BagelPipeline, "_forward_single", capture_base_forward)
    sampling = OmniDiffusionSamplingParams(num_inference_steps=2)
    pipeline._forward_single({"prompt": "test", "modalities": ["image"]}, sampling)

    local_sampling = captured["sampling"]
    assert local_sampling.cfg_text_past_key_values is None
    assert local_sampling.cfg_text_kv_metadata is None
    assert local_sampling.cfg_img_past_key_values is gen_cache


def test_count_conditioned_views_counts_fim_markers() -> None:
    """Conditioned views = <|fim_middle|> markers; string prompts count zero."""
    assert SenseNovaVisionPipeline._count_conditioned_views(_recon3d_request_from_prompt(num_markers=3)) == 3
    assert SenseNovaVisionPipeline._count_conditioned_views(_recon3d_request_from_prompt(num_markers=0)) == 0
    assert SenseNovaVisionPipeline._count_conditioned_views(_recon3d_request(num_views=2)) == 0


def test_forward_recon3d_defaults_to_conditioned_view_count() -> None:
    """Without an explicit num_views, decode one image per conditioned view.

    Regression for the 3-input-views -> 4-output-images bug: upstream
    ``reconstruct_3d`` derives ``num_output_vae`` from the input image count
    (``interleave_inference``: ``max(input_image_count, 1)``), so the port must
    default to the conditioned view count instead of a hardcoded 4.
    """
    pipeline = _recon3d_pipeline()
    out = pipeline._forward_recon3d(_recon3d_request_from_prompt(num_markers=3))
    payload = out.output["payload"]
    assert isinstance(payload["image"], list)
    assert len(payload["image"]) == 3


def test_forward_recon3d_zero_conditioned_views_decodes_one() -> None:
    """No markers (e.g. a server-style prompt) falls back to one output view."""
    pipeline = _recon3d_pipeline()
    out = pipeline._forward_recon3d(_recon3d_request_from_prompt(num_markers=0))
    assert len(out.output["payload"]["image"]) == 1


def test_forward_recon3d_explicit_num_views_override_wins() -> None:
    """An explicit num_views >= the conditioned count still overrides the default."""
    pipeline = _recon3d_pipeline()
    out = pipeline._forward_recon3d(_recon3d_request_from_prompt(num_markers=3, num_views=5))
    assert len(out.output["payload"]["image"]) == 5


def test_forward_recon3d_num_views_must_cover_conditioned_views() -> None:
    """An explicit num_views below the conditioned view count is rejected.

    Mirrors upstream ``reconstruct_3d``, which raises when predictions are
    fewer than input images: every conditioned view must get an output.
    """
    pipeline = _recon3d_pipeline()
    with pytest.raises(ValueError, match="must cover every conditioned view"):
        pipeline._forward_recon3d(_recon3d_request_from_prompt(num_markers=3, num_views=2))


def test_forward_recon3d_cfg_feeds_branch_inputs() -> None:
    """CFG-enabled recon3d passes multi-view branch pids/KVs into generate_image.

    Regression for the sequential-CFG/SP path: ``_forward_recon3d`` must
    supply ``cfg_text_packed_position_ids`` (and ``cfg_img_*`` when image CFG
    is enabled), otherwise ``Bagel.forward`` crashes with ``cat(): expected
    Tensor as element 1, got NoneType`` (bagel_transformer.py:2510).
    """
    captured: dict[str, Any] = {}

    def _capture_generate_image(**kw: Any) -> tuple[list[Any], None, None, None]:
        captured.update(kw)
        return _fake_generate_image(**kw)

    pipeline = _recon3d_pipeline()
    pipeline.bagel.generate_image = _capture_generate_image

    def _fake_cfg_pids(**kw: Any) -> dict[str, torch.Tensor]:
        # Mirrors prepare_vae_latent_cfg: per view, (h*w + 2) ids at the view rope.
        img_shape = kw["image_sizes"][0]
        hw = (img_shape[0] // 8) * (img_shape[1] // 8)
        pids = [[rope] * (hw + 2) for rope in kw["curr_rope"]]
        return {"cfg_packed_position_ids": torch.tensor(pids, dtype=torch.long).reshape(-1)}

    pipeline.bagel.prepare_vae_latent_cfg = _fake_cfg_pids

    params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        extra_args={"sensenova_vision_mode": "recon3d", "num_views": 3, "cfg_text_scale": 4.0, "cfg_img_scale": 1.0},
        past_key_values=_cache(16),
        cfg_text_past_key_values=_cache(16),
        kv_metadata={"ropes": [16], "image_shape": [16, 16]},
    )
    req = OmniDiffusionRequest(prompt="recon3d", request_id="req-recon3d-cfg", sampling_params=params)
    pipeline._forward_recon3d(DiffusionRequestBatch(requests=[req]))

    assert captured["cfg_text_scale"] == 4.0
    # The sequential-CFG path requires a non-None cfg_text branch pid tensor.
    assert captured.get("cfg_text_packed_position_ids") is not None
    assert captured.get("cfg_text_past_key_values") is not None
    # num_views=3 => branch pids length = 3 * (h*w + 2); h*w = (16//8)^2 = 4.
    assert captured["cfg_text_packed_position_ids"].numel() == 3 * 6
    assert "cfg_img_packed_position_ids" not in captured  # img CFG disabled
    # Every branch — gen and each CFG companion — is ONE packed sequence
    # holding all views, and each attends to the full AR KV (no per-view
    # split left behind for any cache), matching upstream inferencer.py:164-166.
    gen_cache = captured["past_key_values"]
    assert gen_cache.key_values_lens == [16]  # collapsed: one sequence, full AR context
    cfg_text_cache = captured["cfg_text_past_key_values"]
    assert cfg_text_cache is not gen_cache
    assert cfg_text_cache.key_values_lens == [16]
    merged = NaiveCache.merge([gen_cache, cfg_text_cache])
    # 2 CFG branches -> 2 per-sequence entries (matches
    # batched_seqlens = packed_seqlens.repeat(2)).
    assert merged.key_values_lens == [16, 16]
    assert len(merged.key_values_lens) == 2


def test_forward_recon3d_collapses_to_single_packed_sequence() -> None:
    """The N views are collapsed into ONE packed query sequence per branch.

    Regression for the garbage recon3d point maps: with the old per-view
    topology only view 0 attended to the AR context (its cache slice was the
    whole cache; the rest got 0-length slices) and no view ever attended to
    the others, so views 1..N-1 decoded from pure noise.  Upstream instead
    builds the per-view entries only for the packed indexes / RoPE / position
    ids and then sums them (``inferencer.py:164-166``), giving a single
    non-causal sequence in which every view sees the full AR KV *and* its
    sibling views.
    """
    captured: dict[str, Any] = {}
    prepare_calls: list[dict[str, Any]] = []

    def _capture_prepare_vae_latent(**kw: Any) -> dict[str, Any]:
        prepare_calls.append(kw)
        return _fake_prepare_vae_latent(**kw)

    def _capture_generate_image(**kw: Any) -> tuple[list[Any], None, None, None]:
        captured.update(kw)
        return _fake_generate_image(**kw)

    pipeline = _recon3d_pipeline()
    pipeline.bagel.prepare_vae_latent = _capture_prepare_vae_latent
    pipeline.bagel.generate_image = _capture_generate_image

    out = pipeline._forward_recon3d(_recon3d_request(num_views=3))

    # prepare_vae_latent still receives the per-view entries (they build the
    # packed indexes / RoPE / position ids).
    [prepare_kw] = prepare_calls
    assert prepare_kw["curr_kvlens"] == [16, 0, 0]
    assert prepare_kw["curr_rope"] == [16, 17, 18]
    assert len(prepare_kw["image_sizes"]) == 3

    # ... but the denoise loop gets ONE packed sequence: all 3 views, h*w + 2
    # tokens each (h*w = 4 for a 16x16 shape at downsample 8) => 18.
    packed_seqlens = captured["packed_seqlens"]
    assert packed_seqlens.numel() == 1
    assert int(packed_seqlens.sum()) == 3 * 6
    # The gen cache exposes the same collapsed shape, so ``_forward_gen``
    # splits the merged cache once per CFG branch and every view sees the
    # full AR KV.
    assert captured["past_key_values"].key_values_lens == [16]
    # The final unpack still uses the per-view lengths (upstream keeps the
    # pre-collapse variable for ``x_0.split`` at inferencer.py:225).
    assert captured["unpack_seqlens"].tolist() == [6, 6, 6]
    assert len(out.output["payload"]["image"]) == 3
