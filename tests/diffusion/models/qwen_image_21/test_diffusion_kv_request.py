# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import PIL.Image
import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from vllm.lora.request import LoRARequest
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

import vllm_omni.diffusion.diffusion_engine as engine_module
import vllm_omni.diffusion.diffusion_kv.kv_cache_utils as kv_utils
import vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 as pipeline_module
import vllm_omni.diffusion.models.qwen_image_21.request_layout as layout_module
from tests.helpers.kv_layout import build_kv_cache_tensor
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    QwenImage21Pipeline,
    get_qwen_image_21_pre_process_func,
)
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Transformer2DModel,
    _validate_paged_prompt_mask,
)
from vllm_omni.diffusion.models.qwen_image_21.request_layout import (
    PROMPT_TEMPLATE_T2I,
    SYS_PROMPT,
    QwenImage21LayoutTokenizer,
    QwenImage21PreparedLayout,
    build_qwen_image_21_diffusion_kv_requests,
    native_kv_covers_cond_images,
    prepare_qwen_image_21_layout,
    prepare_qwen_image_21_prefix_cache,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeCharTokenizer:
    """Deterministic character-level tokenizer with a single token for <|image_pad|>."""

    IMG_PAD_ID = 9999

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        ids: list[int] = []
        idx = 0
        marker = "<|image_pad|>"
        while idx < len(text):
            if text.startswith(marker, idx):
                ids.append(self.IMG_PAD_ID)
                idx += len(marker)
            else:
                ids.append(ord(text[idx]) + 10)
                idx += 1
        return {"input_ids": ids}


def _layout_tokenizer() -> QwenImage21LayoutTokenizer:
    return QwenImage21LayoutTokenizer(_FakeCharTokenizer())


def _make_ref_payloads(
    num_images: int = 2,
    *,
    fill: float = 0.25,
    input_image_sizes: list[tuple[int, int]] | None = None,
):
    input_image_sizes = input_image_sizes or [(64, 64) for _ in range(num_images)]
    assert len(input_image_sizes) == num_images
    vae_images = [torch.full((1, 4, 1, 64, 64), fill + i, dtype=torch.float32) for i in range(num_images)]
    prompt_images = [
        PIL.Image.new("RGB", size, color=(int(50 + i * 20), 60, 70)) for i, size in enumerate(input_image_sizes)
    ]
    return {
        "prompt_image": prompt_images,
        "vae_images": vae_images,
        "input_image_sizes": input_image_sizes,
    }


def _make_request(
    *,
    prompt: str = "restyle card one with card two",
    negative_prompt: str | None = None,
    true_cfg_scale: float = 1.0,
    num_images: int = 2,
    request_id: str = "req-1",
    seed: int | None = 42,
    height: int = 64,
    width: int = 64,
    fill: float = 0.25,
    input_image_sizes: list[tuple[int, int]] | None = None,
) -> OmniDiffusionRequest:
    prompt_dict: dict = {
        "prompt": prompt,
        "additional_information": (
            _make_ref_payloads(num_images, fill=fill, input_image_sizes=input_image_sizes) if num_images > 0 else {}
        ),
    }
    if negative_prompt is not None:
        prompt_dict["negative_prompt"] = negative_prompt
    return OmniDiffusionRequest(
        prompt=prompt_dict,
        sampling_params=OmniDiffusionSamplingParams(
            height=height,
            width=width,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=4,
            seed=seed,
        ),
        request_id=request_id,
    )


def test_dense_legacy_preprocess_does_not_initialize_layout_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(
        layout_module,
        "QwenImage21LayoutTokenizer",
        lambda *_args, **_kwargs: pytest.fail("dense_legacy must not initialize QwenImage21LayoutTokenizer"),
    )

    get_qwen_image_21_pre_process_func(
        SimpleNamespace(model="model", diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY)
    )


def test_paged_preprocess_attaches_layout_and_scheduler_kv_requests(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(
        layout_module,
        "QwenImage21LayoutTokenizer",
        lambda *_args, **_kwargs: _layout_tokenizer(),
    )
    preprocess = get_qwen_image_21_pre_process_func(
        SimpleNamespace(
            model="model",
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            diffusion_kv_max_rows_per_request=2,
        )
    )
    req = OmniDiffusionRequest(
        prompt={
            "prompt": "make the yacht white",
            "negative_prompt": "blurry",
            "multi_modal_data": {"image": [PIL.Image.new("RGB", (512, 512))]},
        },
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            true_cfg_scale=4.0,
            num_inference_steps=4,
            seed=7,
        ),
        request_id="edit-req",
    )

    prepared_req = preprocess(req)

    assert prepared_req is req
    assert isinstance(req.prepared_layout, QwenImage21PreparedLayout)
    assert req.prepared_layout.num_branches == 2
    assert req.diffusion_kv_requests is not None
    assert len(req.diffusion_kv_requests) == 2
    assert req.prepared_layout.target_len == 16
    assert [r.sequence_id for r in req.diffusion_kv_requests] == [0, 1]


def test_paged_preprocess_expands_startup_kv_profile_request(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(
        layout_module,
        "QwenImage21LayoutTokenizer",
        lambda *_args, **_kwargs: _layout_tokenizer(),
    )
    preprocess = get_qwen_image_21_pre_process_func(
        SimpleNamespace(
            model="model",
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            diffusion_kv_max_rows_per_request=2,
        )
    )
    profile_req = OmniDiffusionRequest(
        prompt={"prompt": "dummy run"},
        sampling_params=OmniDiffusionSamplingParams(
            height=1024,
            width=1024,
            num_inference_steps=1,
        ),
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
    )

    preprocess(profile_req)

    assert isinstance(profile_req.prepared_layout, QwenImage21PreparedLayout)
    assert len(profile_req.prepared_layout.input_image_sizes) == 4
    assert profile_req.prepared_layout.num_branches == 2
    assert profile_req.diffusion_kv_requests is not None
    assert len(profile_req.diffusion_kv_requests) == 2
    assert profile_req.diffusion_kv_requests[0].prefix_len > 4 * 4096
    assert profile_req.diffusion_kv_requests[0].prefix_len == profile_req.diffusion_kv_requests[1].prefix_len
    assert profile_req.sampling_params.max_sequence_length == 8192
    assert profile_req.prompt["negative_prompt"] == profile_req.prompt["prompt"]
    profile_tokenizer = _layout_tokenizer()
    assert profile_tokenizer._budgeted_prompt_length(profile_req.prompt["prompt"], 4) == 8192

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine._diffusion_kv_profile_limits = (
        len(profile_req.diffusion_kv_requests),
        max(row.seq_len for row in profile_req.diffusion_kv_requests),
        max(row.target_len for row in profile_req.diffusion_kv_requests),
    )
    engine._validate_diffusion_kv_profile_limits(profile_req)

    within = _make_request(prompt="wide image", num_images=0, height=768, width=1280)
    preprocess(within)
    engine._validate_diffusion_kv_profile_limits(within)

    over_target = _make_request(prompt="slightly over one megapixel", num_images=0, height=864, width=1216)
    preprocess(over_target)
    with pytest.raises(ValueError, match=r"max_target_len=4104 \(profiled=4096\)"):
        engine._validate_diffusion_kv_profile_limits(over_target)


def test_paged_preprocess_validates_profile_text_budget(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(layout_module, "QwenImage21LayoutTokenizer", lambda *_args, **_kwargs: _layout_tokenizer())

    with pytest.raises(ValueError, match="diffusion_kv_profile_max_sequence_length"):
        get_qwen_image_21_pre_process_func(
            SimpleNamespace(
                model="model",
                diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
                additional_config={"diffusion_kv_profile_max_sequence_length": 8193},
            )
        )

    preprocess = get_qwen_image_21_pre_process_func(
        SimpleNamespace(
            model="model",
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            diffusion_kv_max_rows_per_request=2,
            additional_config={"diffusion_kv_profile_max_sequence_length": 512},
        )
    )
    boundary_prompt = _layout_tokenizer().build_profile_prompt(num_images=0, max_sequence_length=512)
    boundary = _make_request(
        prompt=boundary_prompt,
        negative_prompt=boundary_prompt,
        true_cfg_scale=4.0,
        num_images=0,
    )
    preprocess(boundary)
    assert all(
        branch.prefix_len == boundary.prepared_layout.branches[0].prefix_len
        for branch in boundary.prepared_layout.branches
    )

    too_long = _make_request(prompt=boundary_prompt + "x", num_images=0)
    with pytest.raises(ValueError, match=r"`prompt`.*max_sequence_length` is 512"):
        preprocess(too_long)

    negative_too_long = _make_request(
        prompt="short",
        negative_prompt=boundary_prompt + "x",
        true_cfg_scale=4.0,
        num_images=0,
    )
    with pytest.raises(ValueError, match=r"`negative_prompt`.*max_sequence_length` is 512"):
        preprocess(negative_too_long)


@pytest.mark.parametrize(
    ("causal_condition", "causal_block"),
    [(False, True), (True, False), (False, False)],
)
def test_paged_transformer_requires_stable_causal_prefix(causal_condition: bool, causal_block: bool) -> None:
    od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        parallel_config=SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="causal_condition=True and causal_block=True"):
        QwenImage21Transformer2DModel(
            od_config=od_config,
            causal_condition=causal_condition,
            causal_block=causal_block,
        )


def test_paged_prompt_mask_distinguishes_prompt_and_sp_padding() -> None:
    sp_only = torch.tensor([[True, True, True, True, False, False, False, False]])
    _validate_paged_prompt_mask(sp_only, sp_padding=4)

    actual_prompt_padding = torch.tensor([[True, True, False, True, False, False, False, False]])
    with pytest.raises(ValueError, match="padded prompt masks"):
        _validate_paged_prompt_mask(actual_prompt_padding, sp_padding=4)

    with pytest.raises(ValueError, match="SP padding exceeds"):
        _validate_paged_prompt_mask(torch.ones(1, 3, dtype=torch.bool), sp_padding=4)


def test_system_prefix_dropped_and_images_expanded_4x() -> None:
    tok = _layout_tokenizer()
    expected_drop = len(f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n")
    assert tok.drop_idx == expected_drop
    assert tok.baseline_len == len(PROMPT_TEMPLATE_T2I.format(""))

    req = _make_request(prompt="hello", num_images=2, height=64, width=64)
    layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)

    assert layout.num_branches == 1
    branch = layout.branches[0]
    # Each 64x64 image has (64//16)*(64//16) = 16 latent tokens (4 VLM slots).
    assert len(branch.image_spans) == 2
    assert [span.stop - span.start for span, _ in branch.image_spans] == [16, 16]
    assert sum(branch.vlm_image_pad_mask) == 8


@pytest.fixture
def prefix_cache_manager(request):
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
    num_blocks = 256
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[build_kv_cache_tensor(spec, num_blocks, ["layer0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )
    manager = DiffusionKVCacheManager(
        config,
        max_model_len=512,
        scheduler_block_size=4,
        hash_block_size=1,
        enable_prefix_caching=getattr(request, "param", True),
    )
    yield manager
    manager.close()


@pytest.mark.parametrize("prefix_cache_manager", [False], indirect=True)
@pytest.mark.parametrize("true_cfg_scale", [1.0, 4.0])
def test_paged_cache_off_prepares_and_allocates_without_identity_work(
    monkeypatch, prefix_cache_manager, true_cfg_scale
) -> None:
    tok = _layout_tokenizer()
    config = SimpleNamespace(
        model="fake",
        model_class_name="QwenImage21Pipeline",
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=False,
    )
    monkeypatch.setattr(pipeline_module, "_read_vae_scale_factor", lambda _cfg: 16)
    monkeypatch.setattr(layout_module, "QwenImage21LayoutTokenizer", lambda *_args, **_kwargs: tok)
    monkeypatch.setattr(engine_module, "get_diffusion_post_process_func", lambda _config: None)
    fail = Mock(side_effect=AssertionError("disabled prefix cache must not prepare identity"))
    monkeypatch.setattr(engine_module, "get_diffusion_prefix_cache_func", fail)
    monkeypatch.setattr(layout_module, "prepare_qwen_image_21_prefix_cache", fail)
    monkeypatch.setattr(layout_module, "hash_prefix_cache_value", fail)
    monkeypatch.setattr(kv_utils, "hash_prefix_cache_value", fail)

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = config
    engine._init_process_hooks(config)
    engine.prefix_cache_func = fail

    req = _make_request(
        prompt="edit",
        negative_prompt="blurry" if true_cfg_scale > 1.0 else None,
        true_cfg_scale=true_cfg_scale,
    )
    engine._prepare_request_for_admission(req)
    assert isinstance(req.prepared_layout, QwenImage21PreparedLayout)
    assert all(
        row.cache_token_ids == () and row.mm_features == [] and row.cache_dependencies == ()
        for row in req.diffusion_kv_requests
    )
    metadata = prefix_cache_manager.reserve_request(req.request_id, req.diffusion_kv_requests)
    assert metadata is not None
    assert all(row.cached_prefix_len == 0 for row in metadata.sequences)
    prefix_cache_manager.publish_request(req.request_id)
    prefix_cache_manager.free_request(req.request_id)
    fail.assert_not_called()


def test_reference_content_is_hashed_once_across_cfg_rows(monkeypatch) -> None:
    tok = _layout_tokenizer()
    hasher = Mock(wraps=layout_module.hash_prefix_cache_value)
    monkeypatch.setattr(layout_module, "hash_prefix_cache_value", hasher)

    req = _make_request(
        prompt="restyle card",
        negative_prompt="blurry",
        true_cfg_scale=4.0,
        num_images=2,
    )
    layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)
    req.prepared_layout = layout
    req.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(req, layout)
    prepare_qwen_image_21_prefix_cache(req)

    # 2 reference images hashed once total across both CFG branches, while
    # positive and negative branches use isolated cache namespaces.
    assert sum(call.args[0][0] == "qwen-image-21-reference-v1" for call in hasher.call_args_list) == 2
    assert len(req.diffusion_kv_requests[0].cache_dependencies) == 2
    assert req.diffusion_kv_requests[0].cache_dependencies == req.diffusion_kv_requests[1].cache_dependencies
    assert req.diffusion_kv_requests[0].cache_namespace != req.diffusion_kv_requests[1].cache_namespace
    layout_digest = layout_module.hash_prefix_cache_value(
        ("qwen-image-21-reference-layout-v1", layout.input_image_sizes)
    ).hex()
    assert req.diffusion_kv_requests[0].cache_namespace == kv_utils.get_cache_namespace(
        f"qwen-image-21-primary-v1-cond-{layout_digest}", req.sampling_params
    )
    assert req.diffusion_kv_requests[1].cache_namespace == kv_utils.get_cache_namespace(
        f"qwen-image-21-primary-v1-uncond-{layout_digest}", req.sampling_params
    )


@pytest.mark.parametrize("changed_num_images", [1, 4])
def test_reference_image_count_change_isolates_prefix_namespace(
    prefix_cache_manager: DiffusionKVCacheManager,
    changed_num_images: int,
) -> None:
    tok = _layout_tokenizer()

    def build(request_id: str, num_images: int) -> OmniDiffusionRequest:
        req = _make_request(
            prompt="edit the references",
            negative_prompt="blurry",
            true_cfg_scale=4.0,
            num_images=num_images,
            request_id=request_id,
        )
        layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)
        req.prepared_layout = layout
        req.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(req, layout)
        prepare_qwen_image_21_prefix_cache(req)
        return req

    cold = build("cold-two", 2)
    assert prefix_cache_manager.reserve_request("cold-two", cold.diffusion_kv_requests) is not None
    prefix_cache_manager.publish_request("cold-two")
    prefix_cache_manager.free_request("cold-two")

    changed = build(f"changed-{changed_num_images}", changed_num_images)
    metadata = prefix_cache_manager.reserve_request(changed.request_id, changed.diffusion_kv_requests)
    assert metadata is not None
    assert all(sequence.cached_prefix_len == 0 for sequence in metadata.sequences)
    prefix_cache_manager.free_request(changed.request_id)


def test_prefix_lookup_hits_full_partial_image_and_isolates_lora(
    prefix_cache_manager: DiffusionKVCacheManager,
) -> None:
    tok = _layout_tokenizer()

    def build(
        request_id: str,
        *,
        prompt: str = "replace the logo with AnchorCache",
        seed: int = 42,
        num_images: int = 2,
        changed_image_idx: int | None = None,
        input_image_sizes: list[tuple[int, int]] | None = None,
        lora_id: int | None = None,
        lora_scale: float = 1.0,
    ):
        req = _make_request(
            prompt=prompt,
            negative_prompt="blurry",
            true_cfg_scale=4.0,
            num_images=num_images,
            request_id=request_id,
            seed=seed,
            input_image_sizes=input_image_sizes,
        )
        if changed_image_idx is not None:
            req.prompt["additional_information"]["vae_images"][changed_image_idx].fill_(9.0)
        if lora_id is not None:
            req.sampling_params.lora_request = LoRARequest(
                lora_name=f"adapter-{lora_id}",
                lora_int_id=lora_id,
                lora_path=f"/adapters/{lora_id}",
            )
            req.sampling_params.lora_scale = lora_scale
        layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)
        req.prepared_layout = layout
        req.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(req, layout)
        prepare_qwen_image_21_prefix_cache(req)
        return req

    cold_req = build("cold")
    cold_meta = prefix_cache_manager.reserve_request("cold", cold_req.diffusion_kv_requests)
    assert cold_meta is not None
    assert all(seq.cached_prefix_len == 0 for seq in cold_meta.sequences)
    prefix_cache_manager.publish_request("cold")
    prefix_cache_manager.free_request("cold")

    # 1. Different seed with identical images and prompt still hits full common prefix.
    same_seed_diff = build("same-diff-seed", seed=999)
    meta_same = prefix_cache_manager.reserve_request("same-diff-seed", same_seed_diff.diffusion_kv_requests)
    assert meta_same is not None
    image_prefix_end = cold_req.prepared_layout.branches[0].image_prefix_end
    assert meta_same.sequences[0].cached_prefix_len >= image_prefix_end
    assert native_kv_covers_cond_images(same_seed_diff.prepared_layout, meta_same.sequences[0].cached_prefix_len)
    prefix_cache_manager.free_request("same-diff-seed")

    # 2. Prompt text follows the condition-image spans and must not invalidate
    # their KV. The unchanged image prefix remains reusable across edits.
    edited = build("edited-prompt", prompt="turn the background dark blue")
    meta_edited = prefix_cache_manager.reserve_request("edited-prompt", edited.diffusion_kv_requests)
    assert meta_edited is not None
    assert all(sequence.cached_prefix_len >= image_prefix_end for sequence in meta_edited.sequences)
    assert native_kv_covers_cond_images(edited.prepared_layout, meta_edited.sequences[0].cached_prefix_len)
    prefix_cache_manager.free_request("edited-prompt")

    # 3. Changing only the second image reuses through the exact start of that
    # image even though the boundary lies midway through a physical KV page.
    img2_changed = build("img2-changed", changed_image_idx=1)
    meta_img2 = prefix_cache_manager.reserve_request("img2-changed", img2_changed.diffusion_kv_requests)
    assert meta_img2 is not None
    second_img_start = cold_req.prepared_layout.branches[0].image_spans[1][0].start
    assert all(sequence.cached_prefix_len == second_img_start for sequence in meta_img2.sequences)
    assert len(meta_img2.block_copies) == (0 if second_img_start % 4 == 0 else 2)
    assert not native_kv_covers_cond_images(img2_changed.prepared_layout, meta_img2.sequences[0].cached_prefix_len)
    prefix_cache_manager.complete_block_copies("img2-changed")
    prefix_cache_manager.free_request("img2-changed")

    # 4. A different total image layout is isolated because Qwen's upstream
    # condition representation is not invariant to image count or shape.
    changed_layout = build(
        "changed-layout",
        input_image_sizes=[(32, 64), (96, 64)],
    )
    meta_changed_layout = prefix_cache_manager.reserve_request("changed-layout", changed_layout.diffusion_kv_requests)
    assert meta_changed_layout is not None
    assert all(sequence.cached_prefix_len == 0 for sequence in meta_changed_layout.sequences)
    prefix_cache_manager.free_request("changed-layout")

    # 5. Within one arbitrary layout, changing the final image reuses every
    # preceding image independently of the concrete image shapes.
    arbitrary_sizes = [(32, 64), (96, 64), (64, 32)]
    cold_three = build("cold-three", num_images=3, input_image_sizes=arbitrary_sizes)
    meta_cold_three = prefix_cache_manager.reserve_request("cold-three", cold_three.diffusion_kv_requests)
    assert meta_cold_three is not None
    prefix_cache_manager.publish_request("cold-three")
    prefix_cache_manager.complete_block_copies("cold-three")
    prefix_cache_manager.free_request("cold-three")

    img3_changed = build(
        "img3-changed",
        num_images=3,
        changed_image_idx=2,
        input_image_sizes=arbitrary_sizes,
    )
    meta_img3 = prefix_cache_manager.reserve_request("img3-changed", img3_changed.diffusion_kv_requests)
    assert meta_img3 is not None
    third_img_start = cold_three.prepared_layout.branches[0].image_spans[2][0].start
    assert all(sequence.cached_prefix_len == third_img_start for sequence in meta_img3.sequences)
    assert len(meta_img3.block_copies) == (0 if third_img_start % 4 == 0 else 2)
    prefix_cache_manager.complete_block_copies("img3-changed")
    prefix_cache_manager.free_request("img3-changed")

    # 6. Different LoRA adapter misses completely.
    lora_changed = build("lora-changed", lora_id=1, lora_scale=0.8)
    meta_lora = prefix_cache_manager.reserve_request("lora-changed", lora_changed.diffusion_kv_requests)
    assert meta_lora is not None
    assert all(seq.cached_prefix_len == 0 for seq in meta_lora.sequences)
    prefix_cache_manager.publish_request("lora-changed")
    prefix_cache_manager.free_request("lora-changed")

    # 7. The same adapter and scale can reuse its own prefix, while changing
    # only the scale creates a distinct cache namespace.
    same_lora = build("same-lora", lora_id=1, lora_scale=0.8)
    meta_same_lora = prefix_cache_manager.reserve_request("same-lora", same_lora.diffusion_kv_requests)
    assert meta_same_lora is not None
    assert all(seq.cached_prefix_len >= image_prefix_end for seq in meta_same_lora.sequences)
    prefix_cache_manager.free_request("same-lora")

    lora_scale_changed = build("lora-scale-changed", lora_id=1, lora_scale=0.9)
    meta_lora_scale = prefix_cache_manager.reserve_request(
        "lora-scale-changed", lora_scale_changed.diffusion_kv_requests
    )
    assert meta_lora_scale is not None
    assert all(seq.cached_prefix_len == 0 for seq in meta_lora_scale.sequences)
    prefix_cache_manager.free_request("lora-scale-changed")

    # 8. Swapping positive and negative prompts in T2I does not cross-hit across
    # cond/uncond branches (preventing uninitialized cross-rank reads under CFGP2).
    t2i_a = _make_request(
        prompt="bright studio portrait",
        negative_prompt="dark blurry grain",
        true_cfg_scale=4.0,
        num_images=0,
        request_id="t2i-a",
    )
    t2i_a.prepared_layout = prepare_qwen_image_21_layout(t2i_a, layout_tokenizer=tok, vae_scale_factor=16)
    t2i_a.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(t2i_a, t2i_a.prepared_layout)
    prepare_qwen_image_21_prefix_cache(t2i_a)
    meta_t2i_a = prefix_cache_manager.reserve_request("t2i-a", t2i_a.diffusion_kv_requests)
    assert meta_t2i_a is not None
    prefix_cache_manager.publish_request("t2i-a")
    prefix_cache_manager.free_request("t2i-a")

    t2i_swapped = _make_request(
        prompt="dark blurry grain",
        negative_prompt="bright studio portrait",
        true_cfg_scale=4.0,
        num_images=0,
        request_id="t2i-swapped",
    )
    t2i_swapped.prepared_layout = prepare_qwen_image_21_layout(t2i_swapped, layout_tokenizer=tok, vae_scale_factor=16)
    t2i_swapped.diffusion_kv_requests = build_qwen_image_21_diffusion_kv_requests(
        t2i_swapped, t2i_swapped.prepared_layout
    )
    prepare_qwen_image_21_prefix_cache(t2i_swapped)
    meta_t2i_swapped = prefix_cache_manager.reserve_request("t2i-swapped", t2i_swapped.diffusion_kv_requests)
    assert meta_t2i_swapped is not None
    # Only the shared user header ("<|im_start|>user\n", 17 chars -> 4 blocks = 16 tokens)
    # may match within each branch's own namespace; the swapped prompt body must not cross-hit.
    assert all(seq.cached_prefix_len < seq.prefix_len for seq in meta_t2i_swapped.sequences)
    prefix_cache_manager.free_request("t2i-swapped")


def test_sequential_cfg_prepares_single_branch_batches_without_two_row_prefill() -> None:
    from contextlib import contextmanager

    from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
        DiffusionPagedAttentionMetadata,
        DiffusionPagedAttentionRow,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import (
        QwenImage21CFGParallelMixin,
        _set_denoise_step_idx,
    )

    row0 = DiffusionPagedAttentionRow(request_id="r", sequence_id=0, kv_start_pos=0, query_len=20518, seq_len=20518)
    row1 = DiffusionPagedAttentionRow(request_id="r", sequence_id=1, kv_start_pos=0, query_len=20517, seq_len=20517)
    metadata = DiffusionPagedAttentionMetadata(prefill_rows=(row0, row1), denoise_rows=(row0, row1))
    prepared_batches: list[tuple[int, ...]] = []

    class _FakeAdapter:
        def prepare_batch(self, rows):
            num_tokens = sum(r.query_len for r in rows)
            if num_tokens > 32768:
                raise ValueError(f"Paged attention batch has {num_tokens} tokens; Worker capacity is 32768")
            seq_ids = tuple(r.sequence_id for r in rows)
            prepared_batches.append(seq_ids)
            return seq_ids

        @contextmanager
        def activate(self, batch):
            yield batch

    class _FakeRuntime:
        def __init__(self) -> None:
            self.metadata = metadata
            self.adapter = _FakeAdapter()

        def ensure_active(self, step_idx):
            rows = self.metadata.rows_for_step(step_idx)
            return self.adapter.prepare_batch(rows)

        def _deactivate(self) -> None:
            return None

    class _FakePipe(QwenImage21CFGParallelMixin):
        def predict_noise(self, **kwargs):
            return kwargs["out"]

        def combine_cfg_noise(self, pos, neg, scale, normalize):
            del scale, normalize
            return pos[0] + neg[0]

    pipe = _FakePipe()
    runtime = _FakeRuntime()
    with set_forward_context(vllm_config=None, omni_diffusion_config=None, paged_kv_runtime=runtime):
        _set_denoise_step_idx(0, sequential_cfg=True)
        out = pipe.predict_noise_maybe_with_cfg(
            do_true_cfg=True,
            true_cfg_scale=4.0,
            positive_kwargs={"out": torch.ones(1, 4, 2)},
            negative_kwargs={"out": torch.ones(1, 4, 2)},
            output_slice=2,
        )
    assert out.shape == (1, 2, 2)
    assert prepared_batches == [(0,), (1,)]


def test_prepare_generation_context_skips_vae_and_prompt_encode_on_prefix_hit() -> None:
    tok = _layout_tokenizer()
    req = _make_request(prompt="keep card", num_images=2, height=64, width=64)
    layout = prepare_qwen_image_21_layout(req, layout_tokenizer=tok, vae_scale_factor=16)

    pipe = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.vae_scale_factor = 16
    pipe.latent_channels = 4
    pipe._max_length = 8192
    pipe.device = torch.device("cpu")
    pipe.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        parallel_config=SimpleNamespace(cfg_parallel_size=1),
    )
    pipe.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipe.transformer = SimpleNamespace(in_channels=4)
    pipe.encode_prompt = Mock(
        return_value=(
            torch.zeros(1, 20, 8),
            torch.ones(1, 20, dtype=torch.long),
            torch.tensor([layout.branches[0].vlm_image_pad_mask], dtype=torch.bool),
        )
    )
    pipe.prepare_latents = Mock(return_value=(torch.zeros(1, 16, 4), None))
    pipe.prepare_timesteps = Mock(return_value=(torch.tensor([1000.0, 500.0]), 2))
    per_request_images = pipe._extract_request_images([req.prompt])

    # Partial hit covering all condition images: skips condition VAE, runs encode_prompt.
    image_end = layout.branches[0].image_prefix_end
    pipe._prepare_generation_context(
        prompt=["keep card"],
        negative_prompt=None,
        height=64,
        width=64,
        num_inference_steps=2,
        sigmas=None,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(0),
        true_cfg_scale=1.0,
        max_sequence_length=8192,
        per_request_images=per_request_images,
        prepared_layout=layout,
        paged_kv_cached_prefix_len=image_end,
    )
    assert pipe.encode_prompt.call_count == 1
    assert pipe.prepare_latents.call_args.args[0] is None

    # Full prefix hit: skips both encode_prompt and condition VAE.
    pipe.encode_prompt.reset_mock()
    pipe.prepare_latents.reset_mock()
    ctx = pipe._prepare_generation_context(
        prompt=["keep card"],
        negative_prompt=None,
        height=64,
        width=64,
        num_inference_steps=2,
        sigmas=None,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(0),
        true_cfg_scale=1.0,
        max_sequence_length=8192,
        per_request_images=per_request_images,
        prepared_layout=layout,
        paged_kv_cached_prefix_len=layout.branches[0].prefix_len,
    )
    assert pipe.encode_prompt.call_count == 0
    assert pipe.prepare_latents.call_args.args[0] is None
    assert ctx["prompt_embeds"] is None


def test_paged_kv_fp32_matmul_reduction_restores_backend_state() -> None:
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        _paged_kv_fp32_matmul_reduction,
    )

    matmul = torch.backends.cuda.matmul
    has_split_k = hasattr(matmul, "allow_bf16_reduced_precision_reduction_split_k")
    orig_state: bool | tuple[bool, bool] = (
        (
            bool(matmul.allow_bf16_reduced_precision_reduction),
            bool(matmul.allow_bf16_reduced_precision_reduction_split_k),
        )
        if has_split_k
        else bool(matmul.allow_bf16_reduced_precision_reduction)
    )
    try:
        test_states: list[bool | tuple[bool, bool]] = (
            [(True, True), (False, True), (False, False)] if has_split_k else [True, False]
        )
        for state in test_states:
            matmul.allow_bf16_reduced_precision_reduction = state
            with _paged_kv_fp32_matmul_reduction(False):
                assert bool(matmul.allow_bf16_reduced_precision_reduction) is bool(
                    state[0] if isinstance(state, tuple) else state
                )
            with _paged_kv_fp32_matmul_reduction(True):
                assert bool(matmul.allow_bf16_reduced_precision_reduction) is False
                if has_split_k and isinstance(state, tuple):
                    assert bool(matmul.allow_bf16_reduced_precision_reduction_split_k) is state[1]
                with _paged_kv_fp32_matmul_reduction(True):
                    assert bool(matmul.allow_bf16_reduced_precision_reduction) is False
            restored = (
                (
                    bool(matmul.allow_bf16_reduced_precision_reduction),
                    bool(matmul.allow_bf16_reduced_precision_reduction_split_k),
                )
                if has_split_k
                else bool(matmul.allow_bf16_reduced_precision_reduction)
            )
            assert restored == state

            with pytest.raises(RuntimeError, match="boom"):
                with _paged_kv_fp32_matmul_reduction(True):
                    raise RuntimeError("boom")
            restored_after_exc = (
                (
                    bool(matmul.allow_bf16_reduced_precision_reduction),
                    bool(matmul.allow_bf16_reduced_precision_reduction_split_k),
                )
                if has_split_k
                else bool(matmul.allow_bf16_reduced_precision_reduction)
            )
            assert restored_after_exc == state
    finally:
        matmul.allow_bf16_reduced_precision_reduction = orig_state

    # Verify compatibility with older PyTorch versions where split_k attribute
    # does not exist and the setter only accepts a bool.
    class _LegacyMatmul:
        def __init__(self, initial: bool) -> None:
            self._val = initial

        @property
        def allow_bf16_reduced_precision_reduction(self) -> bool:
            return self._val

        @allow_bf16_reduced_precision_reduction.setter
        def allow_bf16_reduced_precision_reduction(self, value: bool) -> None:
            if not isinstance(value, bool):
                raise RuntimeError(f"set_allow_bf16_reduction_cublas expects a bool, but got {type(value).__name__}")
            self._val = value

    real_matmul = torch.backends.cuda.matmul
    try:
        for legacy_init in (True, False):
            legacy = _LegacyMatmul(legacy_init)
            torch.backends.cuda.matmul = legacy  # type: ignore[assignment]
            with _paged_kv_fp32_matmul_reduction(True):
                assert legacy.allow_bf16_reduced_precision_reduction is False
            assert legacy.allow_bf16_reduced_precision_reduction is legacy_init
    finally:
        torch.backends.cuda.matmul = real_matmul
