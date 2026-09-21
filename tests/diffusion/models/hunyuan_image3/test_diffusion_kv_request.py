# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import vllm.v1.core.single_type_kv_cache_manager as native_kv_managers
from transformers.utils.generic import ModelOutput
from vllm.lora.request import LoRARequest
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

import vllm_omni.diffusion.diffusion_engine as engine_module
import vllm_omni.diffusion.diffusion_kv.kv_cache_utils as kv_utils
import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as pipeline_module
import vllm_omni.diffusion.models.hunyuan_image3.request_layout as layout_module
from tests.helpers.kv_layout import build_kv_cache_tensor
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerEncodeOutput
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    ImageInfo,
    JointImageInfo,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
    get_hunyuan_image_3_pre_process_func,
)
from vllm_omni.diffusion.models.hunyuan_image3.request_layout import (
    HunyuanPreparedLayout,
    build_hunyuan_diffusion_kv_requests,
    extract_hunyuan_prompt_inputs,
    hunyuan_num_image_tokens,
    hunyuan_num_special_tokens,
    normalize_hunyuan_cot_text,
    prepare_hunyuan_layout,
    prepare_hunyuan_prefix_cache,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dense_legacy_preprocess_does_not_initialize_layout_tokenizer(monkeypatch) -> None:
    hf_config = SimpleNamespace(vae_downsample_factor=(8, 8), patch_size=2)
    image_processor = SimpleNamespace(vision_encoder_processor=SimpleNamespace(patch_size=1))
    monkeypatch.setattr(pipeline_module, "get_config", lambda *_args, **_kwargs: hf_config)
    monkeypatch.setattr(pipeline_module, "HunyuanImage3ImageProcessor", lambda _config: image_processor)
    monkeypatch.setattr(
        pipeline_module,
        "TokenizerWrapper",
        lambda _model: pytest.fail("dense_legacy must not initialize the Diffusion KV tokenizer"),
    )

    get_hunyuan_image_3_pre_process_func(
        SimpleNamespace(model="model", diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY)
    )


class _FakeTokenizerWrapper:
    def __init__(self, *, prefix_lens: list[int]) -> None:
        self.prefix_lens = prefix_lens
        self.calls: list[dict] = []
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.boi_token_id = 3
        self.end_recaption_token_id = 4
        self.end_answer_token_id = 5
        self.special_token_map = {f"<img_ratio_{index}>": 100 + index for index in range(33)}

    def apply_chat_template(self, **kwargs):
        self.calls.append(kwargs)
        rows = len(self.prefix_lens)
        has_cond_image = kwargs.get("batch_cond_image_info") is not None
        seq_lens = [prefix_len + 20 for prefix_len in self.prefix_lens]
        padded_seq_len = max(seq_lens)
        positions = torch.arange(padded_seq_len)
        # Like the real tokenizer, joint spans enclose VAE + separator + ViT,
        # while all_image_slices contains the individual RoPE spans. Leave one
        # full text block before the image so partial reuse can be tested.
        joint_image_slices = [[slice(5, 10)] if has_cond_image else [] for _ in range(rows)]
        cond_vae_image_slices = [[slice(5, 7)] if has_cond_image else [] for _ in range(rows)]
        cond_vit_image_slices = [[slice(8, 10)] if has_cond_image else [] for _ in range(rows)]
        gen_image_slices = [[slice(length + 1, length + 17)] for length in self.prefix_lens]
        all_image_slices = [
            vae + vit + generated
            for vae, vit, generated in zip(cond_vae_image_slices, cond_vit_image_slices, gen_image_slices)
        ]
        cond_vae_image_mask = (
            torch.stack([(positions >= 5) & (positions < 7) for _ in range(rows)]) if has_cond_image else None
        )
        cond_vit_image_mask = (
            torch.stack([(positions >= 8) & (positions < 10) for _ in range(rows)]) if has_cond_image else None
        )
        sections = []
        for _ in range(rows):
            row_sections = []
            if has_cond_image:
                row_sections.append(
                    {
                        "type": "joint_image",
                        "token_height": [1, 1],
                        "token_width": [2, 2],
                    }
                )
            row_sections.append({"type": "gen_image", "token_height": 4, "token_width": 4})
            sections.append(row_sections)
        return {
            "output": TokenizerEncodeOutput(
                tokens=torch.arange(rows * padded_seq_len, dtype=torch.long).reshape(rows, padded_seq_len),
                gen_timestep_scatter_index=torch.tensor(self.prefix_lens, dtype=torch.long).reshape(rows, 1),
                cond_timestep_scatter_index=(torch.zeros(rows, 1, dtype=torch.long) if has_cond_image else None),
                all_image_slices=all_image_slices,
                gen_image_mask=torch.stack(
                    [(positions >= length + 1) & (positions < length + 17) for length in self.prefix_lens]
                ),
                cond_vae_image_mask=cond_vae_image_mask,
                cond_vit_image_mask=cond_vit_image_mask,
                cond_vae_image_slices=cond_vae_image_slices,
                cond_vit_image_slices=cond_vit_image_slices,
                joint_image_slices=joint_image_slices,
                gen_image_slices=gen_image_slices,
                real_pos=torch.tensor(seq_lens, dtype=torch.long).reshape(rows, 1),
            ),
            "sections": sections,
        }


class _FakeImageProcessor:
    def __init__(self, image_token_length: int = 16) -> None:
        self.image_token_length = image_token_length
        self.image_sizes: list[tuple[int, int]] = []
        self.vision_encoder_processor = SimpleNamespace(patch_size=1)

    def build_image_info(self, image_size, **kwargs):
        self.image_sizes.append(image_size)
        return ImageInfo(
            image_type="gen_image",
            image_width=image_size[1],
            image_height=image_size[0],
            token_width=4,
            token_height=4,
            image_token_length=self.image_token_length,
            base_size=1024,
            ratio_index=0,
            **kwargs,
        )


def _components(prefix_lens: list[int]):
    return _FakeTokenizerWrapper(prefix_lens=prefix_lens), _FakeImageProcessor()


def _prepare(
    request: OmniDiffusionRequest,
    tokenizer: _FakeTokenizerWrapper,
    image_processor: _FakeImageProcessor,
) -> HunyuanPreparedLayout:
    prepared_layout = prepare_hunyuan_layout(
        request,
        tokenizer_wrapper=tokenizer,
        image_processor=image_processor,
        generation_config=SimpleNamespace(sequence_template="instruct", drop_think=False),
        image_base_size=1024,
    )
    request.prepared_layout = prepared_layout
    return prepared_layout


def _request(
    *,
    guidance_scale: float,
    prompt="draw a cat",
    request_id: str = "req",
    seed: int | None = None,
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            height=768,
            width=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=4,
            seed=seed,
        ),
        request_id=request_id,
    )


def test_distilled_layout_uses_embedded_cfg_and_meanflow_tokens() -> None:
    tokenizer, image_processor = _components([12])
    request = _request(guidance_scale=2.5)

    prepared_layout = prepare_hunyuan_layout(
        request,
        tokenizer_wrapper=tokenizer,
        image_processor=image_processor,
        generation_config=SimpleNamespace(sequence_template="instruct", drop_think=False),
        image_base_size=1024,
        cfg_distilled=True,
        use_meanflow=True,
    )

    image_info = prepared_layout.generated_image_info
    assert image_info.add_guidance_token
    assert image_info.add_timestep_r_token
    assert tokenizer.calls[0]["cfg_factor"] == 1
    assert hunyuan_num_special_tokens(image_info) == 3
    assert hunyuan_num_image_tokens(image_info) == 19
    assert len(build_hunyuan_diffusion_kv_requests(request, prepared_layout)) == 1


def test_builds_kv_request_lengths_without_model_execution() -> None:
    tokenizer, image_processor = _components([12])
    request = _request(guidance_scale=1.0)
    prepared_layout = _prepare(request, tokenizer, image_processor)

    kv_requests = build_hunyuan_diffusion_kv_requests(request, prepared_layout)

    assert isinstance(request.prepared_layout, HunyuanPreparedLayout)
    assert all(isinstance(item, DiffusionKVRequest) for item in kv_requests)
    assert [(item.prefix_len, item.target_len, item.seq_len) for item in kv_requests] == [(12, 17, 32)]
    assert image_processor.image_sizes == [(768, 1024)]
    assert tokenizer.calls[0]["sequence_template"] == "instruct"
    assert tokenizer.calls[0]["cfg_factor"] == 1
    assert kv_requests[0].block_hashes == []
    assert kv_requests[0].cache_token_ids == ()
    assert kv_requests[0].mm_features == []
    assert kv_requests[0].kv_contexts == ()
    assert kv_requests[0].skip_reading_prefix_cache is True


def test_builds_one_kv_request_per_cfg_row() -> None:
    tokenizer, image_processor = _components([12, 14])
    request = _request(guidance_scale=5.0)
    prepared_layout = _prepare(request, tokenizer, image_processor)

    kv_requests = build_hunyuan_diffusion_kv_requests(request, prepared_layout)

    assert [item.sequence_id for item in kv_requests] == [0, 1]
    assert [item.prefix_len for item in kv_requests] == [12, 14]
    assert [item.seq_len for item in kv_requests] == [32, 34]
    assert tokenizer.calls[0]["cfg_factor"] == 2


def _reference_image() -> JointImageInfo:
    return JointImageInfo(
        vae_image_info=ImageInfo(
            image_type="vae",
            image_tensor=torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2),
            token_width=8,
            token_height=8,
            image_token_length=64,
        ),
        vision_image_info=ImageInfo(
            image_type="siglip2",
            image_tensor=torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2),
            token_width=4,
            token_height=4,
            image_token_length=16,
        ),
        vision_encoder_kwargs={
            "spatial_shapes": torch.tensor([4, 4]),
            "pixel_attention_mask": torch.ones(4, dtype=torch.bool),
        },
    )


def _prepare_cached_requests(request, tokenizer, image_processor):
    layout = _prepare(request, tokenizer, image_processor)
    request.diffusion_kv_requests = build_hunyuan_diffusion_kv_requests(request, layout)
    prepare_hunyuan_prefix_cache(request)
    return request.diffusion_kv_requests


def test_reference_prefix_identity_includes_image_content_and_vae_seed() -> None:
    tokenizer, image_processor = _components([20])

    def build(request_id: str, image: JointImageInfo, seed: int):
        request = _request(
            guidance_scale=1.0,
            request_id=request_id,
            seed=seed,
            prompt={
                "prompt": "edit this image",
                "additional_information": {"batch_cond_image_info": [image]},
            },
        )
        return _prepare_cached_requests(request, tokenizer, image_processor)[0]

    first = build("first", _reference_image(), 7)
    same = build("same", _reference_image(), 7)
    changed_seed = build("changed-seed", _reference_image(), 8)
    changed_image_value = _reference_image()
    changed_image_value.vae_image_info.image_tensor[0, 0, 0, 0] = 99
    changed_image = build("changed-image", changed_image_value, 7)

    assert first.mm_features == same.mm_features
    assert [(feature.mm_position.offset, feature.mm_position.length) for feature in first.mm_features] == [
        (5, 2),
        (8, 2),
    ]
    for changed in (changed_seed, changed_image):
        # Both subspans must carry the identity, not just the first one.
        assert all(a.identifier != b.identifier for a, b in zip(first.mm_features, changed.mm_features, strict=True))


def test_reference_prefix_rejects_missing_inputs_with_nested_spans() -> None:
    tokenizer, image_processor = _components([20])
    request = _request(
        guidance_scale=1.0,
        prompt={"prompt": "edit this image", "additional_information": {"batch_cond_image_info": [_reference_image()]}},
    )
    layout = _prepare(request, tokenizer, image_processor)
    request.diffusion_kv_requests = build_hunyuan_diffusion_kv_requests(request, layout)
    request.prompt["additional_information"].pop("batch_cond_image_info")

    with pytest.raises(ValueError, match="missing its canonical inputs"):
        prepare_hunyuan_prefix_cache(request)


def test_reference_content_is_hashed_once_across_cfg_rows(monkeypatch):
    tokenizer, image_processor = _components([20, 20])
    image = _reference_image()
    hasher = Mock(wraps=layout_module.hash_prefix_cache_value)
    monkeypatch.setattr(layout_module, "hash_prefix_cache_value", hasher)

    def build(request_id):
        request = _request(
            guidance_scale=5.0,
            request_id=request_id,
            seed=7,
            prompt={"prompt": "edit", "additional_information": {"batch_cond_image_info": [image]}},
        )
        return _prepare_cached_requests(request, tokenizer, image_processor)

    first = build("first")
    assert sum(call.args[0][0] == "hunyuan-reference-v1" for call in hasher.call_args_list) == 1
    assert len(first[0].mm_features) == len(first[1].mm_features) == 2
    assert first[0].mm_features == first[1].mm_features
    image.vae_image_info.image_tensor[0, 0, 0, 0] += 1
    changed = build("changed")
    assert sum(call.args[0][0] == "hunyuan-reference-v1" for call in hasher.call_args_list) == 2
    assert all(a.identifier != b.identifier for a, b in zip(first[0].mm_features, changed[0].mm_features, strict=True))


def test_failed_cfg_cache_preparation_does_not_install_partial_inputs():
    tokenizer, image_processor = _components([20, 20])
    request = _request(
        guidance_scale=5.0,
        prompt={"prompt": "edit", "additional_information": {"batch_cond_image_info": [_reference_image()]}},
    )
    layout = _prepare(request, tokenizer, image_processor)
    request.diffusion_kv_requests = build_hunyuan_diffusion_kv_requests(request, layout)
    # First row prepares successfully; the second exposes inconsistent inputs.
    layout.rope_image_info[0] = []
    request.prompt["additional_information"].pop("batch_cond_image_info")
    with pytest.raises(ValueError, match="missing its canonical inputs"):
        prepare_hunyuan_prefix_cache(request)
    assert all(row.cache_token_ids == () and row.mm_features == [] for row in request.diffusion_kv_requests)


@pytest.mark.parametrize("generator_list", [False, True])
def test_reference_identity_uses_explicit_generator_state_before_seed(generator_list):
    tokenizer, image_processor = _components([20])

    def build(request_id, seed, advance):
        request = _request(
            guidance_scale=1.0,
            request_id=request_id,
            seed=seed,
            prompt={"prompt": "edit", "additional_information": {"batch_cond_image_info": [_reference_image()]}},
        )
        generator = torch.Generator(device="cpu").manual_seed(123)
        if advance:
            torch.rand(1, generator=generator)
        request.sampling_params.generator = [generator] if generator_list else generator
        before = generator.get_state().clone()
        row = _prepare_cached_requests(request, tokenizer, image_processor)[0]
        torch.testing.assert_close(before, generator.get_state())
        row.build_block_hashes(4, get_hash_fn_by_name("sha256"))
        return row.block_hashes

    first = build("first", 7, False)
    assert first == build("same-generator-different-unused-seed", 8, False)
    changed = build("advanced-generator-same-seed", 7, True)
    assert first[:1] == changed[:1]
    assert all(a != b for a, b in zip(first[1:], changed[1:], strict=True))


def test_unseeded_reference_identity_is_request_local():
    tokenizer, image_processor = _components([20])

    def build(request_id):
        request = _request(
            guidance_scale=1.0,
            request_id=request_id,
            prompt={"prompt": "edit", "additional_information": {"batch_cond_image_info": [_reference_image()]}},
        )
        # Real requests normally receive an automatic seed in __post_init__.
        request.sampling_params.seed = None
        row = _prepare_cached_requests(request, tokenizer, image_processor)[0]
        row.build_block_hashes(4, get_hash_fn_by_name("sha256"))
        return row.block_hashes

    first = build("first")
    assert first == build("first")
    changed = build("second")
    assert first[:1] == changed[:1]
    assert all(a != b for a, b in zip(first[1:], changed[1:], strict=True))


@pytest.mark.parametrize("guidance_scale", [1.0, 5.0])
def test_text_prefix_hashes_include_lora_identity_and_scale(guidance_scale: float) -> None:
    tokenizer, image_processor = _components([20] * (1 + int(guidance_scale > 1.0)))

    def build(request_id: str, adapter_id: int | None, scale: float):
        request = _request(guidance_scale=guidance_scale, request_id=request_id)
        if adapter_id is not None:
            request.sampling_params.lora_request = LoRARequest(
                lora_name=f"adapter-{adapter_id}", lora_int_id=adapter_id, lora_path=f"/adapters/{adapter_id}"
            )
        request.sampling_params.lora_scale = scale
        kv_requests = _prepare_cached_requests(request, tokenizer, image_processor)
        for kv_request in kv_requests:
            kv_request.build_block_hashes(4, get_hash_fn_by_name("sha256"))
        return kv_requests

    base = build("base", None, 1.0)
    same_base = build("same-base", None, 0.5)  # Scale has no effect without an adapter.
    first = build("first", 1, 0.5)
    same = build("same", 1, 0.5)
    changed_id = build("changed-id", 2, 0.5)
    changed_scale = build("changed-scale", 1, 1.0)

    for row in range(len(first)):
        assert base[row].block_hashes == same_base[row].block_hashes
        assert first[row].block_hashes == same[row].block_hashes
        assert len(first[row].block_hashes) == 5
        for changed in (base, changed_id, changed_scale):
            assert first[row].cache_token_ids == changed[row].cache_token_ids
            # LoRA changes even the text prefix, so every block must differ.
            assert all(a != b for a, b in zip(first[row].block_hashes, changed[row].block_hashes, strict=True))


@pytest.fixture
def prefix_cache_manager(request):
    native_kv_managers.register_all_kvcache_specs(None)
    spec = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
    num_blocks = 64
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[build_kv_cache_tensor(spec, num_blocks, ["layer0"])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer0"], kv_cache_spec=spec)],
    )
    manager = DiffusionKVCacheManager(
        config,
        max_model_len=64,
        scheduler_block_size=4,
        hash_block_size=4,
        enable_prefix_caching=getattr(request, "param", True),
    )
    yield manager
    manager.close()


@pytest.mark.parametrize("prefix_cache_manager", [False], indirect=True)
@pytest.mark.parametrize("guidance_scale", [1.0, 5.0])
def test_paged_cache_off_prepares_and_allocates_without_identity_work(
    monkeypatch, prefix_cache_manager, guidance_scale
):
    tokenizer, image_processor = _components([20] * (1 + int(guidance_scale > 1.0)))
    config = SimpleNamespace(
        model="fake",
        model_class_name="HunyuanImage3ForCausalMM",
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        enable_prefix_caching=False,
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            vae_downsample_factor=(8, 8),
            patch_size=2,
            image_base_size=1024,
        ),
    )
    monkeypatch.setattr(pipeline_module, "HunyuanImage3ImageProcessor", lambda _config: image_processor)
    monkeypatch.setattr(pipeline_module, "TokenizerWrapper", lambda _model: tokenizer)
    monkeypatch.setattr(pipeline_module.GenerationConfig, "from_pretrained", lambda _model: SimpleNamespace())
    monkeypatch.setattr(engine_module, "get_diffusion_post_process_func", lambda _config: None)
    fail = Mock(side_effect=AssertionError("disabled prefix cache must not prepare identity"))
    monkeypatch.setattr(engine_module, "get_diffusion_prefix_cache_func", fail)
    monkeypatch.setattr(layout_module, "prepare_hunyuan_prefix_cache", fail)
    monkeypatch.setattr(layout_module, "hash_prefix_cache_value", fail)
    monkeypatch.setattr(kv_utils, "hash_prefix_cache_value", fail)
    monkeypatch.setattr(kv_utils.MultiModalHasher, "hash_kwargs", fail)
    monkeypatch.setattr(torch.Tensor, "tolist", fail)
    monkeypatch.setattr(prefix_cache_manager, "_hash_function", fail)
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = config
    engine._init_process_hooks(config)
    # Guard admission too, even if someone injects a hook after initialization.
    engine.prefix_cache_func = fail
    request = _request(
        guidance_scale=guidance_scale,
        prompt={"prompt": "edit", "additional_information": {"batch_cond_image_info": [_reference_image()]}},
    )
    request.sampling_params.seed = None
    request.sampling_params.generator = SimpleNamespace(get_state=fail)
    engine._prepare_request_for_admission(request)
    assert isinstance(request.prepared_layout, HunyuanPreparedLayout)
    assert all(row.cache_token_ids == () and row.mm_features == [] for row in request.diffusion_kv_requests)
    metadata = prefix_cache_manager.reserve_request(request.request_id, request.diffusion_kv_requests)
    assert metadata is not None
    assert all(row.cached_prefix_len == 0 for row in metadata.sequences)
    prefix_cache_manager.publish_request(request.request_id)
    prefix_cache_manager.free_request(request.request_id)
    fail.assert_not_called()


@pytest.mark.parametrize("guidance_scale", [1.0, 5.0])
@pytest.mark.parametrize(
    ("change", "expected_hit"),
    [
        ("same", 20),
        ("prompt_suffix", 12),
        ("target", 20),
        ("vae", 4),
        ("vit", 4),
        ("vision_kwargs", 4),
        ("rope", 4),
        ("seed", 4),
        ("lora_id", 0),
        ("lora_scale", 0),
        ("base", 0),
    ],
)
def test_prefix_lookup_respects_reference_and_lora_identity(
    prefix_cache_manager: DiffusionKVCacheManager, guidance_scale: float, change: str, expected_hit: int
) -> None:
    tokenizer, image_processor = _components([20] * (1 + int(guidance_scale > 1.0)))

    def build(request_id: str, change: str):
        image = _reference_image()
        if change == "vae":
            image.vae_image_info.image_tensor[0, 0, 0, 0] = 99
        elif change == "vit":
            image.vision_image_info.image_tensor[0, 0, 0, 0] = 99
        elif change == "vision_kwargs":
            image.vision_encoder_kwargs["pixel_attention_mask"][0] = False
        request = _request(
            guidance_scale=guidance_scale,
            request_id=request_id,
            seed=8 if change == "seed" else 7,
            prompt={"prompt": "edit this image", "additional_information": {"batch_cond_image_info": [image]}},
        )
        if change != "base":
            adapter_id = 2 if change == "lora_id" else 1
            request.sampling_params.lora_request = LoRARequest(
                lora_name=f"adapter-{adapter_id}", lora_int_id=adapter_id, lora_path=f"/adapters/{adapter_id}"
            )
        request.sampling_params.lora_scale = 1.0 if change == "lora_scale" else 0.5
        layout = _prepare(request, tokenizer, image_processor)
        if change == "rope":
            for spans in layout.rope_image_info:
                spans[0] = (slice(5, 7), (2, 1))
        elif change == "prompt_suffix":
            layout.tokenizer_output.tokens[:, 12] += 100
        elif change == "target":
            layout.tokenizer_output.tokens[:, 20:] += 100
        request.diffusion_kv_requests = build_hunyuan_diffusion_kv_requests(request, layout)
        prepare_hunyuan_prefix_cache(request)
        return request.diffusion_kv_requests

    manager = prefix_cache_manager
    cold = build("cold", "same")
    cold_metadata = manager.reserve_request("cold", cold)
    assert cold_metadata is not None
    assert all(sequence.cached_prefix_len == 0 for sequence in cold_metadata.sequences)
    # CPU-only: simulate successful materialization to test native cache lookup.
    manager.publish_request("cold")
    manager.free_request("cold")

    warm = build("warm", change)
    warm_metadata = manager.reserve_request("warm", warm)
    assert warm_metadata is not None
    for cold_row, warm_row, cold_sequence, warm_sequence in zip(
        cold, warm, cold_metadata.sequences, warm_metadata.sequences, strict=True
    ):
        if change == "prompt_suffix":
            assert cold_row.cache_token_ids[:12] == warm_row.cache_token_ids[:12]
            assert cold_row.cache_token_ids[12] != warm_row.cache_token_ids[12]
        else:
            assert cold_row.cache_token_ids == warm_row.cache_token_ids
        assert warm_sequence.cached_prefix_len == expected_hit
        hit_blocks = expected_hit // 4
        assert cold_sequence.block_ids[0][:hit_blocks] == warm_sequence.block_ids[0][:hit_blocks]
        assert cold_row.block_hashes[:hit_blocks] == warm_row.block_hashes[:hit_blocks]
        assert all(
            a != b for a, b in zip(cold_row.block_hashes[hit_blocks:], warm_row.block_hashes[hit_blocks:], strict=True)
        )


def test_passes_preprocessed_reference_image_geometry_to_tokenizer() -> None:
    tokenizer, image_processor = _components([20])
    joint_image = _reference_image()
    prompt = {
        "prompt": "edit this image",
        "additional_information": {"batch_cond_image_info": [joint_image]},
    }
    request = _request(guidance_scale=1.0, prompt=prompt)
    prepared_layout = _prepare(request, tokenizer, image_processor)

    kv_requests = build_hunyuan_diffusion_kv_requests(request, prepared_layout)

    assert tokenizer.calls[0]["batch_cond_image_info"] == [[joint_image]]
    assert kv_requests[0].prefix_len == 20
    assert kv_requests[0].target_len == 17
    assert kv_requests[0].seq_len == 40
    assert kv_requests[0].kv_contexts == ()


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


@pytest.mark.parametrize(
    ("prefix_lens", "guidance_scale", "with_reference_image"),
    [
        pytest.param([12], 1.0, False, id="cfg-1"),
        pytest.param([12, 14], 5.0, False, id="cfg-2"),
        pytest.param([20], 1.0, True, id="cfg-1-reference-image"),
        pytest.param([20, 22], 5.0, True, id="cfg-2-reference-image"),
    ],
)
def test_prepared_model_inputs_match_local_tokenization(
    monkeypatch,
    prefix_lens: list[int],
    guidance_scale: float,
    with_reference_image: bool,
) -> None:
    tokenizer, image_processor = _components(prefix_lens)
    prompt: str | dict = "draw a cat"
    if with_reference_image:
        prompt = {
            "prompt": "edit this image",
            "additional_information": {"batch_cond_image_info": [_reference_image()]},
        }
    request = _request(guidance_scale=guidance_scale, prompt=prompt)
    prepared_layout = _prepare(request, tokenizer, image_processor)

    prompts, cot_texts, system_prompt, batch_cond_image_info, bot_task = extract_hunyuan_prompt_inputs(
        [request.prompt],
        request.sampling_params.extra_args or {},
        request_id=request.request_id,
        allow_cond_image=True,
    )
    cot_text = (
        [normalize_hunyuan_cot_text(text) for text in cot_texts]
        if any(text is not None for text in cot_texts)
        else None
    )

    pipeline = object.__new__(HunyuanImage3Pipeline)
    pipeline._tkwrapper = tokenizer
    pipeline.image_processor = image_processor
    pipeline.generation_config = SimpleNamespace(sequence_template="instruct", drop_think=False)
    pipeline.config = SimpleNamespace(
        image_base_size=1024,
        attention_head_dim=2,
        rope_theta=10000.0,
    )
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda _self: torch.device("cpu")))

    def fake_encode_cond_image(batch_cond_image_info_list, cfg_factor=1, generator=None):
        del generator
        rows = len(batch_cond_image_info_list) * cfg_factor
        return (
            torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 1, 2, 2),
            torch.arange(rows, dtype=torch.float32),
            torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
        )

    pipeline._encode_cond_image = fake_encode_cond_image

    def fake_build_batch_2d_rope(*, image_infos, seq_len, **_kwargs):
        geometry_sum = sum(
            int(image_slice.start or 0) + int(image_slice.stop or 0) + height + width
            for row in image_infos
            for image_slice, (height, width) in row
        )
        values = torch.tensor([len(image_infos), seq_len, geometry_sum], dtype=torch.float32)
        return values, values + 1

    monkeypatch.setattr(pipeline_module, "build_batch_2d_rope", fake_build_batch_2d_rope)
    common_kwargs = dict(
        prompt=prompts,
        cot_text=cot_text,
        system_prompt=system_prompt,
        mode="gen_image",
        guidance_scale=guidance_scale,
        image_size=(request.sampling_params.height, request.sampling_params.width),
        generator=[torch.Generator().manual_seed(0)],
        batch_cond_image_info=batch_cond_image_info,
        bot_task=bot_task,
    )

    local_inputs = pipeline.prepare_model_inputs(**common_kwargs)
    prepared_inputs = pipeline.prepare_model_inputs(**common_kwargs, prepared_layout=prepared_layout)

    if with_reference_image:
        # Preparation runs without a forward context. Coverage must come from
        # the scheduler metadata copied onto this request, including every CFG row.
        reused = pipeline.prepare_model_inputs(
            # The image span is [5, 10); native KV must cover its end, not
            # merely reach the first image token.
            **common_kwargs, prepared_layout=prepared_layout, kv_computed_tokens=(10,) * len(prefix_lens)
        )
        assert reused["cond_vae_images"] is None and reused["cond_vit_images"] is None
        partial = pipeline.prepare_model_inputs(
            **common_kwargs,
            prepared_layout=prepared_layout,
            kv_computed_tokens=(10,) * (len(prefix_lens) - 1) + (9,),
        )
        assert partial["cond_vae_images"] is not None and partial["cond_vit_images"] is not None

    comparable_fields = (
        "input_ids",
        "position_ids",
        "custom_pos_emb",
        "image_mask",
        "gen_timestep_scatter_index",
        "cond_vae_images",
        "cond_timestep",
        "cond_vae_image_mask",
        "cond_vit_images",
        "cond_vit_image_mask",
        "vit_kwargs",
        "cond_timestep_scatter_index",
    )
    for field in comparable_fields:
        _assert_nested_equal(prepared_inputs[field], local_inputs[field])

    tokenizer_fields = (
        "tokens",
        "gen_image_mask",
        "cond_vae_image_mask",
        "cond_vit_image_mask",
        "gen_timestep_scatter_index",
        "cond_timestep_scatter_index",
        "real_pos",
        "all_image_slices",
        "joint_image_slices",
        "gen_image_slices",
    )
    for field in tokenizer_fields:
        _assert_nested_equal(
            getattr(prepared_inputs["tokenizer_output"], field),
            getattr(local_inputs["tokenizer_output"], field),
        )

    local_mask = pipeline._prepare_attention_mask_for_generation(
        local_inputs["input_ids"],
        pipeline.generation_config,
        local_inputs,
    )
    prepared_mask = pipeline._prepare_attention_mask_for_generation(
        prepared_inputs["input_ids"],
        pipeline.generation_config,
        prepared_inputs,
    )
    torch.testing.assert_close(prepared_mask, local_mask)
    assert prepared_inputs["full_attn_spans"] == local_inputs["full_attn_spans"]

    image_info = prepared_layout.generated_image_info
    generation_token_counts = {
        "num_image_tokens": hunyuan_num_image_tokens(image_info),
        "num_special_tokens": hunyuan_num_special_tokens(image_info),
    }
    local_inputs.update(attention_mask=local_mask, **generation_token_counts)
    prepared_inputs.update(attention_mask=prepared_mask, **generation_token_counts)
    local_step_inputs = pipeline._update_model_kwargs_for_generation(ModelOutput(), local_inputs)
    prepared_step_inputs = pipeline._update_model_kwargs_for_generation(ModelOutput(), prepared_inputs)

    for field in ("position_ids", "attention_mask", "gen_timestep_scatter_index", "full_attn_spans"):
        _assert_nested_equal(prepared_step_inputs[field], local_step_inputs[field])


def test_rejects_tokenizer_cfg_row_mismatch() -> None:
    tokenizer, image_processor = _components([12])
    request = _request(guidance_scale=5.0)
    prepared_layout = _prepare(request, tokenizer, image_processor)

    with pytest.raises(ValueError, match="sequence count does not match"):
        build_hunyuan_diffusion_kv_requests(request, prepared_layout)


def test_sender_endpoint_does_not_change_local_kv_request() -> None:
    tokenizer, image_processor = _components([12])
    request = _request(guidance_scale=1.0)
    request.kv_sender_info = {"host": "127.0.0.1", "zmq_port": 5000}
    prepared_layout = _prepare(request, tokenizer, image_processor)

    kv_requests = build_hunyuan_diffusion_kv_requests(request, prepared_layout)

    assert kv_requests[0].seq_len == 32
    assert kv_requests[0].num_computed_tokens == 0


def test_paged_preprocess_attaches_layout_and_scheduler_kv_requests(monkeypatch) -> None:
    tokenizer, image_processor = _components([12])
    hf_config = SimpleNamespace(
        vae_downsample_factor=(8, 8),
        patch_size=2,
        image_base_size=1024,
    )
    monkeypatch.setattr(pipeline_module, "get_config", lambda *_args, **_kwargs: hf_config)
    monkeypatch.setattr(pipeline_module, "HunyuanImage3ImageProcessor", lambda _config: image_processor)
    monkeypatch.setattr(pipeline_module, "TokenizerWrapper", lambda _model: tokenizer)
    monkeypatch.setattr(
        pipeline_module.GenerationConfig,
        "from_pretrained",
        lambda _model: SimpleNamespace(sequence_template="instruct", drop_think=False),
    )
    preprocess = get_hunyuan_image_3_pre_process_func(
        SimpleNamespace(model="model", diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER)
    )
    request = _request(guidance_scale=1.0)

    prepared_request = preprocess(request)

    assert prepared_request is request
    assert isinstance(request.prepared_layout, HunyuanPreparedLayout)
    assert request.diffusion_kv_requests is not None
    assert [(item.prefix_len, item.target_len, item.seq_len) for item in request.diffusion_kv_requests] == [
        (12, 17, 32)
    ]
    assert len(tokenizer.calls) == 1


@pytest.mark.parametrize("transfer_tokens, expected", [(9, [7, 5]), (4, [4, 4])])
def test_cfg_rows_preserve_reusable_token_ids_and_native_transfer_boundary(transfer_tokens, expected):
    from vllm_omni.diffusion.diffusion_kv.kv_connector import prepare_kv_requests

    tokenizer, image_processor = _components([12, 14])
    request = _request(guidance_scale=5.0)
    # Prompt token IDs are only materialized for the native AR -> DiT
    # transfer path; local prefix caching does not need them.
    request.kv_transfer_params = {"num_transfer_tokens": 0}
    layout = _prepare(request, tokenizer, image_processor)
    layout.tokenizer_output.think_recaption_end_pos = [[7], [5]]
    layout.tokenizer_output.uncond_cfg_start_pos = [[None], [5]]
    rows = build_hunyuan_diffusion_kv_requests(request, layout)
    assert [row.prompt_token_ids for row in rows] == [
        layout.tokenizer_output.tokens[0, :7].tolist(),
        layout.tokenizer_output.tokens[1, :5].tolist(),
    ]
    prepare_kv_requests(rows, {"num_transfer_tokens": transfer_tokens})
    assert [row.num_prompt_tokens for row in rows] == expected
    assert all(row.kv_transfer_params["num_transfer_tokens"] == transfer_tokens for row in rows)
    layout.tokenizer_output.uncond_cfg_start_pos = [[5]]
    with pytest.raises(ValueError, match="one boundary per CFG row"):
        build_hunyuan_diffusion_kv_requests(request, layout)


@pytest.mark.parametrize("computed, expected", [((8, 8), True), ((8, 5), False), ((0, 0), False), ((), False)])
def test_native_image_reuse_requires_coverage_in_every_cfg_row(computed, expected):
    from vllm_omni.diffusion.models.hunyuan_image3.request_layout import native_kv_covers_cond_images

    output = TokenizerEncodeOutput(joint_image_slices=[[slice(2, 8)], [slice(2, 6)]])
    assert native_kv_covers_cond_images(output, computed) is expected
