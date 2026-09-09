# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers.utils.generic import ModelOutput

import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as pipeline_module
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
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
        joint_image_slices = [[slice(1, 3), slice(3, 5)] if has_cond_image else [] for _ in range(rows)]
        gen_image_slices = [[slice(length + 1, length + 17)] for length in self.prefix_lens]
        all_image_slices = [joint + generated for joint, generated in zip(joint_image_slices, gen_image_slices)]
        cond_vae_image_mask = (
            torch.stack([(positions >= 1) & (positions < 3) for _ in range(rows)]) if has_cond_image else None
        )
        cond_vit_image_mask = (
            torch.stack([(positions >= 3) & (positions < 5) for _ in range(rows)]) if has_cond_image else None
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
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            height=768,
            width=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=4,
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
            token_width=8,
            token_height=8,
            image_token_length=64,
        ),
        vision_image_info=ImageInfo(
            image_type="siglip2",
            token_width=4,
            token_height=4,
            image_token_length=16,
        ),
        vision_encoder_kwargs={
            "spatial_shapes": torch.tensor([4, 4]),
            "pixel_attention_mask": torch.ones(4, dtype=torch.bool),
        },
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
