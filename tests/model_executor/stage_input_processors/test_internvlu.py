# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.diffusion.models.internvlu.pipeline_internvlu import (
    InternVLUPipeline,
)
from vllm_omni.model_executor.models.internvlu.internvlu import (
    THINK_COT_TOKEN_LIMIT,
    InternVLUChatModel,
)
from vllm_omni.model_executor.models.internvlu.pipeline import INTERNVLU_PIPELINE
from vllm_omni.model_executor.models.internvlu.processing import (
    DEFAULT_SYSTEM_MESSAGE,
    IMAGE_GENERATION_SYSTEM_MESSAGE,
    _prepare_prompt,
)
from vllm_omni.model_executor.stage_input_processors.internvlu import (
    CFG_ROLES,
    EOS_TOKEN_ID,
    IM_START_TOKEN_ID,
    IMG_CONTEXT_TOKEN_ID,
    IMG_END_TOKEN_ID,
    IMG_START_TOKEN_ID,
    PARTIAL_EDIT_TEXT,
    PARTIAL_SUFFIX,
    PURE_UNCONDITIONAL_TEXT,
    UNCONDITIONAL_SUFFIX,
    ar2diffusion,
    expand_cfg_prompts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _prompt_ids(reference_count: int) -> list[int]:
    token_ids = [
        IM_START_TOKEN_ID,
        10,
        EOS_TOKEN_ID,
        IM_START_TOKEN_ID,
        20,
    ]
    for _ in range(reference_count):
        token_ids.extend([IMG_START_TOKEN_ID] + [IMG_CONTEXT_TOKEN_ID] * 256 + [IMG_END_TOKEN_ID])
    token_ids.extend([EOS_TOKEN_ID, IM_START_TOKEN_ID, 30, IMG_START_TOKEN_ID])
    return token_ids


def _source_output(request_id: str, reference_count: int, *, eos: int = EOS_TOKEN_ID):
    prompt_ids = _prompt_ids(reference_count)
    hidden = torch.zeros(len(prompt_ids), 4096)
    completion = SimpleNamespace(cumulative_token_ids=[eos])
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        outputs=[completion],
        multimodal_output={"hidden_states": {"output": hidden}},
    )


def _three_outputs(parent_id: str, reference_count: int):
    return [
        _source_output(parent_id + UNCONDITIONAL_SUFFIX, 0),
        _source_output(parent_id, reference_count),
        _source_output(parent_id + PARTIAL_SUFFIX, reference_count),
    ]


def test_pipeline_topology_matches_native_ar_to_diffusion_contract():
    assert INTERNVLU_PIPELINE.model_type == "internvlu_chat"
    assert INTERNVLU_PIPELINE.diffusers_class_name == "InternVLUPipeline"
    assert INTERNVLU_PIPELINE.validate() == []

    stage0, stage1 = INTERNVLU_PIPELINE.stages
    assert stage0.model_subdir == "vlm"
    assert stage0.tokenizer_subdir == "processor"
    assert stage0.engine_output_type == "latent"
    assert stage0.final_output_type == "text"
    assert stage1.input_sources == (0,)
    assert stage1.final_output_type == "image"


def test_t2i_expansion_creates_two_prefill_only_pure_unconditional_companions():
    base_params = SamplingParams(max_tokens=99)
    expanded = expand_cfg_prompts(
        {"prompt": "draw a fox", "modalities": ["image"]},
        base_params,
    )

    assert [item.role for item in expanded] == ["partial", "unconditional"]
    assert [item.request_id_suffix for item in expanded] == [PARTIAL_SUFFIX, UNCONDITIONAL_SUFFIX]
    assert all(item.prompt["prompt"] == PURE_UNCONDITIONAL_TEXT for item in expanded)
    assert all(item.prompt["mm_processor_kwargs"]["internvlu_generation_mode"] == "image" for item in expanded)

    patched, sampling_list = expanded[0].apply_overrides(base_params, [base_params, object()])
    assert patched.max_tokens == 1
    assert sampling_list[0] is patched
    assert base_params.max_tokens == 99


def test_plain_string_prompt_does_not_expand_cfg_companions():
    assert expand_cfg_prompts("describe a fox", SimpleNamespace(max_tokens=9)) == []


@pytest.mark.parametrize(
    "prompt",
    [
        {"prompt": "answer a question", "modalities": ["text"]},
        {
            "prompt": "describe the reference image",
            "modalities": ["text"],
            "multi_modal_data": {"image": [object()]},
        },
    ],
)
def test_text_and_image_understanding_prompts_do_not_expand_cfg_companions(prompt):
    assert expand_cfg_prompts(prompt, SimpleNamespace(max_tokens=9)) == []


def test_edit_expansion_retains_all_partial_refs_and_drops_unconditional_refs():
    image0, image1 = object(), object()
    expanded = expand_cfg_prompts(
        {
            "prompt": "replace the sky",
            "modalities": ["img2img"],
            "multi_modal_data": {"img2img": [image0, image1]},
            "multi_modal_uuids": {"img2img": ["ref-0", "ref-1"]},
        },
        SimpleNamespace(max_tokens=9),
    )

    partial, unconditional = expanded
    assert partial.prompt["prompt"] == PARTIAL_EDIT_TEXT
    assert partial.prompt["multi_modal_data"] == {"image": [image0, image1]}
    assert partial.prompt["multi_modal_uuids"] == {"image": ["ref-0", "ref-1"]}
    assert unconditional.prompt["prompt"] == PURE_UNCONDITIONAL_TEXT
    assert unconditional.prompt["mm_processor_kwargs"]["internvlu_generation_mode"] == "editing"
    assert "multi_modal_data" not in unconditional.prompt
    assert "multi_modal_uuids" not in unconditional.prompt


def test_edit_expansion_separates_image_output_from_img2img_processing_mode():
    reference = object()
    expanded = expand_cfg_prompts(
        {
            "prompt": "repaint the scene",
            "modalities": ["image"],
            "multi_modal_data": {"image": [reference]},
            "mm_processor_kwargs": {"modalities": ["img2img"]},
        },
        SimpleNamespace(max_tokens=9),
    )

    partial, unconditional = expanded
    assert partial.prompt["multi_modal_data"] == {"image": [reference]}
    assert partial.prompt["mm_processor_kwargs"]["internvlu_generation_mode"] == "editing"
    assert "multi_modal_data" not in unconditional.prompt


def test_cfg_roles_order_is_the_stage1_batch_layout():
    """Reordering CFG_ROLES would silently swap Stage 1's CFG guidance strengths."""
    assert CFG_ROLES == ("conditional", "partial", "unconditional")


def test_bridge_binds_roles_by_suffix_and_builds_exact_reference_mapping():
    image0, image1 = object(), object()
    prompt = {
        "prompt": "edit",
        "multi_modal_data": {"img2img": [image0, image1]},
    }
    sampling_params = SimpleNamespace(height=770, width=513, resolution=None)
    result = ar2diffusion(
        _three_outputs("req", 2),
        prompt,
        # Diffusion StageMetadata currently reports False; the model-local
        # bridge must still carry its own canonical reference images.
        requires_multimodal_data=False,
        sampling_params=sampling_params,
    )

    assert result["height"] == 768
    assert result["width"] == 512
    assert sampling_params.resolution == 770
    assert result["multi_modal_data"]["image"] == [image0, image1]

    payload = result["extra"]["internvlu"]
    assert payload["requested_height"] == 770
    assert payload["requested_width"] == 513
    assert payload["image_grid_thw_gen"].tolist() == [[1, 96, 64]]

    branches = payload["branches"]
    assert tuple(branches) == ("conditional", "partial", "unconditional")
    for role in ("conditional", "partial"):
        assert branches[role]["reference_image_indices"].tolist() == [0, 1]
        assert branches[role]["image_fhw_cond"].tolist() == [[1, 16, 16], [1, 16, 16]]
        assert branches[role]["encoder_hidden_states"].shape[-1] == 4096
        assert branches[role]["encoder_image_token_mask"].sum().item() == 512
    assert branches["unconditional"]["reference_image_indices"].numel() == 0


def test_bridge_uses_resolution_for_square_shared_example_requests():
    sampling_params = SimpleNamespace(height=None, width=None, resolution=512)
    result = ar2diffusion(
        _three_outputs("resolution", 0),
        {"prompt": "draw a square"},
        sampling_params=sampling_params,
    )

    assert result["height"] == 512
    assert result["width"] == 512
    assert result["extra"]["internvlu"]["requested_height"] == 512
    assert result["extra"]["internvlu"]["requested_width"] == 512


def test_bridge_and_stage1_preserve_stride16_generation_size():
    sampling_params = SimpleNamespace(height=1008, width=1024, resolution=None)
    result = ar2diffusion(
        _three_outputs("stride-16", 0),
        {"prompt": "draw at the released resolution"},
        sampling_params=sampling_params,
    )

    assert (result["height"], result["width"]) == (1008, 1024)
    assert result["extra"]["internvlu"]["image_grid_thw_gen"].tolist() == [[1, 126, 128]]

    pipeline = InternVLUPipeline.__new__(InternVLUPipeline)
    pipeline.vlm_cond_hidden_size = 4096
    parsed = pipeline._parse_prompt(
        result,
        sampling_params,
        is_dummy=False,
    )
    assert (parsed["height"], parsed["width"]) == (1008, 1024)
    assert parsed["image_grid_thw_gen"].tolist() == [[1, 126, 128]]


def _cot_conditional_output(request_id: str, cot_tokens: list[int], *, cot_text: str = "Let me plan."):
    prompt_ids = _prompt_ids(0)[:-1]  # think-mode prompts do not end with <img>
    output_ids = [*cot_tokens, IMG_START_TOKEN_ID, EOS_TOKEN_ID]
    hidden = torch.zeros(len(prompt_ids) + len(output_ids) - 1, 4096)
    completion = SimpleNamespace(cumulative_token_ids=output_ids, cumulative_text=cot_text)
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        outputs=[completion],
        multimodal_output={"hidden_states": {"output": hidden}},
    )


def test_bridge_accepts_cot_conditional_branch_and_surfaces_text():
    cot_tokens = [40, 41, 42]
    outputs = [
        _cot_conditional_output("req", cot_tokens),
        _source_output("req" + PARTIAL_SUFFIX, 0),
        _source_output("req" + UNCONDITIONAL_SUFFIX, 0),
    ]
    result = ar2diffusion(outputs, {"prompt": "reason then draw"})

    payload = result["extra"]["internvlu"]
    assert payload["is_cot"] is True
    assert result["extra"]["ar_generated_text"] == "Let me plan."
    conditional = payload["branches"]["conditional"]
    # Branches carry only what Stage 1 consumes — token ids stay bridge-local.
    # The slice (second <|im_start|> through the terminal <img>, dropping the
    # unaligned forced-EOS tail) is pinned via the shipped row counts.
    assert "token_ids" not in conditional
    prompt_ids = _prompt_ids(0)[:-1]
    slice_start = prompt_ids.index(IM_START_TOKEN_ID, 1)
    aligned_len = len(prompt_ids) + len(cot_tokens) + 2 - 1  # output adds <img> + EOS
    assert conditional["encoder_hidden_states"].shape[0] == aligned_len - slice_start
    assert conditional["encoder_image_token_mask"].shape[0] == aligned_len - slice_start


def test_bridge_marks_direct_image_requests_as_non_cot():
    result = ar2diffusion(_three_outputs("req", 0), {"prompt": "draw"})
    assert result["extra"]["internvlu"]["is_cot"] is False
    assert "ar_generated_text" not in result["extra"]


def test_bridge_conditioning_check_is_width_agnostic_but_structural():
    """Stage 1 owns the conditioning width (generation_decoder
    input_hidden_size); the bridge validates only the row structure its own
    slicing depends on: a 2-D floating-point [T, hidden] matrix."""
    outputs = _three_outputs("req-width", 0)
    for source in outputs:
        rows = source.multimodal_output["hidden_states"]["output"].shape[0]
        source.multimodal_output["hidden_states"]["output"] = torch.zeros(rows, 8)
    result = ar2diffusion(outputs, {"prompt": "draw"}, sampling_params=None)
    branches = result["extra"]["internvlu"]["branches"]
    assert branches["conditional"]["encoder_hidden_states"].shape[-1] == 8

    outputs = _three_outputs("req-1d", 0)
    rows = outputs[1].multimodal_output["hidden_states"]["output"].shape[0]
    outputs[1].multimodal_output["hidden_states"]["output"] = torch.zeros(rows)
    with pytest.raises(ValueError, match="matrix"):
        ar2diffusion(outputs, {"prompt": "draw"}, sampling_params=None)

    outputs = _three_outputs("req-int", 0)
    rows = outputs[1].multimodal_output["hidden_states"]["output"].shape[0]
    outputs[1].multimodal_output["hidden_states"]["output"] = torch.zeros(rows, 8, dtype=torch.long)
    with pytest.raises(TypeError, match="floating point"):
        ar2diffusion(outputs, {"prompt": "draw"}, sampling_params=None)


def test_bridge_rejects_unknown_role_suffix():
    outputs = [
        _source_output("req", 0),
        _source_output("req" + PARTIAL_SUFFIX, 0),
        _source_output("req__internvlu_mystery", 0),
    ]
    with pytest.raises(ValueError, match="Unknown InternVL-U CFG request suffix"):
        ar2diffusion(outputs, {"prompt": "x"})


def test_bridge_rejects_non_eos_lifecycle_token():
    outputs = _three_outputs("req", 0)
    outputs[1] = _source_output("req", 0, eos=42)
    with pytest.raises(ValueError, match="must terminate with forced EOS"):
        ar2diffusion(outputs, {"prompt": "x"})


def test_bridge_rejects_reference_count_mismatch():
    with pytest.raises(ValueError, match="expected 1"):
        ar2diffusion(
            _three_outputs("req", 0),
            {"prompt": "edit", "multi_modal_data": {"image": [object()]}},
            requires_multimodal_data=True,
        )


def test_raw_t2i_prompt_preserves_assistant_newline_before_img():
    prompt = _prepare_prompt(
        "draw a fox",
        num_images=0,
        modalities=("image",),
        generation_mode=None,
        system_prompt=None,
    )

    assert f"<|im_start|>system\n{IMAGE_GENERATION_SYSTEM_MESSAGE}" in prompt
    assert "<|im_start|>user\nDraw a fox.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<img>")


def test_formatted_understanding_prompt_gets_default_system_and_normalization():
    rendered = "<|im_start|>user\n  describe this image  <|im_end|>\n<|im_start|>assistant\n"
    prompt = _prepare_prompt(
        rendered,
        num_images=0,
        modalities=(),
        generation_mode=None,
        system_prompt=None,
    )

    assert prompt.startswith(f"<|im_start|>system\n{DEFAULT_SYSTEM_MESSAGE}<|im_end|>\n")
    assert "<|im_start|>user\nDescribe this image.<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_formatted_prompt_preserves_or_explicitly_replaces_system_turn():
    rendered = (
        "<|im_start|>system\nKeep this system<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )
    preserved = _prepare_prompt(
        rendered,
        num_images=0,
        modalities=(),
        generation_mode=None,
        system_prompt=None,
    )
    replaced = _prepare_prompt(
        rendered,
        num_images=0,
        modalities=(),
        generation_mode=None,
        system_prompt="Replacement system",
    )

    assert "<|im_start|>system\nKeep this system<|im_end|>" in preserved
    assert "<|im_start|>system\nReplacement system<|im_end|>" in replaced
    assert "Keep this system" not in replaced


def test_think_mode_uses_generation_system_but_defers_img_token():
    prompt = _prepare_prompt(
        "reason then draw",
        num_images=0,
        modalities=("image",),
        generation_mode=None,
        system_prompt=None,
        think=True,
    )

    assert f"<|im_start|>system\n{IMAGE_GENERATION_SYSTEM_MESSAGE}" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    assert "<img>" not in prompt


def test_think_default_leaves_understanding_prompts_untouched():
    """The think deploy sets the knob through engine mm_processor_kwargs, so
    it also reaches plain understanding requests and must be inert there."""
    prompt = _prepare_prompt(
        "answer a question",
        num_images=0,
        modalities=(),
        generation_mode=None,
        system_prompt=None,
        think=True,
    )

    assert f"<|im_start|>system\n{DEFAULT_SYSTEM_MESSAGE}<|im_end|>" in prompt
    assert "<img>" not in prompt


def test_explicit_generation_mode_wins_over_think_default():
    """CFG companions pin internvlu_generation_mode, so they keep the direct
    <img>-terminated form even inside a think deployment."""
    prompt = _prepare_prompt(
        PURE_UNCONDITIONAL_TEXT,
        num_images=0,
        modalities=("image",),
        generation_mode="image",
        system_prompt=None,
        think=True,
    )

    assert prompt.endswith("<|im_start|>assistant\n<img>")


def test_rest_generation_kwargs_imply_image_generation():
    """/v1/images paths pass target_h/w without a modalities mm kwarg."""
    prompt = _prepare_prompt(
        "draw a fox",
        num_images=0,
        modalities=(),
        generation_mode=None,
        system_prompt=None,
        has_generation_kwargs=True,
    )

    assert f"<|im_start|>system\n{IMAGE_GENERATION_SYSTEM_MESSAGE}" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<img>")


def test_model_sampler_overwrites_internal_img_rows_after_standard_sampling():
    class FakeSampler(Sampler):
        def forward(self, logits, sampling_metadata, **_kwargs) -> SamplerOutput:
            return SamplerOutput(
                sampled_token_ids=torch.tensor([[91], [92]], dtype=torch.long),
                logprobs_tensors=None,
            )

    model = InternVLUChatModel.__new__(InternVLUChatModel)
    torch.nn.Module.__init__(model)
    model._sampler = FakeSampler()
    model._last_step_input_ids = torch.tensor([3, 4], dtype=torch.long)
    model._last_step_query_start_loc = None
    model.think = False
    model.img_start_token_id = 3
    model.eos_token_id = 2

    output = model.sample(
        torch.zeros(2, 8),
        cast(SamplingMetadata, SimpleNamespace()),
    )

    assert output is not None
    assert output.sampled_token_ids.tolist() == [[2], [92]]


def test_model_sampler_replaces_cot_eos_with_img_only_in_think_deployments():
    class FakeSampler(Sampler):
        def forward(self, logits, sampling_metadata, **_kwargs) -> SamplerOutput:
            # Rows: [0] CoT row sampling EOS, [1] CoT row sampling ordinary
            # text, [2] row whose last input is already <img> (forced-EOS
            # wins over the think substitution).
            return SamplerOutput(
                sampled_token_ids=torch.tensor([[2], [77], [88]], dtype=torch.long),
                logprobs_tensors=None,
            )

    model = InternVLUChatModel.__new__(InternVLUChatModel)
    torch.nn.Module.__init__(model)
    model._sampler = FakeSampler()
    model._last_step_input_ids = torch.tensor([5, 7, 3], dtype=torch.long)
    model._last_step_query_start_loc = None
    # Single deployment-wide knob (mm_processor_kwargs.think), mirrored onto
    # the model at construction time.  Without per-request stage metadata the
    # substitution conservatively applies to every row.
    model.think = True
    model._last_step_image_gen_rows = None
    model._last_step_cot_limit_rows = None
    model.img_start_token_id = 3
    model.eos_token_id = 2

    output = model.sample(
        torch.zeros(3, 8),
        cast(SamplingMetadata, SimpleNamespace()),
    )

    assert output is not None
    assert output.sampled_token_ids.tolist() == [[3], [77], [2]]


def test_model_sampler_keeps_chat_eos_in_think_deployments():
    """The EOS -> <img> substitution is gated per request on the engine's
    omni_final_stage_id tag: chat rows (final stage 0) keep their sampled EOS,
    while image-generation rows and unmarked rows take the swap."""

    class FakeSampler(Sampler):
        def forward(self, logits, sampling_metadata, **_kwargs) -> SamplerOutput:
            # Every row samples EOS; the per-request gate alone decides.
            return SamplerOutput(
                sampled_token_ids=torch.tensor([[2], [2], [2]], dtype=torch.long),
                logprobs_tensors=None,
            )

    model = InternVLUChatModel.__new__(InternVLUChatModel)
    torch.nn.Module.__init__(model)
    model._sampler = FakeSampler()
    model._last_step_input_ids = torch.tensor([5, 7, 9], dtype=torch.long)
    model._last_step_query_start_loc = None
    model.think = True
    model.img_start_token_id = 3
    model.eos_token_id = 2
    model._capture_row_metadata(
        [
            {"omni_final_stage_id": 0},  # chat: finalizes at the AR stage
            {"omni_final_stage_id": 1},  # image generation: continues to diffusion
            {},  # unmarked: conservative image-generation fallback
        ]
    )

    output = model.sample(
        torch.zeros(3, 8),
        cast(SamplingMetadata, SimpleNamespace()),
    )

    assert output is not None
    assert output.sampled_token_ids.tolist() == [[2], [3], [3]]

    # Metadata from a different batch shape must not gate anything.
    model._capture_row_metadata([{"omni_final_stage_id": 0}])
    output = model.sample(
        torch.zeros(3, 8),
        cast(SamplingMetadata, SimpleNamespace()),
    )

    assert output is not None
    assert output.sampled_token_ids.tolist() == [[3], [3], [3]]


def test_model_sampler_forces_img_at_cot_token_limit():
    """Image-generation rows whose CoT hits THINK_COT_TOKEN_LIMIT take <img>
    without sampling EOS; rows under the limit, chat rows, and the
    <img>-terminated forced-EOS step stay untouched."""

    class FakeSampler(Sampler):
        def forward(self, logits, sampling_metadata, **_kwargs) -> SamplerOutput:
            # Every row samples ordinary text; the CoT cap alone decides.
            return SamplerOutput(
                sampled_token_ids=torch.tensor([[77], [77], [77], [77]], dtype=torch.long),
                logprobs_tensors=None,
            )

    model = InternVLUChatModel.__new__(InternVLUChatModel)
    torch.nn.Module.__init__(model)
    model._sampler = FakeSampler()
    # Row 3's final input is <img>: its forced EOS must survive the cap.
    model._last_step_input_ids = torch.tensor([5, 7, 9, 3], dtype=torch.long)
    model._last_step_query_start_loc = None
    model.think = True
    model.img_start_token_id = 3
    model.eos_token_id = 2
    model._capture_row_metadata(
        [
            {"omni_final_stage_id": 1, "generated_len": THINK_COT_TOKEN_LIMIT},
            {"omni_final_stage_id": 1, "generated_len": THINK_COT_TOKEN_LIMIT - 1},
            {"omni_final_stage_id": 0, "generated_len": THINK_COT_TOKEN_LIMIT},
            {"omni_final_stage_id": 1, "generated_len": THINK_COT_TOKEN_LIMIT + 1},
        ]
    )

    output = model.sample(
        torch.zeros(4, 8),
        cast(SamplingMetadata, SimpleNamespace()),
    )

    assert output is not None
    assert output.sampled_token_ids.tolist() == [[3], [77], [77], [2]]
