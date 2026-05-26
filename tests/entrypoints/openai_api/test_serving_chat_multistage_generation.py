# SPDX-License-Identifier: Apache-2.0
"""Regression tests for multistage diffusion generation input construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    return object.__new__(OmniOpenAIServingChat)


def test_build_multistage_generation_inputs_applies_stage_specific_overrides(serving_chat):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.2, seed=11),
            OmniDiffusionSamplingParams(),
            OmniDiffusionSamplingParams(),
        ],
    )
    reference_image = Image.new("RGB", (24, 24), color="green")
    extra_body = {
        "negative_prompt": "blurry",
        "num_inference_steps": 28,
        "guidance_scale": 7.5,
        "true_cfg_scale": 5.0,
        "guidance_scale_2": 1.25,
        "layers": 6,
        "resolution": 1024,
        "lora": {"name": "adapter-a", "path": "/tmp/adapter-a", "scale": 0.6},
    }
    gen_params = OmniDiffusionSamplingParams(height=768, width=1024, seed=0, num_outputs_per_prompt=2)

    engine_prompt, sampling_params_list = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="draw a robot",
        extra_body=extra_body,
        reference_images=[reference_image],
        gen_params=gen_params,
    )

    assert engine_prompt["prompt"] == "draw a robot"
    assert engine_prompt["modalities"] == ["img2img"]
    assert engine_prompt["negative_prompt"] == "blurry"
    assert engine_prompt["mm_processor_kwargs"] == {"target_h": 768, "target_w": 1024}
    assert engine_prompt["multi_modal_data"]["img2img"].size == (24, 24)

    assert len(sampling_params_list) == 3
    assert sampling_params_list[0].temperature == 0.2
    assert sampling_params_list[0].seed == 0
    assert sampling_params_list[0].extra_args == {"target_h": 768, "target_w": 1024}
    assert sampling_params_list[1] is not gen_params
    assert sampling_params_list[2] is not gen_params
    assert sampling_params_list[1] is not sampling_params_list[2]
    assert sampling_params_list[1].height == 768
    assert sampling_params_list[1].width == 1024
    assert sampling_params_list[1].seed == 0
    assert sampling_params_list[1].num_inference_steps == 28
    assert sampling_params_list[1].guidance_scale == 7.5
    assert sampling_params_list[1].num_outputs_per_prompt == 2
    assert sampling_params_list[1].true_cfg_scale == 5.0
    assert sampling_params_list[1].lora_request.name == "adapter-a"
    assert sampling_params_list[1].lora_scale == 0.6
    assert sampling_params_list[2].height == 768
    assert sampling_params_list[2].width == 1024
    assert sampling_params_list[2].seed == 0
    assert sampling_params_list[2].num_inference_steps == 28
    assert sampling_params_list[2].lora_request.name == "adapter-a"
    assert sampling_params_list[2].lora_scale == 0.6
    assert gen_params.lora_request is None
    assert engine.default_sampling_params_list[1].height is None
    assert engine.default_sampling_params_list[1].lora_request is None
    assert engine.default_sampling_params_list[2].resolution == 640
    assert engine.default_sampling_params_list[2].lora_request is None


def test_build_multistage_generation_inputs_multi_image_emits_n_img_placeholders(serving_chat):
    """N reference images with bot_task set must emit N <img> placeholders.

    Regression: prior to the multi-image online fix, build_prompt was
    called without num_images, defaulting to 1. A 2-image edit request
    would only get a single <img> placeholder in the AR prompt; vLLMs
    _process_multimodal then raised
    AssertionError(Failed to apply prompt replacement for mm_items[image][1])
    when trying to replace the second image (no placeholder left for it).

    Pins the contract that build_prompt() is invoked with the actual image
    count so multi-image IT2I is wired correctly through the online
    /v1/images/edits path.
    """
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.0),
            OmniDiffusionSamplingParams(),
        ],
    )
    IMG = "<img>"
    images = [Image.new("RGB", (32, 32), color="red") for _ in range(3)]

    for n in (1, 2, 3):
        engine_prompt, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
            serving_chat,
            engine=engine,
            prompt="edit me",
            extra_body={"bot_task": "think"},
            reference_images=images[:n],
            gen_params=OmniDiffusionSamplingParams(),
        )
        prompt_str = engine_prompt["prompt"]
        assert prompt_str.count("<img>") == n, (
            f"N={n}: expected {n} <img> placeholders, got {prompt_str.count(IMG)} -- prompt: {prompt_str!r}"
        )


def test_build_multistage_generation_inputs_tokenizer_path_emits_prompt_token_ids(serving_chat):
    """When a tokenizer is provided, the helper must emit HF byte-for-byte
    prompt_token_ids and forward use_system_prompt to the engine prompt.

    Regression: prior to the HF-byte-equivalent fix, online IT2I always
    passed the prompt as a single string. The engine then BPE-merged across
    chat-template segment boundaries (e.g. user_prompt-ending punctuation
    plus the trailing \n\n before \"Assistant: \") producing a token
    sequence that differs from HF apply_chat_template / offline
    end2end.py. AR generated different cot_text (706 tokens / 1190 chars
    vs offline 661 / 1118 for the same inputs) and DiT produced a visually
    different image (yin-yang on brushed-metal vs three-blue swirl on
    canvas) under the same seed.

    Pins:
      1. engine_prompt[\"prompt_token_ids\"] is set when tokenizer is passed.
      2. engine_prompt[\"prompt\"] stays as the raw user prompt -- the DiT
         side rebuilds its own system prefix via use_system_prompt.
      3. engine_prompt[\"use_system_prompt\"] == \"en_unified\" so
         ar2diffusion forwards the matching system prompt to DiT.
      4. N reference images emit N <img> token ids in the AR sequence.
    """
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    # Minimal FakeTokenizer mirroring tests/diffusion/.../test_hunyuan_image3_it2i_multi_image.py
    class FakeTokenizer:
        SPECIAL = {
            "<|startoftext|>": 1,
            "<img>": 2,
            "<think>": 3,
            "<recaption>": 4,
        }

        def convert_tokens_to_ids(self, tok: str) -> int:
            return self.SPECIAL.get(tok, 0)

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(range(100, 100 + len(text)))

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.0),
            OmniDiffusionSamplingParams(),
        ],
    )
    PROMPT_KEY = "prompt"
    USP_KEY = "use_system_prompt"
    images = [Image.new("RGB", (32, 32), color="red") for _ in range(3)]

    for n in (1, 2, 3):
        tok = FakeTokenizer()
        engine_prompt, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
            serving_chat,
            engine=engine,
            prompt="edit me",
            extra_body={"bot_task": "think"},
            reference_images=images[:n],
            gen_params=OmniDiffusionSamplingParams(),
            tokenizer=tok,
        )
        # (1) prompt_token_ids must be set and non-empty
        assert "prompt_token_ids" in engine_prompt, f"N={n}: prompt_token_ids missing"
        token_ids = engine_prompt["prompt_token_ids"]
        assert isinstance(token_ids, list) and len(token_ids) > 0, f"N={n}: prompt_token_ids empty"
        # (2) raw prompt preserved (DiT bridge needs raw user text)
        assert engine_prompt["prompt"] == "edit me", (
            f"N={n}: prompt must stay raw user text, got {engine_prompt[PROMPT_KEY]!r}"
        )
        # (3) use_system_prompt forwarded for ar2diffusion bridge
        assert engine_prompt.get("use_system_prompt") == "en_unified", (
            f"N={n}: use_system_prompt must be en_unified, got {engine_prompt.get(USP_KEY)!r}"
        )
        # (4) N <img> token ids (id=2 in FakeTokenizer)
        img_count = token_ids.count(2)
        assert img_count == n, f"N={n}: expected {n} <img> token ids in prompt_token_ids, got {img_count}"


def test_build_multistage_generation_inputs_bot_task_semantic_changes_trigger_and_sys(serving_chat):
    """Passing bot_task=think_recaption (vs default "think") must flip the
    resolved sys_type to en_think_recaption (and trigger tag is still
    <think>). Pins that the API actually plumbs the bot_task semantic
    through to build_prompt rather than ignoring it.
    """
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.0),
            OmniDiffusionSamplingParams(),
        ],
    )
    images = [Image.new("RGB", (32, 32), color="red")]

    # Default bot_task (think) -> en_unified system prompt baked into the
    # legacy string path. Use legacy build_prompt (tokenizer=None) so the
    # rendered prompt is a string we can grep.
    think_prompt, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="edit me",
        extra_body={"task": "it2i", "bot_task": "think"},
        reference_images=images,
        gen_params=OmniDiffusionSamplingParams(),
    )
    # think_recaption -> en_think_recaption system prompt (different content).
    recap_prompt, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="edit me",
        extra_body={"task": "it2i", "bot_task": "think_recaption"},
        reference_images=images,
        gen_params=OmniDiffusionSamplingParams(),
    )
    assert think_prompt["prompt"] != recap_prompt["prompt"], (
        "bot_task semantic must change the rendered system prompt: "
        f"think/think_recaption produced identical strings (len={len(think_prompt['prompt'])})"
    )


def test_build_multistage_generation_inputs_sys_type_override(serving_chat):
    """Caller-supplied sys_type must override the bot_task-derived default.
    Mirrors offline `--bot-task think_recaption --sys-type en_unified`
    where the user wants think_recaptions trigger but the unified system
    prompt body.
    """
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.0),
            OmniDiffusionSamplingParams(),
        ],
    )
    images = [Image.new("RGB", (32, 32), color="red")]

    # think_recaption defaults sys_type -> en_think_recaption.
    default_sys, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="edit me",
        extra_body={"task": "it2i", "bot_task": "think_recaption"},
        reference_images=images,
        gen_params=OmniDiffusionSamplingParams(),
    )
    # sys_type=en_unified overrides -> same system body as bot_task=think.
    overridden, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="edit me",
        extra_body={"task": "it2i", "bot_task": "think_recaption", "sys_type": "en_unified"},
        reference_images=images,
        gen_params=OmniDiffusionSamplingParams(),
    )
    plain_think, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="edit me",
        extra_body={"task": "it2i", "bot_task": "think"},
        reference_images=images,
        gen_params=OmniDiffusionSamplingParams(),
    )

    # Override must (a) differ from the no-override default, and (b) equal
    # the prompt that bot_task=think produces (both end up with
    # en_unified system body + <think> trigger).
    assert overridden["prompt"] != default_sys["prompt"], (
        "sys_type override must change the rendered prompt body vs the bot_task default"
    )
    assert overridden["prompt"] == plain_think["prompt"], (
        "sys_type=en_unified + bot_task=think_recaption must produce the same prompt as "
        "bot_task=think (both = en_unified system body + <think> trigger)"
    )


def test_build_multistage_generation_inputs_custom_system_prompt(serving_chat):
    """`extra_body["system_prompt"]` must reach build_prompt as
    `custom_system_prompt`, enabling sys_type="custom" callers to inject
    a verbatim system body. Without this plumbing the sys_type="custom"
    branch in get_system_prompt() returns None and silently drops the
    user-supplied content.
    """
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.0),
            OmniDiffusionSamplingParams(),
        ],
    )
    images = [Image.new("RGB", (32, 32), color="red")]

    marker = "ZZZ_CUSTOM_SYSTEM_PROMPT_MARKER_ZZZ"

    out, _ = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="edit me",
        extra_body={
            "task": "it2i",
            "bot_task": "think",
            "sys_type": "custom",
            "system_prompt": marker,
        },
        reference_images=images,
        gen_params=OmniDiffusionSamplingParams(),
    )
    assert marker in out["prompt"], (
        f"custom system_prompt content must reach the rendered prompt; "
        f"marker {marker!r} not found in prompt of length {len(out['prompt'])}"
    )
