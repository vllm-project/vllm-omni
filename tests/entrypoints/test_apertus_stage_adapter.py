from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.apertus import (
    merge_image_placeholders,
    prefill_to_decode,
)


def test_merge_image_placeholders_replaces_in_order():
    prompt = "Describe this: <|image|> and then this: <|image|>."
    merged = merge_image_placeholders(
        prompt,
        image_prompts=["<img_a>", "<img_b>"],
    )
    assert merged == "Describe this: <img_a> and then this: <img_b>."


def test_merge_image_placeholders_requires_explicit_placeholders():
    prompt = "What is in the image?"
    with pytest.raises(ValueError, match="Mismatch"):
        merge_image_placeholders(
            prompt,
            image_prompts=["<img_a>", "<img_b>"],
            image_placeholder="<|image|>",
        )


def test_merge_image_placeholders_raises_on_count_mismatch():
    with pytest.raises(ValueError, match="Mismatch"):
        merge_image_placeholders(
            "Look: <|image|> and <|image|>.",
            image_prompts=["<img_a>"],
        )


def test_prefill_to_decode_forwards_prompt_and_generated_tokens():
    source_output = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        outputs=[SimpleNamespace(token_ids=[4, 5])],
    )
    stage = SimpleNamespace(engine_outputs=[source_output])

    outputs = prefill_to_decode(
        stage_list=[stage],
        engine_input_source=[0],
    )

    assert len(outputs) == 1
    assert outputs[0]["prompt_token_ids"] == [1, 2, 3, 4, 5]


def test_prefill_to_decode_raises_on_empty_source():
    with pytest.raises(ValueError, match="engine_input_source"):
        prefill_to_decode(
            stage_list=[],
            engine_input_source=[],
        )
