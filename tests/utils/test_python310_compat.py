import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_comfyui_types_imports_on_python310() -> None:
    types_path = Path(__file__).parents[2] / "apps" / "ComfyUI-vLLM-Omni" / "comfyui_vllm_omni" / "utils" / "types.py"
    spec = importlib.util.spec_from_file_location("comfyui_vllm_omni_test_types", types_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.Spec.__required_keys__ == {"stages", "modes"}
    assert module.Spec.__optional_keys__ == {"payload_preprocessor"}


@pytest.mark.parametrize(
    ("type_name", "optional_keys", "new_optional_keys"),
    [
        (
            "OmniTextPrompt",
            {
                "negative_prompt",
                "modalities",
                "prompt_embeds",
                "negative_prompt_embeds",
                "additional_information",
            },
            {
                "negative_prompt",
                "modalities",
                "prompt_embeds",
                "negative_prompt_embeds",
                "additional_information",
            },
        ),
        (
            "OmniTokensPrompt",
            {
                "negative_prompt",
                "prompt_embeds",
                "negative_prompt_embeds",
                "additional_information",
            },
            {
                "negative_prompt",
                "prompt_embeds",
                "negative_prompt_embeds",
                "additional_information",
            },
        ),
        (
            "OmniTokenInputs",
            {
                "negative_prompt",
                "prompt_embeds",
                "negative_prompt_embeds",
                "additional_information",
            },
            {
                "negative_prompt",
                "prompt_embeds",
                "negative_prompt_embeds",
                "additional_information",
            },
        ),
        (
            "OmniEmbedsPrompt",
            {
                "prompt_embeds",
                "negative_prompt_embeds",
                "additional_information",
            },
            {"negative_prompt_embeds", "additional_information"},
        ),
    ],
)
def test_omni_prompt_types_keep_not_required_metadata(
    type_name: str, optional_keys: set[str], new_optional_keys: set[str]
) -> None:
    from vllm_omni.inputs import data

    prompt_type = getattr(data, type_name)
    assert optional_keys <= prompt_type.__optional_keys__
    assert new_optional_keys.isdisjoint(prompt_type.__required_keys__)
