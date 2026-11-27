# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Offline inference tests: text-to-image.
See examples/offline_inference/text_to_image/README.md
"""

from pathlib import Path

import pytest

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet, models_in_snippet
from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.usefixtures("clean_gpu_memory_between_tests"),
    pytest.mark.full_model,
    pytest.mark.example,
    *hardware_marks(res={"cuda": ["H100", "B200"]}),
]

T2I_SCRIPT = EXAMPLES / "offline_inference" / "text_to_image" / "text_to_image.py"
README_PATH = T2I_SCRIPT.with_name("README.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_t2i"


def _skip_readme_snippet(language: str, code: str, h2_title: str) -> tuple[bool, str]:
    if h2_title == "Web UI Demo":
        return True, f"README section '{h2_title}' is intentionally excluded for examples tests"
    return False, ""


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH, skipif=_skip_readme_snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param(
            snippet,
            id=snippet.test_id,
            marks=hardware_marks(res={"rocm": "MI325"})
            if models_in_snippet(snippet.language, snippet.code) == ("Qwen/Qwen-Image-2.1",)
            else (),
        )
        for snippet in README_SNIPPETS
    ],
)
def test_text_to_image(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_image_valid(asset)
