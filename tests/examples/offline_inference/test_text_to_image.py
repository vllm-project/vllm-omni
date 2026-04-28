"""
Offline inference tests: text-to-image.
See examples/offline_inference/text_to_image/README.md
"""

from pathlib import Path

import pytest

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet
from tests.helpers.assertions import assert_image_valid
from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.full_model, pytest.mark.example, *hardware_marks(res={"cuda": "H100"})]


T2I_SCRIPT = EXAMPLES / "offline_inference" / "text_to_image" / "text_to_image.py"
README_PATH = T2I_SCRIPT.with_name("README.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_t2i"


def _skip_readme_snippet(language: str, code: str, h2_title: str) -> tuple[bool, str]:
    if h2_title == "Web UI Demo":
        return True, f"README section '{h2_title}' is intentionally excluded for examples tests"
    return False, ""


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH, skipif=_skip_readme_snippet)

# Track: https://github.com/vllm-project/vllm-omni/issues/3123 — CLI example subprocess exit 1 in CI; re-enable when fixed.
_SKIP_T2I_SNIPPET_IDS = frozenset(
    {
        "quick_start_002",
        "more_cli_examples_001",
        "more_cli_examples_002",
        "more_cli_examples_003",
        "more_cli_examples_007",
        "more_cli_examples_008",
    }
)
_SKIP_T2I_REASON = (
    "Skipped: text_to_image README examples failing in CI (subprocess exit 1). "
    "See https://github.com/vllm-project/vllm-omni/issues/3123"
)


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_text_to_image(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    if snippet.test_id in _SKIP_T2I_SNIPPET_IDS:
        pytest.skip(_SKIP_T2I_REASON)
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_image_valid(asset)
