"""
Offline inference tests: text-to-video.

See examples/offline_inference/text_to_video/text_to_video.md
"""

from pathlib import Path

import pytest

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet
from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.example,
    *hardware_marks(res={"cuda": "H100"}),
]

T2V_SCRIPT = EXAMPLES / "offline_inference" / "text_to_video" / "text_to_video.py"
README_PATH = T2V_SCRIPT.with_name("text_to_video.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_t2v"
VIDEO_SUFFIXES = {".mp4"}


def _skip_readme_snippet(language: str, code: str, h2_title: str) -> tuple[bool, str]:
    if language == "python":
        return True, "Python snippets are intentionally excluded for text-to-video example tests"

    if language == "bash" and "--output quick_test.mp4" not in code:
        return True, "Only the lightweight Quick Test CLI snippet is exercised in examples tests"

    return False, ""


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(
    README_PATH,
    skipif=_skip_readme_snippet,
)
assert any(not snippet.skip[0] for snippet in README_SNIPPETS), (
    "Expected at least one runnable text-to-video README snippet. "
    "Update _skip_readme_snippet if the Quick Test example changes."
)


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_text_to_video(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    result = example_runner.run(
        snippet,
        output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER),
    )

    assert result.assets, "Expected at least one generated video asset"

    for asset in result.assets:
        assert asset.suffix.lower() in VIDEO_SUFFIXES, f"Unexpected video asset suffix: {asset.suffix}"
        assert asset.exists(), f"Video asset not found: {asset}"
        assert asset.stat().st_size > 0, f"Video asset is empty: {asset}"
        assert_video_valid(asset)
