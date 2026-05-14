"""
Offline inference tests: vace video generation.
See examples/offline_inference/vace/vace_video_generation.md
"""

from pathlib import Path

import pytest

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet
from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.usefixtures("clean_gpu_memory_between_tests"),
    pytest.mark.full_model,
    pytest.mark.example,
    *hardware_marks(res={"cuda": "H100"}),
]

VACE_SCRIPT = EXAMPLES / "offline_inference" / "vace" / "vace_video_generation.py"
README_PATH = VACE_SCRIPT.with_name("vace_video_generation.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_vace"

README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH)


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_vace_video_generation(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)
    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_video_valid(asset)
