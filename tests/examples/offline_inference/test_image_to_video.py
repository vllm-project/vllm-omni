"""
Offline inference tests: image-to-video.
See examples/offline_inference/image_to_video/README.md
"""

import shutil
from pathlib import Path

import pytest
import requests

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet
from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.usefixtures("clean_gpu_memory_between_tests"),
    pytest.mark.full_model,
    pytest.mark.example,
    *hardware_marks(res={"cuda": "H100"}),
]

I2V_SCRIPT = EXAMPLES / "offline_inference" / "image_to_video" / "image_to_video.py"
README_PATH = I2V_SCRIPT.with_name("README.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_i2v"

_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
_IMAGE_NAME = "cherry_blossom.jpg"

_SKIP_SECTIONS = {
    "Prerequisites",
    "Advanced Features",
    "FAQ",
    # VACE conditional tasks are covered by test_vace_video_generation.py, which
    # runs the same README snippets with synthetic assets and smoke settings. The
    # prep block here downloads inputs via wget (not portable, not collected), so
    # exercising this section from the shared runner would only duplicate and fail.
    "Wan2.1 VACE Conditional Tasks",
}


def _ensure_test_image(run_dir: Path) -> None:
    """Download or copy the example image into run_dir so CLI snippets can find it."""
    dest = run_dir / _IMAGE_NAME
    if dest.exists():
        return
    src = I2V_SCRIPT.parent / _IMAGE_NAME
    if src.exists():
        shutil.copy2(src, dest)
        return
    response = requests.get(_IMAGE_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)


def _skip_readme_snippet(language: str, code: str, h2_title: str) -> tuple[bool, str]:
    if h2_title in _SKIP_SECTIONS:
        return True, f"README section '{h2_title}' is intentionally excluded for examples tests"
    if language == "python":
        return True, "Python API snippets produce video files that ExampleRunner does not auto-collect"
    if "/path/to/" in code:
        return True, "Snippet references a placeholder local model path"
    return False, ""


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH, skipif=_skip_readme_snippet)


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_image_to_video(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    run_dir = example_runner.output_root / EXAMPLE_OUTPUT_SUBFOLDER / snippet.test_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_test_image(run_dir)

    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_video_valid(asset)
