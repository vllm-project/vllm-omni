# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Offline inference tests: image-to-video.
See examples/offline_inference/image_to_video/README.md
"""

import shutil
import socket
import time
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
    *hardware_marks(res={"cuda": ["H100", "B200"]}),
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
    for attempt in range(3):
        try:
            with requests.get(_IMAGE_URL, timeout=60) as response:
                response.raise_for_status()
                content = response.content
        except requests.ConnectionError as error:
            # requests wraps urllib3's DNS error; follow the exception chain to
            # distinguish a temporary resolver outage from permanent failures.
            cause: BaseException | None = error
            while cause is not None and not isinstance(cause, socket.gaierror):
                cause = cause.__cause__ or cause.__context__
            if not isinstance(cause, socket.gaierror) or cause.errno != socket.EAI_AGAIN or attempt == 2:
                raise
            time.sleep(2**attempt)
        else:
            dest.write_bytes(content)
            return


@pytest.fixture(scope="session")
def i2v_test_image_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Share one README asset download while giving each CLI its own copy."""
    return tmp_path_factory.mktemp("i2v_readme_image")


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
def test_image_to_video(snippet: ReadmeSnippet, example_runner: ExampleRunner, i2v_test_image_dir: Path):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    run_dir = example_runner.output_root / EXAMPLE_OUTPUT_SUBFOLDER / snippet.test_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_test_image(i2v_test_image_dir)
    shutil.copy2(i2v_test_image_dir / _IMAGE_NAME, run_dir / _IMAGE_NAME)

    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_video_valid(asset)
