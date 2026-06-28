"""VACE coverage through the shared text-to-video and image-to-video examples."""

from pathlib import Path

import pytest
from PIL import Image

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet
from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.usefixtures("clean_gpu_memory_between_tests"),
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.example,
    *hardware_marks(res={"cuda": "H100"}),
]

TEXT_TO_VIDEO_README = EXAMPLES / "offline_inference" / "text_to_video" / "text_to_video.md"
IMAGE_TO_VIDEO_README = EXAMPLES / "offline_inference" / "image_to_video" / "README.md"
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_vace"

TEXT_TO_VIDEO_SNIPPET = next(
    snippet
    for snippet in ReadmeSnippet.extract_readme_snippets(TEXT_TO_VIDEO_README)
    if "vace_t2v_output.mp4" in snippet.code
)
CONDITIONAL_OUTPUTS = (
    "vace_i2v_output.mp4",
    "vace_v2lf_output.mp4",
    "vace_flf2v_output.mp4",
    "vace_inpaint_output.mp4",
    "vace_r2v_output.mp4",
)
IMAGE_TO_VIDEO_SNIPPETS = [
    snippet
    for snippet in ReadmeSnippet.extract_readme_snippets(IMAGE_TO_VIDEO_README)
    if snippet.h2_title == "Wan2.1 VACE Conditional Tasks"
    and any(output in snippet.code for output in CONDITIONAL_OUTPUTS)
]


def _smoke_snippet(snippet: ReadmeSnippet, asset_dir: Path) -> ReadmeSnippet:
    # Six fresh model launches share a 90-minute X2V job. Keep the documented
    # quality settings intact while reducing only the CI executions.
    replacements = {
        "astronaut.jpg": str(asset_dir / "astronaut.jpg"),
        "vace_first_frame.png": str(asset_dir / "vace_first_frame.png"),
        "vace_last_frame.png": str(asset_dir / "vace_last_frame.png"),
        "vace_center_mask.png": str(asset_dir / "vace_center_mask.png"),
        "--num-frames 81": "--num-frames 5",
        "--num-inference-steps 30": "--num-inference-steps 2",
    }
    code = snippet.code
    for source, target in replacements.items():
        code = code.replace(source, target)
    return snippet._replace(code=code)


@pytest.fixture
def vace_asset_dir(tmp_path: Path) -> Path:
    asset_dir = tmp_path / "vace_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (832, 480), (180, 180, 180)).save(asset_dir / "astronaut.jpg")
    Image.new("RGB", (512, 512), (80, 140, 220)).save(asset_dir / "vace_first_frame.png")
    Image.new("RGB", (512, 512), (220, 140, 80)).save(asset_dir / "vace_last_frame.png")
    mask = Image.new("L", (832, 480), 0)
    mask.paste(255, (336, 0, 496, 480))
    mask.save(asset_dir / "vace_center_mask.png")
    return asset_dir


def test_vace_text_to_video_shared_example(example_runner: ExampleRunner) -> None:
    snippet = TEXT_TO_VIDEO_SNIPPET._replace(
        code=TEXT_TO_VIDEO_SNIPPET.code.replace("--num-frames 81", "--num-frames 5").replace(
            "--num-inference-steps 30", "--num-inference-steps 2"
        )
    )
    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_video_valid(asset)


@pytest.mark.parametrize("snippet", IMAGE_TO_VIDEO_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_vace_conditional_shared_examples(
    snippet: ReadmeSnippet,
    example_runner: ExampleRunner,
    vace_asset_dir: Path,
) -> None:
    snippet = _smoke_snippet(snippet, vace_asset_dir)
    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_video_valid(asset)
