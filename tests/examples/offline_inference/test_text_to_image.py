# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Offline inference tests: text-to-image.
See examples/offline_inference/text_to_image/README.md
"""

import json
import shlex
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download, snapshot_download

from tests.examples.helpers import EXAMPLES, ExampleRunner, ReadmeSnippet
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
    if "krea/Krea-2-Raw" in code:
        return True, "krea/Krea-2-Raw is gated on Hugging Face Hub; CI has no access"
    return False, ""


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH, skipif=_skip_readme_snippet)


def _prepare_anima_snippet(snippet: ReadmeSnippet) -> ReadmeSnippet:
    if "--model-class-name AnimaPipeline" not in snippet.code:
        return snippet

    assert "/path/to/models/anima-official/split_files/diffusion_models/anima-base-v1.0.safetensors" in snippet.code, (
        "Anima README checkpoint placeholder changed; update _prepare_anima_snippet to match"
    )

    checkpoint = hf_hub_download(
        repo_id="circlestone-labs/Anima",
        filename="split_files/diffusion_models/anima-base-v1.0.safetensors",
    )
    components = snapshot_download(
        repo_id="circlestone-labs/Anima-Base-v1.0-Diffusers",
        allow_patterns=["text_encoder/*", "vae/*", "tokenizer/*", "t5_tokenizer/*", "scheduler/*"],
    )
    argv = shlex.split(snippet.code)
    argv[argv.index("--model") + 1] = checkpoint
    custom_args_index = argv.index("--custom-pipeline-args") + 1
    custom_args = json.loads(argv[custom_args_index])
    custom_args["components_path"] = components
    argv[custom_args_index] = json.dumps(custom_args)
    return snippet._replace(code=shlex.join(argv))


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_text_to_image(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    snippet = _prepare_anima_snippet(snippet)
    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_image_valid(asset)
