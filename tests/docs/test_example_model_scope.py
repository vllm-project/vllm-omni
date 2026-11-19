# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Doc-test model scope must exclude weights before any GPU fixture starts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.examples.model_scope import doc_item_models, doc_models_allowed, models_in_snippet

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "model",
    [
        "openbmb/MiniCPM-o-4_5",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen-Image-2.1",
        "Qwen/Qwen-Image-2.1-Turbo",
        "tencent/HunyuanImage-3.0-Instruct",
        "tencent/HunyuanImage-3.0-Instruct-Distil",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "MiniMaxAI/MiniMax-H3",
    ],
)
def test_requested_model_versions_are_allowed(model):
    assert doc_models_allowed((model,))


@pytest.mark.parametrize(
    "model",
    [
        "openbmb/MiniCPM-o-2_6",
        "Qwen/Qwen2.5-Omni-7B",
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-2512",
        "tencent/HunyuanImage-2.1",
        "hunyuanvideo-community/HunyuanVideo-1.50-Diffusers",
        "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "Tongyi-MAI/Z-Image-Turbo",
        "circlestone-labs/Anima",
        "black-forest-labs/FLUX.2-dev",
        "Lightricks/LTX-2",
        "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "MiniMaxAI/MiniMax-H3-Image",
        "/path/to/models/checkpoint.safetensors",
    ],
)
def test_other_models_and_versions_are_excluded(model):
    assert not doc_models_allowed((model,))


@pytest.mark.parametrize(
    "language, code",
    [
        ("bash", "python run.py --model=Qwen/Qwen-Image-2.1"),
        ("bash", "vllm serve Qwen/Qwen-Image-2.1 --omni"),
        ("bash", "MODEL=Qwen/Qwen-Image-2.1 bash run_server.sh"),
        ("python", 'omni = Omni(model="Qwen/Qwen-Image-2.1")'),
        ("python", 'name = "Qwen/Qwen-Image-2.1"\nomni = Omni(model=name)'),
        ("python", 'omni = Omni("Qwen/Qwen-Image-2.1")'),
    ],
)
def test_explicit_shell_and_python_models(language, code):
    assert models_in_snippet(language, code) == ("Qwen/Qwen-Image-2.1",)


@pytest.mark.parametrize(
    "language, code",
    [
        ("bash", "python run.py --prompt hello"),
        ("bash", "python run.py --model"),
        ("bash", 'python run.py --model "$MODEL"'),
        ("python", "omni = Omni(model=choose_model())"),
        ("python", 'name = "Qwen/Qwen-Image-2.1"\nname = choose_model()\nomni = Omni(model=name)'),
        ("python", 'omni = Omni(model="Qwen/Qwen-Image-2.1"'),
        ("bash", 'python run.py --model "unterminated'),
    ],
)
def test_unknown_or_dynamic_models_fail_closed(language, code):
    assert not doc_models_allowed(models_in_snippet(language, code))


def test_every_model_in_a_block_must_be_allowed():
    code = "python run.py --model Qwen/Qwen-Image-2.1 && python run.py --model Tongyi-MAI/Z-Image-Turbo"
    assert not doc_models_allowed(models_in_snippet("bash", code))


def test_readme_filter_runs_before_model_specific_preparation(tmp_path, monkeypatch):
    from tests.examples import helpers

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("excluded model must not create LoRA assets")

    monkeypatch.setattr(helpers, "write_zimage_lora", unexpected_preparation)
    (tmp_path / "run.py").write_text("", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Qwen-Image 2.1\n```bash\npython missing.py --model Tongyi-MAI/Z-Image-Turbo --lora-path fake\n```\n"
        "```bash\npython run.py --model Qwen/Qwen-Image-2.1 --output image.png\n```\n",
        encoding="utf-8",
    )
    snippets = helpers.ReadmeSnippet.extract_readme_snippets(readme)
    assert len(snippets) == 1
    assert "Qwen/Qwen-Image-2.1" in snippets[0].code
    assert snippets[0].index_in_section == 2


def test_gpu_collection_filters_server_parameters_but_preserves_cpu_tests():
    from tests.examples.conftest import pytest_collection_modifyitems

    def item(markers, params):
        return SimpleNamespace(
            get_closest_marker=lambda name: name in markers,
            callspec=SimpleNamespace(params=params),
        )

    supported = item({"full_model", "example"}, {"omni_server": SimpleNamespace(model="Qwen/Qwen-Image-2.1")})
    unsupported = item({"full_model", "example"}, {"omni_server": SimpleNamespace(model="Qwen/Qwen-Image")})
    unknown = item({"full_model"}, {})
    cpu = item({"core_model", "cpu", "example"}, {})
    items = [supported, unsupported, unknown, cpu]
    deselected = []
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    pytest_collection_modifyitems(config, items)
    assert items == [supported, cpu]
    assert deselected == [unsupported, unknown]
    assert doc_item_models({"omni_server": SimpleNamespace(model="Qwen/Qwen-Image-2.1")}) == ("Qwen/Qwen-Image-2.1",)


def test_real_readme_matrices_are_nonempty_and_scoped():
    from tests.examples.helpers import ReadmeSnippet

    for name in ("text_to_image", "image_to_video"):
        snippets = ReadmeSnippet.extract_readme_snippets(ROOT / "examples/offline_inference" / name / "README.md")
        assert snippets, name
        for snippet in snippets:
            assert doc_models_allowed(models_in_snippet(snippet.language, snippet.code)), snippet
    image_snippets = ReadmeSnippet.extract_readme_snippets(ROOT / "examples/offline_inference/text_to_image/README.md")
    assert "Qwen/Qwen-Image-2.1" in {
        model for snippet in image_snippets for model in models_in_snippet(snippet.language, snippet.code)
    }


def test_nightly_doc_jobs_and_dependencies_match_scope():
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-nightly.yml").read_text(encoding="utf-8"))
    jobs = [job for group in pipeline["steps"] for job in group.get("steps", []) if "Doc Test" in job.get("label", "")]
    assert jobs
    assert not any(job.get("source_file_dependencies") == "omni_qwen2_5_omni_doc" for job in jobs)
    amd = yaml.safe_load((ROOT / ".buildkite/amd/test-amd-nightly.yml").read_text(encoding="utf-8"))
    assert not any(
        "Qwen2.5-Omni Documentation" in job.get("label", "") for group in amd["steps"] for job in group.get("steps", [])
    )
    omni = next(job for job in jobs if "omni_qwen3_omni_doc" == job.get("source_file_dependencies"))
    assert "tests/examples/online_serving/test_qwen3_omni.py" in "\n".join(omni["commands"])
    dependencies = yaml.safe_load(
        (ROOT / ".buildkite/common/ci_source_file_dependencies.yml").read_text(encoding="utf-8")
    )

    # Locate keyed dependencies independently of the registry wrapper name.
    def find_mapping(value):
        if isinstance(value, dict):
            if "diffusion_text_to_image_doc" in value:
                return value
            for child in value.values():
                found = find_mapping(child)
                if found is not None:
                    return found
        return None

    mapping = find_mapping(dependencies)
    assert mapping is not None
    for key in ("omni_qwen3_omni_doc", "diffusion_text_to_image_doc", "diffusion_image_to_video_doc"):
        assert "tests/examples/model_scope.py" in mapping[key]
