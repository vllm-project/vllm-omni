# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


ROOT_DIR = Path(__file__).parents[2]
HOOK_PATH = ROOT_DIR / "docs/mkdocs/hooks/generate_examples.py"
SPEC = importlib.util.spec_from_file_location("generate_examples", HOOK_PATH)
assert SPEC and SPEC.loader
generate_examples = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_examples)


def test_model_example_titles_match_shared_display_names():
    for slug, display_name in generate_examples.MODEL_DISPLAY_NAMES.items():
        for category, mode_title in generate_examples.SERVING_MODE_TITLES.items():
            example_path = ROOT_DIR / "examples" / category / slug
            example = generate_examples.Example(example_path, category)

            assert example.title == f"{display_name}: {mode_title}"
            assert example.nav_title == display_name


def test_model_example_title_mismatch_is_rejected(tmp_path):
    example_path = tmp_path / "mimo_audio"
    example_path.mkdir()
    (example_path / "README.md").write_text("# A different title\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Model example title mismatch"):
        generate_examples.Example(example_path, "offline_inference")


def test_model_display_names_require_string_mapping(tmp_path, monkeypatch):
    display_names_path = tmp_path / "model_display_names.yml"
    display_names_path.write_text("- not a mapping\n", encoding="utf-8")
    monkeypatch.setattr(generate_examples, "MODEL_DISPLAY_NAMES_FILE", display_names_path)

    with pytest.raises(ValueError, match="MODEL_DISPLAY_NAMES_FILE"):
        generate_examples.load_model_display_names()


def test_only_general_serving_examples_get_generated_pages():
    general_example = SimpleNamespace(category="offline_inference", path=Path("text_to_image"))
    model_example = SimpleNamespace(category="offline_inference", path=Path("qwen3_omni"))

    assert generate_examples.is_general_example(general_example)
    assert not generate_examples.is_general_example(model_example)


@pytest.mark.parametrize(
    "code, expected",
    [
        ('Omni(model="Qwen/Qwen-Image-2.1")', True),
        ('model = "Qwen/Qwen-Image-2.1"\nOmni(model=model)', True),
        ('model = "Tongyi-MAI/Z-Image-Turbo"\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nmodel = "other/model"\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nmodel: str = "other/model"\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nmodel += "-unsupported"\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nmodel, other = ("other/model", 0)\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nfor model in ["other/model"]: pass\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\n(model := "other/model")\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\ndel model\nOmni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nmatch "other/model":\n    case model:\n        Omni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nmatch ["other/model"]:\n    case [*model]:\n        Omni(model=model)', False),
        ('model = "Qwen/Qwen-Image-2.1"\nmatch {"x": 1}:\n    case {**model}:\n        Omni(model=model)', False),
        ("model = get_model()\nOmni(model=model)", False),
        ('Omni(model="Qwen/Qwen-Image-2.1")\nOmni(model="other/model")', False),
    ],
)
def test_doc_scope_python_bindings(code, expected):
    from tests.examples.helpers import doc_models_allowed, models_in_snippet

    assert doc_models_allowed(models_in_snippet("python", code)) is expected


def test_doc_scope_filters_before_preparation(tmp_path, monkeypatch):
    from tests.examples.helpers import ReadmeSnippet

    readme = tmp_path / "README.md"
    readme.write_text(
        "```bash\npython missing.py --model other/model --lora-path missing\n```\n"
        "```bash\npython missing.py --model Qwen/Qwen3-Omni-${SIZE} --lora-path missing\n```\n"
        '```python\nOmni(model="Qwen/Qwen-Image-2.1")\n```\n',
        encoding="utf-8",
    )

    def unexpected_preparation(*args):
        pytest.fail("Excluded model reached command preparation")

    monkeypatch.setattr(ReadmeSnippet, "_normalize_bash_command", unexpected_preparation)
    snippets = ReadmeSnippet.extract_readme_snippets(readme)
    assert len(snippets) == 1
    assert snippets[0].language == "python"


def test_doc_scope_collection_preserves_cpu_examples():
    from tests.examples.conftest import pytest_collection_modifyitems
    from tests.examples.helpers import ReadmeSnippet

    def item(code, *, cpu=False):
        return SimpleNamespace(
            get_closest_marker=lambda mark: mark == "full_model" or (mark == "cpu" and cpu),
            callspec=SimpleNamespace(params={"snippet": ReadmeSnippet("python", code, "Example", 1)}),
        )

    supported = item('Omni(model="Qwen/Qwen-Image-2.1")')
    excluded = item('Omni(model="other/model")')
    mutated = item('model = "Qwen/Qwen-Image-2.1"\nmodel: str = "other/model"\nOmni(model=model)')
    cpu = item('Omni(model="other/model")', cpu=True)
    unknown = item("Omni(model=get_model())")
    items = [supported, excluded, mutated, cpu, unknown]
    deselected = []
    config = SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items)))
    pytest_collection_modifyitems(config, items)
    assert items == [supported, cpu]
    assert deselected == [excluded, mutated, unknown]


def test_doc_scope_readmes_and_ci_dependencies():
    import yaml

    from tests.examples.helpers import ReadmeSnippet, doc_models_allowed, models_in_snippet

    for name in ("text_to_image", "image_to_video"):
        snippets = ReadmeSnippet.extract_readme_snippets(ROOT_DIR / "examples/offline_inference" / name / "README.md")
        assert snippets
        assert all(doc_models_allowed(models_in_snippet(s.language, s.code)) for s in snippets)
    pipeline = yaml.safe_load((ROOT_DIR / ".buildkite/cuda/test-nightly.yml").read_text(encoding="utf-8"))
    jobs = [job for group in pipeline["steps"] for job in group.get("steps", []) if "Doc Test" in job.get("label", "")]
    assert jobs
    assert not any(job.get("source_file_dependencies") == "omni_qwen2_5_omni_doc" for job in jobs)
    dependencies = yaml.safe_load(
        (ROOT_DIR / ".buildkite/common/ci_source_file_dependencies.yml").read_text(encoding="utf-8")
    )

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
    for job in jobs:
        paths = mapping[job["source_file_dependencies"]]
        assert "tests/examples/helpers.py" in paths
        assert "tests/examples/conftest.py" in paths
