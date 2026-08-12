# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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


def test_quantization_examples_keep_source_urls_under_feature_navigation(tmp_path, monkeypatch):
    nav_file = tmp_path / "docs" / ".nav.yml"
    nav_file.parent.mkdir(parents=True)
    nav_file.write_text(
        yaml.safe_dump(
            {
                "nav": [
                    {
                        "User Guide": [
                            {
                                "Examples": [
                                    "examples/README.md",
                                    {"Offline Inference": [{"Old": "user_guide/examples/offline_inference/old.md"}]},
                                ]
                            },
                            {"Features": [{"Quantization": [{"Overview": "user_guide/quantization/overview.md"}]}]},
                        ]
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(generate_examples, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(generate_examples, "NAV_FILE", nav_file)
    monkeypatch.setattr(generate_examples, "EXAMPLE_DOC_DIR", tmp_path / "docs/user_guide/examples")

    quantization_example = SimpleNamespace(
        category="quantization",
        path=tmp_path / "examples/quantization/check_modelopt_fp8_export.py",
        nav_title="Check Modelopt FP8 Export",
    )
    offline_example = SimpleNamespace(
        category="offline_inference",
        path=tmp_path / "examples/offline_inference/demo.py",
        nav_title="Demo",
    )

    generate_examples.update_nav_file([quantization_example, offline_example])
    generate_examples.update_nav_file([quantization_example, offline_example])

    nav_data = yaml.safe_load(nav_file.read_text(encoding="utf-8"))
    user_guide = nav_data["nav"][0]["User Guide"]
    examples = next(item["Examples"] for item in user_guide if "Examples" in item)
    features = next(item["Features"] for item in user_guide if "Features" in item)
    quantization = next(item["Quantization"] for item in features if "Quantization" in item)

    assert not any(isinstance(item, dict) and "Quantization" in item for item in examples)
    generated_examples = [
        item[generate_examples.GENERATED_QUANTIZATION_EXAMPLES_TITLE]
        for item in quantization
        if generate_examples.GENERATED_QUANTIZATION_EXAMPLES_TITLE in item
    ]
    assert len(generated_examples) == 1
    assert generated_examples[0] == [
        {"Check Modelopt FP8 Export": "user_guide/examples/quantization/check_modelopt_fp8_export.md"}
    ]

    # A later build without quantization examples must not leave a stale
    # generated group in the hand-authored Quantization section.
    generate_examples.update_nav_file([offline_example])
    nav_data = yaml.safe_load(nav_file.read_text(encoding="utf-8"))
    user_guide = nav_data["nav"][0]["User Guide"]
    features = next(item["Features"] for item in user_guide if "Features" in item)
    quantization = next(item["Quantization"] for item in features if "Quantization" in item)
    assert not any(
        isinstance(item, dict) and generate_examples.GENERATED_QUANTIZATION_EXAMPLES_TITLE in item
        for item in quantization
    )


def test_quantization_examples_are_not_added_to_examples_without_feature_target(tmp_path, monkeypatch, caplog):
    nav_file = tmp_path / "docs" / ".nav.yml"
    nav_file.parent.mkdir(parents=True)
    nav_file.write_text(
        yaml.safe_dump(
            {
                "nav": [
                    {
                        "User Guide": [
                            {"Examples": ["examples/README.md"]},
                        ]
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(generate_examples, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(generate_examples, "NAV_FILE", nav_file)
    monkeypatch.setattr(generate_examples, "EXAMPLE_DOC_DIR", tmp_path / "docs/user_guide/examples")

    quantization_example = SimpleNamespace(
        category="quantization",
        path=tmp_path / "examples/quantization/check_modelopt_fp8_export.py",
        nav_title="Check Modelopt FP8 Export",
    )

    generate_examples.update_nav_file([quantization_example])

    nav_data = yaml.safe_load(nav_file.read_text(encoding="utf-8"))
    examples = nav_data["nav"][0]["User Guide"][0]["Examples"]
    assert not any(isinstance(item, dict) and "Quantization" in item for item in examples)
    assert "omitting generated quantization examples from navigation" in caplog.text
