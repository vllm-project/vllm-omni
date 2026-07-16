# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest

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
