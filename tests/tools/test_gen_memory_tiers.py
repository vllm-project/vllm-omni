# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contracts for the documented minimum-memory index."""

import json

import pytest
import yaml

from tools import gen_memory_tiers as cli

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_recipe(root, vendor, name, variants, tasks=None):
    path = root / vendor / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "meta": {"title": name, "provider": vendor, "tasks": ["omni"]},
                "model": {"model_id": f"{vendor}/{name}"},
                "omni": {"tasks": tasks if tasks is not None else [{"id": "t2a", "label": "Text to speech"}]},
                "variants": variants,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def recipes(tmp_path):
    models = tmp_path / "recipes" / "models"
    _write_recipe(models, "acme", "big", {"default": {"precision": "bf16", "vram_minimum_gb": 24}})
    _write_recipe(
        models,
        "acme",
        "small",
        {
            "default": {"precision": "bf16", "vram_minimum_gb": 8},
            # A variant without a documented minimum must not invent one.
            "unmeasured": {"precision": "fp8"},
        },
        tasks=[{"id": "t2i"}, {"id": "i2i"}],
    )
    return models


def test_load_variants_sorts_by_memory_and_skips_undocumented(recipes):
    variants, sources = cli.load_variants(recipes)

    assert len(sources) == 2
    assert [(variant.model_id, variant.minimum_gib) for variant in variants] == [
        ("acme/small", 8),
        ("acme/big", 24),
    ]
    assert variants[0].tasks == ("Text-to-image", "Image-to-image")
    assert variants[0].source == "recipes/models/acme/small.yaml"


def test_render_groups_variants_into_the_first_tier_that_fits(recipes):
    variants, sources = cli.load_variants(recipes)
    page = cli.render(variants, len(sources), markdown_recipe_count=0)

    eight = page.split("### 8 GiB")[1].split("### 12 GiB")[0]
    sixteen = page.split("### 16 GiB")[1].split("### 24 GiB")[0]
    twenty_four = page.split("### 24 GiB")[1].split("## All documented minimums")[0]

    assert "`acme/small`" in eight
    assert "`acme/big`" not in eight
    # The 16 GiB band is empty for this fixture, so it only reports the backlog.
    assert "`acme/small`" not in sixteen
    assert "No variant documents a minimum in this band" in sixteen
    assert "`acme/big`" in twenty_four
    assert "`acme/small`" not in twenty_four


def test_render_keeps_the_unmeasured_wording(recipes):
    variants, sources = cli.load_variants(recipes)
    page = cli.render(variants, len(sources), markdown_recipe_count=0)

    assert "not yet measured" in page
    assert "unsupported" in page
    assert "acme/small" in page


def _run_cli(recipes, tmp_path, *extra):
    return cli.main(
        [
            "--recipes-dir",
            str(recipes),
            "--output",
            str(tmp_path / "memory_tiers.md"),
            "--json-output",
            str(tmp_path / "memory_tiers.json"),
            *extra,
        ]
    )


def test_cli_emits_both_artifacts_and_json(recipes, tmp_path, capsys):
    assert _run_cli(recipes, tmp_path) == 0
    assert "acme/small" in (tmp_path / "memory_tiers.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "memory_tiers.json").read_text(encoding="utf-8"))[0]["minimum_gib"] == 8

    assert _run_cli(recipes, tmp_path, "--check") == 0
    capsys.readouterr()

    assert _run_cli(recipes, tmp_path, "--format", "json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert [entry["minimum_gib"] for entry in payload] == [8, 24]
    assert payload[0]["source"] == "recipes/models/acme/small.yaml"


@pytest.mark.parametrize("artifact", ["memory_tiers.md", "memory_tiers.json"])
def test_check_flags_a_stale_artifact(recipes, tmp_path, artifact):
    assert _run_cli(recipes, tmp_path) == 0
    (tmp_path / artifact).write_text("stale", encoding="utf-8")

    assert _run_cli(recipes, tmp_path, "--check") == 1


def test_check_flags_a_missing_artifact(recipes, tmp_path):
    assert _run_cli(recipes, tmp_path, "--check") == 1


def test_checked_in_artifacts_match_the_rendered_output():
    """The docs page and the packaged index must track recipes/models/."""
    page, variants = cli.build()

    assert cli.DEFAULT_OUTPUT.read_text(encoding="utf-8") == page
    assert cli.DEFAULT_JSON_OUTPUT.read_text(encoding="utf-8") == cli.render_json(variants)
