# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ensure diffusion stage YAML configs only use valid OmniDiffusionConfig fields.

Regression test for https://github.com/vllm-project/vllm-omni/issues/2563
"""

from dataclasses import fields
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

try:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
except Exception:
    OmniDiffusionConfig = None


@pytest.mark.skipif(
    OmniDiffusionConfig is None,
    reason="OmniDiffusionConfig could not be imported (missing torch?)",
)
def test_diffusion_stage_configs_only_contain_valid_fields():
    """Diffusion stage engine_args must only contain OmniDiffusionConfig fields.

    Regression test for https://github.com/vllm-project/vllm-omni/issues/2563
    """
    # Scan both main configs and test configs
    repo_root = Path(__file__).parent.parent
    config_dirs = [
        repo_root / "vllm_omni" / "model_executor" / "stage_configs",
    ]
    # Also scan test directories recursively
    test_dir = repo_root / "tests"

    yaml_paths: list[Path] = []
    for config_dir in config_dirs:
        yaml_paths.extend(sorted(config_dir.glob("*.yaml")))
    yaml_paths.extend(sorted(test_dir.rglob("*.yaml")))

    valid_fields = {f.name for f in fields(OmniDiffusionConfig)}
    # model_stage is consumed by the stage init layer, not OmniDiffusionConfig
    valid_fields.add("model_stage")
    # model_arch is consumed by the stage init layer for diffusion model class resolution
    valid_fields.add("model_arch")
    # "quantization" is mapped to "quantization_config" by from_kwargs() backwards-compat
    valid_fields.add("quantization")

    invalid_entries: list[tuple[str, set[str]]] = []
    for yaml_path in yaml_paths:
        with open(yaml_path) as fh:
            config = yaml.safe_load(fh)

        stages = config.get("stage_args", config.get("stages", []))
        for stage in stages:
            if stage.get("stage_type") != "diffusion":
                continue
            engine_args = stage.get("engine_args", {})
            invalid = set(engine_args.keys()) - valid_fields
            if invalid:
                invalid_entries.append((yaml_path.relative_to(repo_root), invalid))

    assert not invalid_entries, "Diffusion stage configs contain fields not in OmniDiffusionConfig:\n" + "\n".join(
        f"  {name}: {sorted(bad)}" for name, bad in invalid_entries
    )
