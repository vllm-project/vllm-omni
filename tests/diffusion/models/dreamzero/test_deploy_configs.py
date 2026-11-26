# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Keep shipped DreamZero deploys on the default engine and AR runner."""

from pathlib import Path

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.experimental.ar_diffusion.engine import AR_DIFFUSION_MODEL_RUNNER_CLS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Discover siblings of a known config so a future dreamzero*.yaml is covered
# without editing this test.
DEPLOY_DIR = Path(get_deploy_config_path("dreamzero.yaml")).parent


def _dreamzero_stages() -> list[tuple[str, int, dict]]:
    stages = []
    for name in sorted(path.name for path in DEPLOY_DIR.glob("dreamzero*.yaml")):
        config = yaml.safe_load(Path(get_deploy_config_path(name)).read_text())
        for index, stage in enumerate(config.get("stages") or []):
            stages.append((name, index, stage))
    return stages


def test_dreamzero_deploy_configs_exist() -> None:
    assert _dreamzero_stages(), f"no DreamZero deploy configs found under {DEPLOY_DIR}"


@pytest.mark.parametrize(
    ("config_name", "stage_index", "stage"),
    _dreamzero_stages(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_dreamzero_stage_selects_ar_diffusion_runner(config_name: str, stage_index: int, stage: dict) -> None:
    assert stage.get("engine_backend", "default") == "default", (config_name, stage_index)
    assert stage.get("diffusion_model_runner_cls") == AR_DIFFUSION_MODEL_RUNNER_CLS, (config_name, stage_index)
