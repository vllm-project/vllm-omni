# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Every shipped DreamZero deploy config must select the AR-Diffusion engine.

``DreamZeroPipeline.__init__`` rejects any other engine, so a config missing
``engine_backend`` fails at worker startup rather than at request time. That
only surfaces in slow multi-GPU e2e serving tests (see issue #5046), so assert
it here on CPU instead.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.experimental.ar_diffusion.engine import ARDiffusionEngine

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
def test_dreamzero_stage_selects_ar_diffusion_engine(config_name: str, stage_index: int, stage: dict) -> None:
    backend = stage.get("engine_backend")
    assert backend, (
        f"{config_name} stage {stage_index} does not set engine_backend; "
        "DreamZeroPipeline requires the AR-Diffusion engine"
    )
    # Resolving also proves the import path is real, not just a lookalike string.
    engine_class = DiffusionEngine.resolve_engine_class(SimpleNamespace(engine_backend=backend))
    assert issubclass(engine_class, ARDiffusionEngine), (
        f"{config_name} stage {stage_index} resolves to {engine_class.__name__}, not an AR-Diffusion engine"
    )
