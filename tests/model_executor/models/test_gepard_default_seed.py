# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Gepard deploy configs must not pin a default ``seed``.

Gepard samples all 32 FSQ heads inside ``forward`` and #5666 threads the
request seed into a per-request ``torch.Generator``, so a ``seed`` in a
stage's ``default_sampling_params`` routes *every* request onto vLLM's
seeded sampling path (per-row ``torch.multinomial``) instead of the batched
path -- the throughput regression PR #4970 removed for ``qwen3_tts.yaml``
(issue #6178). Determinism is unaffected: a caller that wants it still passes
a per-request seed, which reaches the in-model sampler.

The cost only shows up under concurrency in slow GPU serving benchmarks, so
guard the config on CPU here instead.
"""

from pathlib import Path

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_DIR = Path(get_deploy_config_path("gepard.yaml")).parent


def _gepard_stages() -> list[tuple[str, int, dict]]:
    stages = []
    for name in sorted(path.name for path in DEPLOY_DIR.glob("gepard*.yaml")):
        config = yaml.safe_load(Path(get_deploy_config_path(name)).read_text())
        for index, stage in enumerate(config.get("stages") or []):
            stages.append((name, index, stage))
    return stages


def test_gepard_deploy_configs_exist() -> None:
    assert _gepard_stages(), f"no Gepard deploy configs found under {DEPLOY_DIR}"


@pytest.mark.parametrize(
    ("config_name", "stage_index", "stage"),
    _gepard_stages(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_gepard_stage_has_no_default_seed(config_name: str, stage_index: int, stage: dict) -> None:
    params = stage.get("default_sampling_params") or {}
    assert "seed" not in params, (
        f"{config_name} stage {stage_index} pins default_sampling_params.seed="
        f"{params.get('seed')!r}; a default seed forces every request onto the "
        "per-request seeded sampling path (see #6178, precedent #4970). Callers "
        "that want determinism should pass a per-request seed instead."
    )
