# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import pytest

from examples.offline_inference.diffusion.lingbot_world_v2_realtime import (
    _load_events,
    _validate_args,
    parse_args,
)
from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EXAMPLE_DIR = Path(__file__).parents[3] / "examples" / "offline_inference" / "diffusion"
DEPLOY_CONFIG = EXAMPLE_DIR / "lingbot_world_v2_replica_dp_tp2.yaml"
EVENTS = EXAMPLE_DIR / "lingbot_world_v2_realtime_events.jsonl"


def _args(tmp_path: Path, *extra: str):
    image = tmp_path / "input.png"
    image.touch()
    return parse_args(
        [
            "--image",
            str(image),
            "--prompt",
            "drive forward",
            "--events",
            str(EVENTS),
            "--output-dir",
            str(tmp_path / "output"),
            *extra,
        ]
    )


def test_replica_dp_recipe_is_two_replicas_with_tp2() -> None:
    deploy = load_deploy_config(DEPLOY_CONFIG)

    assert deploy.distributed_executor_backend == "mp"
    assert len(deploy.stages) == 1
    stage = deploy.stages[0]
    assert stage.devices == "0,1,2,3"
    assert stage.num_replicas == 2
    assert stage.tensor_parallel_size == 2
    assert stage.max_num_seqs == 1


def test_multisession_cli_accepts_checked_in_recipe(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        "--deploy-config",
        str(DEPLOY_CONFIG),
        "--num-sessions",
        "2",
        "--require-distinct-replicas",
    )

    _, events, _, deploy_config = _validate_args(args)

    assert events == EVENTS.resolve()
    assert deploy_config == DEPLOY_CONFIG.resolve()
    assert len(_load_events(events)) >= 2


def test_distinct_replica_check_requires_multiple_sessions(tmp_path: Path) -> None:
    args = _args(tmp_path, "--require-distinct-replicas")

    with pytest.raises(ValueError, match="at least two sessions"):
        _validate_args(args)
