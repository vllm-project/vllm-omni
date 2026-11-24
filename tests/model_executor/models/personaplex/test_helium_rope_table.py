# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helium refuses a context longer than its RoPE table."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import build_engine_args_dict_from_omni_stage_config
from vllm_omni.model_executor.models.personaplex.modeling_helium import HeliumModel, check_rope_covers_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_context_within_the_rope_table_is_accepted() -> None:
    check_rope_covers_context(46_080, 46_080)
    check_rope_covers_context(4096, 46_080)


def test_context_past_the_rope_table_is_refused() -> None:
    # Raising only Stage 0's max_model_len would let a call read rotations past the table.
    with pytest.raises(ValueError, match="temporal_config.max_position_embeddings"):
        check_rope_covers_context(90_000, 46_080)


@pytest.mark.parametrize(
    ("cli_overrides", "stage_overrides", "resizes_stage1"),
    [
        pytest.param({}, {"0": {"max_model_len": 90_000}}, False, id="stage-override"),
        pytest.param({"max_model_len": 90_000}, None, True, id="global-max-model-len"),
    ],
)
def test_helium_refuses_a_deploy_context_raised_without_its_rope_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_overrides: dict[str, int],
    stage_overrides: dict[str, dict[str, int]] | None,
    resizes_stage1: bool,
) -> None:
    """Both ways of raising Stage 0's context alone reach HeliumModel and are
    refused: ``--stage-overrides`` for Stage 0, and the global
    ``--max-model-len``, which also resizes Stage 1."""
    # vllm-omni sets this for any explicit max_model_len, so vLLM itself only warns.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    (tmp_path / "config.json").write_text("{}")
    model = str(tmp_path)

    resolution = resolve_omni_config(
        model,
        trust_remote_code=True,
        deploy_config_path=get_deploy_config_path("personaplex.yaml"),
        cli_overrides=cli_overrides,
        stage_overrides=stage_overrides,
        strategy_config_path=None,
    )
    stage0, stage1 = (build_engine_args_dict_from_omni_stage_config(stage, model) for stage in resolution.stage_configs)
    assert (stage1["max_model_len"] == 90_000) is resizes_stage1

    model_config = OmniEngineArgs(
        model=model,
        model_arch=stage0["model_arch"],
        hf_overrides=dict(stage0["hf_overrides"]),
        max_model_len=stage0["max_model_len"],
        skip_tokenizer_init=True,
        trust_remote_code=True,
    ).create_model_config()
    assert model_config.max_model_len == 90_000

    with pytest.raises(ValueError, match="exceeds Helium's RoPE table of 46080 positions"):
        HeliumModel(
            vllm_config=SimpleNamespace(model_config=model_config, quant_config=None),
            config=model_config.hf_text_config,
        )
