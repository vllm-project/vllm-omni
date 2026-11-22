# SPDX-License-Identifier: Apache-2.0
"""Keep legacy defaults while reading production sampling and prepared actions."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_cookbook import manifest
from test_cookbook_runner import runner  # noqa: F401 - shared fixture
from test_rollout import fake_pipeline, request

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import prepare_domain_ids
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.control_contract import (
    parse_cosmos3_nano_sim_bimanual_conditioning,
)


def production():
    schema = json.loads((Path(__file__).parent / "fixtures/unified_action_schema.json").read_text())
    return replace(
        manifest(),
        window_frames=None,
        t_list=(1.0, 5 / 6),
        conditioning=parse_cosmos3_nano_sim_bimanual_conditioning({"mode": "action", **schema}),
    )


def test_prepared_actions_are_not_normalized_twice():
    pipe, _ = fake_pipeline()
    pipe.manifest = production()
    pipe.action_normalizers = {}
    x = torch.arange(118, dtype=torch.float32).reshape(2, 59)
    y = pipe._prepare_raw_action(embodiment="agibotworld", action_value=x, action_space="normalized_unified_v1")
    assert torch.equal(y[:, :59].float(), x)
    assert y.shape == (2, 64) and not torch.count_nonzero(y[:, 59:])
    with pytest.raises(ValueError, match="Schema 5"):
        pipe._prepare_raw_action(embodiment="agibotworld", action_value=x, action_space="raw")


def test_per_row_domains_follow_action_chunks():
    domains = prepare_domain_ids([2] * 4 + [15] * 8, rows=12, allowed={2, 15})
    assert domains.tolist() == [2] * 4 + [15] * 8
    with pytest.raises(ValueError):
        prepare_domain_ids([99], rows=12, allowed={2, 15})


@pytest.mark.parametrize("domain", [15, [15], torch.tensor(15), torch.tensor([15])])
def test_singleton_domain_representations_share_session_identity(domain):
    pipe, state = fake_pipeline()
    pipe.forward(request(9, domain_id=domain, close_session=False))
    baseline, baseline_state = fake_pipeline()
    baseline.forward(request(9, domain_id=15, close_session=False))
    assert state.fingerprint == baseline_state.fingerprint


def test_cli_reads_prepared_sidecar(tmp_path, runner):  # noqa: F811
    module = runner
    sidecar = {"action_space": "normalized_unified_v1", "action": [[0.0] * 59] * 8, "domain_names": ["agibotworld"]}
    (tmp_path / "actions.json").write_text(json.dumps(sidecar))
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"action_path": "actions.json", "vision_path": "initial.png"}) + "\n")
    record = module._load_record(path, 0)
    assert record["action_space"] == "normalized_unified_v1" and record["image"] == "initial.png"
