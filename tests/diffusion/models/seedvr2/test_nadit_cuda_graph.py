# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact-profile admission, ragged peer contracts and explicit invalidation."""

import pytest
import torch

from vllm_omni.diffusion.models.seedvr2 import nadit as n
from vllm_omni.diffusion.models.seedvr2.nadit_cuda_graph import (
    CONFIG_KEY,
    NaDiTCoreGraphs,
    NaDiTGraphConfig,
    _versions,
    validate_transition_descriptors,
)

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]


def test_exact_geometry_identity_and_eager_override():
    extras = {CONFIG_KEY: {"profiles": [[3, 44, 80], [3, 80, 44]]}}
    config = NaDiTGraphConfig.from_extras(extras, enforce_eager=False)
    assert config.profiles == ((3, 44, 80), (3, 80, 44))
    assert NaDiTGraphConfig.from_extras(extras, enforce_eager=True) is None
    assert NaDiTGraphConfig.from_extras({}, enforce_eager=False) is None


@pytest.mark.parametrize(
    "value",
    [
        True,
        [],
        {},
        {"profiles": []},
        {"profiles": [[1, 2]]},
        {"profiles": [[1, 3, 4]]},
        {"profiles": [[True, 2, 4]]},
        {"profiles": [[0, 2, 4]]},
        {"profiles": [[1, 2, 4]] * 2},
        {"profiles": [[i, 2, 4] for i in range(1, 6)]},
        {"profiles": [[1, 2, 4]], "max_pool_bytes": 0},
        {"profiles": [[1, 2, 4]], "eager_headroom_bytes": 0.5},
        {"profiles": [[1, 2, 4]], "recapture_on_miss": True},
    ],
)
def test_invalid_graph_profiles_are_startup_errors(value):
    with pytest.raises(ValueError):
        NaDiTGraphConfig.from_extras({CONFIG_KEY: value}, enforce_eager=False)


def descriptors():
    return [
        [dict(layer=1, send=[2, 1], recv=[2, 0], input_rows=3, output_rows=2)],
        [dict(layer=1, send=[0, 3], recv=[1, 3], input_rows=3, output_rows=4)],
    ]


def test_ragged_peer_vectors_and_empty_rank():
    validate_transition_descriptors(descriptors())
    rows = descriptors()
    for rank in rows:
        rank[0]["send"] += [0, 0]
        rank[0]["recv"] += [0, 0]
    rows += [[dict(layer=1, send=[0] * 4, recv=[0] * 4, input_rows=0, output_rows=0)] for _ in range(2)]
    validate_transition_descriptors(rows)


@pytest.mark.parametrize("failure", ["peer", "row_total", "negative", "layers", "world"])
def test_inconsistent_transition_manifest_is_rejected(failure):
    rows = descriptors()
    if failure == "peer":
        rows[1][0]["recv"] = [2, 2]  # valid total, wrong peer routing
    elif failure == "row_total":
        rows[0][0]["input_rows"] = 4
    elif failure == "negative":
        rows[0][0]["send"] = [-1, 4]
    elif failure == "layers":
        rows[1] = []
    else:
        rows[0][0]["recv"] = [2]
    with pytest.raises(ValueError):
        validate_transition_descriptors(rows)


def test_parameter_generation_and_storage_changes():
    model = torch.nn.Linear(2, 2)
    original = _versions(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert _versions(model) != original
    original = _versions(model)
    model.weight = torch.nn.Parameter(model.weight.clone())
    assert _versions(model) != original
    with torch.inference_mode():
        inference = torch.nn.Linear(2, 2)
        assert _versions(inference)  # explicit loader invalidation covers no-version tensors


@pytest.mark.parametrize("operation", ["move", "reload"])
def test_model_retires_graph_before_storage_change(operation):
    model = n.SeedVR2NaDiT.__new__(n.SeedVR2NaDiT)
    torch.nn.Module.__init__(model)
    model.device_plan_cache = n.DeviceWindowPlanCache()
    model.weight = torch.nn.Parameter(torch.ones(2))
    observed = []

    class Graphs:
        def close(self):
            observed.append(model.weight.detach().clone())

    model.core_graphs = Graphs()
    if operation == "move":
        model.to(dtype=torch.float64)
        assert observed[0].dtype == torch.float32
    else:
        model.load_state_dict({"weight": torch.zeros(2)})
        assert observed[0].tolist() == [1, 1]
        assert model.weight.tolist() == [0, 0]
    assert model.core_graphs is None and len(observed) == 1


def test_single_gpu_graph_control_avoids_distributed_collectives(monkeypatch):
    graphs = NaDiTCoreGraphs.__new__(NaDiTCoreGraphs)
    graphs.world = 1
    graphs.cpu_group = None
    synchronized = []

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("single-GPU graphs must not issue a distributed control collective")

    monkeypatch.setattr(torch.distributed, "all_gather_object", unexpected_collective)
    monkeypatch.setattr(torch.distributed, "barrier", unexpected_collective)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: synchronized.append(True))
    payload = {"geometry": [2, 60, 106]}
    assert graphs._gather(payload) == [payload]
    graphs._drain()
    assert synchronized == [True]


def test_single_gpu_nadit_graph_config_is_independent_of_sp():
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import validate_seedvr2_config
    from vllm_omni.diffusion.models.seedvr2.vae_cuda_graph import CONFIG_KEY as VAE_KEY

    config = OmniDiffusionConfig(
        model="SeedVR2",
        dtype=torch.float16,
        enforce_eager=False,
        max_num_seqs=1,
        vae_use_tiling=True,
        parallel_config=DiffusionParallelConfig(ulysses_degree=1, vae_patch_parallel_size=1),
        extras={CONFIG_KEY: {"profiles": [[2, 60, 106]]}},
    )
    validate_seedvr2_config(config)
    config.extras[VAE_KEY] = {"profiles": [[2, 60, 106]]}
    with pytest.raises(ValueError, match="VAE decoder graphs require SP2/SP4"):
        validate_seedvr2_config(config)
