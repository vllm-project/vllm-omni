# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the weight-stack data types and protocols (RFC 9536 01-1)."""

from __future__ import annotations

import dataclasses
import inspect
import json
import subprocess
import sys
import typing
from collections.abc import Mapping

import pytest
import torch

from vllm_omni.diffusion.weight_stack.interfaces import (
    AllGatherGroup,
    AllGatherGroups,
    Elections,
    Fingerprints,
    HostStorage,
    HostWeightManager,
    Loader,
    LoadState,
    LoadTransaction,
    MemoryPools,
    Offloader,
    OffloadRuntime,
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
    StartupContext,
    Topology,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

# Protocol method -> parameter names, exactly as specified by RFC 01 section 2.
EXPECTED_PROTOCOLS: dict[type, dict[str, list[str]]] = {
    MemoryPools: {"weights": ["self"], "slots": ["self"]},
    Loader: {
        "build_structure": ["self"],
        "preflight": ["self", "catalog", "plan"],
        "write": ["self", "model", "storage", "transaction"],
        "finalize": ["self", "model", "storage"],
        "close": ["self"],
    },
    Offloader: {
        "plan": ["self", "catalog", "topology", "policy"],
        "prepare_materialization": ["self", "model", "plan", "storage"],
        "setup_compile": ["self", "model", "catalog", "runtime"],
    },
    LoadTransaction: {
        "target": ["self", "tensor_key"],
        "commit": ["self", "final_manifest"],
        "abort": ["self"],
    },
    OffloadRuntime: {
        "close": ["self"],
        "grad_mode": ["self"],
        "execution": ["self"],
        "before_sleep": ["self"],
        "after_wake": ["self"],
    },
    HostWeightManager: {
        "local_record": ["self"],
        "deployment_id": ["self"],
        "layout": ["self", "plan", "expected_manifest"],
        "share_plan": ["self", "topology", "fingerprints"],
        "elect": ["self", "share_plan"],
        "admit": ["self", "topology", "share_plan", "elections", "layout"],
        "allocate": ["self", "layout", "share_plan", "elections"],
    },
    HostStorage: {
        "view": ["self", "tensor_key"],
        "ranges": ["self", "unit"],
        "seal": ["self", "final_manifest"],
        "connect_writer": ["self"],
        "note_written": ["self", "tensor_key"],
        "close": ["self"],
    },
}


class _FakeMemoryPools:
    def weights(self):
        return None

    def slots(self):
        return None


class _FakeLoader:
    def build_structure(self):
        return None

    def preflight(self, catalog, plan):
        return None

    def write(self, model, storage, transaction):
        return None

    def finalize(self, model, storage):
        return None

    def close(self):
        return None


class _FakeOffloader:
    def plan(self, catalog, topology, policy):
        return None

    def prepare_materialization(self, model, plan, storage):
        return None

    def setup_compile(self, model, catalog, runtime):
        return None


class _FakeLoadTransaction:
    def target(self, tensor_key):
        return None

    def commit(self, final_manifest):
        return None

    def abort(self):
        return None


class _FakeOffloadRuntime:
    def close(self):
        return None

    def grad_mode(self):
        return None

    def execution(self):
        return None

    def before_sleep(self):
        return None

    def after_wake(self):
        return None


class _FakeHostWeightManager:
    def local_record(self):
        return None

    def deployment_id(self):
        return ("0" * 32, False)

    def layout(self, plan, expected_manifest):
        return None

    def share_plan(self, topology, fingerprints):
        return None

    def elect(self, share_plan):
        return None

    def admit(self, topology, share_plan, elections, layout):
        return None

    def allocate(self, layout, share_plan, elections):
        return None


class _FakeHostStorage:
    def view(self, tensor_key):
        return None

    def ranges(self, unit):
        return []

    def seal(self, final_manifest):
        return None

    def connect_writer(self):
        return None

    def note_written(self, tensor_key):
        return None

    def close(self):
        return None


FAKES = {
    MemoryPools: _FakeMemoryPools(),
    Loader: _FakeLoader(),
    Offloader: _FakeOffloader(),
    LoadTransaction: _FakeLoadTransaction(),
    OffloadRuntime: _FakeOffloadRuntime(),
    HostWeightManager: _FakeHostWeightManager(),
    HostStorage: _FakeHostStorage(),
}


def _protocol_methods(protocol: type) -> dict[str, typing.Any]:
    return {name: member for name, member in vars(protocol).items() if not name.startswith("_") and callable(member)}


def _stub_methods(stub: object) -> dict[str, typing.Any]:
    return {name: member for name, member in vars(type(stub)).items() if not name.startswith("_") and callable(member)}


def _coordinates(size: int = 1, rank: int = 0) -> RankCoordinates:
    coordinate = ParallelCoordinate(size=size, rank=rank)
    return RankCoordinates(
        tp=coordinate,
        sp=coordinate,
        ulysses=coordinate,
        ring=coordinate,
        cfg=coordinate,
        dp=coordinate,
        pp=coordinate,
        ep=None,
        fs=None,
    )


def test_load_state_members_and_values():
    assert {state.name: state.value for state in LoadState} == {
        "STRUCTURE_READY": "structure_ready",
        "PLANNED": "planned",
        "SOURCES_PREPARED": "sources_prepared",
        "ADMITTED": "admitted",
        "TRANSACTION_READY": "transaction_ready",
        "WEIGHTS_READY": "weights_ready",
        "FINALIZED": "finalized",
        "COMMITTED": "committed",
        "IN_SERVICE": "in_service",
        "FAILED": "failed",
        "CLOSED": "closed",
    }


def test_parallel_coordinate_and_rank_coordinates_are_frozen_values():
    coordinate = ParallelCoordinate(size=2, rank=1)
    assert (coordinate.size, coordinate.rank) == (2, 1)

    coordinates = _coordinates(size=2, rank=1)
    assert [field.name for field in dataclasses.fields(RankCoordinates)] == [
        "tp",
        "sp",
        "ulysses",
        "ring",
        "cfg",
        "dp",
        "pp",
        "ep",
        "fs",
    ]
    assert coordinates.ep is None
    assert coordinates.fs is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        coordinates.tp = ParallelCoordinate(size=1, rank=0)


def test_topology_fields_are_ordered_by_rank_and_frozen():
    topology = Topology(
        deployment_id="0" * 32,
        deployment_scoped=False,
        stage_id=0,
        world_size=1,
        records=(object(),),
        coordinates=(_coordinates(),),
    )
    assert [field.name for field in dataclasses.fields(Topology)] == [
        "deployment_id",
        "deployment_scoped",
        "stage_id",
        "world_size",
        "records",
        "coordinates",
    ]
    assert topology.world_size == len(topology.records) == len(topology.coordinates)
    with pytest.raises(dataclasses.FrozenInstanceError):
        topology.deployment_id = "f" * 32


def test_rank_fingerprints_and_aliases():
    fingerprints = RankFingerprints(
        rank=0,
        content_identities={"dit": "sha256:aa"},
        plan_fingerprint="sha256:bb",
        collective={"dit": "sha256:cc"},
    )
    assert fingerprints.rank == 0
    assert dict(fingerprints.content_identities) == {"dit": "sha256:aa"}
    assert fingerprints.plan_fingerprint == "sha256:bb"
    assert dict(fingerprints.collective) == {"dit": "sha256:cc"}
    assert [field.name for field in dataclasses.fields(RankFingerprints)] == [
        "rank",
        "content_identities",
        "plan_fingerprint",
        "collective",
    ]
    assert typing.get_origin(Fingerprints) is tuple
    assert typing.get_args(Fingerprints) == (RankFingerprints, Ellipsis)
    assert typing.get_origin(AllGatherGroups) is Mapping
    assert typing.get_args(AllGatherGroups)[0] is str
    assert typing.get_args(AllGatherGroups)[1] == tuple[AllGatherGroup, ...]


def test_all_gather_group_and_elections_contract():
    group = AllGatherGroup(component="dit", ranks=(0, 2), group=None)
    assert [field.name for field in dataclasses.fields(AllGatherGroup)] == ["component", "ranks", "group"]
    assert group.ranks == (0, 2)
    assert group.group is None

    local = object()
    elections = Elections(local=local, records=(object(),))
    assert [field.name for field in dataclasses.fields(Elections)] == ["local", "records"]
    assert elections.local is local


def test_startup_context_fields_and_frozen():
    context = StartupContext(
        od_config=object(),
        vllm_config=object(),
        device=torch.device("cpu"),
        rank=0,
        local_rank=0,
        world_size=1,
        load_format="default",
        custom_pipeline_name=None,
        memory_pools=_FakeMemoryPools(),
        policy=None,
        compile_enabled=False,
        faults=None,
    )
    assert [field.name for field in dataclasses.fields(StartupContext)] == [
        "od_config",
        "vllm_config",
        "device",
        "rank",
        "local_rank",
        "world_size",
        "load_format",
        "custom_pipeline_name",
        "memory_pools",
        "policy",
        "compile_enabled",
        "faults",
    ]
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.rank = 1


@pytest.mark.parametrize("protocol", list(EXPECTED_PROTOCOLS), ids=lambda protocol: protocol.__name__)
def test_protocol_methods_match_rfc_contract(protocol):
    methods = _protocol_methods(protocol)
    assert set(methods) == set(EXPECTED_PROTOCOLS[protocol])
    for name, expected_parameters in EXPECTED_PROTOCOLS[protocol].items():
        assert list(inspect.signature(methods[name]).parameters) == expected_parameters


@pytest.mark.parametrize("protocol", list(EXPECTED_PROTOCOLS), ids=lambda protocol: protocol.__name__)
def test_stub_implements_every_protocol_method(protocol):
    """A structural implementation satisfies the whole protocol surface (01-5's fakes follow this)."""
    stub = FAKES[protocol]
    stub_methods = _stub_methods(stub)
    for name, expected_parameters in EXPECTED_PROTOCOLS[protocol].items():
        assert name in stub_methods, f"{type(stub).__name__} is missing {name}"
        assert list(inspect.signature(stub_methods[name]).parameters) == expected_parameters


def test_importing_interfaces_and_errors_loads_no_type_checking_modules(tmp_path):
    """importing the foundation must not pull in implementations or vLLM config modules."""
    result_path = tmp_path / "loaded-modules.json"
    code = """
import json
import sys

import vllm_omni  # noqa: F401

baseline = set(sys.modules)
from vllm_omni.diffusion.weight_stack import errors, interfaces  # noqa: F401

forbidden = (
    "torch.distributed",
    "vllm.config",
    "vllm_omni.diffusion.data",
    "vllm_omni.diffusion.weight_stack.components",
    "vllm_omni.diffusion.weight_stack.host",
    "vllm_omni.diffusion.weight_stack.loader",
    "vllm_omni.diffusion.weight_stack.offload",
    "vllm_omni.diffusion.weight_stack.policy",
    "vllm_omni.diffusion.weight_stack.startup",
)
loaded = sorted(
    name
    for name in set(sys.modules) - baseline
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(loaded, handle)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(result_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result_path.read_text(encoding="utf-8")) == []
