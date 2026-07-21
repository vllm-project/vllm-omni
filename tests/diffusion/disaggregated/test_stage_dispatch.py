# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-free tests for the runner's stage-dispatch decision table (RFC #4590 §4).

The runner's ``execute_model`` routes on ``resolve_execution_path(model_stage)``,
which is a pure function kept in ``stage_roles`` precisely so the routing is
testable without importing the torch-heavy runner. A ``FakeRunner`` mirrors the
runner's dispatch structure over that same shared function to lock the routing
contract; the full behavioral runner tests (real pipeline mutations, payload
export/import, cleanup) live in ``test_runner_stages.py`` under ``needs_runtime``.
"""

from __future__ import annotations


class FakeRunner:
    """Mirrors DiffusionModelRunner.execute_model dispatch using the shared table.

    Records which stage path each role selects so the routing contract can be
    asserted without torch. Kept structurally identical to the real dispatch so
    a divergence in the table surfaces here.
    """

    def __init__(self, roles_mod, model_stage):
        self._roles = roles_mod
        self._model_stage = model_stage
        self.calls: list[str] = []

    @property
    def model_stage(self):
        return self._roles.normalize_stage_role(self._model_stage)

    def execute_model(self, req=None):
        roles = self._roles
        path = roles.resolve_execution_path(self.model_stage)
        if path == roles.EXECUTION_PATH_ENCODE:
            return self._encode()
        if path == roles.EXECUTION_PATH_DENOISE:
            return self._denoise()
        if path == roles.EXECUTION_PATH_DECODE:
            return self._decode()
        if path == roles.EXECUTION_PATH_MODEL_DEFINED:
            return self._model_defined()
        return self._monolithic()

    def _encode(self):
        self.calls.append("encode")

    def _denoise(self):
        self.calls.append("denoise")

    def _decode(self):
        self.calls.append("decode")

    def _monolithic(self):
        self.calls.append("monolithic")

    def _model_defined(self):
        self.calls.append("model_defined")


def test_resolve_execution_path_table(stage_roles):
    r = stage_roles
    assert r.resolve_execution_path(None) == r.EXECUTION_PATH_MONOLITHIC
    assert r.resolve_execution_path("") == r.EXECUTION_PATH_MONOLITHIC
    assert r.resolve_execution_path("diffusion") == r.EXECUTION_PATH_MONOLITHIC
    assert r.resolve_execution_path("encode") == r.EXECUTION_PATH_ENCODE
    assert r.resolve_execution_path("denoise") == r.EXECUTION_PATH_DENOISE
    assert r.resolve_execution_path("decode") == r.EXECUTION_PATH_DECODE
    assert r.resolve_execution_path("kv_update") == r.EXECUTION_PATH_MODEL_DEFINED


def test_encode_role_calls_only_encode(stage_roles):
    runner = FakeRunner(stage_roles, "encode")
    runner.execute_model()
    assert runner.calls == ["encode"]
    assert "denoise" not in runner.calls and "decode" not in runner.calls


def test_denoise_role_calls_only_denoise(stage_roles):
    runner = FakeRunner(stage_roles, "denoise")
    runner.execute_model()
    assert runner.calls == ["denoise"]


def test_decode_role_calls_only_decode(stage_roles):
    runner = FakeRunner(stage_roles, "decode")
    runner.execute_model()
    assert runner.calls == ["decode"]


def test_monolithic_role_unchanged(stage_roles):
    for role in (None, "", "diffusion"):
        runner = FakeRunner(stage_roles, role)
        runner.execute_model()
        assert runner.calls == ["monolithic"]


def test_custom_role_uses_model_defined(stage_roles):
    runner = FakeRunner(stage_roles, "kv_update")
    runner.execute_model()
    assert runner.calls == ["model_defined"]
