# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline capability-detection tests (RFC #4590 / #4948).

Torch-free: interface.py imports torch only under TYPE_CHECKING, so the
capability helpers can be exercised directly. These lock the #4948 contract:

* :class:`DiffusionV2Atoms` is a single ``@runtime_checkable`` Protocol carrying
  BOTH the whole-request/disaggregation atoms (``init_state`` / ``check_inputs``
  / ``encode`` / ``prepare`` / ``diffuse`` / ``decode`` / ``postprocess`` /
  ``pack_stage_state`` / ``unpack_stage_state``) and the per-step atoms
  (``build_step_batch`` / ``build_step_attention_metadata`` / ``denoise_step`` /
  ``step_scheduler``), plus the RFC #4590 ``required_components_for_stage``;
* ``supports_step_execution`` requires the ``supports_step_execution`` flag plus
  either the ``DiffusionV2Atoms`` surface OR the legacy
  :class:`SupportsStepExecution` surface (``prepare_encode`` / ``denoise_step`` /
  ``step_scheduler`` / ``post_decode``), retained for un-migrated pipelines;
* ``supports_disaggregated_execution`` requires the full ``DiffusionV2Atoms``
  atom surface AND an explicit opt-in ``supports_disaggregated_execution`` flag —
  so a pipeline that merely satisfies the atom surface is still not treated as
  disaggregated unless it carries the flag.
"""

from __future__ import annotations


class _V2Atoms:
    """A full #4948 DiffusionV2Atoms method surface (whole-request + step atoms).

    ``DiffusionV2Atoms`` is a ``@runtime_checkable`` Protocol, so ``isinstance``
    requires every declared member to be *present* — including the
    ``supports_step_execution`` ClassVar (presence, not truthiness). A
    disaggregated pipeline opts in via the ``supports_disaggregated_execution``
    flag; a step pipeline opts in via the ``supports_step_execution`` flag; the
    capability helpers additionally check those flags for truthiness.
    """

    supports_disaggregated_execution = True
    supports_step_execution = False

    # whole-request / disaggregation atoms
    def init_state(self, s): ...
    def check_inputs(self, s): ...
    def encode(self, s): ...
    def prepare(self, s): ...
    def diffuse(self, s): ...
    def decode(self, s): ...
    def postprocess(self, s): ...
    def pack_stage_state(self, s, b): ...
    def unpack_stage_state(self, p, s): ...

    # per-step atoms (folded into the same protocol by #4948)
    def build_step_batch(self, states, *, cached_batch=None): ...
    def build_step_attention_metadata(self, input_batch): ...
    def denoise_step(self, input_batch): ...
    def step_scheduler(self, state, noise_pred): ...

    @classmethod
    def required_components_for_stage(cls, model_stage): ...


# Backwards-compatible alias: the disaggregation atom surface is the V2 surface.
_DisaggAtoms = _V2Atoms


class _StepAtoms:
    """A full legacy SupportsStepExecution method surface (un-migrated pipeline)."""

    supports_step_execution = True

    def prepare_encode(self, s, **k): ...
    def denoise_step(self, b, **k): ...
    def step_scheduler(self, s, n, **k): ...
    def post_decode(self, s, **k): ...


def test_step_pipeline_is_not_disaggregated(interface_mod):
    """A legacy step pipeline satisfies SupportsStepExecution but NOT the
    disaggregation contract (no whole-request atoms, no opt-in flag)."""
    step = _StepAtoms()
    assert interface_mod.supports_step_execution(step) is True
    assert interface_mod.supports_disaggregated_execution(step) is False


def test_disaggregated_requires_flag_and_atoms(interface_mod):
    class WithFlag(_V2Atoms):
        supports_disaggregated_execution = True

    class FlagOff(_V2Atoms):
        supports_disaggregated_execution = False  # explicit opt-out honored

    class NoAtoms:
        supports_disaggregated_execution = True  # flag set, but no atom methods

    assert interface_mod.supports_disaggregated_execution(WithFlag()) is True
    assert interface_mod.supports_disaggregated_execution(FlagOff()) is False
    assert interface_mod.supports_disaggregated_execution(NoAtoms()) is False


def test_disaggregated_pipeline_opts_out_of_step(interface_mod):
    """A disaggregated pipeline satisfies the DiffusionV2Atoms surface but opts
    OUT of step execution via ``supports_step_execution = False``, so the step
    runner never claims it (mirrors DreamZero)."""
    disagg = _V2Atoms()
    assert interface_mod.supports_disaggregated_execution(disagg) is True
    assert interface_mod.supports_step_execution(disagg) is False


def test_v2_step_pipeline_is_step_not_disaggregated(interface_mod):
    """A migrated step pipeline (DiffusionV2Atoms surface + step flag, no disagg
    flag) is claimed by the step runner but not the disaggregation path."""

    class V2Step(_V2Atoms):
        supports_step_execution = True
        supports_disaggregated_execution = False

    v2step = V2Step()
    assert interface_mod.supports_step_execution(v2step) is True
    assert interface_mod.supports_disaggregated_execution(v2step) is False


def test_monolithic_pipeline_needs_no_capability(interface_mod):
    class Monolithic:
        def forward(self, batch): ...

    assert interface_mod.supports_disaggregated_execution(Monolithic()) is False
    assert interface_mod.supports_step_execution(Monolithic()) is False
