# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline capability-detection tests (RFC #4590 / #4948).

Torch-free: interface.py imports torch only under TYPE_CHECKING, so the
capability helpers can be exercised directly. These lock the split contract:

* ``supports_step_execution`` is a pure structural ``isinstance`` against
  :class:`SupportsStepExecution` (``prepare_encode`` / ``denoise_step`` /
  ``step_scheduler`` / ``post_decode``);
* ``supports_disaggregated_execution`` requires the full
  :class:`SupportsDisaggregatedExecution` atom surface AND an explicit opt-in
  ``supports_disaggregated_execution`` flag — so a pipeline that merely satisfies
  the atom surface is still not treated as disaggregated unless it carries the
  flag.
"""

from __future__ import annotations


class _DisaggAtoms:
    """A full SupportsDisaggregatedExecution method surface.

    ``SupportsDisaggregatedExecution`` is a ``@runtime_checkable`` Protocol, so
    ``isinstance`` requires every declared member to be *present* — including the
    ``supports_disaggregated_execution`` ClassVar (presence, not truthiness).
    Pipelines opt in via the ``supports_disaggregated_execution`` flag, which the
    capability helper additionally checks for truthiness.
    """

    supports_disaggregated_execution = True

    def init_state(self, s): ...
    def check_inputs(self, s): ...
    def encode(self, s): ...
    def prepare(self, s): ...
    def diffuse(self, s): ...
    def decode(self, s): ...
    def postprocess(self, s): ...
    def pack_stage_state(self, s, b): ...
    def unpack_stage_state(self, p, s): ...

    @classmethod
    def required_components_for_stage(cls, model_stage): ...


class _StepAtoms:
    """A full SupportsStepExecution method surface (the slim upstream protocol)."""

    supports_step_execution = True

    def prepare_encode(self, s, **k): ...
    def denoise_step(self, b, **k): ...
    def step_scheduler(self, s, n, **k): ...
    def post_decode(self, s, **k): ...


def test_step_pipeline_is_not_disaggregated(interface_mod):
    """A pure step pipeline satisfies SupportsStepExecution but NOT the
    disaggregation contract (different, non-overlapping method surface)."""
    step = _StepAtoms()
    assert interface_mod.supports_step_execution(step) is True
    assert interface_mod.supports_disaggregated_execution(step) is False


def test_disaggregated_requires_flag_and_atoms(interface_mod):
    class WithFlag(_DisaggAtoms):
        supports_disaggregated_execution = True

    class FlagOff(_DisaggAtoms):
        supports_disaggregated_execution = False  # explicit opt-out honored

    class NoAtoms:
        supports_disaggregated_execution = True  # flag set, but no atom methods

    assert interface_mod.supports_disaggregated_execution(WithFlag()) is True
    assert interface_mod.supports_disaggregated_execution(FlagOff()) is False
    assert interface_mod.supports_disaggregated_execution(NoAtoms()) is False


def test_disaggregated_pipeline_is_not_step(interface_mod):
    """A disaggregated pipeline does NOT structurally satisfy SupportsStepExecution
    (it has diffuse/pack_stage_state, not prepare_encode/denoise_step/post_decode)."""
    disagg = _DisaggAtoms()
    assert interface_mod.supports_disaggregated_execution(disagg) is True
    assert interface_mod.supports_step_execution(disagg) is False


def test_monolithic_pipeline_needs_no_capability(interface_mod):
    class Monolithic:
        def forward(self, batch): ...

    assert interface_mod.supports_disaggregated_execution(Monolithic()) is False
    assert interface_mod.supports_step_execution(Monolithic()) is False
