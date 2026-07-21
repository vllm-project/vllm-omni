# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline capability-detection tests (#4948 DiffusionV2Atoms).

Torch-free: interface.py imports torch only under TYPE_CHECKING, so the
capability helpers can be exercised directly. These lock the collapsed contract:
``supports_step_execution`` and ``supports_disaggregated_execution`` both require
the full :class:`DiffusionV2Atoms` method surface AND an explicit opt-in flag —
so a pipeline that satisfies the atom surface is still not treated as
step/disaggregated unless it carries the corresponding flag.
"""

from __future__ import annotations


class _V2Atoms:
    """A full DiffusionV2Atoms method surface.

    ``DiffusionV2Atoms`` is a ``@runtime_checkable`` Protocol, so ``isinstance``
    requires every declared member to be *present* — including the
    ``supports_step_execution`` ClassVar (presence, not truthiness). Pipelines
    opt into step vs disaggregated behavior via the two ``supports_*`` flags
    below, both of which are read by the capability helpers.
    """

    supports_step_execution = False

    def init_state(self, s): ...
    def check_inputs(self, s): ...
    def encode(self, s): ...
    def prepare(self, s): ...
    def diffuse(self, s): ...
    def decode(self, s): ...
    def postprocess(self, s): ...
    def pack_stage_state(self, s, b): ...
    def unpack_stage_state(self, p, s): ...
    def build_step_batch(self, states, *, cached_batch=None): ...
    def build_step_attention_metadata(self, b): ...
    def denoise_step(self, b): ...
    def step_scheduler(self, s, n): ...

    @classmethod
    def required_components_for_stage(cls, model_stage): ...


def test_step_only_pipeline_is_not_disaggregated(interface_mod):
    class StepOnly(_V2Atoms):
        supports_step_execution = True
        # No supports_disaggregated_execution flag.

    assert interface_mod.supports_step_execution(StepOnly()) is True
    assert interface_mod.supports_disaggregated_execution(StepOnly()) is False


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


def test_monolithic_pipeline_needs_no_capability(interface_mod):
    class Monolithic:
        def forward(self, batch): ...

    assert interface_mod.supports_disaggregated_execution(Monolithic()) is False
    assert interface_mod.supports_step_execution(Monolithic()) is False
