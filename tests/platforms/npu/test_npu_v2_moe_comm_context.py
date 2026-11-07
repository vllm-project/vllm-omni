# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V2 MoE communication fields survive being stored on the forward context."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm_ascend")

from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import (
    resolve_registered_moe_comm_method,
    v2_extra_forward_attr,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_EXTRA = ("moe_comm_type", "moe_comm_method", "in_profile_run")


def test_v2_extra_reads_attribute_when_the_dict_omits_it():
    ctx = SimpleNamespace(additional_kwargs={}, moe_comm_type="alltoall")
    assert v2_extra_forward_attr(ctx, "moe_comm_type", _EXTRA) == "alltoall"


def test_v2_extra_prefers_an_explicit_dict_value():
    ctx = SimpleNamespace(additional_kwargs={"moe_comm_type": "mc2"}, moe_comm_type="alltoall")
    assert v2_extra_forward_attr(ctx, "moe_comm_type", _EXTRA) == "mc2"


def test_unshaped_lookup_uses_the_only_registered_method():
    method = object()
    methods = {("alltoall", (256, 128)): method}
    assert resolve_registered_moe_comm_method(methods, "alltoall", (0, 0)) is method
    assert resolve_registered_moe_comm_method(methods, "allgather", (0, 0)) is None


def test_unshaped_lookup_stays_unset_when_shapes_differ():
    methods = {("alltoall", (256, 128)): object(), ("alltoall", (64, 32)): object()}
    assert resolve_registered_moe_comm_method(methods, "alltoall", (0, 0)) is None


def test_v2_extra_rejects_unknown_names():
    ctx = SimpleNamespace(additional_kwargs={})
    with pytest.raises(AttributeError, match="not extra"):
        v2_extra_forward_attr(ctx, "not_an_extra", _EXTRA)
