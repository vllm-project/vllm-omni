# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden characterization of ``_build_model_sampler_output_token_ids``.

The base runner (``OmniGPUModelRunner``) and the AR runner
(``GPUARModelRunner``) implement the SAME async-placeholder backfill, but the
AR copy adds a trailing crop step: after backfill it truncates every history at
the first remaining ``-1``, while the base copy KEEPS residual ``-1``s.

A later change collapses these two copies into one; these goldens pin today's divergence
exactly (crop vs keep) so that merge is provably behaviour-preserving. Pure CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _runner(
    cls,
    *,
    req_ids,
    req_output_token_ids,
    sampled_token_ids_cpu,
    prev_req_id_to_index,
):
    """Build a fake runner exposing only the ``input_batch`` fields the method reads."""
    runner = object.__new__(cls)
    event = SimpleNamespace(synchronize=lambda: None)
    runner.input_batch = SimpleNamespace(
        req_output_token_ids=req_output_token_ids,
        req_ids=req_ids,
        sampled_token_ids_cpu=(None if sampled_token_ids_cpu is None else torch.tensor(sampled_token_ids_cpu)),
        async_copy_ready_event=event,
        prev_req_id_to_index=prev_req_id_to_index,
    )
    return runner


def _build(cls, **kwargs):
    return cls._build_model_sampler_output_token_ids(_runner(cls, **kwargs))


# --------------------------------------------------------------------------- #
# Shared behaviour — base and AR agree                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [OmniGPUModelRunner, GPUARModelRunner])
def test_full_backfill_no_residual_placeholder(cls):
    # One trailing placeholder, one sampled token -> fully resolved for both.
    out = _build(
        cls,
        req_ids=["r0"],
        req_output_token_ids=[[10, 11, -1]],
        sampled_token_ids_cpu=[[42]],
        prev_req_id_to_index={"r0": 0},
    )
    assert out == [[10, 11, 42]]


@pytest.mark.parametrize("cls", [OmniGPUModelRunner, GPUARModelRunner])
def test_spec_decode_trailing_pad_in_sampled_is_trimmed(cls):
    # sampled row has a trailing -1 pad; only the real prefix [42, 43] is used.
    out = _build(
        cls,
        req_ids=["r0"],
        req_output_token_ids=[[10, -1, -1]],
        sampled_token_ids_cpu=[[42, 43, -1]],
        prev_req_id_to_index={"r0": 0},
    )
    assert out == [[10, 42, 43]]


@pytest.mark.parametrize("cls", [OmniGPUModelRunner, GPUARModelRunner])
def test_no_trailing_placeholder_is_untouched(cls):
    out = _build(
        cls,
        req_ids=["r0"],
        req_output_token_ids=[[10, 11, 12]],
        sampled_token_ids_cpu=[[42]],
        prev_req_id_to_index={"r0": 0},
    )
    assert out == [[10, 11, 12]]


@pytest.mark.parametrize("cls", [OmniGPUModelRunner, GPUARModelRunner])
def test_early_return_when_sampled_absent_keeps_placeholders(cls):
    # sampled_token_ids_cpu is None -> both copies return BEFORE the crop step,
    # so even the AR copy keeps residual -1 here.
    out = _build(
        cls,
        req_ids=["r0"],
        req_output_token_ids=[[10, -1]],
        sampled_token_ids_cpu=None,
        prev_req_id_to_index={"r0": 0},
    )
    assert out == [[10, -1]]


# --------------------------------------------------------------------------- #
# Divergence — base KEEPS residual -1, AR CROPS at the first -1                #
# --------------------------------------------------------------------------- #
_RESIDUAL_CASE = dict(
    req_ids=["r0"],
    req_output_token_ids=[[10, -1, -1]],  # two placeholders
    sampled_token_ids_cpu=[[42]],  # only one sampled token -> one -1 remains
    prev_req_id_to_index={"r0": 0},
)


def test_base_keeps_residual_placeholder():
    out = _build(OmniGPUModelRunner, **_RESIDUAL_CASE)
    assert out == [[10, 42, -1]]


def test_ar_crops_residual_placeholder():
    out = _build(GPUARModelRunner, **_RESIDUAL_CASE)
    assert out == [[10, 42]]


_ABSENT_FROM_PREV_CASE = dict(
    req_ids=["r0"],
    req_output_token_ids=[[10, -1]],
    sampled_token_ids_cpu=[[42]],
    prev_req_id_to_index={},  # r0 not present last step -> no backfill happens
)


def test_base_keeps_placeholder_when_request_absent_from_prev_step():
    out = _build(OmniGPUModelRunner, **_ABSENT_FROM_PREV_CASE)
    assert out == [[10, -1]]


def test_ar_crops_placeholder_when_request_absent_from_prev_step():
    # No backfill occurred, but the AR crop pass still truncates at the -1.
    out = _build(GPUARModelRunner, **_ABSENT_FROM_PREV_CASE)
    assert out == [[10]]
