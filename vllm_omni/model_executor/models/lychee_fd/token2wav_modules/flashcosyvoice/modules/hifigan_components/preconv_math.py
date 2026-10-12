# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HiFT conv_pre-only released FP32 plans on independently qualified shapes.

Actual36 and supplementary cropped-real-Mel 29-length proofs admit K7/p3.
Lengths60/62 and other unqualified configurations retain the existing delegate.
Instances share only bounded plan lifetime mechanics with the F0 owner; they do
not alter F0 K3/p1 eligibility, ELU math, weights or process-wide backend flags.
"""

from __future__ import annotations

from .f0_math import (
    RELEASED_KNOBS,
    _F0Plan,
    _needs_grad,
    _qualified_runtime,
    _ReleasedConv1d,
)

QUALIFIED_LENGTHS = frozenset(range(2, 59, 2))


def validate_plan_contract(shape, weight_shape, config, engine, knobs):
    shape, weight_shape = tuple(shape), tuple(weight_shape)
    if len(shape) != 3 or shape[0] != 1 or shape[1] != 80 or shape[2] not in QUALIFIED_LENGTHS:
        raise ValueError("Unqualified HiFT conv_pre input shape")
    if weight_shape != (512, 80, 7):
        raise ValueError("Unqualified HiFT conv_pre weight shape")
    if any(config[key] != [value] for key, value in (("stride", 1), ("padding", 3), ("dilation", 1))):
        raise ValueError("Unqualified HiFT conv_pre convolution configuration")
    if config["groups"] != 1 or engine != 38 or knobs != RELEASED_KNOBS:
        raise ValueError("Unqualified HiFT conv_pre engine or knobs")
    return shape, weight_shape


class _PreconvPlan(_F0Plan):
    padding = 3

    @staticmethod
    def validate_shapes(shape, weight_shape):
        config = {"stride": [1], "padding": [3], "dilation": [1], "groups": 1}
        return validate_plan_contract(shape, weight_shape, config, 38, RELEASED_KNOBS)


class PreconvConv1d(_ReleasedConv1d):
    """Independent conv_pre eligibility with LRU4/16MiB instance ownership."""

    def _make_plan(self, shape, weight_shape, device):
        return _PreconvPlan(shape, weight_shape, device)

    def _eligible(self, inputs, weight, bias):
        return (
            not self.training
            and not _needs_grad(inputs, weight, bias)
            and _qualified_runtime(inputs)
            and inputs.ndim == 3
            and inputs.shape[0] == 1
            and inputs.shape[1] == 80
            and inputs.shape[2] in QUALIFIED_LENGTHS
            and tuple(weight.shape) == (512, 80, 7)
            and weight.dtype == inputs.dtype
            and weight.device == inputs.device
            and bias is not None
            and bias.shape == (512,)
            and bias.dtype == inputs.dtype
            and bias.device == inputs.device
            and self.groups == 1
            and self.stride == (1,)
            and self.padding == (3,)
            and self.dilation == (1,)
            and self.padding_mode == "zeros"
        )

    def _conv_forward(self, inputs, weight, bias):
        return super()._conv_forward(inputs, weight, bias)
