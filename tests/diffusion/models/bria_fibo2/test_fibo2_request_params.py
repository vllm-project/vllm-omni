#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-parameter handling for Bria Fibo-2.0: sampling defaults and the
request-scoped generator slot (PR review #9554)."""

from __future__ import annotations

import logging

import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

from vllm_omni.diffusion.models.bria_fibo2.pipeline_bria_fibo2 import (  # noqa: E402
    resolve_sampling_defaults,
)
from vllm_omni.diffusion.utils.param_utils import unwrap_request_generator  # noqa: E402


@pytest.mark.parametrize(
    ("model", "args", "expected"),
    [
        # The Dev defaults mirror diffusers' own signature defaults.
        ("briaai/Fibo-2.0-dev", None, (30, 5.0)),
        # The official Edit-Turbo repo id resolves to the documented turbo
        # defaults; the repo id is upstream-fixed, unlike a substring match.
        ("briaai/Fibo-2.0-edit-turbo", None, (4, 1.0)),
        # The engine may resolve the model to its HF-cache snapshot path; the
        # fixed models--org--name layout still identifies the repo.
        (
            "/hf-home/hub/models--briaai--Fibo-2.0-edit-turbo/snapshots/9a24ee6810a782dd6aad4639fadcf48db0957ef4",
            None,
            (4, 1.0),
        ),
        ("/hf-home/hub/models--briaai--Fibo-2.0-dev/snapshots/72415bd0", None, (30, 5.0)),
        # A locally hosted checkpoint carries no variant information: a turbo
        # copy at an arbitrary path gets the Dev defaults (as in diffusers)
        # unless the operator configures them.
        ("/models/fibo2", None, (30, 5.0)),
        ("/models/whatever-turbo", None, (30, 5.0)),
        # custom_pipeline_args set the defaults for such local copies...
        ("/models/fibo2", {"fibo2_default_num_inference_steps": 4, "fibo2_default_guidance_scale": 1.0}, (4, 1.0)),
        # ...and override the official repo id inference per field.
        ("briaai/Fibo-2.0-edit-turbo", {"fibo2_default_num_inference_steps": 8}, (8, 1.0)),
        ("briaai/Fibo-2.0-dev", {"fibo2_default_guidance_scale": 3.0}, (30, 3.0)),
    ],
)
def test_sampling_defaults(model, args, expected):
    assert resolve_sampling_defaults(model, args) == expected


def test_generator_slot_unwraps_single_element_list():
    gen = torch.Generator().manual_seed(42)
    assert unwrap_request_generator(gen) is gen
    assert unwrap_request_generator(None) is None
    assert unwrap_request_generator([]) is None
    # The form the review flagged: a single-element list must resolve to the
    # generator, not survive as a nested list.
    assert unwrap_request_generator([gen]) is gen


def test_generator_slot_warns_on_multiple_generators(caplog):
    first, second = torch.Generator(), torch.Generator()
    with caplog.at_level(logging.WARNING, logger="vllm_omni.diffusion.utils.param_utils"):
        assert unwrap_request_generator([first, second]) is first
    assert "2 generators" in caplog.text


def test_generator_slot_feeds_randn_tensor():
    """The unwrapped slot must be consumable by diffusers ``randn_tensor``'s
    batch path, which reads ``generator[0].device`` and indexes one generator
    per batch row: a nested list crashes there."""

    def pipeline_shaped(generator):
        # What the pipeline forward builds per request: unwrap the slot, then
        # collect one generator per row for randn_tensor's batch path.
        gen = unwrap_request_generator(generator)
        if gen is None:
            gen = torch.Generator().manual_seed(7)
        return randn_tensor((1, 4), generator=[gen], dtype=torch.float32)

    seeded = torch.Generator().manual_seed(7)
    as_single = pipeline_shaped(seeded)
    as_list = pipeline_shaped([torch.Generator().manual_seed(7)])
    assert torch.equal(as_single, as_list)
