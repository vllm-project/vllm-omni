# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import copy

import pytest

from vllm_omni.engine.request_snapshot import copy_request_snapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_full_snapshot_preserves_fields_aliases_and_isolation():
    samples = [0.25, 0.5]
    source = {"prompt_token_ids": [1, 2], "additional_information": {"samples": samples}, "alias": samples}
    result = copy_request_snapshot(source)
    assert result == copy.deepcopy(source)
    assert result["alias"] is result["additional_information"]["samples"]
    result["alias"].append(0.75)
    result["prompt_token_ids"][0] = 9
    assert samples == [0.25, 0.5]
    assert source["prompt_token_ids"] == [1, 2]


def test_shared_types_are_returned_as_is_inside_copied_containers():
    torch = pytest.importorskip("torch")

    tensor = torch.arange(4)
    labels = ["a"]
    source = {"codes": {"ref": tensor}, "labels": labels, "pair": [tensor, labels]}
    result = copy_request_snapshot(source, shared_types=(torch.Tensor,))

    assert result["codes"]["ref"] is tensor
    assert result["pair"][0] is tensor
    assert result["codes"] is not source["codes"]
    assert result["labels"] is not labels
    assert result["pair"][1] is result["labels"]
