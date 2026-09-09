# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import copy

import numpy as np
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


def test_snapshot_preserves_cycles_including_tuple_fallback():
    source = []
    node = {"parent": source}
    source.extend([node, (source,)])
    result = copy_request_snapshot(source)
    assert result is not source
    assert result[0]["parent"] is result
    assert result[1][0] is result


def test_snapshot_copies_arrays_and_preserves_shared_identity():
    array = np.array([1.0, 2.0])
    result = copy_request_snapshot({"array": array, "alias": array})
    assert result["array"] is result["alias"]
    result["array"][0] = 5
    assert array[0] == 1


def test_subclasses_keep_custom_deepcopy_contract():
    class SpecialList(list):
        def __deepcopy__(self, memo):
            result = SpecialList(["custom"])
            memo[id(self)] = result
            return result

    source = SpecialList([1])
    result = copy_request_snapshot([source, source])
    assert result[0] == ["custom"]
    assert type(result[0]) is SpecialList
    assert result[0] is result[1]
