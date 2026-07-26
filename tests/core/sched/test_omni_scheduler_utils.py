# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.core.sched.utils import split_free_request_result

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_split_free_request_result_supports_old_and_new_vllm_contracts():
    kv = {"kv": "ready"}
    ec = {"ec": "ready"}

    assert split_free_request_result(None) == (None, None)
    assert split_free_request_result(kv) == (kv, None)
    assert split_free_request_result((kv, ec)) == (kv, ec)


@pytest.mark.parametrize("result", [([], None), (None, []), [None, None]])
def test_split_free_request_result_rejects_non_mapping_wire_values(result):
    with pytest.raises(TypeError):
        split_free_request_result(result)
