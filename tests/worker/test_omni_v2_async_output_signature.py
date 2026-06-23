# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

from vllm_omni.worker_v2.omni_ar_model_runner import OmniAsyncOutput


def test_omni_async_output_does_not_require_external_copy_event():
    param = inspect.signature(OmniAsyncOutput.__init__).parameters["copy_event"]

    assert param.default is None


if __name__ == "__main__":
    test_omni_async_output_does_not_require_external_copy_event()
