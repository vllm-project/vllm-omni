# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Ascend encoder dispatch with weight-free Hunyuan networks on CPU."""

import pytest

from tests.helpers.mark import hardware_marks
from tests.model_executor.models import test_hunyuan_request_rng as hunyuan_rng
from vllm_omni.model_executor.mm_request_rng import MM_RNG_ITEM_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"})]

runner = hunyuan_rng.runner
make_request = hunyuan_rng.make_request


def test_load_model_and_real_ascend_encoder_tag_requests(runner, make_request, mocker):
    pytest.importorskip("torch_npu")
    from vllm_omni.platforms.npu.worker import npu_model_runner as module

    # Reuse the weight-free model setup, but dispatch through the actual Ascend
    # class/MRO. Neither encoder batching nor embed_multimodal is replaced.
    npu_runner = object.__new__(module.OmniNPUModelRunner)
    npu_runner.__dict__.update(runner.__dict__)
    npu_runner._mm_rng_provider = None
    npu_runner.vllm_config = npu_runner.model_config = None
    mocker.patch.object(module.NPUModelRunner, "load_model", return_value=None)
    mocker.patch.object(module, "restore_native_rejection_sampler")
    mocker.patch.object(module, "enable_sp")
    for hook in (
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
    ):
        mocker.patch.object(npu_runner, hook)
    npu_runner.load_model()
    assert npu_runner._mm_rng_provider is not None
    refs = hunyuan_rng.images()
    a, b = make_request("A", refs, 42), make_request("B", refs[:1], 0)
    npu_runner.requests = {"A": a, "B": b}
    encode = mocker.spy(npu_runner.model, "embed_multimodal")
    outputs = npu_runner._execute_mm_encoder(hunyuan_rng.schedule({"B": [0], "A": [1, 2]}))
    handles = npu_runner._mm_rng_provider.handles
    assert encode.call_count == 1
    assert encode.call_args.kwargs[MM_RNG_ITEM_KEY].tolist() == [
        [handles["B"], 0],
        [handles["A"], 1],
        [handles["A"], 2],
    ]
    expected_a, _ = hunyuan_rng.reference(npu_runner, refs, 42)
    expected_b, _ = hunyuan_rng.reference(npu_runner, refs[:1], 0)
    hunyuan_rng.assert_latents(outputs, [*expected_b, *expected_a[1:]])
    assert all(MM_RNG_ITEM_KEY not in feature.data for req in (a, b) for feature in req.mm_features)
