# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


@torch.inference_mode()
def test_conv_graph_replay_fallback_and_output_ownership():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.platforms.npu.layers.codec_graph import NPUCodecGraphs
    from vllm_omni.platforms.npu.models.qwen3_tts import (
        _prepare_npu_code2wav_runtime,
    )

    _prepare_npu_code2wav_runtime()

    decoder = _make_decoder().to(device="npu", dtype=torch.bfloat16).eval()
    decoder.precompute_snake_caches()
    graphs = NPUCodecGraphs(decoder._conv_decode_eager, next(decoder.parameters()), 16, [(1, 2), (2, 4)])
    decoder._npu_conv_graphs = graphs
    try:
        for shape in [(1, 2, 16), (2, 4, 16), (1, 3, 16)]:
            x = torch.randn(shape, device="npu", dtype=torch.bfloat16)
            expected = decoder._conv_decode_eager(x)
            result = decoder._conv_decode(x)
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            retained = result.clone()
            decoder._conv_decode(torch.randn_like(x))
            torch.testing.assert_close(result, retained, atol=0, rtol=0)
        # An outer graph must capture the eager body, not replay an inner graph.
        x = torch.randn(1, 2, 16, device="npu", dtype=torch.bfloat16)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            decoder._conv_decode_eager(x)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        outer = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(outer, stream=stream):
                output = decoder._conv_decode(x)
            x.normal_()
            outer.replay()
            torch.testing.assert_close(output, decoder._conv_decode_eager(x), atol=0, rtol=0)
        finally:
            torch.npu.synchronize()
            outer.reset()
        # Module conversions drop graphs bound to the previous parameters.
        decoder.to(torch.float32)
        assert decoder._npu_conv_graphs is None
        assert not graphs.entries
    finally:
        graphs.close()
