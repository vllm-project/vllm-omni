# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ROCm waveform parity across eager decoder startup initialization."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.qwen3_tts.test_qwen3_tts_incremental_decode import _make_small_decoder
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"rocm": "MI325"}, num_cards=1)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("prefix_frames", [0, 32])
@torch.inference_mode()
def test_eager_warmup_preserves_streaming_waveforms(dtype, prefix_frames):
    torch.manual_seed(42)
    decoder = _make_small_decoder().to(device="cuda", dtype=dtype)
    decoder._initial_codec_chunk_frames = 1
    decoder._incremental_chunk_frames = 25
    decoder._incremental_chunk_ramp = []
    decoder.capture_first_audio_state_only = True
    owner = SimpleNamespace(
        decoder=decoder,
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=8)),
        _decode_batch_max_size=4,
        _decoder_sliding_window=72,
        _num_quantizers=2,
        _decode_chunk_frames=300,
        _decode_left_context_frames=25,
        decoder_cudagraph_modes=("icl", "xvec"),
    )
    # Include a partial batch, padded reference prefix, window growth, rolling
    # decode and a final partial chunk through the actual waveform decoder.
    codes = [torch.randint(0, 32, (3, 2, frames), device="cuda") for frames in (prefix_frames + 1, 25, 25, 25, 25, 7)]

    def decode():
        caches = [{"prefix_frames": prefix_frames} for _ in range(3)]
        outputs: list[torch.Tensor] = []
        for chunk in codes:
            outputs.extend(
                output.clone()
                for output in decoder.batched_chunked_decode(
                    chunk, [chunk.shape[-1]] * 3, caches=caches, max_batch_size=4
                )
            )
        return outputs

    expected = decode()
    Qwen3TTSCode2Wav._warmup_eager_decoder(owner, torch.device("cuda"))
    actual = decode()
    assert len(actual) == len(expected)
    for before, after in zip(expected, actual, strict=True):
        assert torch.isfinite(after).all()
        torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-5)
