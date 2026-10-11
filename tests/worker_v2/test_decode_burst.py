# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode-burst configuration and runner behavior against one-step eager."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.worker_v2.decode_burst import decode_burst_steps

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "v2,mode,processor,expected",
    [
        (False, "duplex", "minicpmo_4_5_omni.tts2code2wav_async_chunk", 1),
        (True, "turn", "minicpmo_4_5_omni.tts2code2wav_async_chunk", 1),
        (True, "duplex", "minicpmo_4_5_omni.tts2code2wav_async_chunk", 4),
        (True, "duplex", "qwen3_omni.thinker2talker", 1),
        (True, "duplex", None, 1),
    ],
)
def test_decode_burst_steps(mocker, v2, mode, processor, expected):
    config = VllmConfig()
    config.model_config = mocker.Mock(
        spec=OmniModelConfig,
        use_v2_model_runner=v2,
        session_mode=mode,
        custom_process_next_stage_input_func=(
            f"vllm_omni.model_executor.stage_input_processors.{processor}" if processor else None
        ),
    )
    assert decode_burst_steps(config) == expected


@pytest.mark.parametrize("stop_after", [1, 2, 5])
def test_runner_burst_matches_eager_and_stops_each_request(mocker, stop_after):
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _TalkerDecodeBurst
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    start = torch.tensor([10, 20])
    last = torch.tensor([[1], [3]])
    computed = start.clone()
    eos = 8
    batch = InputBatch.make_dummy(2, 2, InputBuffers(2, 2, torch.device("cpu")), is_padding=False)
    batch.num_computed_tokens_np = start.numpy().copy()
    batch.seq_lens_cpu_upper_bound = start + 1
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner._decode_burst_context = (object(), object())
    plan = _TalkerDecodeBurst(mocker.Mock(_codec_eos_id=eos), 4, [None, None])

    def draw(previous, position, limit):
        return torch.where(position == limit, eos, (previous + position * 3) % 7 + 1)

    def step(_scheduler, _descriptor, step_batch):
        # Replace only the one-token backend; run the real burst loop and merger.
        offset = step_spy.call_count
        np.testing.assert_array_equal(step_batch.num_computed_tokens_np, start.numpy() + offset)
        torch.testing.assert_close(step_batch.seq_lens_cpu_upper_bound, start + offset + 1)
        tokens = draw(last[:, 0], computed, start + torch.tensor([stop_after - 1, 100])).view(2, 1)
        output = SamplerOutput(tokens, None, None, torch.ones(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32))
        return output, {
            "codes": {"audio": last.clone()},
            "meta": {"codec_frame_valid": last[:, 0] != eos, "finished": tokens[:, 0] == eos},
        }

    def commit(indices, tokens, sampled, rejected, query_start_loc):
        assert torch.equal(indices, batch.idx_mapping)
        assert torch.equal(query_start_loc, batch.query_start_loc)
        computed.add_(1 - rejected)
        last.copy_(torch.where(sampled[:, None] > 0, tokens, last))

    step_spy = mocker.patch.object(runner, "_run_decode_burst_step", side_effect=step)
    mocker.patch.object(runner, "postprocess_sampled", side_effect=commit)
    first, payload = step(*runner._decode_burst_context, batch)
    merged, mm, layout = runner._run_decode_burst(plan, batch, first, payload)

    # Independent one-step eager reference, without the burst's masks or merger.
    for row, limit in enumerate((stop_after, 100)):
        previous = 1 if row == 0 else 3
        tokens, codes = [], []
        for position in range(int(start[row]), int(start[row]) + 4):
            codes.append(previous)
            previous = eos if position == int(start[row]) + limit - 1 else (previous + position * 3) % 7 + 1
            tokens.append(previous)
            if previous == eos:
                break
        count = len(tokens)
        assert (merged.num_sampled[row].item(), merged.num_rejected[row].item()) == (count, 4 - count)
        assert merged.sampled_token_ids[row, :count].tolist() == tokens
        assert computed[row] == start[row] + count and last[row, 0] == previous
        begin, end = layout.query_start_loc_np[row : row + 2]
        assert end - begin == layout.num_scheduled_tokens[row] == 4
        assert mm["codes"]["audio"][begin : begin + count, 0].tolist() == codes
        assert mm["meta"]["codec_frame_valid"][begin:end].tolist() == [True] * count + [False] * (4 - count)
        assert mm["meta"]["finished"][row].item() == (previous == eos)
    assert layout.num_tokens_after_padding == 8 and not layout.is_prefilling_np.any()
    np.testing.assert_array_equal(batch.num_computed_tokens_np, start.numpy())
    torch.testing.assert_close(batch.seq_lens_cpu_upper_bound, start + 1)
