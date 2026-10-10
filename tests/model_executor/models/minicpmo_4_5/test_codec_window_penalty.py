# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA codec-window penalties against independently assembled V1 histories."""

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _apply_batched_repetition_penalty,
    _apply_codec_window_penalty_gpu,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("asynchronous", [False, True])
@torch.inference_mode()
def test_codec_rng_matches_v1_across_conditions_after_async_eos_lookahead(mocker, asynchronous):
    import numpy as np
    from vllm import SamplingParams
    from vllm.config import VllmConfig
    from vllm.v1.sample.sampler import Sampler as LegacySampler
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2 import MiniCPMO45SeededCodecSampler
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    device = torch.device("cuda")
    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.vllm_config = VllmConfig()
    talker.vllm_config.scheduler_config.async_scheduling = asynchronous
    talker._codec_eos_id = 63
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    reqs = mocker.Mock(spec=RequestState, index_to_req_id={0: "r", 2: "r"})
    base = mocker.Mock(spec=Sampler, req_states=reqs)
    core = MiniCPMO45SeededCodecSampler(base, talker)
    talker._mrv2_seeded_codec_sampler = core
    reference = torch.Generator(device=device).manual_seed(42)
    legacy = LegacySampler()
    core._rows = core._accepted = (0,)
    tensor = torch.zeros(1, device=device, dtype=torch.int32)
    for seq in range(3):
        # A returning request can move slots without changing its random stream.
        slot = 0 if seq == 0 else 2
        core.add_request(slot, SamplingParams(seed=42, temperature=1.0))
        talker._request_condition_states["r"] = {"condition_seq": seq}
        talker._request_audio_states["r"] = {}
        previous = 0
        for step in range(3 + (3 if asynchronous else 0)):
            lookahead = step >= 3
            logits = torch.linspace(-2, 2, 64, device=device).reshape(1, 64)
            if step >= 2:
                logits.fill_(float("-inf"))
                logits[:, 63] = 0.0
            base.apply_sampling_params.return_value = logits
            actual, _ = core.sample(
                logits, tensor, tensor, np.array([slot]), tensor, tensor, tensor, np.array([4]), False
            )
            if not lookahead:
                expected, _ = legacy.topk_topp_sampler(logits, {0: reference}, None, None)
                torch.testing.assert_close(actual, expected.long(), rtol=0, atol=0)
                assert torch.equal(core._generators["r"].get_state(), reference.get_state())
            batch = mocker.Mock(
                spec=InputBatch,
                query_start_loc_np=np.array([0, 1]),
                num_scheduled_tokens=np.array([1]),
                is_prefilling_np=np.array([step == 0]),
            )
            finalize = talker.mrv2_codec_history_finalizer(batch, [{"req_id": "r", "native_duplex": True}])
            payload = {
                "codes.audio": torch.tensor([[previous]]),
                "meta.codec_frame_valid": torch.tensor([step != 0 and not lookahead]),
            }
            finalize(payload, [1])
            previous = int(actual[0])
        if asynchronous:
            assert core._generators["r"].get_offset() == reference.get_offset() + 12
        else:
            assert torch.equal(core._generators["r"].get_state(), reference.get_state())
    core.on_requests_finished({"r"})
    assert not core._committed_offsets and not core._condition_seqs and not core._generators


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 16, 65])
@pytest.mark.parametrize("with_prefix", [False, True])
def test_fused_codec_penalty_matches_request_histories(dtype, batch_size, with_prefix):
    generator = torch.Generator().manual_seed(812)
    vocab_size, window = 6562, 16
    slots = torch.randperm(128, generator=generator)[:batch_size]
    all_ids = torch.randint(0, vocab_size, (128, 4096), generator=generator, dtype=torch.int32)
    ids = all_ids[:, ::2]
    prompt = torch.randint(0, 100, (128,), generator=generator, dtype=torch.int32)
    counts = torch.arange(128, dtype=torch.int32) % 33
    total = prompt + counts
    prefix = torch.randint(-2, vocab_size + 2, (128, window), generator=generator) if with_prefix else None
    penalties = torch.linspace(0.8, 1.2, 128)
    penalties[slots[::3]] = 1
    # Assemble each request's actual codec history without the device helper.
    histories = []
    for slot in slots.tolist():
        seed = prefix[slot] if prefix is not None else torch.empty(0, dtype=torch.long)
        history = torch.cat([seed, ids[slot, prompt[slot] : total[slot]]])[-window:]
        histories.append(history[(history >= 0) & (history < vocab_size)].cuda())
    values = torch.randn((batch_size, vocab_size * 2), generator=generator, dtype=dtype).cuda()[:, ::2]
    expected = _apply_batched_repetition_penalty(values, histories, penalty=penalties[slots].cuda(), window_size=window)
    _apply_codec_window_penalty_gpu(
        values,
        slots.int().cuda(),
        all_ids.cuda()[:, ::2],
        total.cuda(),
        prompt.cuda(),
        penalties.cuda(),
        window_size=window,
        prefix_history=prefix.cuda() if prefix is not None else None,
    )
    torch.testing.assert_close(values, expected, rtol=1e-6, atol=1e-7)
