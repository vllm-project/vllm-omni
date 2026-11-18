# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA ordering of request inputs handed to the Ming batch producer."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.ming_flash_omni.talker_batch import (
    BatchedTalkerRequest,
    MingTalkerBatchEngine,
    TalkerSegment,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = pytest.mark.core_model


@hardware_test(res={"cuda": "H100"})
@pytest.mark.nvidia_only
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires NVIDIA CUDA FlashAttention and streams")
@torch.inference_mode()
def test_prefill_waits_for_request_embeddings_and_reference_latents(monkeypatch):
    device = torch.device("cuda")
    model = (
        Qwen2Model(
            Qwen2Config(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        .eval()
        .to(device)
    )
    observed_embeds, observed_references = [], []

    def initial_history(reference, device, dtype):
        observed_references.append(reference.clone())
        return reference.to(device=device, dtype=dtype)

    generator = SimpleNamespace(
        _model=model,
        _cfm=None,
        _aggregator=None,
        _stop_head=None,
        _max_cache_len=32,
        _config=SimpleNamespace(steps=1),
        patch_size=1,
        his_patch_size=1,
        latent_dim=1,
        _init_his_lat=initial_history,
    )
    engine = MingTalkerBatchEngine(generator, max_rows=1, device=device, use_cuda_graphs=False)

    def prefill(embeds, *args):
        observed_embeds.append(embeds.clone())
        return embeds

    monkeypatch.setattr(engine.llm, "forward", prefill)

    def request(embeds, reference):
        return BatchedTalkerRequest(
            [TalkerSegment(embeds, 1)],
            publish=lambda *args: None,
            fail=lambda exc: pytest.fail(str(exc)),
            prompt_wav_lat=reference,
            min_steps=0,
            stop_threshold=0.5,
            cfg=2.0,
            sigma=0.25,
            temperature=0.0,
            seed=None,
        )

    embeds = torch.zeros(1, 2, 32, device=device)
    reference = torch.zeros(1, 1, 1, device=device)
    # Warm the actual prefill allocations so allocation-time synchronization
    # cannot hide a missing dependency on the submitting stream.
    warm = request(embeds, reference)
    warm.row = 0
    engine.prefill = [warm]
    engine.stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(engine.stream):
        engine._run_prefill()
    engine.stream.synchronize()
    observed_embeds.clear()
    observed_references.clear()
    monkeypatch.setattr(engine, "_decode_step", lambda: setattr(engine, "closed", True))
    producer = torch.cuda.Stream(device)
    engine.start()
    try:
        with torch.cuda.stream(producer):
            torch.cuda._sleep(1_000_000_000)
            embeds.fill_(7.0)
            reference.fill_(9.0)
            engine.submit(request(embeds, reference))
        engine.thread.join(10)
        assert not engine.thread.is_alive()
        assert engine.error is None
        producer.synchronize()
        engine.stream.synchronize()
        torch.testing.assert_close(observed_embeds[0], torch.full_like(embeds[0], 7.0))
        torch.testing.assert_close(observed_references[0], torch.full_like(reference, 9.0))
    finally:
        with engine.cond:
            engine.closed = True
            engine.cond.notify_all()
        engine.thread.join(10)


@hardware_test(res={"cuda": "H100"})
@pytest.mark.nvidia_only
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires NVIDIA CUDA FlashAttention and streams")
@pytest.mark.parametrize("use_graphs", [False, True])
@torch.inference_mode()
def test_compaction_preserves_decode_across_cancellation_segments_and_row_reuse(monkeypatch, use_graphs):
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    model = (
        Qwen2Model(
            Qwen2Config(
                vocab_size=32,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        .eval()
        .to(device=device, dtype=dtype)
    )

    class SmallFlow:
        def sample(self, hs, history, noise, *args, **kwargs):
            return noise * 0.1 + hs[..., :8] * 0.1 + history * 0.25

    generator = SimpleNamespace(
        _model=model,
        _cfm=SmallFlow(),
        _aggregator=torch.nn.Linear(8, 128).to(device=device, dtype=dtype),
        _stop_head=torch.nn.Linear(128, 2).to(device=device, dtype=dtype),
        _max_cache_len=512,
        _config=SimpleNamespace(steps=1),
        patch_size=1,
        his_patch_size=1,
        latent_dim=8,
        _init_his_lat=lambda reference, device, dtype: reference.to(device=device, dtype=dtype),
    )
    engines = [
        MingTalkerBatchEngine(generator, max_rows=8, device=device, use_cuda_graphs=use_graphs) for _ in range(2)
    ]
    monkeypatch.setattr(engines[0], "_compact_rows", lambda: None)
    requests: list[dict[int, BatchedTalkerRequest]] = [{}, {}]
    emitted: list[dict[int, list[tuple[torch.Tensor, bool]]]] = [{}, {}]
    for engine in engines:
        engine.capture()

    def admit(ids):
        for request_id in ids:
            segments = [TalkerSegment(torch.randn(1, 5 + request_id, 128, device=device, dtype=dtype), 12)]
            if request_id in (7, 13):
                segments.append(TalkerSegment(torch.randn(1, 3, 128, device=device, dtype=dtype), 5))
            reference = torch.randn(1, 1, 8, device=device, dtype=dtype)
            for arm, engine in enumerate(engines):
                emitted[arm][request_id] = []
                output = emitted[arm][request_id]
                request = BatchedTalkerRequest(
                    segments,
                    publish=lambda latent, last, output=output: output.append((latent.clone(), last)),
                    fail=lambda exc: pytest.fail(str(exc)),
                    prompt_wav_lat=reference,
                    min_steps=100,
                    stop_threshold=2.0,
                    cfg=2.0,
                    sigma=0.25,
                    temperature=1.0,
                    seed=100 + request_id,
                )
                requests[arm][request_id] = request
                engine.submit(request)
                with torch.cuda.stream(engine.stream):
                    engine._admit(list(engine.pending))
                    engine.pending.clear()

    def advance():
        for engine in engines:
            with torch.cuda.stream(engine.stream):
                engine._drop_cancelled()
                if engine.prefill:
                    engine._run_prefill()
                if engine.rows:
                    engine._decode_step()
        for request_id in emitted[0]:
            expected, actual = emitted[0][request_id], emitted[1][request_id]
            assert len(actual) == len(expected)
            for (want, want_last), (got, got_last) in zip(expected, actual, strict=True):
                assert got_last == want_last
                torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
        table = engines[1].llm.block_table
        assert sorted(table[:, 0].tolist()) == list(range(0, 9 * 2, 2))

    try:
        admit(range(8))
        for _ in range(3):
            advance()
        for arm, engine in enumerate(engines):
            for request_id in range(1, 7):
                engine.cancel(requests[arm][request_id])
        advance()
        assert requests[0][7].row == 7 and requests[1][7].row == 1
        admit(range(8, 14))
        for _ in range(2):
            advance()
        for arm, engine in enumerate(engines):
            for request_id in (0, 8, 9, 10, 11, 12):
                engine.cancel(requests[arm][request_id])
        for _ in range(20):
            advance()
        for engine in engines:
            assert engine.rows == {} and not engine.prefill
            assert sorted(engine.free_rows) == list(range(8))
        for arm in range(2):
            assert len(emitted[arm][7]) == len(emitted[arm][13]) == 17
            assert emitted[arm][7][-1][1] and emitted[arm][13][-1][1]
    finally:
        for engine in engines:
            engine.close()
