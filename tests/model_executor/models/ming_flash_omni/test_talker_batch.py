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
