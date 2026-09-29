# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration
from vllm_omni.worker.gpu_ar_model_runner import _snapshot_tensor_payload_to_cpu_async

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def test_async_packed_snapshot_survives_source_reuse():
    values = [
        torch.arange(12, device="cuda").view(1, 12),
        torch.empty((0, 12), device="cuda", dtype=torch.long),
        torch.arange(24, device="cuda").view(2, 12),
    ]
    expected = [v.cpu() for v in values]
    snapshot = _snapshot_tensor_payload_to_cpu_async(
        {"list": values, "tuple": tuple(values)}, copy_stream=torch.cuda.Stream(), pin_memory=True
    )
    for v in values:
        v.fill_(-99)
    snapshot.wait()
    torch.testing.assert_close(snapshot.payload, {"list": expected, "tuple": tuple(expected)}, rtol=0, atol=0)


@pytest.mark.parametrize("offsets", [[0, 1], [1, 4]])
def test_mtp_preserves_rows_and_snapshot_without_cuda_sync(monkeypatch, offsets):
    import vllm_omni.worker.gpu_model_runner as mod
    from tests.worker.test_omni_gpu_model_runner import _make_runner, _noop_forward_context

    monkeypatch.setattr(mod.current_omni_platform, "set_forward_context", _noop_forward_context)
    runner = _make_runner(req_ids=("r1", "r2"), hidden_size=4)
    runner.model.gpu_resident_buffer_keys = {("codes", "audio")}
    for name in ("talker_mtp_input_ids", "talker_mtp_inputs_embeds", "last_talker_hidden", "text_step"):
        buffer = getattr(runner, name)
        buffer.gpu = buffer.gpu.to("cuda")
    embeds = torch.ones((2, 4), device="cuda")
    codes = torch.arange(2, device="cuda").view(2, 1)
    runner.talker_mtp = lambda *args, **kwargs: (embeds, codes)
    output = torch.full((5, 4), -10.0, device="cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        runner._talker_mtp_forward(["r1", "r2"], output, offsets)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    expected = torch.full((5, 4), -10.0)
    expected[offsets] = 1.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    codes.fill_(999)
    assert runner.model_intermediate_buffer["r1"]["codes"]["audio"].item() == 0
    assert runner.model_intermediate_buffer["r2"]["codes"]["audio"].item() == 1


def test_mixed_stop_flags_do_not_synchronize_cuda():
    model = object.__new__(MossTTSLocalTalkerForGeneration)
    torch.nn.Module.__init__(model)
    model.n_vq = 12
    model.audio_pad_token_id = 1024
    valid = torch.arange(12, device="cuda").view(1, 12)
    stopped = torch.full_like(valid, 1024)
    infos = [{"audio_codes": {"current": valid}}, {}, {"audio_codes": {"current": stopped}}]
    hidden = torch.ones((3, 4), device="cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        output = model.make_omni_output(hidden, runtime_additional_information=infos)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert model._batch_should_continue.tolist() == [True, True, False]
    torch.testing.assert_close(output.multimodal_outputs["codes"]["audio"], [valid, valid[:0], stopped])
    assert all("current" not in info.get("audio_codes", {}) for info in infos)
    prefill = model.make_omni_output(hidden[:1], runtime_additional_information=[{}])
    assert model._batch_should_continue is None and prefill.multimodal_outputs == {}
