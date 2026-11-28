# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.encoder_config import EncoderWeightTransfer
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _bare_encoder():
    encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    return encoder


@pytest.mark.parametrize("device_output", [False, True])
def test_return_placement_and_sdp_restore(mocker, device_output):
    encode = MiniMaxH3Qwen3VLEncoder.encode_ids
    raw = mocker.Mock(cpu=mocker.Mock(return_value="host result"))
    obj = mocker.Mock(
        is_loaded=True,
        cudnn_sdp=True,
        device_target=torch.device("cpu"),
        parameters=lambda: iter([torch.zeros(1)]),
        _encode=mocker.Mock(return_value=raw),
    )
    initial = torch.backends.cuda.cudnn_sdp_enabled()
    result = encode(obj, torch.tensor([1]), **({"return_on_device": True} if device_output else {}))
    assert result is raw if device_output else result == "host result"
    assert raw.cpu.call_count == (0 if device_output else 1)
    assert torch.backends.cuda.cudnn_sdp_enabled() == initial


@pytest.mark.parametrize("device_output", [False, True])
def test_exception_restores_sdp(mocker, device_output):
    encode = MiniMaxH3Qwen3VLEncoder.encode_ids
    obj = mocker.Mock(
        is_loaded=True,
        cudnn_sdp=True,
        device_target=torch.device("cpu"),
        parameters=lambda: iter([torch.zeros(1)]),
        _encode=mocker.Mock(side_effect=RuntimeError("failure")),
    )
    initial = torch.backends.cuda.cudnn_sdp_enabled()
    with pytest.raises(RuntimeError, match="failure"):
        encode(obj, torch.tensor([1]), return_on_device=device_output)
    assert torch.backends.cuda.cudnn_sdp_enabled() == initial


def test_forward_keeps_default_call_and_forwards_opt_in(mocker):
    forward = MiniMaxH3Qwen3VLEncoder.forward
    obj = mocker.Mock(encode_ids=mocker.Mock(return_value="result"))
    ids = torch.tensor([1])
    assert forward(obj, ids) == "result"
    assert "return_on_device" not in obj.encode_ids.call_args.kwargs
    forward(obj, ids, return_on_device=True)
    assert obj.encode_ids.call_args.kwargs["return_on_device"] is True


@pytest.mark.parametrize("branch", ["resident", "manual", "sequential"])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("device_type", ["npu", "cuda", "cpu"])
def test_pipeline_routing_lifetime_and_input_ownership(mocker, branch, fail, device_type):
    call = MiniMaxH3Pipeline._encode_text_hidden
    events: list[object] = []
    hidden = torch.arange(12).reshape(3, 4).to(torch.bfloat16)

    def encode(ids, **kwargs):
        events.append(("encode", kwargs.copy()))
        if fail:
            raise RuntimeError("failure")
        return hidden

    encoder = mocker.Mock(side_effect=encode)
    encoder.encode_ids.side_effect = encode

    @contextmanager
    def scope(component):
        events.append("load")
        try:
            yield
        finally:
            events.append("offload")

    obj = mocker.Mock(
        device=SimpleNamespace(type=device_type),
        text_encoder=encoder,
        _model_cpu_offload_modules=["module"] if branch == "sequential" else [],
        _uses_manual_component_offload=lambda c: branch == "manual",
        _component_on_device=scope,
        _encoder_weight_transfer=EncoderWeightTransfer.OFF,
    )
    kwargs = {"pixel_values": torch.zeros(1)}
    if fail:
        with pytest.raises(RuntimeError, match="failure"):
            call(obj, torch.tensor([1]), kwargs)
    else:
        assert call(obj, torch.tensor([1]), kwargs) is hidden
        assert torch.equal(hidden, torch.arange(12).reshape(3, 4).to(torch.bfloat16))
    passed = next(e[1] for e in events if isinstance(e, tuple))
    if device_type == "npu":
        assert passed["return_on_device"] is True
    else:
        assert "return_on_device" not in passed
    assert set(kwargs) == {"pixel_values"}
    if branch == "manual":
        assert events[0] == "load" and events[-1] == "offload"
    if branch == "sequential":
        encoder.assert_called_once()
        encoder.encode_ids.assert_not_called()
    if branch == "resident":
        encoder.load_to_device.assert_called_once()


def test_packed_transfer_prepares_on_normal_encoder_path(mocker):
    call = MiniMaxH3Pipeline._encode_text_hidden
    events = []
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    encoder = _bare_encoder()
    prepare_packed_offload = mocker.Mock(side_effect=lambda: events.append("prepare"))

    def encode_ids(input_ids, *, return_on_device=False):
        assert torch.equal(input_ids, torch.tensor([1]))
        assert return_on_device
        events.append("encode")
        return hidden

    encode_ids_mock = mocker.Mock(side_effect=encode_ids)
    setattr(encoder, "prepare_packed_offload", prepare_packed_offload)
    setattr(encoder, "encode_ids", encode_ids_mock)

    @contextmanager
    def scope(component):
        events.append("load")
        try:
            yield
        finally:
            events.append("offload")

    obj = mocker.Mock(
        device=SimpleNamespace(type="npu"),
        text_encoder=encoder,
        _model_cpu_offload_modules=[],
        _uses_manual_component_offload=lambda component: True,
        _component_on_device=scope,
        _encoder_weight_transfer=EncoderWeightTransfer.PACKED,
    )

    assert call(obj, torch.tensor([1]), {}) is hidden
    assert events == ["prepare", "load", "encode", "offload"]
    prepare_packed_offload.assert_called_once_with()
    encode_ids_mock.assert_called_once_with(torch.tensor([1]), return_on_device=True)


@pytest.mark.parametrize("branch", ["manual", "sequential"])
@pytest.mark.parametrize("fail", [False, True])
def test_real_module_hooks_and_component_cleanup(mocker, branch, fail):
    events = []

    class Encoder(MiniMaxH3Qwen3VLEncoder):
        cudnn_sdp = True

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.text_model = torch.nn.Identity()
            self.device_target = torch.device("cpu")

        def load_to_device(self):
            events.append("load")

        def offload_to_cpu(self):
            events.append("offload")
            self.weight = torch.nn.Parameter(torch.empty(0))

        def _encode(self, ids, **kwargs):
            if fail:
                raise RuntimeError("failure")
            return self.weight + ids.float()

    enc = Encoder()
    obj = mocker.Mock(
        device=SimpleNamespace(type="npu"),
        text_encoder=enc,
        _model_cpu_offload_modules=[enc] if branch == "sequential" else [],
        _uses_manual_component_offload=lambda c: True,
        _encoder_weight_transfer=EncoderWeightTransfer.OFF,
    )
    obj._component_on_device = MethodType(MiniMaxH3Pipeline._component_on_device, obj)
    if branch == "sequential":
        enc.register_forward_pre_hook(lambda m, args: m.load_to_device())
        enc.register_forward_hook(lambda m, args, out: m.offload_to_cpu(), always_call=True)
    call = MiniMaxH3Pipeline._encode_text_hidden
    if fail:
        with pytest.raises(RuntimeError, match="failure"):
            call(obj, torch.arange(4), {})
    else:
        hidden = call(obj, torch.arange(4), {})
        # The real scope/hooks have released the parameter; activation remains usable.
        assert enc.weight.numel() == 0
        assert torch.equal(hidden, torch.arange(4) + 1)
        assert not torch._C._is_alias_of(hidden, enc.weight)
    assert events == ["load", "offload"]


@pytest.mark.parametrize("mode", list(EncoderWeightTransfer))
@pytest.mark.parametrize("rank,tp_size,world_size", [(0, 1, 1), (0, 8, 8), (1, 8, 8), (7, 8, 8), (1, 1, 8)])
@pytest.mark.parametrize("device_type", ["npu", "cuda"])
def test_encode_prompt_preserves_backend_hidden_lifetime(mocker, mode, rank, tp_size, world_size, device_type):
    expected_hidden = torch.arange(12).reshape(3, 4).to(torch.bfloat16)
    expected_tags = torch.tensor([0, 1, 2])
    output_ref: weakref.ReferenceType[torch.Tensor] | None = None
    events = []
    target = SimpleNamespace(type=device_type)

    def encode(input_ids, vision_kwargs):
        nonlocal output_ref
        assert torch.equal(input_ids, torch.tensor([1, 2, 3]))
        hidden = expected_hidden.clone()
        output_ref = weakref.ref(hidden)
        events.append("encode")
        return hidden

    def broadcast(tensor, *, dtype, device):
        assert device is target
        if dtype == torch.bfloat16:
            if rank == 0:
                assert output_ref is not None
                assert tensor is output_ref()
            elif device_type == "npu" or rank >= tp_size:
                assert tensor is None
                # Check lifetime at receive allocation, not just the argument.
                assert output_ref is None or output_ref() is None
            else:
                assert output_ref is not None
                assert tensor is output_ref()
            events.append("receive")
            expected = expected_hidden
        else:
            assert dtype == torch.long
            assert tensor is None if rank != 0 else torch.equal(tensor, expected_tags)
            expected = expected_tags
        return tensor if rank == 0 else expected.clone()

    distribute = mocker.Mock(return_value=torch.tensor([1, 2, 3]))
    encoder = mocker.Mock(side_effect=encode)
    pipeline = SimpleNamespace(
        text_encoder_tp_size=tp_size,
        device=target,
        _encoder_weight_transfer=mode,
        _distribute_encode_inputs=distribute,
        _encode_text_hidden=encoder,
        tokenizer=object(),
        processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2)),
    )
    prepared = SimpleNamespace(
        images=[],
        qwen_videos=[],
        prompt="test",
        media=SimpleNamespace(task="test"),
        condition_labels=[],
        video_timestamps=[],
    )
    check_error = mocker.patch.object(pipeline_minimax_h3, "_broadcast_rank0_exception")
    mocker.patch.object(pipeline_minimax_h3, "_dit_rank_world", return_value=(None, rank, world_size))
    mocker.patch.object(pipeline_minimax_h3, "_broadcast_tensor", side_effect=broadcast)
    mocker.patch.object(
        pipeline_minimax_h3, "build_minimax_h3_presentation", return_value=(torch.tensor([1, 2, 3]), expected_tags)
    )
    hidden, tags = MiniMaxH3Pipeline.encode_prompt(pipeline, prepared if rank == 0 else None)
    assert torch.equal(hidden.view(torch.uint8), expected_hidden.view(torch.uint8))
    assert torch.equal(tags, expected_tags)
    check_error.assert_called_once_with(None)
    assert events == (["encode", "receive"] if rank < tp_size else ["receive"])
    assert distribute.call_count == encoder.call_count == int(rank < tp_size)
