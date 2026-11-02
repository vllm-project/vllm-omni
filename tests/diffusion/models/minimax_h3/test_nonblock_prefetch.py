# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import os
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import pytest
import torch

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion]
SOURCE = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"


def load(name, scope):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    prefix = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=prefix + [fn], type_ignores=[])), str(SOURCE), "exec"), scope
    )
    return scope[name]


def make_fixture(mocker, rank=0, world=2, manual=True, dlo=True, sequential=False, load_error=None, remote_error=None):
    events = []

    class MiniMaxH3Qwen3VLEncoder:
        def __init__(self):
            self._omni_layerwise_enabled = dlo
            self.prepare_packed_offload = mocker.Mock()

    encoder = MiniMaxH3Qwen3VLEncoder()

    @contextmanager
    def component_scope(component):
        events.append("load")
        if load_error:
            # Real component scope owns failed-load cleanup too.
            events.append("cleanup")
            raise load_error
        try:
            yield
        finally:
            events.append("cleanup")

    group = object()

    def reduce(status, **kwargs):
        assert status.device.type == "cpu"
        assert kwargs["group"] is group
        events.append("cpu status")
        if remote_error:
            status.fill_(0)

    def gather(output, local, **kwargs):
        assert kwargs["group"] is group
        events.append("error details")
        for i in range(world):
            output[i] = local if i == rank else None
        if remote_error:
            output[1] = {"type": "RuntimeError", "message": remote_error, "status_code": None, "error_type": None}

    dist = mocker.Mock(all_reduce=reduce, all_gather_object=gather, ReduceOp=mocker.Mock(MIN="min"))
    scope = {
        "torch": torch,
        "os": os,
        "ExitStack": ExitStack,
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "dist": dist,
        "MiniMaxH3Qwen3VLEncoder": MiniMaxH3Qwen3VLEncoder,
        "_dit_rank_world": lambda: (None, rank, world),
        "get_world_group": lambda: mocker.Mock(cpu_group=group),
    }
    coordinate = load("_coordinate_encoder_prefetch_error", scope)
    scope["_coordinate_encoder_prefetch_error"] = coordinate
    owner = mocker.Mock(
        text_encoder_tp_size=2,
        text_encoder=encoder,
        _model_cpu_offload_modules=[encoder] if sequential else [],
        _uses_manual_component_offload=lambda c: manual,
        _component_on_device=component_scope,
    )
    owner._encoder_non_block_prefetch_scope = MethodType(load("_encoder_non_block_prefetch_scope", scope), owner)
    return owner, events, scope


def test_outer_scope_owns_single_release_and_status_precedes_compute(mocker):
    owner, events, _ = make_fixture(mocker)
    for _ in range(2):
        with owner._encoder_non_block_prefetch_scope(0) as staged:
            assert staged
            events.extend(["cpu prepare", "encode"])
    assert events == ["load", "cpu status", "cpu prepare", "encode", "cleanup"] * 2
    assert owner.text_encoder.prepare_packed_offload.call_count == 2


@pytest.mark.parametrize("option", ["outside_tp", "resident", "sequential", "not_dlo"])
def test_unsupported_paths_do_not_load(mocker, option):
    rank = 2 if option == "outside_tp" else 0
    owner, events, _ = make_fixture(
        mocker,
        rank=rank,
        world=3,
        manual=option != "resident",
        dlo=option != "not_dlo",
        sequential=option == "sequential",
    )
    with owner._encoder_non_block_prefetch_scope(rank) as staged:
        assert not staged
    assert events == ["cpu status"]


def test_remote_load_failure_releases_acquired_scope_before_raise(mocker):
    owner, events, _ = make_fixture(mocker, remote_error="rank one failed")
    with pytest.raises(RuntimeError, match="rank one failed"):
        with owner._encoder_non_block_prefetch_scope(0):
            pytest.fail("must not enter preparation")
    assert events == ["load", "cpu status", "error details", "cleanup"]


def test_local_load_failure_retains_original_exception(mocker):
    error = ValueError("bad load")
    owner, events, _ = make_fixture(mocker, rank=1, load_error=error)
    with pytest.raises(ValueError) as exc:
        with owner._encoder_non_block_prefetch_scope(1):
            pytest.fail("must not enter preparation")
    assert exc.value is error
    assert events == ["load", "cleanup", "cpu status", "error details"]


def test_single_rank_does_not_need_process_group(mocker):
    owner, events, _ = make_fixture(mocker, world=1)
    with owner._encoder_non_block_prefetch_scope(0) as staged:
        assert staged
    assert events == ["load", "cleanup"]


@pytest.mark.parametrize("failure", [ValueError("bad input"), KeyboardInterrupt("cancelled")])
def test_body_failure_is_not_hidden_and_scope_released(mocker, failure):
    owner, events, _ = make_fixture(mocker)
    with pytest.raises(type(failure)) as exc:
        with owner._encoder_non_block_prefetch_scope(0):
            raise failure
    assert exc.value is failure
    assert events[-1] == "cleanup"


def test_already_staged_encode_does_not_reload_or_release(mocker, monkeypatch):
    owner, events, scope = make_fixture(mocker)
    owner.text_encoder.encode_ids = mocker.Mock(return_value=torch.ones(1))
    monkeypatch.setenv("VLLM_OMNI_H3_HIDDEN_DEVICE", "0")
    call = load("_encode_text_hidden", scope)
    with owner._encoder_non_block_prefetch_scope(0) as staged:
        call(owner, torch.tensor([1]), {}, encoder_staged=staged)
        assert events == ["load", "cpu status"]
    assert events == ["load", "cpu status", "cleanup"]
    owner.text_encoder.encode_ids.assert_called_once_with(torch.tensor([1]))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("bad_input", [False, True])
def test_encode_prompt_order_and_prepare_failure_cleanup(mocker, monkeypatch, enabled, bad_input):
    owner, events, scope = make_fixture(mocker)
    monkeypatch.setenv("VLLM_OMNI_H3_NONBLOCK_PREFETCH", "1" if enabled else "0")
    monkeypatch.setenv("VLLM_OMNI_H3_HIDDEN_DEVICE", "0")

    def preprocess_impl(*args, **kwargs):
        events.append("prepare")
        if bad_input:
            raise ValueError("input failed")
        return {"pixel_values": torch.ones(1), "image_grid_thw": torch.ones(1, 3, dtype=torch.long)}

    preprocess = mocker.Mock(side_effect=preprocess_impl, merge_size=1)

    def broadcast_error(error):
        if error:
            raise error

    def broadcast(value, **kwargs):
        events.append("broadcast")
        return value

    scope.update(
        _broadcast_rank0_exception=broadcast_error,
        _broadcast_tensor=broadcast,
        build_minimax_h3_presentation=lambda *a, **k: (torch.tensor([1]), torch.tensor([2])),
    )
    owner.processor = mocker.Mock(image_processor=preprocess)
    owner.tokenizer = None
    owner.device = "cpu"

    def distribute(ids, vision):
        events.append("distribute")
        return ids

    def encode(*args, **kwargs):
        events.append("encode")
        return torch.zeros(1, 4)

    owner._distribute_encode_inputs = distribute
    owner.text_encoder.encode_ids = encode
    owner._encode_text_hidden = MethodType(load("_encode_text_hidden", scope), owner)
    prepared = mocker.Mock(
        images=[1],
        qwen_videos=[],
        prompt="test",
        condition_labels=[],
        video_timestamps=[],
        media=mocker.Mock(task="ref2va"),
    )
    call = load("encode_prompt", scope)
    if bad_input:
        with pytest.raises(ValueError, match="input failed"):
            call(owner, prepared)
        assert events == (["load", "cpu status", "prepare", "cleanup"] if enabled else ["prepare"])
    else:
        hidden, tags = call(owner, prepared)
        assert torch.equal(hidden, torch.zeros(1, 4)) and tags.item() == 2
        expected = ["load", "cpu status", "prepare", "distribute", "encode", "broadcast", "broadcast", "cleanup"]
        if not enabled:
            expected = ["prepare", "distribute", "load", "encode", "cleanup", "broadcast", "broadcast"]
        assert events == expected


def test_eligibility_failure_is_also_coordinated(mocker):
    error = ValueError("invalid offload config")
    owner, events, _ = make_fixture(mocker)
    owner._uses_manual_component_offload = mocker.Mock(side_effect=error)
    with pytest.raises(ValueError) as exc:
        with owner._encoder_non_block_prefetch_scope(0):
            pytest.fail("must not enter preparation")
    assert exc.value is error
    assert events == ["cpu status", "error details"]
