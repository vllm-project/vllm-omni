# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import inspect
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_OBJECT_COLLECTIVE_NAMES = frozenset(
    {
        "all_gather_object",
        "broadcast_object_list",
        "gather_object",
        "recv_object_list",
        "scatter_object_list",
        "send_object_list",
    }
)


def _collective_worker(rank, rendezvous, backend):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    from vllm_omni.errors import OmniClientError
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_executor.models.minimax_h3.conditioning import MiniMaxH3EncoderMediaInput
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        current_omni_platform.set_device(device)
    dist.init_process_group(
        backend, init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        module._dit_rank_world = lambda: (dist.group.WORLD, rank, 2)
        module._dit_cpu_rank_world = lambda: (dist.group.WORLD, rank, 2)

        class VisualCodec:
            fail = False

            def is_distributed_enabled(self):
                return True

            def collective(self):
                value = torch.ones(1, device=device)
                dist.all_reduce(value)
                assert value.item() == 2

            def encode_image(self, image):
                self.collective()
                assert image.size == (32, 32)
                return torch.ones(1, 96)

            def encode_video(self, frames):
                self.collective()
                assert frames.shape == (4, 32, 32, 3)
                if self.fail and rank == 1:
                    raise ValueError("codec failed on rank 1")
                return torch.full((1, 96), 2.0), (1, 2, 2)

        class AudioCodec:
            def encode_waveform(self, waveform, sample_rate):
                assert rank == 0
                assert sample_rate == 32000
                assert waveform.shape == (64000,)
                return torch.full((160, 32), 3.0), 80

        model = object.__new__(module.MiniMaxH3Pipeline)
        torch.nn.Module.__init__(model)
        model.device = device
        model.od_config = SimpleNamespace(diffusion_offload_config=None, enable_layerwise_offload=False)
        model.video_vae = VisualCodec()
        model.audio_vae = AudioCodec()
        model.partition = "combined"
        model.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
        model.text_encoder_tp_size = 2
        model.text_encoder_group = SimpleNamespace(
            world_size=2, ranks=[0, 1], rank_in_group=rank, device_group=dist.group.WORLD
        )
        model.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
        model.tokenizer = lambda text, **kwargs: {"input_ids": [1, 2, 3]}

        def encode_text(ids, kwargs):
            assert ids.tolist() == [1, 2, 3]
            assert not kwargs
            hidden = torch.ones(3, 5120, device=device, dtype=torch.bfloat16)
            dist.all_reduce(hidden)
            return hidden

        model._encode_text_hidden = encode_text
        sampling = OmniDiffusionSamplingParams(
            height=32, width=32, num_frames=96, extra_args={"task": "t2va", "aspect_ratio": "1:1"}
        )
        text_conditioning, window_text = model._prepare_local_conditioning({"prompt": "test"}, sampling)
        assert window_text is None
        torch.testing.assert_close(text_conditioning.hidden_states, torch.full_like(text_conditioning.hidden_states, 2))
        assert text_conditioning.visual_condition is None

        media = (
            MiniMaxH3EncoderMediaInput(
                task="ref2va",
                height=32,
                width=32,
                num_frames=107,
                latent_t=32,
                audio_t=178,
                images=(torch.zeros(32, 32, 3, dtype=torch.uint8),),
                videos=(torch.zeros(4, 32, 32, 3, dtype=torch.uint8),),
                video_audios=(None,),
                audios=((torch.zeros(64000), 32000),),
            )
            if rank == 0
            else None
        )
        conditioning = model._encode_local_media(media)
        assert [block["kind"] for block in conditioning.ref_blocks] == ["image", "video", "audio"]
        torch.testing.assert_close(
            conditioning.visual_condition.cpu(), torch.cat([torch.ones(1, 96), torch.full((1, 96), 2.0)])
        )
        torch.testing.assert_close(conditioning.audio_condition.cpu(), torch.full((160, 32), 3.0))
        assert conditioning.audio_condition_lengths == (80,)

        model.video_vae.fail = True
        with pytest.raises(OmniClientError, match="codec failed on rank 1"):
            model._encode_local_media(media)
        with pytest.raises(OmniClientError, match="non-empty prompt"):
            model._prepare_local_conditioning({"prompt": ""}, sampling)
        model.video_vae.fail = False
        assert model._encode_local_media(media).visual_condition_shapes == ((1, 2, 2), (1, 2, 2))
    finally:
        dist.destroy_process_group()


def _run_collectives(tmp_path, backend):
    context = mp.spawn(_collective_worker, args=(str(tmp_path / "rendezvous"), backend), nprocs=2, join=False)
    deadline = time.monotonic() + 150
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("MiniMax H3 collectives did not finish within 150 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.cpu
def test_two_rank_conditioning_and_errors_gloo(tmp_path):
    _run_collectives(tmp_path, "gloo")


@pytest.mark.cpu
def test_object_collectives_use_the_cpu_group_and_tensor_collectives_keep_the_device_group(monkeypatch):
    """Object collectives must not run on the device (HCCL) group.

    ``broadcast_object_list`` and ``all_gather_object`` serialize the payload
    and run collectives over Python objects. On a device group they allocate
    device buffers and drive the device backend, which is what fails with
    ``EI0006 Communication_Error_Get_Socket`` on ``HcclBroadcast`` when the
    device backend has to cross an interconnect domain (issue #10645). The
    payload has to travel host-side over the gloo group instead.

    The tensor broadcasts are the opposite case: they move device tensors and
    have to stay on the device group. Both directions are pinned here, so a
    later change cannot swap them.

    Every object-collective site of the pipeline is driven, so a later partial
    revert of one of them is caught: the rank-0 failure broadcast, the media
    input distribution, the media conditioning broadcast, the local-media
    error-status gather and the local-conditioning header broadcast.
    """
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    device_group = object()
    cpu_group = object()
    world = SimpleNamespace(device_group=device_group, cpu_group=cpu_group)
    monkeypatch.setattr(module, "get_world_group", lambda: world)

    object_groups: list[object] = []
    tensor_groups: list[object] = []
    state = {"rank": 0, "calls": 0, "stop_at": 1}

    class _StopAtObjectCollectiveError(Exception):
        """Stops a driven method right after its object collective."""

    def fake_broadcast_object_list(obj_list, src=0, group=None, **kwargs):
        object_groups.append(group)
        state["calls"] += 1
        if state["calls"] >= state["stop_at"]:
            raise _StopAtObjectCollectiveError

    def fake_all_gather_object(output_list, obj, group=None, **kwargs):
        object_groups.append(group)
        state["calls"] += 1
        if state["calls"] >= state["stop_at"]:
            raise _StopAtObjectCollectiveError

    def fake_broadcast(tensor, src=0, group=None, **kwargs):
        tensor_groups.append(group)

    monkeypatch.setattr(module.dist, "broadcast_object_list", fake_broadcast_object_list)
    monkeypatch.setattr(module.dist, "all_gather_object", fake_all_gather_object)
    monkeypatch.setattr(module.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: state["rank"])
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)

    model = object.__new__(module.MiniMaxH3Pipeline)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.video_vae = SimpleNamespace(is_distributed_enabled=lambda: False)
    model.audio_vae = SimpleNamespace()
    model._broadcast_media_conditioning = lambda conditioning: conditioning
    monkeypatch.setattr(module, "encode_media", lambda media, **kwargs: SimpleNamespace())

    def drive(stop_at, call):
        """Run ``call`` until its object collective(s) happened."""
        state["calls"] = 0
        state["stop_at"] = stop_at
        with pytest.raises(_StopAtObjectCollectiveError):
            call()

    # The rank-0 failure broadcast.
    drive(1, lambda: module._broadcast_rank0_exception(None))
    # The media input distribution.
    media = SimpleNamespace(to_mm_tensors=lambda: [], to_metadata=lambda: {})
    drive(1, lambda: model._distribute_media_inputs(media))
    # The media conditioning broadcast.
    conditioning = module.MiniMaxH3EncoderMediaConditioning(
        task="t2va", height=1, width=1, num_frames=1, latent_t=1, audio_t=1
    )
    drive(1, lambda: module.MiniMaxH3Pipeline._broadcast_media_conditioning(model, conditioning))
    # The local-media error-status gather.
    drive(1, lambda: model._encode_local_media(object()))
    # The local-conditioning header broadcast: the rank-0 failure broadcast runs
    # first inside the method, so the header is its second object collective.
    state["rank"] = 1
    sampling = SimpleNamespace(extra_args={})
    drive(2, lambda: model._prepare_local_conditioning(None, sampling))
    state["rank"] = 0

    # The tensor broadcasts move device tensors and stay on the device group.
    module._broadcast_tensor(torch.ones(2, 3), dtype=torch.float32, device=torch.device("cpu"))

    assert object_groups == [cpu_group] * 6
    assert tensor_groups == [device_group, device_group]
    assert device_group not in object_groups
    assert cpu_group not in tensor_groups


def _is_object_collective(node: ast.Call) -> bool:
    """Whether ``node`` calls a ``torch.distributed`` object collective."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _OBJECT_COLLECTIVE_NAMES:
        return False
    owner = func.value
    if isinstance(owner, ast.Name):
        return owner.id == "dist"
    return (
        isinstance(owner, ast.Attribute)
        and owner.attr == "distributed"
        and isinstance(owner.value, ast.Name)
        and owner.value.id == "torch"
    )


def _object_collective_calls(source: str) -> list[tuple[ast.AST, ast.Call]]:
    """Pair every object-collective call of ``source`` with its enclosing scope."""
    calls: list[tuple[ast.AST, ast.Call]] = []
    scope: list[ast.AST] = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            scope.append(node)
            self.generic_visit(node)
            scope.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            scope.append(node)
            self.generic_visit(node)
            scope.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if _is_object_collective(node):
                calls.append((scope[-1] if scope else node, node))
            self.generic_visit(node)

    _Visitor().visit(ast.parse(source))
    return calls


def _group_names_bound_by(function: ast.AST, helper: str) -> set[str]:
    """Return the names that ``function`` binds from the ``helper`` group view."""
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        value = node.value
        if not (isinstance(value.func, ast.Name) and value.func.id == helper):
            continue
        for target in node.targets:
            elements = target.elts if isinstance(target, ast.Tuple) else [target]
            for element in elements:
                if isinstance(element, ast.Name):
                    names.add(element.id)
    return names


@pytest.mark.cpu
def test_every_object_collective_of_the_pipeline_takes_the_cpu_group():
    """No object collective of the pipeline may run on the device group.

    The behavioural case above drives the five object collectives that exist
    today, so a later partial revert of one of them is caught. It cannot
    catch a *new* object collective, because a new site is not driven by that
    case. This case reads the module source instead: every object-collective
    call has to take its group from ``_dit_cpu_rank_world()``, so a new site
    on the device group fails the case until it is routed host-side. Both
    spellings of the call are covered: ``dist.<name>`` and
    ``torch.distributed.<name>``.
    """
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    calls = _object_collective_calls(inspect.getsource(module))
    assert calls, "the pipeline has no object collective to check"
    for scope, call in calls:
        name = getattr(scope, "name", "<module>")
        group = next((keyword.value for keyword in call.keywords if keyword.arg == "group"), None)
        assert isinstance(group, ast.Name), (
            f"{name}: the object collective has to pass a group name that is bound in this scope"
        )
        cpu_names = _group_names_bound_by(scope, "_dit_cpu_rank_world")
        device_names = _group_names_bound_by(scope, "_dit_rank_world")
        assert group.id in cpu_names, f"{name}: {group.id} does not come from the CPU group view"
        assert group.id not in device_names, f"{name}: {group.id} also comes from the device group view"


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parallel
@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_MINIMAX_H3_COLLECTIVES") != "1",
    reason="set VLLM_TEST_MINIMAX_H3_COLLECTIVES=1 with two reserved GPUs",
)
def test_two_rank_conditioning_and_errors_nccl(tmp_path):
    if torch.accelerator.device_count() < 2:
        pytest.skip("requires two GPUs")
    _run_collectives(tmp_path, "nccl")
