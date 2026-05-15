from __future__ import annotations

import argparse
import queue
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.engine.async_omni_engine import StageRuntimeInfo
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.omni_base import OmniEngineDeadError
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage_meta(*, stage_type: str, final_output: bool, final_output_type: str | None) -> StageRuntimeInfo:
    return StageRuntimeInfo(
        stage_type=stage_type,
        final_output=final_output,
        final_output_type=final_output_type,
    )


THREE_STAGE_META = [
    _stage_meta(stage_type="llm", final_output=True, final_output_type="text"),
    _stage_meta(stage_type="llm", final_output=False, final_output_type=None),
    _stage_meta(stage_type="diffusion", final_output=True, final_output_type="image"),
]

DIFFUSION_ONLY_META = [
    _stage_meta(stage_type="diffusion", final_output=True, final_output_type="image"),
]

LLM_DIFFUSION_META = [
    _stage_meta(stage_type="llm", final_output=True, final_output_type="text"),
    _stage_meta(stage_type="diffusion", final_output=True, final_output_type="image"),
]


def make_output_msg(
    request_id: str,
    stage_id: int,
    *,
    payload: str,
    output_finished: bool,
    finished: bool | None = None,
    images: list[str] | None = None,
    metrics: Any = None,
) -> OutputMessage:
    if finished is None:
        finished = output_finished
    final_output_type = "image" if images else "text"
    engine_output = OmniRequestOutput(
        request_id=request_id,
        finished=output_finished,
        final_output_type=final_output_type,
        images=images or [],
        stage_durations={},
    )
    engine_output.payload = payload
    return OutputMessage(
        request_id=request_id,
        stage_id=stage_id,
        engine_outputs=engine_output,
        finished=finished,
        metrics=metrics,
    )


class FakeAsyncOmniEngine:
    def __init__(
        self,
        model: str = "dummy-model",
        *,
        stage_metadata: list[StageRuntimeInfo] | None = None,
        default_sampling_params_list: list[Any] | None = None,
        on_add_request: Callable[[FakeAsyncOmniEngine, dict[str, Any]], None] | None = None,
        rpc_results: list[Any] | None = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.config_path = None
        self.stage_configs: list[Any] = []
        self.stage_metadata = stage_metadata or [THREE_STAGE_META[-1]]
        self.num_stages = len(self.stage_metadata)
        self.default_sampling_params_list = default_sampling_params_list or [
            SamplingParams(max_tokens=8) for _ in range(self.num_stages)
        ]
        self.supported_tasks = ("generate",)
        self.stage_clients = [SimpleNamespace(is_comprehension=False) for _ in range(self.num_stages)]
        self.stage_vllm_configs = [None for _ in range(self.num_stages)]
        self.output_processors = [SimpleNamespace(tokenizer=None) for _ in range(self.num_stages)]
        self.input_processor = None

        self.output_q: queue.Queue[Any] = queue.Queue()
        self.submitted: list[dict[str, Any]] = []
        self.aborted: list[list[str]] = []
        self.rpc_results = rpc_results or []
        self.on_add_request = on_add_request
        self.shutdown_called = False
        self._alive = True

    def add_request(
        self,
        request_id: str,
        prompt: Any,
        sampling_params_list: list[Any] | None = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
        **kwargs: Any,
    ) -> None:
        msg = {
            "request_id": request_id,
            "prompt": prompt,
            "sampling_params_list": sampling_params_list,
            "final_stage_id": final_stage_id,
            "arrival_time": arrival_time,
        }
        self.submitted.append(msg)
        if self.on_add_request is not None:
            self.on_add_request(self, msg)

    async def add_request_async(self, *args, **kwargs) -> None:
        self.add_request(*args, **kwargs)

    def try_get_output(self, timeout: float = 0.001) -> Any | None:
        try:
            return self.output_q.get_nowait()
        except queue.Empty:
            return None

    async def try_get_output_async(self) -> Any | None:
        return self.try_get_output()

    def get_stage_metadata(self, stage_id: int) -> StageRuntimeInfo:
        return self.stage_metadata[stage_id]

    def abort(self, request_ids: list[str]) -> None:
        self.aborted.append(list(request_ids))

    async def abort_async(self, request_ids: list[str]) -> None:
        self.abort(request_ids)

    async def collective_rpc_async(self, **_: Any) -> list[Any]:
        return list(self.rpc_results)

    def is_alive(self) -> bool:
        return self._alive

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._alive = False


def _patch_engine(monkeypatch: pytest.MonkeyPatch, engine: FakeAsyncOmniEngine) -> None:
    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.AsyncOmniEngine", lambda *args, **kwargs: engine)
    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.omni_snapshot_download", lambda model: model)


def test_direct_omni_with_nullified_parser_only_nulls_untyped_override_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
    from vllm_omni.entrypoints.omni import Omni

    captured: dict[str, Any] = {}

    def fake_engine(*args: Any, **kwargs: Any) -> FakeAsyncOmniEngine:
        captured.update(kwargs)
        return FakeAsyncOmniEngine()

    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.AsyncOmniEngine", fake_engine)
    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.omni_snapshot_download", lambda model: model)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--hsdp-shard-size", type=int, default=-1)
    nullify_stage_engine_defaults(parser)
    args = parser.parse_args([])
    args.model = "fake-model"

    Omni(**vars(args))

    assert captured["gpu_memory_utilization"] is None
    assert captured["hsdp_shard_size"] == -1
    assert "_cli_explicit_keys" not in captured


def test_from_cli_args_warns_and_forwards_without_internal_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    def fake_engine(*args: Any, **kwargs: Any) -> FakeAsyncOmniEngine:
        captured.update(kwargs)
        return FakeAsyncOmniEngine()

    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.AsyncOmniEngine", fake_engine)
    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.omni_snapshot_download", lambda model: model)

    args = argparse.Namespace(model="fake-model", gpu_memory_utilization=0.9, _cli_explicit_keys={"model"})
    with pytest.deprecated_call(match="from_cli_args"):
        Omni.from_cli_args(args)

    assert captured["gpu_memory_utilization"] == 0.9
    assert "_cli_explicit_keys" not in captured


def test_deprecated_from_cli_args_preserves_legacy_parser_nulling(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm_omni.entrypoints.omni import Omni

    captured: dict[str, Any] = {}

    def fake_engine(*args: Any, **kwargs: Any) -> FakeAsyncOmniEngine:
        captured.update(kwargs)
        return FakeAsyncOmniEngine()

    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.AsyncOmniEngine", fake_engine)
    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.omni_snapshot_download", lambda model: model)
    monkeypatch.setattr("sys.argv", ["prog"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--hsdp-shard-size", type=int, default=-1)
    args = parser.parse_args([])
    args.model = "fake-model"

    with pytest.deprecated_call(match="from_cli_args"):
        Omni.from_cli_args(args, parser=parser)

    assert captured["gpu_memory_utilization"] is None
    assert captured["hsdp_shard_size"] == -1


def _make_base():
    from vllm_omni.entrypoints.omni_base import OmniBase

    obj = object.__new__(OmniBase)
    obj.engine = MagicMock()
    obj.request_states = {}
    return obj


def _stage_spec(
    stage_id: int,
    *,
    payloads: list[str],
    finished: bool = False,
    image_payloads: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "stage_id": stage_id,
        "payloads": payloads,
        "finished": finished,
        "image_payloads": image_payloads or [],
    }


def _enqueue_outputs(
    engine: FakeAsyncOmniEngine,
    msg: dict[str, Any],
    *,
    stage_specs: list[dict[str, Any]],
) -> None:
    request_id = msg["request_id"]
    for spec in stage_specs:
        payloads = spec["payloads"]
        image_payloads = spec.get("image_payloads", [])
        last_idx = len(payloads) - 1

        for idx, payload_tmpl in enumerate(payloads):
            images = []
            if idx < len(image_payloads):
                images = [image_payloads[idx].format(request_id=request_id, idx=idx)]

            engine.output_q.put_nowait(
                make_output_msg(
                    request_id,
                    spec["stage_id"],
                    payload=payload_tmpl.format(request_id=request_id, idx=idx),
                    output_finished=(idx == last_idx),
                    finished=bool(spec.get("finished")) and idx == last_idx,
                    images=images,
                )
            )


def _enqueue_omni_final_only_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    sampling_params_list = msg["sampling_params_list"]
    llm_streaming = any(params.output_kind != RequestOutputKind.FINAL_ONLY for params in sampling_params_list[:2])
    stage0_count = 3 if llm_streaming else 1
    stage1_count = 3 if llm_streaming else 1

    _enqueue_outputs(
        engine,
        msg,
        stage_specs=[
            _stage_spec(0, payloads=[f"{{request_id}}-stage0-{idx}" for idx in range(stage0_count)]),
            _stage_spec(1, payloads=[f"{{request_id}}-stage1-{idx}" for idx in range(stage1_count)]),
            _stage_spec(
                2,
                payloads=["{request_id}-stage2-final"],
                finished=True,
                image_payloads=["{request_id}-img-final"],
            ),
        ],
    )


def _enqueue_omni_llm_diffusion_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    sampling_params_list = msg["sampling_params_list"]
    llm_streaming = sampling_params_list[0].output_kind != RequestOutputKind.FINAL_ONLY
    stage0_count = 3 if llm_streaming else 1

    _enqueue_outputs(
        engine,
        msg,
        stage_specs=[
            _stage_spec(0, payloads=[f"{{request_id}}-text-{idx}" for idx in range(stage0_count)]),
            _stage_spec(
                1,
                payloads=["{request_id}-image-final"],
                finished=True,
                image_payloads=["{request_id}-image"],
            ),
        ],
    )


def _enqueue_async_three_stage_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    _enqueue_outputs(
        engine,
        msg,
        stage_specs=[
            _stage_spec(0, payloads=[f"{{request_id}}-stage0-{idx}" for idx in range(3)]),
            _stage_spec(1, payloads=[f"{{request_id}}-stage1-{idx}" for idx in range(3)]),
            _stage_spec(
                2,
                payloads=[f"{{request_id}}-stage2-{idx}" for idx in range(3)],
                finished=True,
                image_payloads=[f"{{request_id}}-img-{idx}" for idx in range(3)],
            ),
        ],
    )


def _enqueue_async_finish_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    _enqueue_outputs(
        engine,
        msg,
        stage_specs=[
            _stage_spec(0, payloads=["{request_id}-stage0"]),
            _stage_spec(
                2,
                payloads=["{request_id}-stage2-final"],
                finished=True,
                image_payloads=["{request_id}-img-final"],
            ),
        ],
    )


def _enqueue_async_diffusion_only_output(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    _enqueue_outputs(
        engine,
        msg,
        stage_specs=[
            _stage_spec(
                0,
                payloads=["{request_id}-diffusion-final"],
                finished=True,
                image_payloads=["{request_id}-image"],
            )
        ],
    )


def _enqueue_async_llm_diffusion_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    _enqueue_outputs(
        engine,
        msg,
        stage_specs=[
            _stage_spec(0, payloads=[f"{{request_id}}-text-{idx}" for idx in range(3)]),
            _stage_spec(
                1,
                payloads=["{request_id}-image-final"],
                finished=True,
                image_payloads=["{request_id}-image"],
            ),
        ],
    )


def _enqueue_error_message(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    engine.output_q.put_nowait(
        ErrorMessage(
            request_id=msg["request_id"],
            stage_id=0,
            error="engine boom",
        )
    )


def _enqueue_fatal_error_message(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    engine.output_q.put_nowait(
        ErrorMessage(
            fatal=True,
            request_id=msg["request_id"],
            stage_id=2,
            error="engine dead",
        )
    )


@pytest.mark.asyncio
async def test_get_supported_tasks_returns_engine_supported_tasks():
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(supported_tasks=("generate", "speech"))

    supported_tasks = await omni.get_supported_tasks()

    assert supported_tasks == ("generate", "speech")


def test_model_config_and_vllm_config_forward_from_comprehension_stage():
    model_config = SimpleNamespace(model="Qwen/Qwen3-TTS")
    vllm_config = SimpleNamespace(model_config=model_config)
    renderer = SimpleNamespace(name="renderer")
    input_processor = SimpleNamespace(renderer=renderer)
    io_processor = SimpleNamespace(name="io-processor")
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        stage_clients=[SimpleNamespace(is_comprehension=False), SimpleNamespace(is_comprehension=True)],
        stage_vllm_configs=[None, vllm_config],
    )
    omni.input_processor = input_processor
    omni.io_processor = io_processor

    assert omni.vllm_config is vllm_config
    assert omni.model_config is model_config
    assert omni.renderer is renderer
    assert omni.input_processor is input_processor
    assert omni.io_processor is io_processor


def test_openai_serving_models_can_consume_async_omni_compat_attrs():
    model_config = SimpleNamespace(model="Qwen/Qwen3-TTS", max_model_len=32768)
    vllm_config = SimpleNamespace(model_config=model_config)
    renderer = SimpleNamespace(name="renderer")
    input_processor = SimpleNamespace(renderer=renderer)
    io_processor = SimpleNamespace(name="io-processor")
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        stage_clients=[SimpleNamespace(is_comprehension=True)],
        stage_vllm_configs=[vllm_config],
    )
    omni.input_processor = input_processor
    omni.io_processor = io_processor

    serving_models = OpenAIServingModels(
        engine_client=omni,
        base_model_paths=[BaseModelPath(name="tts-model", model_path="Qwen/Qwen3-TTS")],
    )

    assert serving_models.model_config is model_config
    assert serving_models.renderer is renderer
    assert serving_models.input_processor is input_processor
    # vLLM 0.20 keeps io_processor on the engine client instead of copying it.
    assert serving_models.engine_client.io_processor is io_processor


def test_get_diffusion_od_config_returns_diffusion_stage_config():
    diffusion_od_config = object()
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        stage_clients=[
            SimpleNamespace(stage_type="llm"),
            SimpleNamespace(stage_type="diffusion", _engine=SimpleNamespace(od_config=diffusion_od_config)),
        ]
    )

    assert omni.get_diffusion_od_config() is diffusion_od_config


@pytest.mark.asyncio
async def test_async_omni_yields_only_final_stage_outputs(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        on_add_request=lambda eng, msg: _enqueue_outputs(
            eng,
            msg,
            stage_specs=[
                _stage_spec(1, payloads=["non-final"]),
                _stage_spec(2, payloads=["final"], finished=True, image_payloads=["final-img"]),
            ],
        ),
    )
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        outputs = []
        async for output in app.generate(prompt="hello", request_id="req-1"):
            outputs.append(output)
    finally:
        app.shutdown()

    assert [output.stage_id for output in outputs] == [2]
    assert [output.request_output.payload for output in outputs] == ["final"]
    assert "req-1" not in app.request_states


@pytest.mark.asyncio
async def test_async_omni_accepts_multiple_final_stage_streams(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META, on_add_request=_enqueue_async_three_stage_outputs)
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        outputs = []
        async for output in app.generate(prompt="hello", request_id="req-1"):
            outputs.append(output)
    finally:
        app.shutdown()

    assert [output.stage_id for output in outputs] == [0, 0, 0, 2, 2, 2]
    assert [output.request_output.payload for output in outputs] == [
        "req-1-stage0-0",
        "req-1-stage0-1",
        "req-1-stage0-2",
        "req-1-stage2-0",
        "req-1-stage2-1",
        "req-1-stage2-2",
    ]


@pytest.mark.asyncio
async def test_async_omni_stops_on_final_stage_finished(monkeypatch: pytest.MonkeyPatch):
    # Intentionally jump from stage 0 to stage 2: stage 1 is a non-final stage
    # and should be filtered out from the client-visible output stream.
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META, on_add_request=_enqueue_async_finish_outputs)
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        outputs = []
        async for output in app.generate(prompt="hello", request_id="req-1"):
            outputs.append(output)
    finally:
        app.shutdown()

    assert [output.request_output.payload for output in outputs] == [
        "req-1-stage0",
        "req-1-stage2-final",
    ]
    assert "req-1" not in app.request_states


@pytest.mark.asyncio
async def test_async_omni_diffusion_only_yields_single_image_output(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(
        stage_metadata=DIFFUSION_ONLY_META,
        on_add_request=_enqueue_async_diffusion_only_output,
    )
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        outputs = []
        async for output in app.generate(prompt="hello", request_id="req-1"):
            outputs.append(output)
    finally:
        app.shutdown()

    assert len(outputs) == 1
    assert outputs[0].stage_id == 0
    assert outputs[0].final_output_type == "image"
    assert outputs[0].images == ["req-1-image"]
    assert outputs[0].request_output.payload == "req-1-diffusion-final"


@pytest.mark.asyncio
async def test_async_omni_llm_diffusion_yields_text_stream_then_image(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(
        stage_metadata=LLM_DIFFUSION_META,
        on_add_request=_enqueue_async_llm_diffusion_outputs,
    )
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        outputs = []
        async for output in app.generate(prompt="hello", request_id="req-1"):
            outputs.append(output)
    finally:
        app.shutdown()

    assert [output.stage_id for output in outputs] == [0, 0, 0, 1]
    assert [output.final_output_type for output in outputs] == ["text", "text", "text", "image"]
    assert [output.request_output.payload for output in outputs] == [
        "req-1-text-0",
        "req-1-text-1",
        "req-1-text-2",
        "req-1-image-final",
    ]
    assert outputs[-1].images == ["req-1-image"]
    assert "req-1" not in app.request_states


@pytest.mark.asyncio
async def test_async_omni_abort_forwards_to_engine(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META)
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        app.request_states["req-1"] = object()
        await app.abort("req-1")
    finally:
        app.shutdown()

    assert engine.aborted == [["req-1"]]
    assert "req-1" not in app.request_states


@pytest.mark.asyncio
async def test_async_omni_propagates_fatal_error_context(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META, on_add_request=_enqueue_fatal_error_message)
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        with pytest.raises(EngineDeadError, match="engine dead") as exc_info:
            async for _ in app.generate(prompt="hello", request_id="req-1"):
                pass
    finally:
        app.shutdown()

    assert isinstance(exc_info.value, OmniEngineDeadError)
    assert str(exc_info.value) == "engine dead"
    assert getattr(exc_info.value, "error_stage_id") == 2


def test_omni_generate_py_generator_yields_final_outputs_for_each_request(monkeypatch: pytest.MonkeyPatch):
    sampling_params = [SamplingParams(max_tokens=8) for _ in range(3)]
    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        default_sampling_params_list=sampling_params,
        on_add_request=_enqueue_omni_final_only_outputs,
    )
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    outputs = list(app.generate(["p1", "p2"], py_generator=True, use_tqdm=False))

    assert len(outputs) == 4
    assert [output.stage_id for output in outputs] == [0, 2, 0, 2]
    assert [output.request_output.payload for output in outputs] == [
        f"{engine.submitted[0]['request_id']}-stage0-0",
        f"{engine.submitted[0]['request_id']}-stage2-final",
        f"{engine.submitted[1]['request_id']}-stage0-0",
        f"{engine.submitted[1]['request_id']}-stage2-final",
    ]
    assert engine.shutdown_called is True


def test_omni_generate_returns_list_when_not_using_generator(monkeypatch: pytest.MonkeyPatch):
    sampling_params = [SamplingParams(max_tokens=8) for _ in range(3)]
    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        default_sampling_params_list=sampling_params,
        on_add_request=_enqueue_omni_final_only_outputs,
    )
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        outputs = app.generate(["p1", "p2"], py_generator=False, use_tqdm=False)
    finally:
        app.shutdown()

    assert isinstance(outputs, list)
    assert len(outputs) == 4
    assert [output.stage_id for output in outputs] == [0, 2, 0, 2]


def test_omni_generate_diffusion_only_yields_single_image_per_request(monkeypatch: pytest.MonkeyPatch):
    sampling_params = [SamplingParams(max_tokens=8)]
    engine = FakeAsyncOmniEngine(
        stage_metadata=DIFFUSION_ONLY_META,
        default_sampling_params_list=sampling_params,
        on_add_request=_enqueue_async_diffusion_only_output,
    )
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    outputs = list(app.generate(["p1", "p2"], py_generator=True, use_tqdm=False))

    assert len(outputs) == 2
    assert [output.stage_id for output in outputs] == [0, 0]
    assert [output.final_output_type for output in outputs] == ["image", "image"]
    assert [output.request_output.payload for output in outputs] == [
        f"{engine.submitted[0]['request_id']}-diffusion-final",
        f"{engine.submitted[1]['request_id']}-diffusion-final",
    ]
    assert [output.images for output in outputs] == [
        [f"{engine.submitted[0]['request_id']}-image"],
        [f"{engine.submitted[1]['request_id']}-image"],
    ]
    assert engine.shutdown_called is True


def test_omni_generate_llm_diffusion_yields_final_text_then_image_per_request(
    monkeypatch: pytest.MonkeyPatch,
):
    sampling_params = [SamplingParams(max_tokens=8) for _ in range(2)]
    engine = FakeAsyncOmniEngine(
        stage_metadata=LLM_DIFFUSION_META,
        default_sampling_params_list=sampling_params,
        on_add_request=_enqueue_omni_llm_diffusion_outputs,
    )
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    outputs = list(app.generate(["p1", "p2"], py_generator=True, use_tqdm=False))

    assert len(outputs) == 4
    assert [output.stage_id for output in outputs] == [0, 1, 0, 1]
    assert [output.final_output_type for output in outputs] == ["text", "image", "text", "image"]
    assert [output.request_output.payload for output in outputs] == [
        f"{engine.submitted[0]['request_id']}-text-0",
        f"{engine.submitted[0]['request_id']}-image-final",
        f"{engine.submitted[1]['request_id']}-text-0",
        f"{engine.submitted[1]['request_id']}-image-final",
    ]
    assert [output.images for output in outputs] == [
        [],
        [f"{engine.submitted[0]['request_id']}-image"],
        [],
        [f"{engine.submitted[1]['request_id']}-image"],
    ]
    assert engine.submitted[0]["sampling_params_list"][0].output_kind == RequestOutputKind.FINAL_ONLY
    assert engine.shutdown_called is True


def test_omni_abort_forwards_to_engine(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META)
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        app.request_states["req-1"] = object()
        app.abort("req-1")
    finally:
        app.shutdown()

    assert engine.aborted == [["req-1"]]
    assert "req-1" not in app.request_states


def test_omni_forces_final_only_on_llm_stages(monkeypatch: pytest.MonkeyPatch):
    sampling_params = [SamplingParams(max_tokens=8) for _ in range(3)]
    original_diffusion_output_kind = sampling_params[2].output_kind
    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        default_sampling_params_list=sampling_params,
        on_add_request=_enqueue_omni_final_only_outputs,
    )
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        outputs = list(app.generate(["p1"], py_generator=True, use_tqdm=False))
    finally:
        if not engine.shutdown_called:
            app.shutdown()

    submitted_params = engine.submitted[0]["sampling_params_list"]
    assert submitted_params[0].output_kind == RequestOutputKind.FINAL_ONLY
    assert submitted_params[1].output_kind == RequestOutputKind.FINAL_ONLY
    assert submitted_params[2].output_kind == original_diffusion_output_kind
    assert len(outputs) == 2


def test_fatal_error_raises_engine_dead():
    base = _make_base()
    msg = ErrorMessage(error="orchestrator crashed", fatal=True)

    with pytest.raises(EngineDeadError, match="orchestrator crashed"):
        base._handle_output_message(msg)


def test_non_fatal_error_raises_runtime():
    base = _make_base()
    msg = ErrorMessage(error="something wrong")

    with pytest.raises(RuntimeError, match="something wrong"):
        base._handle_output_message(msg)


def test_async_omni_errored_property_alive():
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        is_alive=lambda: True,
        stage_clients=[SimpleNamespace(is_comprehension=False)],
    )

    assert omni.errored is False


def test_async_omni_errored_property_dead_engine():
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        is_alive=lambda: False,
        stage_clients=[SimpleNamespace(is_comprehension=False)],
    )

    assert omni.errored is True


def test_async_omni_errored_property_dead_stage():
    omni = object.__new__(AsyncOmni)
    dead_stage = SimpleNamespace(is_comprehension=False, _engine_dead=True)
    omni.engine = SimpleNamespace(
        is_alive=lambda: True,
        stage_clients=[dead_stage],
    )

    assert omni.errored is True


def _enqueue_stage_error(
    engine: FakeAsyncOmniEngine,
    msg,
    *,
    error_text: str,
    kill_engine: bool = False,
):
    """Enqueue a stage error output, optionally killing the engine."""
    if kill_engine:
        engine._alive = False
    engine_output = OmniRequestOutput.from_error(msg["request_id"], error_text)
    engine_output.payload = ""
    engine.output_q.put_nowait(
        OutputMessage(
            request_id=msg["request_id"],
            stage_id=0,
            engine_outputs=engine_output,
            finished=False,
        )
    )


@pytest.mark.asyncio
async def test_async_omni_propagates_engine_dead_error(monkeypatch: pytest.MonkeyPatch):
    """When the engine is dead and an error output arrives, ``generate()``
    must raise ``EngineDeadError`` (not plain ``RuntimeError``)."""

    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        on_add_request=lambda eng, msg: _enqueue_stage_error(eng, msg, error_text="worker OOM", kill_engine=True),
    )
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        with pytest.raises(EngineDeadError, match="worker OOM"):
            async for _ in app.generate(prompt="hello", request_id="req-dead"):
                pass
    finally:
        app.shutdown()


@pytest.mark.asyncio
async def test_async_omni_propagates_engine_generate_error(monkeypatch: pytest.MonkeyPatch):
    """When the engine is alive but a stage error occurs, ``generate()``
    must raise ``EngineGenerateError`` (recoverable, not ``EngineDeadError``)."""

    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        on_add_request=lambda eng, msg: _enqueue_stage_error(eng, msg, error_text="diffusion step failed"),
    )
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        with pytest.raises(EngineGenerateError):
            async for _ in app.generate(prompt="hello", request_id="req-recover"):
                pass
    finally:
        app.shutdown()


# ───────── OmniBase.check_health() aggregation ─────────


def test_check_health_passes_when_all_healthy():
    base = _make_base()
    healthy_stage = MagicMock()
    healthy_stage.check_health = MagicMock()
    base.engine.is_alive.return_value = True
    base.engine.stage_clients = [healthy_stage]
    base.check_health()  # should not raise


def test_check_health_raises_when_stage_dead():
    base = _make_base()
    dead_stage = MagicMock()
    dead_stage.check_health = MagicMock(side_effect=EngineDeadError("Stage-1 dead"))
    base.engine.is_alive.return_value = True
    base.engine.stage_clients = [dead_stage]
    with pytest.raises(EngineDeadError, match="Stage-1 dead"):
        base.check_health()


def test_check_health_raises_when_orchestrator_dead():
    base = _make_base()
    base.engine.is_alive.return_value = False
    base.engine.stage_clients = []
    with pytest.raises(EngineDeadError, match="not alive"):
        base.check_health()


# ───────── OmniBase.errored property ─────────


def test_omni_base_errored_false_when_alive():
    base = _make_base()
    base.engine.is_alive.return_value = True
    base.engine.stage_clients = [SimpleNamespace()]
    assert base.errored is False


def test_omni_base_errored_true_when_orchestrator_dead():
    base = _make_base()
    base.engine.is_alive.return_value = False
    base.engine.stage_clients = []
    assert base.errored is True


def test_omni_base_errored_true_when_stage_engine_dead():
    base = _make_base()
    base.engine.is_alive.return_value = True
    dead_stage = SimpleNamespace(_engine_dead=True)
    base.engine.stage_clients = [dead_stage]
    assert base.errored is True


def test_omni_base_errored_true_when_stage_resources_engine_dead():
    base = _make_base()
    base.engine.is_alive.return_value = True
    dead_stage = SimpleNamespace(resources=SimpleNamespace(engine_dead=True))
    base.engine.stage_clients = [dead_stage]
    assert base.errored is True


# ───────── Omni (sync) EngineDeadError / EngineGenerateError ─────────


def test_omni_propagates_engine_dead_error(monkeypatch: pytest.MonkeyPatch):
    """When the engine is dead and a stage error output arrives,
    ``Omni.generate()`` must raise ``EngineDeadError``."""
    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        on_add_request=lambda eng, msg: _enqueue_stage_error(eng, msg, error_text="worker OOM", kill_engine=True),
    )
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        with pytest.raises(EngineDeadError, match="worker OOM"):
            list(app.generate(["hello"], py_generator=False, use_tqdm=False))
    finally:
        app.shutdown()


def test_omni_propagates_engine_generate_error(monkeypatch: pytest.MonkeyPatch):
    """When the engine is alive but a stage error occurs,
    ``Omni.generate()`` must raise ``EngineGenerateError`` (recoverable)."""
    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        on_add_request=lambda eng, msg: _enqueue_stage_error(eng, msg, error_text="diffusion step failed"),
    )
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        with pytest.raises(EngineGenerateError):
            list(app.generate(["hello"], py_generator=False, use_tqdm=False))
    finally:
        app.shutdown()


def test_omni_errored_property_alive(monkeypatch: pytest.MonkeyPatch):
    """Omni.errored (inherited from OmniBase) returns False when healthy."""
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META)
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        assert app.errored is False
    finally:
        app.shutdown()


def test_omni_errored_property_dead_engine(monkeypatch: pytest.MonkeyPatch):
    """Omni.errored returns True when the orchestrator is dead."""
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META)
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        engine._alive = False
        assert app.errored is True
    finally:
        app.shutdown()


def test_omni_errored_property_dead_stage(monkeypatch: pytest.MonkeyPatch):
    """Omni.errored returns True when a stage client is marked dead."""
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META)
    _patch_engine(monkeypatch, engine)

    app = Omni("dummy-model")
    try:
        engine.stage_clients[0]._engine_dead = True
        assert app.errored is True
    finally:
        app.shutdown()
