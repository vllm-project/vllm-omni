from __future__ import annotations

import queue
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.omni import Omni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


THREE_STAGE_META = [
    {"stage_type": "llm", "final_output": True, "final_output_type": "text"},
    {"stage_type": "llm", "final_output": False, "final_output_type": None},
    {"stage_type": "diffusion", "final_output": True, "final_output_type": "image"},
]

DIFFUSION_ONLY_META = [
    {"stage_type": "diffusion", "final_output": True, "final_output_type": "image"},
]

LLM_DIFFUSION_META = [
    {"stage_type": "llm", "final_output": True, "final_output_type": "text"},
    {"stage_type": "diffusion", "final_output": True, "final_output_type": "image"},
]


class FakeEngineOutput:
    def __init__(
        self,
        *,
        payload: str,
        finished: bool,
        images: list[str] | None = None,
        stage_durations: dict[str, float] | None = None,
    ) -> None:
        self.payload = payload
        self.finished = finished
        self.images = images or []
        self.stage_durations = stage_durations or {}


def make_output_msg(
    request_id: str,
    stage_id: int,
    *,
    payload: str,
    output_finished: bool,
    finished: bool | None = None,
    images: list[str] | None = None,
    metrics: Any = None,
) -> dict[str, Any]:
    if finished is None:
        finished = output_finished
    return {
        "type": "output",
        "request_id": request_id,
        "stage_id": stage_id,
        "engine_outputs": FakeEngineOutput(
            payload=payload,
            finished=output_finished,
            images=images,
        ),
        "finished": finished,
        "metrics": metrics,
    }


class FakeAsyncOmniEngine:
    def __init__(
        self,
        model: str = "dummy-model",
        *,
        stage_metadata: list[dict[str, Any]] | None = None,
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

        self.output_q: queue.Queue[dict[str, Any]] = queue.Queue()
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

    def try_get_output(self, timeout: float = 0.001) -> dict[str, Any] | None:
        try:
            return self.output_q.get_nowait()
        except queue.Empty:
            return None

    async def try_get_output_async(self) -> dict[str, Any] | None:
        return self.try_get_output()

    def get_stage_metadata(self, stage_id: int) -> dict[str, Any]:
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
        {
            "type": "error",
            "request_id": msg["request_id"],
            "stage_id": 0,
            "error": "engine boom",
        }
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
    assert serving_models.io_processor is io_processor
    assert serving_models.input_processor is input_processor


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
async def test_async_omni_propagates_engine_error(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(stage_metadata=THREE_STAGE_META, on_add_request=_enqueue_error_message)
    _patch_engine(monkeypatch, engine)

    app = AsyncOmni("dummy-model")
    try:
        with pytest.raises(RuntimeError, match="engine boom"):
            async for _ in app.generate(prompt="hello", request_id="req-1"):
                pass
    finally:
        app.shutdown()


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
