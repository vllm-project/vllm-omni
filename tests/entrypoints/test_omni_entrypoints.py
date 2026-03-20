from __future__ import annotations

import queue
from types import SimpleNamespace
from typing import Any, Callable

import pytest
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
        on_add_request: Callable[["FakeAsyncOmniEngine", dict[str, Any]], None] | None = None,
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


def _enqueue_async_three_stage_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    request_id = msg["request_id"]
    for idx in range(3):
        engine.output_q.put_nowait(
            make_output_msg(
                request_id,
                0,
                payload=f"{request_id}-stage0-{idx}",
                output_finished=(idx == 2),
                finished=False,
            )
        )
    for idx in range(3):
        engine.output_q.put_nowait(
            make_output_msg(
                request_id,
                1,
                payload=f"{request_id}-stage1-{idx}",
                output_finished=(idx == 2),
                finished=False,
            )
        )
    for idx in range(3):
        engine.output_q.put_nowait(
            make_output_msg(
                request_id,
                2,
                payload=f"{request_id}-stage2-{idx}",
                output_finished=(idx == 2),
                finished=(idx == 2),
                images=[f"{request_id}-img-{idx}"],
            )
        )


def _enqueue_async_finish_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    request_id = msg["request_id"]
    engine.output_q.put_nowait(
        make_output_msg(
            request_id,
            0,
            payload=f"{request_id}-stage0",
            output_finished=True,
            finished=False,
        )
    )
    engine.output_q.put_nowait(
        make_output_msg(
            request_id,
            2,
            payload=f"{request_id}-stage2-final",
            output_finished=True,
            finished=True,
            images=[f"{request_id}-img-final"],
        )
    )


def _enqueue_async_diffusion_only_output(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    request_id = msg["request_id"]
    engine.output_q.put_nowait(
        make_output_msg(
            request_id,
            0,
            payload=f"{request_id}-diffusion-final",
            output_finished=True,
            finished=True,
            images=[f"{request_id}-image"],
        )
    )


def _enqueue_async_llm_diffusion_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    request_id = msg["request_id"]
    for idx in range(3):
        engine.output_q.put_nowait(
            make_output_msg(
                request_id,
                0,
                payload=f"{request_id}-text-{idx}",
                output_finished=(idx == 2),
                finished=False,
            )
        )

    engine.output_q.put_nowait(
        make_output_msg(
            request_id,
            1,
            payload=f"{request_id}-image-final",
            output_finished=True,
            finished=True,
            images=[f"{request_id}-image"],
        )
    )


def _enqueue_omni_final_only_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    request_id = msg["request_id"]
    sampling_params_list = msg["sampling_params_list"]
    llm_streaming = any(
        params.output_kind != RequestOutputKind.FINAL_ONLY for params in sampling_params_list[:2]
    )

    stage0_count = 3 if llm_streaming else 1
    stage1_count = 3 if llm_streaming else 1

    for idx in range(stage0_count):
        engine.output_q.put_nowait(
            make_output_msg(
                request_id,
                0,
                payload=f"{request_id}-stage0-{idx}",
                output_finished=(idx == stage0_count - 1),
                finished=False,
            )
        )

    for idx in range(stage1_count):
        engine.output_q.put_nowait(
            make_output_msg(
                request_id,
                1,
                payload=f"{request_id}-stage1-{idx}",
                output_finished=(idx == stage1_count - 1),
                finished=False,
            )
        )

    engine.output_q.put_nowait(
        make_output_msg(
            request_id,
            2,
            payload=f"{request_id}-stage2-final",
            output_finished=True,
            finished=True,
            images=[f"{request_id}-img-final"],
        )
    )


def _enqueue_omni_llm_diffusion_outputs(engine: FakeAsyncOmniEngine, msg: dict[str, Any]) -> None:
    request_id = msg["request_id"]
    sampling_params_list = msg["sampling_params_list"]
    llm_streaming = sampling_params_list[0].output_kind != RequestOutputKind.FINAL_ONLY
    stage0_count = 3 if llm_streaming else 1

    for idx in range(stage0_count):
        engine.output_q.put_nowait(
            make_output_msg(
                request_id,
                0,
                payload=f"{request_id}-text-{idx}",
                output_finished=(idx == stage0_count - 1),
                finished=False,
            )
        )

    engine.output_q.put_nowait(
        make_output_msg(
            request_id,
            1,
            payload=f"{request_id}-image-final",
            output_finished=True,
            finished=True,
            images=[f"{request_id}-image"],
        )
    )


@pytest.mark.asyncio
async def test_async_omni_yields_only_final_stage_outputs(monkeypatch: pytest.MonkeyPatch):
    engine = FakeAsyncOmniEngine(
        stage_metadata=THREE_STAGE_META,
        on_add_request=lambda eng, msg: eng.output_q.put_nowait(
            make_output_msg(msg["request_id"], 1, payload="non-final", output_finished=True, finished=False)
        )
        or eng.output_q.put_nowait(
            make_output_msg(
                msg["request_id"],
                2,
                payload="final",
                output_finished=True,
                finished=True,
                images=["final-img"],
            )
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
