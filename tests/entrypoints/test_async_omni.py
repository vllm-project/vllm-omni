import asyncio
from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _noop(**kw):
    pass


def get_fake_add_request(submitted_request_ids):
    async def fake_add_request_async(*, request_id, prompt, sampling_params_list, final_stage_id, **kwargs):
        del prompt, sampling_params_list, final_stage_id, kwargs
        submitted_request_ids.append(request_id)

    return fake_add_request_async


def get_fake_abort(aborted_request_batches):
    async def fake_abort_async(request_ids):
        aborted_request_batches.append(list(request_ids))

    return fake_abort_async


async def fake_process_results(request_id, metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts):
    del metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts
    if request_id.startswith("cancel-"):
        await asyncio.Future()
        return
    yield SimpleNamespace(
        stage_id=0,
        request_output=SimpleNamespace(outputs=[]),
        finished=True,
    )


def get_async_omni_instance(fake_add_request=_noop, fake_abort_request=_noop) -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni.engine = SimpleNamespace(
        num_stages=1,
        add_request_async=fake_add_request,
        abort_async=fake_abort_request,
    )
    omni.log_stats = False
    omni.request_states = {}
    omni._final_output_handler = lambda: None
    omni.resolve_sampling_params_list = lambda params, allow_delta_coercion: params
    omni._compute_final_stage_id = lambda output_modalities: 0
    omni._process_orchestrator_results = fake_process_results
    omni._log_summary_and_cleanup = lambda request_id: omni.request_states.pop(request_id, None)
    return omni


def test_generate_accepts_request_after_repeated_cancellations():
    async def run_test():
        submitted_request_ids = []
        aborted_request_batches = []

        async def collect_outputs(request_id):
            outputs = []
            async for output in AsyncOmni.generate(
                omni,
                prompt={"prompt": "prompt"},
                request_id=request_id,
                sampling_params_list=[SimpleNamespace()],
                output_modalities=["image"],
            ):
                outputs.append(output)
            return outputs

        omni = get_async_omni_instance(
            fake_add_request=get_fake_add_request(submitted_request_ids),
            fake_abort_request=get_fake_abort(aborted_request_batches),
        )

        assert len(await collect_outputs("baseline")) == 1

        for idx in range(3):
            task = asyncio.create_task(collect_outputs(f"cancel-{idx}"))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert len(await collect_outputs("after-cancel")) == 1
        assert submitted_request_ids == [
            "baseline",
            "cancel-0",
            "cancel-1",
            "cancel-2",
            "after-cancel",
        ]
        assert aborted_request_batches == [
            ["cancel-0"],
            ["cancel-1"],
            ["cancel-2"],
        ]

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY, RequestOutputKind.CUMULATIVE]
)
def test_output_kind_is_preserved_with_explicit_sampling_params(output_kind):
    """Ensure we don't change the output kind in async generate if params are provided directly."""

    captured_params = []

    async def capturing_add_request(*, request_id, prompt, sampling_params_list, final_stage_id, **kwargs):
        del prompt, final_stage_id, kwargs
        captured_params.extend(sampling_params_list)

    async def run():
        omni = get_async_omni_instance(fake_add_request=capturing_add_request)
        sp = SamplingParams(output_kind=output_kind)
        async for _ in omni.generate(
            prompt={"prompt": "test"},
            request_id="test-req",
            sampling_params_list=[sp],
            output_modalities=["text"],
        ):
            pass

    asyncio.run(run())
    assert captured_params[0].output_kind == output_kind
