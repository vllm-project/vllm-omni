import asyncio
import re
from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model]

DIFFUSION_MODEL = "riverclouds/qwen_image_random"
OMNI_MODEL = "Qwen/Qwen2.5-Omni-7B"
OMNI_STAGE_CONFIG = get_deploy_config_path("ci/qwen2_5_omni_thinker_only.yaml")


async def _noop(*args, **kw):
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
    omni._compute_final_output_stage_ids = lambda output_modalities: [0]
    omni._process_orchestrator_results = fake_process_results
    omni._log_summary_and_cleanup = lambda request_id: omni.request_states.pop(request_id, None)
    return omni


@pytest.mark.cpu
def test_generate_submits_randomized_id_to_engine():
    """Ensure the engine receives a UUID-suffixed ID, not the raw request ID"""

    async def run():
        submitted_ids = []
        omni = get_async_omni_instance(fake_add_request=get_fake_add_request(submitted_ids))

        req_id = "my-req-1"
        async for _ in omni.generate(
            prompt={"prompt": "test"},
            request_id=req_id,
            sampling_params_list=[SimpleNamespace()],
            output_modalities=["text"],
        ):
            pass

        assert len(submitted_ids) == 1
        assert submitted_ids[0] != req_id
        assert submitted_ids[0].startswith(f"{req_id}-")

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "req_ids,cancel_prefix,expected_cancel_count",
    [
        (["cancel-me"], "cancel-me", 1),
        (["cancel-me", "cancel-me"], "cancel-me", 2),
        (["cancel-hello", "cancel-hello-world"], "cancel-hello", 1),
    ],
)
def test_abort_handles_internal_request_mapping(req_ids: list[str], cancel_prefix: str, expected_cancel_count: int):
    """Ensure that abort() with the user-visible ID resolves correctly.

    NOTE: In the case of concurrent / colliding request(s), all requests matching the
    user provided request ID will be aborted."""

    async def run():
        aborted_batches = []
        omni = get_async_omni_instance(
            fake_abort_request=get_fake_abort(aborted_batches),
        )

        async def exhaust(agen):
            async for _ in agen:
                pass

        tasks = []
        for user_request_id in req_ids:
            t = asyncio.create_task(
                exhaust(
                    omni.generate(
                        prompt={"prompt": "test"},
                        request_id=user_request_id,
                        sampling_params_list=[SimpleNamespace()],
                    )
                )
            )
            await asyncio.sleep(0)
            tasks.append(t)

        assert len(omni.request_states) == len(req_ids)
        await omni.abort(cancel_prefix)

        assert len(aborted_batches) == 1
        aborted_ids = aborted_batches[0]
        # Aborted requests will have fmt {ext_id}-{UUID} to avoid collisions
        for rid in aborted_ids:
            assert re.fullmatch(rf"{re.escape(cancel_prefix)}-[0-9a-f]+", rid)
        assert len(aborted_ids) == expected_cancel_count
        assert len(set(aborted_ids)) == expected_cancel_count
        assert len(omni.request_states) == len(req_ids) - expected_cancel_count

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.cpu
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

        # Check prefixes instead of equality since generate will add
        # a random numeric suffix to ensure the request ID is unique
        expected_prefixes = ["baseline-", "cancel-0-", "cancel-1-", "cancel-2-", "after-cancel-"]
        assert len(submitted_request_ids) == len(expected_prefixes)
        for submitted, prefix in zip(submitted_request_ids, expected_prefixes):
            assert submitted.startswith(prefix)

        assert len(aborted_request_batches) == 3
        for batch, prefix in zip(aborted_request_batches, ["cancel-0-", "cancel-1-", "cancel-2-"]):
            assert len(batch) == 1
            assert batch[0].startswith(prefix)

    asyncio.run(run_test())


@pytest.mark.cpu
def test_generate_yields_streaming_diffusion_chunks_before_final():
    """AsyncOmni.generate yields every intermediate diffusion chunk before the final one."""

    async def streaming_process_results(request_id, metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts):
        del metrics, final_stage_id_for_e2e, req_start_ts, wall_start_ts
        yield OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=[],
            final_output_type="image",
            custom_output={"chunk": 0},
            finished=False,
        )
        yield OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=[],
            final_output_type="image",
            custom_output={"chunk": 1},
            finished=True,
        )

    async def run_test():
        omni = get_async_omni_instance()
        omni._process_orchestrator_results = streaming_process_results

        outputs = []
        async for output in AsyncOmni.generate(
            omni,
            prompt={"prompt": "a cat"},
            request_id="req-stream",
            sampling_params_list=[SimpleNamespace()],
            output_modalities=["image"],
        ):
            outputs.append(output)

        assert len(outputs) == 2
        assert [output.finished for output in outputs] == [False, True]
        assert [output.custom_output["chunk"] for output in outputs] == [0, 1]

    asyncio.run(run_test())


@pytest.mark.cpu
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


# End to end tests for ensuring internal manipulation of request ID
# in diffusion / Omni models don't leak back to the user.
#
# One AsyncOmni per test function (all cases in a single asyncio loop) to avoid
# repeated cold starts. Do not use class/module-scoped engine fixtures here:
# pytest-asyncio uses a function-scoped event loop by default, so reusing an
# engine across tests can hang on the second generate() call.


# Covers:
#   * plain client ids (``my-req-1``)
#   * OpenAI-style prefixed ids (``img_gen-*``, ``chatcmpl-*``) that AsyncOmni
#     suffixes internally for engine routing — streamed outputs must still echo
#     the caller-visible id, not the internal UUID-suffixed id
#   * empty ``request_id`` — server assigns a non-empty id for the caller
_DIFFUSION_REQ_IDS = ["my-req-1", "img_gen-abc123", "chatcmpl-xyz"]
_OMNI_REQ_IDS = ["my-req-1", "img_gen-abc123", "chatcmpl-xyz"]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.omni
@pytest.mark.asyncio
async def test_diffusion_generate_request_id():
    """Diffusion E2E request-id contract (``riverclouds/qwen_image_random``).

    Scenarios (one engine, sequential ``generate`` calls):
    - plain id ``my-req-1``
    - image-style prefix ``img_gen-abc123``
    - chat-style prefix ``chatcmpl-xyz``
    - empty ``request_id`` → output id is non-empty (auto-assigned)

    Each streaming output must expose the user-supplied id unchanged; internal
    UUID suffixing must not leak into ``output.request_id``.
    """
    engine = AsyncOmni(model=DIFFUSION_MODEL)
    try:
        for req_id in _DIFFUSION_REQ_IDS:
            async for output in engine.generate("a white cat", request_id=req_id):
                assert output.request_id == req_id
        async for output in engine.generate("a white cat", request_id=""):
            assert output.request_id != ""
    finally:
        engine.shutdown()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.omni
@pytest.mark.asyncio
async def test_omni_generate_request_id():
    """Omni E2E request-id contract (``Qwen/Qwen2.5-Omni-7B``, thinker-only stage).

    Same scenarios as ``test_diffusion_generate_request_id``:
    - plain id ``my-req-1``
    - image-style prefix ``img_gen-abc123``
    - chat-style prefix ``chatcmpl-xyz``
    - empty ``request_id`` → output id is non-empty (auto-assigned)

    Text modality only; asserts caller-visible ids are preserved across the
    multi-stage orchestrator path on H100.
    """
    engine = AsyncOmni(model=OMNI_MODEL, stage_configs_path=OMNI_STAGE_CONFIG)
    try:
        for req_id in _OMNI_REQ_IDS:
            async for output in engine.generate(
                "Say hello in one word.",
                request_id=req_id,
                output_modalities=["text"],
            ):
                assert output.request_id == req_id
        async for output in engine.generate(
            "Say hello in one word.",
            request_id="",
            output_modalities=["text"],
        ):
            assert output.request_id != ""
    finally:
        engine.shutdown()
