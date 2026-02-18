"""End-to-end showcase test for streaming input support.

Demonstrates how the streaming input API works with AsyncOmniLLM:

  1. An async generator yields StreamingInput chunks over time
  2. Each chunk is processed by the engine (prompt is extended, KV cache preserved)
  3. Each sub-request produces intermediate outputs (finished=False)
  4. When the generator exhausts, a final output is emitted (finished=True)

This is the pattern used by multi-turn speech-to-speech pipelines, where
perception tokens, LLM tokens, and TTS tokens are streamed between stages.

Usage from the demo notebook:

    async def input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt="Once upon a time")
        yield StreamingInput(prompt=" in a magical forest")
        yield StreamingInput(prompt=" there lived a dragon who")

    async for output in engine.generate(
        input_generator(),
        sampling_params=sampling_params,
        request_id="my_session",
    ):
        print(output.outputs[0].text, output.finished)
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput
from vllm.v1.engine.output_processor import RequestOutputCollector


def _make_output(request_id: str, finished: bool) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=finished,
    )


@pytest.fixture
def mock_llm():
    """Create a mock AsyncLLM with the real generate() method bound."""
    llm = MagicMock(spec=AsyncLLM)
    llm.vllm_config = MagicMock()
    llm.vllm_config.cache_config.kv_sharing_fast_prefill = False
    llm.model_config = MagicMock()
    llm.model_config.max_model_len = 2048
    llm.log_requests = False
    llm.errored = False
    llm._pause_cond = asyncio.Condition()
    llm._paused = False
    llm._run_output_handler = MagicMock()
    llm.abort = AsyncMock()

    # Bind the real generate() method from AsyncLLM
    llm.generate = AsyncLLM.generate.__get__(llm, AsyncLLM)
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# Showcase 1: Basic streaming input with async generator
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_streaming_input_basic_flow(mock_llm):
    """Showcase: feed 2 prompt chunks via async generator, get 3 outputs.

    Flow:
        input_generator yields "Hello"   → engine processes → output (not finished)
        input_generator yields " world"  → engine processes → output (not finished)
        input_generator exhausts         → engine finalizes → output (finished=True)

    This is the fundamental streaming input pattern. Each yield extends
    the KV cache context without recomputing previous tokens.
    """
    request_id = "streaming_showcase"
    sampling_params = SamplingParams(
        max_tokens=5, output_kind=RequestOutputKind.DELTA
    )

    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)
    inputs_received: list[str] = []

    async def mock_add_request(req_id, prompt, params, *args, **kwargs):
        if isinstance(prompt, AsyncGenerator):
            async def handle_stream():
                async for chunk in prompt:
                    inputs_received.append(chunk.prompt)
                    queue.put(_make_output(req_id, finished=False))
                    await asyncio.sleep(0.01)
                # Final output when generator exhausts
                queue.put(_make_output(req_id, finished=True))

            asyncio.create_task(handle_stream())
            return queue
        return queue

    mock_llm.add_request = mock_add_request

    # ── The user-facing API ──
    async def input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt="Hello", sampling_params=sampling_params)
        yield StreamingInput(prompt=" world", sampling_params=sampling_params)

    outputs: list[RequestOutput] = []
    async for output in mock_llm.generate(
        input_generator(), sampling_params, request_id
    ):
        outputs.append(output)

    # 2 intermediate + 1 final
    assert len(outputs) == 3
    assert outputs[0].finished is False
    assert outputs[1].finished is False
    assert outputs[2].finished is True
    assert inputs_received == ["Hello", " world"]


# ─────────────────────────────────────────────────────────────────────────────
# Showcase 2: Synchronized injection (explicit autoregression)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_streaming_input_synchronized_injection(mock_llm):
    """Showcase: synchronized input injection driven by output reception.

    This pattern is used for explicit autoregression loops where the
    caller feeds one token at a time and waits for the model's output
    before injecting the next token:

        inject "A" → decode → get output → inject "B" → decode → get output → done

    The sync_queue bridges the output consumer and input producer,
    ensuring strict alternation between input injection and output
    reception.
    """
    request_id = "sync_showcase"
    sampling_params = SamplingParams(
        max_tokens=1, output_kind=RequestOutputKind.DELTA
    )

    queue = RequestOutputCollector(RequestOutputKind.DELTA, request_id)
    # Synchronization queue: output consumer signals input producer
    sync_queue: asyncio.Queue[int | None] = asyncio.Queue()
    inject_tokens = ["token_A", "token_B", "token_C"]
    inputs_received: list[str] = []

    async def mock_add_request(req_id, prompt, params, *args, **kwargs):
        if isinstance(prompt, AsyncGenerator):
            async def handle_stream():
                async for chunk in prompt:
                    inputs_received.append(chunk.prompt)
                    queue.put(_make_output(req_id, finished=False))
                    await asyncio.sleep(0.01)
                queue.put(_make_output(req_id, finished=True))

            asyncio.create_task(handle_stream())
            return queue
        return queue

    mock_llm.add_request = mock_add_request

    # ── Input producer: waits for signal before yielding next chunk ──
    async def synchronized_input_generator() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt="initial_prompt", sampling_params=sampling_params)
        for token in inject_tokens:
            signal = await sync_queue.get()
            if signal is None:
                break
            yield StreamingInput(prompt=token, sampling_params=sampling_params)

    # ── Output consumer: signals input producer after each output ──
    outputs: list[RequestOutput] = []
    step = 0
    async for output in mock_llm.generate(
        synchronized_input_generator(), sampling_params, request_id
    ):
        outputs.append(output)
        if not output.finished and step < len(inject_tokens):
            await sync_queue.put(step)
            step += 1

    # initial_prompt + 3 injected tokens = 4 intermediate + 1 final
    assert len(outputs) == 5
    assert all(not o.finished for o in outputs[:-1])
    assert outputs[-1].finished is True
    assert inputs_received == ["initial_prompt", "token_A", "token_B", "token_C"]


# ─────────────────────────────────────────────────────────────────────────────
# Showcase 3: Normal (non-streaming) generate still works
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_normal_generate_unaffected(mock_llm):
    """Non-streaming generate() continues to work as before.

    Passing a plain string prompt (not an AsyncGenerator) goes through
    the normal code path. This ensures the streaming input changes
    don't break existing behavior.
    """
    request_id = "normal_request"
    sampling_params = SamplingParams(max_tokens=10)

    queue = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, request_id)

    async def feed_outputs():
        queue.put(_make_output(request_id, finished=False))
        await asyncio.sleep(0.05)
        queue.put(_make_output(request_id, finished=True))

    asyncio.create_task(feed_outputs())

    async def mock_add_request(*args, **kwargs):
        return queue

    mock_llm.add_request = mock_add_request

    outputs: list[RequestOutput] = []
    async for output in mock_llm.generate(
        prompt="Tell me about Paris",
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        outputs.append(output)

    assert len(outputs) == 2
    assert outputs[-1].finished is True
