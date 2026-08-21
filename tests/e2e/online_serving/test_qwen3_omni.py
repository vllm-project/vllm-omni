"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# Set VLLM_TEST_PD_MODE=1 to test PD disaggregation (follow-up — deploy overlay not yet migrated).
_USE_PD = os.environ.get("VLLM_TEST_PD_MODE", "0") == "1"

_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


# For prefix caching checks against we enable it on the thinker and talker via CLI override
# and enable prompt token details so that we can determine if any tokens were cached.
# We also explicitly set block size so that we can make sure the cached token counts are a
# multiple of the block size.
BLOCK_SIZE = 16
test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_CI_DEPLOY,
            use_stage_cli=True,
            server_args=[
                "--no-async-chunk",
                "--block-size",
                str(BLOCK_SIZE),
                "--stage-overrides",
                '{"0": {"enable_prefix_caching": true}, "1": {"enable_prefix_caching": true}}',
                "--enable-prompt-tokens-details",
            ],
        ),
        id="default",
    )
]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "text_image": "What color are the squares in this image?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.skipif(_USE_PD, reason="Temporarily skip PD mode in this test module.")
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=3 if _USE_PD else 2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test multi-modal input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    Datasets: single request
    """

    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {
            "audio": ["test"],
        },
    }

    # Test single completion
    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.skipif(_USE_PD, reason="Temporarily skip PD mode in this test module.")
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=3 if _USE_PD else 2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: text
    Datasets: few requests
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


def _run_prefix_cache_check(openai_client, request_config: dict):
    """Make two requests given a request config, and validate that:
    1. The second request actually had cached tokens
    2. The number of cached tokens is divisible by the block size used in
    test_params, because currently upstream vLLM does not cache partial
    blocks.
    3. The first request also reports prompt token details; details must be
    present whenever --enable-prompt-tokens-details is on, even when zero
    tokens are cached. (An exact cached_tokens == 0 check would be
    order-dependent here: the module-scoped server may have already cached
    the shared system-prompt blocks from earlier tests.)

    Returns the (first, second) responses for additional per-test assertions.
    """
    first_response = openai_client.send_omni_request(request_config, request_num=1)[0]
    cached_response = openai_client.send_omni_request(request_config, request_num=1)[0]

    assert first_response.cached_tokens is not None

    # Ensure that we have a prefix cache hit on the second request and that only the last
    # partial block is uncached (since currently we don't cache partial blocks).
    num_cached_tokens = cached_response.cached_tokens
    num_prompt_tokens = cached_response.prompt_tokens
    assert num_cached_tokens is not None and num_prompt_tokens is not None
    num_uncached_tokens = num_prompt_tokens % BLOCK_SIZE
    assert num_cached_tokens > 0
    assert num_cached_tokens % BLOCK_SIZE == 0
    assert (num_cached_tokens + num_uncached_tokens) == num_prompt_tokens

    return first_response, cached_response


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_thinker_prefix_caching_text_output(omni_server, openai_client) -> None:
    """
    Test thinker prefix caching by sending identical requests with an image (i.e.,
    a large shared prefix) and verifying that the second request uses cached tokens
    & produces the same output with greedy decoding.

    NOTE: Checking the output of prefix caching directly can be a bit unstable
    due to slight numerical differences as a result of running different scheduled
    sequence lengths. As such, for E2E tests on prefix cache, we only check the cached
    token count and not the output, since the omni tensor cache has solid unit tests,
    and the core prefix cache algorithm is largely tested by upstream vLLM.
    """
    img_res = generate_synthetic_image(224, 224)
    image_data_url = f"data:image/jpeg;base64,{img_res['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text=get_prompt("text_image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
    }
    first_response, _ = _run_prefix_cache_check(openai_client, request_config)

    assert first_response.multimodal_tokens is not None
    assert first_response.multimodal_tokens.get("image", 0) > 0


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_thinker_prefix_caching_audio_output(omni_server, openai_client) -> None:
    """
    Verify that thinker prefix caching does not hang when the request
    produces audio output (text + audio modalities).  Sends two identical
    requests so the second exercises the prefix-cached path through the
    full thinker -> talker -> code2wav pipeline.

    Regression test for https://github.com/vllm-project/vllm-omni/issues/3510
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt(),
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
    }

    _run_prefix_cache_check(openai_client, request_config)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_completions_rejected_for_thinker_talker(omni_server, openai_client) -> None:
    """Ensure Thinker-talker models reject /v1/completions; we do this because the
    thinker-talker handoff implementations currently use ChatML <|im_start|> and
    <|im_end|> markers to segment the input sequence; when we don't have them,
    the talker does not get any embeddings, which breaks the server.
    """
    responses = openai_client.send_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "Hello, how are you?",
                "max_tokens": 10,
            },
        },
        err_code=400,
    )
    assert not responses[0].success


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_batched_completions_text(omni_server, openai_client) -> None:
    """Ensure that we can make a batch chat completions request (text only)."""
    responses = openai_client.send_batched_chat_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "messages": [
                    [{"role": "user", "content": "What color is the sky? Answer in one word."}],
                    [{"role": "user", "content": "What is 2+2? Answer in one word."}],
                ],
                "modalities": ["text"],
                "max_tokens": 20,
            },
        },
    )
    assert responses
    resp = responses[0]
    assert resp.success
    body = resp.json_body
    choices = body["choices"]
    assert len(choices) == 2
    for choice in choices:
        assert choice["message"]["content"]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_batched_completions_audio_out(omni_server, openai_client) -> None:
    """Ensure that we can make a batch chat completions request (audio + text)."""
    messages = [
        [{"role": "user", "content": "Say hello."}],
        [{"role": "user", "content": "Say goodbye."}],
    ]
    num_messages = len(messages)
    responses = openai_client.send_batched_chat_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "messages": messages,
                "modalities": ["text", "audio"],
                "max_tokens": 200,
            },
        },
    )
    assert responses and len(responses) == 1
    resp = responses[0]
    assert resp.success
    choices = resp.json_body["choices"]

    assert len(choices) == num_messages

    # Every message in the batch should contain both the audio and text content
    for choice in choices:
        assert choice["message"]["audio"]["data"]
        assert choice["message"]["content"]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@pytest.mark.parametrize(
    "sad_opts",
    [
        {"stream": "not-a-bool"},  # Fails validation
        {"n": 2},  # vLLM Batched endpoints doesn't support n > 1
    ],
)
def test_batched_completions_with_bad_values(omni_server, openai_client, sad_opts) -> None:
    """Ensure that a bad values are correctly handled as 400s."""
    messages = [
        [{"role": "user", "content": "Say hello."}],
        [{"role": "user", "content": "Say goodbye."}],
    ]
    responses = openai_client.send_batched_chat_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "messages": messages,
                **sad_opts,
            },
        },
    )
    assert responses and len(responses) == 1
    resp = responses[0]
    assert not responses[0].success
    assert resp.status_code == 400
