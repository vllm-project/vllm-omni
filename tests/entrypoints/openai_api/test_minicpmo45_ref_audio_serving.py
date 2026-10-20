from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat(monkeypatch):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    class FakeMediaConnector:
        def __init__(self, **_kwargs):
            pass

        async def fetch_audio_async(self, uri):
            fetched_audio_uris.append(uri)
            return "wave", 24000

    fetched_audio_uris = []
    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(engine_args=SimpleNamespace(model_arch="MiniCPMO45OmniForConditionalGeneration"))
        ]
    )
    instance.model_config = SimpleNamespace(
        allowed_local_media_path="/tmp",
        allowed_media_domains=None,
    )
    instance.fetched_audio_uris = fetched_audio_uris
    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.serving_chat.MediaConnector",
        FakeMediaConnector,
    )
    return instance


def test_extracts_benchmark_ref_audio_from_extra_body():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    request = SimpleNamespace(extra_body={"ref_audio": "file:///tmp/ref.wav"}, model_extra={})
    assert OmniOpenAIServingChat._minicpmo45_ref_audio_uri(request, []) == "file:///tmp/ref.wav"


def test_extracts_demo_ref_audio_from_message():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    request = SimpleNamespace(extra_body={}, model_extra={})
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
            ],
        }
    ]
    assert OmniOpenAIServingChat._minicpmo45_ref_audio_uri(request, messages).startswith("data:audio/wav")


def test_wraps_input_audio_base64_as_data_uri_using_format():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    request = SimpleNamespace(extra_body={}, model_extra={})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            ],
        }
    ]
    assert OmniOpenAIServingChat._minicpmo45_ref_audio_uri(request, messages) == "data:audio/wav;base64,AAAA"


def test_deferred_input_audio_base64_is_materialized_from_data_uri():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    modality, payload = OmniOpenAIServingChat._deferred_multimodal_part(
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        {"audio"},
    )

    assert modality == "audio"
    assert payload == "data:audio/wav;base64,AAAA"


@pytest.mark.asyncio
async def test_materializes_ref_audio_into_deferred_contract(serving_chat):
    request = SimpleNamespace(media_io_kwargs=None)
    prompt = {"additional_information": {}}

    await serving_chat._attach_minicpmo45_ref_audio(request, "file:///tmp/ref.wav", prompt)

    assert prompt["additional_information"]["deferred_multi_modal_data"]["audio"] == [("wave", 24000)]
    assert serving_chat.fetched_audio_uris == ["file:///tmp/ref.wav"]


@pytest.mark.asyncio
async def test_does_not_replace_existing_deferred_audio(serving_chat):
    request = SimpleNamespace(media_io_kwargs=None)
    prompt = {"additional_information": {"deferred_multi_modal_data": {"audio": [("old", 16000)]}}}

    await serving_chat._attach_minicpmo45_ref_audio(request, "file:///tmp/ref.wav", prompt)

    assert prompt["additional_information"]["deferred_multi_modal_data"]["audio"] == [("old", 16000)]
    assert serving_chat.fetched_audio_uris == []


@pytest.mark.asyncio
async def test_reuses_renderer_materialized_audio_without_fetching_again(serving_chat):
    request = SimpleNamespace(media_io_kwargs=None)
    rendered_audio = ("decoded wave", 24000)
    prompt = {"multi_modal_data": {"audio": [rendered_audio]}, "additional_information": {}}

    await serving_chat._attach_minicpmo45_ref_audio(request, "data:audio/wav;base64,AAAA", prompt)

    assert prompt["additional_information"]["deferred_multi_modal_data"]["audio"] == [rendered_audio]
    assert serving_chat.fetched_audio_uris == []


@pytest.mark.asyncio
async def test_fetches_ref_audio_when_renderer_audio_is_empty(serving_chat):
    request = SimpleNamespace(media_io_kwargs=None)
    prompt = {"multi_modal_data": {"audio": []}, "additional_information": {}}

    await serving_chat._attach_minicpmo45_ref_audio(request, "file:///tmp/ref.wav", prompt)

    assert prompt["additional_information"]["deferred_multi_modal_data"]["audio"] == [("wave", 24000)]
    assert serving_chat.fetched_audio_uris == ["file:///tmp/ref.wav"]


@pytest.mark.asyncio
async def test_fetches_input_audio_data_uri_when_renderer_did_not_materialize(serving_chat):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    request = SimpleNamespace(extra_body={}, model_extra={}, media_io_kwargs=None)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            ],
        }
    ]
    uri = OmniOpenAIServingChat._minicpmo45_ref_audio_uri(request, messages)
    prompt = {"additional_information": {}}

    await serving_chat._attach_minicpmo45_ref_audio(request, uri, prompt)

    assert serving_chat.fetched_audio_uris == ["data:audio/wav;base64,AAAA"]
    assert prompt["additional_information"]["deferred_multi_modal_data"]["audio"] == [("wave", 24000)]
