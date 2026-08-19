from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat(monkeypatch):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    class FakeMediaConnector:
        def __init__(self, **_kwargs):
            pass

        async def fetch_audio_async(self, _uri):
            return "wave", 24000

    instance = object.__new__(OmniOpenAIServingChat)
    instance.engine_client = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(
                engine_args=SimpleNamespace(model_arch="MiniCPMO45OmniForConditionalGeneration")
            )
        ]
    )
    instance.model_config = SimpleNamespace(
        allowed_local_media_path="/tmp",
        allowed_media_domains=None,
    )
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


@pytest.mark.asyncio
async def test_materializes_ref_audio_into_deferred_contract(serving_chat):
    request = SimpleNamespace(media_io_kwargs=None)
    prompt = {"additional_information": {}}

    await serving_chat._attach_minicpmo45_ref_audio(request, "file:///tmp/ref.wav", prompt)

    assert prompt["additional_information"]["deferred_multi_modal_data"]["audio"] == [("wave", 24000)]


@pytest.mark.asyncio
async def test_does_not_replace_existing_deferred_audio(serving_chat):
    request = SimpleNamespace(media_io_kwargs=None)
    prompt = {"additional_information": {"deferred_multi_modal_data": {"audio": [("old", 16000)]}}}

    await serving_chat._attach_minicpmo45_ref_audio(request, "file:///tmp/ref.wav", prompt)

    assert prompt["additional_information"]["deferred_multi_modal_data"]["audio"] == [("old", 16000)]
