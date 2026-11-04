from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat(monkeypatch):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    class FakeMediaConnector:
        def __init__(self, **_kwargs):
            pass

        async def fetch_audio_async(self, uri):
            fetched_audio_uris.append(uri)
            if uri == "file:///tmp/reference-b.wav":
                return [0.75, -0.25], 22050
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


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["extra_body", "model_extra", "attribute"])
@pytest.mark.parametrize("question_location", ["rendered", "deferred", "both"])
async def test_explicit_reference_wins_over_distinct_question_audio(serving_chat, source, question_location):
    from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
    from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import llm2tts

    request = SimpleNamespace(media_io_kwargs=None, extra_body={}, model_extra={})
    if source == "attribute":
        request.ref_audio = "file:///tmp/reference-b.wav"
    else:
        getattr(request, source)["ref_audio"] = "file:///tmp/reference-b.wav"
    question_audio = ([0.1, 0.2], 16000)
    prompt = {"multi_modal_data": {"image": object()}, "additional_information": {}}
    if question_location in {"rendered", "both"}:
        prompt["multi_modal_data"]["audio"] = [question_audio]
    if question_location in {"deferred", "both"}:
        prompt["additional_information"]["deferred_multi_modal_data"] = {"audio": [question_audio]}

    uri = serving_chat._minicpmo45_ref_audio_uri(request, [])
    await serving_chat._attach_minicpmo45_ref_audio(request, uri, prompt)

    info = prompt["additional_information"]
    assert torch.equal(info["minicpmo45_ref_audio"], torch.tensor([0.75, -0.25]))
    assert info["minicpmo45_ref_audio_sr"] == 22050
    roundtrip = deserialize_additional_information(serialize_additional_information(info))
    assert torch.equal(roundtrip["minicpmo45_ref_audio"], torch.tensor([0.75, -0.25]))
    assert roundtrip["minicpmo45_ref_audio_sr"] == 22050
    assert serving_chat.fetched_audio_uris == ["file:///tmp/reference-b.wav"]
    if question_location in {"rendered", "both"}:
        assert prompt["multi_modal_data"]["audio"] == [question_audio]
    if question_location in {"deferred", "both"}:
        assert info["deferred_multi_modal_data"]["audio"] == [question_audio]

    completion = SimpleNamespace(
        token_ids=[11, 12, 9002],
        text="hello",
        multimodal_output={
            "latent": torch.arange(20, dtype=torch.float32).reshape(5, 4),
            "meta": {"tts_bos_token_id": 9001, "tts_eos_token_id": 9002},
        },
    )
    source_output = SimpleNamespace(request_id="req-1", prompt_token_ids=[101, 9001], outputs=[completion])
    converted = llm2tts([source_output], prompt=[prompt])[0]["model_intermediate_buffer"]
    assert converted["codes"]["ref"] == [0.75, -0.25]
    assert converted["meta"]["ref_audio_sr"] == 22050
