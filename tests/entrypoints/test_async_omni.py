from types import SimpleNamespace

import pytest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
