from types import SimpleNamespace

import torch

from vllm_omni.config.stage_config import DeployConfig, merge_pipeline_deploy
from vllm_omni.engine.mm_outputs import MultimodalPayload
from vllm_omni.model_executor.models.minimind_o.minimind_omni_code2wav import (
    _codec_ids_from_payload_or_input,
)
from vllm_omni.model_executor.models.minimind_o.pipeline import MINIMIND_OMNI_PIPELINE
from vllm_omni.model_executor.stage_input_processors.minimind_omni import (
    talker2code2wav,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
    thinker2talker,
    thinker2talker_full_payload,
    thinker2talker_token_only,
)


def test_thinker2talker_accepts_multimodal_payload() -> None:
    bridge = torch.randn(3, 8)
    output = SimpleNamespace(
        cumulative_token_ids=[3],
        multimodal_output=MultimodalPayload(
            metadata={"hidden_states": {"bridge": bridge}},
        ),
    )
    thinker_output = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[output])

    [talker_input] = thinker2talker([thinker_output])

    assert talker_input["prompt_token_ids"] == [2049, 2049]
    torch.testing.assert_close(talker_input["additional_information"]["hidden_states"]["bridge"], bridge)


def test_talker2code2wav_accepts_multimodal_payload() -> None:
    audio_codes = torch.tensor([[1, 2], [3, 4]])
    output = SimpleNamespace(
        multimodal_output=MultimodalPayload(
            metadata={"codes": {"audio": audio_codes}},
        ),
    )
    talker_output = SimpleNamespace(request_id="request-0", outputs=[output])

    [code2wav_input] = talker2code2wav([talker_output])

    assert code2wav_input["prompt_token_ids"] == [1, 3, 2, 4]


def test_thinker_connector_full_payload_and_token_only() -> None:
    bridge = torch.randn(3, 8)
    output = SimpleNamespace(
        cumulative_token_ids=[3],
        multimodal_output=MultimodalPayload(
            metadata={"hidden_states": {"bridge": bridge}},
        ),
    )
    thinker_output = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[output])
    request = SimpleNamespace(prompt_token_ids=[1, 2], output_token_ids=[3])

    payload = thinker2talker_full_payload(
        transfer_manager=None,
        pooling_output={"hidden_states.bridge": bridge},
        request=request,
    )
    [talker_input] = thinker2talker_token_only([thinker_output])

    assert payload is not None
    torch.testing.assert_close(payload["hidden_states"]["bridge"], bridge)
    assert payload["ids"] == {"prompt": [1, 2], "output": [3], "all": [1, 2, 3]}
    assert talker_input["prompt_token_ids"] == [2049, 2049]
    assert talker_input["additional_information"] is None


def test_talker_connector_full_payload_and_token_only() -> None:
    audio_codes = torch.tensor([[1, 2], [3, 4]])
    output = SimpleNamespace(
        multimodal_output=MultimodalPayload(
            metadata={"codes": {"audio": audio_codes}},
        ),
    )
    talker_output = SimpleNamespace(request_id="request-0", outputs=[output])

    payload = talker2code2wav_full_payload(
        transfer_manager=None,
        pooling_output={"codes.audio": audio_codes},
        request=SimpleNamespace(request_id="request-0"),
    )
    [code2wav_input] = talker2code2wav_token_only([talker_output])

    assert payload is not None
    assert payload["codes"]["audio"].tolist() == [1, 3, 2, 4]
    assert code2wav_input["prompt_token_ids"] == [0, 0, 0, 0]


def test_code2wav_prefers_connector_codec_ids() -> None:
    placeholder_ids = torch.zeros(4, dtype=torch.long)
    codec_ids = torch.tensor([1, 3, 2, 4])

    actual = _codec_ids_from_payload_or_input(
        placeholder_ids,
        {"codes": {"audio": codec_ids}},
    )

    torch.testing.assert_close(actual, codec_ids)


def test_pipeline_registers_sync_connector_processors() -> None:
    thinker, talker, code2wav = MINIMIND_OMNI_PIPELINE.stages

    assert thinker.custom_process_next_stage_input_func.endswith(".thinker2talker_full_payload")
    assert talker.sync_process_input_func.endswith(".thinker2talker_token_only")
    assert talker.custom_process_next_stage_input_func.endswith(".talker2code2wav_full_payload")
    assert code2wav.sync_process_input_func.endswith(".talker2code2wav_token_only")

    merged = merge_pipeline_deploy(MINIMIND_OMNI_PIPELINE, DeployConfig(async_chunk=False))
    assert merged[0].yaml_engine_args["custom_process_next_stage_input_func"].endswith(".thinker2talker_full_payload")
    assert merged[1].custom_process_input_func.endswith(".thinker2talker_token_only")
    assert merged[1].yaml_engine_args["custom_process_next_stage_input_func"].endswith(".talker2code2wav_full_payload")
    assert merged[2].custom_process_input_func.endswith(".talker2code2wav_token_only")
