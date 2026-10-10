# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    Qwen3TTSPromptEmbedsBuilder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_batch_preprocess_projects_new_non_streaming_texts_once():
    torch.manual_seed(0)
    text_embedding = torch.nn.Embedding(64, 4)
    projection = torch.nn.Linear(4, 4)
    projected_shapes: list[tuple[int, ...]] = []

    def text_projection(x: torch.Tensor) -> torch.Tensor:
        projected_shapes.append(tuple(x.shape))
        return projection(x)

    builder = Qwen3TTSPromptEmbedsBuilder.__new__(Qwen3TTSPromptEmbedsBuilder)
    builder._text_embedding = text_embedding
    builder._text_projection = text_projection
    builder._batched_text_embeds = {}

    ids_a = list(range(1, 13))
    ids_b = list(range(20, 31))
    # Serving stores the assistant-template ids wrapped in a one-element list.
    serving_ids = [ids_a]
    buf: dict[str, dict[str, Any]] = {
        "a": {"req_id": "a", "task_type": ["CustomVoice"], PRECOMPUTED_TEXT_IDS_KEY: [ids_a]},
        "b": {"req_id": "b", "task_type": ["VoiceDesign"], PRECOMPUTED_TEXT_IDS_KEY: [torch.tensor(ids_b)]},
        "streaming": {
            "req_id": "streaming",
            "task_type": ["CustomVoice"],
            "non_streaming_mode": [False],
            PRECOMPUTED_TEXT_IDS_KEY: serving_ids,
        },
        "base": {"req_id": "base", "task_type": ["Base"], PRECOMPUTED_TEXT_IDS_KEY: serving_ids},
        "built": {
            "req_id": "built",
            "task_type": ["CustomVoice"],
            "embed": {"prefill": torch.zeros(1)},
            PRECOMPUTED_TEXT_IDS_KEY: serving_ids,
        },
    }

    builder.preprocess_infos_batch(req_infos=list(buf.values()), device=torch.device("cpu"))

    # One embedding + projection over both requests' text tokens (template stripped).
    assert projected_shapes == [(1, (len(ids_a) - 8) + (len(ids_b) - 8), 4)]
    assert set(builder._batched_text_embeds) == {"a", "b"}
    with torch.no_grad():
        for req_id, ids in (("a", ids_a), ("b", ids_b)):
            expected = projection(text_embedding(torch.tensor([ids[3:-5]])))
            torch.testing.assert_close(builder._batched_text_embeds[req_id], expected)
            assert buf[req_id][PRECOMPUTED_TEXT_IDS_KEY].tolist() == [ids]
    for skipped in ("streaming", "base", "built"):
        assert buf[skipped][PRECOMPUTED_TEXT_IDS_KEY] is serving_ids


class _Tokenizer:
    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[2, 3, 4, 5]])}


def _prompt_builder() -> Qwen3TTSPromptEmbedsBuilder:
    """Use the real constructor and prompt assembly with small embedding tables."""
    torch.manual_seed(42)
    config = Qwen3TTSConfig(tts_bos_token_id=50, tts_eos_token_id=51, tts_pad_token_id=52)
    talker_config = Qwen3TTSTalkerConfig(
        codec_nothink_id=10,
        codec_think_id=11,
        codec_think_bos_id=12,
        codec_think_eos_id=13,
        codec_pad_id=14,
        codec_bos_id=15,
        codec_language_id={"english": 16},
        spk_id={"vivian": 17, "serena": 18},
    )
    builder = Qwen3TTSPromptEmbedsBuilder(
        config=config,
        talker_config=talker_config,
        model_path="",
        text_embedding=torch.nn.Embedding(64, 4),
        text_projection=torch.nn.Linear(4, 4),
        codec_embed=torch.nn.Embedding(64, 4),
        residual_code_embeddings=lambda: [],
        speaker_encoder=torch.nn.Identity(),
        tts_pad_embed=torch.zeros(1, 4),
        encode_ref_audio_batch=lambda *args, **kwargs: [],
    )
    builder._text_tokenizer = _Tokenizer()
    return builder


@pytest.mark.parametrize("task_type", ["CustomVoice", "VoiceDesign"])
@pytest.mark.parametrize("non_streaming", [False, True])
@pytest.mark.parametrize("instruct", ["Speak calmly", ""])
def test_batched_prompt_consumption_matches_serial_and_clears_cache(task_type: str, non_streaming: bool, instruct: str):
    batched, serial = _prompt_builder(), _prompt_builder()

    def request(req_id: str, ids: list[int], speaker: str) -> dict[str, Any]:
        return {
            "req_id": req_id,
            "text": ["hello"],
            "task_type": [task_type],
            "language": ["English"],
            "speaker": [speaker],
            "instruct": [instruct],
            "non_streaming_mode": [non_streaming],
            PRECOMPUTED_TEXT_IDS_KEY: [ids],
        }

    with torch.inference_mode():
        # The same request id can be used again after its earlier prompt was consumed.
        for ids_a, ids_b in [(list(range(12)), list(range(20, 31))), (list(range(10, 24)), list(range(30, 46)))]:
            pending = [request("a", ids_a, "Vivian"), request("b", ids_b, "Serena")]
            expected = [request("a", ids_a, "Vivian"), request("b", ids_b, "Serena")]
            batched.preprocess_infos_batch(req_infos=pending, device=torch.device("cpu"))
            assert set(batched._batched_text_embeds) == ({"a", "b"} if non_streaming else set())
            for actual_info, expected_info in zip(pending, expected, strict=True):
                actual = batched.build_prompt_embeds(task_type=task_type, info_dict=actual_info)
                reference = serial.build_prompt_embeds(task_type=task_type, info_dict=expected_info)
                torch.testing.assert_close(actual[0], reference[0])
                torch.testing.assert_close(actual[1], reference[1])
                assert actual[2:] == reference[2:]
                assert PRECOMPUTED_TEXT_IDS_KEY not in actual_info
            assert not batched._batched_text_embeds
        batched.preprocess_infos_batch(req_infos=[], device=torch.device("cpu"))
        assert not batched._batched_text_embeds


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_projected_text_table_matches_the_projection_per_token():
    builder = Qwen3TTSPromptEmbedsBuilder.__new__(Qwen3TTSPromptEmbedsBuilder)
    torch.manual_seed(0)
    builder._text_embedding = torch.nn.Embedding(1000, 32).cuda()
    mlp = torch.nn.Sequential(torch.nn.Linear(32, 48), torch.nn.SiLU(), torch.nn.Linear(48, 16)).cuda()
    builder._text_projection = mlp
    builder._projected_text_table = None
    builder._projected_token_cache = {("cuda:0", (1,)): torch.zeros(1)}
    ids = torch.tensor([[5, 999, 0, 5, 321]], device="cuda")
    direct = mlp(builder._text_embedding(ids))
    torch.testing.assert_close(builder._project_text_ids(ids), direct)

    builder.build_projected_text_table(chunk_rows=300)  # several chunks, a partial last one
    assert builder._projected_text_table is not None
    assert builder._projected_text_table.shape == (1000, 16)
    assert not builder._projected_token_cache, "constants are re-projected from the table"
    torch.testing.assert_close(builder._project_text_ids(ids), direct, rtol=1e-5, atol=1e-5)
