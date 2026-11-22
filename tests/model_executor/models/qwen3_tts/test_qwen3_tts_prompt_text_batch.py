# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    Qwen3TTSPromptEmbedsBuilder,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_batch_preprocess_projects_new_texts_once():
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
    ids_c = list(range(30, 40))
    ids_s = list(range(40, 52))
    # Serving stores the assistant-template ids wrapped in a one-element list.
    serving_ids = [ids_a]
    buf: dict[str, dict[str, Any]] = {
        "a": {"req_id": "a", "task_type": ["CustomVoice"], PRECOMPUTED_TEXT_IDS_KEY: [ids_a]},
        "b": {"req_id": "b", "task_type": ["VoiceDesign"], PRECOMPUTED_TEXT_IDS_KEY: [torch.tensor(ids_b)]},
        "c": {
            "req_id": "c",
            "task_type": ["VoiceDesign"],
            "non_streaming_mode": [True],
            PRECOMPUTED_TEXT_IDS_KEY: [ids_c],
        },
        "streaming": {
            "req_id": "streaming",
            "task_type": ["CustomVoice"],
            "non_streaming_mode": [False],
            PRECOMPUTED_TEXT_IDS_KEY: [ids_s],
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

    # One embedding + projection over all new CustomVoice / VoiceDesign texts (streaming + non-streaming).
    expected_tokens = sum(len(x) - 8 for x in (ids_a, ids_b, ids_c, ids_s))
    assert projected_shapes == [(1, expected_tokens, 4)]
    assert set(builder._batched_text_embeds) == {"a", "b", "c", "streaming"}
    with torch.no_grad():
        for req_id, ids in (("a", ids_a), ("b", ids_b), ("c", ids_c), ("streaming", ids_s)):
            expected = projection(text_embedding(torch.tensor([ids[3:-5]])))
            torch.testing.assert_close(builder._batched_text_embeds[req_id], expected)
            assert buf[req_id][PRECOMPUTED_TEXT_IDS_KEY].tolist() == [ids]
    for skipped in ("base", "built"):
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


@pytest.mark.cpu
@pytest.mark.parametrize("task_type", ["CustomVoice", "VoiceDesign"])
@pytest.mark.parametrize("non_streaming", [None, False, True])
@pytest.mark.parametrize("instruct", ["Speak calmly", ""])
def test_batched_prompt_consumption_matches_serial_and_clears_cache(
    task_type: str, non_streaming: bool | None, instruct: str
):
    batched, serial = _prompt_builder(), _prompt_builder()

    def request(req_id: str, ids: list[int], speaker: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "req_id": req_id,
            "text": ["hello"],
            "task_type": [task_type],
            "language": ["English"],
            "speaker": [speaker],
            "instruct": [instruct],
            PRECOMPUTED_TEXT_IDS_KEY: [ids],
        }
        if non_streaming is not None:
            info["non_streaming_mode"] = [non_streaming]
        return info

    with torch.inference_mode():
        # The same request id can be used again after its earlier prompt was consumed.
        for ids_a, ids_b in [
            (list(range(12)), list(range(20, 31))),
            ([0, 1, 2, *range(13, 24)], [20, 21, 22, *range(33, 46)]),
        ]:
            pending = [request("a", ids_a, "Vivian"), request("b", ids_b, "Serena")]
            expected = [request("a", ids_a, "Vivian"), request("b", ids_b, "Serena")]
            batched.preprocess_infos_batch(req_infos=pending, device=torch.device("cpu"))
            assert set(batched._batched_text_embeds) == {"a", "b"}
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


@pytest.mark.cpu
def test_streaming_custom_voice_prompt_cache_fast_path_and_cross_mode_parity():
    cached = _prompt_builder()

    def req(req_id: str, ids: list[int], non_streaming: bool | None = None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "req_id": req_id,
            "text": ["hello"],
            "task_type": ["CustomVoice"],
            "language": ["English"],
            "speaker": ["Vivian"],
            "instruct": [""],
            PRECOMPUTED_TEXT_IDS_KEY: [ids],
        }
        if non_streaming is not None:
            info["non_streaming_mode"] = [non_streaming]
        return info

    role = [0, 1, 2]
    ids_1 = [*role, 10, 11, 12, 40, 41, 42, 43, 44]
    ids_2 = [*role, 13, 14, 15, 16, 40, 41, 42, 43, 44]
    ids_3 = [*role, 17, 18, 40, 41, 42, 43, 44]

    with torch.inference_mode():
        # First streaming request (default omitted) populates the prompt cache.
        r1 = req("r1", ids_1)
        cached.preprocess_infos_batch(req_infos=[r1], device=torch.device("cpu"))
        cached.build_prompt_embeds(task_type="CustomVoice", info_dict=r1)
        assert ("Vivian", "English") in cached.__dict__.get("_cv_fast_keys", {})

        # Second streaming request and an explicit non-streaming request must both
        # hit the cached fast path without re-running text_embedding/projection.
        r2 = req("r2", ids_2, non_streaming=False)
        r3 = req("r3", ids_3, non_streaming=True)
        cached.preprocess_infos_batch(req_infos=[r2, r3], device=torch.device("cpu"))

        saved_emb, saved_proj = cached._text_embedding, cached._text_projection
        cached._text_embedding = lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("cached fast path should not call _text_embedding")
        )
        cached._text_projection = lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("cached fast path should not call _text_projection")
        )
        try:
            out2 = cached.build_prompt_embeds(task_type="CustomVoice", info_dict=r2)
            out3 = cached.build_prompt_embeds(task_type="CustomVoice", info_dict=r3)
        finally:
            cached._text_embedding, cached._text_projection = saved_emb, saved_proj

        ref2 = _prompt_builder().build_prompt_embeds(
            task_type="CustomVoice", info_dict=req("ref2", ids_2, non_streaming=False)
        )
        ref3 = _prompt_builder().build_prompt_embeds(
            task_type="CustomVoice", info_dict=req("ref3", ids_3, non_streaming=True)
        )
        torch.testing.assert_close(out2[0], ref2[0])
        torch.testing.assert_close(out2[1], ref2[1])
        assert out2[2:] == (None, None)
        assert out2[0].shape[0] == 10  # role(3) + codec_prefix(6) + first_text(1)
        assert out2[1].shape[0] == 4  # text[1:](3) + tts_eos(1)

        torch.testing.assert_close(out3[0], ref3[0])
        torch.testing.assert_close(out3[1], ref3[1])
        assert out3[2:] == (None, None)


@hardware_test(res={"cuda": ["L4", "H100", "H200"], "rocm": "MI325"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("allow_table", [False, True])
@torch.inference_mode()
def test_projected_text_table_matches_the_projection_per_token(monkeypatch, allow_table):
    use_table = allow_table and current_omni_platform.is_cuda()
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: use_table)
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
    if not use_table:
        assert builder._projected_text_table is None
        assert builder._projected_token_cache, "unsupported platforms retain the existing cache"
        torch.testing.assert_close(builder._project_text_ids(ids), direct)
        return
    assert builder._projected_text_table is not None
    assert builder._projected_text_table.shape == (1000, 16)
    assert not builder._projected_token_cache, "constants are re-projected from the table"
    torch.testing.assert_close(builder._project_text_ids(ids), direct, rtol=1e-5, atol=1e-5)
