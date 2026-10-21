# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent numeric oracle for the Talker prefill row transform.

Synthetic projections exercise the real assembly methods without checkpoints.
The oracle states retained spans and the nine bootstrap rows explicitly; it
must not use the production row planner to calculate expected embeddings.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.core.prefix_cache.adapter import PrefixCacheRequestOwner
from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from vllm_omni.model_executor.models.qwen3_omni.talker_input_plan import plan_talker_prefill
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _compute_talker_prompt_ids_length

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IM_START, SYSTEM, USER, ASSISTANT, NEWLINE, IM_END = 151644, 8948, 872, 77091, 198, 151645
IMAGE, AUDIO, VIDEO = 151655, 151646, 151656
WIDTH = 4


class _HiddenProjection(nn.Module):
    def forward(self, values):
        return values + 20


def _model():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        im_start_token_id=IM_START,
        system_token_id=SYSTEM,
        user_token_id=USER,
        assistant_token_id=ASSISTANT,
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        talker_config=SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=WIDTH),
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_pad_id=5,
            codec_bos_id=6,
        ),
    )
    model.thinker_config = SimpleNamespace(image_token_id=IMAGE, audio_token_id=AUDIO, video_token_id=VIDEO)
    model.talker_config = model.config.talker_config
    model.default_tts_text_spk_type = "test"
    model.tts_text_spk_token_ids = {"test": 4}
    model.model_stage = "talker"
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=False, dtype=torch.bfloat16))
    talker = nn.Module()
    talker.num_code_groups = 3
    talker.text_projection = nn.Identity()
    talker.hidden_projection = _HiddenProjection()
    talker.embed_input_ids = nn.Embedding(16, WIDTH, dtype=torch.bfloat16)
    with torch.no_grad():
        talker.embed_input_ids.weight.copy_(torch.arange(16).unsqueeze(1).expand(-1, WIDTH))
    model.talker = talker
    model.model = talker
    return model


def _replay_model_and_payload(text_count):
    model = _model()
    model.talker_config.num_code_groups = 3
    model.talker_config.text_config.vocab_size = 16
    model.talker_config.code_predictor_config = SimpleNamespace(vocab_size=16, hidden_size=WIDTH)
    talker = model.talker
    talker.config = model.talker_config
    talker.language_model = nn.Module()
    talker.language_model.model = nn.Module()
    talker.language_model.model.codec_embedding = talker.embed_input_ids
    talker.code_predictor = nn.Module()
    talker.code_predictor.model = nn.Module()
    talker.code_predictor.model.codec_embedding = nn.ModuleList(
        [nn.Embedding(16, WIDTH, dtype=torch.bfloat16) for _ in range(2)]
    )
    with torch.no_grad():
        for group, embed in enumerate(talker.code_predictor.model.codec_embedding, start=2):
            embed.weight.copy_(torch.arange(16).unsqueeze(1).expand(-1, WIDTH) * group)
    talker.code_predictor.small_to_mtp_projection = nn.Identity()
    talker.code_predictor.get_input_embeddings = lambda: talker.code_predictor.model.codec_embedding
    talker.replay_codec_embeddings = Qwen3OmniMoeTalkerForConditionalGeneration.replay_codec_embeddings.__get__(talker)
    prompt, sequence, embeddings, hidden, *_ = _case(text_count)
    # Fixed source/admission descriptors stated explicitly, not calculated by
    # the production planner. Twelve user rows followed by nine bootstrap rows.
    prompt_ids = sequence[5:12] + sequence[17:22] + [IM_START, ASSISTANT, NEWLINE] + [151671] * 4
    prompt_ids += [151672, sequence[25] if text_count else 151671]
    payload = {
        "ids": {"prompt": prompt, "all": sequence},
        "embed": {
            "prefill": embeddings,
            "tts_pad": torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16),
            "tts_bos": torch.full((1, 1, WIDTH), 60, dtype=torch.bfloat16),
            "tts_eos": torch.full((1, 1, WIDTH), 70, dtype=torch.bfloat16),
        },
        "hidden_states": {
            "output": hidden,
            "last": torch.full((1, WIDTH), -999, dtype=torch.bfloat16),
            "trailing_text": torch.full((2, WIDTH), -888, dtype=torch.bfloat16),
        },
        "meta": {
            "talker_prefill_plan": [("user", 5, 12), ("user", 17, 22), ("assistant", 22, len(sequence))],
            "next_stage_prompt_ids": prompt_ids,
            "num_processed_tokens": 1000,
            "decode_flag": True,
        },
    }
    return model, payload


@pytest.mark.parametrize("text_count", [0, 1, 3, 8])
@pytest.mark.parametrize("start,end", [(0, 1), (5, 13), (20, 21), (0, 25), (20, 24), (21, 22), (22, 25), (25, 28)])
@torch.inference_mode()
def test_history_reconstruction_uses_absolute_cursor_and_complete_codes(text_count, start, end, monkeypatch, mocker):
    model, payload = _replay_model_and_payload(text_count)
    owner = PrefixCacheRequestOwner(9, 2)
    records = tuple(torch.tensor((index + 1, 2, 3), dtype=torch.long) for index in range(7))
    forbidden = mocker.Mock(side_effect=AssertionError("history must not sample/replan"))
    monkeypatch.setattr(model, "talker_mtp", forbidden)
    monkeypatch.setattr(model.talker.code_predictor, "forward", forbidden)
    monkeypatch.setattr("vllm_omni.model_executor.models.qwen3_omni.qwen3_omni.plan_talker_prefill", forbidden)
    scratch_ids = torch.full((end - start,), -99, dtype=torch.long)
    scratch = torch.full((end - start, WIDTH), -99, dtype=torch.bfloat16)
    actual_ids, actual, update = model.talker_replay_inputs(
        scratch_ids, scratch, row_start=start, prompt_length=21, owner=owner, codec_inputs=records, payload=payload
    )
    embeddings = payload["embed"]["prefill"]
    # Independent full-sequence numeric oracle, including the mixed-block case.
    users = torch.cat((embeddings[5:12], embeddings[17:22])).clone()
    users[3:6] += 60  # image/audio/video: hidden = embedding + 40, projection + 20
    first_text = embeddings[25:26] if text_count else torch.zeros((1, WIDTH), dtype=torch.bfloat16)
    bootstrap_text = torch.cat(
        (embeddings[22:25], torch.full((4, WIDTH), 50), torch.full((1, WIDTH), 60), first_text)
    ).to(torch.bfloat16)
    bootstrap = bootstrap_text + torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6]).unsqueeze(1)
    tail = torch.cat((embeddings[26:], torch.full((1, WIDTH), 70))).to(torch.bfloat16)
    text_rows = torch.cat((tail, torch.full((7, WIDTH), 50)))[:7].to(torch.bfloat16)
    codec_rows = torch.stack(records)
    # Three embeddings have weights id, 2*id and 3*id, respectively.
    codec_sums = (codec_rows * torch.tensor([1, 2, 3])).sum(dim=1).unsqueeze(1)
    full_expected = torch.cat((users, bootstrap, codec_sums + text_rows)).to(torch.bfloat16)
    torch.testing.assert_close(actual, full_expected[start:end], rtol=0, atol=0)
    assert (
        actual_ids.tolist()
        == (payload["meta"]["next_stage_prompt_ids"] + [int(record[0]) for record in records])[start:end]
    )
    expected_codes = torch.cat((torch.zeros(21, 3, dtype=torch.long), codec_rows))
    torch.testing.assert_close(update["codes"]["audio"], expected_codes[start:end], rtol=0, atol=0)
    assert "mtp_inputs" not in update
    assert update["meta"]["num_processed_tokens"] == (1 + end - 21 if end > 21 else end)
    if end > 21:
        expected_tail = tail[end - 21 :] if end - 21 < tail.shape[0] else torch.full((1, WIDTH), 50)
        torch.testing.assert_close(update["hidden_states"]["trailing_text"], expected_tail.to(torch.bfloat16))
        assert update["meta"]["decode_flag"]
    assert payload["meta"]["num_processed_tokens"] == 1000
    assert torch.all(payload["hidden_states"]["trailing_text"] == -888)
    assert torch.all(scratch == -99) and torch.all(scratch_ids == -99)
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    ("case", "start", "count"),
    [
        ("prompt-only-fresh", 0, 5),
        ("prompt-only-resume", 5, 3),
        ("generated-only", 21, 1),
        ("mixed", 20, 2),
    ],
)
@torch.inference_mode()
def test_history_replay_single_part_return_skips_final_singleton_cats(case, start, count, monkeypatch):
    owner = PrefixCacheRequestOwner(9, 2)
    records = tuple(torch.tensor((index + 1, 2, 3), dtype=torch.long) for index in range(7))
    model, payload = _replay_model_and_payload(3)
    captured_prefill = {}
    prefill_args = []
    original_prefill = model.talker_preprocess_prefill

    def capture_prefill(prefill_start, prefill_span):
        fixed_payload = dict(payload)
        fixed_payload["meta"] = {**payload["meta"], "num_processed_tokens": prefill_start}
        captured_prefill[(prefill_start, prefill_span)] = original_prefill(
            torch.full((prefill_span,), -99, dtype=torch.long),
            torch.full((prefill_span, WIDTH), -99, dtype=torch.bfloat16),
            fixed_payload,
        )

    prefill_rows = max(0, min(start + count, 21) - start)
    if prefill_rows:
        capture_prefill(start, prefill_rows)
    if start + count > 21 and not prefill_rows:
        capture_prefill(20, 1)

    def stub_prefill(input_ids, input_embeds, replay_payload):
        key = (replay_payload["meta"]["num_processed_tokens"], input_ids.shape[0])
        prefill_args.append(key)
        return captured_prefill[key]

    monkeypatch.setattr(model, "talker_preprocess_prefill", stub_prefill)
    original_cat = torch.cat
    replay_cat_lengths = []

    def spy_cat(tensors, *args, **kwargs):
        tensor_parts = tuple(tensors)
        replay_cat_lengths.append(len(tensor_parts))
        if len(tensor_parts) == 1:
            raise AssertionError("single-part replay return must not call torch.cat")
        return original_cat(tensor_parts, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", spy_cat)
    actual_ids, actual_embeds, actual_update = model.talker_replay_inputs(
        torch.full((count,), -99, dtype=torch.long),
        torch.full((count, WIDTH), -99, dtype=torch.bfloat16),
        row_start=start,
        prompt_length=21,
        owner=owner,
        codec_inputs=records,
        payload=payload,
    )

    assert prefill_args == list(captured_prefill)
    if case == "mixed":
        assert replay_cat_lengths == [2, 2, 2]
    else:
        assert replay_cat_lengths == []
    if prefill_rows and start + count <= 21:
        prefill_ids, prefill_embeds, _ = captured_prefill[(start, prefill_rows)]
        assert actual_ids.data_ptr() == prefill_ids.data_ptr()
        assert actual_embeds.data_ptr() == prefill_embeds.data_ptr()
        torch.testing.assert_close(
            actual_update["codes"]["audio"],
            torch.zeros((count, 3), dtype=torch.long),
            rtol=0,
            atol=0,
        )
    if case == "generated-only":
        assert actual_ids.tolist() == [1]
        torch.testing.assert_close(actual_update["codes"]["audio"], torch.tensor([[1, 2, 3]]), rtol=0, atol=0)
        actual_update["codes"]["audio"][0, 0] = 99
        assert records[0][0].item() == 1


@pytest.mark.parametrize("bad_kind", ["missing", "width", "dtype", "untyped"])
def test_history_reconstruction_rejects_bad_interval_before_any_embedding(bad_kind, monkeypatch, mocker):
    model, payload = _replay_model_and_payload(3)
    owner = PrefixCacheRequestOwner(9, 2)
    valid = torch.tensor((1, 2, 3), dtype=torch.long)
    bad = {
        "missing": None,
        "width": torch.tensor((2, 2), dtype=torch.long),
        "dtype": torch.tensor((2, 2, 3), dtype=torch.int32),
        "untyped": {"owner": owner, "position": 22, "codes": (2, 2, 3)},
    }[bad_kind]
    forbidden = mocker.Mock(side_effect=AssertionError("bad interval must fail before any projection or gathering"))
    monkeypatch.setattr(model, "talker_preprocess_prefill", forbidden)
    monkeypatch.setattr(model.talker, "replay_codec_embeddings", forbidden)
    with pytest.raises(ValueError, match="Talker"):
        model.talker_replay_inputs(
            torch.zeros(3, dtype=torch.long),
            torch.zeros(3, WIDTH, dtype=torch.bfloat16),
            row_start=20,
            prompt_length=21,
            owner=owner,
            codec_inputs=(valid,) if bad is None else (valid, bad),
            payload=payload,
        )
    forbidden.assert_not_called()


@torch.inference_mode()
def test_first_decode_handoff_reuses_retained_prefill_tail(monkeypatch, mocker):
    model, payload = _replay_model_and_payload(5)
    owner = PrefixCacheRequestOwner(40)
    payload["meta"] = {
        **payload["meta"],
        "decode_flag": False,
        "prefill_consumed_text_tokens": 1,
        "num_processed_tokens": 21,
    }
    payload["hidden_states"]["trailing_text"] = torch.full((2, WIDTH), 80, dtype=torch.bfloat16)
    payload["embed"]["tts_pad_projected"] = torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16)
    monkeypatch.setattr(
        model,
        "talker_preprocess_prefill",
        mocker.Mock(side_effect=AssertionError("first decode handoff must not reconstruct prefill")),
    )
    codes = (torch.tensor((7, 2, 3), dtype=torch.long),)
    codec_embedding = torch.full((1, WIDTH), 11, dtype=torch.bfloat16)

    actual_ids, actual, update = model.talker_replay_inputs(
        torch.tensor([7]),
        torch.zeros(1, WIDTH, dtype=torch.bfloat16),
        row_start=21,
        prompt_length=21,
        owner=owner,
        codec_inputs=codes,
        payload=payload,
        restore_text=False,
        codec_embeddings=codec_embedding,
    )

    assert actual_ids.tolist() == [7]
    torch.testing.assert_close(actual, torch.full((1, WIDTH), 91, dtype=torch.bfloat16), rtol=0, atol=0)
    assert update["meta"] == {"decode_flag": True, "num_processed_tokens": 2}
    torch.testing.assert_close(
        update["hidden_states"]["trailing_text"], torch.full((1, WIDTH), 80, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize(
    "meta",
    [
        {"decode_flag": True, "prefill_consumed_text_tokens": 1, "num_processed_tokens": 21},
        {"decode_flag": False, "prefill_consumed_text_tokens": 1, "num_processed_tokens": 20},
        {"decode_flag": False, "num_processed_tokens": 21},
    ],
)
def test_first_decode_handoff_rejects_bad_prefill_cursor_before_replay(meta):
    model, payload = _replay_model_and_payload(5)
    payload["meta"] = {**payload["meta"], **meta}
    payload["hidden_states"]["trailing_text"] = torch.full((2, WIDTH), 80, dtype=torch.bfloat16)
    payload["embed"]["tts_pad_projected"] = torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="steady replay|first decode"):
        model.talker_replay_inputs(
            torch.tensor([7]),
            torch.zeros(1, WIDTH, dtype=torch.bfloat16),
            row_start=21,
            prompt_length=21,
            owner=PrefixCacheRequestOwner(41),
            codec_inputs=(torch.tensor((7, 2, 3), dtype=torch.long),),
            payload=payload,
            restore_text=False,
            codec_embeddings=torch.zeros(1, WIDTH, dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("consumed", [0, 1, 3, 6])
@torch.inference_mode()
def test_history_restores_queue_for_following_decode_without_user_projection(consumed, monkeypatch, mocker):
    model, payload = _replay_model_and_payload(5)
    owner = PrefixCacheRequestOwner(31)
    records = tuple(torch.tensor((1, 2, 3), dtype=torch.long) for _ in range(7))
    forbidden = mocker.Mock(side_effect=AssertionError("history hit must not project user rows"))
    monkeypatch.setattr(model, "_get_talker_user_parts", forbidden)
    _, _, update = model.talker_replay_inputs(
        torch.tensor([1]),
        torch.zeros(1, WIDTH, dtype=torch.bfloat16),
        row_start=21 + consumed,
        prompt_length=21,
        owner=owner,
        codec_inputs=records,
        payload=payload,
    )
    # Existing steady-state decode must continue at the next row, not start
    # again from the bootstrap's first text. Use an independent text oracle.
    resumed_payload = {**payload, **update}
    _, text_step, _ = model.talker_preprocess_decode(
        torch.tensor([1]), torch.zeros(1, WIDTH, dtype=torch.bfloat16), {}, resumed_payload
    )
    expected_text = [26, 27, 28, 29, 70, 50, 50, 50][consumed + 1]
    assert torch.all(text_step == expected_text)
    forbidden.assert_not_called()


@torch.inference_mode()
def test_history_restoration_is_reentrant_and_uses_current_payload_pad():
    model, payload = _replay_model_and_payload(1)
    owner = PrefixCacheRequestOwner(31)
    records = tuple(torch.tensor((1, 2, 3), dtype=torch.long) for _ in range(5))

    def replay(start, count, current_payload):
        return model.talker_replay_inputs(
            torch.ones(count, dtype=torch.long),
            torch.zeros(count, WIDTH, dtype=torch.bfloat16),
            row_start=start,
            prompt_length=21,
            owner=owner,
            codec_inputs=records,
            payload=current_payload,
        )

    _, first, captured = replay(21, 3, payload)
    retained = first.clone()
    retained_tail = captured["hidden_states"]["trailing_text"].clone()
    # Another request overwrites the model's legacy global special embeds.
    changed = {**payload, "embed": {**payload["embed"], "tts_pad": torch.full((1, 1, WIDTH), 90)}}
    _, second, _ = replay(22, 2, changed)
    assert torch.all(second == 104)  # fixed codes sum to 14; this request's PAD is 90
    # Recompute a previous partial block after eviction; stale global state and
    # the old request's mutable decode queue must have no influence.
    _, recomputed, _ = replay(21, 3, payload)
    torch.testing.assert_close(recomputed, retained, rtol=0, atol=0)
    torch.testing.assert_close(first, retained, rtol=0, atol=0)
    torch.testing.assert_close(captured["hidden_states"]["trailing_text"], retained_tail, rtol=0, atol=0)


def test_fixed_wire_plan_keeps_source_ids_on_cpu_during_payload_upload(monkeypatch):
    import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni as module

    model, payload = _replay_model_and_payload(3)
    payload["meta"]["num_processed_tokens"] = 0
    original_as_tensor = torch.as_tensor
    upload = module.to_device_nonblocking
    payload_tensor_ids = {
        id(payload["embed"]["prefill"]),
        id(payload["hidden_states"]["output"]),
        id(payload["embed"]["tts_pad"]),
        id(payload["embed"]["tts_bos"]),
        id(payload["embed"]["tts_eos"]),
    }
    uploaded_tensors = set()
    source_id_devices = []

    def spy_as_tensor(data, *args, **kwargs):
        if data is payload["ids"]["all"]:
            source_id_devices.append(kwargs.get("device"))
        if data is payload["ids"]["all"] and kwargs.get("device") is not None:
            raise AssertionError("fixed payload source IDs must not be uploaded before CPU classification")
        return original_as_tensor(data, *args, **kwargs)

    def spy_upload(tensor, device):
        if id(tensor) in payload_tensor_ids:
            uploaded_tensors.add(id(tensor))
        return upload(tensor, device)

    monkeypatch.setattr(torch, "as_tensor", spy_as_tensor)
    monkeypatch.setattr(module, "to_device_nonblocking", spy_upload)
    actual_ids, actual, update = model.talker_preprocess_prefill(
        torch.tensor(payload["meta"]["next_stage_prompt_ids"][:1]),
        torch.zeros(1, WIDTH, dtype=torch.bfloat16),
        payload,
    )

    assert actual_ids.tolist() == payload["meta"]["next_stage_prompt_ids"][:1]
    assert actual.shape == (1, WIDTH)
    assert update["meta"]["prefill_consumed_text_tokens"] == 1
    assert source_id_devices == [None]
    assert uploaded_tensors == payload_tensor_ids


@pytest.mark.parametrize(
    ("pad_shape", "selected_row"),
    [
        ("singleton", 0),
        ("leading_aggregate", 0),
        ("sequence_aggregate", 16),
        ("flat_aggregate", 16),
    ],
)
@torch.inference_mode()
def test_prefill_persists_selected_single_pad_for_long_generated_replay(pad_shape, selected_row):
    model, payload = _replay_model_and_payload(3)
    pad_rows = (torch.arange(17, dtype=torch.bfloat16).reshape(17, 1).expand(17, WIDTH) + 80).clone()
    payload["embed"]["tts_pad"] = {
        "singleton": pad_rows[selected_row : selected_row + 1].reshape(1, 1, WIDTH),
        "leading_aggregate": pad_rows.reshape(17, 1, WIDTH),
        "sequence_aggregate": pad_rows.reshape(1, 17, WIDTH),
        "flat_aggregate": pad_rows,
    }[pad_shape]
    payload["meta"]["num_processed_tokens"] = 0

    _, prompt_embeds, update = model.talker_preprocess_prefill(
        torch.tensor(payload["meta"]["next_stage_prompt_ids"][:21]),
        torch.zeros(21, WIDTH, dtype=torch.bfloat16),
        payload,
    )

    selected = pad_rows[selected_row].reshape(1, WIDTH)
    persisted_pad = update["embed"]["tts_pad_projected"]
    assert persisted_pad.shape == (1, 1, WIDTH)
    torch.testing.assert_close(persisted_pad.reshape(1, WIDTH), selected, rtol=0, atol=0)
    bootstrap_pad = prompt_embeds[15:19] - torch.arange(1, 5, dtype=torch.bfloat16).reshape(4, 1)
    torch.testing.assert_close(bootstrap_pad, selected.expand(4, -1), rtol=0, atol=0)

    records = tuple(torch.tensor((index % 15 + 1, 2, 3), dtype=torch.long) for index in range(63))
    _, _, replay_update = model.talker_replay_inputs(
        torch.full((63,), -99, dtype=torch.long),
        torch.full((63, WIDTH), -99, dtype=torch.bfloat16),
        row_start=21,
        prompt_length=21,
        owner=PrefixCacheRequestOwner(88),
        codec_inputs=records,
        payload={
            **payload,
            "embed": {**payload["embed"], **update["embed"]},
            "hidden_states": {**payload["hidden_states"], **update["hidden_states"]},
            "meta": {**payload["meta"], **update["meta"]},
        },
    )

    torch.testing.assert_close(replay_update["codes"]["audio"], torch.stack(records), rtol=0, atol=0)
    replay_update["codes"]["audio"][0, 0] = 999
    assert records[0][0].item() == 1
    trailing = replay_update["hidden_states"]["trailing_text"]
    assert trailing.shape == (1, WIDTH)
    torch.testing.assert_close(trailing, selected.to(trailing.dtype), rtol=0, atol=0)
    _, text_step, _ = model.talker_preprocess_decode(
        torch.tensor([1]),
        torch.zeros(1, WIDTH, dtype=torch.bfloat16),
        {},
        {
            "embed": replay_update["embed"],
            "hidden_states": replay_update["hidden_states"],
            "meta": replay_update["meta"],
        },
    )
    assert text_step.shape == (1, WIDTH)
    torch.testing.assert_close(text_step, selected.to(text_step.dtype), rtol=0, atol=0)


def test_text_only_cpu_mask_user_span_projects_directly_without_bool_gather(monkeypatch):
    model = _model()
    mask = torch.zeros(4, dtype=torch.bool)
    hidden = torch.full((4, WIDTH), 40, dtype=torch.bfloat16)
    embed = torch.arange(4, dtype=torch.bfloat16).unsqueeze(1).expand(-1, WIDTH).clone()
    original_getitem = torch.Tensor.__getitem__
    projected = []

    def spy_getitem(self, key):
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            raise AssertionError("text-only CPU mask path must not use boolean tensor gather")
        return original_getitem(self, key)

    def project(value):
        projected.append(value)
        return value

    monkeypatch.setattr(torch.Tensor, "__getitem__", spy_getitem)
    monkeypatch.setattr(model.talker.text_projection, "forward", project)
    actual = model._get_talker_user_parts(0, 4, mask, hidden, embed)

    assert projected and projected[0].data_ptr() == embed[0:4].data_ptr()
    torch.testing.assert_close(actual, embed, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    assert actual.device == hidden.device


@pytest.mark.parametrize("bad_kind", ["async_chunk", "no_plan", "length", "owner", "negative", "empty", "shape"])
def test_history_rejects_invalid_boundary_contract(bad_kind, monkeypatch, mocker):
    model, payload = _replay_model_and_payload(1)
    args = dict(row_start=0, prompt_length=21, owner=PrefixCacheRequestOwner(3), codec_inputs=(), payload=payload)
    input_ids = torch.ones(1, dtype=torch.long)
    input_embeds = torch.zeros(1, WIDTH, dtype=torch.bfloat16)
    if bad_kind == "async_chunk":
        model.vllm_config.model_config.async_chunk = True
    elif bad_kind == "no_plan":
        payload["meta"].pop("talker_prefill_plan")
    elif bad_kind == "length":
        args["prompt_length"] = 22
    elif bad_kind == "owner":
        args["owner"] = {"admission_id": 3, "generation": 0}
    elif bad_kind == "negative":
        args["row_start"] = -1
    elif bad_kind == "empty":
        input_ids = torch.empty(0, dtype=torch.long)
    else:
        input_embeds = torch.zeros(1, WIDTH + 1, dtype=torch.bfloat16)
    forbidden = mocker.Mock(side_effect=AssertionError("invalid boundary must fail before projection"))
    monkeypatch.setattr(model, "talker_preprocess_prefill", forbidden)
    with pytest.raises(ValueError, match="Talker"):
        model.talker_replay_inputs(input_ids, input_embeds, **args)
    forbidden.assert_not_called()


def _case(text_count, assistant_only=False):
    parts = [
        [IM_START, SYSTEM, NEWLINE, 11, IM_END],
        [IM_START, USER, NEWLINE, IMAGE, AUDIO, VIDEO, IM_END],
        [IM_START, ASSISTANT, NEWLINE, 12, IM_END],
        [IM_START, USER, NEWLINE, 13, IM_END],
        [IM_START, ASSISTANT, NEWLINE],
    ]
    if assistant_only:
        parts = parts[-1:]
    prompt = sum(parts, [])
    sequence = prompt + [40 + index for index in range(text_count)]
    embeddings = torch.arange(len(sequence), dtype=torch.bfloat16).unsqueeze(1).expand(-1, WIDTH).clone()
    hidden = embeddings + 40
    # These source spans are fixed independently of any production planner.
    users = [] if assistant_only else [(5, 12), (17, 22)]
    assistant_start = 0 if assistant_only else 22
    return prompt, sequence, embeddings, hidden, users, assistant_start


@pytest.mark.parametrize("text_count", [0, 1, 3])
@pytest.mark.parametrize("assistant_only", [False, True])
def test_prefill_embedding_math_preserves_user_spans_and_nine_bootstrap_rows(text_count, assistant_only):
    model = _model()
    prompt, sequence, embeddings, hidden, users, assistant_start = _case(text_count, assistant_only)
    pad, bos, eos = (torch.full((1, 1, WIDTH), v, dtype=torch.bfloat16) for v in (50, 60, 70))
    original_embeddings, original_hidden = embeddings.clone(), hidden.clone()
    actual_ids, actual, trailing = model._thinker_to_talker_prefill(
        embeddings,
        hidden,
        None,
        torch.tensor([prompt]),
        torch.tensor(sequence),
        speaker_id=4,
        tts_pad_thinker=pad,
        tts_bos_thinker=bos,
        tts_eos_thinker=eos,
    )
    expected_users = []
    for start, end in users:
        rows = embeddings[start:end].clone()
        for offset, token in enumerate(sequence[start:end]):
            if token in (IMAGE, AUDIO, VIDEO):
                rows[offset] = hidden[start + offset] + 20
        expected_users.append(rows)
    first_text = embeddings[assistant_start + 3 : assistant_start + 4] if text_count else torch.zeros((1, WIDTH))
    bootstrap_text = torch.cat(
        (embeddings[assistant_start : assistant_start + 3], pad[0].expand(4, -1), bos[0], first_text)
    )
    bootstrap_codec = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6]).unsqueeze(1).expand(-1, WIDTH)
    expected = torch.cat((*expected_users, bootstrap_text + bootstrap_codec)).to(torch.bfloat16)
    expected_trailing = torch.cat((embeddings[assistant_start + 4 :], eos[0])) if text_count > 1 else eos[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(trailing, expected_trailing, rtol=0, atol=0)
    torch.testing.assert_close(embeddings, original_embeddings, rtol=0, atol=0)
    torch.testing.assert_close(hidden, original_hidden, rtol=0, atol=0)
    assert actual.shape == (sum(end - start for start, end in users) + 9, WIDTH)
    expected_ids = sum((sequence[start:end] for start, end in users), []) + [
        IM_START,
        ASSISTANT,
        NEWLINE,
        *([model.config.tts_pad_token_id] * 4),
        model.config.tts_bos_token_id,
        sequence[assistant_start + 3] if text_count else model.config.tts_pad_token_id,
    ]
    assert actual_ids.tolist() == expected_ids


@pytest.mark.parametrize("cached_rows", [0, 20])
@pytest.mark.parametrize("supplied_embeds", [False, True])
def test_real_identity_ids_skip_codec_gather_and_initialize_current_prefill_state(cached_rows, supplied_embeds):
    model = _model()
    prompt, sequence, embeddings, hidden, *_ = _case(3)
    span = 21 - cached_rows
    # Every identity ID is outside this test's codec vocabulary of 16 entries.
    input_ids = torch.full((span,), IM_START, dtype=torch.long)
    payload = {
        "ids": {"prompt": prompt, "all": sequence},
        "embed": {
            "prefill": embeddings,
            "tts_pad": torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16),
            "tts_bos": torch.full((1, 1, WIDTH), 60, dtype=torch.bfloat16),
            "tts_eos": torch.full((1, 1, WIDTH), 70, dtype=torch.bfloat16),
        },
        "hidden_states": {"output": hidden},
        "speaker": [" TEST "],
        "meta": {},
        "_omni_is_prefill": True,
        "_omni_num_computed_tokens": cached_rows,
    }
    supplied = torch.full((span, WIDTH), -99, dtype=torch.bfloat16) if supplied_embeds else None
    actual_ids, actual, update = model.talker_preprocess(input_ids, supplied, **payload)
    assert actual_ids.shape == (span,)
    torch.testing.assert_close(actual[-1], embeddings[25] + 6, rtol=0, atol=0)
    assert update["meta"] == {"prefill_consumed_text_tokens": 1, "num_processed_tokens": 21}
    torch.testing.assert_close(
        update["hidden_states"]["trailing_text"], torch.cat((embeddings[26:], payload["embed"]["tts_eos"][0]))
    )
    assert torch.all(update["embed"]["tts_pad_projected"] == 50)
    assert update["codes"]["audio"].shape == (span, 3)


@pytest.mark.parametrize("supplied_embeds", [False, True])
def test_decode_still_embeds_codec_ids_and_hands_off_after_first_text(supplied_embeds):
    model = _model()
    model.tts_pad_embed = torch.full((1, WIDTH), 50, dtype=torch.bfloat16)
    supplied = torch.full((1, WIDTH), 9, dtype=torch.bfloat16) if supplied_embeds else None
    input_ids = torch.tensor([7])
    actual_ids, actual, update = model.talker_preprocess(
        input_ids,
        supplied,
        _omni_is_prefill=False,
        meta={"prefill_consumed_text_tokens": 1, "num_processed_tokens": 21},
        hidden_states={"trailing_text": torch.full((2, WIDTH), 80, dtype=torch.bfloat16)},
    )
    assert actual_ids is input_ids
    # The runner can pass reusable scratch storage, not codec embeddings.
    assert torch.all(actual == 7)
    assert update["meta"] == {"decode_flag": True, "num_processed_tokens": 2}
    assert torch.all(update["mtp_inputs"][1] == 80)


def test_trimmed_first_text_row_is_missing_even_when_sampled_id_is_present():
    model = _model()
    prompt, sequence, embeddings, hidden, *_ = _case(1, assistant_only=True)
    actual_ids, actual, _ = model._thinker_to_talker_prefill(
        embeddings[:-1],
        hidden[:-1],
        None,
        torch.tensor([prompt]),
        torch.tensor(sequence),
        speaker_id=4,
    )
    assert actual_ids.tolist() == [IM_START, ASSISTANT, NEWLINE] + [model.config.tts_pad_token_id] * 4 + [
        model.config.tts_bos_token_id,
        model.config.tts_pad_token_id,
    ]
    # The final row is zero text plus codec BOS, never the unavailable text.
    assert torch.all(actual[-1] == 6)


def test_generic_embedding_and_warmup_never_gather_identity_ids(monkeypatch):
    model = _model()

    def forbidden_gather(_ids):
        raise AssertionError("generic embedding/warmup must not gather identity IDs")

    monkeypatch.setattr(model.talker.embed_input_ids, "forward", forbidden_gather)
    monkeypatch.setattr(model.talker, "forward", lambda **kwargs: kwargs["inputs_embeds"])
    # Include a Thinker token that overlaps the codec vocabulary as well.
    identity_ids = torch.tensor([IM_START, 7, 151671])
    scratch = model.embed_input_ids(identity_ids)
    assert scratch.shape == (3, WIDTH)
    assert scratch.dtype == torch.bfloat16
    assert torch.count_nonzero(scratch) == 0
    warmup = model.forward(input_ids=identity_ids, positions=torch.arange(3))
    torch.testing.assert_close(warmup, scratch)
    prepared = torch.full_like(scratch, 80)
    assert model.forward(input_ids=identity_ids, positions=torch.arange(3), inputs_embeds=prepared) is prepared


def test_trimmed_user_rows_keep_identity_and_embeddings_aligned():
    model = _model()
    sequence = torch.tensor([IM_START, USER, NEWLINE, IMAGE])
    embeddings = torch.arange(3 * WIDTH, dtype=torch.bfloat16).reshape(3, WIDTH)
    actual_ids, actual, _ = model._thinker_to_talker_prefill(
        embeddings,
        embeddings[:2],
        None,
        sequence.unsqueeze(0),
        sequence,
        speaker_id=4,
    )
    assert actual_ids.tolist() == [IM_START, USER]
    torch.testing.assert_close(actual, embeddings[:2], rtol=0, atol=0)


@pytest.mark.parametrize("last_tail_row", [False, True])
def test_full_payload_decode_uses_current_request_pad_after_another_prefill(last_tail_row):
    model = _model()
    model.tts_pad_embed = torch.full((1, WIDTH), -123, dtype=torch.bfloat16)
    tail = torch.full((1, WIDTH), 70, dtype=torch.bfloat16) if last_tail_row else None
    _, _, update = model.talker_preprocess(
        torch.tensor([7]),
        None,
        _omni_is_prefill=False,
        meta={"decode_flag": True, "num_processed_tokens": 2},
        embed={"tts_pad_projected": torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16)},
        hidden_states={"trailing_text": tail},
    )
    assert torch.all(update["mtp_inputs"][1] == (70 if last_tail_row else 50))
    if last_tail_row:
        assert update["hidden_states"]["trailing_text"].shape == (1, WIDTH)
        assert torch.all(update["hidden_states"]["trailing_text"] == 50)


@pytest.mark.parametrize("text_count", [1, 3])
@pytest.mark.parametrize("assistant_only", [False, True])
def test_full_payload_identity_matches_scheduler_hashes_and_actual_model_rows(text_count, assistant_only):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin,
    )
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
        thinker2talker_full_payload,
        thinker2talker_token_only,
    )

    model = _model()
    model.config.talker_config.accept_hidden_layer = 6
    prompt, sequence, embeddings, hidden, users, start = _case(text_count, assistant_only)
    transfer = SimpleNamespace(_get_model_config=lambda: SimpleNamespace(hf_config=model.config))
    source = SimpleNamespace(request_id="source", prompt_token_ids=prompt, all_token_ids=sequence)
    payload = thinker2talker_full_payload(
        transfer,
        {
            "hidden_states.layer_0": embeddings,
            "hidden_states.layer_6": hidden,
            "embed.tts_pad": torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16),
            "embed.tts_bos": torch.full((1, 1, WIDTH), 60, dtype=torch.bfloat16),
            "embed.tts_eos": torch.full((1, 1, WIDTH), 70, dtype=torch.bfloat16),
        },
        source,
    )
    output = SimpleNamespace(cumulative_token_ids=sequence[len(prompt) :])
    admission = thinker2talker_token_only(
        [SimpleNamespace(request_id="source", prompt_token_ids=prompt, outputs=[output])],
        {"cache_salt": "original-caller"},
    )[0]
    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    request = Request(
        request_id="talker",
        prompt_token_ids=admission["prompt_token_ids"],
        cache_salt=admission.get("cache_salt"),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=hasher,
    )
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    metadata = _OmniConnectorPayloadTransportMixin._extract_scheduling_metadata(payload)
    coordinator.update_request_metadata({"talker": request}, {"talker": metadata})
    expected_ids = sum((sequence[first:last] for first, last in users), []) + [
        IM_START,
        ASSISTANT,
        NEWLINE,
        *([151671] * 4),
        151672,
        sequence[start + 3] if text_count > 1 else 151671,
    ]
    assert request.prompt_token_ids == expected_ids
    assert request.num_prompt_tokens == len(expected_ids)
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=expected_ids,
        cache_salt="original-caller",
        sampling_params=request.sampling_params,
        pooling_params=None,
        block_hasher=hasher,
    )
    assert request.block_hashes == fresh.block_hashes
    actual_ids, actual_embeddings, _ = model.talker_preprocess(
        torch.tensor(expected_ids),
        None,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        **payload,
    )
    assert actual_ids.tolist() == expected_ids
    assert actual_embeddings.shape[0] == request.num_prompt_tokens


def test_admission_counts_and_embedding_segments_share_the_same_layout():
    prompt, sequence, *_ = _case(3)
    # Generated text may contain role markers, but they are not new chat parts.
    sequence.extend([IM_START, USER, NEWLINE])
    parts = plan_talker_prefill(prompt, len(sequence))
    assert [(part.kind, part.start, part.end) for part in parts] == [
        ("user", 5, 12),
        ("user", 17, 22),
        ("assistant", 22, len(sequence)),
    ]
    assert sum(part.num_rows for part in parts) == 21
    assert compute_talker_prompt_ids_length(prompt) == 21
    assert _compute_talker_prompt_ids_length({"ids": {"prompt": prompt, "all": sequence}}) == 21


@pytest.mark.parametrize(
    "prompt, length",
    [([IM_START], 1), ([IM_START, ASSISTANT], 2), ([IM_START, 999, NEWLINE], 3), ([IM_START, USER, NEWLINE], 2)],
)
def test_invalid_layout_fails_before_embedding_assembly(prompt, length):
    with pytest.raises(ValueError):
        plan_talker_prefill(prompt, length)


def _fixed_payload(model, text_count=3, *, assistant_only=False, value_offset=0):
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_full_payload

    model.config.talker_config.accept_hidden_layer = 6
    prompt, sequence, embeddings, hidden, *_ = _case(text_count, assistant_only)
    return thinker2talker_full_payload(
        SimpleNamespace(_get_model_config=lambda: SimpleNamespace(hf_config=model.config)),
        {
            "hidden_states.layer_0": embeddings + value_offset,
            "hidden_states.layer_6": hidden + value_offset,
            "embed.tts_pad": torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16),
            "embed.tts_bos": torch.full((1, 1, WIDTH), 60, dtype=torch.bfloat16),
            "embed.tts_eos": torch.full((1, 1, WIDTH), 70, dtype=torch.bfloat16),
        },
        SimpleNamespace(request_id="source", prompt_token_ids=prompt, all_token_ids=sequence),
    )


@pytest.mark.parametrize("fixed_plan", [False, True])
@pytest.mark.parametrize("preface_tokens", [0, 2])
def test_prefill_excludes_tool_call_from_bootstrap_and_decode_tail(fixed_plan, preface_tokens):
    from vllm.sampling_params import SamplingParams

    from vllm_omni.model_executor.models.qwen3_omni.tool_speech import TOOL_CALL_START_TOKEN_ID
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_full_payload
    from vllm_omni.request import OmniRequest

    model = _model()
    model.config.talker_config.accept_hidden_layer = 6
    prompt, sequence, embeddings, hidden, *_ = _case(4, assistant_only=True)
    tool_offset = len(prompt) + preface_tokens
    sequence[tool_offset] = TOOL_CALL_START_TOKEN_ID
    request = OmniRequest(
        request_id="tool", prompt_token_ids=prompt, sampling_params=SamplingParams(max_tokens=4), pooling_params=None
    )
    request.append_output_token_ids(sequence[len(prompt) :])
    payload = thinker2talker_full_payload(
        SimpleNamespace(_get_model_config=lambda: SimpleNamespace(hf_config=model.config)),
        {
            "hidden_states.layer_0": embeddings,
            "hidden_states.layer_6": hidden,
            "embed.tts_pad": torch.full((1, 1, WIDTH), 50, dtype=torch.bfloat16),
            "embed.tts_bos": torch.full((1, 1, WIDTH), 60, dtype=torch.bfloat16),
            "embed.tts_eos": torch.full((1, 1, WIDTH), 70, dtype=torch.bfloat16),
        },
        request,
    )
    assert payload is not None
    assert payload["meta"]["talker_prefill_plan"] == [("assistant", 0, tool_offset)]
    prompt_ids = payload["meta"]["next_stage_prompt_ids"]
    if not fixed_plan:
        payload["meta"].pop("talker_prefill_plan")
    actual_ids, actual, update = model.talker_preprocess(
        torch.tensor(prompt_ids), None, _omni_is_prefill=True, _omni_num_computed_tokens=0, **payload
    )
    assert actual_ids[-1] == (sequence[3] if preface_tokens else model.config.tts_pad_token_id)
    first_text = embeddings[3] if preface_tokens else torch.zeros(WIDTH, dtype=torch.bfloat16)
    torch.testing.assert_close(actual[-1], first_text + 6, rtol=0, atol=0)
    expected_tail = torch.cat((embeddings[4:tool_offset], payload["embed"]["tts_eos"][0]))
    torch.testing.assert_close(update["hidden_states"]["trailing_text"], expected_tail, rtol=0, atol=0)


def _fixed_expected(payload, model):
    """Independent retained-row and nine-bootstrap oracle (no planner)."""
    embeddings = payload["embed"]["prefill"]
    hidden = payload["hidden_states"]["output"]
    sequence = payload["ids"]["all"]
    assistant_only = len(payload["ids"]["prompt"]) == 3
    spans, start = ([], 0) if assistant_only else ([(5, 12), (17, 22)], 22)
    rows, ids = [], []
    for first, last in spans:
        for i in range(first, last):
            rows.append(hidden[i] + 20 if sequence[i] in (IMAGE, AUDIO, VIDEO) else embeddings[i])
            ids.append(sequence[i])
    available_text = len(embeddings) > start + 3
    rows.extend(embeddings[start : start + 3])
    rows.extend(torch.full((WIDTH,), 50 + i, dtype=torch.bfloat16) for i in (1, 2, 3, 4))
    rows.append(torch.full((WIDTH,), 65, dtype=torch.bfloat16))
    rows.append(embeddings[start + 3] + 6 if available_text else torch.full((WIDTH,), 6, dtype=torch.bfloat16))
    ids.extend(
        [
            IM_START,
            ASSISTANT,
            NEWLINE,
            *([151671] * 4),
            151672,
            sequence[start + 3] if available_text else model.config.tts_pad_token_id,
        ]
    )
    tail = torch.cat((embeddings[start + 4 :], payload["embed"]["tts_eos"][0]))
    return ids, torch.stack(rows), tail


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 32])
@pytest.mark.parametrize("text_count", [1, 3])
def test_full_payload_plan_is_built_once_and_sliced_after_connector_round_trip(
    mocker, monkeypatch, chunk_size, text_count
):
    import importlib

    from vllm_omni.data_entry_keys import validate_payload
    from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    producer = importlib.import_module("vllm_omni.model_executor.stage_input_processors.qwen3_omni")
    consumer = importlib.import_module("vllm_omni.model_executor.models.qwen3_omni.qwen3_omni")
    planner = mocker.Mock(wraps=plan_talker_prefill)
    monkeypatch.setattr(producer, "plan_talker_prefill", planner)
    monkeypatch.setattr(
        consumer, "plan_talker_prefill", mocker.Mock(side_effect=AssertionError("consumer replanned fixed input"))
    )
    model = _model()
    payload = _fixed_payload(model, text_count)
    validate_payload(payload)
    payload = OmniMsgpackDecoder().decode(OmniMsgpackEncoder().encode(payload))
    assert planner.call_count == 1
    assert payload["meta"]["talker_prefill_plan"]
    expected_ids, expected, expected_tail = _fixed_expected(payload, model)
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.requests = {"consumer": SimpleNamespace()}
    runner.model_intermediate_buffer = {"consumer": payload}
    user_rows: list[int] = []
    original_user_parts = model._get_talker_user_parts

    def record_user_rows(first, last, *args):
        user_rows.extend(range(first, last))
        return original_user_parts(first, last, *args)

    monkeypatch.setattr(model, "_get_talker_user_parts", record_user_rows)
    actual_ids, actual = [], []
    for start in range(0, len(expected_ids), chunk_size):
        end = min(start + chunk_size, len(expected_ids))
        with monkeypatch.context() as context:
            context.setattr(
                torch.Tensor, "tolist", mocker.Mock(side_effect=AssertionError("fixed input read back token IDs"))
            )
            ids, embeddings, update = model.talker_preprocess(
                torch.tensor(expected_ids[start:end]),
                None,
                _omni_is_prefill=True,
                _omni_num_computed_tokens=start,
                **payload,
            )
        runner._update_intermediate_buffer("consumer", update)
        actual_ids.extend(ids.tolist())
        actual.append(embeddings)
    assert actual_ids == expected_ids
    torch.testing.assert_close(torch.cat(actual), expected, rtol=0, atol=0)
    torch.testing.assert_close(payload["hidden_states"]["trailing_text"], expected_tail, rtol=0, atol=0)
    assert user_rows == [*range(5, 12), *range(17, 22)]
    assert planner.call_count == 1


@pytest.mark.parametrize("start", range(12, 21))
def test_fixed_plan_hit_at_every_bootstrap_boundary_initializes_current_tail(mocker, monkeypatch, start):
    import importlib

    model = _model()
    payload = _fixed_payload(model)
    expected_ids, expected, tail = _fixed_expected(payload, model)
    module = importlib.import_module("vllm_omni.model_executor.models.qwen3_omni.qwen3_omni")
    monkeypatch.setattr(module, "plan_talker_prefill", mocker.Mock(side_effect=AssertionError("unexpected replan")))
    actual_ids, actual, update = model.talker_preprocess(
        torch.tensor(expected_ids[start:]), None, _omni_is_prefill=True, _omni_num_computed_tokens=start, **payload
    )
    assert actual_ids.tolist() == expected_ids[start:]
    torch.testing.assert_close(actual, expected[start:], rtol=0, atol=0)
    torch.testing.assert_close(update["hidden_states"]["trailing_text"], tail, rtol=0, atol=0)
    assert torch.all(update["embed"]["tts_pad_projected"] == 50)


def test_new_full_payload_replaces_plan_without_a_request_id_keyed_cache(mocker, monkeypatch):
    import importlib

    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    model = _model()
    old, new = _fixed_payload(model), _fixed_payload(model, assistant_only=True, value_offset=80)
    old_plan = list(old["meta"]["talker_prefill_plan"])
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner.requests = {"same-id": SimpleNamespace()}
    runner.model_intermediate_buffer = {"same-id": old}
    module = importlib.import_module("vllm_omni.model_executor.models.qwen3_omni.qwen3_omni")
    monkeypatch.setattr(module, "plan_talker_prefill", mocker.Mock(side_effect=AssertionError("unexpected replan")))
    for payload in (old, new, old):
        # Exercise the existing per-request buffer replacement boundary.
        runner._replace_intermediate_buffer("same-id", payload)
        current = runner.model_intermediate_buffer["same-id"]
        expected_ids, expected, tail = _fixed_expected(payload, model)
        actual_ids, actual, update = model.talker_preprocess(
            torch.tensor(expected_ids), None, _omni_is_prefill=True, _omni_num_computed_tokens=0, **current
        )
        assert actual_ids.tolist() == expected_ids
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(update["hidden_states"]["trailing_text"], tail, rtol=0, atol=0)
    assert old["meta"]["talker_prefill_plan"] == old_plan


@pytest.mark.parametrize(
    "bad_plan",
    [
        [],
        "caller-label",
        [("user", 5)],
        [("other", 5, 12)],
        [("user", True, 12)],
        [("user", -1, 12)],
        [("user", 5, 12), ("user", 10, 13)],
        [("user", 5, 999)],
        [("assistant", 22, 24)],
        [("assistant", 22, 27), ("assistant", 27, 30)],
    ],
)
def test_invalid_wire_plan_is_rejected_before_projections(mocker, monkeypatch, bad_plan):
    model = _model()
    payload = _fixed_payload(model)
    payload["meta"]["talker_prefill_plan"] = bad_plan
    monkeypatch.setattr(
        model.talker.text_projection, "forward", mocker.Mock(side_effect=AssertionError("invalid plan projected"))
    )
    with pytest.raises(ValueError, match="Talker prefill plan"):
        model.talker_preprocess(
            torch.tensor([151644]), None, _omni_is_prefill=True, _omni_num_computed_tokens=0, **payload
        )


@pytest.mark.parametrize("invalid", ["missing_ids", "short_ids", "past_end", "negative_start"])
def test_fixed_plan_requires_aligned_identity_and_interval(mocker, monkeypatch, invalid):
    model = _model()
    payload = _fixed_payload(model)
    start = 0
    if invalid == "missing_ids":
        payload["meta"].pop("next_stage_prompt_ids")
    elif invalid == "short_ids":
        payload["meta"]["next_stage_prompt_ids"].pop()
    else:
        start = 21 if invalid == "past_end" else -1
    monkeypatch.setattr(
        model.talker.text_projection, "forward", mocker.Mock(side_effect=AssertionError("invalid plan projected"))
    )
    with pytest.raises(ValueError, match="fixed.*plan|fixed Talker row plan"):
        model.talker_preprocess(
            torch.tensor([151644]), None, _omni_is_prefill=True, _omni_num_computed_tokens=start, **payload
        )


def test_fixed_plan_metadata_alone_does_not_wake_execution():
    from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
        _OmniConnectorPayloadTransportMixin,
    )

    payload = {"meta": {"talker_prefill_plan": [("assistant", 0, 3)]}}
    assert not _OmniConnectorPayloadTransportMixin._payload_is_consumable(payload)
