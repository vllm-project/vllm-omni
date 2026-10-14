# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed-conditioning identity is independent of delivery/request ownership."""

from dataclasses import replace

import pytest

from vllm_omni.model_executor.models.qwen3_omni.talker_conditioning import (
    TalkerProcessedMediaIdentity,
    complete_talker_conditioning_digest,
    thinker_conditioning_digest,
)
from vllm_omni.model_executor.models.qwen3_omni.talker_input_plan import plan_talker_prefill, talker_prefill_token_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IM, SYS, USER, ASSISTANT, NL = 151644, 8948, 872, 77091, 198
IMAGE, AUDIO, VIDEO = 151655, 151646, 151656


def _conditioning():
    prompt = [IM, SYS, NL, 10, IM, USER, NL, IMAGE, AUDIO, VIDEO, IM, ASSISTANT, NL]
    sequence = prompt + [50, 51, 52]
    media = tuple(
        TalkerProcessedMediaIdentity(
            modality=kind,
            content_digest=str(index + 1) * 64,
            preprocessing_digest="f" * 64,
            offset=offset,
            length=1,
        )
        for index, (kind, offset) in enumerate((("image", 7), ("audio", 8), ("video", 9)))
    )
    return dict(
        prompt_ids=prompt,
        sequence_ids=sequence,
        row_plan=plan_talker_prefill(prompt, len(sequence)),
        embedding_rows=len(sequence),
        hidden_rows=len(sequence),
        text_controls=(151671, 151672, 151673),
        codec_controls=(4203, 4204, 4205, 7, 4196, 4197),
        media_token_ids=(IMAGE, AUDIO, VIDEO),
        source_model="thinker-revision-A",
        source_adapter=None,
        talker_model="talker-revision-A",
        talker_adapter=None,
        projection="projection-v1",
        processed_media=media,
    )


def _conditioning_digest(*, talker_model, talker_adapter, projection, codec_controls, **source):
    return complete_talker_conditioning_digest(
        source_digest=thinker_conditioning_digest(**source),
        talker_model=talker_model,
        talker_adapter=talker_adapter,
        projection=projection,
        text_controls=source["text_controls"],
        codec_controls=codec_controls,
    )


def _request_hashes(conditioning, request_id):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    from vllm_omni.core.sched.input_finalization import install_request_input, prepare_request_input

    token_ids = talker_prefill_token_ids(
        conditioning["row_plan"],
        conditioning["sequence_ids"],
        tts_pad_token_id=conditioning["text_controls"][0],
        tts_bos_token_id=conditioning["text_controls"][1],
    )
    init_none_hash(sha256)
    request = Request(
        request_id=request_id,
        prompt_token_ids=[0] * len(token_ids),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=get_request_block_hasher(4, sha256),
        cache_salt="caller",
    )
    candidate = prepare_request_input(
        request,
        prompt_token_ids=token_ids,
        mm_features=[],
        cache_salt="caller",
        conditioning_digest=_conditioning_digest(**conditioning),
        sampling_params=request.sampling_params,
    )
    install_request_input(request, candidate)
    assert request.block_hashes
    return request.block_hashes


def test_equivalent_complete_conditioning_has_canonical_identity():
    first = _conditioning()
    second = _conditioning()
    second["processed_media"] = tuple(reversed(second["processed_media"]))
    assert _conditioning_digest(**first) == _conditioning_digest(**second)
    assert len(_conditioning_digest(**first)) == 64
    assert _request_hashes(first, "producer-A") == _request_hashes(second, "different-consumer-B")
    # There is deliberately no request ID, generation, timestamp or decode
    # state argument. None of these can enter this fixed-input identity.
    for forbidden in ("request_id", "generation", "timestamp", "decode_flag"):
        with pytest.raises(TypeError):
            arguments = {**first, forbidden: "owner-only"}
            _conditioning_digest(**arguments)


@pytest.mark.parametrize(
    "dependency",
    [
        "system",
        "suffix",
        "speaker",
        "control",
        "text-control",
        "embedding_rows",
        "hidden_rows",
        "row_plan",
        "source_model",
        "talker_model",
        "projection",
        "source_adapter",
        "talker_adapter",
        "media",
        "preprocessing",
    ],
)
def test_each_effective_dependency_changes_identity(dependency):
    args = _conditioning()
    before = _conditioning_digest(**args)
    before_hashes = _request_hashes(args, "A")
    if dependency in {"system", "suffix"}:
        args["sequence_ids"][3 if dependency == "system" else -1] += 1
        if dependency == "system":
            args["prompt_ids"][3] += 1
    elif dependency in {"speaker", "control", "text-control"}:
        field = "text_controls" if dependency == "text-control" else "codec_controls"
        values = list(args[field])
        values[3 if dependency == "speaker" else 0] += 1
        args[field] = tuple(values)
    elif dependency in {"embedding_rows", "hidden_rows"}:
        args[dependency] -= 1
        args["row_plan"] = plan_talker_prefill(
            args["prompt_ids"],
            len(args["sequence_ids"]),
            embedding_rows=args["embedding_rows"],
            hidden_rows=args["hidden_rows"],
        )
    elif dependency == "row_plan":
        args["row_plan"] = (*args["row_plan"][:-1], replace(args["row_plan"][-1], end=13))
    elif dependency in {"media", "preprocessing"}:
        field = "content_digest" if dependency == "media" else "preprocessing_digest"
        args["processed_media"] = (
            replace(args["processed_media"][0], **{field: "a" * 64}),
            *args["processed_media"][1:],
        )
    else:
        args[dependency] = "different"
    assert _conditioning_digest(**args) != before
    assert _request_hashes(args, "B") != before_hashes


def test_media_position_and_embedding_mask_are_part_of_identity():
    args = _conditioning()
    before = _conditioning_digest(**args)
    # Include the preceding text row in an explicit sparse placeholder range.
    item = replace(args["processed_media"][0], offset=6, length=2, is_embed=(False, True))
    args["processed_media"] = (item, *args["processed_media"][1:])
    assert _conditioning_digest(**args) != before


@pytest.mark.parametrize("invalid", ["missing", "raw-dict", "label", "bad-mask", "wrong-modality", "outside"])
def test_media_requires_complete_processed_identity_and_valid_layout(invalid):
    args = _conditioning()
    item, *rest = args["processed_media"]
    if invalid == "missing":
        args["processed_media"] = ()
    elif invalid == "raw-dict":
        args["processed_media"] = ({"modality": "image", "identifier": "caller-uuid"}, *rest)
    elif invalid == "label":
        args["processed_media"] = (replace(item, content_digest="caller-uuid"), *rest)
    elif invalid == "bad-mask":
        args["processed_media"] = (replace(item, is_embed=(False,)), *rest)
    elif invalid == "wrong-modality":
        args["processed_media"] = (replace(item, modality="audio"), *rest)
    else:
        args["processed_media"] = (replace(item, offset=len(args["prompt_ids"])), *rest)
    with pytest.raises(ValueError):
        _conditioning_digest(**args)


def test_text_only_inputs_need_no_media_provenance():
    args = _conditioning()
    for index in (7, 8, 9):
        args["prompt_ids"][index] = args["sequence_ids"][index] = 20 + index
    args["processed_media"] = ()
    assert _conditioning_digest(**args)


def test_sequence_must_include_the_declared_source_context():
    args = _conditioning()
    args["sequence_ids"][0] += 1
    with pytest.raises(ValueError, match="source prompt"):
        _conditioning_digest(**args)
