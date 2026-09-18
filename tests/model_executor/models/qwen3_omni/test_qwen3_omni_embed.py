# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run Qwen3-Omni embedding regressions on a Linux/CUDA environment.

The oracle assigns features to token positions independently of the model's
merge helpers. LM call counts additionally catch issue #7451: ordinary
multimodal inputs must not repeat the text embedding lookup. Small synthetic
weights suffice; no checkpoint is downloaded or loaded.
"""

import sys

import pytest
import torch
from pytest_mock import MockerFixture

from benchmarks.qwen3_omni.embedding_harness import (
    ModelShape,
    assert_case_output,
    build_case,
    expected_outputs,
    make_thinker,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_SHAPE = ModelShape(
    vocab_size=64,
    hidden_size=8,
    deepstack_levels=3,
    image_token_id=60,
    video_token_id=61,
    audio_token_id=62,
)

_SCENARIOS = (
    "text",
    "empty",
    "audio",
    "image",
    "video",
    "mixed",
    "interleaved",
    "vision_no_deepstack",
)


@pytest.fixture(scope="module", autouse=True)
def require_cuda_environment():
    if sys.platform != "linux" or torch.version.cuda is None or not torch.cuda.is_available():
        pytest.fail(
            "This validation suite requires a Linux/CUDA environment; "
            "CPU execution and an all-skipped result are not valid substitutes.",
            pytrace=False,
        )
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.fail("This validation suite requires native BF16 support on the current CUDA device.", pytrace=False)


@torch.inference_mode()
def _embed(model, case):
    return model.embed_input_ids(
        case.input_ids,
        multimodal_embeddings=case.fresh_embeddings(),
        is_multimodal=case.is_multimodal,
    )


@pytest.mark.parametrize(
    ("scenario", "dtype"),
    [pytest.param(scenario, torch.bfloat16, id=f"{scenario}-bf16") for scenario in _SCENARIOS]
    + [pytest.param(scenario, torch.float32, id=f"{scenario}-fp32") for scenario in ("image", "mixed", "interleaved")],
)
def test_embedding_positions_and_single_lm_lookup(mocker: MockerFixture, scenario: str, dtype: torch.dtype):
    model = make_thinker(_SHAPE, device="cuda", dtype=dtype, deepstack=scenario != "vision_no_deepstack")
    case = build_case(
        scenario,
        16,
        0 if scenario in {"text", "empty"} else 8,
        _SHAPE,
        device="cuda",
        dtype=dtype,
        mask_device="cpu",
    )
    lm_embed = mocker.spy(model.language_model, "embed_input_ids")

    result = _embed(model, case)

    # Unlike upstream-only mocks, make_thinker returns an instance of Omni's
    # actual thinker class, so its zero-argument super() follows the real MRO.
    lm_embed.assert_called_once()
    assert_case_output(model, case, result)


@pytest.mark.parametrize("scenario", ["image", "mixed", "interleaved"])
def test_repeated_calls_keep_original_deepstack_features(mocker: MockerFixture, scenario: str):
    model = make_thinker(_SHAPE, device="cuda", dtype=torch.bfloat16)
    case = build_case(scenario, 16, 8, _SHAPE, device="cuda", dtype=torch.bfloat16, mask_device="cpu")
    original_embeddings = case.fresh_embeddings()
    assert original_embeddings
    original_shapes = [tensor.shape for tensor in original_embeddings]
    original_modalities = [getattr(tensor, "modality", None) for tensor in original_embeddings]
    lm_embed = mocker.spy(model.language_model, "embed_input_ids")
    previous_list = None

    for _ in range(3):
        embeddings = case.fresh_embeddings()
        assert embeddings is not original_embeddings
        assert embeddings is not previous_list
        assert [tensor.shape for tensor in embeddings] == original_shapes
        assert [getattr(tensor, "modality", None) for tensor in embeddings] == original_modalities
        lm_embed.reset_mock()

        with torch.inference_mode():
            result = model.embed_input_ids(
                case.input_ids,
                multimodal_embeddings=embeddings,
                is_multimodal=case.is_multimodal,
            )

        lm_embed.assert_called_once()
        assert_case_output(model, case, result)
        # torch.split replaces entries in the supplied list with main-scale
        # views. The next call must receive the original multiscale tensors.
        assert [tensor.shape for tensor in original_embeddings] == original_shapes
        model._clear_deepstack_input_embeds(case.input_ids.numel())
        previous_list = embeddings


@pytest.mark.parametrize("scenario", ["audio", "image", "mixed", "interleaved"])
def test_oov_multimodal_ids_are_masked_before_lm_lookup(mocker: MockerFixture, scenario: str):
    model = make_thinker(_SHAPE, device="cuda", dtype=torch.bfloat16, oov=True)
    case = build_case(scenario, 16, 8, _SHAPE, device="cuda", dtype=torch.bfloat16, mask_device="cpu", oov=True)
    original_ids = case.input_ids.clone()
    mask = case.is_multimodal.to(device=case.input_ids.device)
    assert torch.all(case.input_ids[mask] >= _SHAPE.vocab_size)
    lm_embed = mocker.spy(model.language_model, "embed_input_ids")

    result = _embed(model, case)

    lm_embed.assert_called_once()
    received_ids = lm_embed.call_args.args[0]
    torch.testing.assert_close(received_ids, original_ids.masked_fill(mask, 0), rtol=0, atol=0)
    torch.testing.assert_close(case.input_ids, original_ids, rtol=0, atol=0)
    assert_case_output(model, case, result)


@pytest.mark.parametrize("next_scenario", ["text", "empty"])
def test_padded_deepstack_read_and_clear_do_not_leak_into_decode(mocker: MockerFixture, next_scenario: str):
    model = make_thinker(_SHAPE, device="cuda", dtype=torch.bfloat16, buffer_capacity=32)
    image_case = build_case("image", 16, 8, _SHAPE, device="cuda", dtype=torch.bfloat16, mask_device="cpu")
    image_result = _embed(model, image_case)
    assert_case_output(model, image_case, image_result)
    _, expected_deepstack = expected_outputs(model, image_case)
    assert expected_deepstack is not None

    num_valid = image_case.input_ids.numel()
    num_padded = 24
    # Padding may contain old buffer contents. _get must zero the tail while
    # retaining all valid visual features, as the real forward requires.
    for buffer in model.deepstack_input_embeds:
        buffer[num_valid:num_padded].fill_(77)
    padded = model._get_deepstack_input_embeds(num_padded)
    assert padded is not None
    for level in range(_SHAPE.deepstack_levels):
        actual = padded[f"deepstack_input_embeds_{level}"]
        torch.testing.assert_close(actual[:num_valid], expected_deepstack[level])
        assert torch.count_nonzero(actual[num_valid:]).item() == 0

    model._clear_deepstack_input_embeds(num_padded)
    assert model.deepstack_input_embeds_num_tokens == 0
    for buffer in model.deepstack_input_embeds:
        assert torch.count_nonzero(buffer[:num_padded]).item() == 0

    decode_case = build_case(next_scenario, 1, 0, _SHAPE, device="cuda", dtype=torch.bfloat16, mask_device="cpu")
    lm_embed = mocker.spy(model.language_model, "embed_input_ids")
    decode_result = _embed(model, decode_case)
    lm_embed.assert_called_once()
    assert_case_output(model, decode_case, decode_result)
    decode_deepstack = model._get_deepstack_input_embeds(2)
    assert decode_deepstack is not None
    for level in range(_SHAPE.deepstack_levels):
        assert torch.count_nonzero(decode_deepstack[f"deepstack_input_embeds_{level}"]).item() == 0


@torch.inference_mode()
def test_deepstack_set_resizes_for_larger_scheduled_windows(mocker: MockerFixture):
    model = make_thinker(_SHAPE, device="cuda", dtype=torch.bfloat16, buffer_capacity=4)
    lm_embed = mocker.spy(model.language_model, "embed_input_ids")

    for num_tokens in (16, 24):
        case = build_case("mixed", num_tokens, 8, _SHAPE, device="cuda", dtype=torch.bfloat16, mask_device="cpu")
        lm_embed.reset_mock()
        result = _embed(model, case)

        lm_embed.assert_called_once()
        assert_case_output(model, case, result)
        assert model.deepstack_input_embeds_num_tokens == num_tokens
        assert all(buffer.size(0) >= num_tokens for buffer in model.deepstack_input_embeds)
        model._clear_deepstack_input_embeds(num_tokens)


@pytest.mark.parametrize("mask_device", ["cpu", "cuda"])
@pytest.mark.parametrize("scenario", ["image", "interleaved"])
def test_cuda_embeddings_accept_cpu_or_cuda_mask(mocker: MockerFixture, scenario: str, mask_device: str):
    model = make_thinker(_SHAPE, device="cuda", dtype=torch.bfloat16)
    case = build_case(scenario, 16, 8, _SHAPE, device="cuda", dtype=torch.bfloat16, mask_device=mask_device)
    lm_embed = mocker.spy(model.language_model, "embed_input_ids")

    result = _embed(model, case)

    lm_embed.assert_called_once()
    assert result.device.type == "cuda"
    assert_case_output(model, case, result)
