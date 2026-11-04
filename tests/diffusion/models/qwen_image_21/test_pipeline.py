# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU unit tests for QwenImage21Pipeline helpers: prepare_latents duplication
order, KV-cache assembly across step phases, and the post-process func."""

import json
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import (
    QwenImage21Pipeline,
    _should_encode_negative_prompt,
    get_qwen_image_21_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _toy_pipeline() -> QwenImage21Pipeline:
    """A pipeline shell with the attributes prepare_latents reads, no weights."""
    pipeline = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig()
    pipeline.vae_scale_factor = 16
    pipeline.latent_channels = 4
    return pipeline


@pytest.mark.parametrize("image_count", [0, 1, 2])
@pytest.mark.parametrize("use_cfg", [False, True])
@pytest.mark.parametrize("num_outputs", [1, 2])
def test_encode_batch_preserves_encoder_batch_and_request_conditioning(mocker, image_count, use_cfg, num_outputs):
    pipeline = _toy_pipeline()
    pipeline.device = torch.device("cpu")
    pipeline._max_length = 4096
    sizes = [(32, 32), (64, 32)][:image_count]
    requests = []
    for index, name in enumerate(("first", "second")):
        requests.append(
            OmniDiffusionRequest(
                request_id=name,
                prompt={
                    "prompt": name,
                    "negative_prompt": f"negative-{name}" if use_cfg else None,
                    "additional_information": {
                        "prompt_image": [Image.new("RGB", size) for size in sizes],
                        "vae_images": [
                            torch.full((1, 4, 1, height // 16, width // 16), float(10 * index + slot))
                            for slot, (width, height) in enumerate(sizes)
                        ],
                        "vae_images_are_latents": True,
                        "input_image_sizes": sizes,
                    },
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=64,
                    width=96,
                    true_cfg_scale=3.0 if use_cfg else 1.0,
                    num_outputs_per_prompt=num_outputs,
                    generator=[torch.Generator().manual_seed(index + n) for n in range(num_outputs)],
                ),
            )
        )

    def encode_prompt(*, prompt, images_per_prompt, **kwargs):
        # A singleton encoder call would change both batching and row identity.
        assert prompt in (["first", "second"], ["negative-first", "negative-second"])
        if image_count:
            assert [len(images) for images in images_per_prompt] == [image_count, image_count]
        else:
            assert images_per_prompt is None
        embeds = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        image_mask = torch.tensor([[True, False, False], [False, True, False]])
        return embeds, mask, image_mask

    encoder = mocker.patch.object(pipeline, "encode_prompt", side_effect=encode_prompt)
    outputs = pipeline.encode_batch(DiffusionRequestBatch(requests=requests))

    assert encoder.call_count == (2 if use_cfg else 1)
    assert len(outputs) == 2
    for index, output in enumerate(outputs):
        payload = output.custom_output
        conditioning = pipeline._stage_conditioning(payload)
        assert conditioning["prompt_embeds"].shape == (1, 3, 4)
        torch.testing.assert_close(
            conditioning["prompt_embeds"], torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)[index : index + 1]
        )
        assert conditioning["prompt_embeds_mask"].tolist() == ([[1, 1, 0]] if index == 0 else [[1, 1, 1]])
        assert conditioning["image_pad_mask"].tolist() == (
            [[True, False, False]] if index == 0 else [[False, True, False]]
        )
        if use_cfg:
            torch.testing.assert_close(conditioning["negative_prompt_embeds"], conditioning["prompt_embeds"])
            assert conditioning["negative_prompt_embeds_mask"].tolist() == conditioning["prompt_embeds_mask"].tolist()
            assert conditioning["negative_image_pad_mask"].tolist() == conditioning["image_pad_mask"].tolist()
        else:
            assert conditioning["negative_prompt_embeds"] is None
        if image_count:
            for slot, (width, height) in enumerate(sizes):
                image = conditioning["image_latents"][slot]
                assert image.shape == (1, 4, 1, height // 16, width // 16)
                assert torch.all(image == 10 * index + slot)
        else:
            assert conditioning["image_latents"] is None
        assert payload["qwen_image_21_metadata"] == {
            "height": 64,
            "width": 96,
            "input_image_sizes": sizes,
            "has_negative_conditioning": use_cfg,
        }


# ---------------------------------------------------------------------------
# prepare_latents
# ---------------------------------------------------------------------------


def test_prepare_latents_duplicates_condition_latents_request_major():
    """batch=2 prompts x n=2 outputs must order condition latents A,A,B,B.

    ``encode_prompt`` expands prompt embeddings request-major (A,A,B,B); the
    condition latents must follow the same order or generations 2 and 3 swap
    condition images. ``repeat`` would produce the wrong (A,B,A,B) tile-major
    order; ``repeat_interleave`` is correct.
    """
    pipeline = _toy_pipeline()
    cond_a = torch.full((1, 4, 1, 8, 8), 1.0)
    cond_b = torch.full((1, 4, 1, 8, 8), 2.0)
    stacked = torch.cat([cond_a, cond_b], dim=0)  # one row per request

    latents, image_latents = pipeline.prepare_latents(
        [stacked],
        batch_size=4,
        num_channels_latents=4,
        height=64,
        width=64,
        image_is_latent=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )

    assert latents.shape == (4, 16, 4)  # packed: 4x4 latent tokens, 4 channels
    assert image_latents.shape == (4, 64, 4)  # packed: 8x8 condition tokens
    row_means = image_latents.mean(dim=(1, 2))
    torch.testing.assert_close(row_means, torch.tensor([1.0, 1.0, 2.0, 2.0]))


def test_prepare_latents_without_duplication_when_rows_match_batch():
    pipeline = _toy_pipeline()
    stacked = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1, 1).expand(4, 4, 1, 2, 2).contiguous()

    _, image_latents = pipeline.prepare_latents(
        [stacked],
        batch_size=4,
        num_channels_latents=4,
        height=32,
        width=32,
        image_is_latent=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(image_latents.mean(dim=(1, 2)), torch.arange(4, dtype=torch.float32))


def test_prepare_latents_rejects_non_divisible_duplication():
    pipeline = _toy_pipeline()
    stacked = torch.zeros(2, 4, 1, 8, 8)

    with pytest.raises(ValueError, match="Cannot duplicate"):
        pipeline.prepare_latents(
            [stacked],
            batch_size=3,
            num_channels_latents=4,
            height=64,
            width=64,
            image_is_latent=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(0),
        )


def test_prepare_latents_rejects_generator_list_length_mismatch():
    pipeline = _toy_pipeline()

    with pytest.raises(ValueError, match="list of generators"):
        pipeline.prepare_latents(
            None,
            batch_size=2,
            num_channels_latents=4,
            height=32,
            width=32,
            image_is_latent=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=[torch.Generator().manual_seed(0)],
        )


def test_prepare_latents_concatenates_multiple_condition_images():
    pipeline = _toy_pipeline()
    first = torch.full((2, 4, 1, 4, 4), 1.0)
    second = torch.full((2, 4, 1, 2, 2), 3.0)

    _, image_latents = pipeline.prepare_latents(
        [first, second],
        batch_size=2,
        num_channels_latents=4,
        height=32,
        width=32,
        image_is_latent=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )

    # Token dim concatenates the packed condition images: 16 + 4 per row.
    assert image_latents.shape == (2, 20, 4)
    torch.testing.assert_close(image_latents[:, :16].mean(), torch.tensor(1.0))
    torch.testing.assert_close(image_latents[:, 16:].mean(), torch.tensor(3.0))


# ---------------------------------------------------------------------------
# KV-cache assembly across step phases
# ---------------------------------------------------------------------------


def _state(cache) -> SimpleNamespace:
    return SimpleNamespace(extra={"kv_cache": cache})


def _decode_cache(rows: int, value: float) -> list[dict]:
    return [
        {
            "cond": {
                "key": torch.full((rows, 2), value),
                "value": torch.full((rows, 2), value + 0.5),
            }
        }
    ]


def test_kv_cache_phase_classification():
    assert QwenImage21Pipeline._kv_cache_phase(None) == "none"
    assert QwenImage21Pipeline._kv_cache_phase([{}]) == "prefill"
    assert QwenImage21Pipeline._kv_cache_phase(_decode_cache(1, 0.0)) == "decode"


def test_assemble_kv_cache_all_none():
    assert QwenImage21Pipeline._assemble_kv_cache([_state(None), _state(None)]) == (None, False)


def test_assemble_kv_cache_prefill_returns_fresh_owned_cache():
    states = [_state([{} for _ in range(3)]), _state([{} for _ in range(3)])]

    kv_cache, take_ownership = QwenImage21Pipeline._assemble_kv_cache(states)

    assert take_ownership is True
    assert kv_cache == [{}, {}, {}]
    # Fresh blocks, not aliases of either request's cache: the batched prefill
    # writes into these and _scatter_kv_cache splits them back per request.
    for block in kv_cache:
        assert all(block is not request_block for state in states for request_block in state.extra["kv_cache"])


def test_assemble_kv_cache_decode_concatenates_in_request_order():
    first, second = _decode_cache(2, 1.0), _decode_cache(1, 3.0)
    states = [_state(first), _state(second)]

    kv_cache, take_ownership = QwenImage21Pipeline._assemble_kv_cache(states)

    assert take_ownership is False
    assert len(kv_cache) == 1
    cond = kv_cache[0]["cond"]
    torch.testing.assert_close(cond["key"], torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 3.0]]))
    torch.testing.assert_close(cond["value"], torch.tensor([[1.5, 1.5], [1.5, 1.5], [3.5, 3.5]]))
    # Decode only reads the cache: per-request tensors must stay untouched.
    assert first[0]["cond"]["key"].shape == (2, 2)
    assert second[0]["cond"]["key"].shape == (1, 2)


def test_assemble_kv_cache_decode_merges_every_part():
    first = [{"cond": {"key": torch.ones(1, 1), "value": torch.ones(1, 1), "key_scale": torch.ones(1)}}]
    second = [{"cond": {"key": torch.zeros(1, 1), "value": torch.zeros(1, 1), "key_scale": torch.zeros(1)}}]

    kv_cache, take_ownership = QwenImage21Pipeline._assemble_kv_cache([_state(first), _state(second)])

    assert take_ownership is False
    assert set(kv_cache[0]["cond"]) == {"key", "value", "key_scale"}
    torch.testing.assert_close(kv_cache[0]["cond"]["key_scale"], torch.tensor([1.0, 0.0]))


def test_assemble_kv_cache_mixed_phases_raises():
    """Defensive backstop: a late prefill must never join a decoding batch.

    The preprocessor sets ``allow_mixed_step_phases = False`` so the
    StepScheduler defers late requests until the batch finishes decoding; this
    raise is the last line of defense if a mixed batch is ever assembled.
    """
    states = [_state([{}]), _state(_decode_cache(1, 0.0))]

    with pytest.raises(ValueError, match="mixed KV-cache phases"):
        QwenImage21Pipeline._assemble_kv_cache(states)

    with pytest.raises(ValueError, match="mixed KV-cache phases"):
        QwenImage21Pipeline._assemble_kv_cache([_state(None), _state(_decode_cache(1, 0.0))])


def test_assemble_kv_cache_decode_rejects_mismatched_prefill_prompt_seq_len():
    """Homogeneous merge invariant: denoise_step must sub-batch before assembling."""
    short = _state(_decode_cache(1, 1.0))
    short.extra["prefill_prompt_seq_len"] = 4
    long = _state(_decode_cache(1, 2.0))
    long.extra["prefill_prompt_seq_len"] = 8

    with pytest.raises(ValueError, match="sub-batch by prefill_prompt_seq_len"):
        QwenImage21Pipeline._assemble_kv_cache([short, long])

    # Same baked length remains batchable.
    same_a = _state(_decode_cache(1, 1.0))
    same_a.extra["prefill_prompt_seq_len"] = 4
    same_b = _state(_decode_cache(1, 2.0))
    same_b.extra["prefill_prompt_seq_len"] = 4
    _, take_ownership = QwenImage21Pipeline._assemble_kv_cache([same_a, same_b])
    assert take_ownership is False


def test_split_decode_groups_by_prefill_prompt_seq_len():
    """Routine serving: separate prefills → joint decode must split, not fail."""
    short = _state(_decode_cache(1, 1.0))
    short.request_id = "short"
    short.extra["prefill_prompt_seq_len"] = 4
    long = _state(_decode_cache(1, 2.0))
    long.request_id = "long"
    long.extra["prefill_prompt_seq_len"] = 8
    short2 = _state(_decode_cache(1, 3.0))
    short2.request_id = "short2"
    short2.extra["prefill_prompt_seq_len"] = 4

    states = [short, long, short2]
    assert QwenImage21Pipeline._needs_decode_seq_len_split(states) is True
    groups = QwenImage21Pipeline._split_decode_groups(states)
    assert [[s.request_id for s in g] for g in groups] == [["short", "short2"], ["long"]]

    same = [_state(_decode_cache(1, 1.0)), _state(_decode_cache(1, 2.0))]
    for s in same:
        s.extra["prefill_prompt_seq_len"] = 4
    assert QwenImage21Pipeline._needs_decode_seq_len_split(same) is False
    assert QwenImage21Pipeline._split_decode_groups(same) == [same]


def test_scatter_kv_cache_splits_rows_back_per_request():
    states = [_state([{}]), _state([{}])]
    merged = [
        {
            "cond": {
                "key": torch.tensor([[1.0], [2.0], [3.0]]),
                "value": torch.tensor([[4.0], [5.0], [6.0]]),
            }
        }
    ]

    QwenImage21Pipeline._scatter_kv_cache(states, [1, 2], merged)

    torch.testing.assert_close(states[0].extra["kv_cache"][0]["cond"]["key"], torch.tensor([[1.0]]))
    torch.testing.assert_close(states[0].extra["kv_cache"][0]["cond"]["value"], torch.tensor([[4.0]]))
    torch.testing.assert_close(states[1].extra["kv_cache"][0]["cond"]["key"], torch.tensor([[2.0], [3.0]]))
    scattered = states[1].extra["kv_cache"][0]["cond"]["key"]
    assert scattered.is_contiguous()
    # Must own storage (clone), not a view into the merged prefill tensor —
    # otherwise step-mode CUDA-graph decode never matches owner identity.
    assert scattered.data_ptr() != merged[0]["cond"]["key"][1:3].data_ptr()
    merged[0]["cond"]["key"].fill_(0.0)
    torch.testing.assert_close(scattered, torch.tensor([[2.0], [3.0]]))


def test_encode_prompt_expands_masks_request_major():
    """Two prompts × n=2 must keep embeds and masks in A,A,B,B order.

    A 2D ``mask.repeat(1, n, 1).view(B*n, S)`` would wrongly produce A,B,A,B.
    """
    pipeline = _toy_pipeline()
    embeds = torch.tensor([[[1.0], [1.0]], [[2.0], [0.0]]])  # [B=2, S=2, D=1]
    mask = torch.tensor([[True, True], [True, False]])
    image_pad = torch.tensor([[False, True], [False, False]])

    def _fake_get(prompt, images_per_prompt, max_sequence_length=None, prompt_name="prompt"):
        del prompt, images_per_prompt, max_sequence_length, prompt_name
        return embeds.clone(), mask.clone(), image_pad.clone()

    pipeline._get_qwen_prompt_embeds = _fake_get  # type: ignore[method-assign]

    out_embeds, out_mask, out_pad = pipeline.encode_prompt(
        ["short", "loooooonger"],
        num_images_per_prompt=2,
    )

    assert out_embeds.shape[0] == 4
    # embeds request-major: A,A,B,B
    torch.testing.assert_close(out_embeds[:, 0, 0], torch.tensor([1.0, 1.0, 2.0, 2.0]))
    # masks must follow the same order (not A,B,A,B)
    assert out_mask.tolist() == [
        [True, True],
        [True, True],
        [True, False],
        [True, False],
    ]
    assert out_pad.tolist() == [
        [False, True],
        [False, True],
        [False, False],
        [False, False],
    ]


def test_resolve_prompt_seq_len_keeps_prefill_padding_after_batch_shrink():
    """Longest-prompt cancel must not shrink the surviving request's KV layout."""
    short = _state([{}])
    short.extra["image_pad_mask"] = torch.tensor([[True, False]])
    short.extra["prefill_prompt_seq_len"] = 4  # baked in when batched with a longer prompt

    # Current InputBatch only sees the short prompt's native length.
    assert QwenImage21Pipeline._resolve_prompt_seq_len([short], current_seq_len=2) == 4
    padded = QwenImage21Pipeline._pad_rows_to_seq_len([short.extra["image_pad_mask"]], 4)
    assert padded.tolist() == [[True, False, False, False]]


# ---------------------------------------------------------------------------
# post-process func
# ---------------------------------------------------------------------------


@pytest.fixture
def postprocess(tmp_path):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
    return get_qwen_image_21_post_process_func(SimpleNamespace(model=str(tmp_path)))


def test_postprocess_tensor_returns_pil_images(postprocess):
    from PIL import Image

    images = torch.rand(2, 4, 16, 16) * 2 - 1

    result = postprocess(images)

    assert len(result) == 2
    assert all(isinstance(image, Image.Image) for image in result)


def test_postprocess_envelope_unwraps_payload_and_keeps_metadata(postprocess):
    from PIL import Image

    envelope = {
        "payload": {"image": torch.rand(1, 4, 16, 16), "seed": 42},
        "metadata": {"request_id": "abc"},
    }

    result = postprocess(envelope)

    assert result["metadata"] == {"request_id": "abc"}
    assert result["payload"]["seed"] == 42
    assert len(result["payload"]["image"]) == 1
    assert isinstance(result["payload"]["image"][0], Image.Image)


def test_postprocess_envelope_requires_image_payload(postprocess):
    with pytest.raises(ValueError, match=r"payload\['image'\]"):
        postprocess({"payload": {"seed": 42}, "metadata": {}})


def test_release_captured_graphs_delegates_to_transformer() -> None:
    pipeline = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    released: list[bool] = []
    pipeline.transformer = SimpleNamespace(release_captured_graphs=lambda: released.append(True))

    pipeline.release_captured_graphs()

    assert released == [True]


def test_postprocess_envelope_normalizes_non_dict_metadata(postprocess):
    result = postprocess({"payload": {"image": torch.rand(1, 4, 16, 16)}, "metadata": "bogus"})

    assert result["metadata"] == {}


@pytest.mark.parametrize(
    "field,value", [("height", 96), ("width", 128), ("true_cfg_scale", 3), ("max_sequence_length", 128)]
)
def test_encode_rejects_incompatible_direct_batch_before_encoding(mocker, field, value):
    pipeline = _toy_pipeline()
    requests = [
        OmniDiffusionRequest(
            request_id=str(i),
            prompt="text",
            sampling_params=OmniDiffusionSamplingParams(height=64, width=64, true_cfg_scale=1),
        )
        for i in range(2)
    ]
    setattr(requests[1].sampling_params, field, value)
    encode = mocker.patch.object(pipeline, "_encode_conditioning")
    with pytest.raises(ValueError, match=field):
        pipeline.encode_batch(DiffusionRequestBatch(requests=requests))
    encode.assert_not_called()


@pytest.mark.parametrize(
    "first_type", [list, tuple, np.array, torch.tensor, partial(torch.tensor, dtype=torch.bfloat16)]
)
@pytest.mark.parametrize(
    "second_type", [list, tuple, np.array, torch.tensor, partial(torch.tensor, dtype=torch.bfloat16)]
)
@pytest.mark.parametrize("compatible", [False, True])
def test_diffuse_compares_custom_sigma_values_before_generation(mocker, first_type, second_type, compatible):
    pipeline = _toy_pipeline()
    requests = [
        OmniDiffusionRequest(
            request_id=str(i),
            prompt="text",
            sampling_params=OmniDiffusionSamplingParams(height=64, width=64),
        )
        for i in range(2)
    ]
    requests[0].sampling_params.sigmas = first_type([0.75, 0.5, 0.25])
    requests[1].sampling_params.sigmas = second_type([0.75, 0.5, 0.25 if compatible else 0.125])
    collate = mocker.patch.object(pipeline, "_collate_stage_conditioning")
    generate = mocker.patch.object(pipeline, "_generate_latents", return_value=(torch.zeros(2, 16, 4), 64, 64, 1))
    batch = DiffusionRequestBatch(requests=requests)
    if compatible:
        assert len(pipeline.diffuse_batch(batch)) == 2
        generate.assert_called_once()
    else:
        with pytest.raises(ValueError, match="requires matching sigmas"):
            pipeline.diffuse_batch(batch)
        collate.assert_not_called()
        generate.assert_not_called()


@pytest.mark.parametrize("sigma_type", [list, np.array, torch.tensor])
@pytest.mark.parametrize("missing_index", [0, 1])
def test_diffuse_rejects_missing_custom_sigmas(mocker, sigma_type, missing_index):
    pipeline = _toy_pipeline()
    requests = [
        OmniDiffusionRequest(request_id=str(i), prompt="text", sampling_params=OmniDiffusionSamplingParams())
        for i in range(2)
    ]
    requests[1 - missing_index].sampling_params.sigmas = sigma_type([0.75, 0.5])
    with pytest.raises(ValueError, match="requires matching sigmas"):
        pipeline.diffuse_batch(DiffusionRequestBatch(requests=requests))


def test_sigma_comparison_does_not_round_reference_values_to_tensor_dtype():
    requests = [
        OmniDiffusionRequest(request_id=str(i), prompt="text", sampling_params=OmniDiffusionSamplingParams())
        for i in range(2)
    ]
    requests[0].sampling_params.sigmas = torch.tensor([0.1], dtype=torch.float32)
    requests[1].sampling_params.sigmas = [0.1]
    with pytest.raises(ValueError, match="requires matching sigmas"):
        QwenImage21Pipeline._validate_stage_batch_settings(DiffusionRequestBatch(requests=requests), ("sigmas",))


@pytest.mark.parametrize("true_cfg_scale", [None, False, True, 1.0, 3.0, np.float32(3.0)])
@pytest.mark.parametrize("negative_prompt", [None, "", "blurry", [""], ["blurry", ""]])
def test_negative_conditioning_presence_accepts_explicit_string_batches(true_cfg_scale, negative_prompt):
    assert _should_encode_negative_prompt(true_cfg_scale, negative_prompt) is (
        true_cfg_scale == 3.0 and negative_prompt is not None
    )


@pytest.mark.parametrize(
    "true_cfg_scale",
    [
        "bad",
        "1.0",
        [],
        {},
        float("nan"),
        float("inf"),
        -float("inf"),
        torch.tensor(3.0),
        pytest.param(10**400, id="huge-int"),
        pytest.param(-(10**400), id="huge-negative-int"),
    ],
)
def test_encode_conditioning_rejects_invalid_cfg_scale_before_encoding(mocker, true_cfg_scale):
    pipeline = _toy_pipeline()
    encoder = mocker.patch.object(pipeline, "encode_prompt")
    with pytest.raises(ValueError, match="true_cfg_scale"):
        pipeline._encode_conditioning(["prompt"], None, [], true_cfg_scale, 128)
    encoder.assert_not_called()


@pytest.mark.parametrize("negative_prompt", [[], [None], ["blurry", 1], ("blurry",), 1])
def test_encode_conditioning_rejects_invalid_negative_prompts_before_encoding(mocker, negative_prompt):
    pipeline = _toy_pipeline()
    encoder = mocker.patch.object(pipeline, "encode_prompt")
    with pytest.raises(ValueError, match="non-empty list of strings"):
        pipeline._encode_conditioning(["prompt"], negative_prompt, [], 3.0, 128)
    encoder.assert_not_called()


def test_encode_conditioning_requires_one_negative_per_positive_prompt(mocker):
    pipeline = _toy_pipeline()
    encoder = mocker.patch.object(pipeline, "encode_prompt")
    with pytest.raises(ValueError, match="one string per prompt"):
        pipeline._encode_conditioning(["one", "two"], ["blurry"], [], 3.0, 128)
    encoder.assert_not_called()


@pytest.mark.parametrize("true_cfg_scale", [None, 0.0, 1.0])
@pytest.mark.parametrize("negative_prompt", [[], [None], ["one", "two"], ("placeholder",), 1])
def test_cfg_disabled_ignores_negative_prompt_placeholders(mocker, true_cfg_scale, negative_prompt):
    pipeline = _toy_pipeline()
    mocker.patch.object(
        pipeline,
        "encode_prompt",
        return_value=(torch.ones(1, 2, 4), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool)),
    )
    conditioning = pipeline._encode_conditioning(["prompt"], negative_prompt, [], true_cfg_scale, 128)
    assert conditioning["negative_prompt_embeds"] is None
    pipeline.encode_prompt.assert_called_once()


def test_encode_conditioning_treats_bare_positive_prompt_as_one_request(mocker):
    pipeline = _toy_pipeline()
    encoder = mocker.patch.object(
        pipeline,
        "encode_prompt",
        return_value=(torch.ones(1, 2, 4), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool)),
    )
    output = pipeline._encode_conditioning("a long prompt", ["blurry"], [], 3.0, 128)
    assert output["negative_prompt_embeds"] is not None
    assert encoder.call_count == 2


@pytest.mark.parametrize("true_cfg_scale", [None, 1.0, 3.0])
@pytest.mark.parametrize("negative_prompt", [None, "", "blurry"])
@pytest.mark.parametrize("image_count", [0, 1])
def test_negative_conditioning_key_matches_encoded_payload(
    mocker, tmp_path, true_cfg_scale, negative_prompt, image_count
):
    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import get_qwen_image_21_pre_process_func

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps({"temperal_downsample": [False] * 4}))
    preprocess = get_qwen_image_21_pre_process_func(SimpleNamespace(model=str(tmp_path)))
    pipeline = _toy_pipeline()
    pipeline.device = torch.device("cpu")
    pipeline._max_length = 128
    encoder = mocker.patch.object(
        pipeline,
        "encode_prompt",
        return_value=(torch.ones(1, 2, 4), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool)),
    )
    mocker.patch.object(pipeline, "_encode_vae_image", return_value=torch.ones(1, 4, 1, 2, 2))
    request = preprocess(
        OmniDiffusionRequest(
            request_id="raw",
            prompt={
                "prompt": "teapot",
                "negative_prompt": negative_prompt,
                "multi_modal_data": {"image": [Image.new("RGB", (32, 32))] * image_count},
            },
            sampling_params=OmniDiffusionSamplingParams(height=64, width=64, true_cfg_scale=true_cfg_scale),
        )
    )
    output = pipeline.encode_batch(DiffusionRequestBatch(requests=[request]))[0]
    downstream = preprocess(
        OmniDiffusionRequest(request_id="encoded", prompt=output.custom_output, sampling_params=request.sampling_params)
    )
    expected = true_cfg_scale == 3.0 and negative_prompt is not None
    assert request.batch_compatibility_key == downstream.batch_compatibility_key
    assert downstream.batch_compatibility_key[-1] is expected
    assert (output.custom_output["qwen_image_21_conditioning"]["negative_prompt_embeds"] is not None) is expected
    assert encoder.call_count == (2 if expected else 1)


@pytest.mark.parametrize("num_outputs", [1, 2])
@pytest.mark.parametrize("use_cfg", [False, True])
def test_stage_generator_and_decoder_use_native_batches(mocker, num_outputs, use_cfg):
    from vllm_omni.diffusion.data import DiffusionOutput

    pipeline = _toy_pipeline()
    pipeline.device = torch.device("cpu")
    rows = []
    requests = []
    for i, length in enumerate((2, 3)):
        rows.append(
            {
                "version": 1,
                "prompt_embeds": torch.full((1, length, 4), float(i + 1)),
                "prompt_embeds_mask": torch.ones(1, length),
                "image_pad_mask": torch.zeros(1, length, dtype=torch.bool),
                "negative_prompt_embeds": torch.full((1, length + 2, 4), float(i + 3)) if use_cfg else None,
                "negative_prompt_embeds_mask": torch.ones(1, length + 2) if use_cfg else None,
                "negative_image_pad_mask": torch.zeros(1, length + 2, dtype=torch.bool) if use_cfg else None,
                "image_latents": [torch.full((1, 4, 1, 2, 2), float(i + 10))],
                "input_image_sizes": [(32, 32)],
            }
        )
        requests.append(
            OmniDiffusionRequest(
                request_id=str(i),
                prompt={"row": i},
                sampling_params=OmniDiffusionSamplingParams(height=64, width=64, num_outputs_per_prompt=num_outputs),
            )
        )
    mocker.patch.object(pipeline, "_stage_conditioning", side_effect=lambda p: rows[p["row"]])
    packed = torch.arange(2 * num_outputs, dtype=torch.float32).view(-1, 1, 1).expand(-1, 16, 4)
    generate = mocker.patch.object(pipeline, "_generate_latents", return_value=(packed, 64, 64, num_outputs))
    batch = DiffusionRequestBatch(requests=requests)
    outputs = pipeline.diffuse_batch(batch)
    generate.assert_called_once()
    assert generate.call_args.args[0] is batch
    conditioning = generate.call_args.kwargs["conditioning"]
    assert conditioning["prompt_embeds"].shape == (2, 3, 4)
    assert conditioning["prompt_embeds_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert conditioning["prompt_embeds"][0, -1].count_nonzero() == 0
    assert not conditioning["image_pad_mask"].any()
    if use_cfg:
        assert conditioning["negative_prompt_embeds"].shape == (2, 5, 4)
        assert conditioning["negative_prompt_embeds_mask"].tolist() == [[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]
        assert conditioning["negative_prompt_embeds"][0, -1].count_nonzero() == 0
        assert not conditioning["negative_image_pad_mask"].any()
    assert conditioning["image_latents"][0][:, 0, 0, 0, 0].tolist() == [10, 11]
    for i, output in enumerate(outputs):
        torch.testing.assert_close(output.custom_output["latents"], packed[i * num_outputs : (i + 1) * num_outputs])
        requests[i].prompt = output.custom_output
    decoded = torch.arange(2 * num_outputs, dtype=torch.float32).view(-1, 1, 1, 1)
    decode = mocker.patch.object(pipeline, "_decode_latents", return_value=DiffusionOutput(output=decoded))
    outputs = pipeline.decode_batch(batch)
    decode.assert_called_once()
    torch.testing.assert_close(decode.call_args.args[0], packed)
    for i, output in enumerate(outputs):
        torch.testing.assert_close(output.output, decoded[i * num_outputs : (i + 1) * num_outputs])


@pytest.mark.parametrize("already_latent", [False, True])
def test_conditioning_uses_explicit_image_representation_with_equal_channels(mocker, already_latent):
    pipeline = _toy_pipeline()
    pipeline.device = torch.device("cpu")
    # Raw RGBA has four channels too: channel count cannot determine representation.
    image = torch.ones(1, 4, 1, 2, 2)
    encoded = torch.full_like(image, 7)
    mocker.patch.object(
        pipeline,
        "encode_prompt",
        return_value=(torch.ones(1, 2, 4), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool)),
    )
    encode_vae = mocker.patch.object(pipeline, "_encode_vae_image", return_value=encoded)
    info = {
        "prompt_images": [Image.new("RGBA", (32, 32))],
        "vae_images": [image],
        "input_image_sizes": [(32, 32)],
        "vae_images_are_latents": already_latent,
    }
    conditioning = pipeline._encode_conditioning(["prompt"], None, [info], 1.0, 128)
    torch.testing.assert_close(conditioning["image_latents"][0], image if already_latent else encoded)
    if already_latent:
        encode_vae.assert_not_called()
    else:
        encode_vae.assert_called_once()


@pytest.mark.parametrize("already_latent", [False, True])
def test_prepare_latents_uses_explicit_image_representation(mocker, already_latent):
    pipeline = _toy_pipeline()
    image = torch.ones(1, 4, 1, 2, 2)
    encode_vae = mocker.patch.object(pipeline, "_encode_vae_image", return_value=image * 7)
    _, conditioning = pipeline.prepare_latents(
        [image],
        1,
        4,
        32,
        32,
        torch.float32,
        torch.device("cpu"),
        torch.Generator().manual_seed(1),
        image_is_latent=already_latent,
    )
    torch.testing.assert_close(conditioning, torch.full((1, 4, 4), 1.0 if already_latent else 7.0))
    assert encode_vae.call_count == (0 if already_latent else 1)


@pytest.mark.parametrize("max_sequence_length,expected", [(None, 8192), (128, 128)])
def test_generator_dummy_profiles_configured_maximum_text_length(max_sequence_length, expected):
    pipeline = _toy_pipeline()
    pipeline.device = torch.device("cpu")
    pipeline._max_length = 8192
    pipeline.od_config.tf_model_config = {"context_in_dim": 4}
    request = OmniDiffusionRequest(
        request_id="dummy",
        prompt="warmup",
        sampling_params=OmniDiffusionSamplingParams(max_sequence_length=max_sequence_length),
    )
    pipeline._prepare_dummy_denoise_payload(DiffusionRequestBatch(requests=[request]))
    conditioning = request.prompt["qwen_image_21_conditioning"]
    assert conditioning["prompt_embeds"].shape == (1, expected, 4)
    assert conditioning["negative_prompt_embeds"].shape == (1, expected, 4)
    assert request.sampling_params.max_sequence_length == max_sequence_length
