# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interleaved image spans must not introduce extra chat turn boundaries."""

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class PackingCompleteError(Exception):
    pass


@pytest.mark.parametrize(
    "prompt, spans",
    [
        (
            "<|im_start|>user\n<|image_pad|>\nDescribe it.<|im_end|>\n<|im_start|>assistant\n",
            ["user\n", "\nDescribe it.<|im_end|>\n<|im_start|>assistant\n"],
        ),
        ("make it blue <|image_pad|>", ["make it blue ", ""]),
        ("first<|image_pad|>second<|image_pad|>", ["first", "second", ""]),
    ],
)
def test_interleaved_spans_share_one_wrap_and_cfg_tokenizes_joined_text(mocker, prompt, spans):
    pipeline = object.__new__(BagelPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = mocker.Mock(dtype=torch.float32)
    pipeline.tokenizer = mocker.Mock()
    # A tokenizer with a merge across the boundary when the image is removed.
    pipeline.tokenizer.encode.side_effect = lambda text, **_: [3] if text == "".join(spans) else [4, 5]
    pipeline.new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}
    pipeline.language_model = mocker.Mock(vocab_size=10)
    pipeline.image_processor = mocker.Mock()
    pipeline.vae = mocker.Mock()
    pipeline.vae.encode.side_effect = lambda _: torch.randn(1, 4, 8, 8)
    bagel = mocker.MagicMock()
    bagel.max_latent_size = 32
    bagel.latent_downsample = 8
    bagel.config.llm_config.num_hidden_layers = 1
    pipeline.bagel = bagel
    calls, events = [], []
    nonempty = [text for text in spans if text.strip()]

    def prepare_prompts(**kwargs):
        result = Bagel.prepare_prompts(None, **kwargs)
        calls.append((kwargs["prompts"][0], result[0]["packed_text_ids"].tolist()))
        events.append("negative" if kwargs["prompts"] == ["negative"] else "text")
        if kwargs["prompts"] == ["negative"]:
            # Each image contributes three fake VAE tokens; positive text must
            # never enter this independent text-unconditional context.
            assert kwargs["curr_kvlens"] == [3 * (len(spans) - 1)]
        if len(calls) == len(nonempty) + 2:
            raise PackingCompleteError
        return result

    def prepare_image(**kwargs):
        events.append("image")
        assert max(kwargs["images"][0].size) <= 256
        return (
            {"padded_images": torch.zeros(1, 3, 32, 32)},
            [kwargs["curr_kvlens"][0] + 3],
            [kwargs["curr_rope"][0] + 1],
        )

    bagel.prepare_prompts.side_effect = prepare_prompts
    bagel.prepare_vae_images.side_effect = prepare_image

    def prepare_vit(**kwargs):
        assert kwargs["images"][0].size == (2000, 1500)
        assert kwargs["transforms"](kwargs["images"][0]).shape == (3, 728, 980)
        return {}, kwargs["curr_kvlens"], kwargs["curr_rope"]

    bagel.prepare_vit_images.side_effect = prepare_vit
    request = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt={
                    "prompt": prompt,
                    "negative_prompt": "negative",
                    "multi_modal_data": {"image": [Image.new("RGB", (2000, 1500)) for _ in spans[1:]]},
                },
                sampling_params=OmniDiffusionSamplingParams(),
                request_id="interleave",
            )
        ]
    )
    with pytest.raises(PackingCompleteError):
        pipeline.forward(request)

    assert [text for text, _ in calls] == nonempty + ["negative", "".join(spans)]
    gen_ids = [token for _, ids in calls[:-2] for token in ids]
    assert gen_ids.count(1) == gen_ids.count(2) == 1
    assert gen_ids[0] == 1 and gen_ids[-1] == 2
    assert calls[-1][1] == [1, 3, 2]
    expected = []
    for i, span in enumerate(spans):
        if span.strip():
            expected.append("text")
        if i < len(spans) - 1:
            expected.extend(["image", "image"])
    assert events == expected + ["negative", "text"]

    assert pipeline.vae.encode.call_count == len(spans) - 1
    updates = bagel.forward_cache_update_vae.call_args_list
    assert len(updates) == 2 * (len(spans) - 1)
    for gen, cfg in zip(updates[::2], updates[1::2]):
        assert gen.kwargs["padded_latent"] is cfg.kwargs["padded_latent"]
        assert gen.args[1] is not cfg.args[1]
    if len(spans) > 2:
        assert updates[0].kwargs["padded_latent"] is not updates[2].kwargs["padded_latent"]
