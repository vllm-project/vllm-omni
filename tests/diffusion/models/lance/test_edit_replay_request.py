# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.lance import pipeline_lance
from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel
from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def edit_pipeline(mocker):
    pipeline = object.__new__(LancePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.bagel = mocker.Mock()
    pipeline.bagel.config.llm_config.num_hidden_layers = 1
    pipeline.bagel.latent_downsample = 32
    pipeline.bagel.max_latent_size = 1
    pipeline.tokenizer = mocker.Mock()
    pipeline.tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text.split())))
    pipeline.vae = mocker.Mock()
    pipeline.scheduler = None
    pipeline.scheduler_kwargs = {}
    pipeline.new_token_ids = {}
    mocker.patch.object(pipeline, "_segment_strings", return_value=("prefix", "sep end"))
    mocker.patch.object(pipeline, "_autocast_kwargs", return_value={"device_type": "cpu", "enabled": False})
    mocker.patch.object(pipeline, "_regen_init_noise_on_device")
    mocker.patch.object(pipeline, "_decode_image_from_latent", return_value=Image.new("RGB", (32, 32)))
    mocker.patch.object(pipeline, "_decode_video_from_latent", return_value=np.zeros((1, 32, 32, 3), dtype=np.uint8))
    mocker.patch.object(
        LanceBagel,
        "_lance_video_preprocess",
        return_value=(torch.zeros(3, 1, 32, 32), torch.zeros(1), torch.tensor([[1, 2, 2]]), 1),
    )

    def reference_prefill(ctx, *args, collect=None, collect_role=None, **kwargs):
        if collect is not None:
            segment = {"role": collect_role, "rows": torch.ones(2, 4), "positions": torch.arange(2)}
            if collect_role == "ref_vae":
                segment["gen_indexes"] = torch.tensor([1])
            collect.append(segment)
        ctx["kv_lens"][0] += 2
        ctx["ropes"][0] += 2

    for name in ("_vit_image_prefill", "_vit_video_prefill", "_vae_ref_prefill"):
        mocker.patch.object(pipeline, name, side_effect=reference_prefill)
    latent_input = {"packed_position_ids": torch.zeros(3, 2), "packed_vae_position_ids": torch.arange(2)}
    for name in ("prepare_vae_latent", "prepare_video_latent"):
        getattr(pipeline.bagel, name).return_value = latent_input
    for name in ("prepare_vae_latent_cfg", "prepare_video_latent_cfg"):
        getattr(pipeline.bagel, name).return_value = {"cfg_packed_position_ids": torch.zeros(3, 2)}

    def generate_image(**kwargs):
        trajectory = [torch.ones(2)] if kwargs["return_trajectory_latents"] else []
        return torch.zeros(1, 2), trajectory, trajectory, trajectory

    pipeline.bagel.generate_image.side_effect = generate_image
    return pipeline


def edit_request(mode, instruction_tokens, *, export=False, text_cap=None, cfg_text_scale=4.0):
    media = Image.new("RGB", (32, 32)) if mode == "image" else np.zeros((1, 32, 32, 3), dtype=np.uint8)
    prompt = {
        "user_text": " ".join(["edit"] * instruction_tokens),
        "multi_modal_data": {"img2img" if mode == "image" else "video": media},
        "extra_args": {"cfg_text_scale": cfg_text_scale},
    }
    sampling = OmniDiffusionSamplingParams(
        return_trajectory_latents=export, max_sequence_length=text_cap, num_inference_steps=1, seed=1
    )
    return DiffusionRequestBatch(
        requests=[OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id="edit")]
    )


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("text_cap", [None, 8])
@pytest.mark.parametrize("cfg_text_scale", [1.0, 4.0])
def test_ordinary_edit_skips_replay_for_long_instructions(edit_pipeline, mocker, mode, text_cap, cfg_text_scale):
    export = mocker.spy(pipeline_lance, "_edit_replay_condition")
    prefill = mocker.spy(edit_pipeline, "_raw_text_prefill")
    request = edit_request(mode, 129, text_cap=text_cap, cfg_text_scale=cfg_text_scale)

    result = getattr(edit_pipeline, f"_forward_{mode}_edit")(request)

    assert result.output["metadata"] == {mode: {"shape": (32, 32) if mode == "image" else (1, 32, 32)}}
    output = result.output["payload"][mode]
    if mode == "image":
        assert isinstance(output, Image.Image)
    else:
        assert len(output) == 1
        assert isinstance(output[0], Image.Image)
    assert "trajectory" not in result.output["payload"]
    export.assert_not_called()
    assert all(call.kwargs.get("collect") is None for call in prefill.call_args_list)
    edit_pipeline.bagel.generate_image.assert_called_once()


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize(("text_cap", "tokens"), [(None, 0), (2, 0), (None, 3), (8, 3), (None, 126), (8, 6)])
@pytest.mark.parametrize("cfg_text_scale", [1.0, 4.0])
def test_rl_edit_keeps_padded_replay_metadata(edit_pipeline, mode, text_cap, tokens, cfg_text_scale):
    request = edit_request(mode, tokens, export=True, text_cap=text_cap, cfg_text_scale=cfg_text_scale)

    result = getattr(edit_pipeline, f"_forward_{mode}_edit")(request)

    replay = result.output["metadata"]["rl"]
    cap = 128 if text_cap is None else text_cap
    assert replay["condition_gen_tail_ids"].shape == (1, cap)
    tail_length = tokens + 2
    assert replay["condition_gen_tail_ids"][0, :tail_length].tolist() == [*range(tokens), 0, 1]
    assert replay["condition_gen_tail_mask"].tolist() == [[True] * tail_length + [False] * (cap - tail_length)]
    assert replay["condition_cfg_tail_mask"].tolist() == [[True] * 2 + [False] * (cap - 2)]
    assert replay["condition_ref_rows"].shape == (1, 4, 4)
    assert replay["condition_latent_positions"].shape == (1, 3, 2)
    assert set(result.output["payload"]["trajectory"]) == {"latents", "timesteps", "log_probs"}


@pytest.mark.parametrize("export", [False, True])
def test_empty_text_prefill_preserves_context(edit_pipeline, mocker, export):
    cache = mocker.Mock()
    ctx = {"kv_lens": [7], "ropes": [5], "past_key_values": cache}
    segments: list[dict] | None = [] if export else None

    edit_pipeline._raw_text_prefill(ctx, "", collect=segments, collect_role="gen_instruction")

    edit_pipeline.bagel.forward_cache_update_text.assert_not_called()
    assert ctx == {"kv_lens": [7], "ropes": [5], "past_key_values": cache}
    if segments is not None:
        assert len(segments) == 1
        assert segments[0]["role"] == "gen_instruction"
        for name in ("ids", "positions"):
            assert segments[0][name].shape == (0,)
            assert segments[0][name].dtype == torch.long
            assert segments[0][name].device.type == "cpu"


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize(("text_cap", "tokens"), [(None, 127), (8, 7), (0, 1), (-1, 1)])
def test_rl_edit_rejects_over_cap_before_preprocessing(edit_pipeline, mocker, mode, text_cap, tokens):
    prefill = mocker.spy(edit_pipeline, "_raw_text_prefill")
    request = edit_request(mode, tokens, export=True, text_cap=text_cap)

    with pytest.raises(ValueError, match="edit replay segment 'gen_tail'.*above the"):
        getattr(edit_pipeline, f"_forward_{mode}_edit")(request)

    prefill.assert_not_called()
    LanceBagel._lance_video_preprocess.assert_not_called()
    edit_pipeline._vit_image_prefill.assert_not_called()
    edit_pipeline._vit_video_prefill.assert_not_called()
    edit_pipeline._vae_ref_prefill.assert_not_called()
    edit_pipeline.bagel.generate_image.assert_not_called()
    edit_pipeline._decode_image_from_latent.assert_not_called()
    edit_pipeline._decode_video_from_latent.assert_not_called()
