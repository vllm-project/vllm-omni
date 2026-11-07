# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.soprano.vocos import SopranoDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decoder_exposes_generation_worker_protocol():
    from vllm.model_executor.models.interfaces_base import is_text_generation_model

    from vllm_omni.model_executor.models.soprano.soprano_decoder import SopranoDecoderForGeneration

    assert is_text_generation_model(SopranoDecoderForGeneration)


def test_decoder_interpolation_istft_shape_and_precision():
    # Real small CPU forward catches hop/center/interpolation mistakes without weights.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        decoder = SopranoDecoder().float().eval()
        for length in [2, 3, 6]:
            with torch.inference_mode():
                audio = decoder(torch.zeros(1, 512, length))
            assert audio.shape == (1, 2048 * (length - 1))
            assert audio.dtype == torch.float32
            assert torch.isfinite(audio).all()
        with pytest.raises(ValueError, match="float32"):
            decoder(torch.zeros(1, 512, 3, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="T >= 2"):
            decoder(torch.zeros(1, 512, 1))
    finally:
        torch.set_num_threads(previous)


def test_production_loader_rejects_incomplete_decoder_checkpoint(tmp_path, mocker):
    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.soprano.soprano_decoder import SopranoDecoderForGeneration

    config = mocker.Mock(spec=VllmConfig)
    config.model_config = mocker.Mock(model=str(tmp_path), dtype=torch.float32)
    torch.save({"head.out.bias": torch.zeros(2050)}, tmp_path / "decoder.pth")
    model = SopranoDecoderForGeneration(vllm_config=config)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_weights(iter(()))


def test_decoder_loader_does_not_read_stage_zero_weights(tmp_path, mocker):
    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.soprano import soprano_decoder

    decoder = torch.nn.Linear(2, 2)
    state = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}
    torch.save(state, tmp_path / "decoder.pth")
    mocker.patch.object(soprano_decoder, "SopranoDecoder", return_value=decoder)
    config = mocker.Mock(spec=VllmConfig)
    config.model_config = mocker.Mock(model=str(tmp_path), dtype=torch.float32)
    config.device_config = mocker.Mock(device=torch.device("cpu"))

    def stage_zero_weights():
        pytest.fail("The decoder must not materialize AR tensors owned by stage zero")
        yield "lm_head.weight", torch.empty(0)

    model = soprano_decoder.SopranoDecoderForGeneration(vllm_config=config)
    assert model.load_weights(stage_zero_weights()) == {"decoder.weight", "decoder.bias"}
    assert torch.equal(decoder.weight, state["weight"])
    assert torch.equal(decoder.bias, state["bias"])
