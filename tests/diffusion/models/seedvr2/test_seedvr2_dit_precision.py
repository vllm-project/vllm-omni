# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from unittest.mock import patch

import pytest
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torch import nn
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.seedvr2 import pipeline_seedvr2 as sr
from vllm_omni.diffusion.models.seedvr2.nadit import NaDiTOutput, NaMMRotaryEmbedding3d
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("unsupported", ["auto", "fp8", None, 16])
def test_invalid_dit_dtype(unsupported):
    with pytest.raises(ValueError, match="seedvr2_dit_compute_dtype"):
        sr.get_dit_compute_dtype({"seedvr2_dit_compute_dtype": unsupported})


def test_velocity_bridge():
    velocity = torch.tensor([65280.0, -65280.0, 0.25], dtype=torch.bfloat16)
    torch.testing.assert_close(sr.velocity_to_fp16(velocity), velocity.half(), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="BF16"):
        sr.velocity_to_fp16(velocity.float())


@pytest.mark.parametrize("invalid", [65536.0, -65536.0, float("nan"), float("inf"), -float("inf")])
def test_reject_invalid_velocity(invalid):
    with pytest.raises(RuntimeError, match="non-finite"):
        sr.velocity_to_fp16(torch.tensor([1.0, invalid], dtype=torch.bfloat16))


class TinyDiT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.rope = NaMMRotaryEmbedding3d(rotary_dim=24)

    def token_grid_for(self, shape):
        return tuple(shape[0].tolist())

    def build_runtime(self, grid, *, text_len, parallel_config):
        return None

    def forward(self, video, text, shape, text_shape, timestep, runtime):
        self.autocast_enabled = torch.is_autocast_enabled("cpu")
        # 256 squared overflows FP16 before subtracting back into FP16 range.
        intermediate = self.scale * self.scale
        value = intermediate - intermediate + 0.25
        return NaDiTOutput(vid_sample=value.expand(video.shape[0], 16))


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def encode(self, video):
        assert video.dtype == torch.float16
        moments = video.new_zeros((1, 32, 1, 2, 2))
        return DiagonalGaussianDistribution(moments)

    def decode(self, latent):
        self.decoder_input = latent
        return latent[:, :3]


def make_tiny_pipeline(tmp_path, monkeypatch, additional_config, text):
    monkeypatch.setattr(sr, "SeedVR2NaDiT", TinyDiT)
    monkeypatch.setattr(sr, "SeedVR2VAE", TinyVAE)
    monkeypatch.setattr(sr, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sr, "_device_memory", lambda: None)
    monkeypatch.setattr(sr, "set_torch_threads_for_runtime", lambda: None)
    torch.save(text, tmp_path / "pos_emb.pt")
    config = OmniDiffusionConfig(
        model=str(tmp_path), dtype=torch.float16, enforce_eager=True, additional_config=additional_config
    )
    with set_default_torch_dtype(torch.float16):
        return sr.SeedVR2Pipeline(od_config=config)


def make_request(request_id):
    return OmniDiffusionRequest(
        prompt={"prompt": "", "multi_modal_data": {"video": torch.zeros(1, 3, 16, 16)}},
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(
            height=16, width=16, seed=17, fps=24, extra_args={"color_correction_method": "none"}
        ),
    )


@pytest.mark.parametrize(
    "additional_config, dtype", [({}, torch.float16), ({"seedvr2_dit_compute_dtype": "bfloat16"}, torch.bfloat16)]
)
def test_pipeline_component_dtypes(tmp_path, monkeypatch, additional_config, dtype):
    checked: list[torch.dtype] = []
    monkeypatch.setattr(sr.current_platform, "check_if_supports_dtype", checked.append)
    # Direct BF16 rounding of this value differs from rounding through checkpoint FP16.
    text = torch.full((1, 5120), 0.99795, dtype=torch.float32)
    assert not torch.equal(text.bfloat16(), text.half().bfloat16())
    pipeline = make_tiny_pipeline(tmp_path, monkeypatch, additional_config, text)
    bf16 = dtype == torch.bfloat16
    assert checked == ([torch.bfloat16] if bf16 else [])
    assert pipeline.transformer.scale.dtype == dtype
    assert pipeline.vae.weight.dtype == torch.float16
    torch.testing.assert_close(pipeline.text, text.half().to(dtype), rtol=0, atol=0)

    # At least one FP16 checkpoint frequency changes if accidentally rounded to BF16.
    freqs = torch.tensor([1.0, 0.789, 0.246, 0.123], dtype=torch.float16)
    assert not torch.equal(freqs.float(), freqs.bfloat16().float())
    with torch.no_grad():
        pipeline.load_weights(
            [
                ("transformer.scale", torch.tensor(256.0, dtype=torch.float16)),
                ("transformer.rope.rope.freqs", freqs),
                ("vae.weight", torch.ones(())),
            ]
        )
    torch.testing.assert_close(pipeline.transformer.rope.freqs, freqs.float(), rtol=0, atol=0)

    with torch.autocast("cpu", dtype=torch.float16):
        pipeline(DiffusionRequestBatch([make_request("seedvr2-precision")]))
        assert torch.is_autocast_enabled("cpu")
    assert pipeline.transformer.autocast_enabled is (not bf16)
    assert pipeline.vae.decoder_input.dtype == torch.float16
    if bf16:
        assert pipeline.vae.decoder_input.isfinite().all()
    else:
        assert pipeline.vae.decoder_input.isnan().all()


@pytest.mark.parametrize("bad_index", [0, 1], ids=["invalid-first", "invalid-last"])
def test_nonfinite_velocity_isolated_to_request(tmp_path, monkeypatch, bad_index):
    monkeypatch.setattr(sr.current_platform, "check_if_supports_dtype", lambda dtype: None)
    pipeline = make_tiny_pipeline(
        tmp_path, monkeypatch, {"seedvr2_dit_compute_dtype": "bfloat16"}, torch.zeros(1, 5120)
    )
    reference = pipeline(DiffusionRequestBatch([make_request("valid")]))[0]
    # Inject a nonfinite value into the DiT output for just one of two requests.
    bad_request = iter([bad_index == 0, bad_index == 1])

    def inject_nonfinite(module, args, output):
        if next(bad_request):
            return NaDiTOutput(vid_sample=torch.full_like(output.vid_sample, float("nan")))
        return output

    requests = [make_request("invalid" if i == bad_index else "valid") for i in range(2)]
    with (
        pipeline.transformer.register_forward_hook(inject_nonfinite),
        patch.object(pipeline.vae, "decode", wraps=pipeline.vae.decode) as decode,
    ):
        outputs = pipeline(DiffusionRequestBatch(requests))

    assert len(outputs) == 2
    bad, good = outputs[bad_index], outputs[1 - bad_index]
    assert bad.error and "non-finite" in bad.error
    assert bad.output is None
    assert good.error is None
    assert isinstance(good.output, dict) and isinstance(reference.output, dict)
    torch.testing.assert_close(good.output["payload"]["video"], reference.output["payload"]["video"])
    decode.assert_called_once()
