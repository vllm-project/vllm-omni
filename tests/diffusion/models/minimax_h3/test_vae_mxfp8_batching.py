# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize("world", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("windows,tiles", [(1, 1), (3, 7), (21, 28)])
def test_schedule_covers_each_tile_once(world, windows, tiles):
    from vllm_omni.diffusion.models.minimax_h3.vae_batching import jobs

    schedule = jobs(windows, tiles, world)
    flattened = [job for row in schedule for rank in row for job in rank]
    assert len(flattened) == len(set(flattened)) == windows * tiles
    assert set(flattened) == {(window, tile) for window in range(windows) for tile in range(tiles)}
    for row in schedule:
        lengths = [len(rank) for rank in row]
        assert max(lengths) - min(lengths) <= 1
        assert len({window for rank in row for window, _ in rank}) <= 2


@hardware_test(res={"cuda": ["B200"]}, num_cards=1)
@pytest.mark.parametrize("rows", [1, 137])
def test_vae_fp32_weights_quantized_before_fp16_rounding(rows):
    from vllm_omni.diffusion.models.minimax_h3.quantization import VideoVAEMXFP8Linear
    from vllm_omni.platforms import current_omni_platform

    capability = current_omni_platform.get_device_capability()
    if capability is None or capability.major not in (10, 12):
        pytest.skip("requires Blackwell")
    torch.manual_seed(713)
    linear = torch.nn.Linear(128, 64, device="cuda", dtype=torch.float32)
    # Place values across FP8 rounding boundaries that FP16 conversion can move.
    with torch.no_grad():
        linear.weight[:, :32] = torch.linspace(0.0001, 1.7502, 32, device="cuda")
    quantized = VideoVAEMXFP8Linear(linear, torch.device("cuda"))
    weight = linear.weight.detach().reshape(64, 4, 32)
    exponent = torch.ceil(torch.log2(weight.abs().amax(dim=-1) / 448)).clamp(min=-127)
    expected = (weight / torch.exp2(exponent).unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(64, 128)
    assert torch.equal(quantized.weight.view(torch.uint8), expected.view(torch.uint8))
    x = torch.randn(1, rows, 128, device="cuda", dtype=torch.float32)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = quantized(x)
        reference = linear(x)
    assert actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    relative_rms = (
        actual.float() - reference.float()
    ).square().mean().sqrt() / reference.float().square().mean().sqrt()
    assert relative_rms < 0.06
    with pytest.raises(RuntimeError, match="autocast"):
        quantized(x)


@pytest.mark.cpu
@pytest.mark.parametrize("count", [0, 2, 362])
def test_paired_full_output_owns_uint8_storage(monkeypatch, count):
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    monkeypatch.setenv("VLLM_OMNI_H3_VAE_BATCHING", "paired")
    vae = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    tile = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1, 1)

    def decode_chunks(latent, *, on_chunk):
        for _ in range(count):
            on_chunk(tile)

    monkeypatch.setattr(vae, "decode_with_chunks", decode_chunks)
    if count == 2:
        with pytest.raises(RuntimeError, match="expected 362"):
            vae.decode_latent(torch.empty(0))
        return
    output = vae.decode_latent(torch.empty(0))
    assert output.dtype == torch.uint8
    if count == 0:
        assert output.shape == (1, 3, 0, 0, 0)
    else:
        assert output.shape == (1, 3, 362, 1, 1)
        assert torch.equal(output[:, :, 0, 0, 0], torch.tensor([[0, 128, 255]], dtype=torch.uint8))
        tile.zero_()
        assert output[0, 2].eq(255).all()


@pytest.mark.cpu
@pytest.mark.parametrize("world", [2, 4, 8, 16])
def test_mixed_schedule_preserves_consumer_order_without_batch_three(world):
    from vllm_omni.diffusion.models.minimax_h3.vae_batching import batch_plan, jobs

    schedule = jobs(world=world)
    for rank in range(world):
        batches = batch_plan(rank, world)
        assert all(len(batch) in (1, 2, 4) for batch in batches)
        expected = [job for round_jobs in schedule for job in round_jobs[rank]]
        assert [job for batch in batches for job in batch] == expected
        # Batching must not move a tile across its pair's gather boundary.
        assert all(len({window // 2 for window, tile in batch}) == 1 for batch in batches)
