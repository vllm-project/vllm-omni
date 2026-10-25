# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import build_reference_config

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("sp_size", [1, 2])
def test_configuration_matches_reference_hydra_composition(tmp_path, sp_size):
    reference = pytest.importorskip("configs.config", reason="Install the pinned Hy-Nomas reference checkout")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(Path(reference.__file__).parent), version_base="1.3"):
        config = compose(config_name="config", overrides=["+experiment=inference"])
    config.dit_weight = str(tmp_path / "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors")
    config.model.vae.checkpoint = str(tmp_path / "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors")
    config.model.vae.parallel_mode = "none"
    config.model.text_encoder.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    config.model.dit.seg_ar_attention_backend = "flex"
    config.parallelism.sp_size = sp_size
    config.parallelism.shard_text_encoder = False
    config.inference.save_latents = False
    config.inference.save_prompt_txt = False
    config.inference.video_size = [704, 1280]
    expected = reference._flatten_cfg(config)
    actual = build_reference_config(tmp_path, "Qwen/Qwen2.5-VL-7B-Instruct", sp_size)
    assert actual.vae.parallel_mode == "none"
    assert actual.parallelism.sp_size == sp_size
    assert actual.parallelism.shard_text_encoder is False
    assert OmegaConf.to_container(actual, resolve=True) == OmegaConf.to_container(expected, resolve=True)
