# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full RoboTwin checkpoint parity against pinned LeRobot source.

Install LeRobot PR #4897 at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d
without replacing the target runtime. Set LINGBOT_PARITY_MODEL_PATH to its
RoboTwin snapshot and LINGBOT_BACKBONE_PATH to the local Qwen config directory.
Run in a fresh process with CUBLAS_WORKSPACE_CONFIG=:4096:8. Both implementations
use the same torch/transformers runtime; six cases compare preprocessing,
canonical actions and calibrated robot actions. This is not a task-success test.
The pinned reference calls Qwen3-VL vision methods that Transformers removed in
5.17, so run this file with transformers==5.16.1.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H100"})]


@pytest.fixture(scope="module")
def policies():
    model_path = os.environ.get("LINGBOT_PARITY_MODEL_PATH")
    if model_path is None:
        pytest.skip("Set LINGBOT_PARITY_MODEL_PATH to a local full checkpoint")
        return
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
        return
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        pytest.skip("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting this process")
    pytest.importorskip("lerobot.policies.lingbot_vla_v2")
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config
    from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import LingBotVLAV2Pipeline

    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        cfg = LingbotVLAV2Config.from_pretrained(model_path)
        cfg.tokenizer_path = os.environ.get("LINGBOT_BACKBONE_PATH", cfg.tokenizer_path)
        cfg.device, cfg.dtype = "cuda", torch.float32
        reference = LingbotVLAV2Policy(cfg)
        reference.load_state_dict(load_file(Path(model_path) / "model.safetensors"), strict=True)
        reference = reference.float().cuda().eval()
        pre, post = make_pre_post_processors(
            cfg, pretrained_path=model_path, preprocessor_overrides={"device_processor": {"device": "cuda"}}
        )
        runtime = OmniDiffusionConfig(
            model=model_path,
            model_class_name="LingBotVLAV2Pipeline",
            dtype=torch.float32,
            model_config={"tokenizer_path": cfg.tokenizer_path},
            num_gpus=1,
        )
        native = LingBotVLAV2Pipeline(od_config=runtime)
        yield reference, pre, post, native
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


@pytest.mark.parametrize("views", [1, 2, 3])
@pytest.mark.parametrize("steps", [2, 10])
@torch.no_grad()
def test_full_checkpoint_parity(policies, views, steps):
    reference, pre, post, native = policies
    keys = native.config.camera_keys
    chosen = [0] if views == 1 else [0, 2] if views == 2 else [0, 1, 2]
    state = np.linspace(-0.1, 0.1, native.config.state_dim, dtype=np.float32)
    obs = {"state": state, "prompt": "pick up the red block and place it in the bin"}
    rng = np.random.default_rng(12)
    obs.update({keys[i]: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for i in chosen})
    batch = {"observation.state": torch.from_numpy(state).unsqueeze(0), "task": [obs["prompt"]]}
    for i in chosen:
        batch[keys[i]] = torch.from_numpy(obs[keys[i]]).permute(2, 0, 1).float().unsqueeze(0) / 255
    processed = pre(batch)
    names = ("images", "img_masks", "lang_tokens", "lang_masks", "state", "image_grid_thw")
    expected_inputs = dict(zip(names, reference._extract_model_inputs(processed), strict=True))
    actual_inputs = native.processor.build_model_inputs(obs)
    for name in names:
        torch.testing.assert_close(actual_inputs[name], expected_inputs[name], atol=0, rtol=0)
    shape = (1, native.config.chunk_size, native.config.max_action_dim)
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(120 + views)).cuda()
    reference.config.num_steps = steps
    expected = reference.model.sample_actions(**expected_inputs, noise=noise.clone())
    actual = native.model.sample_actions(**actual_inputs, noise=noise.clone(), num_steps=steps)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(
        torch.from_numpy(native.processor.build_model_outputs(actual)), post(expected)[0].cpu(), atol=1e-4, rtol=1e-5
    )
