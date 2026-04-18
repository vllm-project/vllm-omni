# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Source parity checks for the RoboArena → DreamZero input path.

This test targets the non-model part of the OpenPI DreamZero chain:

- `socket_test_optimized_AR.py:ARDroidRoboarenaPolicy._convert_observation()`
- upstream `eval_transform.apply()`
- local `RoboArenaTransform.transform_input()`
- local prompt tokenization + state normalization path used by
  `DreamZeroPipeline.forward()`

The goal is to make sure the local serving pre-processing feeds the same
stitched video, prompt tokens, and normalized state into the model as the
upstream DreamZero source server.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import DreamZeroPipeline
from vllm_omni.diffusion.models.dreamzero.transform.roboarena import (
    RoboArenaTransform,
)

instantiate = pytest.importorskip("hydra.utils").instantiate
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

DREAMZERO_REPO = Path(os.environ.get("DREAMZERO_REPO", "~/code/dreamzero")).expanduser()
CHECKPOINT_DIR = DREAMZERO_REPO / "checkpoints" / "dreamzero"
PROMPT = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
SESSION_ID = "roboarena-transform-source-parity"

pytestmark = [
    pytest.mark.skipif(not DREAMZERO_REPO.exists(), reason="DreamZero source repo is required at ~/code/dreamzero"),
    pytest.mark.skipif(not CHECKPOINT_DIR.exists(), reason="DreamZero local checkpoint is required"),
]


def _load_source_normalized_input():
    if str(DREAMZERO_REPO) not in sys.path:
        sys.path.insert(0, str(DREAMZERO_REPO))

    import test_client_AR as dreamzero_client
    from groot.vla.data.schema import DatasetMetadata, EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import unsqueeze_dict_values
    from socket_test_optimized_AR import ARDroidRoboarenaPolicy

    class DummyPolicy:
        pass

    camera_frames = dreamzero_client.load_camera_frames()
    obs0 = dreamzero_client._make_obs_from_video(camera_frames, [0], PROMPT, SESSION_ID)

    adapter = ARDroidRoboarenaPolicy(groot_policy=DummyPolicy(), signal_group=None)
    converted = unsqueeze_dict_values(adapter._convert_observation(dict(obs0)))

    train_cfg = OmegaConf.load(CHECKPOINT_DIR / "experiment_cfg" / "conf.yaml")
    with open(CHECKPOINT_DIR / "experiment_cfg" / "metadata.json") as f:
        metadatas = json.load(f)

    metadata = DatasetMetadata.model_validate(metadatas[EmbodimentTag.OXE_DROID.value])
    eval_transform = instantiate(train_cfg.transforms[EmbodimentTag.OXE_DROID.value])
    eval_transform.set_metadata(metadata)
    eval_transform.eval()
    normalized = eval_transform.apply(dict(converted))

    return obs0, metadatas, normalized


def test_roboarena_transform_matches_source_video_prompt_and_state():
    obs0, metadatas, source_normalized = _load_source_normalized_input()

    local_transform = RoboArenaTransform()
    local_unified = local_transform.transform_input(dict(obs0))

    source_images = source_normalized["images"].cpu().numpy()
    if source_images.ndim == 5 and source_images.shape[0] == 1:
        source_images = source_images[0]
    assert np.array_equal(local_unified["images"], source_images)

    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    local_text = tokenizer(
        local_unified["prompt"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    assert torch.equal(local_text["input_ids"], source_normalized["text"].cpu())
    assert torch.equal(
        local_text["attention_mask"],
        source_normalized["text_attention_mask"].cpu(),
    )

    pipe = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipe.state_norm_stats = DreamZeroPipeline._parse_state_norm_stats(metadatas)

    raw_state = np.asarray(local_unified["state"], dtype=np.float64)
    padded = np.zeros(64, dtype=np.float64)
    padded[: len(raw_state)] = raw_state
    local_state = torch.from_numpy(padded).reshape(1, 1, 64).to(dtype=torch.float32)
    local_state = DreamZeroPipeline._normalize_state(pipe, local_state, "oxe_droid")

    torch.testing.assert_close(
        local_state,
        source_normalized["state"].float(),
        atol=1e-7,
        rtol=0.0,
    )
