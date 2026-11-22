# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference prompt formatting, optional real-tokenizer IDs, and CPU SDE trajectories."""

import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.fixed_step_sde import fixed_step_sample, initial_noise
from vllm_omni.diffusion.models.cosmos3.multiview_prompts import (
    COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
    COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    format_rig_view_captions,
)
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
FIXTURES = json.loads((Path(__file__).parent / "fixtures/reference_parity.json").read_text())


def _format(fixture, tokenizer):
    cls = Cosmos3OmniDiffusersPipeline
    host = SimpleNamespace(tokenizer=tokenizer, device="cpu", _get_sp_param=lambda sp, key, default=None: default)
    for name in ("_format_and_tokenize_prompts", "_tokenize_prompt"):
        setattr(host, name, MethodType(getattr(cls, name), host))
    host._apply_metadata_templates = cls._apply_metadata_templates
    host._normalize_token_ids = cls._normalize_token_ids
    system = (
        COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT
        if fixture["joint"]
        else COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT
    )
    assert system == fixture["system_prompt"]
    return [
        host._format_and_tokenize_prompts(
            text,
            "",
            fixture["num_frames"],
            fixture["fps"],
            fixture["height"],
            fixture["width"],
            4096,
            None,
            use_system_prompt=True,
            system_prompt=system,
            use_duration_template=True,
            use_resolution_template=True,
            negative_metadata_mode="none",
            aspect_ratio_override="16,9",
            truncate_duration=True,
        )[0]
        .flatten()
        .tolist()
        for text in format_rig_view_captions(fixture["captions"], fixture["cameras"])
    ]


@pytest.mark.parametrize("fixture", FIXTURES["prompts"])
def test_reference_prompt_messages(fixture) -> None:
    messages = []

    def record(conversation, **kwargs):
        messages.append(conversation)
        return [1, 2]

    tokenizer = SimpleNamespace(apply_chat_template=record, eos_token_id=151645, convert_tokens_to_ids=lambda _: 151652)
    _format(fixture, tokenizer)
    # Each positive prompt is followed by an unused negative prompt.
    assert messages[::2] == [
        [{"role": "system", "content": fixture["system_prompt"]}, {"role": "user", "content": text}]
        for text in fixture["formatted"]
    ]


def test_reference_token_ids_with_local_tokenizer() -> None:
    tokenizer_path = os.environ.get("COSMOS3_SIM_AUTO_TOKENIZER")
    if tokenizer_path is None:
        pytest.skip("Set COSMOS3_SIM_AUTO_TOKENIZER to the matching exported text_tokenizer directory")
    from transformers import AutoTokenizer

    path = Path(tokenizer_path)
    for name in (
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ):
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == FIXTURES["tokenizer_files_sha256"][name]
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, use_fast=FIXTURES["tokenizer_class"].endswith("Fast")
    )
    for fixture in FIXTURES["prompts"]:
        actual = _format(fixture, tokenizer)
        assert actual == fixture["token_ids"]
        assert [ids[:4] for ids in actual] == fixture["sink_ids"]
        assert max(map(len, actual)) == fixture["text_extent"]


@pytest.mark.parametrize("fixture", FIXTURES["noise"])
def test_reference_noise_and_sde_trajectory(fixture) -> None:
    noise = initial_noise(fixture["shape"], seed=fixture["seed"], sample_index=fixture["sample_index"], device="cpu")
    torch.testing.assert_close(noise, torch.tensor(fixture["noise"]), rtol=0, atol=0)
    step = 0

    def velocity(x, timestep):
        nonlocal step
        torch.testing.assert_close(x, torch.tensor(fixture["states"][step]), rtol=0, atol=0)
        assert timestep.item() == fixture["timesteps"][step]
        step += 1
        return 0.125 * x + timestep * 0.0001

    actual = fixed_step_sample(
        velocity,
        noise,
        t_list=fixture["t_list"],
        num_train_timesteps=1000,
        seed=fixture["seed"],
        sample_index=fixture["sample_index"],
    )
    torch.testing.assert_close(actual, torch.tensor(fixture["result"]), rtol=0, atol=0)
