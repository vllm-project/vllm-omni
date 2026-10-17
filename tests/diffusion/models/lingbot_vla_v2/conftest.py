# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline tests on a real tiny LingBot inference graph and local processors."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor
from transformers.models.qwen3_vl import Qwen3VLConfig

from vllm_omni.diffusion.models.lingbot_vla_v2.config import LingBotVLAV2Config
from vllm_omni.diffusion.models.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction


def tiny_config(**overrides):
    def slots(key):
        return {
            f"{key}.{name}": {"origin_keys": [{key: {"start": s, "end": e}}]}
            for name, s, e in [("arm.position", 0, 6), ("effector.position", 6, 7)]
        }

    head = {"num_layers": 1, "num_heads": 2, "dim_head": 8, "ff_mult": 1, "num_backbone_tokens": 8, "dim_out": 16}
    cfg = dict(
        chunk_size=4,
        n_action_steps=4,
        num_steps=2,
        vocab_size=128,
        expert_hidden_size=64,
        expert_intermediate_size=128,
        action_num_attention_heads=4,
        action_num_key_value_heads=2,
        action_head_dim=16,
        token_moe_layers=[0, 1],
        token_num_experts=4,
        token_top_k=2,
        token_moe_intermediate_size=32,
        token_shared_intermediate_size=32,
        tokenizer_max_length=16,
        resize_imgs_with_padding=(64, 64),
        image_min_pixels=4096,
        image_max_pixels=4096,
        input_features={
            "observation.state": {"type": "STATE", "shape": [7]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 32, 32]},
        },
        output_features={"action": {"type": "ACTION", "shape": [7]}},
        state_slots=slots("observation.state"),
        action_slots=slots("action"),
        normalization_mapping={"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        align_params={
            "mode": "query",
            "num_task_tokens": 4,
            "use_future_video": True,
            "llm": {"dim_out": 64},
            "depth": {**head, "use_future_depth": True},
            "video": {
                **head,
                "share_future_depth_query": True,
                "use_shared_future_task_proj": True,
                "use_current_patch_loss": True,
            },
        },
    )
    cfg.update(overrides)
    return LingBotVLAV2Config(**cfg)


@pytest.fixture
def assets(tmp_path):
    words = ["<unk>", "<pad>", "<s>", "</s>", "user", "pick", "up", "the", "cube"]
    raw = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>")
    tok.chat_template = "{% for m in messages %}<s> {{ m.role }} {{ m.content }} </s>{% endfor %}"
    for dirname in ("tokenizer", "chat_template_tokenizer"):
        tok.save_pretrained(tmp_path / dirname)
    Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2).save_pretrained(
        tmp_path / "image_processor"
    )
    stats = {}
    for key in ("observation.state", "action"):
        stats[key + ".q01"] = torch.linspace(-2, -1, 7)
        stats[key + ".q99"] = torch.linspace(1, 3, 7)
    save_file(stats, tmp_path / "stats.safetensors")
    for filename, name in (
        ("policy_preprocessor.json", "normalizer_processor"),
        ("policy_postprocessor.json", "unnormalizer_processor"),
    ):
        (tmp_path / filename).write_text(
            json.dumps({"steps": [{"registry_name": name, "config": {"eps": 1e-8}, "state_file": "stats.safetensors"}]})
        )
    return tmp_path


def tiny_backbone():
    cfg = Qwen3VLConfig()
    t, v = cfg.text_config, cfg.vision_config
    t.num_hidden_layers, t.hidden_size, t.intermediate_size = 2, 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.rope_parameters = {**t.rope_parameters, "mrope_section": [2, 3, 3]}
    t.vocab_size, t.eos_token_id = 128, 3
    cfg.image_token_id, cfg.video_token_id = 100, 101
    cfg.vision_start_token_id, cfg.vision_end_token_id = 102, 103
    v.hidden_size, v.intermediate_size, v.num_heads, v.depth, v.out_hidden_size = 64, 128, 2, 2, 64
    v.deepstack_visual_indexes = [0, 1]
    return cfg


@pytest.fixture
def model():
    torch.manual_seed(19)
    net = LingBotVLAV2ForActionPrediction(tiny_config(), vlm_config=tiny_backbone()).float().eval()
    with torch.no_grad():
        for p in net.parameters():
            p.uniform_(-0.04, 0.04)
    return net


def observation():
    return {
        "state": np.linspace(-0.5, 0.5, 7, dtype=np.float32),
        "prompt": "pick up the cube",
        "observation.images.top": np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3),
    }
