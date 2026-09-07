# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The two-stage Thinker runs SigLIP as NaViT (Bagel/modeling/bagel/siglip_navit.py): each image is
one sequence, position ids index the 2-D table by (row, col) without stretching to a square, and
post_layernorm is applied. Compare against the HF modules with the same weights."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import SiglipVisionConfig
from transformers import SiglipVisionModel as HFSiglipVisionModel
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.siglip import SiglipVisionModel

from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's SigLIP needs a GPU attention backend")
SIDE = 4  # 4x4 position table -> images up to 56x56 at patch 14


@pytest.fixture(scope="module", autouse=True)
def _single_rank_env():
    for k, v in {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29511",
    }.items():
        os.environ.setdefault(k, v)
    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
    if not model_parallel_is_initialized():
        initialize_model_parallel(1, 1, 1, 1, 1, 1, 1)
    yield
    destroy_distributed_env()


def navit_reference(hf, img):
    core = getattr(hf, "vision_model", hf)  # transformers 5 flattens SiglipVisionModel
    ids = OmniBagelForConditionalGeneration.get_flattened_position_ids(None, *img.shape[-2:], 14, SIDE).to(img.device)
    x = core.embeddings.patch_embedding(img[None]).flatten(2).transpose(1, 2) + core.embeddings.position_embedding(ids)
    return core.post_layernorm(core.encoder(inputs_embeds=x).last_hidden_state)[0]


def test_two_stage_vit_path_matches_hf_navit():
    torch.manual_seed(0)
    cfg = SiglipVisionConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, image_size=56, patch_size=14
    )
    cfg.vision_use_head = False
    hf = HFSiglipVisionModel(cfg).cuda().eval()
    with set_current_vllm_config(VllmConfig()):
        vit = SiglipVisionModel(cfg).cuda()
    vit.vision_model.load_weights((k.removeprefix("vision_model."), v) for k, v in hf.state_dict().items())

    model = OmniBagelForConditionalGeneration.__new__(OmniBagelForConditionalGeneration)
    nn.Module.__init__(model)
    model.vit_model, model.connector, model.vit_pos_embed = (
        vit,
        nn.Identity(),
        lambda ids: torch.zeros((), device=ids.device),
    )
    model.config = SimpleNamespace(vit_config=cfg, vit_max_num_patch_per_side=SIDE)

    square, wide = torch.randn(3, 56, 56, device="cuda"), torch.randn(3, 28, 42, device="cuda")
    with torch.no_grad():
        assert torch.allclose(
            navit_reference(hf, square), hf(pixel_values=square[None]).last_hidden_state[0], atol=1e-5
        )
        for img in (square, wide):
            torch.testing.assert_close(model._vit_embeddings([img])[0], navit_reference(hf, img), atol=1e-4, rtol=1e-4)
