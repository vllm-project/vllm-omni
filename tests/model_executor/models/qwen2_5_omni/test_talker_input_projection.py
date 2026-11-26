# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Talker's ``thinker_to_talker_proj`` lives inside the compiled ``Qwen2Model``.

It must exist for every Talker config, including one whose ``embedding_size`` equals
``hidden_size`` (a square projection is still learned), so that the checkpoint weight
lands on the mapped path and the BOS/EOS helper always gets a callable.
"""

import socket
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config
from vllm.config import CacheConfig, CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_omni.model_executor.models.qwen2_5_omni import qwen2_5_omni_talker as talker_mod
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_old import Qwen2Model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 8


class _RecordingDecoderLayer(torch.nn.Module):
    def __init__(self, *, config, cache_config, quant_config, prefix):
        super().__init__()
        self.received_hidden_states = []

    def forward(self, positions, hidden_states, residual):
        self.received_hidden_states.append(hidden_states.detach().clone())
        return hidden_states, torch.zeros_like(hidden_states)  # a real layer returns a residual tensor


@pytest.fixture(scope="module", autouse=True)
def _single_rank_vllm_dist():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    initialize_model_parallel(1, 1)
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def _backbone(embedding_size: int) -> Qwen2Model:
    hf_config = Qwen2Config(
        vocab_size=32,
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    hf_config.embedding_size = embedding_size
    vllm_config = VllmConfig(compilation_config=CompilationConfig(mode=0), cache_config=CacheConfig())
    vllm_config.model_config = SimpleNamespace(hf_config=hf_config, dtype=torch.float32, is_multimodal_model=False)
    with set_current_vllm_config(vllm_config):
        return Qwen2Model(vllm_config=vllm_config, prefix="model", decoder_layer_type=_RecordingDecoderLayer)


@pytest.mark.parametrize("embedding_size", [HIDDEN, 2 * HIDDEN], ids=["equal_width", "wider_thinker"])
def test_projection_is_loaded_from_the_checkpoint_path_and_applied(embedding_size):
    model = _backbone(embedding_size)

    proj = model.thinker_to_talker_proj
    assert isinstance(proj, torch.nn.Linear)
    assert (proj.in_features, proj.out_features) == (embedding_size, HIDDEN)

    # The Talker maps `talker.thinker_to_talker_proj.*` into the backbone; the key must resolve to a parameter.
    mapped = talker_mod.Qwen2_5OmniTalkerForConditionalGeneration.hf_to_vllm_mapper._map_name(
        "talker.thinker_to_talker_proj.weight"
    )
    assert mapped is not None and mapped.startswith("language_model.model.")
    params = dict(model.named_parameters())
    assert mapped.removeprefix("language_model.model.") in params

    # A non-identity projection, so a skipped or wrongly routed weight cannot pass unnoticed.
    weight = torch.arange(HIDDEN * embedding_size, dtype=torch.float32).reshape(HIDDEN, embedding_size) / 10
    bias = torch.linspace(-1, 1, HIDDEN)
    params[mapped.removeprefix("language_model.model.")].data.copy_(weight)
    params["thinker_to_talker_proj.bias"].data.copy_(bias)

    x = torch.randn(3, embedding_size)
    first_layer = model.layers[0]
    model(input_ids=torch.zeros(3, dtype=torch.long), positions=torch.arange(3), inputs_embeds=x)
    assert len(first_layer.received_hidden_states) == 1
    torch.testing.assert_close(first_layer.received_hidden_states[0], x @ weight.T + bias, rtol=1e-5, atol=1e-5)
