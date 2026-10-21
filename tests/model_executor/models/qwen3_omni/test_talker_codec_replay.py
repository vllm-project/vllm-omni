# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live small CPU predictor arithmetic, not a checkpoint/device qualification."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig

from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorWrapper, CodePredictorWrapperConfig
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _talker(monkeypatch, dtype, num_groups):
    # Use the real shared transformer/attention/norm/sampler, with small CPU
    # weights and ordinary embeddings instead of distributed vocabulary shards.
    # Only disable compilation/capture: those are not the claim of this test.
    monkeypatch.setattr(current_omni_platform, "supports_torch_inductor", lambda: False)
    config = Qwen3OmniMoeTalkerCodePredictorConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        num_code_groups=num_groups,
    )
    model = object.__new__(Qwen3OmniMoeTalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(code_predictor_config=config)
    model.num_code_groups = num_groups
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.codec_embedding = nn.Embedding(64, 16, dtype=dtype)
    model.code_predictor = CodePredictorWrapper(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(stage_connector_config={})),
        cp_config=config,
        wrapper_config=CodePredictorWrapperConfig(
            use_cuda_graphs=False,
            use_parallel_embedding=False,
            use_projection=False,
            return_proj_buf=True,
            sampling_mode="stored",
        ),
    ).to(dtype=dtype)
    return model


@pytest.mark.parametrize("num_groups", [3, 16])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_replay_matches_live_predictor_without_transformer_or_rng(
    mocker, monkeypatch, model_dtype, input_dtype, batch_size, num_groups
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        model = _talker(monkeypatch, model_dtype, num_groups)
        primary = torch.arange(batch_size).reshape(-1, 1) + 2
        hidden = torch.randn(batch_size, 1, 16, dtype=input_dtype)
        codes, original_sum = model.code_predictor_forward(primary, hidden, last_talker_hidden=hidden)
        state_after_sampling = torch.random.get_rng_state().clone()
        original_codes = codes.clone()
        original_sum = original_sum.clone()
        forbidden = mocker.Mock(side_effect=AssertionError("replay must not sample or run the residual transformer"))
        monkeypatch.setattr(model.code_predictor, "forward", forbidden)
        monkeypatch.setattr(model.code_predictor.model, "forward", forbidden)
        monkeypatch.setattr(model.code_predictor, "_sample_codes_gumbel", forbidden)
        # Non-contiguous views occur when rows come from batched output storage.
        storage = torch.full((batch_size, num_groups * 2), -99, dtype=torch.long)
        storage[:, ::2] = codes.squeeze(-1)
        replayed = model.replay_codec_embeddings(storage[:, ::2], dtype=input_dtype)
        torch.testing.assert_close(replayed, original_sum[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), state_after_sampling, rtol=0, atol=0)
        torch.testing.assert_close(codes, original_codes, rtol=0, atol=0)
        # Returned storage must survive the next input's replay.
        retained = replayed.clone()
        model.replay_codec_embeddings(torch.zeros_like(codes.squeeze(-1)), dtype=input_dtype)
        torch.testing.assert_close(replayed, retained, rtol=0, atol=0)
        forbidden.assert_not_called()


@pytest.mark.parametrize(
    "codes", [torch.zeros(2, 3), torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 3, 1, dtype=torch.long)]
)
def test_replay_rejects_wrong_tensor_contract_before_gather(mocker, monkeypatch, codes):
    model = _talker(monkeypatch, torch.float32, 3)
    forbidden = mocker.Mock(side_effect=AssertionError("invalid replay must fail before gathering"))
    monkeypatch.setattr(model.language_model.model.codec_embedding, "forward", forbidden)
    with pytest.raises(ValueError, match="complete int64"):
        model.replay_codec_embeddings(codes, dtype=torch.float32)
    forbidden.assert_not_called()
