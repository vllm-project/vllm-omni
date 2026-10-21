# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in replay arithmetic with all real checkpoint predictor/codec weights.

VLLM_OMNI_TEST_QWEN3_MODEL must name a verified local snapshot. Only its
complete predictor and primary codec embedding are loaded. Select cpu (default)
or cuda via VLLM_OMNI_TEST_TALKER_DEVICE. No network access, full Talker forward,
KV cache, audio generation, WER, latency or distributed qualification is implied.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_marks
from vllm_omni.core.prefix_cache.adapter import PrefixCacheRequestOwner
from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorWrapper, CodePredictorWrapperConfig
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_talker import Qwen3OmniMoeTalkerForConditionalGeneration
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.talker_history import bind_talker_history, preprocess_talker_history

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cpu,
    pytest.mark.omni,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]


@pytest.fixture(scope="module")
def checkpoint_talker():
    checkpoint = os.environ.get("VLLM_OMNI_TEST_QWEN3_MODEL")
    if not checkpoint:
        pytest.skip("set VLLM_OMNI_TEST_QWEN3_MODEL to qualify checkpoint codec replay")
    assert checkpoint is not None
    snapshot = Path(checkpoint)
    assert (snapshot / "config.json").is_file(), "qualification requires a local checkpoint"
    device = torch.device(os.environ.get("VLLM_OMNI_TEST_TALKER_DEVICE", "cpu"))
    assert device.type in ("cpu", "cuda"), "this qualification supports CPU or CUDA only"
    if device.type == "cuda":
        assert torch.cuda.is_available(), "explicit CUDA qualification requires an available device"
    config = Qwen3OmniMoeConfig.from_pretrained(snapshot, local_files_only=True).talker_config
    weights = {}
    primary = None
    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("talker.code_predictor."):
                    weights[key.removeprefix("talker.code_predictor.")] = handle.get_tensor(key)
                elif key == "talker.model.codec_embedding.weight":
                    primary = handle.get_tensor(key)
    assert primary is not None and weights, "complete codec/predictor checkpoint shards are required"
    model = object.__new__(Qwen3OmniMoeTalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = config
    model.num_code_groups = config.num_code_groups
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    # TP1-equivalent unsharded tables; this does not test distributed loading.
    model.language_model.model.codec_embedding = nn.Embedding.from_pretrained(primary)
    model.code_predictor = CodePredictorWrapper(
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(stage_connector_config={})),
        cp_config=config.code_predictor_config,
        wrapper_config=CodePredictorWrapperConfig(
            use_cuda_graphs=False,
            use_parallel_embedding=False,
            use_projection=False,
            return_proj_buf=True,
            sampling_mode="stored",
        ),
    ).to(dtype=primary.dtype)
    loaded = model.code_predictor.load_weights(weights.items())
    assert loaded == set(dict(model.code_predictor.named_parameters())), "no randomly initialized predictor parameters"
    del weights
    model = model.to(device=device).eval()
    # Exercise the real eager transformer and sampling. Compiler/graph parity
    # has its own gate; don't spend qualification time compiling random warmups.
    with pytest.MonkeyPatch.context() as context:
        context.setattr(current_omni_platform, "supports_torch_inductor", lambda: False)
        model.code_predictor._setup_compile()
    return model


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_checkpoint_replay_matches_predictor_buffer_and_preserves_rng(
    checkpoint_talker, batch_size, input_dtype, mocker
):
    model = checkpoint_talker
    device = next(model.parameters()).device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(901)
        primary = (torch.arange(batch_size, device=device) + 400).reshape(-1, 1)
        # Controlled hidden inputs, not states from a full Talker inference.
        hidden = torch.randn(batch_size, 1, model.config.text_config.hidden_size, device=device, dtype=input_dtype)
        codes, summed = model.code_predictor_forward(primary, hidden, last_talker_hidden=hidden)
        rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
        original_codes, original_sum = codes.clone(), summed.clone()
        forbidden = mocker.Mock(side_effect=AssertionError("checkpoint replay must not sample"))
        with pytest.MonkeyPatch.context() as context:
            context.setattr(model.code_predictor, "forward", forbidden)
            context.setattr(model.code_predictor, "_sample_codes_gumbel", forbidden)
            replayed = model.replay_codec_embeddings(codes.squeeze(-1), dtype=input_dtype)
        torch.testing.assert_close(replayed, original_sum[:, 0], rtol=0, atol=0)
        current_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
        torch.testing.assert_close(current_rng, rng_state, rtol=0, atol=0)
        torch.testing.assert_close(codes, original_codes, rtol=0, atol=0)
        assert torch.isfinite(replayed).all()
        assert codes.shape == (batch_size, model.config.num_code_groups, 1)
        forbidden.assert_not_called()


@pytest.mark.parametrize("batch_size", [1, 3])
@torch.inference_mode()
def test_checkpoint_runtime_producer_preserves_samples_and_owned_snapshot(
    checkpoint_talker, batch_size, monkeypatch, mocker
):
    talker = checkpoint_talker
    device = next(talker.parameters()).device
    model = SimpleNamespace(talker=talker, talker_config=talker.config)
    model.talker_replay_inputs = mocker.Mock(
        side_effect=lambda input_ids, input_embeds, **_: (input_ids, input_embeds, {})
    )
    runner = object.__new__(GPUARModelRunner)
    runner.model = model
    runner.talker_mtp_input_ids = SimpleNamespace(gpu=torch.empty(batch_size, device=device, dtype=torch.long))
    runner.talker_mtp_inputs_embeds = SimpleNamespace(
        gpu=torch.empty(batch_size, talker.config.text_config.hidden_size, device=device, dtype=torch.bfloat16)
    )
    runner.last_talker_hidden = SimpleNamespace(
        gpu=torch.empty(batch_size, talker.config.text_config.hidden_size, device=device, dtype=torch.bfloat16)
    )
    runner.text_step = SimpleNamespace(
        gpu=torch.empty(batch_size, talker.config.text_config.hidden_size, device=device, dtype=torch.bfloat16)
    )
    runner._predict_talker_codes = GPUARModelRunner._predict_talker_codes.__get__(runner, GPUARModelRunner)

    invoke_rows = []
    poison_value = 13

    def invoke_talker_mtp(rows):
        invoke_rows.append(rows)
        primary = runner.talker_mtp_input_ids.gpu[:rows].reshape(rows, 1)
        req_embeds = runner.talker_mtp_inputs_embeds.gpu[:rows]
        hidden = runner.last_talker_hidden.gpu[:rows].reshape(rows, 1, -1)
        torch.testing.assert_close(
            req_embeds,
            torch.full_like(req_embeds, poison_value),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            runner.text_step.gpu[:rows],
            torch.zeros_like(runner.text_step.gpu[:rows]),
            rtol=0,
            atol=0,
        )
        codes, summed = talker.code_predictor_forward(primary, req_embeds, last_talker_hidden=hidden)
        return summed[:, 0] + runner.text_step.gpu[:rows], codes.squeeze(-1)

    runner._invoke_talker_mtp = invoke_talker_mtp
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(808)
        primary = (torch.arange(batch_size, device=device) + 400).reshape(batch_size, 1)
        hidden = torch.randn(batch_size, talker.config.text_config.hidden_size, device=device, dtype=torch.bfloat16)
        rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
        zero_input_embeds = torch.zeros_like(hidden).reshape(batch_size, 1, -1)
        runner.talker_mtp_inputs_embeds.gpu.fill_(poison_value)
        runner.text_step.gpu.fill_(poison_value)
        states = {}
        items = []
        for row in range(batch_size):
            state = SimpleNamespace(
                prompt_token_ids=[100, 101, 102],
                output_token_ids=[400 + row],
                sampling_params=SamplingParams(max_tokens=8),
            )
            bind_talker_history(state, PrefixCacheRequestOwner(row + 1))
            req_id = str(row)
            states[req_id] = state
            items.append(
                (
                    req_id,
                    0,
                    torch.tensor([100, 101], device=device),
                    torch.zeros(2, talker.config.text_config.hidden_size, device=device, dtype=torch.bfloat16),
                    {"hidden_states": {"last": hidden[row]}, "meta": {}},
                )
            )
        runner.requests = states
        with monkeypatch.context() as context:
            context.setattr(
                torch.cuda, "synchronize", mocker.Mock(side_effect=AssertionError("no device-wide barrier"))
            )
            OmniGPUModelRunner._materialize_pending_talker_history(runner, items)
        assert sum(invoke_rows) == batch_size
        # Scalar and batched producers consume global RNG in different orders.
        # Compare each actual call to native prediction with the same row group.
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng, device)
        else:
            torch.random.set_rng_state(rng)
        expected_code_rows, expected_embedding_rows = [], []
        offset = 0
        for rows in invoke_rows:
            end = offset + rows
            codes, summed = talker.code_predictor_forward(
                primary[offset:end],
                zero_input_embeds[offset:end],
                last_talker_hidden=hidden[offset:end].reshape(rows, 1, -1),
            )
            expected_code_rows.append(codes.clone())
            expected_embedding_rows.append(summed.clone())
            offset = end
        expected_codes = torch.cat(expected_code_rows)
        expected_sum = torch.cat(expected_embedding_rows)
        frozen = {}
        for row, req_id in enumerate(states):
            state = states[req_id]
            actual = state.talker_codec_inputs[0].reshape(1, -1)
            torch.testing.assert_close(actual, expected_codes[row].squeeze(-1).reshape(1, -1), rtol=0, atol=0)
            torch.testing.assert_close(
                state.talker_next_input_embedding, expected_sum[row, 0].reshape(1, -1), rtol=0, atol=0
            )
            preprocess_talker_history(
                model,
                state,
                row_start=0,
                input_ids=torch.tensor([100, 101], device=device),
                input_embeds=torch.zeros(2, talker.config.text_config.hidden_size, device=device, dtype=torch.bfloat16),
                payload=items[row][4],
            )
            replayed = talker.replay_codec_embeddings(actual, dtype=torch.bfloat16)
            torch.testing.assert_close(replayed, expected_sum[row, 0].reshape(1, -1), rtol=0, atol=0)
            frozen[req_id] = tuple(record.clone() for record in state.talker_codec_inputs)
        runner.talker_mtp_input_ids.gpu.fill_(0)
        runner.talker_mtp_inputs_embeds.gpu.fill_(0)
        runner.last_talker_hidden.gpu.fill_(0)
        runner.text_step.gpu.fill_(0)
        talker.code_predictor_forward(
            primary, hidden.reshape(batch_size, 1, -1), last_talker_hidden=hidden.reshape(batch_size, 1, -1)
        )
        for req_id, state in states.items():
            torch.testing.assert_close(state.talker_codec_inputs[0], frozen[req_id][0], rtol=0, atol=0)
