# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.ace_step.ace_step_transformer import (
    AceStepTransformer1DModel,
)
from vllm_omni.diffusion.models.ace_step.modeling_ace_step import (
    AceStepConditionEncoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def _init_single_process_tp():
    """Set up a single-process tensor-model-parallel group so the forward-shape
    tests can instantiate modules that use ``ReplicatedLinear`` (which calls
    ``get_tensor_model_parallel_rank`` in its ``__init__``).
    """
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29503")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1)
    yield


# --------------------------------------------------------------------------- #
#                         parallelism / acceleration contract                  #
# --------------------------------------------------------------------------- #


def test_ace_step_exposes_hsdp_shard_conditions_for_transformer_blocks():
    """``_hsdp_shard_conditions`` must select every block in ``self.layers`` so
    FSDP2 shards weights per-block. Mirrors the stable_audio HSDP contract.
    """
    model = object.__new__(AceStepTransformer1DModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    model.proj_out_conv = nn.Linear(4, 4)

    conditions = getattr(model, "_hsdp_shard_conditions", None)
    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["layers.0", "layers.1", "layers.2"]


def test_ace_step_repeated_blocks_lists_block_class_name():
    """``_repeated_blocks`` is used by torch.compile to compile a single block
    type once and reuse the artifact across the stack.
    """
    assert AceStepTransformer1DModel._repeated_blocks == ["AceStepTransformerBlock"]


def test_ace_step_layerwise_offload_attr_points_at_block_list():
    """``_layerwise_offload_blocks_attrs`` tells the CPU-offload backend the
    attribute name of the ``nn.ModuleList`` to offload per layer.
    """
    assert AceStepTransformer1DModel._layerwise_offload_blocks_attrs == ["layers"]


# --------------------------------------------------------------------------- #
#         default __init__ kwargs match the official v15-turbo config         #
# --------------------------------------------------------------------------- #
# Source of truth:
#   https://huggingface.co/ACE-Step/Ace-Step1.5/blob/main/acestep-v15-turbo/config.json
# If these drift, the model will silently load with wrong dimensions and
# weight loading will fail with cryptic shape mismatches at runtime.


_OFFICIAL_TURBO_DIT_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "in_channels": 192,
    "audio_acoustic_hidden_dim": 64,
    "patch_size": 2,
    "rope_theta": 1000000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
}

_OFFICIAL_TURBO_COND_ENCODER_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "text_hidden_dim": 1024,
    "timbre_hidden_dim": 64,
    "num_lyric_encoder_hidden_layers": 8,
    "num_timbre_encoder_hidden_layers": 4,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rope_theta": 1000000.0,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
}


@pytest.mark.parametrize("key,expected", list(_OFFICIAL_TURBO_DIT_CONFIG.items()))
def test_transformer_default_matches_official_turbo_config(key, expected):
    sig = inspect.signature(AceStepTransformer1DModel.__init__)
    assert sig.parameters[key].default == expected, (
        f"AceStepTransformer1DModel default for {key!r} = "
        f"{sig.parameters[key].default!r} but ACE-Step 1.5 turbo expects {expected!r}"
    )


@pytest.mark.parametrize("key,expected", list(_OFFICIAL_TURBO_COND_ENCODER_CONFIG.items()))
def test_condition_encoder_default_matches_official_turbo_config(key, expected):
    sig = inspect.signature(AceStepConditionEncoder.__init__)
    assert sig.parameters[key].default == expected, (
        f"AceStepConditionEncoder default for {key!r} = "
        f"{sig.parameters[key].default!r} but ACE-Step 1.5 turbo expects {expected!r}"
    )


# --------------------------------------------------------------------------- #
#                         forward shape checks (tiny config)                   #
# --------------------------------------------------------------------------- #
# These instantiate a SMALL model so the test runs in seconds on CPU. The
# proportions match the real model (GQA factor 2, patch_size 2, channel split
# context+acoustic) so any reshape / projection bug would still surface.


_TINY_DIT_KWARGS = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    in_channels=24,
    audio_acoustic_hidden_dim=8,
    patch_size=2,
    sliding_window=4,
)

_TINY_COND_ENCODER_KWARGS = dict(
    hidden_size=64,
    intermediate_size=64,
    text_hidden_dim=32,
    timbre_hidden_dim=16,
    num_lyric_encoder_hidden_layers=2,
    num_timbre_encoder_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    sliding_window=4,
)


def test_transformer_forward_returns_expected_shape(_init_single_process_tp):
    """End-to-end shape check: `hidden_states` enters as
    [B, T, audio_acoustic_hidden_dim], goes through channel-concat + Conv1d
    patchify (stride=patch_size) -> stack of blocks -> ConvTranspose1d
    de-patchify -> slice back to original seq_len, and must exit at
    [B, T, audio_acoustic_hidden_dim].
    """
    torch.manual_seed(0)
    model = AceStepTransformer1DModel(**_TINY_DIT_KWARGS).eval()

    batch_size, seq_len = 1, 8
    audio_dim = _TINY_DIT_KWARGS["audio_acoustic_hidden_dim"]
    context_dim = _TINY_DIT_KWARGS["in_channels"] - audio_dim
    encoder_seq_len = 6
    encoder_hidden_size = _TINY_DIT_KWARGS["hidden_size"]

    hidden_states = torch.randn(batch_size, seq_len, audio_dim)
    timestep = torch.tensor([0.5])
    timestep_r = torch.tensor([0.5])
    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, encoder_hidden_size)
    context_latents = torch.randn(batch_size, seq_len, context_dim)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            encoder_hidden_states=encoder_hidden_states,
            context_latents=context_latents,
            return_dict=False,
        )

    assert isinstance(output, tuple)
    sample = output[0]
    assert sample.shape == (batch_size, seq_len, audio_dim), (
        f"DiT forward returned shape {tuple(sample.shape)}; expected ({batch_size}, {seq_len}, {audio_dim})"
    )


def test_condition_encoder_forward_returns_expected_shape(_init_single_process_tp):
    """Packed-sequence shape check: lyric (L) + timbre (1 token after CLS pool)
    + text (T) get concatenated and stable-sorted, so the encoder output must
    be [B, L + 1 + T, hidden_size] with a matching mask.
    """
    torch.manual_seed(0)
    encoder = AceStepConditionEncoder(**_TINY_COND_ENCODER_KWARGS).eval()

    batch_size = 1
    text_seq_len = 4
    lyric_seq_len = 8
    timbre_k = 4  # per-sample timbre token count before CLS pool
    text_hidden_dim = _TINY_COND_ENCODER_KWARGS["text_hidden_dim"]
    timbre_hidden_dim = _TINY_COND_ENCODER_KWARGS["timbre_hidden_dim"]
    hidden_size = _TINY_COND_ENCODER_KWARGS["hidden_size"]

    text_hidden_states = torch.randn(batch_size, text_seq_len, text_hidden_dim)
    text_attention_mask = torch.ones(batch_size, text_seq_len, dtype=torch.long)
    lyric_hidden_states = torch.randn(batch_size, lyric_seq_len, text_hidden_dim)
    lyric_attention_mask = torch.ones(batch_size, lyric_seq_len, dtype=torch.long)
    refer_audio_acoustic = torch.randn(batch_size, timbre_k, timbre_hidden_dim)
    # One timbre sample, assigned to batch 0.
    refer_audio_order_mask = torch.zeros(batch_size, dtype=torch.long)

    with torch.no_grad():
        encoder_hidden_states, encoder_attention_mask = encoder(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic,
            refer_audio_order_mask=refer_audio_order_mask,
        )

    # After packing: lyric (L) + timbre (1 CLS token) + text (T) = L + 1 + T.
    expected_seq_len = lyric_seq_len + 1 + text_seq_len
    assert encoder_hidden_states.shape == (batch_size, expected_seq_len, hidden_size), (
        f"Condition encoder returned shape {tuple(encoder_hidden_states.shape)}; "
        f"expected ({batch_size}, {expected_seq_len}, {hidden_size})"
    )
    assert encoder_attention_mask.shape == (batch_size, expected_seq_len), (
        f"Condition encoder mask shape {tuple(encoder_attention_mask.shape)}; "
        f"expected ({batch_size}, {expected_seq_len})"
    )
