# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""UMT5 differs from T5 by giving every block its own relative attention bias.

``T5Stack`` has to do two things together for that to be correct: build a
``relative_attention_bias`` table on every block, AND stop threading block 0's
``position_bias`` through the rest. Doing only the first leaves 23 of DreamZero's
24 tables dead; doing only the second feeds zeros into every block after the
first. Neither mistake raises — the output is just silently wrong — so both
halves are pinned here, along with the T5 default staying untouched for the
pipelines that already ship on it (flux, hunyuan_video_1_5, ming_flash_omni).
"""

import pytest
import torch
from torch import nn
from transformers import T5Config
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.models.t5_encoder.t5_encoder import (
    T5EncoderModel,
    T5Stack,
    UMT5EncoderModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_T5_MODULE = "vllm_omni.diffusion.models.t5_encoder.t5_encoder"

# Same shape as test_t5_encoder_tp.py's config, with DreamZero's gated-gelu.
SMALL_UMT5_CONFIG = dict(
    d_model=64,
    d_kv=8,
    d_ff=128,
    num_heads=8,
    num_layers=4,
    vocab_size=256,
    relative_attention_num_buckets=32,
    relative_attention_max_distance=128,
    is_gated_act=True,
    dense_act_fn="gelu_new",
    layer_norm_epsilon=1e-6,
    feed_forward_proj="gated-gelu",
)


@pytest.fixture(scope="module")
def umt5_config() -> T5Config:
    return T5Config(**SMALL_UMT5_CONFIG)


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group(monkeypatch, mocker):
    """TP=2, rank 0, on CPU — mirrors test_t5_encoder_tp.py's fixture."""
    device_config = DeviceConfig(device="cpu")

    monkeypatch.setattr("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_world_size",
        lambda: 2,
    )

    monkeypatch.setattr(f"{_T5_MODULE}.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        "vllm.model_executor.layers.vocab_parallel_embedding.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    mock_tp_group = mocker.MagicMock()
    mock_tp_group.world_size = 2
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_tp_group)

    monkeypatch.setattr(f"{_T5_MODULE}.get_act_fn", lambda _: torch.nn.GELU())

    with set_current_vllm_config(VllmConfig(device_config=device_config)):
        yield


def _bias_blocks(stack: T5Stack) -> list[int]:
    """Indices of blocks that own a relative_attention_bias table.

    T5SelfAttention only *assigns* the attribute when it has a bias, so this
    probes with getattr rather than comparing against None.
    """
    return [
        i
        for i, block in enumerate(stack.block)
        if getattr(block.layer[0].SelfAttention, "relative_attention_bias", None) is not None
    ]


class TestPerLayerBiasConstruction:
    def test_umt5_builds_a_bias_table_on_every_block(self, umt5_config):
        model = UMT5EncoderModel(umt5_config)

        assert _bias_blocks(model.encoder) == list(range(umt5_config.num_layers))

    def test_t5_default_is_unchanged(self, umt5_config):
        """Guards the pipelines already shipping on T5EncoderModel."""
        model = T5EncoderModel(umt5_config)

        assert _bias_blocks(model.encoder) == [0]
        assert model.encoder.per_layer_relative_attention_bias is False

    def test_bias_tables_are_distinct_objects(self, umt5_config):
        model = UMT5EncoderModel(umt5_config)

        tables = {id(b.layer[0].SelfAttention.relative_attention_bias.weight) for b in model.encoder.block}
        assert len(tables) == umt5_config.num_layers

    def test_bias_table_is_full_width_not_sharded(self, umt5_config):
        """The table is [buckets, heads] and is sliced per rank in compute_bias.

        Sharding it as a parameter would break the bucket lookup, so it must
        stay full width even at TP=2.
        """
        model = UMT5EncoderModel(umt5_config)

        table = model.encoder.block[0].layer[0].SelfAttention.relative_attention_bias.weight
        assert table.shape == (
            umt5_config.relative_attention_num_buckets,
            umt5_config.num_heads,
        )


class TestPositionBiasThreading:
    """The other half of the coupled change: who computes the bias, per block."""

    class _RecordingBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen: list[torch.Tensor | None] = []

        def forward(self, hidden_states, mask=None, position_bias=None):
            self.seen.append(position_bias)
            # Stand in for a real block's returned bias.
            return hidden_states, torch.zeros(1)

    def _run_with_stub_blocks(self, config: T5Config, per_layer: bool) -> list[list]:
        shared = nn.Embedding(config.vocab_size, config.d_model)
        stack = T5Stack(config, shared, per_layer_relative_attention_bias=per_layer)
        stubs = [self._RecordingBlock() for _ in range(config.num_layers)]
        stack.block = nn.ModuleList(stubs)
        stack.embed_tokens = shared

        stack(torch.zeros(1, 3, dtype=torch.long))
        return [stub.seen for stub in stubs]

    def test_umt5_gives_every_block_a_fresh_none(self, umt5_config):
        seen = self._run_with_stub_blocks(umt5_config, per_layer=True)

        # Every block recomputes its own bias (and re-adds the mask), matching
        # HF UMT5Attention, which is constructed with
        # has_relative_attention_bias=True for every layer.
        assert all(s == [None] for s in seen)

    def test_t5_threads_block_zero_forward(self, umt5_config):
        seen = self._run_with_stub_blocks(umt5_config, per_layer=False)

        assert seen[0] == [None]
        for later in seen[1:]:
            assert later[0] is not None


class TestDreamZeroKeyNamespace:
    """The exact key namespace DreamZeroPipeline._remap_text_encoder_key emits.

    DreamZero routes its `action_head.text_encoder.*` stream through this
    module's load_weights(). Its only weight source declares prefix="", so the
    loader's strict check requires EVERY parameter to be filled — a namespace
    drift here surfaces as a startup ValueError, or worse, as a partially
    random encoder.
    """

    @staticmethod
    def _dreamzero_weights(config: T5Config) -> list[tuple[str, torch.Tensor]]:
        inner_dim = config.num_heads * config.d_kv
        weights: list[tuple[str, torch.Tensor]] = [
            ("shared.weight", torch.randn(config.vocab_size, config.d_model)),
            ("encoder.final_layer_norm.weight", torch.randn(config.d_model)),
        ]
        for i in range(config.num_layers):
            attn = f"encoder.block.{i}.layer.0.SelfAttention."
            ff = f"encoder.block.{i}.layer.1.DenseReluDense."
            weights += [
                (attn + "q.weight", torch.randn(inner_dim, config.d_model)),
                (attn + "k.weight", torch.randn(inner_dim, config.d_model)),
                (attn + "v.weight", torch.randn(inner_dim, config.d_model)),
                (attn + "o.weight", torch.randn(config.d_model, inner_dim)),
                (
                    attn + "relative_attention_bias.weight",
                    torch.randn(config.relative_attention_num_buckets, config.num_heads),
                ),
                (f"encoder.block.{i}.layer.0.layer_norm.weight", torch.randn(config.d_model)),
                (ff + "wi_0.weight", torch.randn(config.d_ff, config.d_model)),
                (ff + "wi_1.weight", torch.randn(config.d_ff, config.d_model)),
                (ff + "wo.weight", torch.randn(config.d_model, config.d_ff)),
                (f"encoder.block.{i}.layer.1.layer_norm.weight", torch.randn(config.d_model)),
            ]
        return weights

    def test_every_parameter_is_filled(self, umt5_config):
        model = UMT5EncoderModel(umt5_config)

        reported = model.load_weights(self._dreamzero_weights(umt5_config))

        expected = set(dict(model.named_parameters()))
        filled = {name for name in reported if name in expected}
        assert expected - filled == set(), f"unfilled parameters: {sorted(expected - filled)}"

    def test_streaming_one_key_at_a_time_is_equivalent(self, umt5_config):
        """DreamZero feeds keys individually to keep the checkpoint stream lazy."""
        model = UMT5EncoderModel(umt5_config)

        filled: set[str] = set()
        for entry in self._dreamzero_weights(umt5_config):
            filled |= set(model.load_weights([entry]))

        expected = set(dict(model.named_parameters()))
        assert expected - {name for name in filled if name in expected} == set()

    def test_qkv_and_gated_ffn_are_fused_and_sharded(self, umt5_config):
        model = UMT5EncoderModel(umt5_config)
        model.load_weights(self._dreamzero_weights(umt5_config))

        local_heads = umt5_config.num_heads // 2
        attn = model.encoder.block[0].layer[0].SelfAttention
        assert attn.qkv_proj.weight.shape == (3 * local_heads * umt5_config.d_kv, umt5_config.d_model)

        # wi_0 and wi_1 fuse into one MergedColumnParallelLinear, each half
        # column-sharded: 2 * (d_ff / tp).
        ffn = model.encoder.block[0].layer[1].DenseReluDense
        assert ffn.wi.weight.shape == (2 * (umt5_config.d_ff // 2), umt5_config.d_model)

    def test_unknown_key_fills_nothing(self, umt5_config):
        model = UMT5EncoderModel(umt5_config)

        reported = model.load_weights([("encoder.block.0.layer.0.SelfAttention.nope.weight", torch.randn(4))])

        assert {name for name in reported if name in dict(model.named_parameters())} == set()
