from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.models.sd3.sd3_transformer as sd3_transformer
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_sd3_sampling(**overrides):
    values = {
        "height": 32,
        "width": 32,
        "num_inference_steps": 2,
        "sigmas": None,
        "max_sequence_length": None,
        "num_outputs_per_prompt": 0,
        "generator": None,
        "latents": None,
        "guidance_scale": 4.0,
        "output_type": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_sd3_pipeline():
    pipeline = object.__new__(StableDiffusion3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.patch_size = 2
    pipeline.default_sample_size = 128
    pipeline.transformer = SimpleNamespace(in_channels=1)
    return pipeline


def _make_pretrained_stub():
    stub = MagicMock()
    stub.config = SimpleNamespace(block_out_channels=[1, 2, 4])
    stub.to.return_value = stub
    return stub


def _make_sd3_model_config(**overrides):
    values = {
        "num_layers": 2,
        "sample_size": 32,
        "in_channels": 16,
        "out_channels": 16,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "caption_projection_dim": 8,
        "pooled_projection_dim": 8,
        "joint_attention_dim": 8,
        "patch_size": 2,
        "dual_attention_layers": (),
        "qk_norm": "rms_norm",
        "pos_embed_max_size": 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_fake_linear(records, kind):
    class _FakeLinear(nn.Module):
        def __init__(self, *args, quant_config=None, prefix="", total_num_heads=None, **kwargs):
            super().__init__()
            records.append(
                {
                    "kind": kind,
                    "quant_config": quant_config,
                    "prefix": prefix,
                }
            )
            self.quant_config = quant_config
            self.prefix = prefix
            self.num_heads = total_num_heads or kwargs.get("total_num_heads") or 1

        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states, None

    return _FakeLinear


class _FakeAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise AssertionError("attention forward should not be called during constructor-only tests")


def test_forward_collates_request_prompt_tensors_for_sd3():
    pipeline = _make_sd3_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    encode_calls = []
    diffuse_call = {}

    def _fake_encode_prompt(**kwargs):
        encode_calls.append(kwargs)
        prompt_embeds = kwargs["prompt_embeds"]
        if prompt_embeds is None:
            prompt_embeds = torch.empty(2, 2, 3)
        return prompt_embeds, kwargs.get("pooled_prompt_embeds")

    def _fake_diffuse(**kwargs):
        diffuse_call.update(kwargs)
        raise StopAfterDiffuseError

    pipeline.encode_prompt = _fake_encode_prompt
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(2, 1, 1, 1)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0]), 1)
    pipeline.diffuse = _fake_diffuse

    prompt_embeds_a = torch.zeros(2, 3)
    prompt_embeds_b = torch.ones(2, 3)
    pooled_prompt_embeds_a = torch.full((4,), 2.0)
    pooled_prompt_embeds_b = torch.full((4,), 3.0)
    negative_prompt_embeds_a = torch.full((2, 3), 4.0)
    negative_prompt_embeds_b = torch.full((2, 3), 5.0)
    negative_pooled_prompt_embeds_a = torch.full((4,), 6.0)
    negative_pooled_prompt_embeds_b = torch.full((4,), 7.0)

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="sd3-prompt-a",
                prompt={
                    "prompt": "prompt-a",
                    "negative_prompt": "negative-a",
                    "prompt_embeds": prompt_embeds_a,
                    "pooled_prompt_embeds": pooled_prompt_embeds_a,
                    "negative_prompt_embeds": negative_prompt_embeds_a,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds_a,
                },
                sampling_params=_make_sd3_sampling(),
            ),
            SimpleNamespace(
                request_id="sd3-prompt-b",
                prompt={
                    "prompt": "prompt-b",
                    "negative_prompt": "negative-b",
                    "additional_information": {
                        "prompt_embeds": [prompt_embeds_b],
                        "pooled_prompt_embeds": [pooled_prompt_embeds_b],
                        "negative_prompt_embeds": [negative_prompt_embeds_b],
                        "negative_pooled_prompt_embeds": [negative_pooled_prompt_embeds_b],
                    },
                },
                sampling_params=_make_sd3_sampling(),
            ),
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(batch)

    assert encode_calls[0]["prompt"] is None
    assert encode_calls[0]["prompt_2"] is None
    assert encode_calls[0]["prompt_3"] is None
    torch.testing.assert_close(
        encode_calls[0]["prompt_embeds"],
        torch.stack([prompt_embeds_a, prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        encode_calls[0]["pooled_prompt_embeds"],
        torch.stack([pooled_prompt_embeds_a, pooled_prompt_embeds_b], dim=0),
    )

    assert encode_calls[1]["prompt"] is None
    assert encode_calls[1]["prompt_2"] is None
    assert encode_calls[1]["prompt_3"] is None
    torch.testing.assert_close(
        encode_calls[1]["prompt_embeds"],
        torch.stack([negative_prompt_embeds_a, negative_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        encode_calls[1]["pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        diffuse_call["pooled_prompt_embeds"],
        torch.stack([pooled_prompt_embeds_a, pooled_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        diffuse_call["negative_pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )


def test_encode_prompt_preserves_direct_pooled_prompt_embeds():
    pipeline = _make_sd3_pipeline()
    prompt_embeds = torch.zeros(1, 2, 3)
    pooled_prompt_embeds = torch.ones(1, 4)

    actual_prompt_embeds, actual_pooled_prompt_embeds = pipeline.encode_prompt(
        prompt=None,
        prompt_2=None,
        prompt_3=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
    )

    assert actual_prompt_embeds is prompt_embeds
    assert actual_pooled_prompt_embeds is pooled_prompt_embeds


@patch(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3.prefetch_subfolders",
)
@patch(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3.from_pretrained_with_prefetch",
)
@patch(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3.T5Tokenizer.from_pretrained",
    return_value=MagicMock(),
)
@patch(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3.CLIPTokenizer.from_pretrained",
    return_value=MagicMock(),
)
@patch(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3.FlowMatchEulerDiscreteScheduler.from_pretrained",
    return_value=MagicMock(),
)
@patch(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3.get_local_device",
    return_value="cpu",
)
def test_sd3_pipeline_passes_quant_config_to_transformer(
    _mock_get_local_device,
    _mock_scheduler,
    _mock_clip_tokenizer,
    _mock_t5_tokenizer,
    mock_from_pretrained,
    _mock_prefetch,
):
    captured_kwargs = {}

    class FakeTransformer:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    mock_from_pretrained.side_effect = lambda *args, **kwargs: _make_pretrained_stub()

    with patch("vllm_omni.diffusion.models.sd3.pipeline_sd3.SD3Transformer2DModel", FakeTransformer):
        fake_quant_config = MagicMock()
        od_config = SimpleNamespace(
            model="fake-sd3-model",
            dtype=torch.bfloat16,
            tf_model_config=SimpleNamespace(),
            quantization_config=fake_quant_config,
            output_type="pil",
            enable_diffusion_pipeline_profiler=False,
        )

        StableDiffusion3Pipeline(od_config=od_config)

    assert captured_kwargs["od_config"] is od_config
    assert captured_kwargs["quant_config"] is fake_quant_config


def test_sd3_transformer_propagates_quant_config_to_key_submodules():
    records = []
    fake_quant_config = MagicMock(name="fake_quant_config")
    od_config = SimpleNamespace(
        tf_model_config=_make_sd3_model_config(),
        parallel_config=SimpleNamespace(),
        quantization_config=fake_quant_config,
    )

    with (
        patch.object(sd3_transformer, "QKVParallelLinear", _make_fake_linear(records, "qkv")),
        patch.object(sd3_transformer, "RowParallelLinear", _make_fake_linear(records, "row")),
        patch.object(sd3_transformer, "ColumnParallelLinear", _make_fake_linear(records, "column")),
        patch.object(sd3_transformer, "ReplicatedLinear", _make_fake_linear(records, "replicated")),
        patch.object(sd3_transformer, "Attention", _FakeAttention),
    ):
        sd3_transformer.SD3Transformer2DModel(od_config=od_config, quant_config=fake_quant_config)

    record_by_prefix = {item["prefix"]: item for item in records}
    expected_prefixes = {
        "context_embedder",
        "transformer_blocks.0.attn.to_qkv",
        "transformer_blocks.0.attn.add_kv_proj",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.ff.net.2",
        "transformer_blocks.0.ff_context.net.0.proj",
        "transformer_blocks.0.ff_context.net.2",
        "proj_out",
    }

    assert expected_prefixes <= set(record_by_prefix)
    for prefix in expected_prefixes:
        assert record_by_prefix[prefix]["quant_config"] is fake_quant_config
