# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vllm_omni.model_executor.models.indextts2 import indextts2_s2mel_decoder
from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
    IndexTTS2S2MelDecoder,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules import (
    flow_matching,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.flow_matching import (
    BASECFM,
    CFM,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.gpt_fast import (
    model as gpt_fast_model,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeEnhancedCodec:
    downsample_scale = 2

    def __init__(self):
        self.decode_calls = 0

    def decode(self, codes):
        self.decode_calls += 1
        return torch.ones(codes.shape[0], codes.shape[1] * 2, 4)


class _FakeS2Mel:
    def forward_gpt(self, latent):
        return torch.full((*latent.shape[:-1], 4), 2.0)


class _FakeBigVGAN:
    def __init__(self):
        self.events = []

    def remove_weight_norm(self):
        self.events.append("remove_weight_norm")

    def to(self, *, device, dtype):
        self.events.append(("to", device, dtype))
        return self

    def eval(self):
        self.events.append("eval")
        return self

    def parameters(self):
        return []


class _DefaultDtypeBigVGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.fold_weight_dtype = None
        self.fold_default_dtype = None

    def remove_weight_norm(self):
        self.fold_weight_dtype = self.weight.dtype
        self.fold_default_dtype = torch.get_default_dtype()


class _LengthRecordingBigVGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_lengths = []

    def forward(self, mel):
        self.input_lengths.append(mel.shape[-1])
        return torch.zeros(
            mel.shape[0],
            1,
            mel.shape[-1] * 256,
            dtype=mel.dtype,
            device=mel.device,
        )


class _NoiseEchoCFM(BASECFM):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.in_channels = 2

    def solve_euler(
        self,
        x,
        x_lens,
        prompt,
        mu,
        style,
        f0,
        t_span,
        inference_cfg_rate=0.5,
        prompt_lens=None,
        unpad_data=None,
    ):
        del unpad_data
        return x


class _PromptRecordingEstimator(torch.nn.Module):
    style_as_token = False
    time_as_token = False
    is_causal = False

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        x,
        prompt,
        x_lens,
        _t,
        _style,
        mu,
        *,
        pre_mask,
        unpad_data=None,
    ):
        self.calls.append(
            {
                "x": x.clone(),
                "prompt": prompt.clone(),
                "x_lens": x_lens.clone(),
                "mu": mu.clone(),
                "pre_mask": tuple(mask.clone() for mask in pre_mask),
                "unpad_data": unpad_data,
            }
        )
        return torch.ones_like(x)


class _VariablePromptCFM(BASECFM):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.in_channels = 2
        self.zero_prompt_speech_token = True
        self.estimator = _PromptRecordingEstimator()


class _GroupingEstimator:
    def __init__(self):
        layers = [SimpleNamespace(attention=SimpleNamespace()) for _ in range(2)]
        self.transformer = SimpleNamespace(
            layers=layers,
            freqs_cis=torch.ones(1),
            max_batch_size=8,
        )


class _GroupingCFM:
    in_channels = 2

    def __init__(self):
        self.estimator = _GroupingEstimator()
        self.calls = []

    def inference(
        self,
        mu,
        x_lens,
        prompt,
        style,
        f0,
        n_timesteps,
        temperature=1.0,
        inference_cfg_rate=0.5,
        initial_noise=None,
        prompt_lens=None,
        unpad_data=None,
    ):
        self.calls.append(
            {
                "mu": mu.clone(),
                "x_lens": x_lens.clone(),
                "prompt": prompt.clone(),
                "prompt_lens": prompt_lens.clone(),
                "style": style.clone(),
                "f0": f0,
                "n_timesteps": n_timesteps,
                "temperature": temperature,
                "inference_cfg_rate": inference_cfg_rate,
                "initial_noise": initial_noise.clone(),
                "unpad_data": unpad_data,
                "full_mask": all(layer.attention._assume_full_mask for layer in self.estimator.transformer.layers),
            }
        )
        return mu[..., 0].unsqueeze(1).expand(-1, self.in_channels, -1).clone()


class _OrderRecordingBigVGAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, mel):
        self.calls.append((int(mel.shape[-1]), float(mel[0, 0, 0])))
        return mel[:, :1].repeat_interleave(4, dim=-1)


class _MetadataRecordingDiffusionAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metadata = []

    def forward(self, query, _key, _value, *, attn_metadata):
        self.metadata.append(attn_metadata)
        return query


class _IdentityQKV(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x, x, x], dim=-1)


def _make_decoder(*, use_gpt_latent: bool):
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.use_gpt_latent = use_gpt_latent
    decoder.semantic_codec_type = "enhanced"
    decoder.s2mel = _FakeS2Mel()
    return decoder


def test_v25_code_only_semantic_embedding_uses_enhanced_decode():
    decoder = _make_decoder(use_gpt_latent=False)
    codec = _FakeEnhancedCodec()

    result = decoder._build_semantic_embedding(
        codec=codec,
        mel_codes=torch.tensor([[1, 2, 3]]),
        latent=None,
        dtype=torch.float32,
    )

    assert codec.decode_calls == 1
    assert result.shape == (1, 6, 4)
    assert torch.equal(result, torch.ones(1, 6, 4))


def test_v25_latent_mode_upsamples_projected_latent_before_addition():
    decoder = _make_decoder(use_gpt_latent=True)
    codec = _FakeEnhancedCodec()

    result = decoder._build_semantic_embedding(
        codec=codec,
        mel_codes=torch.tensor([[1, 2, 3]]),
        latent=torch.ones(1, 3, 8),
        dtype=torch.float32,
    )

    assert result.shape == (1, 6, 4)
    assert torch.equal(result, torch.full((1, 6, 4), 3.0))


def test_s2mel_payload_latent_policy_must_match_stage_config():
    decoder = _make_decoder(use_gpt_latent=False)

    decoder._validate_payload_policy([{"use_gpt_latent": False}])
    with pytest.raises(ValueError, match="use_gpt_latent"):
        decoder._validate_payload_policy([{"use_gpt_latent": True}])


@pytest.mark.parametrize("missing_key", ["mel_codes", "S_ref", "ref_mel", "style"])
def test_s2mel_payload_contract_requires_non_empty_conditioning(missing_key):
    payload = {
        "mel_codes": torch.ones(2, dtype=torch.long),
        "S_ref": torch.ones(2, 4),
        "ref_mel": torch.ones(80, 2),
        "style": torch.ones(192),
    }
    payload[missing_key] = torch.empty(0)

    with pytest.raises(ValueError, match=f"non-empty {missing_key}"):
        IndexTTS2S2MelDecoder._validate_payload_contract([payload])


def test_s2mel_target_lengths_apply_request_local_duration_factors():
    assert IndexTTS2S2MelDecoder._target_lengths(
        [100, 80],
        semantic_time_scale=2,
        mel_code_to_frame_ratio=1.72,
        duration_factors=[0.5, 2.0],
    ) == [172, 550]


def test_s2mel_duration_factors_default_to_one_without_request_infos():
    assert IndexTTS2S2MelDecoder._duration_factors([], batch_size=2) == [1.0, 1.0]


def test_s2mel_duration_factors_default_missing_request_value_to_one():
    assert IndexTTS2S2MelDecoder._duration_factors(
        [{}, {"duration_factor": 0.5}],
        batch_size=2,
    ) == [1.0, 0.5]


def test_s2mel_duration_factors_accept_supported_bounds():
    assert IndexTTS2S2MelDecoder._duration_factors(
        [{"duration_factor": 0.5}, {"duration_factor": 2.0}],
        batch_size=2,
    ) == [0.5, 2.0]


def test_s2mel_duration_factor_count_must_match_batch_size():
    with pytest.raises(ValueError) as exc_info:
        IndexTTS2S2MelDecoder._duration_factors(
            [{"duration_factor": 1.0}],
            batch_size=2,
        )

    assert str(exc_info.value) == "IndexTTS duration-factor batch mismatch: factors=1 batch=2"


def test_s2mel_target_length_count_must_match_duration_factors():
    with pytest.raises(ValueError) as exc_info:
        IndexTTS2S2MelDecoder._target_lengths(
            [100, 80],
            semantic_time_scale=2,
            mel_code_to_frame_ratio=1.72,
            duration_factors=[1.0],
        )

    assert str(exc_info.value) == "IndexTTS target-length batch mismatch: codes=2 factors=1"


@pytest.mark.parametrize("duration_factor", [float("nan"), float("inf"), float("-inf"), 0.0, -0.5, 0.49, 2.01])
def test_s2mel_duration_factors_must_be_finite_and_in_supported_range(duration_factor):
    with pytest.raises(ValueError) as exc_info:
        IndexTTS2S2MelDecoder._duration_factors(
            [{"duration_factor": duration_factor}],
            batch_size=1,
        )

    assert str(exc_info.value) == "IndexTTS duration_factor values must be finite and between 0.5 and 2.0"


def test_cfm_unpad_data_builder_accounts_for_token_offsets_and_cfg_row_order():
    indices, cu_seqlens = flow_matching.build_cfm_unpad_data(
        [3, 5],
        padded_length=5,
        token_offset=2,
        repeat_for_cfg=True,
        device=torch.device("cpu"),
    )

    # Effective lengths are [5, 7] and CFG order is
    # [cond row 0, cond row 1, uncond row 0, uncond row 1]. The second
    # copy must use the actual padded row offsets (14 and 21), not copied
    # indices from the conditional half.
    assert indices.dtype == torch.int64
    assert torch.equal(
        indices,
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
            ],
            dtype=torch.int64,
        ),
    )
    assert cu_seqlens.dtype == torch.int32
    assert torch.equal(cu_seqlens, torch.tensor([0, 5, 12, 17, 24], dtype=torch.int32))


def test_cfm_inference_uses_explicit_initial_noise():
    cfm = _NoiseEchoCFM()
    initial_noise = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

    result = cfm.inference(
        mu=torch.zeros(1, 6, 4),
        x_lens=torch.tensor([6]),
        prompt=torch.zeros(1, 80, 0),
        style=torch.zeros(1, 192),
        f0=None,
        n_timesteps=1,
        temperature=0.5,
        initial_noise=initial_noise,
    )

    assert torch.equal(result, initial_noise * 0.5)


def test_cfm_variable_prompt_lens_are_applied_per_row_under_cfg_each_step():
    cfm = _VariablePromptCFM()
    initial_noise = torch.arange(20, dtype=torch.float32).reshape(2, 2, 5)
    prompt = torch.tensor(
        [
            [[11.0, 91.0, 92.0], [12.0, 93.0, 94.0]],
            [[21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
        ]
    )
    mu = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2) + 1
    prompt_lens = torch.tensor([1, 3], dtype=torch.long)
    unpad_data = (
        torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]),
        torch.tensor([0, 4, 9, 13, 18], dtype=torch.int32),
    )

    result = cfm.inference(
        mu=mu,
        x_lens=torch.tensor([4, 5], dtype=torch.long),
        prompt=prompt,
        prompt_lens=prompt_lens,
        style=torch.ones(2, 3),
        f0=None,
        n_timesteps=2,
        inference_cfg_rate=0.7,
        initial_noise=initial_noise,
        unpad_data=unpad_data,
    )

    assert len(cfm.estimator.calls) == 2
    for call in cfm.estimator.calls:
        # CFG packs conditional rows first and unconditional rows second.
        assert torch.count_nonzero(call["x"][0, :, :1]) == 0
        assert torch.count_nonzero(call["x"][1, :, :3]) == 0
        assert torch.count_nonzero(call["x"][2, :, :1]) == 0
        assert torch.count_nonzero(call["x"][3, :, :3]) == 0
        assert torch.equal(call["prompt"][0, :, :1], prompt[0, :, :1])
        assert torch.count_nonzero(call["prompt"][0, :, 1:]) == 0
        assert torch.equal(call["prompt"][1, :, :3], prompt[1])
        assert torch.count_nonzero(call["prompt"][2:]) == 0
        assert torch.count_nonzero(call["mu"][0, :1]) == 0
        assert torch.equal(call["mu"][0, 1:], mu[0, 1:])
        assert torch.count_nonzero(call["mu"][1, :3]) == 0
        assert torch.equal(call["mu"][1, 3:], mu[1, 3:])
        assert torch.count_nonzero(call["mu"][2:]) == 0
        assert torch.equal(call["x_lens"], torch.tensor([4, 5]))
        assert call["unpad_data"] is unpad_data

    assert torch.count_nonzero(result[0, :, :1]) == 0
    assert torch.count_nonzero(result[1, :, :3]) == 0


def test_seeded_cfm_noise_is_request_local_and_batch_order_independent():
    kwargs = {
        "channels": 2,
        "length": 6,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }

    noise_40_42 = IndexTTS2S2MelDecoder._sample_cfm_noise(
        seeds=[40, 42],
        **kwargs,
    )
    torch.randn(257)
    noise_42_40 = IndexTTS2S2MelDecoder._sample_cfm_noise(
        seeds=[42, 40],
        **kwargs,
    )

    assert noise_40_42 is not None
    assert noise_42_40 is not None
    assert torch.equal(noise_40_42[0], noise_42_40[1])
    assert torch.equal(noise_40_42[1], noise_42_40[0])
    assert not torch.equal(noise_40_42[0], noise_40_42[1])


def test_unseeded_cfm_noise_preserves_upstream_global_rng_behavior():
    assert (
        IndexTTS2S2MelDecoder._sample_cfm_noise(
            seeds=[None, None],
            channels=2,
            length=6,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )


def test_variable_length_cfm_groups_pack_batch_crop_and_restore_request_order():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.diffusion_steps = 2
    decoder.inference_cfg_rate = 0.7
    decoder.s2mel_dit_cuda_graph = False
    cfm = _GroupingCFM()

    # Original request order is deliberately different from total-length order.
    # Stable sort by ref+target must produce groups [1, 3] and [2, 0].
    ref_lengths = [4, 1, 2, 2]
    target_lengths = [3, 2, 3, 1]
    seeds = [101, 202, 303, 404]
    prompt_markers = [110.0, 120.0, 130.0, 140.0]
    target_markers = [10.0, 20.0, 30.0, 40.0]

    prompt_condition = torch.full((4, 4, 2), -777.0)
    cond = torch.full((4, 3, 2), -999.0)
    ref_mel = torch.full((4, 2, 4), -555.0)
    style = torch.zeros(4, 3)
    for index, (ref_len, target_len) in enumerate(zip(ref_lengths, target_lengths)):
        prompt_condition[index, :ref_len] = prompt_markers[index]
        cond[index, :target_len] = target_markers[index]
        ref_mel[index, :, :ref_len] = prompt_markers[index] + 1
        style[index] = index + 1

    mels = decoder._run_cfm_groups(
        cfm=cfm,
        cond=cond,
        prompt_condition=prompt_condition,
        ref_mel=ref_mel,
        style=style,
        target_lengths=target_lengths,
        ref_lengths=ref_lengths,
        seeds=seeds,
        max_group_size=2,
    )

    assert len(cfm.calls) == 2
    assert [call["x_lens"].tolist() for call in cfm.calls] == [[3, 3], [5, 7]]
    assert [call["prompt_lens"].tolist() for call in cfm.calls] == [[1, 2], [2, 4]]
    assert [call["style"][:, 0].tolist() for call in cfm.calls] == [[2.0, 4.0], [3.0, 1.0]]

    first, second = cfm.calls
    assert torch.equal(first["mu"][0, :1], torch.full((1, 2), 120.0))
    assert torch.equal(first["mu"][0, 1:3], torch.full((2, 2), 20.0))
    assert torch.equal(first["mu"][1, :2], torch.full((2, 2), 140.0))
    assert torch.equal(first["mu"][1, 2:3], torch.full((1, 2), 40.0))
    assert torch.equal(second["mu"][0, :2], torch.full((2, 2), 130.0))
    assert torch.equal(second["mu"][0, 2:5], torch.full((3, 2), 30.0))
    assert torch.count_nonzero(second["mu"][0, 5:]) == 0
    assert torch.equal(second["mu"][1, :4], torch.full((4, 2), 110.0))
    assert torch.equal(second["mu"][1, 4:7], torch.full((3, 2), 10.0))

    assert torch.equal(first["prompt"][0, :, :1], torch.full((2, 1), 121.0))
    assert torch.count_nonzero(first["prompt"][0, :, 1:]) == 0
    assert torch.equal(second["prompt"][0, :, :2], torch.full((2, 2), 131.0))
    assert torch.count_nonzero(second["prompt"][0, :, 2:]) == 0

    # Equal total lengths have no tail padding and may use the full-mask path;
    # the mixed-length group must explicitly disable it.
    assert [call["full_mask"] for call in cfm.calls] == [True, False]
    assert first["unpad_data"] is None
    ragged_indices, ragged_cu_seqlens = second["unpad_data"]
    assert torch.equal(
        ragged_indices,
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
            ]
        ),
    )
    assert torch.equal(ragged_cu_seqlens, torch.tensor([0, 5, 12, 17, 24], dtype=torch.int32))

    for call, sorted_indices in zip(cfm.calls, ([1, 3], [2, 0])):
        for row, request_index in enumerate(sorted_indices):
            x_len = ref_lengths[request_index] + target_lengths[request_index]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seeds[request_index])
            expected = torch.randn((2, x_len), generator=generator)
            assert torch.equal(call["initial_noise"][row, :, :x_len], expected)
            assert torch.count_nonzero(call["initial_noise"][row, :, x_len:]) == 0

    assert [tuple(mel.shape) for mel in mels] == [
        (2, 3),
        (2, 2),
        (2, 3),
        (2, 1),
    ]
    for mel, marker in zip(mels, target_markers):
        assert torch.equal(mel, torch.full_like(mel, marker))


def test_bigvgan_folds_weight_norm_before_low_precision_cast(monkeypatch):
    from vllm_omni.model_executor.models.indextts2.s2mel.modules import bigvgan

    model = _FakeBigVGAN()
    monkeypatch.setattr(
        bigvgan.BigVGAN,
        "from_pretrained",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_patch_bigvgan_compat",
        lambda _cls: None,
    )
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_repo_prefetch_lock",
        lambda _name: None,
    )
    indextts2_s2mel_decoder._bigvgan_models.clear()

    loaded = indextts2_s2mel_decoder._load_bigvgan(
        "fake-bigvgan",
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert loaded is model
    assert model.events == [
        "remove_weight_norm",
        ("to", torch.device("cpu"), torch.bfloat16),
        "eval",
    ]


def test_bigvgan_folds_weight_norm_in_fp32_under_low_precision_loader_context(
    monkeypatch,
):
    from vllm_omni.model_executor.models.indextts2.s2mel.modules import bigvgan

    monkeypatch.setattr(
        bigvgan.BigVGAN,
        "from_pretrained",
        lambda *_args, **_kwargs: _DefaultDtypeBigVGAN(),
    )
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_patch_bigvgan_compat",
        lambda _cls: None,
    )
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_repo_prefetch_lock",
        lambda _name: None,
    )
    indextts2_s2mel_decoder._bigvgan_models.clear()

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        loaded = indextts2_s2mel_decoder._load_bigvgan(
            "fake-bigvgan",
            torch.device("cpu"),
            torch.bfloat16,
        )

        assert loaded.fold_weight_dtype == torch.float32
        assert loaded.fold_default_dtype == torch.float32
        assert loaded.weight.dtype == torch.bfloat16
        assert torch.get_default_dtype() == torch.bfloat16
    finally:
        torch.set_default_dtype(previous_dtype)


def _cfm_estimator_with_mixed_parameter_types():
    return torch.nn.ModuleDict(
        {
            "linear": torch.nn.Linear(3, 2),
            "conv": torch.nn.Conv1d(2, 4, kernel_size=3, bias=False),
            "norm": torch.nn.LayerNorm(4),
        }
    )


def test_cfm_estimator_bf16_precast_only_casts_linear_and_conv1d():
    estimator = _cfm_estimator_with_mixed_parameter_types()

    parameter_count = indextts2_s2mel_decoder._precast_cfm_estimator_bf16(
        estimator,
        enabled=True,
    )

    assert parameter_count == 32
    assert estimator["linear"].weight.dtype == torch.bfloat16
    assert estimator["linear"].bias.dtype == torch.bfloat16
    assert estimator["conv"].weight.dtype == torch.bfloat16
    assert estimator["norm"].weight.dtype == torch.float32
    assert estimator["norm"].bias.dtype == torch.float32


def test_cfm_estimator_bf16_precast_disabled_preserves_fp32():
    estimator = _cfm_estimator_with_mixed_parameter_types()

    parameter_count = indextts2_s2mel_decoder._precast_cfm_estimator_bf16(
        estimator,
        enabled=False,
    )

    assert parameter_count == 0
    assert all(parameter.dtype == torch.float32 for parameter in estimator.parameters())


def test_v25_full_mask_fast_path_is_guarded_independently_of_cuda_graph():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.s2mel_dit_cuda_graph = False
    layers = [SimpleNamespace(attention=SimpleNamespace()) for _ in range(13)]
    estimator = SimpleNamespace(
        transformer=SimpleNamespace(layers=layers),
    )

    decoder._set_dit_full_mask_fast_path(estimator, enabled=True)
    decoder._enable_dit_cuda_graph(estimator)

    assert all(layer.attention._assume_full_mask is True for layer in layers)
    assert not hasattr(estimator, "_cuda_graph_runner")

    decoder._set_dit_full_mask_fast_path(estimator, enabled=False)

    assert all(layer.attention._assume_full_mask is False for layer in layers)


def test_indextts_attention_keeps_padding_metadata_model_local(monkeypatch):
    attention = object.__new__(gpt_fast_model.Attention)
    torch.nn.Module.__init__(attention)
    attention.n_head = 1
    attention.n_local_heads = 1
    attention.head_dim = 2
    attention.dim = 2
    attention.wqkv = _IdentityQKV()
    attention.wo = torch.nn.Identity()
    attention.diff_attn = _MetadataRecordingDiffusionAttention()
    monkeypatch.setattr(gpt_fast_model, "apply_rotary_emb", lambda tensor, _freqs: tensor)

    x = torch.ones(2, 3, 2)
    mask = torch.tensor(
        [
            [[[True, True, True], [True, True, True], [True, True, True]]],
            [[[True, True, False], [True, True, False], [True, True, False]]],
        ]
    )
    attention._assume_full_mask = False
    unpad_data = (
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    attention(x, torch.empty(0), mask, unpad_data=unpad_data)

    padded_metadata = attention.diff_attn.metadata[-1]
    assert padded_metadata.extra == {}
    assert torch.equal(padded_metadata.attn_mask, mask[:, 0, 0, :])

    attention._assume_full_mask = True
    attention(x, torch.empty(0), mask)

    assert attention.diff_attn.metadata[-1] is None

    all_true_mask = torch.ones_like(mask)
    for unknown_state in ("missing", None):
        if unknown_state == "missing":
            del attention._assume_full_mask
        else:
            attention._assume_full_mask = unknown_state

        attention(x, torch.empty(0), all_true_mask)

        fallback_metadata = attention.diff_attn.metadata[-1]
        assert fallback_metadata.extra == {}
        assert torch.equal(fallback_metadata.attn_mask, all_true_mask[:, 0, 0, :])


def test_v25_ref_mel_lengths_come_from_unpadded_shapes():
    ref_mels = [
        torch.zeros(80, 5),
        torch.ones(80, 3),
    ]

    assert IndexTTS2S2MelDecoder._ref_mel_lengths(ref_mels) == [5, 3]
    assert IndexTTS2S2MelDecoder._ref_mel_lengths([None, ref_mels[1]]) == []


def test_v25_resolved_vocoder_source_is_cached(monkeypatch):
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.model_path = "/models/indextts25"
    decoder.config = SimpleNamespace(vocoder={"name": "fake-bigvgan"})
    decoder._resolved_vocoder_source = None
    calls = []

    def fake_resolve(model_path, vocoder_name):
        calls.append((model_path, vocoder_name))
        return "/models/indextts25/hf_cache/bigvgan"

    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_resolve_bigvgan_source",
        fake_resolve,
    )

    first = decoder._get_resolved_vocoder_source()
    second = decoder._get_resolved_vocoder_source()

    assert first == "/models/indextts25/hf_cache/bigvgan"
    assert second == first
    assert calls == [("/models/indextts25", "fake-bigvgan")]


def test_stage1_preload_does_not_swallow_cuda_runtime_failure(monkeypatch):
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.s2mel = torch.nn.Linear(1, 1)
    decoder.model_path = "/models/indextts25"
    decoder.semantic_codec_type = "enhanced"
    decoder.s2mel_vocoder_bf16 = False
    decoder.s2mel_vocoder_cuda_graph = False
    decoder._resolved_vocoder_source = "/models/bigvgan"
    decoder.config = SimpleNamespace(
        semantic_codec={},
        semantic_codec_checkpoint="codec.pth",
    )

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("CUDA error: illegal memory access")

    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_load_bigvgan",
        fail_load,
    )

    with pytest.raises(RuntimeError, match="CUDA error"):
        decoder._warmup_external_models()


def _runtime_decoder(*, compile_dit: bool):
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    calls = []
    decoder.s2mel = SimpleNamespace(
        enable_torch_compile=lambda *, mode="default": calls.append("compile"),
    )
    decoder.s2mel_dit_torch_compile = compile_dit
    decoder.s2mel_dit_torch_compile_mode = "default"
    decoder.s2mel_dit_cuda_graph = False
    decoder._dit_runtime_configured = False
    return decoder, calls


def test_configure_dit_runtime_compiles_only_on_cuda():
    cpu_decoder, cpu_calls = _runtime_decoder(compile_dit=True)
    cuda_decoder, cuda_calls = _runtime_decoder(compile_dit=True)

    cpu_decoder._configure_dit_runtime(torch.device("cpu"))
    cuda_decoder._configure_dit_runtime(torch.device("cuda"))
    cuda_decoder._configure_dit_runtime(torch.device("cuda"))

    assert cpu_calls == []
    assert cuda_calls == ["compile"]


def test_cfm_torch_compile_forwards_reduce_overhead_mode(monkeypatch):
    cfm = object.__new__(CFM)
    torch.nn.Module.__init__(cfm)
    eager_estimator = torch.nn.Identity()
    cfm.estimator = eager_estimator
    object.__setattr__(cfm, "_eager_estimator", eager_estimator)
    calls = []

    def fake_compile(model, **kwargs):
        calls.append((model, kwargs))
        return model

    monkeypatch.setattr(flow_matching.torch, "compile", fake_compile)

    cfm.enable_torch_compile(mode="reduce-overhead")

    assert calls == [
        (
            eager_estimator,
            {
                "mode": "reduce-overhead",
                "fullgraph": True,
                "dynamic": True,
            },
        )
    ]


def test_configure_dit_runtime_forwards_compile_mode_once():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    calls = []
    decoder.s2mel = SimpleNamespace(
        enable_torch_compile=lambda *, mode="default": calls.append(mode),
    )
    decoder.s2mel_dit_torch_compile = True
    decoder.s2mel_dit_torch_compile_mode = "reduce-overhead"
    decoder.s2mel_dit_cuda_graph = False
    decoder._dit_runtime_configured = False

    decoder._configure_dit_runtime(torch.device("cuda"))
    decoder._configure_dit_runtime(torch.device("cuda"))

    assert calls == ["reduce-overhead"]


def _vocoder_decoder(*, compile_vocoder: bool, graph_vocoder: bool = False):
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.s2mel_vocoder_torch_compile = compile_vocoder
    decoder.s2mel_vocoder_cuda_graph = graph_vocoder
    decoder._compiled_vocoder = None
    return decoder


def test_vocoder_torch_compile_disabled_returns_original_model(monkeypatch):
    decoder = _vocoder_decoder(compile_vocoder=False)
    bigvgan = _LengthRecordingBigVGAN()
    runner_factory = getattr(decoder, "_get_vocoder_runner", None)

    assert runner_factory is not None

    def unexpected_compile(*_args, **_kwargs):
        pytest.fail("torch.compile must stay disabled")

    monkeypatch.setattr(indextts2_s2mel_decoder.torch, "compile", unexpected_compile)

    runner = runner_factory(
        bigvgan,
        torch.device("cuda"),
        torch.bfloat16,
    )

    assert runner is bigvgan


def test_vocoder_torch_compile_is_cached_and_preserves_exact_lengths(monkeypatch):
    decoder = _vocoder_decoder(compile_vocoder=True)
    bigvgan = _LengthRecordingBigVGAN()
    runner_factory = getattr(decoder, "_get_vocoder_runner", None)
    compile_calls = []

    assert runner_factory is not None

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(indextts2_s2mel_decoder.torch, "compile", fake_compile)

    first = runner_factory(
        bigvgan,
        torch.device("cuda"),
        torch.bfloat16,
    )
    second = runner_factory(
        bigvgan,
        torch.device("cuda"),
        torch.bfloat16,
    )
    first(torch.zeros(1, 80, 137))
    second(torch.zeros(1, 80, 221))

    assert first is second
    assert bigvgan.input_lengths == [137, 221]
    assert len(compile_calls) == 1
    assert compile_calls[0][0] == bigvgan.forward
    assert compile_calls[0][1] == {
        "mode": "default",
        "fullgraph": False,
        "dynamic": True,
    }


def test_bigvgan_vocodes_cropped_mels_one_request_at_a_time_in_order():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    vocoder = _OrderRecordingBigVGAN()
    mels = [
        torch.full((2, 3), 0.1),
        torch.full((2, 2), 0.2),
        torch.full((2, 4), 0.3),
    ]

    wavs = decoder._vocode_mels(
        mels=mels,
        vocode=vocoder,
        voc_dtype=torch.float32,
    )

    assert [length for length, _ in vocoder.calls] == [3, 2, 4]
    assert [marker for _, marker in vocoder.calls] == pytest.approx([0.1, 0.2, 0.3])
    assert [tuple(wav.shape) for wav in wavs] == [(12,), (8,), (16,)]
    assert [float(wav[0]) for wav in wavs] == pytest.approx([0.1, 0.2, 0.3])


def test_vocoder_torch_compile_rejects_cuda_graph_combination():
    decoder = _vocoder_decoder(
        compile_vocoder=True,
        graph_vocoder=True,
    )
    runner_factory = getattr(decoder, "_get_vocoder_runner", None)

    assert runner_factory is not None
    with pytest.raises(ValueError, match="mutually exclusive"):
        runner_factory(
            _LengthRecordingBigVGAN(),
            torch.device("cuda"),
            torch.bfloat16,
        )


@pytest.mark.parametrize("recipe_name", ["indextts2_5.yaml"])
def test_v25_recipe_enables_dynamic_dit_compile(recipe_name):
    repo_root = Path(__file__).parents[4]
    recipe_path = repo_root / "vllm_omni" / "deploy" / recipe_name
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    stage1 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 1)
    overrides = stage1["hf_overrides"]

    assert overrides["s2mel_dit_cuda_graph"] is False
    assert overrides["s2mel_dit_bf16"] is True
    assert overrides["s2mel_dit_torch_compile"] is True
    assert overrides.get("s2mel_dit_torch_compile_mode") == "default"
    assert overrides.get("s2mel_vocoder_torch_compile") is False
    assert overrides.get("s2mel_vocoder_cuda_graph", False) is False


def test_v25_default_recipe_selects_validated_triton_backend_for_stage0():
    repo_root = Path(__file__).parents[4]
    recipe_path = repo_root / "vllm_omni" / "deploy" / "indextts2_5.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    stage0 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 0)

    assert stage0["attention_backend"] == "TRITON_ATTN"
    assert stage0["enable_chunked_prefill"] is False
    assert stage0["max_num_batched_tokens"] == stage0["max_model_len"]
