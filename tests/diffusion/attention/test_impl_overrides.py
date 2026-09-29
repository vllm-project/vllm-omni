# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.attention import layer
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import FastVideoVSABackend, FastVideoVSAImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.data import AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _SpecializedImpl(FastVideoVSAImpl):
    pass


@pytest.fixture
def local_attention(monkeypatch):
    monkeypatch.setattr(layer, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_specialization_preserves_selected_backend_and_options(mocker, local_attention, explicit, override):
    spec = AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=7) if explicit else None
    select = mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(FastVideoVSABackend, spec))
    attention = layer.Attention(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        role="video.self",
        role_category="self",
        qkv_layout="BSND",
        impl_overrides={"FASTVIDEO_VSA": _SpecializedImpl} if override else None,
    )

    assert attention.attn_backend is FastVideoVSABackend
    assert type(attention.attention) is (_SpecializedImpl if override else FastVideoVSAImpl)
    assert attention.attention.topk == (7 if explicit else 64)
    assert attention.attention.qkv_layout == "BSND"
    assert attention.backend_pref == "FASTVIDEO_VSA"
    assert attention.backend_explicit is explicit
    assert attention.attn_spec is spec
    assert isinstance(attention.parallel_strategy, NoParallelAttention)
    select.assert_called_once_with(
        role="video.self", head_size=8, attention_config=None, role_category="self", allow_trtllm_default=False
    )


def test_unselected_specialization_does_not_replace_dense_backend(mocker, local_attention):
    mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(SDPABackend, None))
    attention = layer.Attention(
        num_heads=2,
        head_size=8,
        softmax_scale=8**-0.5,
        causal=False,
        impl_overrides={"FASTVIDEO_VSA": _SpecializedImpl},
    )
    assert attention.attn_backend is SDPABackend
    assert type(attention.attention) is SDPABackend.get_impl_cls()


def test_specialization_does_not_bypass_selection_errors(mocker, local_attention):
    mocker.patch.object(layer, "get_attn_backend_for_role", side_effect=ImportError("optional kernel unavailable"))
    with pytest.raises(ImportError, match="optional kernel unavailable"):
        layer.Attention(
            num_heads=2,
            head_size=8,
            softmax_scale=8**-0.5,
            causal=False,
            impl_overrides={"FASTVIDEO_VSA": _SpecializedImpl},
        )


def test_specialization_must_extend_selected_implementation(mocker, local_attention):
    mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(FastVideoVSABackend, None))
    with pytest.raises(TypeError, match="must subclass the selected implementation"):
        layer.Attention(
            num_heads=2,
            head_size=8,
            softmax_scale=8**-0.5,
            causal=False,
            impl_overrides={"FASTVIDEO_VSA": SDPABackend.get_impl_cls()},
        )


def test_platform_backend_capabilities_survive_model_specialization(mocker, local_attention):
    from vllm_omni.diffusion.models.minimax_h3.attention.fastvideo_h3 import MiniMaxH3VSAImpl

    class PlatformBackend(FastVideoVSABackend):
        @classmethod
        def supports_packed_mask_free(cls):
            return False

        @staticmethod
        def get_supported_head_sizes():
            return [128]

    # A registry/platform may refine capabilities without changing the
    # implementation. The model must not replace those capability decisions.
    mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(PlatformBackend, None))
    attention = layer.Attention(
        num_heads=2,
        head_size=128,
        softmax_scale=128**-0.5,
        causal=False,
        impl_overrides={"FASTVIDEO_VSA": MiniMaxH3VSAImpl},
    )
    assert attention.attn_backend is PlatformBackend
    assert not attention.attn_backend.supports_packed_mask_free()
    assert attention.attn_backend.get_supported_head_sizes() == [128]
    assert type(attention.attention) is MiniMaxH3VSAImpl


def test_incompatible_platform_implementation_is_not_discarded(mocker, local_attention):
    class PlatformImpl(FastVideoVSAImpl):
        pass

    class PlatformBackend(FastVideoVSABackend):
        @staticmethod
        def get_impl_cls():
            return PlatformImpl

    mocker.patch.object(layer, "get_attn_backend_for_role", return_value=(PlatformBackend, None))
    with pytest.raises(TypeError, match="selected implementation .*PlatformImpl"):
        layer.Attention(
            num_heads=2,
            head_size=8,
            softmax_scale=8**-0.5,
            causal=False,
            impl_overrides={"FASTVIDEO_VSA": _SpecializedImpl},
        )
