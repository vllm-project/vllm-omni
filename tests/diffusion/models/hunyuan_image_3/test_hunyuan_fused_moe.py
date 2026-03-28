# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HunyuanFusedMoE (Support HunyuanImage3 Diffusion Model, 5a779b4)."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSetForwardContextNumTokens:
    """Test _set_forward_context_num_tokens defensive fix for vLLM 0.18.0."""

    def test_sets_num_tokens_when_context_available(self, mocker):
        """num_tokens should be set on ForwardContext when available."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        mock_ctx = mocker.MagicMock()
        del mock_ctx.in_profile_run  # simulate missing attr
        mocker.patch.object(hunyuan_moe._vllm_fc, "is_forward_context_available", return_value=True)
        mocker.patch.object(hunyuan_moe._vllm_fc, "get_forward_context", return_value=mock_ctx)

        hunyuan_moe._set_forward_context_num_tokens(1024)

        assert mock_ctx.num_tokens == 1024
        assert mock_ctx.in_profile_run is False

    def test_sets_in_profile_run_only_if_missing(self, mocker):
        """in_profile_run should not be overwritten if already set."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        mock_ctx = mocker.MagicMock()
        mock_ctx.in_profile_run = True  # already set
        mocker.patch.object(hunyuan_moe._vllm_fc, "is_forward_context_available", return_value=True)
        mocker.patch.object(hunyuan_moe._vllm_fc, "get_forward_context", return_value=mock_ctx)

        hunyuan_moe._set_forward_context_num_tokens(512)

        assert mock_ctx.num_tokens == 512
        assert mock_ctx.in_profile_run is True  # not overwritten

    def test_noop_when_context_unavailable(self, mocker):
        """Should do nothing when ForwardContext is not available."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        mocker.patch.object(hunyuan_moe._vllm_fc, "is_forward_context_available", return_value=False)
        mock_get = mocker.patch.object(hunyuan_moe._vllm_fc, "get_forward_context")

        hunyuan_moe._set_forward_context_num_tokens(256)

        mock_get.assert_not_called()


class TestHunyuanFusedMoEPlatformDispatch:
    """Test platform dispatch via platform qualname hooks."""

    def test_default_platform_uses_default_impl_qualname(self, mocker):
        """HunyuanFusedMoE should resolve the impl class from the platform hook."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        mock_platform = mocker.MagicMock()
        mock_platform.get_diffusion_model_impl_qualname.return_value = (
            "vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe.HunyuanFusedMoEDefault"
        )

        mocker.patch.object(
            hunyuan_moe,
            "current_omni_platform",
            mock_platform,
        )
        mock_resolve = mocker.patch.object(hunyuan_moe, "resolve_obj_by_qualname")
        mock_impl = mocker.MagicMock()
        mock_resolve.return_value = mock_impl

        from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe import (
            HunyuanFusedMoE,
        )

        HunyuanFusedMoE(prefix="")

        mock_platform.prepare_diffusion_op_runtime.assert_called_once_with("hunyuan_fused_moe")
        mock_platform.get_diffusion_model_impl_qualname.assert_called_once_with("hunyuan_fused_moe")
        mock_resolve.assert_called_once_with(
            "vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe.HunyuanFusedMoEDefault"
        )
        mock_impl.assert_called_once_with(prefix="")


class TestHunyuanFusedMoEFactory:
    """Test HunyuanFusedMoE factory __new__ and make_expert_params_mapping delegation."""

    def test_new_delegates_to_impl_class(self, mocker):
        """HunyuanFusedMoE(prefix=..., **kwargs) should instantiate and return impl instance."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        class MockImpl:
            def __init__(self, *, prefix: str = "", **kwargs):
                self.prefix = prefix
                self.kwargs = kwargs

        mock_platform = mocker.MagicMock()
        mock_platform.get_diffusion_model_impl_qualname.return_value = "mock.impl.Qualname"
        mocker.patch.object(hunyuan_moe, "current_omni_platform", mock_platform)

        mock_impl_class = mocker.MagicMock(return_value=MockImpl(prefix="test", a=1))
        mocker.patch.object(hunyuan_moe, "resolve_obj_by_qualname", return_value=mock_impl_class)

        from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe import (
            HunyuanFusedMoE,
        )

        result = HunyuanFusedMoE(prefix="test", a=1)

        assert isinstance(result, MockImpl)
        assert result.prefix == "test"
        assert result.kwargs == {"a": 1}
        mock_platform.prepare_diffusion_op_runtime.assert_called_once_with("hunyuan_fused_moe")
        mock_platform.get_diffusion_model_impl_qualname.assert_called_once_with("hunyuan_fused_moe")
        mock_impl_class.assert_called_once_with(prefix="test", a=1)

    def test_make_expert_params_mapping_delegates_to_impl(self, mocker):
        """make_expert_params_mapping should delegate to impl class method."""
        import vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe as hunyuan_moe

        expected_mapping = [("a", "b", 0, "c")]
        mock_platform = mocker.MagicMock()
        mock_platform.get_diffusion_model_impl_qualname.return_value = "mock.impl.Qualname"
        mocker.patch.object(hunyuan_moe, "current_omni_platform", mock_platform)

        mock_impl_class = mocker.MagicMock()
        mock_impl_class.make_expert_params_mapping = mocker.MagicMock(return_value=expected_mapping)
        mocker.patch.object(hunyuan_moe, "resolve_obj_by_qualname", return_value=mock_impl_class)

        from vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe import (
            HunyuanFusedMoE,
        )

        result = HunyuanFusedMoE.make_expert_params_mapping(
            model=None,
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=4,
            num_redundant_experts=0,
        )

        assert result == expected_mapping
        mock_platform.get_diffusion_model_impl_qualname.assert_called_once_with("hunyuan_fused_moe")
        mock_impl_class.make_expert_params_mapping.assert_called_once_with(
            None,
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=4,
            num_redundant_experts=0,
        )
