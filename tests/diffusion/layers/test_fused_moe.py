# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the diffusion FusedMoE adapter."""

import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSetForwardContextNumTokens:
    """Test _set_forward_context_num_tokens defensive fix for vLLM 0.18.0."""

    def test_sets_num_tokens_when_context_available(self, mocker):
        """num_tokens should be set on ForwardContext when available."""
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        mock_ctx = mocker.MagicMock()
        del mock_ctx.in_profile_run  # simulate missing attr
        mocker.patch.object(fused_moe._vllm_fc, "is_forward_context_available", return_value=True)
        mocker.patch.object(fused_moe._vllm_fc, "get_forward_context", return_value=mock_ctx)

        fused_moe._set_forward_context_num_tokens(1024)

        assert mock_ctx.num_tokens == 1024
        assert mock_ctx.in_profile_run is False

    def test_sets_in_profile_run_only_if_missing(self, mocker):
        """in_profile_run should not be overwritten if already set."""
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        mock_ctx = mocker.MagicMock()
        mock_ctx.in_profile_run = True  # already set
        mocker.patch.object(fused_moe._vllm_fc, "is_forward_context_available", return_value=True)
        mocker.patch.object(fused_moe._vllm_fc, "get_forward_context", return_value=mock_ctx)

        fused_moe._set_forward_context_num_tokens(512)

        assert mock_ctx.num_tokens == 512
        assert mock_ctx.in_profile_run is True  # not overwritten

    def test_noop_when_context_unavailable(self, mocker):
        """Should do nothing when ForwardContext is not available."""
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        mocker.patch.object(fused_moe._vllm_fc, "is_forward_context_available", return_value=False)
        mock_get = mocker.patch.object(fused_moe._vllm_fc, "get_forward_context")

        fused_moe._set_forward_context_num_tokens(256)

        mock_get.assert_not_called()


class TestSetForwardContextDPMetadata:
    """Test diffusion MoE DP metadata setup for vLLM 0.24 dispatch."""

    def test_sets_dp_metadata_from_vllm_dp_group(self, mocker):
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        mock_ctx = mocker.MagicMock()
        mock_ctx.dp_metadata = None
        fake_group = mocker.MagicMock()
        fake_group.world_size = 2
        fake_group.cpu_group = object()

        def _all_gather_object(output, obj, group):
            assert group is fake_group.cpu_group
            output[:] = [obj, obj + 1]

        mocker.patch.object(fused_moe._vllm_fc, "is_forward_context_available", return_value=True)
        mocker.patch.object(fused_moe._vllm_fc, "get_forward_context", return_value=mock_ctx)
        mocker.patch.object(fused_moe._vllm_ps, "_DP", fake_group, create=True)
        mocker.patch.object(fused_moe.torch.distributed, "all_gather_object", side_effect=_all_gather_object)

        fused_moe._set_forward_context_dp_metadata(1024)

        assert mock_ctx.dp_metadata is not None
        assert mock_ctx.dp_metadata.num_tokens_across_dp_cpu.tolist() == [1024, 1025]

    def test_keeps_existing_dp_metadata(self, mocker):
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        existing_metadata = object()
        mock_ctx = mocker.MagicMock()
        mock_ctx.dp_metadata = existing_metadata
        mocker.patch.object(fused_moe._vllm_fc, "is_forward_context_available", return_value=True)
        mocker.patch.object(fused_moe._vllm_fc, "get_forward_context", return_value=mock_ctx)
        mock_all_gather = mocker.patch.object(fused_moe.torch.distributed, "all_gather_object")

        fused_moe._set_forward_context_dp_metadata(1024)

        assert mock_ctx.dp_metadata is existing_metadata
        mock_all_gather.assert_not_called()


class TestFusedMoEFactory:
    """Test FusedMoE factory __new__ and mapping delegation."""

    def test_new_returns_vllm_runner_and_registers_hooks(self, mocker, monkeypatch):
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        mock_platform = mocker.MagicMock()
        mocker.patch.object(fused_moe, "current_omni_platform", mock_platform)

        mock_runner = mocker.MagicMock()
        mock_vllm_fused_moe = mocker.MagicMock(return_value=mock_runner)
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.fused_moe",
            SimpleNamespace(FusedMoEFactory=mock_vllm_fused_moe),
        )

        result = fused_moe.FusedMoE(prefix="test", a=1, reduce_results=True)

        assert result is mock_runner
        mock_platform.prepare_diffusion_op_runtime.assert_called_once_with("fused_moe")
        mock_vllm_fused_moe.assert_called_once_with(prefix="test", a=1)
        mock_runner.register_forward_pre_hook.assert_called_once_with(
            mocker.ANY,
            with_kwargs=True,
        )
        mock_platform.register_additional_diffusion_fused_moe_hooks.assert_called_once_with(mock_runner)

    def test_num_tokens_pre_hook_uses_kwargs_hidden_states(self, mocker, monkeypatch):
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        captured_hook = None

        class MockRunner:
            def register_forward_pre_hook(self, hook, *, with_kwargs):
                nonlocal captured_hook
                assert with_kwargs is True
                captured_hook = hook

        mock_runner = MockRunner()
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.fused_moe",
            SimpleNamespace(FusedMoEFactory=mocker.MagicMock(return_value=mock_runner)),
        )
        mocker.patch.object(fused_moe, "current_omni_platform", mocker.MagicMock())
        mock_set_num_tokens = mocker.patch.object(fused_moe, "_set_forward_context_num_tokens")
        mock_set_dp_metadata = mocker.patch.object(fused_moe, "_set_forward_context_dp_metadata")

        fused_moe.FusedMoE(prefix="")
        assert captured_hook is not None

        hidden_states = torch.empty(17, 32)
        captured_hook(mock_runner, (), {"hidden_states": hidden_states})

        mock_set_num_tokens.assert_called_once_with(17)
        mock_set_dp_metadata.assert_called_once_with(17)

    def test_num_tokens_pre_hook_uses_positional_hidden_states(self, mocker, monkeypatch):
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        captured_hook = None

        class MockRunner:
            def register_forward_pre_hook(self, hook, *, with_kwargs):
                nonlocal captured_hook
                assert with_kwargs is True
                captured_hook = hook

        mock_runner = MockRunner()
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.fused_moe",
            SimpleNamespace(FusedMoEFactory=mocker.MagicMock(return_value=mock_runner)),
        )
        mocker.patch.object(fused_moe, "current_omni_platform", mocker.MagicMock())
        mock_set_num_tokens = mocker.patch.object(fused_moe, "_set_forward_context_num_tokens")
        mock_set_dp_metadata = mocker.patch.object(fused_moe, "_set_forward_context_dp_metadata")

        fused_moe.FusedMoE(prefix="")
        assert captured_hook is not None

        hidden_states = torch.empty(23, 32)
        captured_hook(mock_runner, (hidden_states,), {})

        mock_set_num_tokens.assert_called_once_with(23)
        mock_set_dp_metadata.assert_called_once_with(23)

    def test_make_expert_params_mapping_delegates_to_vllm_function(self, mocker, monkeypatch):
        import vllm_omni.diffusion.layers.fused_moe as fused_moe

        expected_mapping = [("a", "b", 0, "c")]
        mock_mapping_fn = mocker.MagicMock(return_value=expected_mapping)
        monkeypatch.setitem(
            sys.modules,
            "vllm.model_executor.layers.fused_moe",
            SimpleNamespace(fused_moe_make_expert_params_mapping=mock_mapping_fn),
        )

        result = fused_moe.FusedMoE.make_expert_params_mapping(
            model=None,
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=4,
            num_redundant_experts=0,
        )

        assert result == expected_mapping
        mock_mapping_fn.assert_called_once_with(
            None,
            ckpt_gate_proj_name="gate",
            ckpt_down_proj_name="down",
            ckpt_up_proj_name="up",
            num_experts=4,
            num_redundant_experts=0,
        )
