"""Unit tests for OmniBase and AsyncOmni profiler methods."""

from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniBaseProfiler:
    """Test suite for OmniBase profiler methods (start_profile, stop_profile)."""

    @pytest.fixture
    def mock_engine(self, mocker: MockerFixture):
        """Create a mock AsyncOmniEngine for testing."""
        engine = mocker.MagicMock()
        engine.num_stages = 3
        engine.is_alive.return_value = True
        engine.default_sampling_params_list = [mocker.MagicMock() for _ in range(3)]
        engine.get_stage_metadata.side_effect = lambda i: SimpleNamespace(
            final_output_type="text" if i == 0 else "audio",
            final_output=True,
        )
        engine.collective_rpc.return_value = [None, None, None]
        return engine

    @pytest.fixture
    def omni_base_instance(self, mock_engine, mocker: MockerFixture):
        """Create an OmniBase instance with mocked dependencies."""
        mocker.patch("vllm_omni.entrypoints.omni_base.AsyncOmniEngine", return_value=mock_engine)
        mocker.patch("vllm_omni.entrypoints.omni_base.omni_snapshot_download", side_effect=lambda x: x)
        mocker.patch("vllm_omni.entrypoints.omni_base.weakref.finalize")
        from vllm_omni.entrypoints.omni_base import OmniBase

        instance = OmniBase(model="test-model")
        return instance

    def test_start_profile_calls_collective_rpc(self, omni_base_instance, mock_engine):
        """Test that start_profile calls collective_rpc with correct arguments."""
        omni_base_instance.start_profile()

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(True, None),
            stage_ids=None,
        )

    def test_start_profile_with_prefix(self, omni_base_instance, mock_engine):
        """Test that start_profile passes profile_prefix to collective_rpc."""
        omni_base_instance.start_profile(profile_prefix="test_trace")

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(True, "test_trace"),
            stage_ids=None,
        )

    def test_start_profile_with_stages(self, omni_base_instance, mock_engine):
        """Test that start_profile passes stages to collective_rpc."""
        omni_base_instance.start_profile(stages=[0, 2])

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(True, None),
            stage_ids=[0, 2],
        )

    def test_start_profile_with_prefix_and_stages(self, omni_base_instance, mock_engine):
        """Test that start_profile passes both prefix and stages."""
        omni_base_instance.start_profile(profile_prefix="my_prefix", stages=[1])

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(True, "my_prefix"),
            stage_ids=[1],
        )

    def test_start_profile_returns_rpc_result(self, omni_base_instance, mock_engine):
        """Test that start_profile returns the result from collective_rpc."""
        expected_result = [{"stage_0": "started"}, {"stage_1": "started"}]
        mock_engine.collective_rpc.return_value = expected_result

        result = omni_base_instance.start_profile()

        assert result == expected_result

    def test_stop_profile_calls_collective_rpc(self, omni_base_instance, mock_engine):
        """Test that stop_profile calls collective_rpc with correct arguments."""
        omni_base_instance.stop_profile()

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(False, None),
            stage_ids=None,
        )

    def test_stop_profile_with_stages(self, omni_base_instance, mock_engine):
        """Test that stop_profile passes stages to collective_rpc."""
        omni_base_instance.stop_profile(stages=[0])

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(False, None),
            stage_ids=[0],
        )

    def test_stop_profile_returns_rpc_result(self, omni_base_instance, mock_engine):
        """Test that stop_profile returns the result from collective_rpc."""
        expected_result = [{"stage_0": "stopped"}, {"stage_1": "stopped"}]
        mock_engine.collective_rpc.return_value = expected_result

        result = omni_base_instance.stop_profile()

        assert result == expected_result

    def test_start_stop_profile_workflow(self, omni_base_instance, mock_engine):
        """Test a typical start/stop profiling workflow."""
        # Start profiling on specific stages
        omni_base_instance.start_profile(profile_prefix="workflow_test", stages=[0, 1])

        # Verify start was called correctly
        mock_engine.collective_rpc.assert_called_with(
            method="profile",
            args=(True, "workflow_test"),
            stage_ids=[0, 1],
        )

        # Reset mock to check stop call
        mock_engine.collective_rpc.reset_mock()

        # Stop profiling on the same stages
        omni_base_instance.stop_profile(stages=[0, 1])

        # Verify stop was called correctly
        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(False, None),
            stage_ids=[0, 1],
        )

    def test_start_profile_empty_stages_list(self, omni_base_instance, mock_engine):
        """Test that start_profile handles empty stages list."""
        omni_base_instance.start_profile(stages=[])

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(True, None),
            stage_ids=[],
        )

    def test_stop_profile_empty_stages_list(self, omni_base_instance, mock_engine):
        """Test that stop_profile handles empty stages list."""
        omni_base_instance.stop_profile(stages=[])

        mock_engine.collective_rpc.assert_called_once_with(
            method="profile",
            args=(False, None),
            stage_ids=[],
        )


class TestOmniBaseProfilerSignatureConsistency:
    """Test that profiler methods have consistent signatures with vLLM."""

    def test_start_profile_signature(self):
        """Verify start_profile has the expected signature parameters."""
        import inspect

        from vllm_omni.entrypoints.omni_base import OmniBase

        sig = inspect.signature(OmniBase.start_profile)
        params = list(sig.parameters.keys())

        # Should have: self, profile_prefix, stages
        assert "self" in params
        assert "profile_prefix" in params
        assert "stages" in params

    def test_stop_profile_signature(self):
        """Verify stop_profile has the expected signature parameters."""
        import inspect

        from vllm_omni.entrypoints.omni_base import OmniBase

        sig = inspect.signature(OmniBase.stop_profile)
        params = list(sig.parameters.keys())

        # Should have: self, stages
        assert "self" in params
        assert "stages" in params

    def test_start_profile_default_values(self):
        """Verify start_profile has correct default parameter values."""
        import inspect

        from vllm_omni.entrypoints.omni_base import OmniBase

        sig = inspect.signature(OmniBase.start_profile)

        # profile_prefix should default to None
        assert sig.parameters["profile_prefix"].default is None
        # stages should default to None
        assert sig.parameters["stages"].default is None

    def test_stop_profile_default_values(self):
        """Verify stop_profile has correct default parameter values."""
        import inspect

        from vllm_omni.entrypoints.omni_base import OmniBase

        sig = inspect.signature(OmniBase.stop_profile)

        # stages should default to None
        assert sig.parameters["stages"].default is None


class TestAsyncOmniProfilerSignatureConsistency:
    """Test that AsyncOmni profiler methods have consistent signatures."""

    def test_async_start_profile_signature(self):
        """Verify AsyncOmni.start_profile has the expected signature parameters."""
        import inspect

        from vllm_omni.entrypoints.async_omni import AsyncOmni

        sig = inspect.signature(AsyncOmni.start_profile)
        params = list(sig.parameters.keys())

        # Should have: self, profile_prefix, stages
        assert "self" in params
        assert "profile_prefix" in params
        assert "stages" in params

    def test_async_stop_profile_signature(self):
        """Verify AsyncOmni.stop_profile has the expected signature parameters."""
        import inspect

        from vllm_omni.entrypoints.async_omni import AsyncOmni

        sig = inspect.signature(AsyncOmni.stop_profile)
        params = list(sig.parameters.keys())

        # Should have: self, stages
        assert "self" in params
        assert "stages" in params

    def test_async_start_profile_is_coroutine(self):
        """Verify AsyncOmni.start_profile is an async method."""
        import inspect

        from vllm_omni.entrypoints.async_omni import AsyncOmni

        assert inspect.iscoroutinefunction(AsyncOmni.start_profile)

    def test_async_stop_profile_is_coroutine(self):
        """Verify AsyncOmni.stop_profile is an async method."""
        import inspect

        from vllm_omni.entrypoints.async_omni import AsyncOmni

        assert inspect.iscoroutinefunction(AsyncOmni.stop_profile)
