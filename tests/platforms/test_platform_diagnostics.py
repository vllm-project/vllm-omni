# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for platform diagnostic infrastructure.

These tests ensure that when no hardware accelerator is detected, the
UnspecifiedOmniPlatform error messages contain actionable details about
*why* each platform was skipped — not just a generic "no platform found".

The key invariant: built-in plugins internally catch exceptions (to keep
normal multi-platform startup quiet), so the diagnostics must collect
errors via a side-channel (:data:`_plugin_error_details`) rather than
relying on exceptions propagating through the resolver.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestBuildUnspecifiedDiagnostics:
    """Unit tests for :func:`vllm_omni.platforms._build_unspecified_diagnostics`."""

    @staticmethod
    def _call(errors: dict[str, str]) -> str:
        from vllm_omni.platforms import _build_unspecified_diagnostics

        return _build_unspecified_diagnostics(errors)

    def test_empty_errors_still_produces_helpful_message(self) -> None:
        """Even without per-platform details the message must guide the user."""
        msg = self._call({})
        assert "No hardware accelerator detected" in msg
        assert "nvidia-smi" in msg
        assert "rocm-smi" in msg

    def test_single_error_appears_in_output(self) -> None:
        """Each recorded error must mention both the platform name and failure reason."""
        msg = self._call({"cuda": "No module named 'pynvml'"})
        assert "cuda" in msg
        assert "No module named 'pynvml'" in msg

    def test_multiple_errors_all_listed(self) -> None:
        """When several platforms fail, all of them must appear in the message."""
        errors = {
            "cuda": "pynvml.nvmlInit failed: Driver Not Found",
            "rocm": "No module named 'amdsmi'",
            "npu": "torch.npu.is_available() returned False",
        }
        msg = self._call(errors)
        for name, detail in errors.items():
            assert name in msg, f"{name!r} missing from diagnostics"
            assert detail in msg, f"{detail!r} missing from diagnostics"


class TestMakeUnsupportedStub:
    """Tests for :func:`vllm_omni.platforms.interface._make_unsupported_stub`."""

    @staticmethod
    def _make_stub(method_name: str):
        from vllm_omni.platforms.interface import _make_unsupported_stub

        return _make_unsupported_stub(method_name)

    def test_stub_name_and_qualname_are_set(self) -> None:
        stub = self._make_stub("set_device")
        assert stub.__name__ == "set_device"
        assert stub.__qualname__ == "UnspecifiedOmniPlatform.set_device"

    def test_stub_raises_not_implemented_with_method_name(self) -> None:
        stub = self._make_stub("synchronize")
        with pytest.raises(NotImplementedError, match="synchronize is not implemented"):
            stub.__func__(None)

    def test_stub_includes_diagnostics_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When _platform_diagnostics is populated, the error message includes it."""
        import vllm_omni.platforms

        monkeypatch.setattr(
            vllm_omni.platforms,
            "_platform_diagnostics",
            "No hardware accelerator detected.\n  - cuda: nvmlInit failed.",
        )
        stub = self._make_stub("get_free_memory")
        with pytest.raises(NotImplementedError, match="nvmlInit failed"):
            stub.__func__(None)

    def test_unspecified_error_suffix_has_key_info(self) -> None:
        """The fallback suffix must mention concrete troubleshooting steps."""
        from vllm_omni.platforms.interface import _UNSPECIFIED_ERROR_SUFFIX

        assert "UnspecifiedOmniPlatform" in _UNSPECIFIED_ERROR_SUFFIX
        assert "nvidia-smi" in _UNSPECIFIED_ERROR_SUFFIX
        assert "pynvml" in _UNSPECIFIED_ERROR_SUFFIX
        assert "CPU-only" in _UNSPECIFIED_ERROR_SUFFIX


class TestPluginErrorSideChannel:
    """Tests that the side-channel dict is populated / cleared properly.

    These verify the mechanism that records *why* each built-in plugin
    couldn't activate.  The key invariant: ImportError means the
    hardware stack isn't installed (expected, not recorded); other
    exceptions mean the stack IS installed but the probe failed (real
    error, recorded for diagnostics).
    """

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _clear_side_channel() -> None:
        from vllm_omni.platforms import _plugin_error_details

        _plugin_error_details.clear()

    def _assert_side_channel_empty(self) -> None:
        from vllm_omni.platforms import _plugin_error_details

        assert not _plugin_error_details, f"Side-channel must be empty; got {dict(_plugin_error_details)}"

    # -- import errors (expected: not recorded) ---------------------------

    def test_import_error_not_recorded_for_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pynvml not installed → silently return None, don't record."""
        from vllm_omni.platforms import cuda_omni_platform_plugin

        self._clear_side_channel()

        import vllm.utils.import_utils  # noqa: F401

        def _raise_import_error():
            raise ImportError("No module named 'pynvml'")

        monkeypatch.setattr(vllm.utils.import_utils, "import_pynvml", _raise_import_error)

        result = cuda_omni_platform_plugin()
        assert result is None
        self._assert_side_channel_empty()

    def test_import_error_not_recorded_for_rocm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """amdsmi not installed → silently return None, don't record."""
        from vllm_omni.platforms import rocm_omni_platform_plugin

        self._clear_side_channel()

        # Replace the 'import amdsmi' statement to raise ImportError.
        import builtins

        _original_import = builtins.__import__

        def _block_amdsmi(name, *args, **kwargs):
            if name == "amdsmi":
                raise ImportError("No module named 'amdsmi'")
            return _original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_amdsmi)

        result = rocm_omni_platform_plugin()
        assert result is None
        self._assert_side_channel_empty()

    # -- real errors (recorded) -------------------------------------------

    def test_init_error_recorded_for_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pynvml installs fine but nvmlInit fails → record the error."""
        from vllm_omni.platforms import _plugin_error_details, cuda_omni_platform_plugin

        self._clear_side_channel()

        import vllm.utils.import_utils  # noqa: F401

        def _fake_import_pynvml():
            """Return a mock pynvml whose nvmlInit raises."""

            class _MockPynvml:
                @staticmethod
                def nvmlInit():
                    raise RuntimeError("NVML Shared Library Not Found")

                @staticmethod
                def nvmlShutdown():
                    pass

                @staticmethod
                def nvmlDeviceGetCount():
                    return 0

            return _MockPynvml()

        monkeypatch.setattr(vllm.utils.import_utils, "import_pynvml", _fake_import_pynvml)

        result = cuda_omni_platform_plugin()
        assert result is None, "Plugin must return None when init fails"
        assert "cuda" in _plugin_error_details, (
            f"Side-channel must contain 'cuda'; got {list(_plugin_error_details.keys())}"
        )
        assert "NVML Shared Library Not Found" in _plugin_error_details["cuda"], (
            "Error message must contain the real failure reason"
        )

    # -- resolver lifecycle -----------------------------------------------

    def test_side_channel_cleared_on_each_resolution(self) -> None:
        """resolve_current_omni_platform_cls_qualname must clear stale entries."""
        # Use attribute access (not ``from ... import``) because the resolver
        # rebinds _plugin_error_details to a fresh dict via ``global ... = {}``,
        # which a local reference from ``from ... import`` would not see.
        import vllm_omni.platforms

        vllm_omni.platforms._plugin_error_details["stale"] = "should be cleared"
        vllm_omni.platforms.resolve_current_omni_platform_cls_qualname()
        assert "stale" not in vllm_omni.platforms._plugin_error_details, (
            "Side-channel dict must be cleared at the start of each resolution"
        )
