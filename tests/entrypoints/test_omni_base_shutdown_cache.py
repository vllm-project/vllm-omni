# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniBase shutdown dispatching CosyVoice3 cache cleanup.

``OmniBase._shutdown_base`` releases model-side process-wide caches after the
engine stops, but only for model modules that were actually imported — it looks
them up via ``sys.modules`` so shutdown never forces a heavy optional import.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.omni_base import OmniBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_COSYVOICE3_MOD = "vllm_omni.model_executor.models.cosyvoice3.cosyvoice3"


def _fake_self(engine):
    """Minimal stand-in carrying only the attributes _shutdown_base reads."""
    return SimpleNamespace(_shutdown_called=False, _weak_finalizer=None, engine=engine)


class TestShutdownCacheDispatch:
    def test_calls_cleanup_when_module_imported(self, mocker):
        engine = mocker.MagicMock()
        cleanup = mocker.MagicMock()
        fake_mod = SimpleNamespace(clear_process_runtime_caches=cleanup)
        mocker.patch.dict(sys.modules, {_COSYVOICE3_MOD: fake_mod})

        OmniBase._shutdown_base(_fake_self(engine))

        engine.shutdown.assert_called_once()
        cleanup.assert_called_once_with()

    def test_skips_when_module_not_imported(self, mocker):
        engine = mocker.MagicMock()
        # Ensure the module is absent; shutdown must not import it.
        mocker.patch.dict(sys.modules)
        sys.modules.pop(_COSYVOICE3_MOD, None)

        OmniBase._shutdown_base(_fake_self(engine))

        engine.shutdown.assert_called_once()
        assert _COSYVOICE3_MOD not in sys.modules

    def test_cleanup_failure_does_not_propagate(self, mocker):
        engine = mocker.MagicMock()
        cleanup = mocker.MagicMock(side_effect=RuntimeError("cleanup boom"))
        fake_mod = SimpleNamespace(clear_process_runtime_caches=cleanup)
        mocker.patch.dict(sys.modules, {_COSYVOICE3_MOD: fake_mod})

        # Must not raise even though the model cleanup blew up.
        OmniBase._shutdown_base(_fake_self(engine))
        cleanup.assert_called_once_with()

    def test_second_call_is_a_noop(self, mocker):
        engine = mocker.MagicMock()
        cleanup = mocker.MagicMock()
        fake_mod = SimpleNamespace(clear_process_runtime_caches=cleanup)
        mocker.patch.dict(sys.modules, {_COSYVOICE3_MOD: fake_mod})

        slf = _fake_self(engine)
        OmniBase._shutdown_base(slf)
        OmniBase._shutdown_base(slf)

        # _shutdown_called guard prevents a second engine.shutdown / cleanup.
        engine.shutdown.assert_called_once()
        cleanup.assert_called_once_with()
