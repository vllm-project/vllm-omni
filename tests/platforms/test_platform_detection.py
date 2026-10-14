# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import logging

import pytest

import vllm_omni.platforms as platforms

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_failing_oot_plugin_is_logged_and_falls_back(monkeypatch, caplog):
    plugin_name = "raising_oot_plugin"

    def raising_plugin():
        raise RuntimeError("injected platform detection failure")

    monkeypatch.setattr(platforms, "builtin_omni_platform_plugins", {})
    monkeypatch.setattr(
        platforms,
        "load_omni_plugins_by_group",
        lambda _group: {plugin_name: raising_plugin},
    )

    target_logger = logging.getLogger(platforms.__name__)
    target_logger.addHandler(caplog.handler)
    previous_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        platform_cls = platforms.resolve_current_omni_platform_cls_qualname()
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(previous_level)

    assert platform_cls == "vllm_omni.platforms.interface.UnspecifiedOmniPlatform"
    assert plugin_name in caplog.text
    assert "RuntimeError: injected platform detection failure" in caplog.text
