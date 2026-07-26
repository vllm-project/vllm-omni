# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.platforms.npu.worker.compat import async_exponential_enabled, profiling_chunk_enabled

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(), False),
        (SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=True)), True),
        (
            SimpleNamespace(
                scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=True)),
            ),
            True,
        ),
        (
            SimpleNamespace(
                scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False)),
                profiling_chunk_config=SimpleNamespace(enabled=True),
            ),
            False,
        ),
    ],
)
def test_profiling_chunk_enabled_supports_old_and_new_layouts(config, expected):
    assert profiling_chunk_enabled(config) is expected


def test_async_exponential_defaults_off_but_preserves_legacy_flag():
    assert async_exponential_enabled(SimpleNamespace()) is False
    assert async_exponential_enabled(SimpleNamespace(enable_async_exponential=True)) is True
