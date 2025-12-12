# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

try:
    from vllm.logger import init_logger as _vllm_init_logger
except Exception:  # pragma: no cover - optional dependency
    _vllm_init_logger = None


def get_connector_logger(name: str) -> logging.Logger:
    """Return a logger preferring vLLM's init_logger when available."""
    return _vllm_init_logger(name) if _vllm_init_logger else logging.getLogger(name)
