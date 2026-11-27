# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility for checkpoint processors with legacy string registration."""

from contextlib import contextmanager
from threading import RLock

from transformers import AutoImageProcessor

_PROCESSOR_LOAD_LOCK = RLock()


@contextmanager
def legacy_image_processor_registration():
    """Ignore redundant checkpoint registrations while their auto_map loads.

    Transformers now expects a config class rather than a processor name.
    Restore the public method even when remote-code loading fails.
    """
    with _PROCESSOR_LOAD_LOCK:
        original_register = AutoImageProcessor.register

        def register(config_class, *args, **kwargs):
            if config_class in ("MiniCPMVImageProcessor", "RealtimeVenusOmniImageProcessor"):
                return None
            return original_register(config_class, *args, **kwargs)

        AutoImageProcessor.register = staticmethod(register)
        try:
            yield
        finally:
            AutoImageProcessor.register = staticmethod(original_register)
