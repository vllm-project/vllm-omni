# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class MareyVAEInitializationError(RuntimeError):
    """Raised when Marey cannot serve decoded video because the VAE is unavailable."""
