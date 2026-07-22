# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Connector-specific exceptions."""


class ConnectorUnavailableError(RuntimeError):
    """Raised when a connector cannot run in the current environment."""


class MooncakeUnavailableError(ConnectorUnavailableError):
    """Raised when Mooncake Transfer Engine support is unavailable."""
