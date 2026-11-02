# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

if os.environ.get("WAN_TRACE_MODE"):
    import wan_instrument  # noqa: F401
