# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


def is_warmup_request(req_id: str) -> bool:
    """Recognize vLLM kernel and mixed-attention warmup requests."""
    return req_id.startswith(("_warmup_", "_v2_mixed_warmup_"))
