# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Metadata ownership for the H3 chunked reverse-output schedule."""

import os

H3_VSA_O_BUNDLE_ENV = "VLLM_OMNI_FASTVIDEO_VSA_O_BUNDLE"
H3_VSA_O_BUNDLE_ACTIVE_KEY = "vsa_h3_o_bundle_active"
H3_VSA_O_BUNDLE_STATE_KEY = "vsa_h3_o_bundle_state"


def h3_vsa_o_bundle_enabled() -> bool:
    """Return whether reverse-O compact-coarse piggybacking was requested."""
    raw = os.environ.get(H3_VSA_O_BUNDLE_ENV, "0").strip()
    if raw not in {"0", "1"}:
        raise ValueError(f"{H3_VSA_O_BUNDLE_ENV} must be exactly '0' or '1', got {raw!r}")
    return raw == "1"
