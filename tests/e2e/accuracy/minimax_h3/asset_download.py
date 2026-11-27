# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Asset download used by the MiniMax-H3 accuracy tests.

Deliberately free of model and engine imports: the retry policy is exercised
by ``test_asset_download_retry.py`` in the CPU lane, which must not pull the
serving stack into a unit test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests
import urllib3


def _asset_download_retry() -> urllib3.util.Retry:
    """Retry transient connection failures and 5xx, never a 4xx."""
    return urllib3.util.Retry(
        total=4,
        connect=4,
        read=2,
        backoff_factor=2,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET",),
        # urllib3 additionally retries 413/429 responses carrying Retry-After;
        # keep retries to the explicit 5xx list above so 4xx fails fast.
        respect_retry_after_header=False,
    )


def download_reference_asset(
    output_dir: Path,
    *,
    filename: str,
    url: str,
    sha256: str,
    label: str,
) -> Path:
    """Download one conditioning asset and gate it on its pinned SHA256.

    Conditioning inputs are hosted on cdn.hailuoai.com, whose DNS resolution
    is intermittent from CI runners (#6295, #10661). Transient connection
    failures and 5xx are retried; the digest still gates content integrity.
    """
    reference_path = output_dir / filename
    download_url = url.replace("/blob/", "/resolve/", 1)
    session = requests.Session()
    retry_adapter = requests.adapters.HTTPAdapter(max_retries=_asset_download_retry())
    session.mount("https://", retry_adapter)
    # Production assets are https; mounting http keeps the policy identical
    # for the local-server CPU regression.
    session.mount("http://", retry_adapter)
    response = session.get(download_url, timeout=120)
    response.raise_for_status()
    content = response.content
    digest = hashlib.sha256(content).hexdigest()
    assert digest == sha256, f"Official MiniMax H3 {label} reference changed: got {digest}, expected {sha256}"
    reference_path.write_bytes(content)
    return reference_path
