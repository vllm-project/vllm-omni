#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared CI definitions: release-line gates, pipeline whitelists, and uploader constants.

Shell counterpart: ``release_line_branch.sh`` uses the same version pattern.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

PLATFORMS = ("cuda", "amd", "intel", "npu")
L23_LEVELS = frozenset({"l2", "l3"})

L2_YAML_FILES: dict[str, str] = {
    ".buildkite/cuda/test-ready.yml": "cuda",
    ".buildkite/npu/test-npu-ready.yml": "npu",
    ".buildkite/amd/test-amd-ready.yml": "amd",
    ".buildkite/intel/pipeline-intel.yml": "intel",
}

L3_YAML_FILES: dict[str, str] = {
    ".buildkite/cuda/test-merge.yml": "cuda",
    ".buildkite/amd/test-amd-merge.yml": "amd",
}

L45_YAML_FILES: dict[str, str] = {
    ".buildkite/cuda/test-nightly.yml": "cuda",
    ".buildkite/cuda/test-weekly.yml": "cuda",
    ".buildkite/amd/test-amd-nightly.yml": "amd",
    ".buildkite/amd/test-amd-weekly.yml": "amd",
    ".buildkite/npu/test-npu-nightly.yml": "npu",
}

BOOTSTRAP_STEPS_FILENAME = "bootstrap-upload-steps.yml"
BOOTSTRAP_IMAGE_BUILD_KEYS = frozenset(
    {
        "image-build",
        "image-build-a2",
        "image-build-a3",
        "image-build-a5",
        "image-build-310p",
    }
)
BOOTSTRAP_UPLOAD_IF_KEYS = {
    "upload-ready-pipeline": "ready",
    "upload-merge-pipeline": "merge",
    "upload-nightly-pipeline": "nightly",
    "upload-weekly-pipeline": "weekly",
}
E2E_GROUP_MARKER = "E2E Test"
ROOT = Path(__file__).resolve().parent.parent.parent.parent
CI_MIRROR_HARDWARES_PATH = ROOT / ".buildkite/common/ci_mirror_hardwares.yml"
CI_SOURCE_FILE_DEPENDENCIES_PATH = ROOT / ".buildkite/common/ci_source_file_dependencies.yml"
# Registry key whose prefixes bypass source filtering. Paths live in
# ci_source_file_dependencies.yml; do not attach this key to a job.
SOURCE_FILTER_FALLBACK_KEY = "source_filter_fallback"
CUDA_HF_TOKEN_ENV = "VLLM_CI_HF_TOKEN"
CUDA_HF_TOKEN_EXPORT = f'if [ -n "$${{{CUDA_HF_TOKEN_ENV}:-}}" ]; then export HF_TOKEN="$${{{CUDA_HF_TOKEN_ENV}}}"; fi'

# Release lines run the same post-merge / scheduled pipelines as main.
# Branch name is exactly X.Y.Z (for example 0.30.0). Keep this pattern identical
# in release_line_branch.sh.
RELEASE_LINE_VERSION_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+$"
_RELEASE_LINE_VERSION_RE = re.compile(RELEASE_LINE_VERSION_PATTERN)

# Bootstrap Buildkite ``if`` expressions.
# ``*_MAIN_IF``: release line (main or X.Y.Z) + env schedule.
# ``*_LABEL_IF``: PR label (and/or composed with MAIN).
# ``*_UPLOAD_IF``: full gate for uploading that child pipeline.
# Ruby regex; built from RELEASE_LINE_VERSION_PATTERN.
_RELEASE_LINE_BRANCH_IF = f'(build.branch == "main" || build.branch =~ /{RELEASE_LINE_VERSION_PATTERN}/)'
NIGHTLY_MAIN_IF = f'{_RELEASE_LINE_BRANCH_IF} && build.env("NIGHTLY") == "1"'
NIGHTLY_LABEL_IF = (
    f'({NIGHTLY_MAIN_IF}) || (build.branch != "main" && build.pull_request.labels includes "nightly-test")'
)
WEEKLY_E2E_IF = f'{_RELEASE_LINE_BRANCH_IF} && build.env("WEEKLY") == "1"'
WEEKLY_MAIN_IF = f'{_RELEASE_LINE_BRANCH_IF} && (build.env("WEEKLY") == "1" || build.env("NON_CRITICAL") == "1")'
WEEKLY_LABEL_IF = f'({WEEKLY_MAIN_IF}) || (build.branch != "main" && build.pull_request.labels includes "weekly-test")'
READY_LABEL_IF = 'build.branch != "main" && build.pull_request.labels includes "ready"'
MERGE_LABEL_IF = 'build.branch != "main" && build.pull_request.labels includes "merge-test"'
MERGE_MAIN_IF = (
    f'{_RELEASE_LINE_BRANCH_IF} && build.env("NIGHTLY") != "1" && '
    'build.env("WEEKLY") != "1" && build.env("NON_CRITICAL") != "1"'
)
READY_UPLOAD_IF = f"({WEEKLY_E2E_IF}) || ({READY_LABEL_IF})"
MERGE_UPLOAD_IF = f"({WEEKLY_E2E_IF}) || (({MERGE_MAIN_IF}) || ({MERGE_LABEL_IF}))"
BOOTSTRAP_DISABLED_IF = "false"
BOOTSTRAP_ENABLED_IF = "true"


def is_release_line_branch(branch: str | None = None) -> bool:
    """Return True for ``main`` and for a branch named exactly ``X.Y.Z``."""
    if branch is None:
        branch = os.environ.get("BUILDKITE_BRANCH", "")
    return branch == "main" or _RELEASE_LINE_VERSION_RE.fullmatch(branch) is not None
