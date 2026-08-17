# SPDX-License-Identifier: Apache-2.0
"""Ratchet: block new model-specific Python files added under examples/.

Per issue #6260, all new Python paths under examples/ are blocked by default.
Paths that existed when this ratchet was introduced are grandfathered in the
baseline file (tools/pre_commit/examples_policy_baseline.txt).

Checks added (A), copied (C), and renamed (R) destination paths in the diff
between the PR head and its merge base. Modifications and deletions always
pass. A new path passes only if it appears in the approved exceptions list.

Lowering the baseline (by removing migrated paths) is encouraged.
Adding a new exception requires an explicit entry in APPROVED_EXCEPTIONS
with a written justification, and must be reviewed by a maintainer.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_FILE = Path(__file__).parent / "examples_policy_baseline.txt"

# Approved exceptions: paths that are genuinely model-neutral task/protocol
# runners. Each entry must include a justification comment above it.
APPROVED_EXCEPTIONS: set[str] = {
    # examples/offline_inference/x_to_video_audio/x_to_video_audio.py
    # Shared task runner for video/audio generation, not model-specific.
    "examples/offline_inference/x_to_video_audio/x_to_video_audio.py",
    # examples/offline_inference/text_to_video/text_to_video.py
    # Shared task runner for text-to-video, not model-specific.
    "examples/offline_inference/text_to_video/text_to_video.py",
    # examples/offline_inference/text_to_audio/text_to_audio.py
    # Shared task runner for text-to-audio, not model-specific.
    "examples/offline_inference/text_to_audio/text_to_audio.py",
    # examples/offline_inference/image_to_video/image_to_video.py
    # Shared task runner for image-to-video, not model-specific.
    "examples/offline_inference/image_to_video/image_to_video.py",
    # examples/offline_inference/x_to_text/x_to_text.py
    # Shared task runner for multimodal-to-text, not model-specific.
    "examples/offline_inference/x_to_text/x_to_text.py",
    # examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py
    # Generic OpenAI-compatible client, not model-specific.
    "examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py",
}

_GUIDANCE = """
Do not add new model-specific Python files under examples/.
Instead:
  - Move model defaults and behavior into production model modules or
    vllm_omni/model_extras/.
  - Use an existing shared task runner (text_to_video.py, x_to_video_audio.py,
    x_to_text.py, etc.) with model_extras or explicit config.
  - Put model commands, hardware notes, and validation evidence in docs/recipes.

If this path is a genuinely model-neutral task/protocol runner, add it to
APPROVED_EXCEPTIONS in tools/pre_commit/check_examples_policy.py with a
written justification and request maintainer review.
""".strip()


def _load_baseline() -> set[str]:
    if not BASELINE_FILE.is_file():
        return set()
    lines = BASELINE_FILE.read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def _get_merge_base(base_ref: str) -> str:
    result = subprocess.run(
        ["git", "merge-base", "HEAD", base_ref],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"check_examples_policy: could not find merge base with {base_ref}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def _get_added_paths(merge_base: str) -> list[str]:
    # --diff-filter=ACR: Added, Copied, Renamed destination paths only
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACR", merge_base, "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("check_examples_policy: git diff failed", file=sys.stderr)
        sys.exit(1)
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("examples/") and line.strip().endswith(".py")
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Ref to diff against (default: origin/main)",
    )
    # pre-commit passes staged filenames; accepted and ignored.
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args(argv)

    baseline = _load_baseline()
    merge_base = _get_merge_base(args.base_ref)
    added_paths = _get_added_paths(merge_base)

    errors: list[str] = []
    for path in added_paths:
        if path in baseline:
            continue
        if path in APPROVED_EXCEPTIONS:
            continue
        errors.append(path)

    if errors:
        listing = "\n".join(f"  {p}" for p in sorted(errors))
        print(
            f"check_examples_policy: {len(errors)} new model-specific Python "
            f"path(s) blocked under examples/:\n{listing}\n\n{_GUIDANCE}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())