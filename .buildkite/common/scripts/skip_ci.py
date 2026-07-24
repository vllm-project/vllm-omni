#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared Buildkite skip-ci and CI-YAML level targeting logic.

``resolve_ci_decision(changed_files, ...)`` buckets explicit file lists (tests, dry-run).
``resolve_ci_context_from_git()`` resolves git diff once and returns decision + changed files
for Buildkite bootstrap / upload scripts.

CLI (for bootstrap scripts; exit code is the signal):
  python3 skip_ci.py gate <platform> <level>
      Exit 0 when bootstrap should stop (skip-all, or that L2/L3 target is off).
      Prints ``skip-all`` or ``skip-l23`` on stdout; logs the decision once.
  python3 skip_ci.py check-skip-all
      Exit 0 when only docs/skip-mark changes → bootstrap skips entire CI upload.
  python3 skip_ci.py check-skip-l2-l3 <platform> <level>
      Exit 0 when that L2/L3 pipeline should be skipped (CI config YAML-only diffs).
  python3 skip_ci.py print-annotate
      Print decision message for Buildkite annotation.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

LOG = "[skip-ci]"
ROOT = Path(__file__).resolve().parent.parent.parent.parent

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
    ".buildkite/npu/test-npu-nightly.yml": "npu",
}


def _empty_platform_buckets() -> dict[str, list[str]]:
    return {platform: [] for platform in PLATFORMS}


@dataclass(frozen=True)
class CiDevice:
    """Per-platform / per-level skip flags for yaml-gated bootstrap (``skip_l2_l3``)."""

    skip_cuda_l2: bool = False
    skip_cuda_l3: bool = False
    skip_amd_l2: bool = False
    skip_amd_l3: bool = False
    skip_intel_l2: bool = False
    skip_intel_l3: bool = False
    skip_npu_l2: bool = False
    skip_npu_l3: bool = False

    def is_skip(self, platform: str, level: str) -> bool:
        return bool(getattr(self, f"skip_{platform}_{level}", True))


@dataclass
class DiffBuckets:
    """Changed files grouped by doc / skip-mark / CI yaml / other."""

    docs: list[str] = field(default_factory=list)
    skip_tests: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    l2: dict[str, list[str]] = field(default_factory=_empty_platform_buckets)
    l3: dict[str, list[str]] = field(default_factory=_empty_platform_buckets)
    l45: dict[str, list[str]] = field(default_factory=_empty_platform_buckets)

    @property
    def has_doc_changes(self) -> bool:
        return bool(self.docs)

    @property
    def has_skip_test_changes(self) -> bool:
        return bool(self.skip_tests)

    @property
    def has_other_changes(self) -> bool:
        """Product code or paths outside skip-ci scope."""
        return bool(self.other)

    @property
    def has_l23_yaml_changes(self) -> bool:
        """At least one whitelisted L2/L3 CI yaml file changed."""
        return any(self.l2[platform] or self.l3[platform] for platform in PLATFORMS)

    @property
    def has_l45_yaml_changes(self) -> bool:
        """At least one whitelisted L4/L5 (nightly/weekly) CI yaml file changed."""
        return any(self.l45[platform] for platform in PLATFORMS)

    @property
    def has_skip_ci_scope_changes(self) -> bool:
        """At least one doc / skip-mark / whitelisted CI yaml change."""
        return (
            self.has_doc_changes or self.has_skip_test_changes or self.has_l23_yaml_changes or self.has_l45_yaml_changes
        )

    def l2_platforms(self) -> frozenset[str]:
        return frozenset(platform for platform in PLATFORMS if self.l2[platform])

    def l3_platforms(self) -> frozenset[str]:
        return frozenset(platform for platform in PLATFORMS if self.l3[platform])


@dataclass(frozen=True)
class SkipL2L3Basis:
    skip_all_l2: bool
    skip_all_l3: bool

    @classmethod
    def from_buckets(cls, buckets: DiffBuckets) -> SkipL2L3Basis | None:
        """L4/L5 yaml → skip both L2 and L3 (L2/L3 buckets may rescue).

        Docs / skip-mark changes do not block this path; they are ignored for
        L2/L3 targeting when CI YAML is also present.
        """
        if buckets.has_other_changes:
            return None
        if not buckets.has_l45_yaml_changes:
            return None
        return cls(
            skip_all_l2=not buckets.l2_platforms(),
            skip_all_l3=not buckets.l3_platforms(),
        )


@dataclass(frozen=True)
class SkipL3Basis:
    enable_l2_platforms: frozenset[str]

    @classmethod
    def from_buckets(cls, buckets: DiffBuckets) -> SkipL3Basis | None:
        """L2 yaml changed → skip L3; keep L2 on touched platforms."""
        platforms = buckets.l2_platforms()
        if not platforms:
            return None
        return cls(enable_l2_platforms=platforms)


@dataclass(frozen=True)
class SkipL2Basis:
    enable_l3_platforms: frozenset[str]

    @classmethod
    def from_buckets(cls, buckets: DiffBuckets) -> SkipL2Basis | None:
        """L3 yaml changed → skip L2; keep L3 on touched platforms."""
        platforms = buckets.l3_platforms()
        if not platforms:
            return None
        return cls(enable_l3_platforms=platforms)


@dataclass(frozen=True)
class CiDecision:
    """Final CI skip decision for one diff."""

    skip_all: bool = False
    skip_l2_l3: bool = False
    device: CiDevice = field(default_factory=CiDevice)
    message: str = ""

    @classmethod
    def skip_all_basis(cls, buckets: DiffBuckets, *, diff_range: str | None) -> bool:
        """Docs / pytest skip-mark only → suppress entire default CI."""
        if buckets.has_other_changes:
            return False
        if (buckets.has_l23_yaml_changes or buckets.has_l45_yaml_changes) and (
            buckets.has_doc_changes or buckets.has_skip_test_changes
        ):
            return False
        if not buckets.has_doc_changes and not buckets.has_skip_test_changes:
            return False
        if diff_range is None and not buckets.has_doc_changes:
            return False
        return True

    @classmethod
    def from_yaml_basis(
        cls,
        l2_l3_basis: SkipL2L3Basis | None,
        l3_basis: SkipL3Basis | None,
        l2_basis: SkipL2Basis | None,
    ) -> CiDecision:
        """Build a yaml-gated decision by merging L2/L3 skip bases.

        Bases are applied as a union of enables: L2 yaml keeps L2 on touched
        platforms, L3 yaml keeps L3 on touched platforms. Cross-platform mixes
        (e.g. CUDA ready + AMD merge) must preserve both enables. The resulting
        per-platform / per-level matrix is stored on ``device`` as ``skip_*``.
        """
        run = {platform: {"l2": True, "l3": True} for platform in PLATFORMS}

        if l2_l3_basis is not None:
            if l2_l3_basis.skip_all_l2:
                for platform in PLATFORMS:
                    run[platform]["l2"] = False
            if l2_l3_basis.skip_all_l3:
                for platform in PLATFORMS:
                    run[platform]["l3"] = False

        if l3_basis is not None:
            for platform in PLATFORMS:
                run[platform]["l3"] = False
                run[platform]["l2"] = platform in l3_basis.enable_l2_platforms

        if l2_basis is not None:
            for platform in PLATFORMS:
                if platform in l2_basis.enable_l3_platforms:
                    run[platform]["l3"] = True
                    if l3_basis is None or platform not in l3_basis.enable_l2_platforms:
                        run[platform]["l2"] = False
                elif l3_basis is None:
                    # L3-yaml-only: disable platforms that did not change L3.
                    run[platform]["l2"] = False
                    run[platform]["l3"] = False
                # else: leave L2 enables from SkipL3Basis; keep this platform's L3 off.

        return cls(
            skip_l2_l3=True,
            device=cls._device_from_run(run),
        )

    @staticmethod
    def _device_from_run(run: dict[str, dict[str, bool]]) -> CiDevice:
        return CiDevice(
            **{f"skip_{platform}_{level}": not run[platform][level] for platform in PLATFORMS for level in ("l2", "l3")}
        )

    def is_run(self, platform: str, level: str) -> bool:
        if self.skip_all:
            return False
        if not self.skip_l2_l3:
            return True
        if level not in L23_LEVELS:
            return True
        return not self.device.is_skip(platform, level)


@dataclass(frozen=True)
class CiContext:
    """Skip-ci decision plus git diff inputs used to compute it."""

    decision: CiDecision
    changed_files: list[str] | None
    diff_range: str | None


def _log(message: str) -> None:
    print(f"{LOG}: {message}", file=sys.stderr)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )


# --- Diff helpers ---


def _resolve_diff_range() -> str | None:
    """Return a git diff range for PR or main builds, or None when unavailable."""
    is_pr = os.environ.get("BUILDKITE_PULL_REQUEST", "false") != "false" and os.environ.get(
        "BUILDKITE_PULL_REQUEST", ""
    )
    commit = os.environ.get("BUILDKITE_COMMIT", "")

    if is_pr:
        base_branch = os.environ.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH", "main")
        base_ref = f"origin/{base_branch}"
        if _git("rev-parse", "--verify", base_ref).returncode != 0:
            _log(f"origin/{base_branch} not found locally; trying fetch")
            _git("fetch", "--depth=200", "origin", base_branch)
        if _git("rev-parse", "--verify", base_ref).returncode != 0:
            if _git("rev-parse", "--verify", base_branch).returncode == 0:
                base_ref = base_branch
            else:
                _log(f"cannot resolve PR base {base_branch}; using safe defaults")
                return None
        return f"{base_ref}...{commit}"

    if os.environ.get("BUILDKITE_BRANCH", "") == "main":
        if _git("rev-parse", "--verify", f"{commit}^").returncode != 0:
            _log("main commit has no parent; using safe defaults")
            return None
        return f"{commit}^..{commit}"

    _log("not PR/main build; using safe defaults")
    return None


def _changed_files_for_diff_range(diff_range: str | None) -> list[str] | None:
    """Return changed file paths for *diff_range*, or None when unavailable."""
    if diff_range is None:
        return None

    result = _git("diff", "--name-only", diff_range)
    if result.returncode != 0:
        _log(f"git diff failed ({diff_range}); using safe defaults")
        return None

    files = [line for line in result.stdout.splitlines() if line.strip()]
    _log(f"{len(files)} changed file(s)")
    return files


def resolve_ci_context_from_git() -> CiContext:
    """Resolve git diff once and return skip-ci decision plus changed files."""
    diff_range = _resolve_diff_range()
    changed_files = _changed_files_for_diff_range(diff_range)
    decision = resolve_ci_decision(changed_files, diff_range=diff_range)
    return CiContext(decision=decision, changed_files=changed_files, diff_range=diff_range)


# --- Doc / skip-mark helpers ---


_SKIP_MARK_RE = re.compile(r"pytest\.mark\.skip(?:if)?\b|pytest\.skip\s*\(")
_PYTESTMARK_SKIP_RE = re.compile(r"pytest\.mark\.skip\b")
_PYTEST_MARK_ONLY_RE = re.compile(r"pytest\.mark\.\w+")


def _diff_only_contains_skip_mark_changes(diff_text: str) -> bool:
    """Return True when a unified diff only *adds* pytest skip marks.

    Deleting a skip mark re-enables the test and must not suppress CI. Editing an
    existing skip (``-`` old / ``+`` new) still qualifies when at least one skip
    mark is added.
    """

    def paren_balance(line: str) -> int:
        return line.count("(") - line.count(")")

    def is_blank_or_comment(line: str) -> bool:
        stripped = line.strip()
        return not stripped or stripped.startswith("#")

    def is_skip_mark_line(line: str) -> bool:
        return _SKIP_MARK_RE.search(line.strip()) is not None

    def is_pytestmark_adjacent_line(line: str) -> bool:
        stripped = line.strip().rstrip(",")
        if not stripped or stripped in {"[", "]"} or stripped.startswith("#"):
            return True
        if stripped.startswith("pytestmark"):
            return True
        return _PYTEST_MARK_ONLY_RE.search(stripped) is not None

    pending_depth = 0
    saw_change = False
    has_added_skip_mark = False
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("@@"):
            pending_depth = 0
            continue
        if not (raw_line.startswith("+") or raw_line.startswith("-")):
            continue
        if raw_line.startswith("+++") or raw_line.startswith("---"):
            continue

        saw_change = True
        is_add = raw_line.startswith("+")
        content = raw_line[1:]

        if pending_depth > 0:
            pending_depth += paren_balance(content)
            if pending_depth < 0:
                pending_depth = 0
            continue

        if is_blank_or_comment(content):
            continue

        if is_skip_mark_line(content):
            if is_add:
                has_added_skip_mark = True
            pending_depth = max(0, paren_balance(content))
            continue

        if is_pytestmark_adjacent_line(content):
            continue

        return False

    return saw_change and has_added_skip_mark


def _is_skip_testcase_change(file_path: str, *, diff_range: str) -> bool:
    """Return True when a test file diff is skip-mark-only (including new-file cases)."""
    diff_result = _git("diff", diff_range, "--", file_path)
    if diff_result.returncode != 0:
        _log(f"skip-mark-only: git diff failed for {file_path}")
        return False

    diff_text = diff_result.stdout
    if not diff_text.strip():
        _log(f"skip-mark-only: empty diff for {file_path}")
        return False

    if _diff_only_contains_skip_mark_changes(diff_text):
        return True

    if "new file mode" not in diff_text and "--- /dev/null" not in diff_text:
        _log(f"skip-mark-only: non-skip changes in {file_path}")
        return False

    commit = os.environ.get("BUILDKITE_COMMIT", "")
    if not commit:
        return False

    content_result = _git("show", f"{commit}:{file_path}")
    if content_result.returncode != 0:
        _log(f"skip-mark-only: cannot read {file_path} at {commit}")
        return False

    content = content_result.stdout
    match = re.search(r"^pytestmark\s*=\s*\[(.*?)^\s*\]", content, re.MULTILINE | re.DOTALL)
    if match is not None and _PYTESTMARK_SKIP_RE.search(match.group(1)) is not None:
        _log(f"skip-mark-only: new test file with module-level pytest.mark.skip: {file_path}")
        return True
    if re.search(r"^pytestmark\s*=\s*pytest\.mark\.skip\b", content, re.MULTILINE):
        _log(f"skip-mark-only: new test file with module-level pytest.mark.skip: {file_path}")
        return True

    _log(f"skip-mark-only: new test file without module-level pytest.mark.skip: {file_path}")
    return False


def _classify_changed_files_into_buckets(changed_files: list[str], *, diff_range: str | None) -> DiffBuckets:
    """Classify each changed file path into a ``DiffBuckets`` category."""
    buckets = DiffBuckets()
    for file_path in changed_files:
        if not file_path:
            continue
        if file_path.startswith("docs/") or file_path.endswith(".md") or file_path == "mkdocs.yml":
            buckets.docs.append(file_path)
            continue
        path = Path(file_path)
        if path.suffix == ".py" and path.parts and path.parts[0] == "tests":
            if diff_range is not None and _is_skip_testcase_change(file_path, diff_range=diff_range):
                buckets.skip_tests.append(file_path)
            else:
                buckets.other.append(file_path)
            continue
        if file_path in L2_YAML_FILES:
            platform = L2_YAML_FILES[file_path]
            buckets.l2[platform].append(file_path)
        elif file_path in L3_YAML_FILES:
            platform = L3_YAML_FILES[file_path]
            buckets.l3[platform].append(file_path)
        elif file_path in L45_YAML_FILES:
            platform = L45_YAML_FILES[file_path]
            buckets.l45[platform].append(file_path)
        else:
            buckets.other.append(file_path)
    return buckets


# --- Decision ---


def _finish(decision: CiDecision, message: str) -> CiDecision:
    finished = replace(decision, message=message)
    _log(message)
    return finished


def _format_yaml_gated_message(buckets: DiffBuckets, decision: CiDecision) -> str:
    """Human-readable summary of which CI YAML changed and which L2/L3 targets run."""
    changed_parts: list[str] = []
    for level_name, by_platform in (("L2", buckets.l2), ("L3", buckets.l3), ("L4/L5", buckets.l45)):
        files = [path for platform in PLATFORMS for path in by_platform[platform]]
        if files:
            changed_parts.append(f"{level_name}=[{', '.join(files)}]")

    run_targets = [
        f"{platform}/{level}" for platform in PLATFORMS for level in ("l2", "l3") if decision.is_run(platform, level)
    ]
    skip_targets = [
        f"{platform}/{level}"
        for platform in PLATFORMS
        for level in ("l2", "l3")
        if not decision.is_run(platform, level)
    ]

    changed = "; ".join(changed_parts) if changed_parts else "none"
    run = ", ".join(run_targets) if run_targets else "none"
    skip = ", ".join(skip_targets) if skip_targets else "none"
    if buckets.has_doc_changes or buckets.has_skip_test_changes:
        return (
            "CI config YAML + docs/skip-mark changed — follow CI YAML gating; "
            f"changed: {changed}; run: {run}; skip: {skip}"
        )
    return f"only CI config YAML changed — changed: {changed}; run: {run}; skip: {skip}"


def resolve_ci_decision(
    changed_files: list[str] | None = None,
    *,
    diff_range: str | None = None,
) -> CiDecision:
    if changed_files is None:
        return _finish(CiDecision(), "could not resolve changed files; run normal CI")

    buckets = _classify_changed_files_into_buckets(changed_files, diff_range=diff_range)
    if buckets.has_other_changes or not buckets.has_skip_ci_scope_changes:
        return _finish(CiDecision(), "product or non-whitelisted changes present; run normal CI")

    if CiDecision.skip_all_basis(buckets, diff_range=diff_range):
        return _finish(
            CiDecision(skip_all=True),
            "CI skipped — docs or pytest skip-mark changes only",
        )

    # Docs / skip-mark mixed with whitelisted CI YAML → same yaml-gated path
    # as CI-YAML-only (docs/skip-mark do not widen to normal CI).
    if not buckets.has_l23_yaml_changes and not buckets.has_l45_yaml_changes:
        if buckets.has_skip_test_changes and diff_range is None:
            message = (
                "only pytest skip-mark test changes detected, but no git diff range "
                "is available to verify them (skip-all requires diff_range); run normal CI"
            )
        elif buckets.has_skip_test_changes:
            message = "pytest skip-mark test changes did not qualify for skip-all and no ci yaml changed; run normal CI"
        else:
            message = "doc/skip-mark changes did not qualify for skip-all; run normal CI"
        return _finish(CiDecision(), message)

    decision = CiDecision.from_yaml_basis(
        SkipL2L3Basis.from_buckets(buckets),
        SkipL3Basis.from_buckets(buckets),
        SkipL2Basis.from_buckets(buckets),
    )
    return _finish(decision, _format_yaml_gated_message(buckets, decision))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gate_parser = subparsers.add_parser(
        "gate",
        help="exit 0 when bootstrap should stop (skip-all or PLATFORM/LEVEL disabled); logs once",
    )
    gate_parser.add_argument("platform", help="cuda, amd, intel, or npu")
    gate_parser.add_argument("level", choices=("l2", "l3"), help="test level")

    subparsers.add_parser(
        "check-skip-all",
        help="exit 0 when docs/skip-mark-only diff should skip entire CI upload",
    )

    skip_parser = subparsers.add_parser(
        "check-skip-l2-l3",
        help="exit 0 when PLATFORM l2/l3 pipeline should be skipped (CI config YAML-only diffs)",
    )
    skip_parser.add_argument("platform", help="cuda, amd, intel, or npu")
    skip_parser.add_argument("level", choices=("l2", "l3"), help="test level")

    subparsers.add_parser(
        "print-annotate",
        help="print human-readable skip reason for Buildkite annotation",
    )

    args = parser.parse_args()
    decision = resolve_ci_context_from_git().decision

    if args.command == "gate":
        if decision.skip_all:
            print("skip-all")
            return 0
        if not decision.is_run(args.platform, args.level):
            print("skip-l23")
            return 0
        return 1

    if args.command == "check-skip-all":
        return 0 if decision.skip_all else 1

    if args.command == "check-skip-l2-l3":
        return 0 if not decision.is_run(args.platform, args.level) else 1

    if args.command == "print-annotate":
        print(decision.message or "CI will run normally")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
