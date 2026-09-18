"""Shared report filenames and titles — use generation date (UTC), not log-dir suffixes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def utc_report_date_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def validate_report_date_iso(value: str) -> str:
    datetime.strptime(value.strip(), "%Y-%m-%d")
    return value.strip()


def resolve_report_date_iso(explicit: str | None = None) -> str:
    if explicit:
        return validate_report_date_iso(explicit)
    return utc_report_date_iso()


def nightly_report_basename(date_iso: str | None = None) -> str:
    return f"nightly-report-buildkite-latest-{resolve_report_date_iso(date_iso)}.html"


def release_report_basename(date_iso: str | None = None) -> str:
    return f"vllm-omni-test-report-{resolve_report_date_iso(date_iso)}.html"


def release_report_preview_basename(date_iso: str | None = None) -> str:
    return f"vllm-omni-test-report-preview-{resolve_report_date_iso(date_iso)}.html"


def development_report_basename(date_iso: str | None = None) -> str:
    """Filename pattern for the **Development** variant (``compose_full_report.py --kind development``).

    Distinct suffix so it does not collide with the default **release** filename when both are
    generated on the same UTC day.
    """
    return f"vllm-omni-test-report-development-{resolve_report_date_iso(date_iso)}.html"


def development_report_preview_basename(date_iso: str | None = None) -> str:
    return f"vllm-omni-test-report-development-preview-{resolve_report_date_iso(date_iso)}.html"


def nightly_report_title(date_iso: str | None = None) -> str:
    return f"Nightly Buildkite report - {resolve_report_date_iso(date_iso)}"


def default_nightly_html_path(skill_dir: Path, date_iso: str | None = None) -> Path:
    return skill_dir / nightly_report_basename(date_iso)


def default_release_html_path(skill_dir: Path, date_iso: str | None = None) -> Path:
    return skill_dir / release_report_basename(date_iso)


def default_development_html_path(skill_dir: Path, date_iso: str | None = None) -> Path:
    return skill_dir / development_report_basename(date_iso)


def di_top20_report_basename(date_iso: str | None = None) -> str:
    """Filename pattern for the **Outstanding DI Top 20** standalone skill
    (`vllm-omni-issue-monitor`). Mirrors the nightly/development naming so the
    archive can be slotted alongside the existing daily HTML without
    colliding with another kind on the same UTC day.
    """
    return f"di-top20-report-{resolve_report_date_iso(date_iso)}.html"


def issue_monitor_report_basename(date_iso: str | None = None) -> str:
    """Filename pattern for the **Issue Monitor** standalone skill
    (`vllm-omni-issue-monitor`). Renamed from the ``di-top20-report-*``
    prefix in 2026-08-27 to match the new skill name; older filenames still
    resolve via :func:`di_top20_report_basename` for backwards compatibility.
    """
    return f"issue-monitor-report-{resolve_report_date_iso(date_iso)}.html"


def issue_monitor_report_preview_basename(date_iso: str | None = None) -> str:
    return f"issue-monitor-report-preview-{resolve_report_date_iso(date_iso)}.html"


def stale_prs_report_basename(date_iso: str | None = None) -> str:
    """Filename pattern for the **Stale PRs (Bot-Mentioned Reviewers)**
    standalone skill (`vllm-omni-pr-monitor`). Renamed from
    ``vllm-omni-stale-prs`` in 2026-08-27; old ``stale-prs-report-*``
    filenames still resolve via this helper for backwards compatibility.
    """
    return f"stale-prs-report-{resolve_report_date_iso(date_iso)}.html"


def pr_monitor_report_basename(date_iso: str | None = None) -> str:
    """Filename pattern for the **PR Monitor** standalone skill
    (`vllm-omni-pr-monitor`).
    """
    return f"pr-monitor-report-{resolve_report_date_iso(date_iso)}.html"


def stale_prs_report_preview_basename(date_iso: str | None = None) -> str:
    return f"stale-prs-report-preview-{resolve_report_date_iso(date_iso)}.html"


def pr_monitor_report_preview_basename(date_iso: str | None = None) -> str:
    return f"pr-monitor-report-preview-{resolve_report_date_iso(date_iso)}.html"


def di_top20_report_title(date_iso: str | None = None) -> str:
    return f"vLLM-Omni Outstanding DI Top 20 - {resolve_report_date_iso(date_iso)}"


def issue_monitor_report_title(date_iso: str | None = None) -> str:
    """Issue Monitor report title.

    The ``date_iso`` argument is accepted for API symmetry with the dated
    basename helpers but **not** rendered into the title — the standalone
    Issue Monitor report always reflects the most recent Outstanding DI
    snapshot, so the date lives in the filename (and the
    ``Generated <UTC>`` footer) instead of the heading.
    """
    return "vLLM-Omni Issue Monitor"


def stale_prs_report_title(date_iso: str | None = None) -> str:
    return f"vLLM-Omni Stale PRs (Bot-Mentioned) - {resolve_report_date_iso(date_iso)}"


def pr_monitor_report_title(date_iso: str | None = None) -> str:
    """PR Monitor report title.

    The ``date_iso`` argument is accepted for API symmetry with the dated
    basename helpers but **not** rendered into the title — the standalone
    PR Monitor report always reflects the most recent stale-PR snapshot,
    so the date lives in the filename (and the ``Generated <UTC>`` footer)
    instead of the heading.
    """
    return "vLLM-Omni PR Monitor"


def default_di_top20_html_path(skill_dir: Path, date_iso: str | None = None) -> Path:
    return skill_dir / di_top20_report_basename(date_iso)


def default_issue_monitor_html_path(skill_dir: Path, date_iso: str | None = None) -> Path:
    return skill_dir / issue_monitor_report_basename(date_iso)


def default_stale_prs_html_path(skill_dir: Path, date_iso: str | None = None) -> Path:
    return skill_dir / stale_prs_report_basename(date_iso)


def default_pr_monitor_html_path(skill_dir: Path, date_iso: str | None = None) -> Path:
    return skill_dir / pr_monitor_report_basename(date_iso)
