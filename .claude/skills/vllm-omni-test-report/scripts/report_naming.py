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
