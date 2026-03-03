#!/usr/bin/env python3
"""
Send the nightly performance Excel report by email to DAILY_EMAIL_LIST.

Reads SMTP and recipient config from environment variables. Use --dry-run to
validate config and print subject/body without sending.
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

LOGGER = logging.getLogger(__name__)

# Env keys for SMTP and recipients.
ENV_SMTP_HOST = "SMTP_HOST"
ENV_SMTP_PORT = "SMTP_PORT"
ENV_SMTP_USERNAME = "SMTP_USERNAME"
ENV_SMTP_PASSWORD = "SMTP_PASSWORD"
ENV_DAILY_EMAIL_LIST = "DAILY_EMAIL_LIST"
ENV_EMAIL_SENDER = "EMAIL_SENDER"
ENV_EMAIL_SUBJECT_PREFIX = "EMAIL_SUBJECT_PREFIX"
ENV_BUILD_URL = "BUILDKITE_BUILD_URL"
ENV_COMMIT = "BUILDKITE_COMMIT"

DEFAULT_OUTPUT_DIR = os.getenv("DEFAULT_OUTPUT_DIR")

# Do not attach Excel if size >= this (bytes); body will suggest downloading from build URL.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

SMTP_RETRIES = 3
SMTP_RETRY_DELAY_SEC = 5


def _get_required_env() -> dict[str, str]:
    """Read required env vars; raise SystemExit with clear message if any missing."""
    required = {
        ENV_SMTP_HOST: os.environ.get(ENV_SMTP_HOST),
        ENV_SMTP_PORT: os.environ.get(ENV_SMTP_PORT),
        ENV_SMTP_USERNAME: os.environ.get(ENV_SMTP_USERNAME),
        ENV_SMTP_PASSWORD: os.environ.get(ENV_SMTP_PASSWORD),
        ENV_DAILY_EMAIL_LIST: os.environ.get(ENV_DAILY_EMAIL_LIST),
    }
    missing = [k for k, v in required.items() if not (v and str(v).strip())]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}. Set them (e.g. in Buildkite secrets).")
    return {k: str(v).strip() for k, v in required.items()}


def _get_latest_file(folder_path: str) -> str:
    """Get the latest modified file from the folder path."""
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".xlsx")]
    if not files:
        raise SystemExit(f"No Excel files found in {folder_path}")
    return max(files, key=os.path.getmtime)


def _recipients_list(comma_separated: str) -> list[str]:
    """Parse DAILY_EMAIL_LIST into a list of addresses."""
    return [a.strip() for a in comma_separated.split(",") if a.strip()]


def _build_body(
    date_str: str,
    commit_sha: str | None,
    build_url: str | None,
    attachment_skipped: bool,
) -> str:
    """Plain-text body with build metadata; note when attachment was skipped."""
    lines = [
        f"Nightly performance report date: {date_str}",
        "",
        f"Commit: {commit_sha or 'N/A'}",
        f"Build: {build_url or 'N/A'}",
        "",
    ]
    if attachment_skipped:
        lines.append(
            "Report file was too large to attach. Please download the Excel from the build artifacts (Build URL above)."
        )
        lines.append("")
    return "\n".join(lines)


def _build_subject(prefix: str | None, date_str: str) -> str:
    """Subject line: optional prefix + date."""
    base = f"Nightly Perf {date_str}"
    if prefix and prefix.strip():
        return f"{prefix.strip()} {base}"
    return base


def _send_mail(
    report_file: str,
    date_str: str,
    dry_run: bool,
) -> None:
    """Load config, build message, and send (or dry-run)."""
    cfg = _get_required_env()
    recipients = _recipients_list(cfg[ENV_DAILY_EMAIL_LIST])
    if not recipients:
        raise SystemExit("DAILY_EMAIL_LIST is empty after parsing.")

    commit_sha = os.environ.get(ENV_COMMIT)
    build_url = os.environ.get(ENV_BUILD_URL)
    sender = os.environ.get(ENV_EMAIL_SENDER) or cfg[ENV_SMTP_USERNAME]
    prefix = os.environ.get(ENV_EMAIL_SUBJECT_PREFIX)

    size = os.path.getsize(report_file) if os.path.isfile(report_file) else 0
    attach_excel = size < MAX_ATTACHMENT_BYTES and size > 0
    attachment_skipped = os.path.isfile(report_file) and not attach_excel

    body = _build_body(
        commit_sha=commit_sha, build_url=build_url, date_str=date_str, attachment_skipped=attachment_skipped
    )
    subject = _build_subject(prefix=prefix, date_str=date_str)

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if attach_excel:
        with open(report_file, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(report_file))
        msg.attach(part)

    if dry_run:
        LOGGER.info("dry-run: not sending mail")
        print("To:", recipients, file=sys.stderr)
        print("Subject:", subject, file=sys.stderr)
        print("Attachment:", "yes" if attach_excel else "no (size limit)", file=sys.stderr)
        print("Body preview:", body[:300] + ("..." if len(body) > 300 else ""), file=sys.stderr)
        return

    port = int(cfg[ENV_SMTP_PORT], 10)
    last_err: Exception | None = None
    for attempt in range(SMTP_RETRIES):
        try:
            with smtplib.SMTP(cfg[ENV_SMTP_HOST], port=port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(cfg[ENV_SMTP_USERNAME], cfg[ENV_SMTP_PASSWORD])
                smtp.sendmail(sender, recipients, msg.as_string())
            LOGGER.info("sent nightly perf email to %d recipient(s)", len(recipients))
            return
        except Exception as e:
            last_err = e
            LOGGER.warning("SMTP attempt %d/%d failed: %s", attempt + 1, SMTP_RETRIES, e)
            if attempt < SMTP_RETRIES - 1:
                import time

                time.sleep(SMTP_RETRY_DELAY_SEC)
    raise SystemExit(f"Failed to send email after {SMTP_RETRIES} attempts.") from last_err


def _date_from_filename(path: str) -> str:
    """Try to derive a date string from report filename (e.g. nightly_perf_20260211-020704.xlsx -> 2026-02-11)."""
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("nightly_perf_") and len(base) > 13:
        raw = base.replace("nightly_perf_", "")[:8]
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return base or "unknown"


def _vllm_omni_root() -> str:
    """Resolve vllm-omni repo root: directory that contains a 'tests' subdir (and usually 'tools')."""
    path = os.path.dirname(os.path.abspath(__file__))
    while path and path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "tests")):
            return path
        path = os.path.dirname(path)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _default_output_dir() -> str:
    """Default: vllm-omni root / DEFAULT_OUTPUT_DIR (where performance .xlsx files live)."""
    root = _vllm_omni_root()
    return os.path.join(root, DEFAULT_OUTPUT_DIR)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Send nightly performance Excel report by email (config from env).",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=_default_output_dir(),
        help="Folder/file path to the nightly_perf_*.xlsx file; default is DEFAULT_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string for subject/body (default: inferred from report filename).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recipient, subject, and body; do not send.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    report_file = _get_latest_file(args.report_file) if os.path.isdir(args.report_file) else args.report_file
    if not os.path.isfile(report_file):
        raise SystemExit(f"Report file not found: {report_file}")

    date_str = args.date or _date_from_filename(report_file)
    _send_mail(report_file=report_file, date_str=date_str, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
