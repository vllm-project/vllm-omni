#!/usr/bin/env python3
"""
Send the nightly report by email to DAILY_EMAIL_LIST.

Reads SMTP and recipient config from environment variables. Use --dry-run to
validate config and print subject/body without sending.
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import smtplib
import sys
from datetime import date
from email.header import Header
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

# Do not attach file if size >= this (bytes); body will suggest downloading from build URL.
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

SMTP_RETRIES = 3
SMTP_RETRY_DELAY_SEC = 5


def _strip_quoted(s: str) -> str:
    """Remove one layer of surrounding quote characters from env values.

    Handles both ASCII quotes ('", ''') and common Unicode quotes (“ ” ‘ ’).
    """
    if not s:
        return s
    s = s.strip()
    if not s:
        return s
    quotes = {'"', "'", "“", "”", "‘", "’"}
    if len(s) >= 2 and s[0] in quotes and s[-1] in quotes:
        s = s[1:-1].strip()
    return s


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
    return {k: _strip_quoted(str(v)) for k, v in required.items()}


def _get_latest_file(path: str) -> str | list[str]:
    """
    Resolve report file patterns into concrete files.

    - If path is a directory: return all files in that directory (list).
    - If path contains wildcards (e.g. *.xlsx, *.html): expand using glob and
      return all matching files (list).
    - If path is a plain filename or file path: treat as (dir, filename),
      search that directory for files with the same basename and return the
      latest modified one (str).

    Note: higher-level callers will turn a single str into [str] when they
    want to attach multiple files.
    """
    import glob

    # Support comma-/semicolon-separated list: e.g. "*.xlsx, *.html; reports/foo.xlsx"
    parts = [p.strip() for p in path.split(";") for p in p.split(",") if p and p.strip()]
    if len(parts) > 1:
        files: list[str] = []
        for part in parts:
            files_part = _get_latest_file(part)
            if isinstance(files_part, str):
                files.append(files_part)
            else:
                files.extend(files_part)
        if not files:
            raise SystemExit(f"No files resolved from list: {path}")
        return sorted(set(files))

    # Single expression after splitting
    path = parts[0] if parts else path

    # Directory: return all files inside
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if not files:
            raise SystemExit(f"No files found in {path}")
        return sorted(files)

    # Wildcard pattern: expand via glob
    if any(ch in path for ch in ("*", "?", "[")) and not os.path.exists(path):
        matches = [p for p in glob.glob(path) if os.path.isfile(p)]
        if not matches:
            raise SystemExit(f"No files match pattern {path!r}")
        return sorted(matches)

    # Plain filename or explicit file path: search its directory for same basename
    parent = os.path.dirname(path) or "."
    base = os.path.basename(path)
    if not base:
        raise SystemExit(f"Invalid path (no filename): {path}")
    if not os.path.isdir(parent):
        raise SystemExit(f"Directory not found: {parent}")
    candidates = [
        p
        for p in (os.path.join(parent, f) for f in os.listdir(parent))
        if os.path.basename(p) == base and os.path.isfile(p)
    ]
    if not candidates:
        raise SystemExit(f"No file named {base!r} found in {parent}")
    return max(candidates, key=os.path.getmtime)


def _recipients_list(comma_separated: str) -> list[str]:
    """Parse DAILY_EMAIL_LIST into a list of addresses."""
    return [_strip_quoted(a.strip()) for a in comma_separated.split(",") if a and a.strip()]


def _build_body(
    date_str: str,
    commit_sha: str | None,
    build_url: str | None,
    attachment_skipped: bool,
) -> str:
    """Plain-text body with build metadata; note when attachment was skipped."""
    lines = [
        f"Nightly report date: {date_str}",
        "",
        f"Commit: {commit_sha or 'N/A'}",
        f"Build: {build_url or 'N/A'}",
        "",
    ]
    if attachment_skipped:
        lines.append(
            "Some report file(s) were too large to attach. Please download from the build artifacts (Build URL above)."
        )
        lines.append("")
    return "\n".join(lines)


def _build_subject(prefix: str | None, date_str: str) -> str:
    """Subject line: optional prefix + date."""
    base = f"Nightly Report {date_str}"
    if prefix and prefix.strip():
        return f"{prefix.strip()} {base}"
    return base


def _mime_type_and_subtype(file_path: str) -> tuple[str, str]:
    """Return (maintype, subtype) for the file; fallback to application/octet-stream."""
    full_type, _ = mimetypes.guess_type(file_path, strict=False)
    if not full_type:
        return "application", "octet-stream"
    parts = full_type.split("/", 1)
    if len(parts) != 2:
        return "application", "octet-stream"
    return parts[0], parts[1]


def _attach_file(msg: MIMEMultipart, file_path: str, max_bytes: int) -> bool:
    """
    Attach one file to msg with correct MIME type. Return True if attached, False if skipped (e.g. size).
    """
    if not os.path.isfile(file_path):
        return False
    size = os.path.getsize(file_path)
    if size <= 0 or size >= max_bytes:
        return False
    maintype, subtype = _mime_type_and_subtype(file_path)
    name = os.path.basename(file_path)
    try:
        if maintype == "text":
            with open(file_path, encoding="utf-8", errors="replace") as f:
                part = MIMEText(f.read(), subtype, "utf-8")
        else:
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), _subtype=subtype)
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)
        return True
    except OSError:
        return False


def _send_mail(
    report_files: list[str],
    date_str: str,
    dry_run: bool,
) -> None:
    """Load config, build message (attach each file with correct MIME type), and send (or dry-run)."""
    cfg = _get_required_env()
    recipients = _recipients_list(cfg[ENV_DAILY_EMAIL_LIST])
    if not recipients:
        raise SystemExit("DAILY_EMAIL_LIST is empty after parsing.")

    commit_sha = os.environ.get(ENV_COMMIT)
    build_url = os.environ.get(ENV_BUILD_URL)
    # Allow EMAIL_SENDER / EMAIL_SUBJECT_PREFIX to be quoted; strip quotes if present.
    raw_sender = os.environ.get(ENV_EMAIL_SENDER)
    sender = _strip_quoted(raw_sender) if raw_sender else cfg[ENV_SMTP_USERNAME]
    raw_prefix = os.environ.get(ENV_EMAIL_SUBJECT_PREFIX)
    prefix = _strip_quoted(raw_prefix) if raw_prefix else None

    attachment_skipped = any(os.path.isfile(p) and os.path.getsize(p) >= MAX_ATTACHMENT_BYTES for p in report_files)
    body = _build_body(
        commit_sha=commit_sha, build_url=build_url, date_str=date_str, attachment_skipped=attachment_skipped
    )
    subject = _build_subject(prefix=prefix, date_str=date_str)

    msg = MIMEMultipart()
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(body, "plain", "utf-8"))

    attached_count = 0
    for file_path in report_files:
        if os.path.isfile(file_path) and os.path.getsize(file_path) >= MAX_ATTACHMENT_BYTES:
            attachment_skipped = True
        else:
            if _attach_file(msg, file_path, MAX_ATTACHMENT_BYTES):
                attached_count += 1
            elif os.path.isfile(file_path):
                attachment_skipped = True

    if dry_run:
        LOGGER.info("dry-run: not sending mail")
        print("To:", recipients, file=sys.stderr)
        print("Subject:", subject, file=sys.stderr)
        print("Attachments:", attached_count, "(skipped large:", attachment_skipped, ")", file=sys.stderr)
        print("Body preview:", body[:300] + ("..." if len(body) > 300 else ""), file=sys.stderr)
        return

    host = cfg[ENV_SMTP_HOST]
    port = int(cfg[ENV_SMTP_PORT], 10)
    use_ssl = port == 465
    last_err: Exception | None = None
    for attempt in range(SMTP_RETRIES):
        try:
            if use_ssl:
                with smtplib.SMTP_SSL(host, port=port, timeout=30) as smtp:
                    smtp.login(cfg[ENV_SMTP_USERNAME], cfg[ENV_SMTP_PASSWORD])
                    smtp.sendmail(sender, recipients, msg.as_string())
            else:
                with smtplib.SMTP(host, port=port, timeout=30) as smtp:
                    smtp.starttls()
                    smtp.login(cfg[ENV_SMTP_USERNAME], cfg[ENV_SMTP_PASSWORD])
                    smtp.sendmail(sender, recipients, msg.as_string())
            LOGGER.info("sent nightly email to %d recipient(s)", len(recipients))
            return
        except Exception as e:
            last_err = e
            LOGGER.warning("SMTP attempt %d/%d failed: %s", attempt + 1, SMTP_RETRIES, e)
            if attempt < SMTP_RETRIES - 1:
                import time

                time.sleep(SMTP_RETRY_DELAY_SEC)
    hint = ""
    if last_err and "getaddrinfo" in str(last_err).lower():
        hint = f" (check SMTP_HOST={host!r} is correct and reachable, e.g. nslookup {host})"
    raise SystemExit(f"Failed to send email after {SMTP_RETRIES} attempts.{hint}") from last_err


def _vllm_omni_root() -> str:
    """Resolve vllm-omni repo root: directory that contains a 'tests' subdir (and usually 'tools')."""
    path = os.path.dirname(os.path.abspath(__file__))
    while path and path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "tests")):
            return path
        path = os.path.dirname(path)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def _default_output_dir() -> str:
    """Default: vllm-omni root / DEFAULT_OUTPUT_DIR (where nightly report files live)."""
    root = _vllm_omni_root()
    subdir = DEFAULT_OUTPUT_DIR if (DEFAULT_OUTPUT_DIR and DEFAULT_OUTPUT_DIR.strip()) else "."
    return os.path.join(root, subdir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Send nightly report by email (config from env).",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=_default_output_dir(),
        help="Folder/file path to the sent email file; default is DEFAULT_OUTPUT_DIR.",
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

    result = _get_latest_file(args.report_file)
    report_files = result if isinstance(result, list) else [result]
    report_files = [p for p in report_files if os.path.isfile(p)]
    if not report_files:
        raise SystemExit("No report file(s) found.")

    date_str = args.date or date.today().strftime("%Y-%m-%d")
    _send_mail(report_files=report_files, date_str=date_str, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
