"""
Pure helpers for L5(b) reliability fault injection (RFC: fault_inject.py).

Abnormal-input paths use raw HTTP so malformed bodies reach the server (OpenAI SDK
would reject them client-side).
"""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from typing import Any

FaultVariant = str


@dataclass(frozen=True)
class FaultPhaseResult:
    """Observed HTTP statuses during fault injection (before recovery request)."""

    http_statuses: list[int]
    notes: str


@dataclass(frozen=True)
class RecoveryResult:
    """Structured outcome after fault + recovery attempt (RFC Detailed Design)."""

    recovered: bool
    recovery_time_sec: float | None
    health_check_ok: bool
    post_fault_success_count: int
    post_fault_error_count: int
    notes: str


def post_chat_completions_raw(
    host: str,
    port: int,
    body: bytes | str,
    *,
    content_type: str = "application/json",
    timeout_sec: int = 120,
) -> tuple[int, bytes]:
    """POST /v1/chat/completions with raw bytes; returns (status, response_body)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout_sec)
    try:
        headers = {"Content-Type": content_type}
        payload = body.encode("utf-8") if isinstance(body, str) else body
        conn.request("POST", "/v1/chat/completions", body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, data
    finally:
        conn.close()


def inject_abnormal_input_faults(
    host: str,
    port: int,
    model: str,
    fault_params: dict[str, Any],
) -> FaultPhaseResult:
    """
    Run configured abnormal-input HTTP variants against the running server.

    Args:
        host: Server host.
        port: Server port.
        model: Deployed model id (for JSON bodies that must reference it).
        fault_params: Must include ``variants`` (list of ``FaultVariant`` keys).

    Returns:
        FaultPhaseResult with one HTTP status per variant.
    """
    variants: list[FaultVariant] = fault_params.get(
        "variants",
        ["malformed_json", "invalid_messages_type"],
    )
    statuses: list[int] = []
    notes_parts: list[str] = []

    for key in variants:
        if key == "malformed_json":
            status, _ = post_chat_completions_raw(host, port, b"not json {{{")
            statuses.append(status)
            notes_parts.append(f"malformed_json:{status}")
        elif key == "invalid_messages_type":
            bad = json.dumps({"model": model, "messages": "must_be_a_list_not_string"})
            status, body = post_chat_completions_raw(host, port, bad)
            statuses.append(status)
            notes_parts.append(f"invalid_messages_type:{status}:{len(body)}B")
        else:
            raise ValueError(f"Unknown abnormal_input variant: {key!r}")

    return FaultPhaseResult(http_statuses=statuses, notes="; ".join(notes_parts))


def assert_fault_http_expectation(
    statuses: list[int],
    expect: dict[str, Any],
) -> None:
    """Assert fault-phase HTTP statuses match ``expect.fault_http_status_class``."""
    cls = expect.get("fault_http_status_class", "4xx")
    if cls == "4xx":
        for s in statuses:
            assert 400 <= s < 500, f"expected 4xx for abnormal input, got {s}"
        return
    raise ValueError(f"Unsupported fault_http_status_class: {cls!r}")


def build_recovery_result(
    *,
    recovered: bool,
    recovery_time_sec: float | None,
    health_check_ok: bool,
    post_fault_success_count: int,
    post_fault_error_count: int,
    notes: str,
) -> RecoveryResult:
    """Factory for RecoveryResult (single place for field defaults)."""
    return RecoveryResult(
        recovered=recovered,
        recovery_time_sec=recovery_time_sec,
        health_check_ok=health_check_ok,
        post_fault_success_count=post_fault_success_count,
        post_fault_error_count=post_fault_error_count,
        notes=notes,
    )
