#!/usr/bin/env python3
"""Shared pytest log parsing: FAILURE/ERROR lines, reasons, traceback excerpts, heuristic analysis."""

from __future__ import annotations

import re
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Buildkite agent terminal metadata prefix (before pytest output on each line).
BUILDKITE_BK_PREFIX_RE = re.compile(r"\x1b_bk;[^\x07]*\x07")
BUILDKITE_BK_LITERAL_PREFIX_RE = re.compile(r"^_bk;t=\d+\x07?")
BUILDKITE_TS_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
# Pytest final session lines (6.x/7.x/8.x variants).
SESSION_LINE_RE = re.compile(r"^=+\s*.+\s+in\s+[\d.]+s\s*=+\s*$")
COUNTS_FRAGMENT_RE = re.compile(
    r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+skipped|(\d+)\s+errors?\b",
    re.IGNORECASE,
)
SECTION_RULE_RE = re.compile(r"^=+\s*.+?\s*=+$")
_FAILURES_BANNER_INLINE_RE = re.compile(r"=+\s*FAILURES\s*=+", re.I)
_ERRORS_BANNER_INLINE_RE = re.compile(r"=+\s*ERRORS?\s*=+", re.I)
# Banner line after FAILURES: next major pytest section (not another FAILURES).
_POST_FAILURES_SECTION_RE = re.compile(
    r"(short\s+test\s+summary|warnings?\s+summary|^errors$|\berrors\b|^\s*warnings\s*$|^passes$|\bpasses\b)",
    re.I,
)
_UNDERSCORE_BLOCK_HEADER = re.compile(r"^(_{3,})\s*(.+?)\s*(_{3,})$")


def _pytest_banner_inner(line: str) -> str | None:
    s = line.strip()
    if len(s) < 10 or not s.startswith("=") or not s.endswith("="):
        return None
    inner = s.strip("=").strip()
    return inner or None


def _pytest_banner_inner_from_raw(raw: str) -> str | None:
    """Parse ``=== SECTION ===`` inner text when the line has a leading timestamp/prefix."""
    t = raw.strip()
    eq = t.find("=")
    if eq < 0:
        return None
    return _pytest_banner_inner(t[eq:])


def _line_has_failures_banner(raw: str) -> bool:
    s = raw.strip()
    if len(s) < 15:
        return False
    if "short test summary" in s.lower():
        return False
    return bool(_FAILURES_BANNER_INLINE_RE.search(s))


def _line_has_errors_banner(raw: str) -> bool:
    s = raw.strip()
    if len(s) < 15:
        return False
    return bool(_ERRORS_BANNER_INLINE_RE.search(s))


def _failures_section_range(lines: list[str]) -> tuple[int, int] | None:
    start: int | None = None
    for i, raw in enumerate(lines):
        if _line_has_failures_banner(raw):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if _line_has_failures_banner(lines[j]):
            continue
        inner = _pytest_banner_inner_from_raw(lines[j])
        if inner is None:
            continue
        if _POST_FAILURES_SECTION_RE.search(inner):
            return start, j
    return start, len(lines)


def _errors_section_range(lines: list[str]) -> tuple[int, int] | None:
    start: int | None = None
    for i, raw in enumerate(lines):
        if _line_has_errors_banner(raw):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if _line_has_errors_banner(lines[j]):
            continue
        inner = _pytest_banner_inner_from_raw(lines[j])
        if inner is None:
            continue
        il = inner.lower()
        if il == "failures" or "short test summary" in il or "warnings" in il:
            return start, j
    return start, len(lines)


def _short_summary_section_range(lines: list[str]) -> tuple[int, int] | None:
    """Lines under pytest ``short test summary info`` until the next major ``===`` banner."""
    start: int | None = None
    for i, raw in enumerate(lines):
        low = raw.lower()
        if "short test summary" in low and "info" in low:
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        raw = lines[j].strip()
        if not SECTION_RULE_RE.match(raw) or len(raw) <= 10:
            continue
        inner = _pytest_banner_inner(raw)
        if inner is None:
            continue
        if "short test summary" in inner.lower():
            continue
        return start, j
    return start, len(lines)


def _underscore_header_title(line: str) -> str | None:
    m = _UNDERSCORE_BLOCK_HEADER.match(line.strip())
    if not m:
        return None
    mid = m.group(2).strip()
    return mid or None


def _extract_underscore_blocks(lines: list[str], a: int, b: int) -> list[tuple[str, str]]:
    """Titles and bodies between pytest ``____ title ____`` headers (e.g. under FAILURES)."""
    out: list[tuple[str, str]] = []
    i = a + 1
    n = min(b, len(lines))
    while i < n:
        title = _underscore_header_title(lines[i])
        if title is None:
            i += 1
            continue
        i += 1
        body_lines: list[str] = []
        while i < n:
            if _underscore_header_title(lines[i]) is not None:
                break
            body_lines.append(lines[i])
            i += 1
        text = "\n".join(body_lines).strip()
        out.append((title, text))
    return out


def _tail_node_id(node_id: str) -> str:
    return node_id.split("::")[-1].strip()


def _node_id_match_variants(node_id: str) -> list[str]:
    """Substrings useful for matching ERRORS/FAILURES underscore titles and bodies."""
    n = (node_id or "").strip().replace("\\", "/")
    if not n:
        return []
    seen: dict[str, None] = {}
    out: list[str] = []

    def add(s: str) -> None:
        t = s.strip()
        if t and t not in seen:
            seen[t] = None
            out.append(t)

    add(n)
    if "::" in n:
        path, tail = n.split("::", 1)
        add(path)
        if "[" in tail:
            add(f"{path}::{tail.split('[', 1)[0]}")
        base = path.split("/")[-1]
        if base:
            add(f"{base}::{tail}")
            if "[" in tail:
                add(f"{base}::{tail.split('[', 1)[0]}")
    else:
        add(n.split("/")[-1])
    return out


def _canonical_error_node_id(node_id: str) -> str:
    """Strip accidental `` (ERROR)`` suffix (HTML/report display) for matching."""
    n = (node_id or "").strip()
    if n.upper().endswith("(ERROR)"):
        n = n[: -len("(ERROR)")].strip()
    return n


def _resolve_errors_section_body(
    node_id: str,
    blocks: list[tuple[str, str]],
) -> str | None:
    """Best-effort ERRORS underscore block body for this node id."""
    nid = _canonical_error_node_id(node_id)
    if not nid or not blocks:
        return None
    d = _best_detail_body(nid, blocks, errors_section=True)
    if d:
        return d
    d = _errors_block_fallback_by_path(nid, blocks)
    if d:
        return d
    d = _errors_block_longest_mentioning_node(nid, blocks)
    if d:
        return d
    path_part = nid.split("::", 1)[0].strip().replace("\\", "/") if nid else ""
    base = path_part.split("/")[-1] if path_part else ""
    # Path appears in title or early body (collection / import errors).
    if path_part:
        best: str | None = None
        best_rank = -1
        for title, body in blocks:
            bt = (body or "").strip()
            if not bt:
                continue
            tl = title.replace("\\", "/")
            blob_head = (title + "\n" + bt[:4800]).replace("\\", "/")
            rank = 0
            if path_part in tl:
                rank = 4
            elif base and base in tl:
                rank = 3
            elif path_part in blob_head:
                rank = 2
            elif base and base in blob_head:
                rank = 1
            if rank > best_rank or (rank == best_rank and rank > 0 and len(bt) > len(best or "")):
                best_rank = rank
                best = body
        if best_rank > 0 and best:
            return best
    # Collection errors: often a single "ERROR collecting …/test_x.py" block.
    if len(blocks) == 1:
        title, body = blocks[0]
        bt = (body or "").strip()
        if not bt:
            return None
        tl = title.replace("\\", "/")
        if base and base in tl:
            return body
        if path_part and path_part in tl:
            return body
    return None


def _errors_block_longest_mentioning_node(
    node_id: str,
    blocks: list[tuple[str, str]],
) -> str | None:
    """If titles miss the exact node id (e.g. parametrize brackets), pick the longest body that mentions it."""
    variants = [v for v in _node_id_match_variants(_canonical_error_node_id(node_id)) if len(v) >= 8]
    if not variants:
        return None
    best: str | None = None
    best_score = (-1, -1)  # (len(matched_variant), len(body))
    for title, body in blocks:
        bt = (body or "").strip()
        if not bt:
            continue
        blob = f"{title}\n{body}"
        mv = max((len(v) for v in variants if v in blob), default=0)
        if mv <= 0:
            continue
        key = (mv, len(bt))
        if key > best_score:
            best_score = key
            best = body
    return best


def _best_detail_body(
    node_id: str,
    blocks: list[tuple[str, str]],
    *,
    errors_section: bool,
) -> str | None:
    """Match a short-summary node id to a FAILURES/ERRORS underscore block."""
    if not blocks or not node_id:
        return None
    nid_norm = node_id.replace("\\", "/")
    path_part = nid_norm.split("::", 1)[0].strip() if nid_norm else ""
    path_base = path_part.split("/")[-1] if path_part else ""
    tail = _tail_node_id(node_id)
    tail_np = tail.split("[", 1)[0].strip() if tail else ""
    best_body: str | None = None
    best_key: tuple[int, int] = (-1, -1)  # (rank, title_len)

    for title, body in blocks:
        bt = (body or "").strip()
        if not bt:
            continue
        tl = title.strip()
        tl_norm = tl.replace("\\", "/")
        rank = 0
        if node_id == tl:
            rank = 5
        elif node_id.endswith("::" + tl):
            rank = 5
        elif tail and tl == tail:
            rank = 4
        elif tail_np and tl == tail_np:
            rank = 4
        elif tail_np and tail_np in tl and (".py" in tl or "::" in tl):
            rank = 3
        elif "::" in tl:
            ttail = _tail_node_id(tl)
            ttail_np = ttail.split("[", 1)[0].strip() if ttail else ""
            if ttail == tail or node_id.endswith("::" + ttail):
                rank = 4
            elif ttail_np and tail_np and ttail_np == tail_np:
                rank = 4
            elif node_id == tl or node_id.endswith(tl):
                rank = 3
        elif tail and (tl in node_id or node_id in tl):
            rank = 3 if tail in tl else 2
        elif tail and tl.endswith(tail):
            rank = 2
        elif tail_np and tl.endswith(tail_np):
            rank = 2
        if errors_section:
            m_of = re.search(r"\bof\s+(\S.+)$", tl, re.I)
            if m_of:
                ref = m_of.group(1).strip().rstrip("_").strip()
                for v in _node_id_match_variants(node_id):
                    if (
                        ref == v
                        or v.endswith(ref)
                        or ref.endswith(v)
                        or (len(ref) >= 10 and ref in v)
                        or (len(v) >= 10 and v in ref)
                    ):
                        rank = max(rank, 4)
                        break
            low_tl = tl.lower()
            if path_base and "collecting" in low_tl and path_base in tl_norm:
                rank = max(rank, 4)
            elif (
                path_base
                and path_base in tl_norm
                and "error" in low_tl
                and ("collect" in low_tl or "import" in low_tl or ".py" in tl_norm)
            ):
                rank = max(rank, 4)
            elif (
                path_part
                and len(path_part) > 5
                and path_part in tl_norm
                and ("collecting" in low_tl or "import" in low_tl)
            ):
                rank = max(rank, 4)
        key = (rank, len(tl))
        if rank > 0 and key > best_key:
            best_key = key
            best_body = body
    return best_body


def _enrich_from_pytest_detail_sections(
    lines: list[str],
    order_fail: list[str],
    order_err: list[str],
    *,
    failure_excerpts: dict[str, str],
    failed_reasons: dict[str, str],
    error_excerpts: dict[str, str],
    error_reasons: dict[str, str],
    failure_inline: dict[str, str],
    error_inline: dict[str, str],
) -> None:
    fr = _failures_section_range(lines)
    if fr:
        a, b = fr
        blocks = _extract_underscore_blocks(lines, a, b)
        for nid in order_fail:
            detail = _best_detail_body(nid, blocks, errors_section=False)
            if not detail:
                continue
            failure_excerpts[nid] = detail
            failed_reasons[nid] = _reason_from_excerpt(
                detail,
                failure_inline.get(nid, ""),
            )
    er = _errors_section_range(lines)
    if er:
        a, b = er
        blocks = _extract_underscore_blocks(lines, a, b)
        for nid in order_err:
            detail = _resolve_errors_section_body(nid, blocks)
            if not detail:
                continue
            error_excerpts[nid] = detail
            error_reasons[nid] = _reason_from_excerpt(
                detail,
                error_inline.get(nid, ""),
            )


def _is_interruption_or_collection_banner_reason(reason: str) -> bool:
    """Pytest session/collection banners, not a real failure message."""
    if not (reason and reason.strip()):
        return False
    t = reason.strip()
    low = t.lower()
    if "error during collection" in low:
        return True
    if "interrupted" in low and "collection" in low:
        return True
    if t.count("!") >= 16 and ("interrupted" in low or "collection" in low):
        return True
    if re.match(r"^[=!\s]+", t) and "interrupted" in low:
        return True
    return False


def _errors_block_fallback_by_path(
    node_id: str,
    blocks: list[tuple[str, str]],
) -> str | None:
    """Match ERRORS underscore blocks when node id is ``path.py`` (no ``::``) or title only mentions path."""
    nid = _canonical_error_node_id(node_id).strip().replace("\\", "/")
    if not nid:
        return None
    if "::" in nid:
        path_part = nid.split("::", 1)[0].strip()
    else:
        path_part = nid
    base = path_part.split("/")[-1] if path_part else ""
    variants = _node_id_match_variants(node_id)
    best_body: str | None = None
    best_score = -1
    for title, body in blocks:
        bt = (body or "").strip()
        if not bt:
            continue
        tl = title.strip()
        hay = f"{tl}\n{bt[:2000]}"
        score = -1
        if any(v and v in hay for v in variants if len(v) >= 8):
            score = len(bt) + 100
        elif nid in hay or path_part in hay:
            score = len(bt) + 100
        elif base and base in tl:
            score = len(bt) + 50
        elif base and base in bt[:2000]:
            score = len(bt)
        if score > best_score:
            best_score = score
            best_body = body
    return best_body


def _rewrite_noise_reasons_from_errors_section(
    lines: list[str],
    order_fail: list[str],
    order_err: list[str],
    *,
    failed_reasons: dict[str, str],
    failure_excerpts: dict[str, str],
    error_reasons: dict[str, str],
    error_excerpts: dict[str, str],
    failure_inline: dict[str, str],
    error_inline: dict[str, str],
) -> None:
    """Replace ``Interrupted: … collection`` style reasons with ERRORS-section summary."""
    er = _errors_section_range(lines)
    if not er:
        return
    a, b = er
    blocks = _extract_underscore_blocks(lines, a, b)
    if not blocks:
        return

    def patch(
        nid: str,
        reasons: dict[str, str],
        excerpts: dict[str, str],
        inlines: dict[str, str],
        *,
        allow_placeholder_replace: bool,
    ) -> None:
        cur = (reasons.get(nid) or "").strip()
        ex = (excerpts.get(nid) or "").strip()
        inl = (inlines.get(nid) or "").strip()
        noisy_cur = _is_interruption_or_collection_banner_reason(cur)
        noisy_ex = bool(ex) and (
            _is_interruption_or_collection_banner_reason(ex[:1200]) or _excerpt_looks_like_log_tail_noise(ex)
        )
        noisy_inl = _is_interruption_or_collection_banner_reason(inl)
        placeholder = allow_placeholder_replace and cur == _REASON_SINGLE_LINE_FALLBACK
        if not (noisy_cur or noisy_ex or noisy_inl or placeholder):
            return
        detail = _resolve_errors_section_body(nid, blocks)
        if not detail:
            return
        excerpts[nid] = detail
        reasons[nid] = _reason_from_excerpt(detail, inlines.get(nid, ""))

    for nid in order_err:
        patch(
            nid,
            error_reasons,
            error_excerpts,
            error_inline,
            allow_placeholder_replace=True,
        )
    for nid in order_fail:
        patch(
            nid,
            failed_reasons,
            failure_excerpts,
            failure_inline,
            allow_placeholder_replace=False,
        )


def _is_pytest_test_node_id(text: str) -> bool:
    """True if ``text`` is (or starts with) ``path/file.py::node[...]`` pytest node id."""
    t = text.strip()
    i = t.find("::")
    if i < 0:
        return False
    head = t[:i].strip()
    if not head or " " in head or "\t" in head:
        return False
    last_seg = head.replace("\\", "/").split("/")[-1]
    return bool(re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*\.py$", last_seg))


def _short_summary_rest_looks_like_pytest(rest: str) -> bool:
    """
    Keep pytest *short test summary* lines only.

    Rejects application / vllm log lines accidentally matched via ``] ERROR `` /
    ``] FAILED `` (e.g. ``[multiproc_executor.py:236] ERROR worker died ...``).
    """
    t = rest.strip()
    if not t or len(t) > 4000:
        return False
    if _is_pytest_test_node_id(t):
        return True
    if re.match(r"^collecting\s+[A-Za-z0-9_./\\-]+\.py\b", t, re.I):
        return True
    if re.match(r"^[A-Za-z0-9_./\\-]+\.py(\s+-\s|\s*$)", t):
        return True
    return False


def _match_failed_summary_rest(line: str) -> str | None:
    """Match pytest ``FAILED path::node`` short-summary line; allow log prefixes (timestamps, etc.)."""
    m = re.match(r"^\s*FAILED\s+(.+)$", line)
    if m:
        rest = m.group(1).strip()
        return rest if _short_summary_rest_looks_like_pytest(rest) else None
    m2 = re.search(r"(^|[\s\]])FAILED\s+(\S.+)$", line.rstrip())
    if m2:
        rest = m2.group(2).strip()
        return rest if _short_summary_rest_looks_like_pytest(rest) else None
    return None


def _relaxed_match_failed_summary_rest(line: str) -> str | None:
    """
    Match ``FAILED …`` when strict patterns miss (e.g. agent/extra tokens before ``FAILED``).

    Only use inside the *short test summary* region to limit false positives.
    """
    m = re.search(r"\bFAILED\s+(\S.+)$", line.rstrip())
    if not m:
        return None
    rest = m.group(1).strip()
    return rest if _short_summary_rest_looks_like_pytest(rest) else None


def _relaxed_match_error_summary_rest(line: str) -> str | None:
    """Same as `_relaxed_match_failed_summary_rest` for ``ERROR …`` summary lines."""
    m = re.search(r"\bERROR\s+(\S.+)$", line.rstrip())
    if not m:
        return None
    rest = m.group(1).strip()
    if rest.lower().startswith("collecting ") and "::" not in rest:
        return None
    return rest if _short_summary_rest_looks_like_pytest(rest) else None


def _match_error_summary_rest(line: str) -> str | None:
    m = re.match(r"^\s*ERROR\s+(.+)$", line)
    if m:
        rest = m.group(1).strip()
        return rest if _short_summary_rest_looks_like_pytest(rest) else None
    m2 = re.search(r"(^|[\s\]])ERROR\s+(\S.+)$", line.rstrip())
    if m2:
        rest = m2.group(2).strip()
        return rest if _short_summary_rest_looks_like_pytest(rest) else None
    return None


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def normalize_log_line(line: str) -> str:
    """Strip Buildkite/ANSI line prefixes so pytest section parsers see clean text."""
    s = line or ""
    s = BUILDKITE_BK_PREFIX_RE.sub("", s)
    s = ANSI_RE.sub("", s)
    s = s.lstrip("\x07")
    s = BUILDKITE_BK_LITERAL_PREFIX_RE.sub("", s)
    s = BUILDKITE_TS_PREFIX_RE.sub("", s)
    return s


def normalize_log_text(text: str) -> str:
    return "\n".join(normalize_log_line(line) for line in (text or "").splitlines())


def _line_is_buildkite_or_session_tail_noise(raw: str) -> bool:
    t = raw.strip()
    if not t:
        return False
    if "_bk;t=" in t or t.startswith("_bk;"):
        return True
    if "Error: The command exited with status" in t:
        return True
    if "user command error:" in t.lower():
        return True
    if t.startswith("^^^ +++"):
        return True
    if SESSION_LINE_RE.match(t) and COUNTS_FRAGMENT_RE.search(t):
        return True
    return False


def _excerpt_looks_like_log_tail_noise(excerpt: str) -> bool:
    t = (excerpt or "").strip()
    if not t:
        return False
    if any(_line_is_buildkite_or_session_tail_noise(ln) for ln in t.splitlines()[:6]):
        return True
    if "ERRORS" in t or "FAILURES" in t or "Traceback" in t:
        return False
    first = t.splitlines()[0].strip() if t else ""
    if COUNTS_FRAGMENT_RE.search(first) and re.search(r"\bpassed\b", first, re.I):
        return True
    return False


def _split_failed_rest(rest: str) -> tuple[str, str]:
    """Split ``path::node - message`` into node key and inline message."""
    rest = rest.strip()
    if " - " in rest:
        node, msg = rest.rsplit(" - ", 1)
        return node.strip(), msg.strip()
    return rest, ""


def _line_looks_traceback(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if t.startswith("E") and len(t) > 1 and t[1] in " \t":
        return True
    if t.startswith(">"):
        return True
    if t.startswith('File "') or t.startswith("File '"):
        return True
    if t.startswith("Traceback"):
        return True
    if re.match(r"^\s*raise\s+\w+", t):
        return True
    if re.match(
        r"^[\w.]*(Error|Exception|Warning|Exit|KeyboardInterrupt)\b",
        t,
    ):
        return True
    if t.startswith("___") and "___" in t[3:15]:
        return True
    return False


def _collect_excerpt(
    lines: list[str],
    start_idx: int,
    *,
    max_lines: int = 100,
    max_chars: int = 6000,
) -> str:
    buf: list[str] = []
    n = len(lines)
    end = min(start_idx + 1 + max_lines, n)
    for j in range(start_idx + 1, end):
        raw = lines[j].rstrip()
        if SECTION_RULE_RE.match(raw.strip()) and len(raw.strip()) > 10:
            if re.search(r"=+\s*FAILURES\s*=+|=+\s*ERRORS?\s*=+", raw, re.I):
                continue
            break
        if raw.strip().startswith("FAILED ") or raw.strip().startswith("ERROR "):
            if j > start_idx + 3:
                break
        if _line_is_buildkite_or_session_tail_noise(raw):
            continue
        if _line_looks_traceback(raw) or raw.strip():
            buf.append(raw)
        if sum(len(x) + 1 for x in buf) > max_chars:
            buf.append("... [truncated]")
            break
    return "\n".join(buf).strip()


_REASON_SINGLE_LINE_FALLBACK = "(Could not extract a single-line reason from the log; see excerpt below)"


def _line_stripping_pytest_e_prefix(s: str) -> str:
    """Strip pytest ``E   ...`` / ``E\t...`` log prefix, return content or original."""
    t = s.strip()
    if len(t) >= 2 and t[0] == "E" and t[1] in " \t":
        inner = t[1:].lstrip()
        return inner if inner else t
    return t


def _reason_from_excerpt(excerpt: str, inline: str) -> str:
    if (
        inline
        and inline not in ("(no inline reason; see log)",)
        and not _is_interruption_or_collection_banner_reason(inline)
    ):
        return inline
    elines = excerpt.splitlines()
    i = 0
    while i < len(elines) and (
        not elines[i].strip() or _is_interruption_or_collection_banner_reason(elines[i].strip())
    ):
        i += 1
    elines = elines[i:]
    excerpt = "\n".join(elines)
    # Deepest / most specific errors first (collection tracebacks)
    for line in reversed(elines):
        s = _line_stripping_pytest_e_prefix(line).strip()
        if re.search(r"\bModuleNotFoundError\b", s) or re.search(
            r"\bImportError\s*:\s*",
            s,
        ):
            return s[:2000]
    # Collection / import one-liners (often before Traceback)
    for line in elines:
        s = _line_stripping_pytest_e_prefix(line)
        low = s.lower()
        if not s:
            continue
        if "while importing test module" in low or "import error while" in low:
            return s[:2000]
        if re.match(
            r"^[\w.]*(ModuleNotFoundError|ImportError|SyntaxError|OSError)\b",
            s,
        ):
            return s[:2000]
    # Prefer last substantive E-line (skip caret-only / File-only noise)
    for line in reversed(elines):
        s = _line_stripping_pytest_e_prefix(line).strip()
        if not s or s == "^" or s.startswith('File "') or s.startswith("File '"):
            continue
        if s.startswith("Traceback"):
            continue
        if s.startswith("> ") or s.startswith(">"):
            continue
        if s.startswith("___") and "___" in s[1:12]:
            continue
        if re.match(r"^[\w.]*(Error|Exception|Warning|Exit)\b", s):
            return s[:2000]
        if _is_interruption_or_collection_banner_reason(s):
            continue
        if re.match(r"^[\W_=]+$", s):
            continue
        if len(s) > 12 and not re.match(r"^[=!\s]+$", s):
            return s[:2000]
    for line in reversed(elines):
        s = line.strip()
        if re.match(r"^[\w.]*(Error|Exception)\b", s):
            return s[:2000]
    if excerpt:
        cand = excerpt[:1500].strip() if len(excerpt) > 1500 else excerpt.strip()
        if not _is_interruption_or_collection_banner_reason(cand):
            return cand
    return _REASON_SINGLE_LINE_FALLBACK


def analyze_failure_cn(reason: str, excerpt: str) -> str:
    """Heuristic, short analysis for operators."""
    blob = f"{reason}\n{excerpt}".lower()
    if "assertionerror" in blob or "assertion error" in blob:
        return (
            "Category: assertion failure. Expected vs actual mismatch — check test logic, edge cases, and data setup."
        )
    if "timeouterror" in blob or "timeout" in blob and ("deadline" in blob or "pytest" in blob):
        return (
            "Category: timeout. Case or environment may be slow, deadlocked, or waiting on "
            "external resources — check timeout settings and dependencies."
        )
    if "cuda out of memory" in blob or "out of memory" in blob and ("gpu" in blob or "cuda" in blob):
        return (
            "Category: GPU/memory OOM. Try smaller batch, model size, or parallelism; "
            "check for unreleased tensors/cache."
        )
    if "oom" in blob and ("kill" in blob or "memory" in blob):
        return "Category: suspected OOM or process kill. Watch process memory and GPU peak usage."
    if "importerror" in blob or "modulenotfounderror" in blob:
        return (
            "Category: import failure. Often missing deps, PYTHONPATH, or env mismatch — "
            "compare CI image and local deps."
        )
    if "filenotfounderror" in blob or "no such file" in blob:
        return "Category: missing file. Check data paths, permissions, or build artifacts not generated/mounted."
    if "connection" in blob and ("refused" in blob or "error" in blob or "reset" in blob):
        return "Category: network connection. Downstream service may be down, or port/DNS misconfigured."
    if "permission denied" in blob:
        return "Category: permission denied. Check file/dir permissions or container user and mount options."
    if "keyboardinterrupt" in blob:
        return (
            "Category: interrupted. Build/run stopped manually or by the system — not necessarily a test logic failure."
        )
    if "xfail" in blob or "skip" in blob and "reason" in blob:
        return "Category: skip/xfail related. Confirm markers match intended behavior for this version."
    if "fixture" in blob and "error" in blob:
        return (
            "Category: fixture/setup error. Failure may occur before test logic — "
            "check conftest and resource init first."
        )
    if "error during collection" in blob or (
        "collecting" in blob and ("error" in blob or "import" in blob or "traceback" in blob)
    ):
        return (
            "Category: collection phase failure. "
            "The Reason column pulls from ERRORS/Traceback when possible; full stack is in the "
            "excerpt or raw log."
        )
    if not reason.strip() and not excerpt.strip():
        return "No excerpt available for auto-classification; open the full step log near the failure."
    return (
        "Could not auto-classify into a common type; use Reason and the excerpt "
        "(exception type, stack top file/line) for further triage."
    )


def _upsert_failure(
    key: str,
    inline: str,
    lines: list[str],
    idx: int,
    failed_reasons: dict[str, str],
    failure_excerpts: dict[str, str],
    order: list[str],
) -> None:
    ex = _collect_excerpt(lines, idx)
    reason = _reason_from_excerpt(ex, inline)
    if key not in failed_reasons:
        failed_reasons[key] = reason
        failure_excerpts[key] = ex
        order.append(key)
        return
    if len(reason) > len(failed_reasons[key]):
        failed_reasons[key] = reason
    if len(ex) > len(failure_excerpts[key]):
        failure_excerpts[key] = ex


def _upsert_error(
    key: str,
    inline: str,
    lines: list[str],
    idx: int,
    error_reasons: dict[str, str],
    error_excerpts: dict[str, str],
    order_err: list[str],
) -> None:
    ex = _collect_excerpt(lines, idx)
    reason = _reason_from_excerpt(ex, inline)
    if key not in error_reasons:
        error_reasons[key] = reason
        error_excerpts[key] = ex
        order_err.append(key)
        return
    if len(reason) > len(error_reasons[key]):
        error_reasons[key] = reason
    if len(ex) > len(error_excerpts[key]):
        error_excerpts[key] = ex


def _ingest_short_summary_extra(
    lines: list[str],
    failed_reasons: dict[str, str],
    error_reasons: dict[str, str],
    failure_excerpts: dict[str, str],
    error_excerpts: dict[str, str],
    order_fail: list[str],
    order_err: list[str],
    failure_inline: dict[str, str],
    error_inline: dict[str, str],
) -> None:
    """
    Re-scan *short test summary* only: ``… FAILED`` / ``… ERROR`` lines that strict
    matchers miss (extra tokens before keyword) still appear in this block.
    """
    sr = _short_summary_section_range(lines)
    if not sr:
        return
    a, b = sr
    for i in range(a + 1, b):
        line = lines[i]
        fail_rest = _match_failed_summary_rest(line)
        if fail_rest is None:
            fail_rest = _relaxed_match_failed_summary_rest(line)
        if fail_rest is not None:
            node, inline = _split_failed_rest(fail_rest)
            if not node:
                continue
            if inline:
                failure_inline[node] = inline
            _upsert_failure(
                node,
                inline,
                lines,
                i,
                failed_reasons,
                failure_excerpts,
                order_fail,
            )
            continue
        err_rest = _match_error_summary_rest(line)
        if err_rest is None:
            err_rest = _relaxed_match_error_summary_rest(line)
        if err_rest is not None:
            node, inline = _split_failed_rest(err_rest)
            if not node:
                continue
            if inline:
                error_inline[node] = inline
            _upsert_error(
                node,
                inline,
                lines,
                i,
                error_reasons,
                error_excerpts,
                order_err,
            )


def _ingest_failures_from_failures_section(
    lines: list[str],
    order_fail: list[str],
    failed_reasons: dict[str, str],
    failure_excerpts: dict[str, str],
    failure_inline: dict[str, str],
) -> None:
    """Add ``failed_nodes`` from ``==== FAILURES ====`` underscore titles when summary missed them."""
    fr = _failures_section_range(lines)
    if not fr:
        return
    a, b = fr
    blocks = _extract_underscore_blocks(lines, a, b)
    have = set(order_fail)
    for title, body in blocks:
        t = title.strip()
        if not t or t in have:
            continue
        if not _is_pytest_test_node_id(t):
            continue
        have.add(t)
        order_fail.append(t)
        body_text = (body or "").strip()
        inl = failure_inline.get(t, "")
        failed_reasons[t] = _reason_from_excerpt(body_text, inl)
        failure_excerpts[t] = body_text


def _error_key_already_present(key: str, order_err: list[str]) -> bool:
    key_tail = _tail_node_id(key)
    for existing in order_err:
        if existing == key:
            return True
        if key_tail and key_tail == _tail_node_id(existing):
            return True
        ev = set(_node_id_match_variants(existing))
        if any(v in ev for v in _node_id_match_variants(key) if len(v) >= 8):
            return True
    return False


def _parse_errors_section_node_key(title: str) -> str | None:
    t = title.strip().rstrip("_").strip()
    if not t:
        return None
    if _is_pytest_test_node_id(t):
        return t
    m = re.match(r"^ERROR at setup of\s+(.+)$", t, re.I)
    if m:
        return m.group(1).strip().rstrip("_").strip()
    m2 = re.match(r"^ERROR collecting\s+(.+)$", t, re.I)
    if m2:
        return m2.group(1).strip().rstrip("_").strip()
    return None


def _ingest_errors_from_errors_section(
    lines: list[str],
    order_err: list[str],
    error_reasons: dict[str, str],
    error_excerpts: dict[str, str],
    error_inline: dict[str, str],
) -> None:
    """Add ``error_nodes`` from ``==== ERRORS ====`` underscore titles when summary missed them."""
    er = _errors_section_range(lines)
    if not er:
        return
    a, b = er
    blocks = _extract_underscore_blocks(lines, a, b)
    for title, body in blocks:
        key = _parse_errors_section_node_key(title)
        if not key or _error_key_already_present(key, order_err):
            continue
        order_err.append(key)
        body_text = (body or "").strip()
        inl = error_inline.get(key, "")
        error_reasons[key] = _reason_from_excerpt(body_text, inl)
        error_excerpts[key] = body_text


def parse_pytest_log(text: str) -> dict[str, Any]:
    """
    Parse pytest-flavored log text.

    Returns keys: failed_nodes, error_nodes, failed_reasons, error_reasons,
    failure_excerpts, error_excerpts, failure_analyses, error_analyses, summary.
    """
    text = normalize_log_text(strip_ansi(text or ""))
    lines = text.splitlines()

    failed_reasons: dict[str, str] = {}
    error_reasons: dict[str, str] = {}
    failure_excerpts: dict[str, str] = {}
    error_excerpts: dict[str, str] = {}
    order_fail: list[str] = []
    order_err: list[str] = []
    failure_inline: dict[str, str] = {}
    error_inline: dict[str, str] = {}

    for i, line in enumerate(lines):
        fail_rest = _match_failed_summary_rest(line)
        if fail_rest is not None:
            node, inline = _split_failed_rest(fail_rest)
            if not node:
                continue
            if inline:
                failure_inline[node] = inline
            _upsert_failure(node, inline, lines, i, failed_reasons, failure_excerpts, order_fail)
            continue
        err_rest = _match_error_summary_rest(line)
        if err_rest is not None:
            node, inline = _split_failed_rest(err_rest)
            if not node:
                continue
            if inline:
                error_inline[node] = inline
            _upsert_error(node, inline, lines, i, error_reasons, error_excerpts, order_err)
            continue
        s = line.strip()
        prog_fail = re.match(r"^(.+::.+)\s+FAILED(?:\s+\[\d+%\])?\s*$", s)
        if prog_fail:
            node_pf = prog_fail.group(1).strip()
            if _is_pytest_test_node_id(node_pf):
                _upsert_failure(node_pf, "", lines, i, failed_reasons, failure_excerpts, order_fail)
                continue
        pf_suf = re.search(
            r"([\w/\.\-]+\.py(?:::[^\s\[]+(?:\[[^\]]+\])?)+)\s+FAILED(?:\s+\[\d+%\])?\s*$",
            s,
        )
        if pf_suf:
            node = pf_suf.group(1).strip()
            if _is_pytest_test_node_id(node):
                _upsert_failure(node, "", lines, i, failed_reasons, failure_excerpts, order_fail)
            continue
        prog_err = re.match(r"^(.+::.+)\s+ERROR(?:\s+\[\d+%\])?\s*$", s)
        if prog_err:
            node_pe = prog_err.group(1).strip()
            if _is_pytest_test_node_id(node_pe):
                _upsert_error(node_pe, "", lines, i, error_reasons, error_excerpts, order_err)
                continue
        pe_suf = re.search(
            r"([\w/\.\-]+\.py(?:::[^\s\[]+(?:\[[^\]]+\])?)+)\s+ERROR(?:\s+\[\d+%\])?\s*$",
            s,
        )
        if pe_suf:
            node = pe_suf.group(1).strip()
            if _is_pytest_test_node_id(node):
                _upsert_error(node, "", lines, i, error_reasons, error_excerpts, order_err)

    _ingest_short_summary_extra(
        lines,
        failed_reasons,
        error_reasons,
        failure_excerpts,
        error_excerpts,
        order_fail,
        order_err,
        failure_inline,
        error_inline,
    )
    _ingest_failures_from_failures_section(
        lines,
        order_fail,
        failed_reasons,
        failure_excerpts,
        failure_inline,
    )
    _ingest_errors_from_errors_section(
        lines,
        order_err,
        error_reasons,
        error_excerpts,
        error_inline,
    )

    _enrich_from_pytest_detail_sections(
        lines,
        order_fail,
        order_err,
        failure_excerpts=failure_excerpts,
        failed_reasons=failed_reasons,
        error_excerpts=error_excerpts,
        error_reasons=error_reasons,
        failure_inline=failure_inline,
        error_inline=error_inline,
    )
    _rewrite_noise_reasons_from_errors_section(
        lines,
        order_fail,
        order_err,
        failed_reasons=failed_reasons,
        failure_excerpts=failure_excerpts,
        error_reasons=error_reasons,
        error_excerpts=error_excerpts,
        failure_inline=failure_inline,
        error_inline=error_inline,
    )

    summary = None
    for line in reversed(lines):
        s = line.strip()
        if SESSION_LINE_RE.match(s) and COUNTS_FRAGMENT_RE.search(s):
            summary = s.strip().strip("=").strip()
            break
    if summary is None:
        for line in reversed(lines):
            s = line.strip()
            if "short test summary" in s.lower():
                continue
            if COUNTS_FRAGMENT_RE.search(s) and len(s) < 300:
                summary = s
                break

    failure_analyses = {k: analyze_failure_cn(failed_reasons[k], failure_excerpts.get(k, "")) for k in order_fail}
    error_analyses = {k: analyze_failure_cn(error_reasons[k], error_excerpts.get(k, "")) for k in order_err}

    return {
        "failed_nodes": list(order_fail),
        "error_nodes": list(order_err),
        "failed_reasons": failed_reasons,
        "error_reasons": error_reasons,
        "failure_excerpts": failure_excerpts,
        "error_excerpts": error_excerpts,
        "failure_analyses": failure_analyses,
        "error_analyses": error_analyses,
        "summary": summary,
    }


def extract_pytest_counts(summary: str | None) -> dict[str, int]:
    out = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    if not summary:
        return out
    low = summary.lower()
    for m in re.finditer(
        r"(\d+)\s+(passed|failed|skipped|errors?)\b",
        low,
    ):
        n = int(m.group(1))
        kind = m.group(2)
        if kind == "passed":
            out["passed"] = n
        elif kind == "failed":
            out["failed"] = n
        elif kind == "skipped":
            out["skipped"] = n
        elif kind == "error" or kind == "errors":
            out["error"] = n
    return out


_PYTEST_DURATION_RE = re.compile(r"\bin\s+([\d.]+)\s*s\b", re.IGNORECASE)


def extract_pytest_duration_display(summary: str | None) -> str:
    """
    Parse pytest session footer fragment ``in <seconds>s`` (e.g. ``... in 123.4s ...``).

    Returns a short display like ``123.4s``, or empty string if not found.
    """
    if not summary:
        return ""
    matches = list(_PYTEST_DURATION_RE.finditer(summary))
    if not matches:
        return ""
    return f"{matches[-1].group(1)}s"
