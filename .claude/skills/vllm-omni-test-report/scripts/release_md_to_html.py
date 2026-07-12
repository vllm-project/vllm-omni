#!/usr/bin/env python3
"""Convert compose_full_report.py Markdown output to a standalone HTML document (stdlib only)."""

from __future__ import annotations

import base64
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from report_html_theme import EDITORIAL_THEME_CSS, RELEASE_MARKDOWN_DOC_CSS  # noqa: E402

# Inserted after ``## Test conclusion``; replaced with interactive HTML (release) or static MD
# table (archive / .md only).
RELEASE_CONCLUSION_PLACEHOLDER = "@@RELEASE_CONCLUSION_WIDGET@@"

RELEASE_CONCLUSION_ITEMS: tuple[str, ...] = (
    # 0 — "guide" row (user-selectable, excluded from the final verdict)
    "UT coverage meets this iteration requirement(Guide)",
    # 1 — "guide" row (user-selectable, excluded from the final verdict)
    "Performance regression < 5%(Guide)",
    # 2 — auto: latest finished ready + merge builds have no failed/broken jobs
    # (Upload * Pipeline upload-only steps are skipped).
    "Latest L2&L3 pass rate is 100%",
    # 3 — manual (user-selectable)
    "Requirement completion rate > 85%",
    # 4 — auto: cumulative Outstanding DI from all open `label:bug` (self-calculated
    #     the same way as the Development report). Threshold rule: DI ≤ 30 → Pass,
    #     DI > 30 → Fail.
    "Remaining DI < 30",
    # 5 — auto: compose checks for open issues labeled ``critical``
    "No remaining critical issues",
    # 6 — manual (user-selectable): the assignee check is now a case-by-case
    #     judgement rather than a single auto-computed rule.
    "All remaining bugs have assignees",
)

# Indices in ``RELEASE_CONCLUSION_ITEMS`` whose cells are **(Guide)** markers.
# They render in the table and the user can still pick Pass/Fail, but the
# final Go / Rejected verdict ignores them.
RELEASE_CONCLUSION_GUIDE_ROW_INDICES: frozenset[int] = frozenset({0, 1})

# "Latest L2&L3 pass rate is 100%": latest finished ready + merge builds have no failed/broken jobs
CONCLUSION_L2_L3_ROW_INDEX = 2
# "Remaining DI < 30": auto-computed cumulative Outstanding DI; threshold ≤ 30 ⇒ Pass.
CONCLUSION_DI_ROW_INDEX = 4
# "No remaining critical issues": compose checks for open issues labeled ``critical``
CONCLUSION_CRITICAL_ROW_INDEX = 5
# "All remaining bugs have assignees" is now manual (user-selectable) — kept as
# a name for back-compat with any external caller that still imports the symbol.
CONCLUSION_ASSIGNEE_ROW_INDEX = 6


# Release chapter heading icons (24×24 stroke; same visual language as nightly HTML).
_RELEASE_SVG_CHECK = '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>'
_RELEASE_SVG_CHART = (
    '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
)
_RELEASE_SVG_LIST = (
    '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/>'
    '<line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/>'
    '<line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>'
)
_RELEASE_SVG_ALERT = (
    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
)
_RELEASE_SVG_INBOX = '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
_RELEASE_SVG_DATABASE = (
    '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
    '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
)
_RELEASE_SVG_LAYOUT = (
    '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>'
    '<rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>'
)
_RELEASE_SVG_CLOUD = '<path d="M18 10h-1.26A8 8 0 1 0 9 22h9a5 5 0 1 0 0-12z"/>'
_RELEASE_SVG_SERVER = (
    '<rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>'
    '<rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>'
    '<line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>'
)


def _release_inline_svg(paths: str, *, size: int = 22, extra_class: str = "") -> str:
    c = f"ico {extra_class}".strip()
    return (
        f'<svg class="{c}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'aria-hidden="true" focusable="false" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{paths}</svg>"
    )


def _release_h2_heading_plain(inner_html: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", inner_html)).strip()


def _release_section_theme(title_plain: str) -> tuple[str, str]:
    """
    Map H2 title to a CSS modifier (``release-section-card--*``) and inline SVG paths.
    """
    t = title_plain.strip()
    low = t.lower()
    if "test conclusion" in low:
        return "conclusion", _RELEASE_SVG_CHECK
    if "metrics" in low:
        return "metrics", _RELEASE_SVG_CHART
    if "failure analysis" in low:
        return "failure", _RELEASE_SVG_ALERT
    if "test result" in low:
        return "tests", _RELEASE_SVG_LIST
    if "issue tracking" in low:
        return "tracking", _RELEASE_SVG_ALERT
    if "open issues" in low:
        return "open-issues", _RELEASE_SVG_INBOX
    if "data source" in low:
        return "data", _RELEASE_SVG_DATABASE
    return "default", _RELEASE_SVG_LAYOUT


def test_conclusion_markdown_for_archive(
    *,
    l2_l3_row_ok: bool | None = None,
    l2_l3_row_detail: str = "",
    di_row_ok: bool | None = None,
    di_row_detail: str = "",
    critical_row_ok: bool | None = None,
    critical_row_detail: str = "",
    assignee_row_ok: bool | None = None,
    assignee_row_detail: str = "",
) -> str:
    """Static Markdown block (no ``##`` heading): table + **Test conclusion:** Go / Rejected.

    Guide rows (see :data:`RELEASE_CONCLUSION_GUIDE_ROW_INDICES`) still render
    in the table — they just don't influence the final Go / Rejected verdict.
    The ``All remaining bugs have assignees`` row is manual (user-selectable)
    in the archive .md it defaults to "Pass" and never affects the verdict.
    """
    lines = [
        "| Check item | Result |",
        "| --- | --- |",
    ]
    verdict_ok = True
    for i, item in enumerate(RELEASE_CONCLUSION_ITEMS):
        safe = item.replace("|", "\\|")
        cell = "Pass"
        extra = ""
        affects_verdict = i not in RELEASE_CONCLUSION_GUIDE_ROW_INDICES
        if i == CONCLUSION_L2_L3_ROW_INDEX and l2_l3_row_ok is not None:
            cell = "Pass" if l2_l3_row_ok else "Fail"
            if affects_verdict and not l2_l3_row_ok:
                verdict_ok = False
            if l2_l3_row_detail:
                extra = f" ({l2_l3_row_detail.replace('|', '/')})"
        elif i == CONCLUSION_DI_ROW_INDEX and di_row_ok is not None:
            cell = "Pass" if di_row_ok else "Fail"
            if affects_verdict and not di_row_ok:
                verdict_ok = False
            if di_row_detail:
                extra = f" ({di_row_detail.replace('|', '/')})"
        elif i == CONCLUSION_CRITICAL_ROW_INDEX and critical_row_ok is not None:
            cell = "Pass" if critical_row_ok else "Fail"
            if affects_verdict and not critical_row_ok:
                verdict_ok = False
            if critical_row_detail:
                extra = f" ({critical_row_detail.replace('|', '/')})"
        elif i == CONCLUSION_ASSIGNEE_ROW_INDEX and assignee_row_ok is not None:
            # Manual: never auto-affects the verdict (caller may still pass a
            # value, but the static archive always treats it as a guide row).
            cell = "Pass" if assignee_row_ok else "Fail"
            if assignee_row_detail:
                extra = f" ({assignee_row_detail.replace('|', '/')})"
        lines.append(f"| {safe} | {cell}{extra} |")
    vtxt = "Go" if verdict_ok else "Rejected"
    lines.extend(["", f"**Test conclusion:** {vtxt}", ""])
    return "\n".join(lines)


def release_conclusion_widget_html(
    *,
    l2_l3_row_ok: bool | None = None,
    l2_l3_row_detail: str = "",
    di_row_ok: bool | None = None,
    di_row_detail: str = "",
    critical_row_ok: bool | None = None,
    critical_row_detail: str = "",
    assignee_row_ok: bool | None = None,
    assignee_row_detail: str = "",
) -> str:
    """Interactive table + verdict (Go / Rejected) for ``.release-doc`` HTML.

    Automatic rows (non-clickable when ``*_row_ok`` is not ``None``): **L2&L3**,
    **Remaining DI**, **critical issues**. The ``All remaining bugs have
    assignees`` row is **manual** (always user-selectable, per the
    "用例自己选择结果" rule). Rows whose index is in
    :data:`RELEASE_CONCLUSION_GUIDE_ROW_INDICES` render in the table and stay
    user-selectable, but the final Go / Rejected verdict ignores them.
    """
    rows: list[str] = []
    for i, item in enumerate(RELEASE_CONCLUSION_ITEMS):
        auto_ok: bool | None = None
        row_detail = ""
        if i == CONCLUSION_L2_L3_ROW_INDEX:
            auto_ok = l2_l3_row_ok
            row_detail = l2_l3_row_detail
        elif i == CONCLUSION_DI_ROW_INDEX:
            auto_ok = di_row_ok
            row_detail = di_row_detail
        elif i == CONCLUSION_CRITICAL_ROW_INDEX:
            auto_ok = critical_row_ok
            row_detail = critical_row_detail
        elif i == CONCLUSION_ASSIGNEE_ROW_INDEX:
            # Manual: always user-selectable. The auto-computed signal
            # (assignee_row_ok) is intentionally ignored so the operator can
            # make a case-by-case judgement in the HTML widget.
            auto_ok = None
        is_auto = auto_ok is not None
        pass_on = bool(auto_ok) if is_auto else True
        pass_cls = "is-on" if pass_on else ""
        fail_cls = "" if pass_on else "is-on"
        pass_pressed = "true" if pass_on else "false"
        fail_pressed = "false" if pass_on else "true"
        auto_cls = " conc-auto" if is_auto else ""
        guide_attr = "1" if i in RELEASE_CONCLUSION_GUIDE_ROW_INDICES else "0"
        hint = ""
        if is_auto and row_detail:
            hint = f'<div class="conc-auto-hint">{html.escape(row_detail)}</div>'
        rows.append(
            f'<tr data-conc-row="{i}" data-conc-auto="{"1" if is_auto else "0"}"'
            f' data-conc-guide="{guide_attr}">'
            f"<td>{html.escape(item)}</td>"
            "<td>"
            f'<div class="conc-btns{auto_cls}" role="group" aria-label="Check result">'
            f'<button type="button" class="conc-btn conc-pass {pass_cls}" data-conc="pass"'
            f' aria-pressed="{pass_pressed}">Pass</button>'
            f'<button type="button" class="conc-btn conc-fail {fail_cls}" data-conc="fail"'
            f' aria-pressed="{fail_pressed}">Fail</button>'
            f"</div>{hint}</td></tr>"
        )
    rows_s = "\n".join(rows)
    return f"""<div class="release-conclusion-wrap">
<table class="release-conclusion-table">
<thead><tr><th>Check item</th><th>Result</th></tr></thead>
<tbody>
{rows_s}
</tbody>
</table>
<p class="release-verdict-line">Test conclusion:
<strong class="release-verdict" id="release-verdict-label">Go</strong></p>
</div>
<script>
(function () {{
  var wrap = document.querySelector('.release-conclusion-wrap');
  if (!wrap) return;
  function allPass() {{
    // Verdict ignores (Guide) rows: those are advisory only and must not
    // change the final Go / Rejected outcome.
    var rows = wrap.querySelectorAll('tbody tr');
    for (var i = 0; i < rows.length; i++) {{
      var tr = rows[i];
      if (tr.getAttribute('data-conc-guide') === '1') continue;
      var on = tr.querySelector('.conc-btn.conc-pass.is-on');
      if (!on) return false;
    }}
    return rows.length > 0;
  }}
  function syncVerdict() {{
    var el = document.getElementById('release-verdict-label');
    if (el) el.textContent = allPass() ? 'Go' : 'Rejected';
  }}
  wrap.addEventListener('click', function (e) {{
    var t = e.target;
    if (!t.classList || !t.classList.contains('conc-btn')) return;
    var tr = t.closest('tr');
    if (!tr) return;
    if (tr.getAttribute('data-conc-auto') === '1') return;
    var pass = t.classList.contains('conc-pass');
    var bp = tr.querySelector('.conc-pass');
    var bf = tr.querySelector('.conc-fail');
    if (!bp || !bf) return;
    if (pass) {{
      bp.classList.add('is-on');
      bf.classList.remove('is-on');
      bp.setAttribute('aria-pressed', 'true');
      bf.setAttribute('aria-pressed', 'false');
    }} else {{
      bf.classList.add('is-on');
      bp.classList.remove('is-on');
      bf.setAttribute('aria-pressed', 'true');
      bp.setAttribute('aria-pressed', 'false');
    }}
    syncVerdict();
  }});
  syncVerdict();
}})();
</script>"""


def apply_release_conclusion_placeholder(
    fragment: str,
    *,
    l2_l3_row_ok: bool | None = None,
    l2_l3_row_detail: str = "",
    di_row_ok: bool | None = None,
    di_row_detail: str = "",
    critical_row_ok: bool | None = None,
    critical_row_detail: str = "",
    assignee_row_ok: bool | None = None,
    assignee_row_detail: str = "",
) -> str:
    """Replace paragraph-wrapped placeholder with interactive widget."""
    escaped = html.escape(RELEASE_CONCLUSION_PLACEHOLDER, quote=False)
    p_wrap = f"<p>{escaped}</p>"
    widget = release_conclusion_widget_html(
        l2_l3_row_ok=l2_l3_row_ok,
        l2_l3_row_detail=l2_l3_row_detail,
        di_row_ok=di_row_ok,
        di_row_detail=di_row_detail,
        critical_row_ok=critical_row_ok,
        critical_row_detail=critical_row_detail,
        assignee_row_ok=assignee_row_ok,
        assignee_row_detail=assignee_row_detail,
    )
    if p_wrap in fragment:
        return fragment.replace(p_wrap, widget, 1)
    if RELEASE_CONCLUSION_PLACEHOLDER in fragment:
        return fragment.replace(RELEASE_CONCLUSION_PLACEHOLDER, widget, 1)
    return fragment


def materialize_release_conclusion_in_markdown(
    md: str,
    *,
    l2_l3_row_ok: bool | None = None,
    l2_l3_row_detail: str = "",
    di_row_ok: bool | None = None,
    di_row_detail: str = "",
    critical_row_ok: bool | None = None,
    critical_row_detail: str = "",
    assignee_row_ok: bool | None = None,
    assignee_row_detail: str = "",
) -> str:
    """Replace placeholder with static Markdown (archived .md or ``--format markdown`` output).

    Also materializes the UT-coverage editable-cell placeholder
    (``@@UT_CELL_INSERTION_POINT@@``) with static text so the archived
    Markdown is readable without the interactive JS handler.
    """
    result = md
    if RELEASE_CONCLUSION_PLACEHOLDER in result:
        block = test_conclusion_markdown_for_archive(
            l2_l3_row_ok=l2_l3_row_ok,
            l2_l3_row_detail=l2_l3_row_detail,
            di_row_ok=di_row_ok,
            di_row_detail=di_row_detail,
            critical_row_ok=critical_row_ok,
            critical_row_detail=critical_row_detail,
            assignee_row_ok=assignee_row_ok,
            assignee_row_detail=assignee_row_detail,
        )
        result = result.replace(RELEASE_CONCLUSION_PLACEHOLDER, block, 1)
    # Replace UT-coverage manual-edit placeholder with static Markdown text.
    # In the archived .md the cell reads as "(manual edit — editable in HTML)".
    result = result.replace("@@UT_CELL_INSERTION_POINT@@", "*manual edit — editable in HTML*")
    return result


def _italic_in_plain(s: str) -> str:
    """Apply *em* to raw ``s``; escape remaining text."""
    parts = re.split(r"(\*[^*]+\*)", s)
    out: list[str] = []
    for p in parts:
        if len(p) >= 2 and p[0] == "*" and p[-1] == "*" and not p.startswith("**"):
            inner = p[1:-1]
            out.append("<em>" + html.escape(inner) + "</em>")
        else:
            out.append(html.escape(p))
    return "".join(out)


def _bold_italic_plain(s: str) -> str:
    parts = re.split(r"(\*\*[^*]+\*\*)", s)
    res: list[str] = []
    for p in parts:
        if p.startswith("**") and p.endswith("**") and len(p) >= 4:
            res.append("<strong>" + _italic_in_plain(p[2:-2]) + "</strong>")
        else:
            res.append(_italic_in_plain(p))
    return "".join(res)


def _inline_text_with_links(s: str) -> str:
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", s):
        out.append(_bold_italic_plain(s[pos : m.start()]))
        url = html.escape(m.group(2), quote=True)
        inner = _bold_italic_plain(m.group(1))
        out.append(f'<a href="{url}">{inner}</a>')
        pos = m.end()
    out.append(_bold_italic_plain(s[pos:]))
    return "".join(out)


def inline_md_to_html(s: str) -> str:
    if not s:
        return ""
    # Preserve <a id="…"></a> anchor tags (used by per-GPU Failure Analysis
    # subsections and the matching Failed-column jumps in Execution Results
    # summary tables). Without this carve-out the helpers below would
    # html.escape the angle brackets, turning the anchor into literal text.
    _anchor_re = re.compile(r'(<a\s+id="[^"]*"\s*>\s*</a>)')
    anchors: list[str] = []

    def _stash_anchor(m: re.Match[str]) -> str:
        idx = len(anchors)
        anchors.append(m.group(1))
        return f"\x00ANCHOR_{idx}\x00"

    carved = _anchor_re.sub(_stash_anchor, s)

    # Preserve <span class="dev-snapshot-alert">…</span> as raw HTML so the
    # Development Metrics overview's red-alert rows render correctly (without
    # this carve-out the helper would html.escape the angle brackets, turning
    # the span into literal text). Nested Markdown inside the span is still
    # processed (bold/italic/links/backticks).
    spans: list[tuple[int, str]] = []

    def _stash(m: re.Match[str]) -> str:
        idx = len(spans)
        inner_html = inline_md_to_html(m.group(1))
        rendered = f'<span class="dev-snapshot-alert">{inner_html}</span>'
        spans.append((idx, rendered))
        return f"\x00DEV_SNAPSHOT_SPAN_{idx}\x00"

    carved = re.sub(
        r'<span class="dev-snapshot-alert">(.*?)</span>',
        _stash,
        carved,
        flags=re.DOTALL,
    )
    chunks: list[tuple[str, str]] = []
    last = 0
    for m in re.finditer(r"`([^`]+)`", carved):
        chunks.append(("t", carved[last : m.start()]))
        chunks.append(("c", m.group(1)))
        last = m.end()
    chunks.append(("t", carved[last:]))
    out: list[str] = []
    for kind, content in chunks:
        if kind == "c":
            out.append("<code>" + html.escape(content) + "</code>")
        else:
            out.append(_inline_text_with_links(content))
    rendered = "".join(out)
    if spans:
        for idx, span_html in spans:
            rendered = rendered.replace(f"\x00DEV_SNAPSHOT_SPAN_{idx}\x00", span_html)
    if anchors:
        for idx, anchor_html in enumerate(anchors):
            rendered = rendered.replace(f"\x00ANCHOR_{idx}\x00", anchor_html)
    return rendered


def _parse_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(bool(re.match(r"^:?-{3,}:?$", (c or "").strip())) for c in cells)


def _render_md_table(tbl_lines: list[str]) -> str:
    rows = [_parse_table_row(L) for L in tbl_lines]
    if not rows:
        return ""
    i = 0
    header = rows[i]
    i += 1
    if i < len(rows) and _is_separator_row(rows[i]):
        i += 1
    body_rows = rows[i:]
    parts = ["<table>", "<thead><tr>"]
    for h in header:
        parts.append(f"<th>{inline_md_to_html(h)}</th>")
    parts.extend(["</tr></thead>", "<tbody>"])
    for r in body_rows:
        parts.append("<tr>")
        # pad short rows
        while len(r) < len(header):
            r.append("")
        for c in r[: len(header)]:
            parts.append(f"<td>{inline_md_to_html(c)}</td>")
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>"])
    inner = "\n".join(parts)
    return f'<div class="table-scroll">\n{inner}\n</div>'


def convert_markdown_to_html_body(md: str) -> str:
    lines = md.splitlines()
    html_parts: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("|"):
            tbl_lines: list[str] = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            html_parts.append(_render_md_table(tbl_lines))
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = min(len(m.group(1)), 6)
            content = m.group(2)
            tag = f"h{level}"
            html_parts.append(f"<{tag}>{inline_md_to_html(content)}</{tag}>")
            i += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            items: list[str] = []
            while i < n:
                s = lines[i].strip()
                if re.match(r"^[-*]\s+", s):
                    items.append(re.sub(r"^[-*]\s+", "", s))
                    i += 1
                elif not s:
                    i += 1
                    break
                else:
                    break
            lis = "\n".join(f"<li>{inline_md_to_html(it)}</li>" for it in items)
            html_parts.append(f"<ul>\n{lis}\n</ul>")
            continue
        para: list[str] = []
        while i < n:
            s = lines[i]
            st = s.strip()
            if not st:
                break
            if st.startswith("|") or re.match(r"^#{1,6}\s", st) or re.match(r"^[-*]\s+", st):
                break
            para.append(s)
            i += 1
        text = " ".join(p.strip() for p in para)
        if text:
            html_parts.append(f"<p>{inline_md_to_html(text)}</p>")
    return "\n".join(html_parts)


def _wrap_release_report_h2_sections(html_fragment: str) -> str:
    """Wrap each top-level ``<h2>…`` block in a themed dashboard card (icon + accent)."""
    frag = html_fragment.strip()
    if not frag:
        return frag
    chunks = re.split(r"(?=<h2\b)", frag)
    out: list[str] = []
    for chunk in chunks:
        piece = chunk.strip()
        if not piece:
            continue
        hm = re.match(r"(?s)^<h2>([\s\S]*?)</h2>\s*([\s\S]*)$", piece)
        if not hm:
            out.append(f'<section class="panel release-section-card release-section-card--intro">\n{piece}\n</section>')
            continue
        h2_inner_html, rest = hm.group(1), hm.group(2)
        title_plain = _release_h2_heading_plain(h2_inner_html)
        theme, svg_paths = _release_section_theme(title_plain)
        icon = _release_inline_svg(svg_paths, size=22, extra_class="release-section-ico")
        new_h2 = (
            '<h2 class="release-section-h2">'
            '<span class="release-section-h2-row">'
            f'<span class="release-section-h2-ico" aria-hidden="true">{icon}</span>'
            f'<span class="release-section-h2-label">{h2_inner_html}</span>'
            "</span></h2>"
        )
        out.append(
            f'<section class="panel release-section-card release-section-card--{theme}">\n{new_h2}\n{rest}\n</section>'
        )
    return "\n".join(out)


def _test_result_h3_is_gpu_card(h3_block: str) -> bool:
    """True if the block opens with an ``h3`` for H100 / H200 / H800 / A100 (not Common stack)."""
    m = re.match(r"\s*<h3>([\s\S]*?)</h3>", h3_block.strip())
    if not m:
        return False
    inner_text = re.sub(r"<[^>]+>", "", m.group(1))
    inner_text = html.unescape(inner_text).strip()
    if not inner_text or inner_text.lower().startswith("common stack"):
        return False
    if re.fullmatch(r"H200", inner_text, re.IGNORECASE):
        return True
    if re.fullmatch(r"H800", inner_text, re.IGNORECASE):
        return True
    if re.fullmatch(r"A100", inner_text, re.IGNORECASE):
        return True
    # ``### H100``, ``### H100 (CI ...)``, ``### H100（CI ...）`` — reject ``H1000``-style labels.
    # Allow optional whitespace between ``H100`` and the opening paren so that the
    # Buildkite-side heading ``H100 (CI — Buildkite scheduled nightly)`` is still
    # treated as a GPU card and folded.
    return bool(re.match(r"H100(?:\s*[（(]|\Z)", inner_text, re.IGNORECASE))


def _balanced_outer_section_end(html: str, section_open_lt: int) -> int | None:
    """Index one past the matching ``</section>`` for outer ``<section`` at ``section_open_lt``."""
    if section_open_lt < 0 or not html.startswith("<section", section_open_lt):
        return None
    depth = 0
    i = section_open_lt
    n = len(html)
    while i < n:
        if html.startswith("</section>", i):
            depth -= 1
            if depth < 0:
                return None
            i += len("</section>")
            if depth == 0:
                return i
            continue
        if html.startswith("<section", i):
            depth += 1
            gt = html.find(">", i)
            if gt < 0:
                return None
            i = gt + 1
            continue
        i += 1
    return None


def _wrap_test_result_gpu_subcards(html_fragment: str) -> str:
    """Inside **Test Result**, wrap H100/H200/H800/A100 ``h3`` sections in nested cards."""
    # Anchor on ``--tests`` so we cannot match a later ``Test Result`` label inside another
    # section: a naïve ``[\s\S]*?`` between ``<h2…>`` and the label can span past ``</h2>``
    # and glue the wrong outer ``<section>`` to the tests heading (empty ``inner``, bad HTML).
    # Themed card: ``class="panel release-section-card release-section-card--tests"``.
    open_re = re.compile(
        r'<section\s+class="[^"]*\brelease-section-card--tests\b[^"]*">\s*'
        r"(?:"
        r'<h2 class="release-section-h2">(?:(?!</h2>).)*?'
        r'<span class="release-section-h2-label">\s*Test Result\s*</span>(?:(?!</h2>).)*?</h2>'
        r"|<h2>\s*Test Result\s*</h2>"
        r")\s*",
        re.IGNORECASE,
    )
    m = open_re.search(html_fragment)
    if not m:
        return html_fragment
    sec_end = _balanced_outer_section_end(html_fragment, m.start())
    if sec_end is None:
        return html_fragment
    close_start = sec_end - len("</section>")
    head = html_fragment[m.start() : m.end()]
    inner = html_fragment[m.end() : close_start].strip()
    chunks = re.split(r"(?=<h3\b)", inner)
    out_chunks: list[str] = []
    for i, raw in enumerate(chunks):
        piece = raw.strip()
        if not piece:
            continue
        if i > 0 and _test_result_h3_is_gpu_card(piece):
            out_chunks.append(f'<section class="panel test-result-gpu-card">\n{piece}\n</section>')
        else:
            out_chunks.append(piece)
    new_inner = "\n".join(out_chunks)
    return (
        html_fragment[: m.start()]
        + head
        + "\n"
        + new_inner
        + "\n"
        + html_fragment[close_start:sec_end]
        + html_fragment[sec_end:]
    )


_GPU_SECTION_OPEN = '<section class="panel test-result-gpu-card">'


def _plain_text_from_heading_inner(heading_el: str) -> str:
    """Plain text inside ``<hN>…</hN>`` (release MD→HTML headings are tag-only)."""
    m = re.match(r"(?s)^\s*<h[1-6]>([\s\S]*?)</h[1-6]>\s*$", heading_el.strip())
    if not m:
        m = re.search(r"<h[1-6]>([\s\S]*?)</h[1-6]>", heading_el)
        if not m:
            return ""
    frag = re.sub(r"<[^>]+>", "", m.group(1))
    return html.unescape(frag).strip()


def _wrap_section_h4_in_details(
    html_fragment: str,
    section_label: str,
) -> str:
    """Wrap ``#### ...`` headings inside a named top-level section (``<h2>...<span>section_label</span>...``)
    as collapsible ``<details class="report-subcard release-h-fold release-h4-fold">`` blocks.

    Each h4 becomes a fold; its body content is the paragraphs and tables that
    follow until the next h4 (or end of the section). The function uses
    :func:`_balanced_outer_section_end` to bound the body so the next top-level
    section (e.g. Open issues) is not eaten.
    """
    label_esc = re.escape(section_label)
    SECTION_RE = re.compile(
        rf"<h2\b(?:(?!</h2>).)*?release-section-h2-label[^>]*>\s*{label_esc}\s*</span>(?:(?!</h2>).)*?</h2>",
        re.IGNORECASE | re.DOTALL,
    )

    m = SECTION_RE.search(html_fragment)
    if not m:
        return html_fragment
    sec_start = html_fragment.rfind("<section", 0, m.start())
    if sec_start < 0:
        return html_fragment
    sec_end = _balanced_outer_section_end(html_fragment, sec_start)
    if sec_end is None:
        return html_fragment
    # The regex pattern ends at the closing </h2>, so ``m.end()`` already
    # points right past it. The body starts there.
    body_start = m.end()
    # sec_end points one past the matching ``</section>``; back up so the
    # closing tag itself stays in the tail (otherwise it gets swallowed into
    # the last wrapped <details>'s body_html).
    section_close = sec_end - len("</section>")
    body_end = section_close
    while body_end > body_start and html_fragment[body_end - 1] in " \t\r\n":
        body_end -= 1
    intro = html_fragment[:body_start]
    body = html_fragment[body_start:body_end]
    tail = html_fragment[section_close:]

    parts = re.split(r"(?=<h4\b)", body)
    out: list[str] = []
    if parts and parts[0].strip():
        out.append(parts[0])
    for p in parts[1:]:
        stripped = p.strip()
        pm = re.match(r"(?s)(<h4[^>]*>[\s\S]*?</h4>)([\s\S]*)", stripped)
        if not pm:
            if stripped:
                out.append(p)
            continue
        h4_el, rest = pm.group(1), pm.group(2)
        title = _plain_text_from_heading_inner(h4_el)
        # Nest any ``##### ...`` headings inside the h4 body as their own
        # collapsible details so per-model rows fold under their parent GPU
        # (Performance Data Comparison / Failure Analysis).
        body_html = _wrap_h5_blocks_in_details(rest.strip())
        out.append(
            '<details class="report-subcard release-h-fold release-h4-fold">'
            '<summary class="report-subcard-summary">'
            f'<span class="report-subcard-title">{html.escape(title)}</span>'
            "</summary>"
            f'<div class="report-subcard-body">{body_html}</div>'
            "</details>"
        )
    return intro + "\n".join(out) + tail


def _wrap_pdc_h4_in_details(html_fragment: str) -> str:
    """Wrap ``#### {GPU}`` headings inside **Performance Data Comparison** as collapsible
    ``<details>`` blocks. Each h4 becomes a fold; its body content is the paragraphs
    and tables that follow until the next h4 (or end of PDC section).
    """
    return _wrap_section_h4_in_details(html_fragment, "Performance Data Comparison")


def _wrap_failure_analysis_h4_in_details(html_fragment: str) -> str:
    """Wrap ``#### {GPU} failures`` headings inside **Failure Analysis** as collapsible
    ``<details>`` blocks. Each h4 becomes a fold; its body content is the paragraphs
    and tables that follow until the next h4 (or end of the Failure Analysis section).
    """
    return _wrap_section_h4_in_details(html_fragment, "Failure Analysis")


_BUGFIX_MONITOR_H3_RE = re.compile(
    r"<h3>\s*(Open|Closed)\s+bugfix\s+PRs\s*\([^)]+\)\s*</h3>",
    re.IGNORECASE,
)


def _wrap_bugfix_monitor_h3_in_details(html_fragment: str) -> str:
    """Wrap ``### Open bugfix PRs (N)`` / ``### Closed bugfix PRs (N)`` headings
    inside **Bugfix Monitor** as collapsible ``<details>`` blocks (the custom
    markdown→HTML converter doesn't preserve raw ``<details>`` HTML, so we
    emit ``### h3`` headings and post-process the body to wrap them).

    Each h3 becomes a fold; its body content is the tables that follow until
    the next h3 (or end of the section). The wrapper reuses the same
    ``report-subcard`` CSS class so the styling matches the rest of the report.
    """
    # Locate the actual ``<h2 class="release-section-h2">Bugfix Monitor`` heading
    # by matching the *complete* h2 element up to and including the closing
    # ``</h2>`` so the regex is bounded to a single h2 (the inner label text
    # is what we actually want to match). The non-greedy ``.*?`` inside the
    # ico span is safe; the ``</h2>`` at the tail anchors the match.
    h2_match = re.search(
        r'<h2 class="release-section-h2"[^>]*>'
        r"(?:(?!</h2>).)*?"
        r'<span class="release-section-h2-label">\s*Bugfix Monitor[^<]*</span>'
        r"(?:(?!</h2>).)*?"
        r"</h2>",
        html_fragment,
        re.DOTALL,
    )
    if not h2_match:
        return html_fragment
    # The enclosing <section class="panel release-section-card"> is the most
    # recent one before the h2 — rfind the full class string so we don't
    # accidentally pick up the intro section.
    sec_start = html_fragment.rfind('<section class="panel release-section-card', 0, h2_match.start())
    if sec_start < 0:
        return html_fragment
    # The regex match is bounded to the full ``<h2 …>…</h2>`` element, so
    # ``h2_match.end()`` points one past the closing ``</h2>``.
    head_end = h2_match.end()
    # _balanced_outer_section_end returns the index one past the matching
    # ``</section>``. The body sits between ``</h2>`` and that closing tag.
    sec_close = _balanced_outer_section_end(html_fragment, sec_start)
    if sec_close is None:
        return html_fragment
    prefix = html_fragment[:sec_start]
    head = html_fragment[sec_start:head_end]
    body = html_fragment[head_end : sec_close - len("</section>")]
    tail = html_fragment[sec_close - len("</section>") :]

    parts = re.split(r"(?=<h3\b)", body)
    out = [prefix, head]
    if parts and parts[0].strip():
        out.append(parts[0])
    for p in parts[1:]:
        stripped = p.strip()
        hm = re.match(r"(?s)(<h3[^>]*>[\s\S]*?</h3>)([\s\S]*)", stripped)
        if not hm:
            if stripped:
                out.append(p)
            continue
        h3_el, rest = hm.group(1), hm.group(2)
        title_text = re.sub(r"<[^>]+>", "", h3_el).strip()
        if not _BUGFIX_MONITOR_H3_RE.search(h3_el):
            out.append(p)
            continue
        out.append(
            '<details class="report-subcard release-h-fold release-bugfix-monitor-fold" open>'
            '<summary class="report-subcard-summary">'
            f'<span class="report-subcard-title">{html.escape(title_text)}</span>'
            "</summary>"
            f'<div class="report-subcard-body">{rest.strip()}</div>'
            "</details>"
        )
    out.append(tail)
    return "".join(out)


def _wrap_h5_blocks_in_details(fragment: str) -> str:
    fragment = fragment.strip()
    if not fragment or "<h5" not in fragment:
        return fragment
    parts = re.split(r"(?=<h5\b)", fragment)
    chunks: list[str] = []
    pre = parts[0].strip()
    if pre:
        chunks.append(pre)
    for p in parts[1:]:
        stripped = p.strip()
        pm = re.match(r"(?s)(<h5>[\s\S]*?</h5>)([\s\S]*)", stripped)
        if not pm:
            chunks.append(p)
            continue
        h5_el, rest = pm.group(1), pm.group(2)
        title = _plain_text_from_heading_inner(h5_el)
        body_html = rest.strip()
        chunks.append(
            '<details class="report-subcard release-h-fold release-h5-fold">'
            '<summary class="report-subcard-summary">'
            f'<span class="report-subcard-title">{html.escape(title)}</span>'
            "</summary>"
            f'<div class="report-subcard-body">{body_html}</div>'
            "</details>"
        )
    return "\n".join(chunks)


def _upgrade_status_cells_in_failure_tables(html_fragment: str) -> str:
    """Upgrade ``<td>Filed / Not an issue</td>`` cells in failure tables into interactive Status cells.

    The development report's ``### Summary`` section emits Markdown tables whose ``Status``
    column renders as plain text ``Filed / Not an issue``. This post-processor walks the HTML,
    finds those cells (only inside tables with a ``Status`` column header), and replaces
    each cell with the interactive ``<td class="fail-status-cell">…</td>`` markup that
    the ``fail-status-submit`` script (``nightly_local_log_report._fail_status_submit_script``)
    handles. A stable ``data-row-id`` is derived from the chain of preceding section
    headings (h2-h5) — *not* the inline h6 "Failures & errors" label — combined with
    the row index within the table so ``localStorage`` keys remain unique across
    report sections. The full heading **chain** is used (not only the nearest heading)
    because the same job name can appear under multiple GPU subsections in the Failure
    Analysis — e.g. ``Local job: full_moon_TTS_Function_Test_with_L4`` exists in both
    ``H200 failures`` and ``A100 failures``. Using only the nearest (h5) heading would
    collapse both rows into the same ``data-row-id`` and ``localStorage`` would apply
    the user's status choice across machines. By joining the chain
    ``"<h2>::<h3/h4>::<h5>::row-N"`` each GPU's row gets a unique key.
    """
    try:
        from nightly_local_log_report import _fail_status_cell_html  # local import
    except Exception:
        return html_fragment

    STATUS_HEADER = "<th>Status</th>"
    STATUS_CELL_RE = re.compile(r"<td>Filed\s*/\s*Not an issue</td>")
    TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
    SECTION_HEADING_RE = re.compile(r"<h([2-5])[^>]*>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)
    _heading_text_re = re.compile(r"<[^>]+>")

    def _table_replace(table_html: str, ctx_chain: list[str]) -> str:
        if STATUS_HEADER not in table_html:
            return table_html
        row_index = [0]

        def _cell_sub(m: re.Match[str]) -> str:
            chain_label = "::".join(ctx_chain) if ctx_chain else "report"
            row_id = f"{chain_label}::row-{row_index[0]}"
            row_index[0] += 1
            return _fail_status_cell_html(row_id)

        return STATUS_CELL_RE.sub(_cell_sub, table_html)

    # Pre-compute (position, level, text) tuples for h2-h5 headings so we can
    # rebuild the heading chain (in source order, descending level) that
    # precedes each table without re-scanning.
    headings = [
        (
            m.start(),
            int(m.group(1)),
            _heading_text_re.sub("", m.group(2)).strip()[:80] or "report",
        )
        for m in SECTION_HEADING_RE.finditer(html_fragment)
    ]

    rebuilt: list[str] = []
    pos = 0
    for tm in TABLE_RE.finditer(html_fragment):
        rebuilt.append(html_fragment[pos : tm.start()])
        # Build the heading chain for every heading that precedes this
        # table. We track each entry's level so a shallower heading (e.g.
        # an h4 "A100 failures" that follows an h5 "Local job: ..." inside
        # the previous h4 "H200 failures") properly pops the deeper entries
        # before being added — otherwise the same job name under multiple
        # GPU subsections would map to the same row-id.
        chain: list[tuple[int, str]] = []
        for h_pos, h_level, h_text in headings:
            if h_pos >= tm.start():
                break
            # Pop any deeper-or-equal entries so the chain reflects current
            # DOM scope, then add the new heading.
            while chain and chain[-1][0] >= h_level:
                chain.pop()
            chain.append((h_level, h_text))
        chain_texts = [t for _, t in chain]
        rebuilt.append(_table_replace(tm.group(0), chain_texts))
        pos = tm.end()
    rebuilt.append(html_fragment[pos:])
    return "".join(rebuilt)


def _upgrade_excerpt_cells_in_failure_tables(html_fragment: str) -> str:
    """Upgrade the 4th-column (Excerpt) cells in failure tables into a View log button + modal.

    The development report's ``### Summary`` section emits Markdown tables whose Excerpt
    column shows the truncated log content as plain text. This post-processor replaces
    the excerpt text with a ``View error log`` button + a hidden ``<pre>`` block,
    reusing ``nightly_local_log_report._excerpt_cell_html`` so the existing modal in
    ``_log_excerpt_modal_html`` works for it. Applies only to tables whose column
    headers include ``Excerpt (truncated)``; rows with ``—`` placeholders are left as-is.
    """
    try:
        from nightly_local_log_report import (  # local import
            _excerpt_cell_html,
            _excerpt_storage_id,
        )
    except Exception:
        return html_fragment

    EXCERPT_HEADER = "<th>Excerpt (truncated)</th>"
    TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
    SECTION_HEADING_RE = re.compile(r"<h([2-5])[^>]*>(.*?)</h\1>", re.DOTALL | re.IGNORECASE)
    _heading_text_re = re.compile(r"<[^>]+>")
    ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)

    headings = [
        (
            m.start(),
            int(m.group(1)),
            _heading_text_re.sub("", m.group(2)).strip()[:80] or "report",
        )
        for m in SECTION_HEADING_RE.finditer(html_fragment)
    ]

    def _table_replace(table_html: str, ctx_chain: list[str]) -> str:
        if EXCERPT_HEADER not in table_html:
            return table_html

        ctx_label = "::".join(ctx_chain) if ctx_chain else "report"

        rows = list(ROW_RE.finditer(table_html))
        if not rows:
            return table_html

        rebuilt_rows: list[str] = []
        rebuilt_rows.append(table_html[: rows[0].start()])
        for ri, rm in enumerate(rows):
            cells = list(TD_RE.finditer(rm.group(1)))
            if len(cells) < 4:
                rebuilt_rows.append(rm.group(0))
                continue
            excerpt_match = cells[3]
            excerpt_text = excerpt_match.group(1).strip()
            if not excerpt_text or excerpt_text == "&mdash;" or excerpt_text == "—":
                rebuilt_rows.append(rm.group(0))
                continue
            # Decode simple HTML entities to plain text for the excerpt storage.
            plain = re.sub(r"<[^>]+>", "", excerpt_text)
            plain = (
                plain.replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&#39;", "'")
            ).strip()
            # First cell holds the test-node title (used for the modal header).
            title_text = re.sub(r"<[^>]+>", "", cells[0].group(1)).strip()
            title_text = title_text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            storage_id = _excerpt_storage_id(ctx_label, title_text, ri)
            new_cell_inner = _excerpt_cell_html(plain, storage_id=storage_id, title=title_text)
            new_cell = f'<td class="excerpt-cell">{new_cell_inner}</td>'
            # Rebuild the row with the new excerpt cell.
            before = rm.group(1)[: excerpt_match.start()]
            after = rm.group(1)[excerpt_match.end() :]
            new_row_inner = before + new_cell + after
            rebuilt_rows.append(rm.group(0).replace(rm.group(1), new_row_inner, 1))
        rebuilt_rows.append(table_html[rows[-1].end() :])
        # Stitch back
        out_parts: list[str] = []
        cursor = 0
        for rm, replacement in zip(rows, rebuilt_rows[1:-1]):
            out_parts.append(table_html[cursor : rm.start()])
            out_parts.append(replacement)
            cursor = rm.end()
        out_parts.append(table_html[cursor:])
        return "".join(out_parts)

    rebuilt: list[str] = []
    pos = 0
    for tm in TABLE_RE.finditer(html_fragment):
        rebuilt.append(html_fragment[pos : tm.start()])
        # Build heading chain (same GPU-aware scoping logic as the Status
        # column upgrade above).
        chain: list[tuple[int, str]] = []
        for h_pos, h_level, h_text in headings:
            if h_pos >= tm.start():
                break
            while chain and chain[-1][0] >= h_level:
                chain.pop()
            chain.append((h_level, h_text))
        chain_texts = [t for _, t in chain]
        rebuilt.append(_table_replace(tm.group(0), chain_texts))
        pos = tm.end()
    rebuilt.append(html_fragment[pos:])
    return "".join(rebuilt)


def _upgrade_submit_issue_cells_in_failure_tables(html_fragment: str) -> str:
    """Upgrade ``<td>Submit issue</td>`` placeholder cells into interactive GitHub issue buttons.

    Mirrors the ``Submit issue`` button rendered by ``nightly_local_log_report._render_failures_table_html``
    (``_github_issue_button_cell`` + ``_github_issue_submit_script``). Rows in the Buildkite
    Failed test jobs table pass the Buildkite step id through the cell; local failure rows
    pass through ``local-<sha1>`` so the row id remains unique per issue template prefill.
    """
    try:
        from nightly_local_log_report import _github_issue_button_cell  # local import
    except Exception:
        return html_fragment

    SUBMIT_HEADER = "<th>Submit Issue</th>"
    TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
    ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
    HEAD_RE = re.compile(r"<th[^>]*>(.*?)</th>", re.DOTALL | re.IGNORECASE)

    def _table_replace(table_html: str) -> str:
        if SUBMIT_HEADER not in table_html:
            return table_html
        rows = list(ROW_RE.finditer(table_html))
        if not rows:
            return table_html
        rebuilt_rows: list[str] = []
        for ri, rm in enumerate(rows):
            inner = rm.group(1)
            if ri == 0:
                rebuilt_rows.append(rm.group(0))
                continue
            cells = list(TD_RE.finditer(inner))
            if not cells:
                rebuilt_rows.append(rm.group(0))
                continue
            # Identify Submit Issue column by header index. (Tables are uniform per row.)
            head_cells = list(HEAD_RE.finditer(rows[0].group(1)))
            submit_col = -1
            for hi, hm in enumerate(head_cells):
                if "Submit Issue" in hm.group(1):
                    submit_col = hi
                    break
            if submit_col < 0 or submit_col >= len(cells):
                rebuilt_rows.append(rm.group(0))
                continue
            submit_match = cells[submit_col]
            cell_text = submit_match.group(1).strip()
            # Accept either placeholder ("Submit issue", "—") or a bare buildkite step id.
            if cell_text not in {"Submit issue", "&mdash;", "—"} and not re.match(r"^[0-9a-f-]{6,}$", cell_text):
                # Already upgraded or unknown content — leave alone.
                rebuilt_rows.append(rm.group(0))
                continue
            new_cell = _github_issue_button_cell()
            new_inner = inner[: submit_match.start()] + new_cell + inner[submit_match.end() :]
            rebuilt_rows.append(rm.group(0).replace(inner, new_inner, 1))
        # Stitch back
        out_parts: list[str] = []
        cursor = 0
        for rm, replacement in zip(rows, rebuilt_rows):
            out_parts.append(table_html[cursor : rm.start()])
            out_parts.append(replacement)
            cursor = rm.end()
        out_parts.append(table_html[cursor:])
        return "".join(out_parts)

    rebuilt: list[str] = []
    pos = 0
    for tm in TABLE_RE.finditer(html_fragment):
        rebuilt.append(html_fragment[pos : tm.start()])
        rebuilt.append(_table_replace(tm.group(0)))
        pos = tm.end()
    rebuilt.append(html_fragment[pos:])
    return "".join(rebuilt)


def _wrap_summary_section_in_details(html_fragment: str) -> str:
    """Wrap the ``### Summary`` section (Test Result → first h3 after Common stack) in a collapsible ``<details>``.

    The Summary section is rendered as Markdown ``### Summary`` followed by intro text
    and per-GPU failure-analysis blocks. Here we wrap the content between
    ``<h3>Summary…</h3>`` and the **next** sibling element — either a GPU card
    (``<details class="test-result-gpu-card release-gpu-details…``) or an h2 (e.g.
    Open issues / Data source) — in a ``<details class="release-section-card release-section-details">``
    block (open by default) so it stays grouped and can be collapsed. This boundary
    detection is critical because the GPU cards use ``<span>`` titles (not h3), so
    a naive "next h3" search would walk past the cards and consume Open issues /
    Data source too.
    """
    SUMMARY_RE = re.compile(r"<h3[^>]*>Summary</h3>", re.IGNORECASE)
    NEXT_GPU_CARD_RE = re.compile(r'<details\s+class="[^"]*\btest-result-gpu-card\b[^"]*"', re.IGNORECASE)
    NEXT_H2_RE = re.compile(r"<h2\b", re.IGNORECASE)

    m = SUMMARY_RE.search(html_fragment)
    if not m:
        return html_fragment
    # Boundary = first GPU card OR first h2 after Summary. Whichever comes first.
    gpu_match = NEXT_GPU_CARD_RE.search(html_fragment, m.end())
    h2_match = NEXT_H2_RE.search(html_fragment, m.end())
    candidates = [c.start() for c in (gpu_match, h2_match) if c is not None]
    body_end = min(candidates) if candidates else len(html_fragment)
    intro = html_fragment[: m.start()]
    body = html_fragment[m.end() : body_end].strip()
    tail = html_fragment[body_end:]
    wrapped = (
        '<details class="panel test-result-gpu-card release-gpu-details release-gpu-details--summary" open>\n'
        '<summary class="release-gpu-details-summary">\n'
        '<span class="release-gpu-summary-row">'
        '<span class="release-gpu-summary-ico" aria-hidden="true">'
        '<svg class="ico release-gpu-summary-ico" width="20" height="20" viewBox="0 0 24 24" '
        'aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 12h18M3 6h18M3 18h18"/></svg>'
        "</span>"
        '<span class="release-gpu-details-title">Summary</span>'
        "</span></summary>\n"
        f'<div class="release-gpu-details-body">\n{body}\n</div>\n'
        "</details>\n"
    )
    return intro + wrapped + tail


def _wrap_h4_blocks_in_details(fragment: str) -> str:
    fragment = fragment.strip()
    if not fragment or "<h4" not in fragment:
        return fragment
    parts = re.split(r"(?=<h4\b)", fragment)
    chunks: list[str] = []
    pre = parts[0].strip()
    if pre:
        chunks.append(pre)
    for p in parts[1:]:
        stripped = p.strip()
        pm = re.match(r"(?s)(<h4>[\s\S]*?</h4>)([\s\S]*)", stripped)
        if not pm:
            chunks.append(p)
            continue
        h4_el, rest = pm.group(1), pm.group(2)
        title = _plain_text_from_heading_inner(h4_el)
        body_html = _wrap_h5_blocks_in_details(rest.strip())
        chunks.append(
            '<details class="report-subcard release-h-fold release-h4-fold">'
            '<summary class="report-subcard-summary">'
            f'<span class="report-subcard-title">{html.escape(title)}</span>'
            "</summary>"
            f'<div class="report-subcard-body">{body_html}</div>'
            "</details>"
        )
    return "\n".join(chunks)


def _gpu_details_extra_classes(title: str) -> str:
    t = (title or "").strip()
    if re.fullmatch(r"H200", t, re.IGNORECASE):
        return " release-gpu-details--h200"
    if re.fullmatch(r"H800", t, re.IGNORECASE):
        return " release-gpu-details--h800"
    if re.fullmatch(r"A100", t, re.IGNORECASE):
        return " release-gpu-details--a100"
    if re.match(r"H100", t, re.IGNORECASE):
        return " release-gpu-details--h100"
    return ""


def _gpu_summary_icon_markup(title: str) -> str:
    t = (title or "").strip()
    paths = _RELEASE_SVG_CLOUD if re.match(r"H100", t, re.IGNORECASE) else _RELEASE_SVG_SERVER
    return _release_inline_svg(paths, size=20, extra_class="release-gpu-summary-ico")


def _gpu_short_title(title: str) -> str:
    """Reduce GPU h3 title to its short token (H100 / H200 / H800 / A100).

    ``### H100 (CI — Buildkite scheduled nightly)`` should still display as ``H100`` in the
    collapsible summary. Falls back to the original title when no token is found.
    """
    t = (title or "").strip()
    if not t:
        return t
    m = re.match(r"\s*(H100|H200|H800|A100)\b", t, re.IGNORECASE)
    return m.group(1).upper() if m else t


def _convert_gpu_section_to_collapsible_details(full_section: str) -> str:
    """Turn GPU ``section`` into default-closed ``details``; fold ``h4`` / ``h5`` inside."""
    fs = full_section.strip()
    mo = re.match(r'^<section class="panel test-result-gpu-card">\s*', fs)
    if not mo:
        return full_section
    end = _balanced_outer_section_end(fs, 0)
    if end is None or end != len(fs):
        return full_section
    inner_close = end - len("</section>")
    inner = fs[mo.end() : inner_close].strip()
    hm = re.match(r"(?s)^(<h3>[\s\S]*?</h3>)\s*([\s\S]*)", inner)
    title = ""
    if hm:
        title = _plain_text_from_heading_inner(hm.group(1))
        body = _wrap_h4_blocks_in_details(hm.group(2).strip())
        short = _gpu_short_title(title)
        title_esc = html.escape(short) if short else "…"
    else:
        title_esc = "…"
        body = _wrap_h4_blocks_in_details(inner)
    gpu_x = _gpu_details_extra_classes(title)
    g_ico = _gpu_summary_icon_markup(title)
    return (
        f'<details class="panel test-result-gpu-card release-gpu-details{gpu_x}">'
        '<summary class="release-gpu-details-summary">'
        '<span class="release-gpu-summary-row">'
        f'<span class="release-gpu-summary-ico" aria-hidden="true">{g_ico}</span>'
        f'<span class="release-gpu-details-title">{title_esc}</span>'
        "</span>"
        "</summary>"
        f'<div class="release-gpu-details-body">{body}</div>'
        "</details>"
    )


def _fold_test_result_gpu_sections(html_fragment: str) -> str:
    """Fold **Test Result** GPU panels: outer ``details`` + inner ``h4``/``h5`` cards (all default-closed)."""
    pos = 0
    out: list[str] = []
    while True:
        idx = html_fragment.find(_GPU_SECTION_OPEN, pos)
        if idx < 0:
            out.append(html_fragment[pos:])
            break
        out.append(html_fragment[pos:idx])
        end = _balanced_outer_section_end(html_fragment, idx)
        if end is None:
            out.append(html_fragment[idx:])
            break
        block = html_fragment[idx:end]
        out.append(_convert_gpu_section_to_collapsible_details(block))
        pos = end
    return "".join(out)


_RELEASE_SECTION_CARD_MARKER = '<section class="panel release-section-card'


def _fold_release_report_section_cards(html_fragment: str) -> str:
    """Turn each H2-headed ``release-section-card`` (Test conclusion / Metrics / …) into default-closed ``details``."""
    pos = 0
    out: list[str] = []
    while True:
        idx = html_fragment.find(_RELEASE_SECTION_CARD_MARKER, pos)
        if idx < 0:
            out.append(html_fragment[pos:])
            break
        out.append(html_fragment[pos:idx])
        end = _balanced_outer_section_end(html_fragment, idx)
        if end is None:
            out.append(html_fragment[idx:])
            break
        close_start = end - len("</section>")
        gt = html_fragment.find(">", idx)
        if gt < 0 or gt >= close_start:
            out.append(html_fragment[idx:end])
            pos = end
            continue
        open_tag = html_fragment[idx : gt + 1].strip()
        inner = html_fragment[gt + 1 : close_start].strip()
        mo = re.match(r"^<section\s+class=\"([^\"]+)\"\s*>$", open_tag, re.IGNORECASE)
        if not mo:
            out.append(html_fragment[idx:end])
            pos = end
            continue
        classes = mo.group(1).strip()
        if "release-section-details" in classes.split():
            out.append(html_fragment[idx:end])
            pos = end
            continue
        hm = re.match(
            r"^(<h2 class=\"release-section-h2\">[\s\S]*?</h2>)\s*([\s\S]*)$",
            inner,
            re.DOTALL,
        )
        if not hm:
            out.append(html_fragment[idx:end])
            pos = end
            continue
        h2_block, body = hm.group(1), hm.group(2).strip()
        new_classes = f"{classes} release-section-details"
        out.append(
            f'<details class="{new_classes}">\n'
            f'<summary class="release-section-fold-summary">\n{h2_block}\n</summary>\n'
            f'<div class="release-section-fold-body">\n{body}\n</div>\n'
            "</details>\n"
        )
        pos = end
    return "".join(out)


def _markdown_skip_document_h1(md: str) -> str:
    """Remove the first ``# document title`` so it is not repeated below the top bar."""
    lines = md.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return md
    head = lines[i]
    if not re.match(r"^#\s+\S", head) or head.startswith("##"):
        return md
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


def _release_brand_clipboard_svg() -> str:
    return (
        '<svg class="ico brand-ico" width="30" height="30" viewBox="0 0 24 24" '
        'aria-hidden="true" focusable="false" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
        '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
        "</svg>"
    )


def _default_archive_filename(title: str, generated_utc: str) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", generated_utc.strip())
    date_part = m.group(1) if m else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = "vllm-omni-test-report"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", title).strip("-")[:72]
    if slug:
        return f"{slug}-{date_part}.md"
    return f"{base}-{date_part}.md"


# Ensure <details> toggles even when inline SVG / ::before hit-testing blocks native behavior.
_RELEASE_DETAILS_TOGGLE_SCRIPT = """<script>
(function () {
  document.querySelectorAll(".release-doc details").forEach(function (d) {
    var s = d.querySelector(":scope > summary");
    if (!s || s.getAttribute("data-release-sum-tog") === "1") return;
    s.setAttribute("data-release-sum-tog", "1");
    s.addEventListener(
      "click",
      function (ev) {
        if (ev.button !== 0) return;
        if (ev.target && ev.target.closest && ev.target.closest("a, button")) return;
        ev.preventDefault();
        d.open = !d.open;
      },
      true
    );
  });
})();
</script>"""


# Interactive Status column handler for failure-analysis tables. Two-button flow:
# Filed -> prompt for issue number; Not an issue -> switch directly. State persisted in
# localStorage keyed by ``data-row-id`` so reloading the report keeps the chosen
# status. Defined in ``nightly_local_log_report._fail_status_submit_script``.
try:
    from nightly_local_log_report import _fail_status_submit_script as _fail_status_fn  # noqa: E402

    _FAIL_STATUS_SCRIPT = _fail_status_fn()
except Exception:  # pragma: no cover — graceful fallback when import fails
    _FAIL_STATUS_SCRIPT = ""

# Modal markup + click handler for the "View error log" button rendered into the
# upgraded Excerpt cells. Defined in ``nightly_local_log_report._log_excerpt_modal_html``
# + the click handler inside ``_github_issue_submit_script`` (or alongside).
try:
    from nightly_local_log_report import _log_excerpt_modal_html as _log_modal_fn  # noqa: E402

    _LOG_EXCERPT_MODAL_HTML = _log_modal_fn()
except Exception:  # pragma: no cover
    _LOG_EXCERPT_MODAL_HTML = ""

# GitHub issue submit handler — same script registered by the dedicated nightly
# report. Provides the modal that the upgraded Submit Issue cells open.
try:
    from nightly_local_log_report import _github_issue_submit_script as _gh_submit_fn  # noqa: E402

    _GITHUB_ISSUE_SUBMIT_SCRIPT = _gh_submit_fn()
except Exception:  # pragma: no cover
    _GITHUB_ISSUE_SUBMIT_SCRIPT = ""

# Editable UT coverage cell handler — used by compose_full_report.py
# ``--kind development`` Metrics overview so report owners can fill the value
# in manually and persist it across reloads via localStorage. Defined in
# ``nightly_local_log_report._ut_coverage_submit_script``.
try:
    from nightly_local_log_report import _ut_coverage_submit_script as _ut_cov_fn  # noqa: E402

    _UT_COVERAGE_SUBMIT_SCRIPT = _ut_cov_fn()
except Exception:  # pragma: no cover
    _UT_COVERAGE_SUBMIT_SCRIPT = ""

# In-page modal for the UT coverage cell. Same pattern as ``_log_excerpt_modal_html``
# (works in iframe contexts where ``window.prompt`` is blocked by ``sandbox``).
try:
    from nightly_local_log_report import _ut_coverage_modal_html as _ut_cov_modal_fn  # noqa: E402

    _UT_COVERAGE_MODAL_HTML = _ut_cov_modal_fn()
except Exception:  # pragma: no cover
    _UT_COVERAGE_MODAL_HTML = ""

# In-page modal for the fail-status cells in failure-analysis tables
# (``Filed`` / ``Not an issue`` flow). Same pattern as ``_log_excerpt_modal_html``;
# replaces the legacy ``window.prompt()`` so the cells stay editable when the
# report is embedded via iframe (kanban Reports page).
try:
    from nightly_local_log_report import _fail_status_modal_html as _fsm_fn  # noqa: E402

    _FAIL_STATUS_MODAL_HTML = _fsm_fn()
except Exception:  # pragma: no cover
    _FAIL_STATUS_MODAL_HTML = ""


def wrap_html_document(
    *,
    title: str,
    body_inner: str,
    generated_utc: str | None = None,
    tagline: str = "Release · CI test report",
    archive_markdown: str | None = None,
    archive_download_name: str | None = None,
) -> str:
    t = html.escape(title)
    when = generated_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = f'<p class="meta generated-meta">Generated: {html.escape(when)}</p>'
    brand = _release_brand_clipboard_svg()
    tl = html.escape(tagline)
    css = EDITORIAL_THEME_CSS + "\n" + RELEASE_MARKDOWN_DOC_CSS
    dl_name = archive_download_name or _default_archive_filename(title, when)
    dl_name_esc = html.escape(dl_name, quote=True)
    archive_top = ""
    archive_scripts = ""
    if archive_markdown is not None:
        b64 = base64.b64encode(archive_markdown.encode("utf-8")).decode("ascii")
        b64_json = json.dumps(b64)
        archive_top = (
            '<div class="top-bar-actions">'
            '<button type="button" class="btn-release-archive" '
            'id="release-archive-md-btn" '
            f'data-download-name="{dl_name_esc}" '
            'title="Download a Markdown file matching this report (for archive or patch_report_*.py)">'
            "Archive Markdown</button>"
            "</div>"
        )
        archive_scripts = (
            f'<script type="application/json" id="release-archive-md-b64">{b64_json}</script>\n'
            """<script>
(function () {
  var btn = document.getElementById("release-archive-md-btn");
  var el = document.getElementById("release-archive-md-b64");
  if (!btn || !el) return;
  btn.addEventListener("click", function () {
    try {
      var b64 = JSON.parse(el.textContent || '""');
      var bin = atob(b64);
      var bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      var text = new TextDecoder("utf-8").decode(bytes);
      var blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = btn.getAttribute("data-download-name") || "vllm-omni-test-report.md";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Archive failed: " + e);
    }
  });
})();
</script>"""
        )
    top_bar = (
        '<div class="top-bar"><div class="shell top-bar-inner">'
        '<div class="brand">'
        f'<div class="brand-mark">{brand}</div>'
        '<div class="brand-copy">'
        f"<h1>{t}</h1>"
        f'<p class="tagline">{tl}</p>'
        "</div></div>"
        f"{archive_top}"
        "</div></div>"
    )
    shell = f'<div class="shell"><div class="release-doc">{meta}\n{body_inner}</div></div>'
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>
{css}
</style>
</head>
<body>
{top_bar}
{shell}
{_LOG_EXCERPT_MODAL_HTML}
{_FAIL_STATUS_MODAL_HTML}
{_UT_COVERAGE_MODAL_HTML}
{archive_scripts}
{_RELEASE_DETAILS_TOGGLE_SCRIPT}
{_FAIL_STATUS_SCRIPT}
{_GITHUB_ISSUE_SUBMIT_SCRIPT}
{_UT_COVERAGE_SUBMIT_SCRIPT}
</body>
</html>
"""


def convert_release_report_markdown(
    md: str,
    *,
    archive_download_name: str | None = None,
    l2_l3_row_ok: bool | None = None,
    l2_l3_row_detail: str = "",
    di_row_ok: bool | None = None,
    di_row_detail: str = "",
    critical_row_ok: bool | None = None,
    critical_row_detail: str = "",
    assignee_row_ok: bool | None = None,
    assignee_row_detail: str = "",
) -> str:
    """Full HTML document from a release report Markdown string.

    Pass ``*_row_ok`` for automatic conclusion rows; when omitted, that row defaults to Pass in the archive table.
    """
    title = "vLLM-Omni Test Report"
    for line in md.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_body = _markdown_skip_document_h1(md)
    body = convert_markdown_to_html_body(md_body)
    # Substitute the editable UT coverage placeholder with raw HTML. The cell
    # is intentionally rendered as an inline ``<span>`` (not its own ``<td>``)
    # so the value sits inside the Result column and inherits the surrounding
    # font/format. The JS handler is registered in ``wrap_html_document`` below.
    try:
        from nightly_local_log_report import _ut_coverage_cell_html as _ut_cell_fn  # noqa: E402

        _ut_cell_html = _ut_cell_fn("dev-metrics::ut-coverage", initial_value="—")
    except Exception:
        _ut_cell_html = (
            '<span class="ut-coverage-cell" data-row-id="dev-metrics::ut-coverage" '
            'data-original="—"><span class="ut-coverage-display">'
            '<button type="button" class="ut-coverage-btn" data-ut-action="edit">—</button>'
            "</span></span>"
        )
    body = body.replace("@@UT_CELL_INSERTION_POINT@@", _ut_cell_html)
    body = apply_release_conclusion_placeholder(
        body,
        l2_l3_row_ok=l2_l3_row_ok,
        l2_l3_row_detail=l2_l3_row_detail,
        di_row_ok=di_row_ok,
        di_row_detail=di_row_detail,
        critical_row_ok=critical_row_ok,
        critical_row_detail=critical_row_detail,
        assignee_row_ok=assignee_row_ok,
        assignee_row_detail=assignee_row_detail,
    )
    body = _wrap_release_report_h2_sections(body)
    body = _wrap_test_result_gpu_subcards(body)
    body = _fold_test_result_gpu_sections(body)
    body = _upgrade_excerpt_cells_in_failure_tables(body)
    body = _upgrade_submit_issue_cells_in_failure_tables(body)
    body = _upgrade_status_cells_in_failure_tables(body)
    body = _wrap_summary_section_in_details(body)
    body = _wrap_failure_analysis_h4_in_details(body)
    body = _wrap_pdc_h4_in_details(body)
    body = _wrap_bugfix_monitor_h3_in_details(body)
    body = _fold_release_report_section_cards(body)
    archive_markdown = materialize_release_conclusion_in_markdown(
        md,
        l2_l3_row_ok=l2_l3_row_ok,
        l2_l3_row_detail=l2_l3_row_detail,
        di_row_ok=di_row_ok,
        di_row_detail=di_row_detail,
        critical_row_ok=critical_row_ok,
        critical_row_detail=critical_row_detail,
        assignee_row_ok=assignee_row_ok,
        assignee_row_detail=assignee_row_detail,
    )
    return wrap_html_document(
        title=title,
        body_inner=body,
        generated_utc=when,
        archive_markdown=archive_markdown,
        archive_download_name=archive_download_name,
    )
