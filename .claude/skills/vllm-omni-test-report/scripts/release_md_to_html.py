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
    "UT coverage meets this iteration requirement",
    "Latest L2&L3 pass rate is 100%",
    "Requirement completion rate > 85%",
    "Performance regression < 5%",
    "Remaining DI < 30",
    "No remaining critical issues",
    "All remaining bugs have assignees",
)

# "Latest L2&L3 pass rate is 100%": latest finished ready + merge builds have no failed/broken jobs
CONCLUSION_L2_L3_ROW_INDEX = 1
# "Remaining DI < 30": precomputed by compose from open bugs in stats window by priority labels
CONCLUSION_DI_ROW_INDEX = 4
# "No remaining critical issues": compose checks for open issues labeled ``critical``
CONCLUSION_CRITICAL_ROW_INDEX = 5
# "All remaining bugs have assignees": open label:bug assignees
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
    """Static Markdown block (no ``##`` heading): table + **Test conclusion:** Go / Rejected."""
    lines = [
        "| Check item | Result |",
        "| --- | --- |",
    ]
    verdict_ok = True
    for i, item in enumerate(RELEASE_CONCLUSION_ITEMS):
        safe = item.replace("|", "\\|")
        cell = "Pass"
        extra = ""
        if i == CONCLUSION_L2_L3_ROW_INDEX and l2_l3_row_ok is not None:
            cell = "Pass" if l2_l3_row_ok else "Fail"
            if not l2_l3_row_ok:
                verdict_ok = False
                if l2_l3_row_detail:
                    extra = f" ({l2_l3_row_detail.replace('|', '/')})"
        elif i == CONCLUSION_DI_ROW_INDEX and di_row_ok is not None:
            cell = "Pass" if di_row_ok else "Fail"
            if not di_row_ok:
                verdict_ok = False
            if di_row_detail:
                extra = f" ({di_row_detail.replace('|', '/')})"
        elif i == CONCLUSION_CRITICAL_ROW_INDEX and critical_row_ok is not None:
            cell = "Pass" if critical_row_ok else "Fail"
            if not critical_row_ok:
                verdict_ok = False
                if critical_row_detail:
                    extra = f" ({critical_row_detail.replace('|', '/')})"
        elif i == CONCLUSION_ASSIGNEE_ROW_INDEX and assignee_row_ok is not None:
            cell = "Pass" if assignee_row_ok else "Fail"
            if not assignee_row_ok:
                verdict_ok = False
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
    **Remaining DI**, **critical issues**, **bug assignees**.
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
            auto_ok = assignee_row_ok
            row_detail = assignee_row_detail
        is_auto = auto_ok is not None
        pass_on = bool(auto_ok) if is_auto else True
        pass_cls = "is-on" if pass_on else ""
        fail_cls = "" if pass_on else "is-on"
        pass_pressed = "true" if pass_on else "false"
        fail_pressed = "false" if pass_on else "true"
        auto_cls = " conc-auto" if is_auto else ""
        hint = ""
        if is_auto and row_detail:
            hint = f'<div class="conc-auto-hint">{html.escape(row_detail)}</div>'
        rows.append(
            f'<tr data-conc-row="{i}" data-conc-auto="{"1" if is_auto else "0"}">'
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
    var rows = wrap.querySelectorAll('tbody tr');
    for (var i = 0; i < rows.length; i++) {{
      var tr = rows[i];
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
    """Replace placeholder with static Markdown (archived .md or ``--format markdown`` output)."""
    if RELEASE_CONCLUSION_PLACEHOLDER not in md:
        return md
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
    return md.replace(RELEASE_CONCLUSION_PLACEHOLDER, block, 1)


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
    chunks: list[tuple[str, str]] = []
    last = 0
    for m in re.finditer(r"`([^`]+)`", s):
        chunks.append(("t", s[last : m.start()]))
        chunks.append(("c", m.group(1)))
        last = m.end()
    chunks.append(("t", s[last:]))
    out: list[str] = []
    for kind, content in chunks:
        if kind == "c":
            out.append("<code>" + html.escape(content) + "</code>")
        else:
            out.append(_inline_text_with_links(content))
    return "".join(out)


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
    # ``### H100`` or ``H100（CI …`` — reject ``H1000``-style labels
    return bool(re.match(r"H100(?:[（(]|\Z)", inner_text, re.IGNORECASE))


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
        title_esc = html.escape(title) if title else "…"
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
{archive_scripts}
{_RELEASE_DETAILS_TOGGLE_SCRIPT}
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
