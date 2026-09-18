#!/usr/bin/env python3
"""Convert compose_full_report.py Markdown output to a standalone HTML document (stdlib only)."""

from __future__ import annotations

import html
import math
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
    "Performance regression < 10%(Guide)",
    # 2 — auto: latest finished ready + merge builds have no failed/broken jobs
    # (Upload * Pipeline upload-only steps are skipped).
    "Latest GPU CI(L1-L5) pass rate is 100%",
    # 3 — manual (user-selectable): NPU CI pass rate — placeholder until the
    # NPU pipeline gate is wired into compose_full_report.py.
    "Latest NPU CI(L1-L4) pass rate is 100%",
    # 4 — manual (user-selectable)
    "Requirement completion rate > 85%",
    # 5 — auto: cumulative Outstanding DI from all open `label:bug` (self-calculated
    #     the same way as the Development report). Threshold rule: DI ≤ 30 → Pass,
    #     DI > 30 → Fail.
    "Remaining DI < 30",
    # 6 — auto: compose checks for open issues labeled ``critical``
    "No remaining critical issues",
)

# Indices in ``RELEASE_CONCLUSION_ITEMS`` whose cells are **(Guide)** markers.
# They render in the table and the user can still pick Pass/Fail, but the
# final Go / Rejected verdict ignores them.
RELEASE_CONCLUSION_GUIDE_ROW_INDICES: frozenset[int] = frozenset({0, 1})

# "Latest GPU CI(L1-L5) pass rate is 100%": latest finished ready + merge builds have no failed/broken jobs
CONCLUSION_L2_L3_ROW_INDEX = 2
# "Latest NPU CI(L1-L4) pass rate is 100%": manual row (placeholder until NPU gate lands).
CONCLUSION_NPU_ROW_INDEX = 3
# "Remaining DI < 30": auto-computed cumulative Outstanding DI; threshold ≤ 30 ⇒ Pass.
CONCLUSION_DI_ROW_INDEX = 5
# "No remaining critical issues": compose checks for open issues labeled ``critical``
CONCLUSION_CRITICAL_ROW_INDEX = 6


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
_RELEASE_SVG_CLIPBOARD = (
    '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
    '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
    '<line x1="9" y1="11" x2="15" y2="11"/><line x1="9" y1="15" x2="15" y2="15"/><line x1="9" y1="19" x2="13" y2="19"/>'
)
_RELEASE_SVG_HOURGLASS = (
    '<line x1="6" y1="2" x2="18" y2="2"/><line x1="6" y1="22" x2="18" y2="22"/>'
    '<path d="M6 2h12v6a6 6 0 0 1-6 6 6 6 0 0 1-6-6V2z"/>'
    '<path d="M6 22h12v-6a6 6 0 0 0-6-6 6 6 0 0 0-6 6v6z"/>'
)
_RELEASE_SVG_SHIELD = (
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
    '<polyline points="9 12 11 14 15 10"/>'
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
    if "outstanding items" in low:
        return "outstanding", _RELEASE_SVG_CLIPBOARD
    if "quality defense" in low or "quality radar" in low or "quality line" in low:
        return "quality-defense", _RELEASE_SVG_SHIELD
    if "stability run results" in low:
        return "stability", _RELEASE_SVG_HOURGLASS
    if "data source" in low:
        return "data", _RELEASE_SVG_DATABASE
    return "default", _RELEASE_SVG_LAYOUT


def release_conclusion_widget_html(
    *,
    l2_l3_row_ok: bool | None = None,
    l2_l3_row_detail: str = "",
    di_row_ok: bool | None = None,
    di_row_detail: str = "",
    critical_row_ok: bool | None = None,
    critical_row_detail: str = "",
) -> str:
    """Interactive table + verdict (Go / Rejected) for ``.release-doc`` HTML.

    Automatic rows (non-clickable when ``*_row_ok`` is not ``None``):
    **Latest GPU CI(L1-L5)**, **Remaining DI**, **critical issues**.
    The **Latest NPU CI(L1-L4)** row is **manual** (always user-selectable,
    per the "用例自己选择结果" rule).
    Rows whose index is in :data:`RELEASE_CONCLUSION_GUIDE_ROW_INDICES` render
    in the table and stay user-selectable, but the final Go / Rejected verdict
    ignores them.
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
    )
    if p_wrap in fragment:
        return fragment.replace(p_wrap, widget, 1)
    if RELEASE_CONCLUSION_PLACEHOLDER in fragment:
        return fragment.replace(RELEASE_CONCLUSION_PLACEHOLDER, widget, 1)
    return fragment


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
    # Stash Markdown links whose **label contains a code span**, e.g.
    # ``[`tests/foo.py`](https://…)`` (emitted by the Skip Test Case Monitoring
    # file links). The backtick chunker below runs before link parsing, so
    # without this carve-out such a link is torn into "[", <code>…</code>,
    # "](url)" and renders as literal Markdown. Only backtick-bearing labels are
    # stashed so a link *inside* a code span (`` `[a](b)` ``) keeps its current
    # literal rendering.
    code_links: list[str] = []

    def _stash_code_link(m: re.Match[str]) -> str:
        idx = len(code_links)
        label_html = inline_md_to_html(m.group(1))
        url = html.escape(m.group(2), quote=True)
        code_links.append(f'<a href="{url}">{label_html}</a>')
        return f"\x00CODELINK_{idx}\x00"

    carved = re.sub(r"\[([^\]`]*`[^\]]*)\]\(([^)\s]+)\)", _stash_code_link, carved)

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
    if code_links:
        for idx, link_html in enumerate(code_links):
            rendered = rendered.replace(f"\x00CODELINK_{idx}\x00", link_html)
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
    """True if the block opens with an ``h3`` for H100 / H200 / H800 / A100 / A3 (not Common stack)."""
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
    if re.fullmatch(r"A3", inner_text, re.IGNORECASE):
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


def _group_skip_monitor_table_by_issue(html_fragment: str) -> str:
    """Fold the ``## Skip Test Case Monitoring`` table into per-issue collapsible groups.

    ``skip_issue_monitor.render_skip_monitor_table`` emits a flat table whose
    **first** column is ``Issue #`` and whose rows are already sorted so that
    every site referencing the same issue is contiguous. This post-processor
    rewrites that table's ``<tbody>`` into:

    * one **group row** per distinct issue (``<tr class="skip-issue-group">``)
      carrying the four issue-level cells (Issue # / Title / State / Updated),
      a toggle button, a site count, and a ``colspan`` summary cell; and
    * the original rows as **child rows** (``<tr class="skip-issue-child" hidden>``)
      whose issue-level cells are blanked (the group row already shows them).

    An *Expand all / Collapse all* toolbar is inserted above the table. All
    groups start collapsed; ``_SKIP_GROUP_SCRIPT`` handles the toggling.
    Markdown output is untouched (it keeps the flat, Issue-#-first table).
    """
    H2_RE = re.compile(r"<h2\b[^>]*>(.*?)</h2>", re.DOTALL | re.IGNORECASE)
    TAG_RE = re.compile(r"<[^>]+>")

    target = None
    for m in H2_RE.finditer(html_fragment):
        if "skip test case monitoring" in TAG_RE.sub("", m.group(1)).strip().lower():
            target = m
            break
    if target is None:
        return html_fragment

    next_h2 = H2_RE.search(html_fragment, target.end())
    section_end = next_h2.start() if next_h2 else len(html_fragment)

    table_m = re.search(
        r"<table\b[^>]*>.*?</table>",
        html_fragment[target.end() : section_end],
        re.DOTALL | re.IGNORECASE,
    )
    if table_m is None:
        return html_fragment
    t_start = target.end() + table_m.start()
    t_end = target.end() + table_m.end()
    table_html = table_m.group(0)

    headers = re.findall(r"<th\b[^>]*>(.*?)</th>", table_html, re.DOTALL | re.IGNORECASE)
    if not headers or "issue #" not in TAG_RE.sub("", headers[0]).strip().lower():
        return html_fragment
    n_cols = len(headers)
    issue_cols = min(4, n_cols)
    detail_cols = n_cols - issue_cols
    if detail_cols <= 0:
        return html_fragment

    body_m = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", table_html, re.DOTALL | re.IGNORECASE)
    if body_m is None:
        return html_fragment
    raw_rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", body_m.group(1), re.DOTALL | re.IGNORECASE)
    if not raw_rows:
        return html_fragment

    parsed: list[list[str]] = []
    for row in raw_rows:
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE)
        if len(cells) != n_cols:
            return html_fragment  # unexpected shape -> leave the table alone
        parsed.append(cells)

    # Group contiguous rows that share the same issue label (e.g. "#4636" or
    # "vllm-project/vllm#43060"). Non-contiguous repeats would each get their
    # own group, which is why the renderer sorts by issue first.
    groups: list[tuple[str, list[list[str]]]] = []
    for cells in parsed:
        key = TAG_RE.sub("", cells[0]).strip()
        if groups and groups[-1][0] == key:
            groups[-1][1].append(cells)
        else:
            groups.append((key, [cells]))

    out_rows: list[str] = []
    for gi, (key, rows) in enumerate(groups):
        gid = f"skip-issue-g{gi}"
        n = len(rows)
        count_label = "1 site" if n == 1 else f"{n} sites"
        head = rows[0]
        summary_txt = "1 skipped test — expand to view" if n == 1 else f"{n} skipped tests — expand to view"
        issue_cells = [
            (
                '<td class="skip-group-issue">'
                f'<button type="button" class="skip-group-toggle" aria-expanded="false" '
                f'aria-controls="{gid}" data-skip-group="{gid}" '
                'title="Show / hide the skipped tests for this issue">'
                '<span class="skip-group-caret" aria-hidden="true">▸</span>'
                "</button> "
                f"{head[0]} "
                f'<span class="skip-group-count">{count_label}</span>'
                "</td>"
            )
        ]
        issue_cells += [f"<td>{head[i]}</td>" for i in range(1, issue_cols)]
        out_rows.append(
            f'<tr class="skip-issue-group" data-skip-group="{gid}" data-expanded="false">'
            + "".join(issue_cells)
            + f'<td class="skip-group-summary" colspan="{detail_cols}">{summary_txt}</td>'
            + "</tr>"
        )
        for cells in rows:
            blanks = '<td class="skip-child-issue"></td>' + "<td></td>" * (issue_cols - 1)
            details = "".join(f"<td>{c}</td>" for c in cells[issue_cols:])
            out_rows.append(
                f'<tr class="skip-issue-child" data-skip-group="{gid}" hidden>' + blanks + details + "</tr>"
            )

    new_tbody = "<tbody>\n" + "\n".join(out_rows) + "\n</tbody>"
    new_table = table_html[: body_m.start()] + new_tbody + table_html[body_m.end() :]
    toolbar = (
        '<div class="skip-group-tools">'
        f'<span class="skip-group-tools-label">{len(groups)} issue(s) · {len(parsed)} skipped site(s)</span>'
        '<button type="button" class="skip-group-all" data-skip-all="expand">Expand all</button>'
        '<button type="button" class="skip-group-all" data-skip-all="collapse">Collapse all</button>'
        "</div>"
    )
    # Place the toolbar *outside* the horizontal scroll wrapper when present so
    # it stays visible while the table scrolls sideways.
    insert_at = t_start
    scroll_open = html_fragment.rfind('<div class="table-scroll">', target.end(), t_start)
    if scroll_open != -1:
        insert_at = scroll_open
    return (
        html_fragment[:insert_at]
        + toolbar
        + "\n"
        + html_fragment[insert_at:t_start]
        + new_table
        + html_fragment[t_end:]
    )


_SKIP_GROUP_SCRIPT = """<script>
(function () {
  function setGroup(gid, expanded, scope) {
    var root = scope || document;
    var rows = root.querySelectorAll('tr.skip-issue-child[data-skip-group="' + gid + '"]');
    for (var i = 0; i < rows.length; i++) {
      if (expanded) { rows[i].removeAttribute("hidden"); }
      else { rows[i].setAttribute("hidden", ""); }
    }
    var head = root.querySelector('tr.skip-issue-group[data-skip-group="' + gid + '"]');
    if (head) {
      head.setAttribute("data-expanded", expanded ? "true" : "false");
      var btn = head.querySelector(".skip-group-toggle");
      if (btn) { btn.setAttribute("aria-expanded", expanded ? "true" : "false"); }
      var caret = head.querySelector(".skip-group-caret");
      if (caret) { caret.textContent = expanded ? "\\u25BE" : "\\u25B8"; }
    }
  }

  document.addEventListener("click", function (ev) {
    var all = ev.target.closest ? ev.target.closest("[data-skip-all]") : null;
    if (all) {
      ev.preventDefault();
      var expand = all.getAttribute("data-skip-all") === "expand";
      var heads = document.querySelectorAll("tr.skip-issue-group");
      for (var i = 0; i < heads.length; i++) {
        setGroup(heads[i].getAttribute("data-skip-group"), expand, document);
      }
      return;
    }
    if (ev.target.closest && ev.target.closest("a")) return;
    var row = ev.target.closest ? ev.target.closest("tr.skip-issue-group") : null;
    if (!row) return;
    ev.preventDefault();
    var gid = row.getAttribute("data-skip-group");
    if (!gid) return;
    setGroup(gid, row.getAttribute("data-expanded") !== "true", document);
  });
})();
</script>"""


#: Allowed values of the ``Follow-up action`` dropdown in the Open issues table.
OPEN_ISSUE_FOLLOWUP_OPTIONS: tuple[str, ...] = (
    "Fix in a later iteration",
    "Blocked by dependency",
    "Won't fix (evaluated)",
)


def _upgrade_open_issue_action_cells(html_fragment: str) -> str:
    """Upgrade the ``Follow-up action`` / ``Remarks`` columns of the Open issues table.

    ``compose_full_report.OPEN_ISSUES_HEADERS`` appends two manual-entry columns
    to the ``## Open issues`` table (release *and* development variants); the
    Markdown renders them as ``—`` placeholders. Here each placeholder becomes:

    * **Follow-up action** — a native ``<select>`` offering ``Fix in a later
      iteration`` / ``Blocked by dependency`` / ``Won't fix (evaluated)`` (plus
      an empty "not set" option); and
    * **Remarks** — a click-to-edit note box (button → inline ``<textarea>``
      with Save / Cancel; Ctrl+Enter saves, Esc cancels).

    Both are persisted in ``localStorage`` by ``_OPEN_ISSUE_ACTION_SCRIPT``,
    keyed by the row's **issue number** (e.g. ``open-issue-followup:#4700``) so
    the operator's triage survives a report regeneration and is shared between
    the release and development variants of the same issue. Rows without a
    recognisable issue number fall back to a positional key.
    """
    TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
    TAG_RE = re.compile(r"<[^>]+>")

    def _upgrade_table(table_html: str) -> str:
        headers = [
            TAG_RE.sub("", h).strip()
            for h in re.findall(r"<th\b[^>]*>(.*?)</th>", table_html, re.DOTALL | re.IGNORECASE)
        ]
        if "Follow-up action" not in headers or "Remarks" not in headers:
            return table_html
        f_idx = headers.index("Follow-up action")
        n_idx = headers.index("Remarks")
        n_cols = len(headers)

        body_m = re.search(r"<tbody\b[^>]*>(.*?)</tbody>", table_html, re.DOTALL | re.IGNORECASE)
        if body_m is None:
            return table_html

        row_no = [0]

        def _row_sub(row_m: re.Match[str]) -> str:
            row_html = row_m.group(0)
            cells = re.findall(r"(<td\b[^>]*>)(.*?)(</td>)", row_html, re.DOTALL | re.IGNORECASE)
            idx = row_no[0]
            row_no[0] += 1
            if len(cells) != n_cols:
                return row_html
            issue_txt = TAG_RE.sub("", cells[0][1]).strip()
            m_issue = re.search(r"#(\d+)", issue_txt)
            key = f"#{m_issue.group(1)}" if m_issue else f"row-{idx}"
            key_attr = html.escape(key, quote=True)

            opts = ['<option value="">—</option>']
            for label in OPEN_ISSUE_FOLLOWUP_OPTIONS:
                esc = html.escape(label)
                opts.append(f'<option value="{esc}">{esc}</option>')
            followup_td = (
                f'<td class="oi-followup-cell" data-oi-key="{key_attr}" data-oi-state="empty">'
                f'<select class="oi-followup-select" data-oi-key="{key_attr}" '
                'aria-label="Follow-up action">' + "".join(opts) + "</select></td>"
            )
            note_td = (
                f'<td class="oi-note-cell" data-oi-key="{key_attr}" data-oi-state="empty" data-oi-value="">'
                '<button type="button" class="oi-note-btn oi-note-empty" '
                'data-oi-note-action="edit" title="Click to add a remark">'
                "Click to add a remark</button></td>"
            )

            parts = [row_html[: row_html.index(">") + 1]]
            for i, (open_tag, inner, close_tag) in enumerate(cells):
                if i == f_idx:
                    parts.append(followup_td)
                elif i == n_idx:
                    parts.append(note_td)
                else:
                    parts.append(f"{open_tag}{inner}{close_tag}")
            parts.append("</tr>")
            return "".join(parts)

        new_body = re.sub(r"<tr\b[^>]*>.*?</tr>", _row_sub, body_m.group(1), flags=re.DOTALL | re.IGNORECASE)
        return table_html[: body_m.start(1)] + new_body + table_html[body_m.end(1) :]

    return TABLE_RE.sub(lambda m: _upgrade_table(m.group(0)), html_fragment)


_OPEN_ISSUE_ACTION_SCRIPT = """<script>
(function () {
  function fKey(k) { return "open-issue-followup:" + k; }
  function nKey(k) { return "open-issue-note:" + k; }
  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) {
    try { if (v) { localStorage.setItem(k, v); } else { localStorage.removeItem(k); } }
    catch (e) { /* localStorage unavailable (e.g. file:// in Chrome) */ }
  }

  // In-memory store is the source of truth for rendering; localStorage is a
  // best-effort persistence layer on top. Without this, opening the report
  // from a file:// URL (where Chrome denies localStorage) would silently drop
  // every remark the moment the user pressed Save, because the cell used to
  // re-read the value straight back out of storage.
  var mem = {};
  function memGet(k) {
    if (!Object.prototype.hasOwnProperty.call(mem, k)) { mem[k] = lsGet(k) || ""; }
    return mem[k];
  }
  function memSet(k, v) { mem[k] = v || ""; lsSet(k, v); }

  // --- Follow-up action (select) ---
  // The selected value is stored in the <td>'s data-oi-value attribute so
  // that browser "Save As" serialises it into the HTML file.  On init we
  // prefer the data-attribute value (preserved by Save As) over localStorage
  // so that a saved-as copy retains the user's choice.
  function getFollowupValue(td) {
    var attr = td ? td.getAttribute("data-oi-value") : null;
    if (attr) return attr;
    var ls = memGet(fKey(td.getAttribute("data-oi-key") || ""));
    return ls || "";
  }
  function setFollowupValue(td, val) {
    td.setAttribute("data-oi-value", val || "");
    td.setAttribute("data-oi-state", val ? "set" : "empty");
    memSet(fKey(td.getAttribute("data-oi-key") || ""), val);
  }

  // --- Remarks (note) ---
  // The note text is stored in the <td>'s data-oi-value attribute so that
  // browser "Save As" serialises it into the HTML file.
  function getNoteValue(cell) {
    var attr = cell ? cell.getAttribute("data-oi-value") : null;
    if (attr) return attr;
    var ls = memGet(nKey(cell.getAttribute("data-oi-key") || ""));
    return ls || "";
  }
  function setNoteValue(cell, val) {
    cell.setAttribute("data-oi-value", val || "");
    cell.setAttribute("data-oi-state", val ? "set" : "empty");
    memSet(nKey(cell.getAttribute("data-oi-key") || ""), val);
  }

  function renderNote(cell) {
    var val = getNoteValue(cell);
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "oi-note-btn" + (val ? "" : " oi-note-empty");
    btn.setAttribute("data-oi-note-action", "edit");
    btn.title = val ? "Click to edit this remark" : "Click to add a remark";
    btn.textContent = val || "Click to add a remark";
    cell.innerHTML = "";
    cell.appendChild(btn);
    cell.setAttribute("data-oi-state", val ? "set" : "empty");
    if (val) { cell.setAttribute("data-oi-value", val); }
  }

  function editNote(cell) {
    var val = getNoteValue(cell);
    cell.innerHTML =
      '<div class="oi-note-editor">' +
      '<textarea class="oi-note-input" rows="3" ' +
      'placeholder="e.g. re-verify once the upstream PR lands"></textarea>' +
      '<div class="oi-note-actions">' +
      '<button type="button" class="oi-note-save">Save</button>' +
      '<button type="button" class="oi-note-cancel">Cancel</button>' +
      "</div></div>";
    cell.setAttribute("data-oi-state", "editing");
    var ta = cell.querySelector("textarea");
    if (ta) { ta.value = val; ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
  }

  function commitNote(cell) {
    var ta = cell.querySelector("textarea");
    var val = ta ? ta.value.trim() : "";
    setNoteValue(cell, val);
    renderNote(cell);
  }

  function initAll() {
    var selects = document.querySelectorAll("select.oi-followup-select");
    for (var i = 0; i < selects.length; i++) {
      var sel = selects[i];
      var td = sel.closest ? sel.closest("td") : null;
      var val = getFollowupValue(td);
      if (val) {
        var ok = false;
        for (var j = 0; j < sel.options.length; j++) {
          if (sel.options[j].value === val) { ok = true; break; }
        }
        if (ok) { sel.value = val; }
      }
      if (td) { setFollowupValue(td, sel.value); }
    }
    var notes = document.querySelectorAll("td.oi-note-cell");
    for (var k = 0; k < notes.length; k++) { renderNote(notes[k]); }
    var summaries = document.querySelectorAll(".fa-summary-editable");
    for (var s = 0; s < summaries.length; s++) { renderNote(summaries[s]); }
  }

  document.addEventListener("change", function (ev) {
    var sel = ev.target;
    if (!sel || !sel.classList || !sel.classList.contains("oi-followup-select")) return;
    var td = sel.closest ? sel.closest("td") : null;
    if (td) { setFollowupValue(td, sel.value); }
  });

  document.addEventListener("click", function (ev) {
    if (!ev.target || !ev.target.closest) return;
    var cell = ev.target.closest("td.oi-note-cell, .fa-summary-editable");
    if (!cell) return;
    if (ev.target.closest(".oi-note-save")) { ev.preventDefault(); commitNote(cell); return; }
    if (ev.target.closest(".oi-note-cancel")) { ev.preventDefault(); renderNote(cell); return; }
    if (ev.target.closest("[data-oi-note-action='edit']")) { ev.preventDefault(); editNote(cell); }
  });

  document.addEventListener("keydown", function (ev) {
    if (!ev.target || !ev.target.classList) return;
    if (!ev.target.classList.contains("oi-note-input")) return;
    var cell = ev.target.closest ? ev.target.closest("td.oi-note-cell, .fa-summary-editable") : null;
    if (!cell) return;
    if (ev.key === "Escape") { ev.preventDefault(); renderNote(cell); }
    else if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); commitNote(cell); }
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
</script>"""


def _upgrade_di_top10_input_cells(html_fragment: str) -> str:
    """Make the **Assignee** and **Maintainer** columns editable in the **DI Top10** sub-table.

    The development report's ``### DI Top10 (SLO-escalating: ...)`` table is a
    plain Markdown table. Operators want to fill in assignees and maintainers
    directly in the report; this post-processor walks that table only and
    rewrites each **Assignee** and **Maintainer** cell.

    Filled cells (GitHub-sourced value present) become a plain
    ``<span class="di-top10-{assignee,maintainer}-text">`` — editing would
    invite drift between the report and GitHub, so we just render the value
    and skip any input affordance. Empty cells become an editable inline
    ``<input>`` (placeholder ``+ Add assignee`` / ``+ Add maintainer``) so
    the operator can fill them in directly.

    The editable input path shares the runtime JS in :data:`_DI_TOP10_INPUT_SCRIPT`:
    every keystroke and ``blur`` writes to (a) the input's
    ``data-di-*-value`` attribute (so a browser "Save Page As" captures it)
    and (b) ``localStorage`` keyed by ``di-top10-assignee:#N`` /
    ``di-top10-maintainer:#N``. The DOM attribute is preferred on reload so
    a saved copy retains the user's edits across origins
    (``file://`` → ``github.io``).

    Implementation pattern mirrors the Open-issues Follow-up action columns
    (see :func:`_upgrade_open_issue_action_cells`).
    """
    if "DI Top10" not in html_fragment:
        return html_fragment

    # Find the DI Top10 sub-table. Strategy: locate the <h3>DI Top10 heading,
    # then walk forward to the FIRST <table> that follows it.
    h3_match = re.search(r"<h3>[^<]*DI\s*Top10[\s\S]*?</h3>", html_fragment)
    if not h3_match:
        return html_fragment
    after_h3 = html_fragment[h3_match.end() :]
    table_match = re.search(r"<table\b[^>]*>(.*?)</table>", after_h3, re.DOTALL | re.IGNORECASE)
    if not table_match:
        return html_fragment
    table_start = h3_match.end() + table_match.start()
    table_end = h3_match.end() + table_match.end()
    table_html = html_fragment[table_start:table_end]

    # Find the Assignee + Maintainer column indices from the header row.
    th_matches = re.findall(r"<th[^>]*>(.*?)</th>", table_html, re.DOTALL | re.IGNORECASE)
    if not th_matches:
        return html_fragment
    column_specs: list[tuple[int, str]] = []  # (col_idx, header_lower)
    for idx, th in enumerate(th_matches):
        plain = re.sub(r"<[^>]+>", "", th).strip().lower()
        if plain == "assignee":
            column_specs.append((idx, "assignee"))
        elif plain == "maintainer":
            column_specs.append((idx, "maintainer"))
    if not column_specs:
        return html_fragment

    def _upgrade_row(tr_html: str) -> str:
        tds = re.findall(r"<td[^>]*>([\s\S]*?)</td>", tr_html)
        if not tds:
            return tr_html
        # Pull the issue number from the first <td>'s "#1234" content.
        first_td = tds[0]
        issue_num_match = re.search(r"#(\d+)", first_td)
        if not issue_num_match:
            return tr_html
        issue_num = issue_num_match.group(1)
        # Build a map of (column_index → replacement HTML) for both columns.
        new_cells: dict[int, str] = {}
        for col_idx, header in column_specs:
            if col_idx >= len(tds):
                continue
            original_value = re.sub(r"<[^>]+>", "", tds[col_idx]).strip()
            # Treat em-dash / dash placeholders as empty (matches the original
            # behaviour so filled rows still pre-fill the input with the real
            # GitHub-sourced value, not the literal "—").
            is_empty = original_value in ("", "—", "—", "-", "N/A")
            effective = "" if is_empty else original_value
            initial = html.escape(effective, quote=True)
            placeholder_text = f"+ Add {header}"
            cell_class = f"di-top10-{header}-cell {'is-filled' if effective else 'is-empty'}"
            cell_attr = f'data-di-{header}-issue="{issue_num}"'
            if effective:
                # Filled cell: plain text, no input. Editing would invite
                # drift between the report and GitHub, so we just render the
                # value. The data-di-*-issue attribute is still emitted so
                # any inspector / future script can still identify the row.
                new_cells[col_idx] = (
                    f'<td class="{cell_class}" {cell_attr}>'
                    f'<span class="di-top10-{header}-text" '
                    f'data-di-{header}-issue="{issue_num}">'
                    f"{initial}</span></td>"
                )
            else:
                # Empty cell: editable input with localStorage persistence.
                new_cells[col_idx] = (
                    f'<td class="{cell_class}" {cell_attr}>'
                    f'<input type="text" class="di-top10-{header}-input" '
                    f'data-di-{header}-issue="{issue_num}" '
                    f'data-di-{header}-persist="1" '
                    f'data-di-{header}-value="" '
                    f'placeholder="{placeholder_text}" '
                    f'value="" />'
                    f"</td>"
                )
        # Replace each targeted cell in document order. We can't use a single regex
        # sub with a counter when both columns share the same row, so apply
        # replacements one-by-one from the highest column index down so the
        # lower indices keep their offsets.
        if not new_cells:
            return tr_html
        for col_idx in sorted(new_cells, reverse=True):
            replacement = new_cells[col_idx]
            pattern = re.compile(r"<td[^>]*>([\s\S]*?)</td>", re.DOTALL | re.IGNORECASE)
            counter = {"i": 0}
            replaced = {"done": False}

            def _replace_td(
                m: re.Match[str],
                _ci: int = col_idx,
                _rep: str = replacement,
                _st: dict[str, bool] = replaced,
            ) -> str:
                if _st["done"]:
                    return m.group(0)
                if counter["i"] == _ci:
                    counter["i"] += 1
                    _st["done"] = True
                    return _rep
                counter["i"] += 1
                return m.group(0)

            tr_html = pattern.sub(_replace_td, tr_html, count=len(tds))
        return tr_html

    new_table = re.sub(
        r"<tr[^>]*>([\s\S]*?)</tr>",
        lambda m: _upgrade_row(m.group(0)),
        table_html,
    )
    return html_fragment[:table_start] + new_table + html_fragment[table_end:]


def _upgrade_next_steps_outstanding_cells(html_fragment: str) -> str:
    """Replace the Outstanding Items markdown table with a clean thead-only
    table and an "Add Item" button.

    The markdown emits a 3-column table (Item / Assignee / Status) with one
    placeholder data row.  In HTML we replace the entire ``<table>`` with:

    * ``<thead>``: Item / Assignee / Status headers + empty 4th column for
      the delete button
    * Empty ``<tbody>`` — all data rows are created dynamically by clicking
      "Add Item" and persisted in localStorage

    The JS in ``_NEXT_STEPS_OUTSTANDING_SCRIPT`` handles row creation, inline
    editing, and localStorage persistence.
    """
    TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)
    TAG_RE = re.compile(r"<[^>]+>")

    def _upgrade_table(table_html: str) -> str:
        headers = [
            TAG_RE.sub("", h).strip()
            for h in re.findall(r"<th\b[^>]*>(.*?)</th>", table_html, re.DOTALL | re.IGNORECASE)
        ]
        has_item = any("Item" in h or "\u4e8b\u9879" in h for h in headers)
        has_assignee = any("Assignee" in h or "\u8d23\u4efb\u4eba" in h for h in headers)
        has_status = any("Status" in h or "\u72b6\u6001" in h for h in headers)
        if not (has_item and has_assignee and has_status):
            return table_html

        add_btn = (
            '<div class="ns-add-item-wrap">'
            '<button type="button" class="ns-add-item-btn" data-ns-add-item="1">'
            "\uff0b Add Item</button></div>"
        )
        new_table = (
            '<table class="ns-outstanding-table" data-ns-table="outstanding-items">\n'
            "<thead><tr>"
            "<th>Item</th>"
            "<th>Assignee</th>"
            "<th>Status</th>"
            '<th class="ns-del-th"></th>'
            "</tr></thead>\n"
            "<tbody></tbody>\n"
            "</table>"
        )
        return add_btn + new_table

    return TABLE_RE.sub(lambda m: _upgrade_table(m.group(0)), html_fragment)


# ── Device-Hours / Build (7-day avg) — manual metric row ────────────────
# Operator-editable metric appended to the release Metrics overview table
# by ``compose_full_report._append_device_hours_build_row``. The row ships
# with a marker placeholder ``@@DEVICE_HOURS_PER_BUILD_CELL@@`` in the
# **Success rate/UT coverage** cell; this upgrade substitutes an inline
# editable ``<input>`` whose value persists in ``localStorage['device-hours-per-build']``.
DEVICE_HOURS_PER_BUILD_CELL_HTML = (
    '<input type="text" class="dhpb-input" data-dhpb-persist="1" '
    'data-dhpb-marker="1" placeholder="click to fill (e.g. 132.4 h)" '
    'value="" />'
)


def _upgrade_device_hours_cell(html_fragment: str) -> str:
    """Swap the ``@@DEVICE_HOURS_PER_BUILD_CELL@@`` marker for an editable input.

    The marker is emitted by ``compose_full_report._append_device_hours_build_row``
    inside the **Success rate/UT coverage** cell of the final row of the
    release Metrics overview table. The Markdown ``|`` cell wrapper is
    plain text so we do a literal ``str.replace`` rather than regex; the
    marker string is unique enough that a global swap is safe. JS handler
    lives in :data:`_DEVICE_HOURS_BUILD_SCRIPT`.
    """
    marker = "@@DEVICE_HOURS_PER_BUILD_CELL@@"
    if marker not in html_fragment:
        return html_fragment
    return html_fragment.replace(marker, DEVICE_HOURS_PER_BUILD_CELL_HTML)


_DEVICE_HOURS_BUILD_SCRIPT = """<script>
(function () {
  "use strict";

  // In-memory fallback when localStorage throws (e.g. Chrome file://).
  var mem = {};
  function lsGet(k) {
    try { return localStorage.getItem(k); } catch (e) { return mem.hasOwnProperty(k) ? mem[k] : null; }
  }
  function lsSet(k, v) {
    try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); }
    catch (e) { if (v) mem[k] = v; else delete mem[k]; }
  }

  var STORAGE_KEY = "device-hours-per-build";

  function hydrate(input) {
    // Prefer the DOM attribute first (captured by "Save Page As"),
    // fall back to localStorage (reload on the same origin).
    var attr = input.getAttribute("data-dhpb-value");
    var saved = null;
    if (attr === null || attr === undefined) {
      saved = lsGet(STORAGE_KEY);
    } else if (attr) {
      saved = attr;
    }
    if (saved !== null && saved !== undefined && saved !== "") {
      input.value = saved;
    }
    if (input.value) {
      input.setAttribute("data-dhpb-persisted", "1");
    }
    function persist() {
      var v = input.value || "";
      input.setAttribute("data-dhpb-value", v);
      lsSet(STORAGE_KEY, v);
      if (v) {
        input.setAttribute("data-dhpb-persisted", "1");
      } else {
        input.removeAttribute("data-dhpb-persisted");
      }
    }
    input.addEventListener("input", persist);
    input.addEventListener("blur", persist);
  }

  function initAll() {
    var inputs = document.querySelectorAll("input.dhpb-input[data-dhpb-marker=\"1\"]");
    for (var i = 0; i < inputs.length; i++) { hydrate(inputs[i]); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
</script>"""


_DI_TOP10_INPUT_SCRIPT = """<script>
(function () {
  "use strict";

  // In-memory fallback when localStorage throws (e.g. Chrome file://).
  var mem = {};
  function lsGet(k) {
    try { return localStorage.getItem(k); }
    catch (e) { return Object.prototype.hasOwnProperty.call(mem, k) ? mem[k] : null; }
  }
  function lsSet(k, v) {
    try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); }
    catch (e) { if (v) mem[k] = v; else delete mem[k]; }
  }

  function keyFor(kind, issue) { return "di-top10-" + kind + ":#" + issue; }

  function hydrate(input, kind) {
    // Prefer the DOM attribute (captured by "Save Page As"), fall back to
    // localStorage (reload on the same origin). Both editors share this
    // exact pattern, so the only difference is the localStorage key prefix.
    var issue = input.getAttribute("data-di-" + kind + "-issue");
    if (!issue) return;
    var key = keyFor(kind, issue);
    var attr = input.getAttribute("data-di-" + kind + "-value");
    if (attr === null || attr === undefined) {
      var saved = lsGet(key);
      if (saved !== null && saved !== undefined && saved !== "") {
        attr = saved;
        input.value = saved;
      }
    } else if (attr) {
      input.value = attr;
    }
    if (input.value) {
      input.setAttribute("data-di-" + kind + "-persisted", "1");
      var td0 = input.closest ? input.closest("td.di-top10-" + kind + "-cell") : null;
      if (td0) td0.classList.remove("is-empty");
    }
    function persist() {
      input.setAttribute("data-di-" + kind + "-value", input.value || "");
      lsSet(key, input.value || "");
    }
    input.addEventListener("input", persist);
    input.addEventListener("blur", persist);
  }

  function initAll() {
    var aInputs = document.querySelectorAll("input.di-top10-assignee-input");
    for (var i = 0; i < aInputs.length; i++) { hydrate(aInputs[i], "assignee"); }
    var mInputs = document.querySelectorAll("input.di-top10-maintainer-input");
    for (var j = 0; j < mInputs.length; j++) { hydrate(mInputs[j], "maintainer"); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
</script>"""


# Backward-compatible alias — older call sites still reference this name.
_DI_TOP10_ASSIGNEE_SCRIPT = _DI_TOP10_INPUT_SCRIPT


_NEXT_STEPS_OUTSTANDING_SCRIPT = """<script>
(function () {
  "use strict";

  // --- Storage helpers (with in-memory fallback when localStorage is unavailable) ---
  var mem = {};
  function lsGet(k) {
    try { return localStorage.getItem(k); } catch (e) { return mem.hasOwnProperty(k) ? mem[k] : null; }
  }
  function lsSet(k, v) {
    try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); }
    catch (e) { if (v) mem[k] = v; else delete mem[k]; }
  }

  // --- Constants ---
  var TABLE_ID = "outstanding-items-table";
  var STATUSES = ["Open", "In Progress", "Blocked", "Won't fix", "Fixed"];
  var PLACEHOLDER = "\u2014";

  // Row keys are generated as ``ns-row-<base36 timestamp>-<random>``. This
  // is globally unique per row creation so two reports / regenerations /
  // browser sessions can't collide \u2014 and the key travels with the row
  // because it is written to ``data-ns-row-key`` on every save and persisted
  // inside ``data-ns-row-value`` (the JSON payload) for "Save Page As".
  function newRowKey() {
    return "ns-row-" + Date.now().toString(36) + "-" + Math.floor(Math.random() * 1e9).toString(36);
  }

  // --- Build a fresh editable row. All three columns are inputs from the start
  //     (no "Click to set" placeholder cell) so users can type immediately. ---
  function buildRow(key) {
    var tr = document.createElement("tr");
    tr.setAttribute("data-ns-row-key", key);

    // Item column \u2014 text input
    var tdItem = document.createElement("td");
    tdItem.className = "ns-item-cell";
    var inpItem = document.createElement("input");
    inpItem.type = "text";
    inpItem.className = "ns-item-input ns-row-input";
    inpItem.setAttribute("data-ns-field", "item");
    inpItem.setAttribute("data-ns-key", key);
    inpItem.placeholder = "Item description";
    tdItem.appendChild(inpItem);

    // Assignee column \u2014 text input
    var tdAssignee = document.createElement("td");
    tdAssignee.className = "ns-assignee-cell";
    var inpAssignee = document.createElement("input");
    inpAssignee.type = "text";
    inpAssignee.className = "ns-assignee-input ns-row-input";
    inpAssignee.setAttribute("data-ns-field", "assignee");
    inpAssignee.setAttribute("data-ns-key", key);
    inpAssignee.placeholder = "Assignee";
    tdAssignee.appendChild(inpAssignee);

    // Status column \u2014 select
    var tdStatus = document.createElement("td");
    tdStatus.className = "ns-status-cell";
    var sel = document.createElement("select");
    sel.className = "ns-status-select";
    sel.setAttribute("data-ns-field", "status");
    sel.setAttribute("data-ns-key", key);
    sel.setAttribute("aria-label", "Status");
    var optEmpty = document.createElement("option");
    optEmpty.value = "";
    optEmpty.textContent = PLACEHOLDER;
    sel.appendChild(optEmpty);
    for (var i = 0; i < STATUSES.length; i++) {
      var opt = document.createElement("option");
      opt.value = STATUSES[i];
      opt.textContent = STATUSES[i];
      sel.appendChild(opt);
    }
    tdStatus.appendChild(sel);

    // Delete column \u2014 button
    var tdDel = document.createElement("td");
    tdDel.className = "ns-del-cell";
    var btnDel = document.createElement("button");
    btnDel.type = "button";
    btnDel.className = "ns-del-btn";
    btnDel.setAttribute("data-ns-del-row", key);
    btnDel.title = "Delete this row";
    btnDel.textContent = "\u2715";
    tdDel.appendChild(btnDel);

    tr.appendChild(tdItem);
    tr.appendChild(tdAssignee);
    tr.appendChild(tdStatus);
    tr.appendChild(tdDel);
    return tr;
  }

  function readRowValue(tr) {
    var inpItem = tr.querySelector("input[data-ns-field='item']");
    var inpAssignee = tr.querySelector("input[data-ns-field='assignee']");
    var selStatus = tr.querySelector("select[data-ns-field='status']");
    return {
      item: inpItem ? inpItem.value : "",
      assignee: inpAssignee ? inpAssignee.value : "",
      status: selStatus ? selStatus.value : "",
    };
  }

  function writeRowValue(tr, entry) {
    var inpItem = tr.querySelector("input[data-ns-field='item']");
    var inpAssignee = tr.querySelector("input[data-ns-field='assignee']");
    var selStatus = tr.querySelector("select[data-ns-field='status']");
    if (inpItem) inpItem.value = entry.item || "";
    if (inpAssignee) inpAssignee.value = entry.assignee || "";
    if (selStatus) selStatus.value = entry.status || "";
  }

  // --- Persistence: serialize/deserialize the full table state. ---
  function saveTable(table) {
    var rows = table.querySelectorAll("tbody tr");
    var data = [];
    for (var i = 0; i < rows.length; i++) {
      var tr = rows[i];
      var key = tr.getAttribute("data-ns-row-key") || newRowKey();
      tr.setAttribute("data-ns-row-key", key);  // ensure stable identity
      var payload = readRowValue(tr);
      // Per-row DOM writeback: a single JSON attribute on the <tr> that
      // browser "Save Page As" captures verbatim. On reload from a saved
      // file the script reads from this attribute first (see loadTable),
      // then falls back to localStorage for same-origin reloads without
      // a saved copy.
      tr.setAttribute("data-ns-row-value", JSON.stringify({
        key: key, item: payload.item, assignee: payload.assignee, status: payload.status,
      }));
      data.push({key: key, item: payload.item, assignee: payload.assignee, status: payload.status});
    }
    lsSet("outstanding-items:" + TABLE_ID, JSON.stringify(data));
  }

  function loadTable(table) {
    // First, harvest rows already present in the DOM (captured by
    // "Save Page As"). The Python post-processor currently emits an empty
    // <tbody>, but a saved copy may contain a hydrated <tr data-ns-row-value>
    // for each row \u2014 preserve those as-is so a saved copy displays correctly
    // even with localStorage cleared / on a different origin.
    var tbody = table.querySelector("tbody");
    if (!tbody) return false;
    var savedRows = [];
    var existingTrs = tbody.querySelectorAll("tr");
    for (var i = 0; i < existingTrs.length; i++) {
      var tr = existingTrs[i];
      var raw = tr.getAttribute("data-ns-row-value");
      if (!raw) continue;
      try {
        var payload = JSON.parse(raw);
        savedRows.push({
          key: payload.key || tr.getAttribute("data-ns-row-key") || newRowKey(),
          item: payload.item || "",
          assignee: payload.assignee || "",
          status: payload.status || "",
        });
      } catch (e) { /* skip */ }
    }
    tbody.innerHTML = "";

    // Then merge with localStorage so an unsaved edit on the same origin
    // is honoured (localStorage wins on conflict because it represents the
    // most recent in-memory change).
    var rawLs = lsGet("outstanding-items:" + TABLE_ID);
    if (rawLs) {
      try {
        var lsData = JSON.parse(rawLs);
        if (Array.isArray(lsData)) {
          var byKey = {};
          for (var j = 0; j < savedRows.length; j++) { byKey[savedRows[j].key] = savedRows[j]; }
          for (var k = 0; k < lsData.length; k++) {
            var entry = lsData[k];
            if (entry && entry.key) byKey[entry.key] = entry;  // LS overrides DOM
          }
          savedRows = Object.keys(byKey).map(function (k) { return byKey[k]; });
        }
      } catch (e) { /* ignore */ }
    }

    for (var m = 0; m < savedRows.length; m++) {
      var rowKey = savedRows[m].key;
      var tr2 = buildRow(rowKey);
      tbody.appendChild(tr2);
      writeRowValue(tr2, savedRows[m]);
      // Persist the freshly written row back into the DOM so the next
      // "Save Page As" picks up exactly this state.
      tr2.setAttribute("data-ns-row-value", JSON.stringify({
        key: rowKey,
        item: savedRows[m].item || "",
        assignee: savedRows[m].assignee || "",
        status: savedRows[m].status || "",
      }));
    }
    return savedRows.length > 0;
  }

  // --- Row management ---
  function addRowToTable(table) {
    var tbody = table.querySelector("tbody");
    if (!tbody) {
      tbody = document.createElement("tbody");
      table.appendChild(tbody);
    }
    var tr = buildRow(newRowKey());
    tbody.appendChild(tr);
    saveTable(table);
    var firstInput = tr.querySelector("input[data-ns-field='item']");
    if (firstInput) {
      try { firstInput.focus(); } catch (e) { /* no-op */ }
    }
  }

  function deleteRow(btn) {
    var tr = btn.closest ? btn.closest("tr") : null;
    if (!tr) return;
    var table = tr.closest ? tr.closest("table") : null;
    if (tr.parentNode) tr.parentNode.removeChild(tr);
    if (table) saveTable(table);
  }

  // --- Initialise each outstanding-items table in the document ---
  function initTable(table) {
    loadTable(table);

    table.addEventListener("input", function (ev) {
      if (ev.target && ev.target.classList && ev.target.classList.contains("ns-row-input")) {
        saveTable(table);
      }
    });
    table.addEventListener("change", function (ev) {
      var t = ev.target;
      if (t && t.classList && t.classList.contains("ns-status-select")) {
        saveTable(table);
      }
    });
    table.addEventListener("click", function (ev) {
      if (!ev.target || !ev.target.closest) return;
      var delBtn = ev.target.closest("[data-ns-del-row]");
      if (delBtn) {
        ev.preventDefault();
        deleteRow(delBtn);
      }
    });
  }

  function initAll() {
    var tables = document.querySelectorAll("table.ns-outstanding-table");
    for (var i = 0; i < tables.length; i++) { initTable(tables[i]); }

    var addBtns = document.querySelectorAll("[data-ns-add-item]");
    for (var j = 0; j < addBtns.length; j++) {
      (function (btn) {
        btn.addEventListener("click", function (ev) {
          ev.preventDefault();
          ev.stopPropagation();
          var wrap = btn.closest ? btn.closest(".ns-add-item-wrap") : null;
          var table = null;
          if (wrap) {
            var next = wrap.nextElementSibling;
            if (next && next.tagName === "TABLE") {
              table = next;
            } else {
              var parent = wrap.parentElement;
              if (parent) table = parent.querySelector("table");
            }
          }
          if (!table) {
            table = document.querySelector("table.ns-outstanding-table");
          }
          if (table) addRowToTable(table);
        });
      })(addBtns[j]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
</script>"""


_RESOURCE_USAGE_BLOCK_HTML = (
    '<div class="resource-usage-block" data-uri-state="empty" data-uri-value="">'
    '<div class="resource-usage-toolbar resource-usage-toolbar--top">'
    '<button type="button" class="resource-usage-add">+ Add module</button>'
    '<span class="resource-usage-status" data-resource-usage-state="pristine">No modules yet — click "Add module" to start</span>'
    '</div>'
    '<div class="resource-usage-modules"></div>'
    '</div>'
)


def _upgrade_resource_usage_block(html_fragment: str) -> str:
    """Replace the ``@@RESOURCE_USAGE_INSERTION_POINT@@`` marker with an editor.

    The development-variant **Resource Usage Analysis** section
    is a manual-entry engineering artefact: the Markdown body is empty by
    design and the editor is injected here so that:

    * The H2 heading stays a single, stable Markdown heading (so it can be
      matched by ``_wrap_release_report_h2_sections`` and wrapped in the
      themed collapsible ``<details>`` card).
    * The H2 → ``<details>`` wrapper added by
      ``_fold_release_report_section_cards`` turns the heading itself into a
      click-to-expand handle — the user clicks the title and the editor
      appears below.
    * Persistence is split between ``data-uri-value`` (so a browser *Save
      Page As* download captures the analysis into the saved HTML) and
      ``localStorage['resource-usage-analysis']`` (so reloads on http(s)
      origins keep the value).

    Note: the report's custom Markdown converter wraps every paragraph in a
    ``<p>``, so the substituted block ends up inside ``<p><div …></div></p>``.
    Browsers' HTML5 parser implicitly closes the ``<p>`` before the ``<div>``
    so the editor block is a direct child of the section body in the DOM; the
    empty surrounding ``<p>`` tags are harmless. The JavaScript handler
    queries ``.resource-usage-block`` directly so the wrapping does not
    affect behaviour.

    The handler lives in :data:`_RESOURCE_USAGE_SCRIPT`.
    """
    marker = "@@RESOURCE_USAGE_INSERTION_POINT@@"
    if marker not in html_fragment:
        return html_fragment
    return html_fragment.replace(marker, _RESOURCE_USAGE_BLOCK_HTML)


# ── Quality Defense Radar ──────────────────────────────────────────────
# Per-model 5-axis coverage radar. Nine flagship models are rendered as a
# 3×3 grid of small pentagon SVGs; each radar carries 8 clickable segments
# (5 axes, of which Functionality / Performance / Stability are split into
# GPU + NPU halves that share one circle, while Documentation / Reliability
# are single circles). Click any segment to toggle gray ↔ green; state is
# mirrored to a ``data-quality-on`` attribute on the ``<g class="qd-segment">``
# (so a Save-Page-As download preserves the toggled state) **and** to
# ``localStorage["quality-defense:<model>:<axis>"]`` for reload persistence.
#
# The geometry below is hand-precomputed from a regular pentagon:
#   * ViewBox 300×300, centre at (150, 150).
#   * Outer pentagon radius 105, inner pentagon radius 50.
#   * Split-axis midpoints sit at radius 75 (between inner and outer).
#   * Half-circle radius 18 for split segments; full-circle radius 18 for
#     single segments.
#   * Pentagon vertex angles (degrees, SVG Y-down, clockwise from top):
#       Functionality  -90°   (top)
#       Performance   -18°   (top-right)
#       Documentation  54°   (bottom-right)
#       Stability     126°   (bottom-left)
#       Reliability   198°  (top-left)
_QUALITY_DEFENSE_MODELS = (
    ("qwen-omni",   "Qwen3-Omni"),
    ("minicpm",     "MiniCPM"),
    ("qwen-tts",    "Qwen-TTS"),
    ("qwen-image",  "Qwen-Image"),
    ("HunyuanImage",  "HunyuanImage"),
    ("HunyuanVideo",  "HunyuanVideo"),
    ("Wan",         "Wan"),
    ("MinimaxH3",   "MinimaxH3"),
    ("Cosmos",      "Cosmos"),
)

_QUALITY_DEFENSE_AXIS_KEYS = ("func", "perf", "doc", "stab", "rel")
_QUALITY_DEFENSE_AXIS_LABELS = {
    "func": "Functionality",
    "perf": "Performance",
    "doc":  "Documentation",
    "stab": "Stability",
    "rel":  "Reliability",
}
_QUALITY_DEFENSE_AXIS_ANGLES = {
    "func": -90,
    "perf": -18,
    "doc":  54,
    "stab": 126,
    "rel":  198,
}
# Visual offset (in SVG units) from the outer pentagon vertex where the
# axis label is drawn. Keys map the axis name to (dx, dy).
_QUALITY_DEFENSE_LABEL_OFFSETS = {
    "func": (0, -10),
    "perf": (10, -4),
    "doc":  (0, 16),
    "stab": (0, 16),
    "rel":  (-10, -4),
}


def _qd_pentagon_point(angle_deg: float, radius: float) -> tuple[float, float]:
    """Regular pentagon vertex at the given angle and radius from (150,150)."""
    rad = math.radians(angle_deg)
    return (150 + radius * math.cos(rad), 150 + radius * math.sin(rad))


def _qd_model_radar_svg(model_id: str) -> str:
    """Inline SVG for a single model's 5-axis coverage radar."""
    # Pentagon rings (outer and inner) — drawn once as decoration.
    outer_pts = [
        _qd_pentagon_point(_QUALITY_DEFENSE_AXIS_ANGLES[k], 105)
        for k in _QUALITY_DEFENSE_AXIS_KEYS
    ]
    inner_pts = [
        _qd_pentagon_point(_QUALITY_DEFENSE_AXIS_ANGLES[k], 50)
        for k in _QUALITY_DEFENSE_AXIS_KEYS
    ]
    outer_str = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in outer_pts)
    inner_str = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in inner_pts)

    axis_lines = "\n        ".join(
        f'<line x1="150" y1="150" x2="{p[0]:.1f}" y2="{p[1]:.1f}"/>'
        for p in outer_pts
    )

    axis_label_xml = "\n      ".join(
        f'<text class="qd-axis-label" x="{outer_pts[i][0] + _QUALITY_DEFENSE_LABEL_OFFSETS[k][0]:.1f}" '
        f'y="{outer_pts[i][1] + _QUALITY_DEFENSE_LABEL_OFFSETS[k][1]:.1f}" '
        f'text-anchor="middle">{_QUALITY_DEFENSE_AXIS_LABELS[k]}</text>'
        for i, k in enumerate(_QUALITY_DEFENSE_AXIS_KEYS)
    )

    # Split axes (Functionality / Performance / Stability) — two halves of
    # one circle each. Each half is rendered inside a `<g transform="…">` so
    # the canonical vertical-diameter half-circles get rotated to align with
    # the axis direction. The canonical LEFT half (sweep=0, bulges to −X)
    # labels "GPU"; the canonical RIGHT half (sweep=1, bulges to +X) labels
    # "NPU". After rotation the GPU half always bulges opposite to the axis
    # direction (closer to centre); NPU bulges along the axis (further out).
    #
    # GPU vs NPU are visually distinguished two ways (so reviewers can tell
    # them apart at a glance):
    #   1. ``data-qd-side="gpu"|"npu"`` attribute drives CSS colour rules
    #      (GPU on = light green; NPU on = light blue — see the radar CSS).
    #   2. The NPU halves also carry ``stroke-dasharray`` in the default
    #      state so even before clicking, the two halves read as "solid
    #      outline" vs "dashed outline".
    split_xml_parts: list[str] = []
    for axis_key in ("func", "perf", "stab"):
        angle = _QUALITY_DEFENSE_AXIS_ANGLES[axis_key]
        mx, my = _qd_pentagon_point(angle, 75)
        axis_label = _QUALITY_DEFENSE_AXIS_LABELS[axis_key]
        split_xml_parts.append(
            f'<g class="qd-segment" data-quality-key="{model_id}:{axis_key}-gpu" '
            f'data-qd-side="gpu" tabindex="0" role="button" '
            f'aria-label="{model_id} {axis_label} GPU" aria-pressed="false">'
            f'<g transform="translate({mx:.1f} {my:.1f}) rotate({angle})">'
            f'<path class="qd-half qd-half--gpu" d="M 0 18 A 18 18 0 0 0 0 -18 Z"/>'
            f"</g></g>"
        )
        split_xml_parts.append(
            f'<g class="qd-segment" data-quality-key="{model_id}:{axis_key}-npu" '
            f'data-qd-side="npu" tabindex="0" role="button" '
            f'aria-label="{model_id} {axis_label} NPU" aria-pressed="false">'
            f'<g transform="translate({mx:.1f} {my:.1f}) rotate({angle})">'
            f'<path class="qd-half qd-half--npu" d="M 0 18 A 18 18 0 0 1 0 -18 Z"/>'
            f"</g></g>"
        )
    split_xml = "\n      ".join(split_xml_parts)

    # Single axes (Documentation / Reliability) — one circle each.
    single_xml_parts: list[str] = []
    for axis_key in ("doc", "rel"):
        angle = _QUALITY_DEFENSE_AXIS_ANGLES[axis_key]
        mx, my = _qd_pentagon_point(angle, 75)
        axis_label = _QUALITY_DEFENSE_AXIS_LABELS[axis_key]
        single_xml_parts.append(
            f'<g class="qd-segment" data-quality-key="{model_id}:{axis_key}" '
            f'tabindex="0" role="button" '
            f'aria-label="{model_id} {axis_label}" aria-pressed="false">'
            f'<circle class="qd-circle" cx="{mx:.1f}" cy="{my:.1f}" r="18"/>'
            f"</g>"
        )
    single_xml = "\n      ".join(single_xml_parts)

    return f"""<svg class="qd-radar" viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{model_id} quality defense radar">
      <g class="qd-grid" aria-hidden="true">
        <polygon class="qd-pentagon-outer" points="{outer_str}"/>
        <polygon class="qd-pentagon-inner" points="{inner_str}"/>
        {axis_lines}
      </g>
      {single_xml}
      {split_xml}
      {axis_label_xml}
    </svg>"""


def _quality_defense_block_html() -> str:
    """Assemble the full Quality Defense Radar block — intro + 3×3 grid + legend.

    The grid is rendered as nine ``<div class="qd-cell">`` cards; each card
    carries its own ``<svg class="qd-radar">`` so click handlers remain
    isolated per model (and per segment). The ``data-quality-key`` namespace
    is ``<model-id>:<axis>[-gpu|-npu]`` so 72 unique localStorage entries are
    produced for the full grid.
    """
    cells: list[str] = []
    for model_id, display_name in _QUALITY_DEFENSE_MODELS:
        cells.append(
            f'<div class="qd-cell" data-qd-model="{model_id}">'
            f'<h3 class="qd-cell-title">{display_name}</h3>'
            f"{_qd_model_radar_svg(model_id)}"
            f"</div>"
        )
    cells_xml = "\n      ".join(cells)
    return (
        '<div class="qd-radar-wrap">'
        '<p class="qd-intro">Per-model 5-axis coverage radar across '
        '<strong>9 flagship models</strong> (Functionality / Performance / '
        'Documentation / Stability / Reliability). Click any segment to mark '
        'it as confirmed. The three split axes (Functionality / Performance / '
        'Stability) expose <strong>GPU</strong> and <strong>NPU</strong> '
        'halves of a single circle independently: GPU halves turn '
        '<strong style="color:#16a34a">green</strong> on click, NPU halves '
        'turn <strong style="color:#0284c7">blue</strong>; NPU halves also '
        'carry a dashed outline so the two sides are distinguishable even '
        'before clicking. State is persisted in <code>localStorage</code>.</p>'
        f'<div class="qd-grid">\n      {cells_xml}\n    </div>'
        '<p class="qd-legend">Click a module: gray → '
        '<strong style="color:#16a34a">green</strong> (GPU, confirmed) or '
        '<strong style="color:#0284c7">blue</strong> (NPU, confirmed); '
        'click again to revert. GPU halves carry a solid outline, NPU halves '
        'carry a dashed outline. State saved in localStorage.</p>'
        '</div>'
    )


_QUALITY_DEFENSE_BLOCK_HTML = _quality_defense_block_html()


_QUALITY_DEFENSE_MARKER = "@@QUALITY_DEFENSE_INSERTION_POINT@@"


def _upgrade_quality_defense_block(html_fragment: str) -> str:
    """Replace the ``@@QUALITY_DEFENSE_INSERTION_POINT@@`` marker with the
    inline SVG radar.

    The H2 ``## 模型质量防线`` stays as a stable Markdown heading so it
    matches :func:`_release_section_theme`'s substring match (returns the
    ``quality-defense`` modifier + shield icon) and so
    :func:`_wrap_release_report_h2_sections` wraps it in the themed
    collapsible ``<details>`` card. Click handling lives in
    :data:`_QUALITY_DEFENSE_SCRIPT`.

    Safe no-op when the section is absent (development variant / nightly
    f-string never emits the marker).
    """
    if _QUALITY_DEFENSE_MARKER not in html_fragment:
        return html_fragment
    return html_fragment.replace(_QUALITY_DEFENSE_MARKER, _QUALITY_DEFENSE_BLOCK_HTML)


_QUALITY_DEFENSE_SCRIPT = """<script>
(function () {
  var mem = {};
  function lsGet(k) {
    try { var v = localStorage.getItem(k); if (v !== null) { mem[k] = v; } return v; }
    catch (e) { return mem[k] || null; }
  }
  function lsSet(k, v) {
    try {
      if (v && v !== "0") { localStorage.setItem(k, v); mem[k] = v; }
      else { localStorage.removeItem(k); delete mem[k]; }
    } catch (e) {
      if (v && v !== "0") { mem[k] = v; } else { delete mem[k]; }
    }
  }
  function k(key) { return "quality-defense:" + key; }
  function apply(g) {
    var key = g.getAttribute("data-quality-key");
    var attr = g.getAttribute("data-quality-on");
    var v = (attr && attr !== "") ? attr : (lsGet(k(key)) || "0");
    if (v !== "1" && v !== "0") { v = "0"; }
    g.setAttribute("data-quality-on", v);
    g.setAttribute("aria-pressed", v === "1" ? "true" : "false");
  }
  function toggle(g) {
    var key = g.getAttribute("data-quality-key");
    var next = g.getAttribute("data-quality-on") === "1" ? "0" : "1";
    g.setAttribute("data-quality-on", next);
    g.setAttribute("aria-pressed", next === "1" ? "true" : "false");
    lsSet(k(key), next);
  }
  function init() {
    var nodes = document.querySelectorAll("g.qd-segment[data-quality-key]");
    for (var i = 0; i < nodes.length; i++) {
      (function (g) {
        apply(g);
        g.addEventListener("click", function () { toggle(g); });
        g.addEventListener("keydown", function (e) {
          if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
            e.preventDefault();
            toggle(g);
          }
        });
      })(nodes[i]);
    }
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
</script>"""


_RESOURCE_USAGE_SCRIPT = """<script>
(function () {
  var KEY = "resource-usage-analysis";
  function lsGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) {
    try {
      if (v === null || v === undefined) { localStorage.removeItem(k); }
      else { localStorage.setItem(k, v); }
    }
    catch (e) { /* localStorage unavailable (e.g. file:// in Chrome) */ }
  }

  // In-memory store is the source of truth for rendering; localStorage is a
  // best-effort persistence layer on top. Without this, opening the report
  // from a file:// URL would silently drop every change because the inputs
  // would otherwise re-read straight from storage on every render.
  var mem = { raw: null, modules: null };
  function readRaw() {
    if (mem.raw !== null) return mem.raw;
    mem.raw = lsGet(KEY) || "";
    return mem.raw;
  }
  function writeRaw(raw) {
    mem.raw = raw || "";
    if (mem.raw) { lsSet(KEY, mem.raw); } else { lsSet(KEY, null); }
  }

  // --- Migration: old format was a single textarea string under the same
  // key. If we see a non-JSON value, treat it as the body of a single module.
  function loadModules() {
    if (Array.isArray(mem.modules)) return mem.modules;
    var raw = readRaw();
    var parsed = null;
    if (raw) {
      try { parsed = JSON.parse(raw); } catch (e) { parsed = null; }
    }
    if (Array.isArray(parsed)) {
      // New format already.
      mem.modules = parsed.filter(function (m) { return m && typeof m === "object"; })
                          .map(function (m) { return normalize(m); });
    } else if (raw && typeof raw === "string") {
      // Legacy migration: one module carrying the old single textarea content.
      mem.modules = [makeModule({ title: "", body: raw, collapsed: false })];
      writeRaw(JSON.stringify(mem.modules));
    } else {
      mem.modules = [];
    }
    return mem.modules;
  }
  function persist() {
    var list = loadModules();
    writeRaw(JSON.stringify(list));
    var blocks = document.querySelectorAll(".resource-usage-block");
    for (var i = 0; i < blocks.length; i++) { syncBlock(blocks[i]); }
  }

  function newId() {
    return "m_" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
  }
  function makeModule(p) {
    p = p || {};
    return {
      id: p.id || newId(),
      title: typeof p.title === "string" ? p.title : "",
      body: typeof p.body === "string" ? p.body : "",
      collapsed: !!p.collapsed,
    };
  }
  function normalize(m) {
    return {
      id: typeof m.id === "string" && m.id ? m.id : newId(),
      title: typeof m.title === "string" ? m.title : "",
      body: typeof m.body === "string" ? m.body : "",
      collapsed: !!m.collapsed,
    };
  }

  // --- Rendering ----------------------------------------------------
  function escapeAttr(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/"/g, "&quot;")
      .replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function renderModule(m) {
    var node = document.createElement("div");
    node.className = "resource-usage-module";
    node.setAttribute("data-module-id", m.id);
    node.setAttribute("data-collapsed", m.collapsed ? "true" : "false");
    if (!m.title && !m.body) node.classList.add("resource-usage-module-empty");
    node.innerHTML =
      '<div class="resource-usage-module-header" role="button" tabindex="0" aria-expanded="' + (m.collapsed ? "false" : "true") + '">'
      + '<button type="button" class="resource-usage-toggle" aria-label="Collapse / expand module">' + (m.collapsed ? "▸" : "▾") + '</button>'
      + '<input type="text" class="resource-usage-module-title" placeholder="Module title…" value="' + escapeAttr(m.title) + '">'
      + '<button type="button" class="resource-usage-delete" aria-label="Delete module" title="Delete module">×</button>'
      + '</div>'
      + '<div class="resource-usage-module-body">'
      + '<textarea class="resource-usage-module-textarea" rows="14" placeholder="Module body — describe the observation, peak/avg figures, follow-up actions…">' + escapeAttr(m.body) + '</textarea>'
      + '<div class="resource-usage-module-footer">'
      + '<span class="resource-usage-module-status" data-resource-usage-state="saved">Saved</span>'
      + '</div>'
      + '</div>';
    return node;
  }

  function renderAll(block) {
    var list = loadModules();
    var host = block.querySelector(".resource-usage-modules");
    if (!host) return;
    host.innerHTML = "";
    for (var i = 0; i < list.length; i++) { host.appendChild(renderModule(list[i])); }
    syncBlock(block);
  }

  function syncBlock(block) {
    var list = loadModules();
    var total = list.length;
    var saved = 0;
    for (var i = 0; i < total; i++) {
      if (list[i].title || list[i].body) saved++;
    }
    var raw = readRaw();
    block.setAttribute("data-uri-value", raw || "");
    block.setAttribute("data-uri-state", raw ? "saved" : "empty");
    var status = block.querySelector(".resource-usage-toolbar--top .resource-usage-status");
    if (status) {
      if (total === 0) {
        status.setAttribute("data-resource-usage-state", "pristine");
        status.textContent = 'No modules yet — click "Add module" to start';
      } else {
        status.setAttribute("data-resource-usage-state", "saved");
        status.textContent = total + " module(s), " + saved + " with content";
      }
    }
  }

  // --- Mutation helpers ----------------------------------------------
  function addModule(block) {
    var list = loadModules();
    var m = makeModule({ title: "", body: "", collapsed: false });
    list.push(m);
    persist();
    renderAll(block);
    // Focus the new module's title input.
    var host = block.querySelector(".resource-usage-modules");
    if (host) {
      var last = host.lastElementChild;
      if (last) {
        var ti = last.querySelector(".resource-usage-module-title");
        if (ti) ti.focus();
      }
    }
    return m;
  }

  function deleteModule(block, id) {
    var list = loadModules();
    var idx = -1;
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { idx = i; break; } }
    if (idx === -1) return;
    list.splice(idx, 1);
    persist();
    renderAll(block);
  }

  function updateModule(block, id, patch) {
    var list = loadModules();
    var m = null;
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { m = list[i]; break; } }
    if (!m) return;
    if (Object.prototype.hasOwnProperty.call(patch, "title")) { m.title = patch.title; }
    if (Object.prototype.hasOwnProperty.call(patch, "body")) { m.body = patch.body; }
    if (Object.prototype.hasOwnProperty.call(patch, "collapsed")) { m.collapsed = !!patch.collapsed; }
    persist();
    // Light sync: status text in the toolbar + module footer.
    syncBlock(block);
    var node = block.querySelector('.resource-usage-module[data-module-id="' + id + '"]');
    if (node) {
      var footer = node.querySelector(".resource-usage-module-status");
      if (footer) {
        footer.setAttribute("data-resource-usage-state", "saved");
        footer.textContent = "Saved";
      }
      if (m.title || m.body) node.classList.remove("resource-usage-module-empty");
      else node.classList.add("resource-usage-module-empty");
    }
  }

  function toggleModule(block, id) {
    var list = loadModules();
    var m = null;
    for (var i = 0; i < list.length; i++) { if (list[i].id === id) { m = list[i]; break; } }
    if (!m) return;
    updateModule(block, id, { collapsed: !m.collapsed });
    // Reflect DOM state for collapsed / caret / aria without a full re-render.
    var node = block.querySelector('.resource-usage-module[data-module-id="' + id + '"]');
    if (node) {
      var collapsed = m.collapsed;
      node.setAttribute("data-collapsed", collapsed ? "true" : "false");
      var caret = node.querySelector(".resource-usage-toggle");
      if (caret) caret.textContent = collapsed ? "▸" : "▾";
      var header = node.querySelector(".resource-usage-module-header");
      if (header) header.setAttribute("aria-expanded", collapsed ? "false" : "true");
    }
  }

  // --- Event delegation ---------------------------------------------
  function findModule(node) {
    while (node && node !== document) {
      if (node.classList && node.classList.contains("resource-usage-module")) return node;
      node = node.parentNode;
    }
    return null;
  }
  function moduleIdFromNode(node) {
    return node ? node.getAttribute("data-module-id") : null;
  }

  // Click on Add / toggle / delete / header background.
  document.addEventListener("click", function (ev) {
    var target = ev.target;
    if (!target || !target.classList) return;

    // Add module.
    if (target.classList.contains("resource-usage-add")) {
      var block = target.closest(".resource-usage-block");
      if (block) { ev.preventDefault(); addModule(block); }
      return;
    }

    var mod = findModule(target);
    if (!mod) return;
    var block = mod.closest(".resource-usage-block");
    if (!block) return;
    var id = moduleIdFromNode(mod);

    // Delete.
    if (target.classList.contains("resource-usage-delete")) {
      ev.preventDefault();
      ev.stopPropagation();
      deleteModule(block, id);
      return;
    }
    // Toggle caret.
    if (target.classList.contains("resource-usage-toggle")) {
      ev.preventDefault();
      ev.stopPropagation();
      toggleModule(block, id);
      return;
    }
    // Header background (but not the input itself, so clicking the input
    // focuses it for editing instead of toggling).
    if (target.classList.contains("resource-usage-module-header")) {
      ev.preventDefault();
      toggleModule(block, id);
      return;
    }
  });

  // Title / body input persistence + caret click.
  document.addEventListener("input", function (ev) {
    var target = ev.target;
    if (!target || !target.classList) return;
    var mod = findModule(target);
    if (!mod) return;
    var block = mod.closest(".resource-usage-block");
    if (!block) return;
    var id = moduleIdFromNode(mod);
    if (target.classList.contains("resource-usage-module-title")) {
      updateModule(block, id, { title: target.value });
    } else if (target.classList.contains("resource-usage-module-textarea")) {
      updateModule(block, id, { body: target.value });
    }
  });

  // Keyboard: Enter on the title collapses the module; Esc on title / body blurs.
  document.addEventListener("keydown", function (ev) {
    var target = ev.target;
    if (!target || !target.classList) return;
    if (target.classList.contains("resource-usage-module-title")) {
      if (ev.key === "Enter") {
        ev.preventDefault();
        var mod = findModule(target);
        if (mod) {
          var block = mod.closest(".resource-usage-block");
          if (block) toggleModule(block, moduleIdFromNode(mod));
        }
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        target.blur();
      }
    } else if (target.classList.contains("resource-usage-module-textarea")) {
      if (ev.key === "Escape") {
        ev.preventDefault();
        target.blur();
      }
    }
  });

  function initAll() {
    var blocks = document.querySelectorAll(".resource-usage-block");
    for (var i = 0; i < blocks.length; i++) {
      // Force a fresh in-memory cache per block; the underlying localStorage
      // is shared so all blocks stay in sync.
      mem.modules = null;
      renderAll(blocks[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAll);
  } else {
    initAll();
  }
})();
</script>"""


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
    if re.fullmatch(r"A3", t, re.IGNORECASE):
        return " release-gpu-details--a3"
    if re.match(r"H100", t, re.IGNORECASE):
        return " release-gpu-details--h100"
    return ""


def _gpu_summary_icon_markup(title: str) -> str:
    t = (title or "").strip()
    paths = _RELEASE_SVG_CLOUD if re.match(r"H100", t, re.IGNORECASE) else _RELEASE_SVG_SERVER
    return _release_inline_svg(paths, size=20, extra_class="release-gpu-summary-ico")


def _gpu_short_title(title: str) -> str:
    """Reduce GPU h3 title to its short token (H100 / H200 / H800 / A100 / A3).

    ``### H100 (CI — Buildkite scheduled nightly)`` should still display as ``H100`` in the
    collapsible summary. Falls back to the original title when no token is found.
    """
    t = (title or "").strip()
    if not t:
        return t
    m = re.match(r"\s*(H100|H200|H800|A100|A3)\b", t, re.IGNORECASE)
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
) -> str:
    t = html.escape(title)
    when = generated_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = f'<p class="meta generated-meta">Generated: {html.escape(when)}</p>'
    brand = _release_brand_clipboard_svg()
    tl = html.escape(tagline)
    css = EDITORIAL_THEME_CSS + "\n" + RELEASE_MARKDOWN_DOC_CSS
    top_bar = (
        '<div class="top-bar"><div class="shell top-bar-inner">'
        '<div class="brand">'
        f'<div class="brand-mark">{brand}</div>'
        '<div class="brand-copy">'
        f"<h1>{t}</h1>"
        f'<p class="tagline">{tl}</p>'
        "</div></div>"
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
{_RELEASE_DETAILS_TOGGLE_SCRIPT}
{_FAIL_STATUS_SCRIPT}
{_GITHUB_ISSUE_SUBMIT_SCRIPT}
{_UT_COVERAGE_SUBMIT_SCRIPT}
{_SKIP_GROUP_SCRIPT}
{_OPEN_ISSUE_ACTION_SCRIPT}
{_NEXT_STEPS_OUTSTANDING_SCRIPT}
{_DI_TOP10_ASSIGNEE_SCRIPT}
{_QUALITY_DEFENSE_SCRIPT}
{_RESOURCE_USAGE_SCRIPT}
{_DEVICE_HOURS_BUILD_SCRIPT}
</body>
</html>
"""


def convert_release_report_markdown(
    md: str,
    *,
    l2_l3_row_ok: bool | None = None,
    l2_l3_row_detail: str = "",
    di_row_ok: bool | None = None,
    di_row_detail: str = "",
    critical_row_ok: bool | None = None,
    critical_row_detail: str = "",
) -> str:
    """Full HTML document from a release report Markdown string.

    Pass ``*_row_ok`` for automatic conclusion rows; when omitted, that row defaults to Pass in the conclusion table.
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
    )
    body = _wrap_release_report_h2_sections(body)
    body = _wrap_test_result_gpu_subcards(body)
    body = _fold_test_result_gpu_sections(body)
    body = _upgrade_excerpt_cells_in_failure_tables(body)
    body = _upgrade_submit_issue_cells_in_failure_tables(body)
    body = _upgrade_status_cells_in_failure_tables(body)
    body = _group_skip_monitor_table_by_issue(body)
    body = _upgrade_open_issue_action_cells(body)
    body = _upgrade_di_top10_input_cells(body)
    body = _upgrade_device_hours_cell(body)
    body = _upgrade_next_steps_outstanding_cells(body)
    body = _upgrade_quality_defense_block(body)
    body = _upgrade_resource_usage_block(body)
    body = _wrap_summary_section_in_details(body)
    body = _wrap_failure_analysis_h4_in_details(body)
    body = _wrap_pdc_h4_in_details(body)
    body = _fold_release_report_section_cards(body)
    return wrap_html_document(
        title=title,
        body_inner=body,
        generated_utc=when,
    )
