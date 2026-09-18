"""Generate a preview HTML for the Top 20 遗留 PR section without real GitHub data.

Calls the real `_render_top_legacy_pr_section_html` from the skill so the
preview reflects the exact production renderer (CSS class names, table
layout, draft row styling, "showing top N of M waiting" suffix, etc.).
Wrap it in a minimal HTML shell so the user can preview the UI in a browser.
"""

import sys
from pathlib import Path

# Make the skill script importable
SKILL = Path("/home/wy/.claude/skills/vllm-omni-test-report/scripts")
sys.path.insert(0, str(SKILL))

import nightly_local_log_report as m  # noqa: E402

# ---------------------------------------------------------------------------
# Mock data: shaped exactly like _compute_top_legacy_prs return value.
# Hand-tuned to show every UI variant (capped vs. uncapped group, draft rows,
# single-reviewer group, fresh PR, stale PR).
# ---------------------------------------------------------------------------

MOCK_TOP_PRS = [
    {
        "reviewer": "david6666666",
        "max_days_open": 27.2,
        "pr_count": 47,
        "shown_pr_count": 20,
        "prs": [
            # High-priority PR (should surface to row 1)
            {
                "number": 1450,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1450",
                "title": "[Bug] Critical memory leak in Wan2.2 inference",
                "author": "leak-hunter",
                "created_at": "2026-08-21T05:00:00Z",
                "days_open": 5.0,
                "draft": False,
                "priority": "high priority",
            },
            {
                "number": 1330,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1330",
                "title": "[Core] Support KV Cache CPU Offloading",
                "author": "yangxueyan",
                "created_at": "2026-08-04T10:14:33Z",
                "days_open": 21.7,
                "draft": False,
            },
            {
                "number": 1365,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1365",
                "title": "[Feat] support teacache for WAN2.2",
                "author": "kekekuli",
                "created_at": "2026-08-08T03:01:11Z",
                "days_open": 17.6,
                "draft": False,
            },
            {
                "number": 1380,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1380",
                "title": "[Bugfix] Fix chunked prefill with sliding window attention",
                "author": "yangfan",
                "created_at": "2026-08-12T22:45:01Z",
                "days_open": 13.1,
                "draft": False,
            },
            {
                "number": 1392,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1392",
                "title": "[CI] Add nightly job for MiniCPM-o on Ascend A3",
                "author": "lintong",
                "created_at": "2026-08-15T07:11:42Z",
                "days_open": 10.5,
                "draft": False,
            },
            {
                "number": 1411,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1411",
                "title": "[model]Add UltraFlux-v1-image support",
                "author": "alice-ultra",
                "created_at": "2026-08-19T14:33:18Z",
                "days_open": 6.3,
                "draft": False,
            },
            {
                "number": 1424,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1424",
                "title": "[Doc] Update README with new benchmark numbers",
                "author": "docwriter",
                "created_at": "2026-08-22T09:00:01Z",
                "days_open": 3.6,
                "draft": True,
            },  # draft row
            {
                "number": 1430,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1430",
                "title": "[Refactor] Consolidate pipeline registry",
                "author": "zengchuang-hw",
                "created_at": "2026-08-24T16:22:45Z",
                "days_open": 1.4,
                "draft": False,
            },
        ],
    },
    {
        "reviewer": "Isotr0py",
        "max_days_open": 24.0,
        "pr_count": 8,
        "shown_pr_count": 8,
        "prs": [
            {
                "number": 1502,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1502",
                "title": "[Bug] Ascend NPU accuracy regression on Qwen2-VL",
                "author": "spencer",
                "created_at": "2026-08-02T05:12:00Z",
                "days_open": 24.0,
                "draft": False,
            },
            {
                "number": 1511,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1511",
                "title": "[Perf] Optimize Wan2.1 attention kernel",
                "author": "wangfan",
                "created_at": "2026-08-10T11:34:00Z",
                "days_open": 15.7,
                "draft": False,
            },
            {
                "number": 1519,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1519",
                "title": "[CI] Add H800 stable-build lane",
                "author": "hsliuustc0106",
                "created_at": "2026-08-17T20:01:00Z",
                "days_open": 8.5,
                "draft": False,
            },
        ],
    },
    {
        "reviewer": "tzhouam",
        "max_days_open": 19.9,
        "pr_count": 3,
        "shown_pr_count": 3,
        "prs": [
            {
                "number": 1601,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1601",
                "title": "[Bugfix][MiniMax-Music3] Resolve stage subdirs against a real cache path",
                "author": "zhangsan",
                "created_at": "2026-08-06T12:00:00Z",
                "days_open": 19.9,
                "draft": False,
            },
            {
                "number": 1605,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1605",
                "title": "[Hardware] Improve A3 NPU placement heuristic",
                "author": "lihao",
                "created_at": "2026-08-13T08:00:00Z",
                "days_open": 12.8,
                "draft": False,
            },
            {
                "number": 1614,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1614",
                "title": "[Doc] TTS serving tutorial",
                "author": "wendy",
                "created_at": "2026-08-23T03:00:00Z",
                "days_open": 3.0,
                "draft": True,
            },
        ],
    },
    {
        "reviewer": "alex-jw-brooks",
        "max_days_open": 12.5,
        "pr_count": 1,
        "shown_pr_count": 1,
        "prs": [
            {
                "number": 1700,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1700",
                "title": "[CI] Pin torch version for stable lane",
                "author": "bounty-hunter",
                "created_at": "2026-08-14T01:00:00Z",
                "days_open": 12.5,
                "draft": False,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

section_html = m._render_top_legacy_pr_section_html(MOCK_TOP_PRS)
heading_html = '<h3 class="focus-table-title">Top 20 Stale PRs (by Bot-Mentioned Reviewers)</h3>'
subtitle_html = (
    '<p class="focus-table-sub">Open PRs whose '
    "<code>vllm-omni-review-bot</code> comment mentions the reviewer "
    "(the bot posts a cc-style triage comment per PR). High-priority "
    "PRs surface first; remaining slots are filled by the "
    "longest-pending PRs.</p>"
)

# Minimal standalone HTML so the preview opens directly in a browser.
# Inherit the production CSS from the skill's theme module so the colors
# match the actual report exactly. The legacy-pr block lives in
# ``nightly_local_log_report.emit_report_html`` (not in the theme module),
# so we extract it directly from the source rather than duplicating the
# CSS in two places.
THEME_CSS = ""
LEGACY_PR_CSS = ""
try:
    from report_html_theme import EDITORIAL_THEME_CSS  # type: ignore

    THEME_CSS = EDITORIAL_THEME_CSS
except Exception:
    pass

try:
    import nightly_local_log_report as _nlr  # noqa: E402

    src = open(_nlr.__file__, encoding="utf-8").read()
    # Find the contiguous `+ "..."` block that contains the
    # `Top 20 遗留 PR` comment and ends at the closing `)`
    import re as _re

    block_match = _re.search(
        r"(/\* Top 20 遗留 PR.*?)(?=\n    \)\n)",
        src,
        _re.S,
    )
    if block_match:
        # Convert the literal `+ "..."` source into the actual CSS string.
        # The pieces use `\"` to escape inner double-quotes (e.g. CSS attr
        # selectors like ``[data-collapsed="true"]``), so allow escaped
        # quotes in the match. After capture, unescape ``\"`` → ``"`` and
        # ``\\`` → ``\``.
        css_pieces = _re.findall(r'\+\s*"((?:[^"\\]|\\.)*)"', block_match.group(1))
        LEGACY_PR_CSS = "".join(p.replace('\\"', '"').replace("\\\\", "\\") for p in css_pieces)
except Exception:
    pass

if not THEME_CSS:
    # Fallback to inline CSS so the preview is still usable
    THEME_CSS = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f8fafc; color: #1f2937; margin: 0; padding: 2rem; }
    section.panel { max-width: 1100px; margin: 0 auto; background: #fff;
                    border-radius: 12px; padding: 1.5rem 2rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    .focus-table-title { font-size: 1.15rem; font-weight: 600;
                          margin: 1.5rem 0 0.5rem; color: #1e293b;
                          border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; }
    .focus-table-sub { color: #475569; font-size: 0.9rem; margin: 0 0 1rem; }
    .legacy-pr-group { margin: 0.75rem 0 1rem; border: 1px solid #e2e8f0;
                        border-radius: 8px; overflow: hidden; }
    .legacy-pr-group > summary { padding: 0.7rem 1rem; cursor: pointer;
                                  background: #f1f5f9; font-weight: 500; }
    .report-subcard-body { padding: 0 1rem 1rem; overflow-x: auto; }
    .top-legacy-pr-table { width: 100%; border-collapse: collapse; }
    .top-legacy-pr-table th, .top-legacy-pr-table td { padding: 0.4rem 0.6rem;
                                                       border-bottom: 1px solid #e5e7eb; }
    .top-legacy-pr-table th { text-align: left; background: #f8fafc;
                               font-weight: 600; font-size: 0.85rem; color: #334155; }
    .top-legacy-pr-table td:nth-child(5), .top-legacy-pr-table th:nth-child(5) {
        text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;
    }
    .top-legacy-pr-table tr.legacy-pr-row--draft td { color: #94a3b8; }
    .top-legacy-pr-table .legacy-pr-title { color: #2563eb;
                                              text-decoration: none; font-weight: 500; }
    .top-legacy-pr-table .legacy-pr-title:hover { text-decoration: underline; }
    .top-legacy-pr-table .legacy-pr-count {
        display: inline-block; margin: 0.5rem 0 0;
        color: #475569; font-size: 0.85em;
    }
    /* New: show-5 default + expand button (mirrors production CSS) */
    .top-legacy-pr-table[data-collapsed="true"] tr.legacy-pr-row--collapsed {
        display: none;
    }
    .legacy-pr-expand-btn {
        display: inline-block;
        margin: 0.6rem 0 0;
        padding: 0.3rem 0.9rem;
        background: #475569;
        color: #fff;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.85em;
        font-family: inherit;
    }
    .legacy-pr-expand-btn:hover { opacity: 0.85; }
    .legacy-pr-expand-btn:focus-visible {
        outline: 2px solid #1d4ed8;
        outline-offset: 2px;
    }
    """

preview_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Preview — Top 20 遗留 PR (按 Bot @ 提及的 Reviewers 分组)</title>
<style>{THEME_CSS}
{LEGACY_PR_CSS}</style>
</head>
<body>
<section class="panel nightly-focus">
<h2>Daily focus</h2>
<p class="note">Mock preview — synthetic data, not pulled from GitHub.</p>

<h3 class="focus-table-title">Top 20 DI contributors (Outstanding DI)</h3>
<p class="focus-table-sub">(placeholder — the new section appears below this in the real report.)</p>

{heading_html}
{subtitle_html}
{section_html}

<h3 class="focus-table-title">All major regressions</h3>
<p class="focus-table-sub">(placeholder — appears after the new section in the real report.)</p>
</section>
</body>
</html>
"""

out_path = Path("/tmp/preview-bot-owners-legacy-prs.html")
out_path.write_text(preview_html, encoding="utf-8")
print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
print(f"section bytes: {len(section_html):,}")
print(f"reviewer groups: {len(MOCK_TOP_PRS)}")
