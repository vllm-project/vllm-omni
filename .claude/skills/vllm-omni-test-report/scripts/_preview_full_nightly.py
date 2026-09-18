"""Generate a preview of the FULL nightly HTML report (no real data).

Calls `nightly_local_log_report.main()` after monkey-patching the network /
disk-dependent helpers so the report renders with mock content but the
production renderer end-to-end (Daily focus, Outstanding DI, Top 20
Stale PRs, all CSS, all modals, all inline JS).

The mock dataset exercises every variant of the new Top 20 Stale PRs
section:
  - 4 reviewer groups with varying PR counts
  - 1 PR carrying a "high priority" label (surfaces to row 1)
  - 1 draft PR (visually muted)
  - 1 small group with 1 PR (no expand button)
  - 1 capped group with 8 PRs (rows 6-8 collapsed by default)

DI table has 5 mock issues spread across severities so the Outstanding
DI snapshot is non-empty.
"""

import sys
from pathlib import Path

SKILL = Path("/home/wy/.claude/skills/vllm-omni-test-report/scripts")
sys.path.insert(0, str(SKILL))

# Make sure GitHub helpers see a token so they actually try to fetch
# (we patch the fetchers below, so a fake token is fine).
import os  # noqa: E402

os.environ.setdefault("GITHUB_TOKEN", "preview-token")
os.environ.setdefault("GH_TOKEN", "preview-token")

import nightly_local_log_report as n  # noqa: E402

# ---------------------------------------------------------------------------
# Mock legacy PR groups — mirrors what _compute_top_legacy_prs returns.
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
            {
                "number": 1440,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1440",
                "title": "[Feat] Add streaming support to audio pipeline",
                "author": "yangxueyan",
                "created_at": "2026-08-25T03:00:00Z",
                "days_open": 1.0,
                "draft": False,
            },
            {
                "number": 1441,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1441",
                "title": "[Bugfix] Off-by-one in CI mirror retry counter",
                "author": "kekekuli",
                "created_at": "2026-08-25T05:00:00Z",
                "days_open": 0.95,
                "draft": False,
            },
            {
                "number": 1442,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1442",
                "title": "[CI] Add GPU memcheck lane for Wan2.x",
                "author": "lintong",
                "created_at": "2026-08-25T08:00:00Z",
                "days_open": 0.9,
                "draft": False,
            },
            {
                "number": 1443,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1443",
                "title": "[Hardware] Pin CUDA driver on H800 hosts",
                "author": "wangfan",
                "created_at": "2026-08-25T10:00:00Z",
                "days_open": 0.85,
                "draft": False,
            },
            {
                "number": 1444,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1444",
                "title": "[Doc] Add troubleshooting guide for A3 NPU",
                "author": "hsliuustc0106",
                "created_at": "2026-08-25T12:00:00Z",
                "days_open": 0.8,
                "draft": False,
            },
            {
                "number": 1445,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1445",
                "title": "[Feat] New prefill scheduler for multi-modal",
                "author": "alice-ultra",
                "created_at": "2026-08-25T14:00:00Z",
                "days_open": 0.75,
                "draft": False,
            },
            {
                "number": 1446,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1446",
                "title": "[Bug] Empty batch crashes async engine",
                "author": "docwriter",
                "created_at": "2026-08-25T16:00:00Z",
                "days_open": 0.7,
                "draft": True,
            },  # draft row
            {
                "number": 1447,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1447",
                "title": "[Test] Add regression test for KV cache eviction",
                "author": "zengchuang-hw",
                "created_at": "2026-08-25T18:00:00Z",
                "days_open": 0.65,
                "draft": False,
            },
            {
                "number": 1448,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1448",
                "title": "[Perf] Reduce GPU memory in TTS path",
                "author": "spencer",
                "created_at": "2026-08-25T19:00:00Z",
                "days_open": 0.6,
                "draft": False,
            },
            {
                "number": 1449,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1449",
                "title": "[Refactor] Split model registry into submodules",
                "author": "yangfan",
                "created_at": "2026-08-25T20:00:00Z",
                "days_open": 0.55,
                "draft": False,
            },
            {
                "number": 1451,
                "html_url": "https://github.com/vllm-project/vllm-omni/pull/1451",
                "title": "[CI] Run flake-hunter weekly on H100",
                "author": "bounty-hunter",
                "created_at": "2026-08-25T22:00:00Z",
                "days_open": 0.4,
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
# Mock Outstanding DI — 5 issues, varied severities.
# Shape: list of (di_value, title, days_open, issue_number, priority_class,
#                 assignee, linked_bugfix_pr_numbers)
# ---------------------------------------------------------------------------

MOCK_OUTSTANDING_DI = [
    (42.0, "[Bug] Qwen2-VL outputs repeat tokens on long context", 56, 1024, "critical", "alice-ultra", []),
    (28.5, "[Bug] Ascend NPU memory leak under continuous load", 38, 1058, "high", "lintong", [1614]),
    (15.0, "[Bug] Wan2.1 attention kernel regression on H800", 21, 1091, "high", "spencer", []),
    (10.0, "[Bug] MiniCPM-o fails to load on A3 with NPU=8", 14, 1112, "medium", "wangfan", []),
    (5.0, "[Bug] Stale entry in V1 engine registry after reload", 7, 1140, "medium", "docwriter", []),
]


# ---------------------------------------------------------------------------
# Monkey-patches — replace network/disk-dependent helpers with mocks.
# ---------------------------------------------------------------------------


def _fake_compute_top_legacy_prs(gh_token=None, *, now=None):
    return MOCK_TOP_PRS


def _fake_compute_outstanding_di(gh_token=None, *, now=None, overrides=None):
    # Returns the same dict shape as the real function:
    #   total_di, n_issues, counts, per_issue (7-tuple list), value, detail, severity.
    # per_issue tuples: (di, priority, days_open, issue_number, title, assignee,
    #                    linked_bugfix_prs).
    total = sum(di for di, *_ in MOCK_OUTSTANDING_DI)
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unclassified": 0}
    per_issue = []
    for di_value, title, days_open, number, priority_class, assignee, linked_bugfix_prs in MOCK_OUTSTANDING_DI:
        if priority_class in counts:
            counts[priority_class] += 1
        else:
            counts["unclassified"] += 1
        per_issue.append((di_value, priority_class, days_open, number, title, assignee, linked_bugfix_prs))
    per_issue_sorted = sorted(per_issue, key=lambda t: -t[0])
    return {
        "total_di": total,
        "n_issues": len(MOCK_OUTSTANDING_DI),
        "counts": counts,
        "per_issue": per_issue_sorted,
        "value": n._format_di_value_nightly(total),
        "detail": "5 issue(s); critical=1 high=2 medium=2",
        "severity": "fail" if total > 30 else "ok",
    }


def _fake_buildkite_perf_rows(kanban_cfg, *, log_dir=None, exclude_local_overlap=False):
    return ({}, [])


def _fake_compute_history_fail_lookup(assets_dir):
    return {}


def _fake_resolve_buildkite_for_report(include, build_no, targets=None):
    from nightly_local_log_report import ALL_BK_TARGETS

    return {t: (None, None, "preview — Buildkite section skipped") for t in ALL_BK_TARGETS}


def _fake_parse_previous_nightly_di_overrides(repo_root, *, today=None):
    return {}


n._compute_top_legacy_prs = _fake_compute_top_legacy_prs
n._compute_outstanding_di = _fake_compute_outstanding_di
n._buildkite_perf_rows = _fake_buildkite_perf_rows
n._compute_history_fail_lookup = _fake_compute_history_fail_lookup
n._resolve_buildkite_for_report = _fake_resolve_buildkite_for_report
n._parse_previous_nightly_di_overrides = _fake_parse_previous_nightly_di_overrides


# ---------------------------------------------------------------------------
# Drive the actual `main()` with empty local logs and --no-buildkite.
# ---------------------------------------------------------------------------

# Empty temp log dir so discover_job_logs returns nothing.
import tempfile  # noqa: E402

tmpdir = Path(tempfile.mkdtemp(prefix="nightly_preview_logs_"))
empty_log_dir = tmpdir / "nightly_jobs"
empty_log_dir.mkdir(parents=True, exist_ok=True)
# touch a placeholder so the dir is "real" — discover_job_logs reads it
(empty_log_dir / ".placeholder").write_text("preview", encoding="utf-8")

# Pin a fixed report date so the filename is predictable.
report_date = "2026-08-26"

out_path = Path("/tmp/preview-full-nightly.html")

# main() reads sys.argv via argparse — feed it our args.
sys.argv = [
    "nightly_local_log_report.py",
    "--no-buildkite",
    "--log-dir",
    str(empty_log_dir),
    "--report-date",
    report_date,
    "--html-report",
    str(out_path),
    "--title",
    "Nightly Test Report — 2026-08-26 (preview)",
]

print("Running nightly_local_log_report.main() with mock data…")
n.main()
print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
