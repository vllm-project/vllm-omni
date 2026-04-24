#!/usr/bin/env python3
"""
Parse pytest commands from Buildkite test-ready.yml, test-merge.yml, test-nightly.yml;
collect test cases (including parametrized) via pytest --collect-only -q and produce an HTML report.

Usage (run from repo root):
  python scripts/buildkite_testcase_statistics.py -o buildkite_testcase_statistics.html

Requires: PyYAML (pip install pyyaml)
"""

from __future__ import annotations

import argparse
import ast
import html
import re
import subprocess
import sys
from pathlib import Path

import yaml

# Repo root (parent of the directory containing this script)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BUILDKITE_DIR = REPO_ROOT / ".buildkite"
PIPELINE_FILES = ["test-ready.yml", "test-merge.yml", "test-nightly.yml"]


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


PYTEST_CMD_RE = re.compile(r"(?:timeout\s+\S+\s+)?(?:python3? -m\s+)?pytest\s+[^\n&|;]*")


def normalize_commands(step: dict) -> list[str]:
    raw = step.get("commands") or []
    if isinstance(raw, str):
        raw = [raw]
    text = "\n".join((c.strip() if isinstance(c, str) else "") for c in raw if c)
    text = text.replace("$$", "$")
    result = []
    for m in PYTEST_CMD_RE.finditer(text):
        line = m.group(0).strip()
        line_start = text.rfind("\n", 0, m.start()) + 1
        before_match = text[line_start : m.start()]
        if before_match.lstrip().startswith("#"):
            continue
        if line:
            result.append(line)
    return result


def extract_pytest_targets_from_line(line: str) -> list[str]:
    """
    Parse pytest test targets (file path or marker expression) from a single command line.
    Returns e.g. ['tests/xxx.py'].
    When there is no explicit test file but only -m/--run-level, returns
    ['-m expr'] or ['-m expr --run-level level'] with the full marker args.
    """
    if "pytest" not in line:
        return []
    # Strip leading timeout and anything before pytest
    idx = line.find("pytest")
    if idx == -1:
        return []
    rest = line[idx:].strip()
    if not rest.startswith("pytest"):
        return []
    # Simple split on whitespace (no handling of quoted spaces)
    parts = re.split(r"\s+", rest, maxsplit=1)
    args = parts[1].strip() if len(parts) > 1 else ""
    if not args:
        return []
    targets = []
    # Match tests/.../*.py
    for m in re.finditer(r"tests/[^\s'\"#]+\.py", args):
        path = m.group(0)
        if path not in targets:
            targets.append(path)
    if not targets:
        # May be -m/--run-level only; record full marker arguments instead of a generic marker.
        extra = _parse_extra_args_from_line(line)
        if extra:
            # Join into a single string, e.g. "-m core and cpu --run-level nightly"
            targets.append(" ".join(extra))
    return targets


def get_pytest_targets_from_step(step: dict) -> list[tuple[str, str]]:
    """
    Extract all pytest-related targets from a step.
    Returns [(target, raw_line), ...] where target is tests/xxx.py or '(marker/run-level)'.
    """
    lines = normalize_commands(step)
    out = []
    seen = set()
    for line in lines:
        for t in extract_pytest_targets_from_line(line):
            if t not in seen:
                seen.add(t)
                out.append((t, line))
    return out


def _parse_extra_args_from_line(raw_line: str) -> list[str]:
    """Extract -m and --run-level args from a raw pytest command line. Strips extra spaces and quotes.
    Correctly handles quoted expressions with spaces, e.g. -m 'core_model and cpu'.
    """
    extra: list[str] = []

    # -m: match -m followed by '...', "..." or a single token (so -m expression is one argv)
    m = re.search(r"-m\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", raw_line)
    if m:
        value = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if value:
            extra.extend(["-m", value])

    r = re.search(r"--run-level\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))", raw_line)
    if r:
        value = (r.group(1) or r.group(2) or r.group(3) or "").strip()
        if value:
            extra.extend(["--run-level", value])

    return extra


def _parse_collect_only_stdout(stdout: str, *, raise_on_empty: bool = True, stderr: str = "") -> list[str]:
    """
    Parse pytest --collect-only -q stdout into a list of node ids.
    Normalizes \\r\\n and \\r to \\n before splitting.
    If raise_on_empty and no node ids found, raises RuntimeError; otherwise returns [].
    """
    raw = (stdout or "").replace("\r\n", "\n").replace("\r", "\n")
    out: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "::" not in line or "test_" not in line:
            continue
        out.append(line)
    if not out and raise_on_empty:
        raise RuntimeError(
            f"Failed to parse pytest --collect-only output: no node ids found.\nstdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    return out


def _resolve_pytest_target(target: str, repo_root: Path, extra: list[str], raw_line: str) -> tuple[list[str], str, int]:
    """
    Resolve target + repo_root + extra to (path_args, fallback_path, timeout).
    Raises FileNotFoundError/RuntimeError on invalid config.
    """
    if target.endswith(".py"):
        # Target extracted from buildkite yaml may include glob patterns like
        # tests/e2e/online_serving/test_*_expansion.py. Pytest does not reliably
        # expand globs across platforms, so expand them here.
        rel = Path(target)
        if any(ch in target for ch in ["*", "?", "["]):
            matches = sorted(repo_root.glob(rel.as_posix()))
            if not matches:
                raise FileNotFoundError(f"Pytest target glob did not match any files: {repo_root / target}")
            return [str(p.resolve()) for p in matches], target.replace("\\", "/"), 90

        path = (repo_root / rel).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Pytest target path does not exist: {path}")
        return [str(path)], target.replace("\\", "/"), 60
    if not extra:
        raise RuntimeError(f"Failed to parse -m/--run-level from pytest line: {raw_line!r}")
    if not (repo_root / "tests").exists():
        raise FileNotFoundError("tests/ directory not found under repo root")
    return ["tests/"], "tests/", 120


def collect_test_names(target: str, repo_root: Path, raw_line: str) -> list[str]:
    """
    Run pytest --collect-only -q for a target and return test node id list (incl. parametrized).
    target: path like tests/foo.py or marker string (e.g. "-m expr"). raw_line used for -m/--run-level.
    """
    extra = _parse_extra_args_from_line(raw_line)
    path_args, _fallback, timeout_quiet = _resolve_pytest_target(target, repo_root, extra, raw_line)

    cmd_quiet = [sys.executable, "-m", "pytest", *path_args, "--collect-only", "-q"]
    cmd_quiet.extend(extra)
    result = subprocess.run(
        cmd_quiet,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout_quiet,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pytest --collect-only failed for {path_args} with args {extra}: {result.stderr.strip()}")
    print(f"pytest --collect-only success for {path_args} with args {extra}: {result.stdout.strip()}")
    return _parse_collect_only_stdout(result.stdout or "", stderr=result.stderr or "")


def _extra_with_skip_marker(extra: list[str]) -> list[str]:
    """Build extra args that add -m skip: if extra has -m expr then use 'expr and skip', else -m skip."""
    out: list[str] = []
    i = 0
    has_m = False
    while i < len(extra):
        if extra[i] == "-m" and i + 1 < len(extra):
            out.extend(["-m", extra[i + 1] + " and skip"])
            has_m = True
            i += 2
        elif extra[i] == "--run-level" and i + 1 < len(extra):
            out.extend(["--run-level", extra[i + 1]])
            i += 2
        else:
            i += 1
    if not has_m:
        out = ["-m", "skip"] + out
    return out


def get_skip_status(target: str, raw_line: str, repo_root: Path) -> set[str]:
    """
    Use pytest -m skip --collect-only -q to collect tests marked with skip; return their node ids.
    Does not run tests; only collects. Supports both pytest xxx.py and pytest -m "marker".
    """
    extra = _parse_extra_args_from_line(raw_line)
    try:
        path_args, _, timeout_quiet = _resolve_pytest_target(target, repo_root, extra, raw_line)
    except (FileNotFoundError, RuntimeError):
        return set()

    extra_skip = _extra_with_skip_marker(extra)
    cmd = [sys.executable, "-m", "pytest", *path_args, "--collect-only", "-q"]
    cmd.extend(extra_skip)
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout_quiet,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return set()
    if result.returncode != 0:
        return set()
    ids = _parse_collect_only_stdout(result.stdout or "", raise_on_empty=False)
    return set(ids)


def get_docstring_for_node_id(repo_root: Path, node_id: str) -> str:
    """
    Parse the docstring of the test_ function/method from source by pytest node id.
    Supports path::test_func, path::TestClass::test_method, path::test_func[param], etc.
    """
    if not node_id or "::" not in node_id or node_id.startswith("["):
        return ""
    parts = node_id.split("::")
    # Path accepts forward slashes; no platform-specific replacement needed
    path = (repo_root / parts[0]).resolve()
    if not path.exists():
        return ""
    # Strip parametrize suffix, e.g. test_foo[param] -> test_foo
    func_part = parts[-1]
    if "[" in func_part:
        func_part = func_part.split("[", 1)[0]
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return ""
    # parts: [file, maybe_class, func] or [file, func]
    if len(parts) == 2:
        # path::test_func
        target_func = func_part
        target_class = None
    else:
        # path::TestClass::test_method
        target_class = parts[1]
        target_func = func_part
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == target_func and target_class is None:
            doc = ast.get_docstring(node)
            return (doc or "").strip()
        if isinstance(node, ast.ClassDef) and node.name == target_class:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == target_func:
                    doc = ast.get_docstring(item)
                    return (doc or "").strip()
            break
    return ""


def get_test_file_for_node_id(node_id: str) -> str:
    """Extract test file path from a pytest node id."""
    if not node_id:
        return ""
    if "::" in node_id:
        return node_id.split("::", 1)[0]
    if node_id.endswith(".py"):
        return node_id
    return "(marker-only / no collection)"


def write_html(all_stats: list[tuple], out_path: Path, total: int, repo_root: Path | None = None) -> None:
    """Write stats to HTML with styling and responsive tables."""
    root = repo_root if repo_root is not None else REPO_ROOT
    pipeline_badges = {
        "test-ready": "ready",
        "test-merge": "merge",
        "test-nightly": "nightly",
    }

    def esc(s: str) -> str:
        return html.escape(str(s))

    def tr_class(i: int) -> str:
        return "odd" if i % 2 == 0 else "even"

    pipeline_totals: dict[str, int] = {}
    for item in all_stats:
        _, pipeline_name, _, _, _, _, count, _, _, _ = item
        pipeline_totals[pipeline_name] = pipeline_totals.get(pipeline_name, 0) + count
    pipeline_summary_rows = []
    for idx, (pipeline_name, pipeline_count) in enumerate(sorted(pipeline_totals.items())):
        badge = pipeline_badges.get(pipeline_name, "")
        badge_span = f'<span class="badge {badge}">{esc(pipeline_name)}</span>' if badge else esc(pipeline_name)
        pipeline_summary_rows.append(
            f'<tr class="{tr_class(idx)}"><td>{badge_span}</td><td class="num">{pipeline_count}</td></tr>'
        )

    # Summary rows by module (item: ..., names, docstrings, skipped_ids)
    summary_rows = []
    for i, item in enumerate(all_stats):
        _, pipeline_name, label, test_files, _, marker_targets, count, _, _, _ = item
        if test_files:
            files_html = "<br>".join(esc(f) for f in test_files)
        elif marker_targets:
            # Display full marker / run-level arguments when no explicit test files.
            files_html = "<br>".join(esc(m) for m in marker_targets)
        else:
            files_html = "<em>No explicit tests or markers</em>"
        badge = pipeline_badges.get(pipeline_name, "")
        badge_span = f'<span class="badge {badge}">{esc(pipeline_name)}</span>' if badge else esc(pipeline_name)
        summary_rows.append(
            f'<tr class="{tr_class(i)}">'
            f'<td>{badge_span}</td><td>{esc(label)}</td><td class="num">{count}</td>'
            f'<td class="files">{files_html}</td></tr>'
        )

    # Detail rows grouped by Pipeline, then by Test Suite (two-level collapsible)
    case_details_thead = """
            <tr>
              <th>Pipeline</th>
              <th>Test Suite</th>
              <th>Test File</th>
              <th>No.</th>
              <th>Test Name</th>
              <th>Status</th>
              <th>Description</th>
            </tr>"""
    # Per-item: (pipeline_name, label, count, badge_span, suite_rows_html)
    pipeline_suites: dict[str, list[tuple[str, int, str, str]]] = {}  # pipeline -> [(label, count, summary, body)]
    for item in all_stats:
        _, pipeline_name, label, _, _, _, count, names, docstrings, skipped_ids = item
        badge = pipeline_badges.get(pipeline_name, "")
        badge_span = f'<span class="badge {badge}">{esc(pipeline_name)}</span>' if badge else esc(pipeline_name)
        suite_summary = f"{esc(label)} ({count} cases)"
        file_sections: list[str] = []
        if not names:
            suite_rows = [
                f'<tr class="{tr_class(0)}">'
                f"<td>{badge_span}</td><td>{esc(label)}</td>"
                f'<td class="files"><em>(marker-only / no collection)</em></td><td class="num">0</td>'
                f'<td class="name"><em>No collection or only -m/--run-level</em></td>'
                f'<td class="status"></td><td class="desc"></td></tr>'
            ]
            file_sections.append(
                '<details class="file-details" open>'
                "<summary>(marker-only / no collection)</summary>"
                f'<div class="table-wrap"><table><thead>{case_details_thead}</thead>'
                f"<tbody>{''.join(suite_rows)}</tbody></table></div></details>"
            )
        else:
            file_groups: dict[str, list[tuple[str, str, bool]]] = {}
            for idx, name in enumerate(names):
                desc = (docstrings[idx] if idx < len(docstrings) else "") or ""
                if not desc and "::" in name and not name.startswith("["):
                    desc = get_docstring_for_node_id(root, name)
                file_key = get_test_file_for_node_id(name)
                file_groups.setdefault(file_key, []).append((name, desc, name in skipped_ids))

            for file_name in sorted(file_groups.keys()):
                suite_rows = []
                for i, (name, desc, is_skipped) in enumerate(file_groups[file_name], 1):
                    desc_html = esc(desc).replace("\n", "<br>") if desc else ""
                    status_html = '<span class="badge skip">Skipped</span>' if is_skipped else "—"
                    suite_rows.append(
                        f'<tr class="{tr_class(i - 1)}">'
                        f'<td>{badge_span}</td><td>{esc(label)}</td><td class="files">{esc(file_name)}</td>'
                        f'<td class="num">{i}</td>'
                        f'<td class="name"><code>{esc(name)}</code></td>'
                        f'<td class="status">{status_html}</td><td class="desc">{desc_html}</td></tr>'
                    )
                file_sections.append(
                    f'<details class="file-details" data-test-file="{esc(file_name.lower())}">'
                    f"<summary>{esc(file_name)} ({len(file_groups[file_name])} cases)</summary>"
                    f'<div class="table-wrap"><table><thead>{case_details_thead}</thead>'
                    f"<tbody>{''.join(suite_rows)}</tbody></table></div></details>"
                )
        body = "".join(file_sections)
        if pipeline_name not in pipeline_suites:
            pipeline_suites[pipeline_name] = []
        pipeline_suites[pipeline_name].append((label, count, suite_summary, body))

    # Build two-level HTML: Pipeline (outer) -> Test Suite (inner) -> table
    detail_sections_html_parts = []
    for pipeline_name in sorted(pipeline_badges.keys()):
        if pipeline_name not in pipeline_suites:
            continue
        suites = pipeline_suites[pipeline_name]
        pipeline_total = sum(c for _, c, _, _ in suites)
        badge = pipeline_badges.get(pipeline_name, "")
        badge_span = f'<span class="badge {badge}">{esc(pipeline_name)}</span>' if badge else esc(pipeline_name)
        pipeline_summary = f"{badge_span} ({pipeline_total} cases)"
        inner_html = "".join(
            f'<details class="suite-details"><summary>{suite_summary}</summary>'
            f'<div class="case-files-inner">{body}</div></details>'
            for _, _, suite_summary, body in suites
        )
        detail_sections_html_parts.append(
            f'<details class="pipeline-details"><summary>{pipeline_summary}</summary>'
            f'<div class="case-details-inner">{inner_html}</div></details>'
        )
    # Include pipelines that might be in all_stats but not in pipeline_badges (e.g. custom names)
    for pipeline_name in sorted(pipeline_suites.keys()):
        if pipeline_name in pipeline_badges:
            continue
        suites = pipeline_suites[pipeline_name]
        pipeline_total = sum(c for _, c, _, _ in suites)
        pipeline_summary = f"{esc(pipeline_name)} ({pipeline_total} cases)"
        inner_html = "".join(
            f'<details class="suite-details"><summary>{suite_summary}</summary>'
            f'<div class="case-files-inner">{body}</div></details>'
            for _, _, suite_summary, body in suites
        )
        detail_sections_html_parts.append(
            f'<details class="pipeline-details"><summary>{pipeline_summary}</summary>'
            f'<div class="case-details-inner">{inner_html}</div></details>'
        )
    detail_sections_html = "".join(detail_sections_html_parts)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Buildkite Test Case Statistics</title>
  <style>
    :root {{
      --bg: #0f1419;
      --card: #1a2332;
      --border: #2d3a4f;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --ready: #3fb950;
      --merge: #a371f7;
      --nightly: #f0883e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      margin: 0;
      padding: 2rem;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{
      font-size: 1.75rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }}
    .meta {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
    section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.5rem;
    }}
    section h2 {{
      font-size: 1.15rem;
      font-weight: 600;
      margin: 0 0 1rem 0;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
      color: var(--accent);
    }}
    .filter-bar {{
      display: flex;
      gap: 0.75rem;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }}
    .filter-bar label {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .filter-input {{
      min-width: 320px;
      max-width: 520px;
      width: 100%;
      padding: 0.7rem 0.9rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color: var(--text);
      outline: none;
    }}
    .filter-input:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(88, 166, 255, 0.18);
    }}
    .filter-hint {{
      color: var(--muted);
      font-size: 0.82rem;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th {{
      text-align: left;
      padding: 0.65rem 0.85rem;
      background: rgba(88, 166, 255, 0.12);
      color: var(--accent);
      font-weight: 600;
      border: 1px solid var(--border);
    }}
    th:first-child {{ border-radius: 8px 0 0 0; }}
    th:last-child {{ border-radius: 0 8px 0 0; }}
    td {{
      padding: 0.6rem 0.85rem;
      border: 1px solid var(--border);
      vertical-align: top;
    }}
    tr.odd td {{ background: rgba(255,255,255,0.02); }}
    tr.even td {{ background: transparent; }}
    tr:hover td {{ background: rgba(88, 166, 255, 0.06); }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    td.files {{ font-size: 0.85rem; color: var(--muted); }}
    td.name code {{ font-size: 0.8rem; background: rgba(0,0,0,0.3); padding: 0.15em 0.4em; border-radius: 4px; }}
    td.desc {{ font-size: 0.85rem; color: var(--muted); max-width: 320px; }}
    .badge {{
      display: inline-block;
      padding: 0.2em 0.5em;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
    }}
    .badge.ready {{ background: rgba(63, 185, 80, 0.2); color: var(--ready); }}
    .badge.merge {{ background: rgba(163, 113, 247, 0.2); color: var(--merge); }}
    .badge.nightly {{ background: rgba(240, 136, 62, 0.2); color: var(--nightly); }}
    .badge.skip {{ background: rgba(139, 148, 158, 0.25); color: var(--muted); }}
    td.status {{ color: var(--muted); font-size: 0.85rem; }}
    .case-details details {{
      margin-bottom: 0.5rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .case-details details summary {{
      padding: 0.65rem 0.85rem;
      cursor: pointer;
      font-weight: 500;
      list-style: none;
      user-select: none;
    }}
    .case-details details summary::-webkit-details-marker {{ display: none; }}
    .case-details details summary::before {{
      content: "▶ ";
      font-size: 0.7rem;
      color: var(--muted);
      margin-right: 0.35rem;
    }}
    .case-details details[open] summary::before {{ content: "▼ "; }}
    .case-details details summary:hover {{ background: rgba(88, 166, 255, 0.06); }}
    .case-details details .table-wrap {{ margin: 0; border-top: 1px solid var(--border); }}
    .case-details details table {{ margin: 0; }}
    .case-details details th {{ border-top: 0; }}
    .case-details .pipeline-details {{ margin-bottom: 0.5rem; }}
    .case-details .pipeline-details .case-details-inner {{ padding-left: 0.5rem; margin-top: 0.25rem; }}
    .case-details .suite-details {{ margin-bottom: 0.35rem; margin-left: 0.5rem; }}
    .case-details .suite-details summary::before {{
      content: "▶ "; font-size: 0.7rem; color: var(--muted); margin-right: 0.35rem;
    }}
    .case-details .suite-details[open] summary::before {{ content: "▼ "; }}
    .case-details .case-files-inner {{ padding-left: 0.5rem; margin-top: 0.35rem; }}
    .case-details .file-details {{
      margin-bottom: 0.35rem;
      margin-left: 0.75rem;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }}
    .case-details .file-details summary {{
      padding: 0.55rem 0.8rem;
      cursor: pointer;
      user-select: none;
      background: rgba(255,255,255,0.015);
    }}
    .case-details .file-details summary::-webkit-details-marker {{ display: none; }}
    .case-details .file-details summary::before {{
      content: "▶ ";
      font-size: 0.7rem;
      color: var(--muted);
      margin-right: 0.35rem;
    }}
    .case-details .file-details[open] summary::before {{ content: "▼ "; }}
    .back-to-top {{
      position: fixed;
      right: 24px;
      bottom: 24px;
      display: inline-block;
      padding: 0.7rem 0.9rem;
      border-radius: 999px;
      background: rgba(88, 166, 255, 0.18);
      border: 1px solid var(--border);
      color: var(--text);
      text-decoration: none;
      font-size: 0.9rem;
      backdrop-filter: blur(6px);
    }}
    .back-to-top:hover {{ background: rgba(88, 166, 255, 0.28); }}
  </style>
</head>
<body id="top">
  <div class="container">
    <h1>Buildkite Pytest Case Statistics</h1>
    <p class="meta">Test Suites: {len(all_stats)} · Total cases (sum of steps): {total} · </p>

    <section>
      <h2>Pipeline Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Pipeline</th><th>Total Cases</th></tr>
          </thead>
          <tbody>
            {"".join(pipeline_summary_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>Summary by Test Suite</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Pipeline</th><th>Test Suite</th><th>Cases</th><th>Test Files</th></tr>
          </thead>
          <tbody>
            {"".join(summary_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="case-details">
      <h2>Case Details</h2>
      <p class="meta">Expand by Pipeline, then by Test Suite, then by Test File.</p>
      <div class="filter-bar">
        <label for="test-file-filter">Filter by Test File</label>
        <input
          id="test-file-filter"
          class="filter-input"
          type="text"
          placeholder="Type part of a test file path, e.g. tests/e2e/online_serving"
        />
        <span class="filter-hint">Filter applies to the Test File fold sections below.</span>
      </div>
      {detail_sections_html}
    </section>
  </div>
  <a href="#top" class="back-to-top" aria-label="Back to top">↑ Top</a>
  <script>
    (() => {{
      const input = document.getElementById("test-file-filter");
      if (!input) return;

      const applyFilter = () => {{
        const query = input.value.trim().toLowerCase();
        document.querySelectorAll(".pipeline-details").forEach((pipeline) => {{
          let pipelineVisible = false;
          pipeline.querySelectorAll(".suite-details").forEach((suite) => {{
            let suiteVisible = false;
            suite.querySelectorAll(".file-details").forEach((fileDetail) => {{
              const fileName = (fileDetail.dataset.testFile || fileDetail.textContent || "").toLowerCase();
              const matched = !query || fileName.includes(query);
              fileDetail.style.display = matched ? "" : "none";
              if (matched) suiteVisible = true;
            }});
            suite.style.display = suiteVisible ? "" : "none";
            if (suiteVisible) pipelineVisible = true;
          }});
          pipeline.style.display = pipelineVisible ? "" : "none";
        }});
      }};

      input.addEventListener("input", applyFilter);
      applyFilter();
    }})();
  </script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Count pytest cases from Buildkite pipelines")
    parser.add_argument(
        "--buildkite-dir",
        type=Path,
        default=BUILDKITE_DIR,
        help="Path to .buildkite directory",
    )
    default_out = REPO_ROOT / "buildkite_testcase_statistics.html"
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_out,
        help=f"Output HTML file path (default: {default_out})",
    )
    args = parser.parse_args()

    all_stats = []

    for filename in PIPELINE_FILES:
        yml_path = args.buildkite_dir / filename
        if not yml_path.exists():
            print(f"Skipped (not found): {yml_path}", file=sys.stderr)
            continue
        data = load_yaml(yml_path)
        steps = data.get("steps") or []
        pipeline_name = filename.replace(".yml", "")

        for step in steps:
            if not isinstance(step, dict):
                continue
            label = step.get("label") or "(no label)"
            # Skip commented-out steps (usually no label or commands)
            if not step.get("commands"):
                continue
            targets = get_pytest_targets_from_step(step)
            if not targets:
                continue

            module_key = f"{pipeline_name} / {label}"
            # test_files: only real .py files; marker-only steps will have an empty list.
            test_files = [t for t, _ in targets if t.endswith(".py")]
            # marker_targets: synthetic marker targets (e.g. "-m expr", "-m expr --run-level level").
            marker_targets = [t for t, _ in targets if not t.endswith(".py")]
            # marker_only: all targets are marker expressions (no explicit .py files).
            marker_only = bool(targets) and all(not t.endswith(".py") for t, _ in targets)

            names = []
            skipped_ids: set[str] = set()
            for target, raw in targets:
                collected = collect_test_names(target, REPO_ROOT, raw)
                skipped_ids.update(get_skip_status(target, raw, REPO_ROOT))
                names.extend(collected)
            count = len(names) - len(skipped_ids)
            docstrings = [
                get_docstring_for_node_id(REPO_ROOT, n) if "::" in n and not n.startswith("[") else "" for n in names
            ]
            all_stats.append(
                (
                    module_key,
                    pipeline_name,
                    label,
                    test_files,
                    marker_only,
                    marker_targets,
                    count,
                    names,
                    docstrings,
                    skipped_ids,
                )
            )

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(item[6] for item in all_stats)
    write_html(all_stats, out_path, total, REPO_ROOT)
    print(f"Written: {out_path}")
    print(f"Test Suites: {len(all_stats)}, total cases (sum of steps): {total}")


if __name__ == "__main__":
    main()
