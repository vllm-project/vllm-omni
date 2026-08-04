"""Static scan of pytest skips in vllm-omni/tests/** whose reason references
a GitHub issue, and the dev-report ``## Skip Test Case Monitoring`` section.

Public entry points consumed by ``compose_full_report.py``:

    render_skip_issue_monitor_section(*, repo_root, gh_token, pull) -> str
    render_skip_issue_monitor_preview_section() -> str

The renderers never raise. Every failure path degrades to a one-line note
inside the section body so report generation is never blocked.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make sibling scripts (``md_table``) importable when this module is loaded
# directly. ``compose_full_report.py`` does the same trick for itself.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from md_table import render_markdown_table  # noqa: E402

try:  # ``requests`` is the primary HTTP client; the bugfix monitor path in
    # ``compose_full_report`` already imports it, so the dependency is
    # already required by this skill at runtime.
    import requests  # type: ignore
except Exception:  # pragma: no cover - bare environment fallback
    requests = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_REPO = "vllm-project/vllm-omni"
HOME_OWNER, HOME_REPO_NAME = HOME_REPO.split("/", 1)

# Short alias -> "owner/repo" mapping. Used when the reason string is
# "vllm issue#N" (and any other cross-repo shorthand that may appear in
# future). Unknown aliases fall back to the home repo.
_CROSS_REPO_ALIASES: dict[str, str] = {
    "vllm": "vllm-project/vllm",
}

# Pytest file globs from ``pyproject.toml [tool.pytest.ini_options]``.
# Hardcoded here (no TOML dependency) per finding 4 in the plan; the
# values are stable under ``--strict-config``.
TEST_FILE_GLOBS: tuple[str, ...] = ("test_*.py", "*_test.py")

# Issue-reference regex. Applied to reason strings only. First match wins.
# Order matters: URL first, then cross-repo, then keyword, then bare hash.
ISSUE_REF_RE = re.compile(
    r"https?://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/issues/(?P<url_n>\d+)"
    r"|(?P<xrepo>[A-Za-z][\w.-]*)\s+issue\s*#\s*(?P<xrepo_n>\d+)"
    r"|issue\s*#\s*(?P<kw_n>\d+)"
    r"|(?<![\w#])#(?P<bare_n>\d{3,})\b",
    re.IGNORECASE,
)

# Cell rendering limits.
_REASON_MAX = 80
_TITLE_MAX = 90

USER_AGENT = "vllm-omni-compose-report"
GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IssueRef:
    owner: str
    repo: str
    number: int

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def url(self) -> str:
        return f"https://github.com/{self.slug}/issues/{self.number}"

    def label(self, *, home_repo: str = HOME_REPO) -> str:
        if self.slug == home_repo:
            return f"#{self.number}"
        return f"{self.slug}#{self.number}"


@dataclass(frozen=True)
class SkipSite:
    test_file: str  # repo-relative posix
    lineno: int
    scope: str
    mark: str
    reason: str
    issue: IssueRef


@dataclass
class IssueInfo:
    ref: IssueRef
    title: str = ""
    state: str = ""
    updated_at: str = ""
    error: str = ""


@dataclass
class SkipMonitorResult:
    sites: list[SkipSite] = field(default_factory=list)
    issues: dict[tuple[str, str, int], IssueInfo] = field(default_factory=dict)
    repo_root: Path | None = None
    pull_note: str = ""
    scan_note: str = ""
    files_scanned: int = 0
    files_failed: int = 0


# ---------------------------------------------------------------------------
# Issue-reference parser
# ---------------------------------------------------------------------------


def parse_issue_ref(
    reason: str,
    *,
    home_repo: str = HOME_REPO,
) -> IssueRef | None:
    """Return the IssueRef for a reason string, or None if no issue link.

    Only the FIRST match is used. The home_repo argument controls which
    IssueRef is returned for the ``issue#N`` and ``#N`` idioms: the cross-repo
    branch is the one that escapes the home-repo default.
    """
    if not reason:
        return None
    m = ISSUE_REF_RE.search(reason)
    if not m:
        return None

    if m.group("url_n"):
        owner = m.group("owner")
        repo = m.group("repo")
        try:
            n = int(m.group("url_n"))
        except (TypeError, ValueError):
            return None
        return IssueRef(owner=owner, repo=repo, number=n)

    if m.group("xrepo_n"):
        alias = (m.group("xrepo") or "").lower()
        target = _CROSS_REPO_ALIASES.get(alias, home_repo)
        owner, repo = target.split("/", 1)
        try:
            n = int(m.group("xrepo_n"))
        except (TypeError, ValueError):
            return None
        return IssueRef(owner=owner, repo=repo, number=n)

    if m.group("kw_n"):
        owner, repo = home_repo.split("/", 1)
        try:
            n = int(m.group("kw_n"))
        except (TypeError, ValueError):
            return None
        return IssueRef(owner=owner, repo=repo, number=n)

    if m.group("bare_n"):
        owner, repo = home_repo.split("/", 1)
        try:
            n = int(m.group("bare_n"))
        except (TypeError, ValueError):
            return None
        return IssueRef(owner=owner, repo=repo, number=n)

    return None


# ---------------------------------------------------------------------------
# Repo root + pull
# ---------------------------------------------------------------------------


def _skill_ancestor_repo_root() -> Path | None:
    """Return the containing checkout if this module lives under one.

    The skill lives at ``<repo>/.claude/skills/vllm-omni-test-report/scripts/``,
    so the containing checkout is ``parents[4]``. This is the most reliable
    candidate in laptop environments where the operator is already in the
    vllm-omni working tree.
    """
    try:
        candidate = Path(__file__).resolve().parents[4]
    except IndexError:
        return None
    return candidate if (candidate / "tests").is_dir() else None


def resolve_omni_repo_root(explicit: Path | None = None) -> Path | None:
    """Resolve the vllm-omni checkout for scanning.

    Resolution order (first whose ``tests/`` is a directory wins):

    1. ``explicit`` (the ``--omni-repo-root`` CLI flag).
    2. ``$OMNI_REPO_ROOT`` env, then ``$REPO_ROOT`` env.
    3. The skill's containing checkout.
    4. ``~/vllm-omni`` (the documented laptop default).

    Returns ``None`` when no candidate has a ``tests/`` directory.
    """
    candidates: list[tuple[str, Path | None]] = []

    if explicit is not None:
        candidates.append(("explicit", Path(explicit).expanduser().resolve()))

    for env in ("OMNI_REPO_ROOT", "REPO_ROOT"):
        raw = (os.environ.get(env) or "").strip()
        if raw:
            candidates.append((f"${env}", Path(raw).expanduser().resolve()))

    ancestor = _skill_ancestor_repo_root()
    if ancestor is not None:
        candidates.append(("skill ancestor", ancestor))

    try:
        from laptop_path_defaults import resolve_laptop_repo_root

        candidates.append(("~/vllm-omni", resolve_laptop_repo_root()))
    except Exception:
        candidates.append(("~/vllm-omni", Path("~/vllm-omni").expanduser().resolve()))

    for _, path in candidates:
        if path is None:
            continue
        try:
            if (path / "tests").is_dir():
                return path
        except OSError:
            continue
    return None


def pull_omni_repo(
    repo_root: Path,
    *,
    remote: str = "origin",
    enabled: bool = True,
) -> str:
    """Best-effort fast-forward-only pull. Never raises."""
    if not enabled:
        return "pull skipped (--no-repo-pull)"
    try:
        if not (repo_root / ".git").exists():
            return f"pull skipped: {repo_root} is not a git checkout"
    except OSError:
        return f"pull skipped: {repo_root} is not a git checkout"

    try:
        branch_proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        branch = (branch_proc.stdout or "").strip() or "main"
    except (subprocess.SubprocessError, OSError) as exc:
        return f"pull skipped: failed to read current branch: {str(exc)[:120]}"

    try:
        proc = subprocess.run(
            ["git", "pull", "--ff-only", remote, branch],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "pull timed out after 180s; scanned the on-disk tree"
    except (subprocess.SubprocessError, OSError) as exc:
        return f"pull failed to start: {str(exc)[:120]}; scanned the on-disk tree"

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        first = err.splitlines()[0][:160] if err else f"exit {proc.returncode}"
        return f"pull failed ({first}); scanned the on-disk tree"

    sha_proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    sha = (sha_proc.stdout or "").strip() or "(unknown)"
    return f"pulled {remote}/{branch} -> {sha}"


# ---------------------------------------------------------------------------
# AST extractor
# ---------------------------------------------------------------------------


def iter_test_files(tests_root: Path) -> Iterator[Path]:
    """Yield every Python test file under ``tests_root``.

    Mirrors ``pyproject.toml [tool.pytest.ini_options] python_files``. Skips
    ``__pycache__``, ``.git``, and symlinked directories to avoid cycles.
    """
    if not tests_root.is_dir():
        return
    for path in tests_root.rglob("*.py"):
        if not path.is_file():
            continue
        if path.is_symlink():
            continue
        name = path.name
        if name == "__init__.py":
            continue
        if name == "conftest.py":
            continue
        if any(part == "__pycache__" or part == ".git" for part in path.parts):
            continue
        if not any(name.endswith(g[1:]) or name == g.replace("*", "") for g in TEST_FILE_GLOBS):
            # The globs are either "test_*.py" or "*_test.py"; both end in
            # ".py". Use a simple prefix/suffix check to avoid fnmatch.
            if not (name.startswith("test_") and name.endswith(".py")) and not (name.endswith("_test.py")):
                continue
        yield path


def _skip_call_kind(node: ast.AST) -> str | None:
    """Match the four skip call shapes we care about."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
        if (
            func.value.attr == "mark"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pytest"
            and func.attr in ("skip", "skipif", "xfail")
        ):
            return f"pytest.mark.{func.attr}"
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "pytest" and func.attr == "skip":
            return "pytest.skip"
    return None


def _literal_str(node: ast.AST | None) -> str | None:
    """Return the string value of a literal node, or None."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_str(node.left)
        right = _literal_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _reason_of(call: ast.Call, kind: str) -> tuple[str | None, int | None]:
    """Return ``(reason, mark_positional_index)`` for a skip call.

    ``reason=`` keyword first; otherwise positional — index 1 for ``skipif``
    (arg 0 is the condition), index 0 for the rest.
    """
    for kw in call.keywords:
        if kw.arg == "reason":
            return _literal_str(kw.value), None
    if kind == "pytest.mark.skipif" and len(call.args) >= 2:
        return _literal_str(call.args[1]), 1
    if call.args:
        return _literal_str(call.args[0]), 0
    return None, None


class _SkipVisitor(ast.NodeVisitor):
    """Walk one test file and yield ``(lineno, scope, mark, reason)`` tuples."""

    def __init__(self, declared_named_marks: dict[str, str]) -> None:
        self._named_marks = declared_named_marks
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []
        self._param_counter = 0
        self.hits: list[tuple[int, str, str, str]] = []

    # --- scope helpers -------------------------------------------------

    def _qualname(self, suffix: str = "") -> str:
        parts = [*self._class_stack, *self._func_stack]
        if not parts:
            return "<module>"
        base = "::".join(parts)
        return f"{base}::{suffix}" if suffix else base

    # --- module-level pytestmark ---------------------------------------

    def _visit_pytestmark(self, node: ast.Assign) -> None:
        value = node.value
        items: list[ast.AST]
        if isinstance(value, (ast.List, ast.Tuple)):
            items = list(value.elts)
        else:
            items = [value]
        for elt in items:
            kind = _skip_call_kind(elt) if isinstance(elt, ast.Call) else None
            if not kind:
                continue
            reason, _ = _reason_of(elt, kind)  # type: ignore[arg-type]
            if reason is None:
                continue
            self.hits.append((elt.lineno, "<module: all tests in file>", kind, reason))

    # --- named mark declared in module body ----------------------------

    def _collect_named_marks(self, tree: ast.Module) -> None:
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            kind = _skip_call_kind(value)
            if not kind:
                continue
            reason, _ = _reason_of(value, kind)
            if reason is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._named_marks[target.id] = reason

    # --- visit overrides ----------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        # Pass 1: collect module-level named marks.
        self._collect_named_marks(node)
        # Pass 2: process pytestmark assignments and visit children.
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets
            ):
                self._visit_pytestmark(stmt)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class-level decorators.
        for dec in node.decorator_list:
            self._handle_decorator(dec, scope_override=node.name)
        self._class_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._handle_decorators(node.decorator_list, node.name)
        self._func_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._handle_decorators(node.decorator_list, node.name)
        self._func_stack.append(node.name)
        try:
            self.generic_visit(node)
        finally:
            self._func_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        kind = _skip_call_kind(node)
        if kind == "pytest.skip" and self._func_stack:
            reason, _ = _reason_of(node, kind)
            if reason is not None:
                self.hits.append((node.lineno, f"{self._qualname()} (runtime)", kind, reason))
        # Detect marks=NAMED inside pytest.param(...)
        if (
            isinstance(node.func, ast.Name) and node.func.id == "pytest" and getattr(node.func, "attr", None) is None
        ) or (isinstance(node.func, ast.Attribute) and node.func.attr == "param"):
            self._visit_pytest_param(node)
        self.generic_visit(node)

    # --- decorator + param helpers -------------------------------------

    def _handle_decorators(self, decorators: list[ast.AST], func_name: str) -> None:
        for dec in decorators:
            self._handle_decorator(dec, scope_override=None, func_name=func_name)

    def _handle_decorator(
        self,
        dec: ast.AST,
        *,
        scope_override: str | None,
        func_name: str | None = None,
    ) -> None:
        # Inline pytest.mark.skip(...).
        if isinstance(dec, ast.Call):
            kind = _skip_call_kind(dec)
            if kind:
                reason, _ = _reason_of(dec, kind)
                if reason is not None:
                    scope = scope_override or (f"{self._qualname()}{'::' + func_name if func_name else ''}")
                    self.hits.append((dec.lineno, scope, kind, reason))
            return
        # @NAMED_MARK
        if isinstance(dec, ast.Name) and dec.id in self._named_marks:
            reason = self._named_marks[dec.id]
            if scope_override is not None:
                scope = f"{scope_override} (all tests in class)"
            elif func_name is not None:
                scope = self._qualname(func_name)
            else:
                scope = self._qualname()
            self.hits.append((dec.lineno, scope, f"pytest.mark.skip (via {dec.id})", reason))

    def _visit_pytest_param(self, call: ast.Call) -> None:
        """Detect ``marks=NAMED`` or ``marks=pytest.mark.skip(...)`` in pytest.param."""
        for kw in call.keywords:
            if kw.arg != "marks":
                continue
            self._handle_marks_kwarg(kw.value, call)

    def _handle_marks_kwarg(self, value: ast.AST, call: ast.Call) -> None:
        marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        for mark in marks:
            if isinstance(mark, ast.Name) and mark.id in self._named_marks:
                reason = self._named_marks[mark.id]
                scope = self._param_scope(call, mark.id)
                self.hits.append((mark.lineno, scope, f"pytest.mark.skip (via {mark.id})", reason))
            elif isinstance(mark, ast.Call):
                kind = _skip_call_kind(mark)
                if not kind:
                    continue
                reason, _ = _reason_of(mark, kind)
                if reason is None:
                    continue
                scope = self._param_scope(call, kind)
                self.hits.append((mark.lineno, scope, kind, reason))

    def _param_scope(self, call: ast.Call, tag: str) -> str:
        # Prefer the ``id=`` keyword.
        for kw in call.keywords:
            if kw.arg == "id":
                lit = _literal_str(kw.value)
                if lit:
                    return f"<param: {lit}>"
        # Fall back to the first positional literal.
        if call.args:
            lit = _literal_str(call.args[0])
            if lit:
                snippet = lit if len(lit) <= 40 else lit[:37] + "…"
                return f"<param: {snippet}>"
        return f"<param via {tag}>"


def extract_skip_sites_from_source(source: str, *, rel_path: str) -> list[tuple[int, str, str, str]]:
    """Return ``[(lineno, scope, mark, reason), ...]`` for one file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    visitor = _SkipVisitor(declared_named_marks={})
    visitor._collect_named_marks(tree)
    visitor.visit(tree)
    return visitor.hits


def scan_skip_sites(
    repo_root: Path,
) -> tuple[list[SkipSite], int, str]:
    """Scan ``repo_root/tests`` and return ``(sites, files_scanned, note)``."""
    tests_root = repo_root / "tests"
    sites: list[SkipSite] = []
    files_scanned = 0
    files_failed = 0
    if not tests_root.is_dir():
        return [], 0, f"no tests/ directory under {repo_root}"
    for path in iter_test_files(tests_root):
        files_scanned += 1
        rel = path.relative_to(repo_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            files_failed += 1
            continue
        try:
            hits = extract_skip_sites_from_source(source, rel_path=rel)
        except SyntaxError:
            files_failed += 1
            continue
        for lineno, scope, mark, reason in hits:
            issue = parse_issue_ref(reason)
            if issue is None:
                continue
            sites.append(
                SkipSite(
                    test_file=rel,
                    lineno=lineno,
                    scope=scope,
                    mark=mark,
                    reason=reason,
                    issue=issue,
                )
            )
    note = ""
    if files_failed:
        note = f"{files_failed} file(s) failed to parse"
    return sites, files_scanned, note


# ---------------------------------------------------------------------------
# GitHub fetch
# ---------------------------------------------------------------------------


def _github_tls_verify() -> bool:
    v = (os.environ.get("GITHUB_INSECURE_SSL") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        # Suppress the InsecureRequestWarning the same way the rest of the
        # skill does in compose_full_report.py (line 123-129).
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        return False
    return True


def _http_get(url: str, *, gh_token: str | None, timeout: int = 30) -> tuple[int, dict[str, str], bytes]:
    """Single GET. Returns ``(status_code, headers, body)``."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
    }
    if gh_token:
        headers["Authorization"] = f"Bearer {gh_token}"
    if requests is not None:
        resp = requests.get(url, headers=headers, timeout=timeout, verify=_github_tls_verify())
        return resp.status_code, dict(resp.headers), resp.content
    # Fallback to urllib.
    import urllib.request

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        return exc.code, dict(exc.headers or {}), exc.read() or b""


def fetch_issue(ref: IssueRef, gh_token: str | None, *, timeout: int = 30) -> IssueInfo:
    """Fetch a single issue. Never raises."""
    info = IssueInfo(ref=ref)
    url = f"{GITHUB_API}/repos/{ref.slug}/issues/{ref.number}"
    try:
        status, headers, body = _http_get(url, gh_token=gh_token, timeout=timeout)
    except Exception as exc:  # pragma: no cover - defensive
        info.error = str(exc)[:120]
        return info

    if status == 200:
        try:
            import json as _json

            payload = _json.loads(body.decode("utf-8"))
        except Exception as exc:
            info.error = f"json decode failed: {str(exc)[:80]}"
            return info
        info.title = (payload.get("title") or "").strip()
        info.state = (payload.get("state") or "").strip().lower()
        upd = (payload.get("updated_at") or "").strip()
        info.updated_at = upd[:10] if upd else ""
        return info
    if status in (404, 410):
        info.error = "not found"
        return info
    if status == 301:
        # Issue transferred between repos. Follow Location once.
        location = headers.get("Location") or headers.get("location") or ""
        if location:
            try:
                status, _, body = _http_get(location, gh_token=gh_token, timeout=timeout)
            except Exception:
                info.error = "redirect fetch failed"
                return info
            if status == 200:
                import json as _json

                try:
                    payload = _json.loads(body.decode("utf-8"))
                except Exception as exc:
                    info.error = f"json decode failed: {str(exc)[:80]}"
                    return info
                info.title = (payload.get("title") or "").strip()
                info.state = (payload.get("state") or "").strip().lower()
                upd = (payload.get("updated_at") or "").strip()
                info.updated_at = upd[:10] if upd else ""
                return info
        info.error = "redirect without Location"
        return info
    if status in (403, 429):
        info.error = "rate limited"
        return info
    info.error = f"HTTP {status}"
    return info


def fetch_issues(
    refs: Iterable[IssueRef],
    gh_token: str | None,
    *,
    max_fetches: int = 40,
    timeout: int = 30,
) -> dict[tuple[str, str, int], IssueInfo]:
    """Fetch issues; dedupes; respects rate limits and ``max_fetches``."""
    unique = sorted({(r.owner, r.repo, r.number) for r in refs}, reverse=True)
    out: dict[tuple[str, str, int], IssueInfo] = {}
    rate_limited = False
    fetched = 0
    for key in unique:
        if rate_limited:
            ref = IssueRef(*key)
            out[key] = IssueInfo(ref=ref, error="rate limited")
            continue
        if fetched >= max_fetches:
            ref = IssueRef(*key)
            out[key] = IssueInfo(ref=ref, error="max_fetches reached")
            continue
        ref = IssueRef(*key)
        info = fetch_issue(ref, gh_token, timeout=timeout)
        out[key] = info
        fetched += 1
        if info.error == "rate limited":
            rate_limited = True
    return out


# ---------------------------------------------------------------------------
# Orchestration + render
# ---------------------------------------------------------------------------


def collect_skip_monitor(
    *,
    repo_root: Path | None = None,
    gh_token: str | None = None,
    pull: bool = True,
    max_fetches: int = 40,
) -> SkipMonitorResult:
    """End-to-end: resolve, pull (optional), scan, fetch."""
    result = SkipMonitorResult()
    resolved = resolve_omni_repo_root(explicit=repo_root)
    if resolved is None:
        result.repo_root = None
        result.pull_note = (
            "no vllm-omni repo root resolved "
            "(checked explicit, $OMNI_REPO_ROOT/$REPO_ROOT, skill ancestor, ~/vllm-omni)"
        )
        result.scan_note = "scan skipped"
        return result
    result.repo_root = resolved
    result.pull_note = pull_omni_repo(resolved, enabled=pull)

    sites, files_scanned, scan_note = scan_skip_sites(resolved)
    result.sites = sites
    result.files_scanned = files_scanned
    result.files_failed = sum(1 for _ in [])  # placeholder; updated below
    result.scan_note = scan_note
    if sites:
        result.issues = fetch_issues((s.issue for s in sites), gh_token, max_fetches=max_fetches)
    return result


# ---------- cell rendering -------------------------------------------------


def _md_cell(value: Any) -> str:
    """Escape a value for inclusion in a Markdown table cell.

    ``render_markdown_table`` does not escape pipes, so we do it here.
    Newlines collapse to spaces; backticks would break inline code, but
    the backticked substrings are produced by us, not user input.
    """
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    return text.replace("|", "\\|")


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _file_link(test_file: str, lineno: int) -> str:
    return f"[`{test_file}`](https://github.com/{HOME_REPO}/blob/main/{test_file}#L{lineno})"


def _issue_link(ref: IssueRef) -> str:
    return f"[{ref.label()}]({ref.url})"


#: Column order for the ``## Skip Test Case Monitoring`` table.
#: The four issue-level columns lead so the HTML post-processor
#: (``release_md_to_html._group_skip_monitor_table_by_issue``) can fold every
#: site that references the same issue number under one collapsible group row.
#: Keep this in sync with ``render_skip_monitor_table`` /
#: ``render_skip_issue_monitor_preview_section`` row order.
SKIP_MONITOR_HEADERS: list[str] = [
    "Issue #",
    "Issue Title",
    "Issue State",
    "Issue Updated",
    "Test File",
    "Test",
    "Skip Mark",
    "Skip Reason",
]

#: Number of leading issue-level columns (blanked on child rows in HTML).
SKIP_MONITOR_ISSUE_COLS = 4


def render_skip_monitor_table(result: SkipMonitorResult) -> tuple[list[str], list[list[str]]]:
    """Return ``(headers, rows)`` for the dev-report table.

    Rows are ordered so that every site sharing an issue is **contiguous**
    (newest issue number first); the HTML layer relies on that adjacency to
    build one collapsible group per issue.
    """
    headers = list(SKIP_MONITOR_HEADERS)

    if not result.sites:
        return headers, []

    def sort_key(site: SkipSite) -> tuple[int, str, str, str, int]:
        return (
            -site.issue.number,
            site.issue.owner,
            site.issue.repo,
            site.test_file,
            site.lineno,
        )

    sites = sorted(result.sites, key=sort_key)
    rows: list[list[str]] = []
    for site in sites:
        info = result.issues.get((site.issue.owner, site.issue.repo, site.issue.number))
        title = info.title if info else ""
        state = (info.state if info else "").lower()
        updated = info.updated_at if info else ""

        if state == "closed":
            state_cell = "**closed**"
        elif state == "open":
            state_cell = "open"
        else:
            state_cell = "—"

        rows.append(
            [
                _issue_link(site.issue),
                _md_cell(_truncate(title, _TITLE_MAX)) if title else "—",
                state_cell,
                updated or "—",
                _file_link(site.test_file, site.lineno),
                f"`{_md_cell(site.scope)}`",
                f"`{_md_cell(site.mark)}`",
                _md_cell(_truncate(site.reason, _REASON_MAX)),
            ]
        )
    return headers, rows


def render_skip_issue_monitor_section(
    *,
    repo_root: Path | None = None,
    gh_token: str | None = None,
    pull: bool = True,
) -> str:
    """Return the full ``## Skip Test Case Monitoring`` section as Markdown.

    Never raises. Always returns a string ending in exactly one newline.
    """
    try:
        result = collect_skip_monitor(repo_root=repo_root, gh_token=gh_token, pull=pull)
    except Exception as exc:  # pragma: no cover - defensive
        return f"## Skip Test Case Monitoring\n\n*Collector failed: {str(exc)[:160]}*\n"

    lines: list[str] = ["## Skip Test Case Monitoring", ""]
    lines.append(
        "Static scan of `tests/**` for pytest skips whose reason links a "
        "GitHub issue (`https://github.com/<repo>/issues/N`, `issue#N`, "
        "`issue #N`, `#N`, or `vllm issue#N` cross-repo)."
    )
    lines.append("")

    n_sites = len(result.sites)
    n_issues = len({(s.issue.owner, s.issue.repo, s.issue.number) for s in result.sites})
    n_closed = sum(1 for info in result.issues.values() if info.state == "closed")

    if result.repo_root is None:
        lines.append(f"- **{result.pull_note}***")
        return "\n".join(lines).rstrip("\n") + "\n"

    lines.append(f"- **Skipped sites:** {n_sites} - **distinct issues:** {n_issues} - **already closed:** {n_closed}")
    lines.append(f"- **Repo:** `{result.repo_root}` - {result.pull_note}")
    lines.append(
        f"- **Scanned:** {result.files_scanned} test file(s) under `tests/`"
        + (f" - {result.scan_note}" if result.scan_note else "")
    )
    if gh_token is None and n_sites:
        lines.append(
            "- *GitHub fetch ran unauthenticated; Issue Title / State / "
            "Updated may show `—` if the API rate-limited the request.*"
        )
    lines.append("")

    if not result.sites:
        lines.append("*No issue-linked skips found - every skip in `tests/` is environmental.*")
        return "\n".join(lines).rstrip("\n") + "\n"

    headers, rows = render_skip_monitor_table(result)
    lines.append(render_markdown_table(headers, rows))
    return "\n".join(lines).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Preview (no git, no network)
# ---------------------------------------------------------------------------


def render_skip_issue_monitor_preview_section() -> str:
    """Hardcoded 5-row preview used by ``--preview --kind development``.

    Two rows deliberately share issue ``#4636`` so the HTML collapsible
    grouping (one group row per issue number) is visible in preview mode.
    """
    headers = list(SKIP_MONITOR_HEADERS)
    rows: list[list[str]] = [
        [
            f"[#4636](https://github.com/{HOME_REPO}/issues/4636)",
            "Expansion pipeline: model loader returns 503 on cold start",
            "open",
            "2026-07-21",
            f"[`tests/e2e/offline_inference/test_sensenova_u1_img2img_expansion.py`](https://github.com/{HOME_REPO}/blob/main/tests/e2e/offline_inference/test_sensenova_u1_img2img_expansion.py#L158)",  # noqa: E501
            "`test_pipeline_runs`",
            "`pytest.mark.skip`",
            "https://github.com/vllm-project/vllm-omni/issues/4636",
        ],
        [
            f"[#4636](https://github.com/{HOME_REPO}/issues/4636)",
            "Expansion pipeline: model loader returns 503 on cold start",
            "open",
            "2026-07-21",
            f"[`tests/e2e/offline_inference/test_sensenova_u1_img2img_expansion.py`](https://github.com/{HOME_REPO}/blob/main/tests/e2e/offline_inference/test_sensenova_u1_img2img_expansion.py#L204)",  # noqa: E501
            "`test_pipeline_batch`",
            "`pytest.mark.skip`",
            "issue #4636",
        ],
        [
            f"[#4285](https://github.com/{HOME_REPO}/issues/4285)",
            "VoxCPM2 long-stream drops frames after 2h",
            "**closed**",
            "2026-06-30",
            f"[`tests/dfx/reliability/test_reliability_voxcpm2.py`](https://github.com/{HOME_REPO}/blob/main/tests/dfx/reliability/test_reliability_voxcpm2.py#L196)",  # noqa: E501
            "`TestVoxCPM2Reliability::test_long_stream`",
            "`pytest.mark.skip`",
            "issue#4285",
        ],
        [
            f"[#3256](https://github.com/{HOME_REPO}/issues/3256)",
            "GeBench H100 smoke needs CUDA 12.4 driver",
            "open",
            "2026-05-12",
            f"[`tests/e2e/accuracy/test_gebench_h100_smoke.py`](https://github.com/{HOME_REPO}/blob/main/tests/e2e/accuracy/test_gebench_h100_smoke.py#L17)",  # noqa: E501
            "`<module: all tests in file>`",
            "`pytest.mark.skip`",
            "#3256",
        ],
        [
            "[vllm-project/vllm#43060](https://github.com/vllm-project/vllm/issues/43060)",
            "Upstream: serve-root SIGINT handler swallows reaped children",
            "open",
            "2026-07-09",
            f"[`tests/dfx/reliability/test_reliability_qwen3_omni.py`](https://github.com/{HOME_REPO}/blob/main/tests/dfx/reliability/test_reliability_qwen3_omni.py#L126)",  # noqa: E501
            "`TestQwen3OmniReliability::test_serve_root_signal`",
            "`pytest.mark.skip`",
            "vllm issue#43060",
        ],
    ]
    body = render_markdown_table(headers, rows)
    return (
        "## Skip Test Case Monitoring\n\n"
        "*This section uses **preview placeholder data** — no git pull, no "
        "AST scan, no GitHub API call was made.*\n\n"
        "- **Skipped sites:** 5 (preview) - **distinct issues:** 4 - "
        "**already closed:** 1\n"
        "- **Repo:** (preview; not resolved)\n"
        "- **Scanned:** 5 row(s) (preview)\n\n"
        f"{body}\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Scan vllm-omni/tests for issue-linked pytest skips and render "
            "the dev-report section. Use --no-pull and --markdown for offline "
            "debugging."
        )
    )
    parser.add_argument(
        "--omni-repo-root",
        type=Path,
        default=None,
        help="Explicit vllm-omni checkout path (overrides env and defaults).",
    )
    parser.add_argument("--no-pull", action="store_true", help="Skip the git pull step.")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit the section Markdown to stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the scan result as JSON to stdout.",
    )
    parser.add_argument(
        "--max-fetches",
        type=int,
        default=40,
        help="Maximum number of GitHub issue fetches (default 40).",
    )
    args = parser.parse_args(argv)

    gh_token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip() or None
    result = collect_skip_monitor(
        repo_root=args.omni_repo_root,
        gh_token=gh_token,
        pull=not args.no_pull,
        max_fetches=args.max_fetches,
    )

    if args.json:
        payload = {
            "repo_root": str(result.repo_root) if result.repo_root else None,
            "pull_note": result.pull_note,
            "scan_note": result.scan_note,
            "files_scanned": result.files_scanned,
            "files_failed": result.files_failed,
            "sites": [
                {
                    "test_file": s.test_file,
                    "lineno": s.lineno,
                    "scope": s.scope,
                    "mark": s.mark,
                    "reason": s.reason,
                    "issue": {
                        "owner": s.issue.owner,
                        "repo": s.issue.repo,
                        "number": s.issue.number,
                    },
                }
                for s in result.sites
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.markdown:
        section = render_skip_issue_monitor_section(
            repo_root=args.omni_repo_root,
            gh_token=gh_token,
            pull=not args.no_pull,
        )
        print(section, end="")
        return 0

    # Default: short summary.
    print(f"repo_root: {result.repo_root}")
    print(f"pull: {result.pull_note}")
    print(f"files_scanned: {result.files_scanned}")
    print(f"sites: {len(result.sites)}")
    distinct = {(s.issue.owner, s.issue.repo, s.issue.number) for s in result.sites}
    print(f"distinct_issues: {len(distinct)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
