"""Default laptop paths for local vLLM-Omni and vllm-omni-kanban checkouts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Documented defaults (tilde form for agent prompts and help text).
DEFAULT_LAPTOP_REPO_ROOT_DISPLAY = "~/vllm-omni"
DEFAULT_KANBAN_REPO_ROOT_DISPLAY = "~/vllm-omni-kanban"

DEFAULT_LAPTOP_REPO_ROOT = Path(DEFAULT_LAPTOP_REPO_ROOT_DISPLAY)
DEFAULT_KANBAN_REPO_ROOT = Path(DEFAULT_KANBAN_REPO_ROOT_DISPLAY)

# Canonical upstream remote URL pattern. Used to prefer the official
# ``vllm-project/vllm-omni`` checkout over a fork (e.g. ``yenuo26/vllm-omni``)
# when multiple siblings exist under ``/home/*``.
_CANONICAL_OMNI_ORIGIN_HOSTS = ("github.com/vllm-project/vllm-omni",)
_CANONICAL_OMNI_ORIGIN_SUFFIXES = ("/vllm-project/vllm-omni.git", "/vllm-project/vllm-omni")


def _git_remote_url(candidate: Path) -> str:
    """Return ``candidate``'s ``origin`` remote URL, or ``""`` on any failure."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(candidate), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return ""
    return (proc.stdout or "").strip()


def _is_canonical_omni_remote(url: str) -> bool:
    if not url:
        return False
    norm = url.strip().rstrip("/")
    return any(suffix in norm for suffix in _CANONICAL_OMNI_ORIGIN_SUFFIXES)


def _partition_canonical_first(paths: list[Path]) -> list[Path]:
    """Stable-sort ``paths`` so canonical-upstream checkouts come first."""
    canonical: list[Path] = []
    others: list[Path] = []
    for path in paths:
        url = _git_remote_url(path)
        (canonical if _is_canonical_omni_remote(url) else others).append(path)
    return canonical + others


def _home_user() -> str | None:
    """Best-effort detection of the real interactive user's home dir.

    ``$HOME`` is unreliable in containerised agents (often ``/root``); prefer
    ``pwd.getpwuid(os.getuid()).pw_dir`` and fall back to the env var. Returns
    ``None`` if neither yields a usable path.
    """
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_dir
    except (KeyError, ImportError):
        pass
    home = os.environ.get("HOME")
    return home.strip() if home else None


def resolve_kanban_repo_root(env_value: str | None = None) -> Path:
    """Resolve KANBAN_REPO_ROOT from env or ``~/vllm-omni-kanban``.

    Robust to HOME mismatches: when ``$HOME`` points at a path that doesn't
    contain a real ``vllm-omni-kanban`` checkout (typical for cc-connect /
    containerised agents where ``HOME=/root`` but the user's checkout lives
    under ``/home/<user>``), the resolver scans ``/home/*/vllm-omni-kanban``
    and returns the first match. This keeps the inheritance of yesterday's
    DI Top Contributors edits working without forcing the user to pass
    ``--kanban-repo-root`` on every nightly run.
    """
    raw = env_value if env_value is not None else (os.environ.get("KANBAN_REPO_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    default = DEFAULT_KANBAN_REPO_ROOT.expanduser().resolve()
    if default.is_dir():
        return default
    # HOME-mismatch fallback: scan /home/* for the canonical kanban checkout.
    home_root = Path("/home")
    if home_root.is_dir():
        for candidate in sorted(home_root.iterdir()):
            ck = candidate / "vllm-omni-kanban"
            if ck.is_dir():
                return ck.resolve()
    return default  # last resort; downstream callers treat missing path as no-op


def resolve_laptop_repo_root(env_value: str | None = None) -> Path:
    """Resolve REPO_ROOT from env or ``~/vllm-omni``.

    Same HOME-mismatch fallback as :func:`resolve_kanban_repo_root`: when
    ``$HOME`` doesn't contain a real ``vllm-omni`` checkout, scan
    ``/home/*/vllm-omni`` for the first existing match.

    When multiple checkouts exist (e.g. ``/home/wy/vllm-omni`` for upstream
    and ``/home/wy/vllm-omni-yn`` for the user's fork), the canonical
    ``vllm-project/vllm-omni`` remote wins over forks so the report is
    always generated against the upstream tree, not the fork.

    When the operator is currently cd'd into a fork workspace
    (``$PWD = ~/vllm-omni-yn`` or ``~/vllm-omni-fork``) and the canonical
    upstream clone sits next to it (``~/vllm-omni``), the sibling wins
    over an unrelated ``/home/<other>/vllm-omni`` whose origin also points
    at ``vllm-project/vllm-omni`` — the latter is somebody else's clone
    on a multi-user laptop layout.
    """
    raw = env_value if env_value is not None else (os.environ.get("REPO_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    # Prefer the real-user's ``vllm-omni`` (i.e. ``<real-home>/vllm-omni``)
    # over the tilde expansion, since ``$HOME`` is often wrong under
    # containerised agents. Falling back to the tilde-expanded default keeps
    # the legacy behaviour intact. A candidate is only valid when it is both
    # a directory and a git checkout (``.git`` is present) — random folders
    # named ``vllm-omni`` (e.g. a log directory) should not win the race.
    def _is_repo(p: Path) -> bool:
        try:
            return p.is_dir() and (p / ".git").exists()
        except OSError:
            return False

    real_home = _home_user()
    candidate_dirs: list[Path] = []
    # CWD sibling: when the agent runs from a workspace whose parent also
    # contains a sibling ``vllm-omni`` directory, that sibling is by far the
    # most likely "your upstream clone". Inserted FIRST so it wins over
    # alphabetical ``/home/*/vllm-omni`` collisions from other operators.
    try:
        cwd_parent = Path(os.getcwd()).resolve().parent
    except OSError:
        cwd_parent = None
    if cwd_parent is not None:
        cwd_sibling = cwd_parent / "vllm-omni"
        if _is_repo(cwd_sibling):
            candidate_dirs.append(cwd_sibling.resolve())
    if real_home:
        candidate_dirs.append(Path(real_home) / "vllm-omni")
    candidate_dirs.append(DEFAULT_LAPTOP_REPO_ROOT.expanduser().resolve())
    existing = [p for p in candidate_dirs if _is_repo(p)]
    if existing:
        return _partition_canonical_first(existing)[0]
    # HOME-mismatch fallback: scan /home/* for the canonical vllm-omni checkout,
    # preferring upstream over forks.
    home_root = Path("/home")
    if home_root.is_dir():
        matches: list[Path] = []
        for candidate in sorted(home_root.iterdir()):
            ck = candidate / "vllm-omni"
            if _is_repo(ck):
                matches.append(ck.resolve())
        if matches:
            return _partition_canonical_first(matches)[0]
    return candidate_dirs[0] if candidate_dirs else DEFAULT_LAPTOP_REPO_ROOT
