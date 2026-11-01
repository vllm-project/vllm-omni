#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Restore or publish the unmasked BF16 AITER MHA module.

Z-Image and the quarantined Stable Audio layerwise job both cold-compile
``mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale``. AITER loads an existing
``.so`` from ``AITER_JIT_DIR`` instead of compiling it. This script stores that
module as a Buildkite artifact keyed by the Python, pybind, compiler,
libstdc++, image dependency, AITER core, Torch, and ROCm identity.

The artifact service, rather than a PR-writable shared filesystem, owns the
cross-build provenance boundary. A consumer accepts artifacts only from:

* the current build, only when ``VLLM_OMNI_AITER_MHA_SAME_BUILD_STEP`` names a
  step with exactly one finished script job that exited 0. The download is
  scoped to that job. Ordinary discovery still ignores the current build, so
  the publisher cannot restore its own in-progress upload; or
* an earlier build of the same PR, head commit, merge base, and checkout; or
* a non-PR Buildkite build of the exact trusted commit on ``main``, or on the
  current release line (``release/vX.Y.Z``) when this build is itself a non-PR
  release-line push; or
* a recent non-PR build on those trusted branches whose published ABI identity
  matches this runtime. The ``.so`` is produced by AITER inside the ROCm image,
  so the identity hash—not the Omni merge-base SHA—is the compatibility gate.
  Exact trusted-commit sources are tried first; ABI-compatible recent main
  (or release-line) publishes fill the gap when the merge-base never published.

Every consumer receives a writable job-local copy. Checksums, ``gfx942``
device code, runtime ABI identity, Buildkite source metadata, and the recorded
producer provenance are verified before the module is restored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_COMMON_SCRIPTS = Path(__file__).resolve().parents[2] / "common" / "scripts"
if str(_COMMON_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_COMMON_SCRIPTS))

from ci_conditions import is_release_line_push  # noqa: E402, I001

MODULE_NAME = "mha_fwd_bf16_nbias_nmask_nlse_ndropout_nqscale"
ARCH = "gfx942"
ARCH_MARKER = b"amdhsa--gfx942"
CACHE_DIRNAME = "vllm-omni-aiter-mha-artifact"
SCHEMA = "3"
TRUSTED_REPOSITORY = "vllm-project/vllm-omni"
BUILDKITE_PIPELINE_URL = "https://buildkite.com/vllm/vllm-omni-amd-ci"
BUILDKITE_PIPELINE_SLUG = "vllm-omni-amd-ci"
SAME_BUILD_STEP_ENV = "VLLM_OMNI_AITER_MHA_SAME_BUILD_STEP"
_STEP_KEY_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
KEY_FIELDS = (
    "schema",
    "module",
    "arch",
    "aiter",
    "torch",
    "hip",
    "python",
    "python_cache_tag",
    "abi_flags",
    "pybind11",
    "compiler",
    "glibcxx",
    "image_deps",
    "core",
)
PROVENANCE_FIELDS = (
    "build_id",
    "build_number",
    "job_id",
    "commit",
    "checkout_commit",
    "branch",
    "pull_request",
    "trusted_commit",
    "repository",
    "cache_key",
)
_IMAGE_DEPENDENCY_PATHS = (
    "docker/Dockerfile.rocm",
    "docker/Dockerfile.rocm.dockerignore",
    "pyproject.toml",
    "setup.py",
    "tools/install_torchcodec_rocm.sh",
)
_READ_SIZE = 1024 * 1024
_MAX_HTTP_BYTES = 2 * 1024 * 1024
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class BuildSource:
    build_id: str
    number: int
    commit: str
    branch: str
    pull_request: str
    exact_checkout: bool
    # Set only for the verified same-build publisher. Empty for historical sources.
    job_id: str = ""


def log(message: str) -> None:
    print(f"aiter-mha-cache: {message}", file=sys.stderr, flush=True)


def module_filename() -> str:
    return f"{MODULE_NAME}.so"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_READ_SIZE):
            digest.update(block)
    return digest.hexdigest()


def contains_marker(path: Path, marker: bytes = ARCH_MARKER) -> bool:
    """Return whether ``marker`` occurs in ``path`` without reading it whole."""
    if not path.is_file() or path.stat().st_size < len(marker):
        return False
    overlap = len(marker) - 1
    previous = b""
    with path.open("rb") as handle:
        while block := handle.read(_READ_SIZE):
            window = previous + block
            if marker in window:
                return True
            previous = window[-overlap:] if overlap else b""
    return False


def cache_key(identity: Mapping[str, str]) -> str:
    missing = [field for field in KEY_FIELDS if not identity.get(field)]
    if missing:
        raise ValueError(f"AITER cache identity is missing {', '.join(missing)}")
    payload = "\n".join(f"{field}={identity[field]}" for field in KEY_FIELDS)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def artifact_prefix(identity: Mapping[str, str]) -> Path:
    return Path(CACHE_DIRNAME) / cache_key(identity) / "published"


def _read_identity(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    identity: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            return None
        field, value = line.split("=", 1)
        identity[field] = value
    if set(identity) != set(KEY_FIELDS):
        return None
    return {field: identity[field] for field in KEY_FIELDS}


def _read_provenance(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(value, dict) or set(value) != set(PROVENANCE_FIELDS):
        return None
    if not all(isinstance(value[field], str) for field in PROVENANCE_FIELDS):
        return None
    return {field: value[field] for field in PROVENANCE_FIELDS}


def is_valid_publication(published: Path) -> bool:
    module = published / module_filename()
    checksum_path = published / "module.sha256"
    ready = published / "READY"
    identity = _read_identity(published / "identity.txt")
    provenance = _read_provenance(published / "provenance.json")
    if not (module.is_file() and checksum_path.is_file() and ready.is_file()):
        return False
    if identity is None or provenance is None:
        log(f"ignoring cache without complete identity and provenance: {published}")
        return False
    recorded = checksum_path.read_text(encoding="utf-8").strip()
    if recorded != sha256_file(module):
        log(f"ignoring cache with a checksum mismatch: {module}")
        return False
    if not contains_marker(module):
        log(f"ignoring cache without {ARCH} device code: {module}")
        return False
    try:
        key = cache_key(identity)
    except ValueError as exc:
        log(str(exc))
        return False
    if (
        published.name != "published"
        or published.parent.name != key
        or published.parent.parent.name != CACHE_DIRNAME
        or provenance["cache_key"] != key
    ):
        log(f"ignoring cache whose artifact path does not match its ABI identity: {published}")
        return False
    return True


def dynamic_library_paths() -> list[Path]:
    candidates: list[Path] = []
    try:
        torch_distribution = importlib.metadata.distribution("torch")
        candidates.append(Path(torch_distribution.locate_file("torch/lib")))
    except importlib.metadata.PackageNotFoundError:
        pass
    candidates.extend((Path("/opt/rocm/lib"), Path("/opt/rocm/lib64")))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate.is_dir()))


def ldd_resolves(
    path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    library_paths: Iterable[Path] | None = None,
) -> bool:
    """Return whether the dynamic loader can resolve ``path``."""
    environment = os.environ.copy()
    search_paths = [str(candidate) for candidate in (library_paths or dynamic_library_paths())]
    search_paths.extend(filter(None, environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)))
    if search_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(search_paths))
    try:
        completed = runner(
            ["ldd", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"could not run ldd on {path}: {exc}")
        return True
    missing = [line.strip() for line in completed.stdout.splitlines() if "not found" in line]
    if completed.returncode != 0 or missing:
        detail = f": {', '.join(missing[:3])}" if missing else ""
        log(f"cached module does not load on this runtime: {path}{detail}")
        return False
    return True


def publication_matches(
    published: Path,
    identity: Mapping[str, str],
    *,
    ldd_ok: Callable[[Path], bool] = ldd_resolves,
) -> bool:
    if not is_valid_publication(published):
        return False
    recorded = _read_identity(published / "identity.txt")
    if recorded != {field: identity[field] for field in KEY_FIELDS}:
        log(f"cached ABI identity does not match this runtime: {published}")
        return False
    return ldd_ok(published / module_filename())


def restore_module(published: Path, job_dir: Path) -> bool:
    """Copy a verified module into ``job_dir``. Leave it unchanged on failure."""
    if not is_valid_publication(published):
        return False
    source = published / module_filename()
    expected = (published / "module.sha256").read_text(encoding="utf-8").strip()
    partial = job_dir / f".{module_filename()}.partial"
    shutil.copyfile(source, partial)
    if sha256_file(partial) != expected or not contains_marker(partial):
        partial.unlink(missing_ok=True)
        log(f"restored copy failed verification: {source}")
        return False
    os.replace(partial, job_dir / module_filename())
    return True


def _write_identity(path: Path, identity: Mapping[str, str]) -> None:
    lines = [f"{field}={identity[field]}" for field in KEY_FIELDS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_publication(
    job_dir: Path,
    staging_root: Path,
    identity: Mapping[str, str],
    provenance: Mapping[str, str],
) -> Path | None:
    local = job_dir / module_filename()
    if not local.is_file() or not contains_marker(local):
        log("job-local BF16 AITER module is absent; nothing to publish")
        return None
    if set(provenance) != set(PROVENANCE_FIELDS):
        log("refusing to stage an artifact with incomplete provenance")
        return None

    published = staging_root / artifact_prefix(identity)
    published.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(local, published / module_filename())
    digest = sha256_file(published / module_filename())
    (published / "module.sha256").write_text(f"{digest}\n", encoding="utf-8")
    (published / "READY").write_text("ok\n", encoding="utf-8")
    _write_identity(published / "identity.txt", identity)
    (published / "provenance.json").write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not is_valid_publication(published):
        log("staged Buildkite artifact failed verification")
        return None
    return published


def seed_prebuilt_modules(job_dir: Path, jit_dir: Path | None) -> None:
    """Copy image-shipped AITER modules so ``AITER_JIT_DIR`` can import them."""
    if jit_dir is None or not jit_dir.is_dir():
        return
    for source in sorted(jit_dir.glob("*.so")):
        if source.name == module_filename():
            continue
        destination = job_dir / source.name
        if destination.exists():
            continue
        shutil.copyfile(source, destination)


def installed_jit_dir() -> Path | None:
    candidate = Path(sysconfig.get_path("purelib")) / "aiter" / "jit"
    if candidate.is_dir():
        return candidate
    return None


def _sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value)


def job_directory() -> Path:
    job_id = _sanitize(os.environ.get("BUILDKITE_JOB_ID") or "local")
    return Path(f"/tmp/vllm-omni-aiter-mha-{job_id}")


def read_hip_version(version_file: Path = Path("/opt/rocm/.info/version")) -> str:
    if not version_file.is_file():
        return ""
    text = version_file.read_text(encoding="utf-8", errors="replace").strip()
    return text.splitlines()[0].strip() if text else ""


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name).strip()
        except importlib.metadata.PackageNotFoundError:
            continue
    return ""


def dependency_identity(root: Path) -> str:
    """Hash the same ROCm image inputs used by ``build-ci-image.sh``."""
    requirement_dir = root / "requirements"
    if not requirement_dir.is_dir():
        return ""
    requirement_files = sorted(path for path in requirement_dir.glob("*.txt") if path.is_file())
    files = [root / relative for relative in _IMAGE_DEPENDENCY_PATHS]
    files.extend(requirement_files)
    if any(not path.is_file() for path in files):
        return ""
    digest = hashlib.sha256()
    digest.update(f"rocm-arch={ARCH}\n".encode())
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"path={relative}\n".encode())
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _max_glibcxx(path: Path) -> str:
    versions: list[str] = []
    data = path.read_bytes()
    for match in re.finditer(rb"GLIBCXX_([0-9.]+)", data):
        versions.append(match.group(1).decode())
    if not versions:
        return ""

    def sort_key(value: str) -> tuple[int, ...]:
        return tuple(int(part) for part in value.split(".") if part.isdigit())

    return max(versions, key=sort_key)


def glibcxx_version() -> str:
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6"),
        Path("/lib/x86_64-linux-gnu/libstdc++.so.6"),
        Path("/usr/lib64/libstdc++.so.6"),
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_symlink():
            version = _max_glibcxx(candidate)
            if version:
                return version
    return ""


def compiler_id() -> str:
    for binary in ("hipcc", "clang"):
        try:
            completed = subprocess.run(
                [binary, "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout or completed.stderr).splitlines()
        if lines and lines[0].strip():
            return f"{binary} {lines[0].strip()}"
    return platform.python_compiler().strip()


def collect_identity(root: Path | None = None, jit_dir: Path | None = None) -> dict[str, str] | None:
    """Return the ABI identity, or None when a required field is unavailable."""
    hip = read_hip_version()
    if not hip:
        try:
            torch = __import__("torch")
            hip = str(getattr(torch.version, "hip", "") or "").strip()
        except ImportError:
            hip = ""
    checkout = Path(__file__).resolve().parents[3] if root is None else root
    core_path = (installed_jit_dir() if jit_dir is None else jit_dir) or Path()
    core_file = core_path / "module_aiter_core.so"
    identity = {
        "schema": SCHEMA,
        "module": MODULE_NAME,
        "arch": ARCH,
        "aiter": _distribution_version("amd-aiter", "aiter"),
        "torch": _distribution_version("torch"),
        "hip": hip,
        "python": sys.version.split()[0],
        "python_cache_tag": sys.implementation.cache_tag or "",
        "abi_flags": getattr(sys, "abiflags", "") or "none",
        "pybind11": _distribution_version("pybind11") or "absent",
        "compiler": compiler_id(),
        "glibcxx": glibcxx_version(),
        "image_deps": dependency_identity(checkout),
        "core": sha256_file(core_file) if core_file.is_file() else "",
    }
    try:
        cache_key(identity)
    except ValueError as exc:
        log(str(exc))
        return None
    return identity


def _normalize_repository(value: str) -> str:
    normalized = value.rstrip("/").removesuffix(".git")
    return normalized.removeprefix("https://github.com/").removeprefix("git@github.com:")


def _pull_request(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("id") or "")
    return ""


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "vllm-omni-aiter-cache/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(_MAX_HTTP_BYTES + 1)
    if len(data) > _MAX_HTTP_BYTES:
        raise ValueError(f"response is larger than {_MAX_HTTP_BYTES} bytes")
    return data.decode("utf-8")


def _source_from_details(
    details: Mapping[str, Any],
    *,
    commit: str,
    branch: str,
    pull_request: str,
    exact_checkout: bool,
) -> BuildSource | None:
    project = details.get("project")
    provider = project.get("provider") if isinstance(project, dict) else None
    if not isinstance(project, dict) or not isinstance(provider, dict):
        return None
    if project.get("slug") != BUILDKITE_PIPELINE_SLUG:
        return None
    if _normalize_repository(str(provider.get("url") or "")) != TRUSTED_REPOSITORY:
        return None
    if details.get("commit_id") != commit or details.get("branch_name") != branch:
        return None
    if _pull_request(details.get("pull_request")) != pull_request:
        return None
    build_id = str(details.get("id") or "")
    number = details.get("number")
    if not build_id or not isinstance(number, int):
        return None
    return BuildSource(build_id, number, commit, branch, pull_request, exact_checkout)


def _build_numbers_from_listing(listing: str) -> list[int]:
    return list(
        dict.fromkeys(
            int(value)
            for value in re.findall(
                rf'href="/vllm/{re.escape(BUILDKITE_PIPELINE_SLUG)}/builds/([0-9]+)"',
                listing,
            )
        )
    )


def discover_build_sources(
    *,
    branch: str,
    commit: str,
    pull_request: str,
    current_build_id: str,
    exact_checkout: bool,
    fetch: Callable[[str], str] = _fetch_text,
) -> list[BuildSource]:
    if not branch or not _SHA_PATTERN.fullmatch(commit):
        return []
    query = urllib.parse.urlencode({"branch": branch, "commit": commit})
    try:
        listing = fetch(f"{BUILDKITE_PIPELINE_URL}/builds?{query}")
    except (OSError, UnicodeError, ValueError, urllib.error.URLError):
        # Commit, branch, and exception text can contain sensitive values.
        log("could not query exact Buildkite artifact candidates")
        return []
    sources: list[BuildSource] = []
    for number in _build_numbers_from_listing(listing)[:10]:
        try:
            details = json.loads(fetch(f"{BUILDKITE_PIPELINE_URL}/builds/{number}.json"))
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError, urllib.error.URLError):
            log(f"could not verify Buildkite build #{number}")
            continue
        if not isinstance(details, dict):
            continue
        source = _source_from_details(
            details,
            commit=commit,
            branch=branch,
            pull_request=pull_request,
            exact_checkout=exact_checkout,
        )
        if source is not None and source.build_id != current_build_id:
            sources.append(source)
    return sources


def discover_recent_branch_sources(
    *,
    branch: str,
    current_build_id: str,
    limit: int = 15,
    fetch: Callable[[str], str] = _fetch_text,
) -> list[BuildSource]:
    """Return recent non-PR builds on ``branch`` for ABI-compatible fallback.

    Used when the exact trusted merge-base never published (common right after
    an ROCm image ABI change). Callers still verify identity and provenance.
    """
    if not branch or limit <= 0:
        return []
    query = urllib.parse.urlencode({"branch": branch})
    try:
        listing = fetch(f"{BUILDKITE_PIPELINE_URL}/builds?{query}")
    except (OSError, UnicodeError, ValueError, urllib.error.URLError):
        # The branch and exception may contain credentials or other secrets.
        log("could not query recent trusted-branch Buildkite builds")
        return []
    sources: list[BuildSource] = []
    for number in _build_numbers_from_listing(listing)[:limit]:
        try:
            details = json.loads(fetch(f"{BUILDKITE_PIPELINE_URL}/builds/{number}.json"))
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError, urllib.error.URLError):
            log(f"could not verify recent Buildkite build #{number}")
            continue
        if not isinstance(details, dict):
            continue
        commit = str(details.get("commit_id") or "")
        if not _SHA_PATTERN.fullmatch(commit):
            continue
        if _pull_request(details.get("pull_request")):
            continue
        source = _source_from_details(
            details,
            commit=commit,
            branch=branch,
            pull_request="",
            exact_checkout=False,
        )
        if source is not None and source.build_id != current_build_id:
            sources.append(source)
    return sources


def _trusted_branches(branch: str, pull_request: str) -> tuple[str, ...]:
    """Non-PR branches that may publish the trusted parent artifact.

    A non-PR ``release/vX.Y.Z`` push looks up its own previous build first, then
    ``main``. A pull request opened from that branch trusts only ``main``.
    """
    if branch != "main" and is_release_line_push(branch, pull_request):
        return (branch, "main")
    return ("main",)


def _passed_publisher_job(records: Any, step_key: str) -> Mapping[str, Any] | None:
    """Return the one finished publisher job, or nothing when the set is ambiguous."""
    if not isinstance(records, list):
        return None
    matches: list[Mapping[str, Any]] = []
    for job in records:
        if not isinstance(job, dict) or job.get("retried_at"):
            continue
        job_id = job.get("id")
        if (
            job.get("type") == "script"
            and job.get("step_key") == step_key
            and job.get("state") == "finished"
            and job.get("exit_status") == 0
            and job.get("passed") is True
            and job.get("soft_failed") is False
            and isinstance(job_id, str)
            and job_id
            and not any(character.isspace() for character in job_id)
        ):
            matches.append(job)
    if len(matches) != 1:
        return None
    return matches[0]


def verified_same_build_source(
    env: Mapping[str, str],
    *,
    fetch: Callable[[str], str] = _fetch_text,
) -> BuildSource | None:
    """Return the current build only after its named publisher job exits 0.

    Default discovery excludes ``BUILDKITE_BUILD_ID`` so a job cannot restore
    an upload from a sibling that is still running. Callers opt in with
    ``VLLM_OMNI_AITER_MHA_SAME_BUILD_STEP``. A missing, failed, soft-failed,
    or duplicated publisher leaves the current build excluded.
    """
    step_key = env.get(SAME_BUILD_STEP_ENV, "")
    number = env.get("BUILDKITE_BUILD_NUMBER", "")
    current_build_id = env.get("BUILDKITE_BUILD_ID", "")
    if not _STEP_KEY_PATTERN.fullmatch(step_key) or not number.isdigit() or not current_build_id:
        return None
    commit = source_commit(env)
    branch = env.get("BUILDKITE_BRANCH", "")
    pull_request = env.get("BUILDKITE_PULL_REQUEST", "")
    if pull_request in ("", "false"):
        pull_request = ""
    try:
        details = json.loads(fetch(f"{BUILDKITE_PIPELINE_URL}/builds/{number}.json"))
        jobs_payload = json.loads(fetch(f"{BUILDKITE_PIPELINE_URL}/builds/{number}/data/jobs"))
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError, urllib.error.URLError):
        log("could not verify the same-build AITER publisher")
        return None
    if not isinstance(details, dict) or not isinstance(jobs_payload, dict):
        return None
    if jobs_payload.get("has_next_page") is not False:
        log("same-build job listing was truncated; leaving the current build excluded")
        return None
    source = _source_from_details(
        details,
        commit=commit,
        branch=branch,
        pull_request=pull_request,
        exact_checkout=True,
    )
    if source is None or source.build_id != current_build_id:
        return None
    publisher = _passed_publisher_job(jobs_payload.get("records"), step_key)
    if publisher is None:
        log("same-build AITER publisher is not one passed job; leaving the current build excluded")
        return None
    return BuildSource(
        source.build_id,
        source.number,
        source.commit,
        source.branch,
        source.pull_request,
        True,
        str(publisher["id"]),
    )


def candidate_sources(
    env: Mapping[str, str] | None = None,
    *,
    fetch: Callable[[str], str] = _fetch_text,
) -> list[BuildSource]:
    values = os.environ if env is None else env
    commit = source_commit(values)
    branch = values.get("BUILDKITE_BRANCH", "")
    pull_request = values.get("BUILDKITE_PULL_REQUEST", "")
    if pull_request in ("", "false"):
        pull_request = ""
    current_build_id = values.get("BUILDKITE_BUILD_ID", "")
    same_build = verified_same_build_source(values, fetch=fetch)
    sources = discover_build_sources(
        branch=branch,
        commit=commit,
        pull_request=pull_request,
        current_build_id=current_build_id,
        exact_checkout=True,
        fetch=fetch,
    )

    trusted = values.get("VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT", "")
    if _SHA_PATTERN.fullmatch(trusted):
        for trusted_branch in _trusted_branches(branch, pull_request):
            # The exact-checkout query above already covered this commit.
            if trusted_branch == branch and trusted == commit:
                continue
            sources.extend(
                discover_build_sources(
                    branch=trusted_branch,
                    commit=trusted,
                    pull_request="",
                    current_build_id=current_build_id,
                    exact_checkout=False,
                    fetch=fetch,
                )
            )
        # After an image ABI bump the merge-base often has no publish yet.
        # Recent non-PR builds on the trusted branches can still restore a
        # module whose identity matches this runtime.
        seen = {source.build_id for source in sources}
        for trusted_branch in _trusted_branches(branch, pull_request):
            for source in discover_recent_branch_sources(
                branch=trusted_branch,
                current_build_id=current_build_id,
                fetch=fetch,
            ):
                if source.build_id not in seen:
                    sources.append(source)
                    seen.add(source.build_id)
    deduped = list({source.build_id: source for source in sources}.values())
    if same_build is None:
        return deduped
    return [same_build, *[source for source in deduped if source.build_id != same_build.build_id]]


def source_commit(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    pull_request = values.get("BUILDKITE_PULL_REQUEST", "")
    pull_request_head = values.get("BUILDKITE_PULL_REQUEST_HEAD_COMMIT", "")
    if pull_request not in ("", "false") and _SHA_PATTERN.fullmatch(pull_request_head):
        return pull_request_head
    commit = values.get("BUILDKITE_COMMIT", "")
    return commit if _SHA_PATTERN.fullmatch(commit) else ""


def checkout_commit(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    commit = values.get("BUILDKITE_COMMIT", "")
    if _SHA_PATTERN.fullmatch(commit):
        return commit
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    value = completed.stdout.strip()
    return value if _SHA_PATTERN.fullmatch(value) else ""


def provenance_from_env(
    identity: Mapping[str, str],
    env: Mapping[str, str] | None = None,
    *,
    checkout: str | None = None,
) -> dict[str, str] | None:
    values = os.environ if env is None else env
    pull_request = values.get("BUILDKITE_PULL_REQUEST", "")
    if pull_request in ("", "false"):
        pull_request = ""
    provenance = {
        "build_id": values.get("BUILDKITE_BUILD_ID", ""),
        "build_number": values.get("BUILDKITE_BUILD_NUMBER", ""),
        "job_id": values.get("BUILDKITE_JOB_ID", ""),
        "commit": source_commit(values),
        "checkout_commit": checkout_commit(values) if checkout is None else checkout,
        "branch": values.get("BUILDKITE_BRANCH", ""),
        "pull_request": pull_request,
        "trusted_commit": values.get("VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT", ""),
        "repository": _normalize_repository(values.get("BUILDKITE_REPO", "")),
        "cache_key": cache_key(identity),
    }
    if (
        provenance["repository"] != TRUSTED_REPOSITORY
        or not provenance["build_id"]
        or not provenance["build_number"].isdigit()
        or not provenance["job_id"]
        or not _SHA_PATTERN.fullmatch(provenance["commit"])
        or not _SHA_PATTERN.fullmatch(provenance["checkout_commit"])
        or not provenance["branch"]
        or not _SHA_PATTERN.fullmatch(provenance["trusted_commit"])
    ):
        return None
    return provenance


def provenance_matches_source(
    published: Path,
    source: BuildSource,
    identity: Mapping[str, str],
    env: Mapping[str, str] | None = None,
    *,
    current_checkout: str | None = None,
) -> bool:
    provenance = _read_provenance(published / "provenance.json")
    if provenance is None:
        return False
    values = os.environ if env is None else env
    expected = {
        "build_id": source.build_id,
        "build_number": str(source.number),
        "commit": source.commit,
        "branch": source.branch,
        "pull_request": source.pull_request,
        "repository": TRUSTED_REPOSITORY,
        "cache_key": cache_key(identity),
    }
    if any(provenance[field] != value for field, value in expected.items()):
        log(f"artifact provenance does not match Buildkite build #{source.number}")
        return False
    if source.job_id and provenance["job_id"] != source.job_id:
        log(f"artifact job does not match the verified publisher: build #{source.number}")
        return False
    if source.exact_checkout:
        current_pull_request = values.get("BUILDKITE_PULL_REQUEST", "")
        if current_pull_request in ("", "false"):
            current_pull_request = ""
        if (
            source.commit != source_commit(values)
            or source.branch != values.get("BUILDKITE_BRANCH", "")
            or source.pull_request != current_pull_request
        ):
            return False
        checkout = checkout_commit(values) if current_checkout is None else current_checkout
        trusted = values.get("VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT", "")
        if provenance["checkout_commit"] != checkout or provenance["trusted_commit"] != trusted:
            log(f"artifact is not from the same checkout/base identity: build #{source.number}")
            return False
    else:
        # Trusted-branch non-PR producers. Exact trusted-commit sources are
        # preferred in candidate_sources ordering; ABI-compatible recent
        # main/release publishes are also accepted because AITER compiles the
        # module from the image stack, not from the Omni merge-base tree.
        # publication_matches already verified the runtime identity hash.
        trusted_branches = _trusted_branches(
            values.get("BUILDKITE_BRANCH", ""),
            values.get("BUILDKITE_PULL_REQUEST", ""),
        )
        trusted = values.get("VLLM_OMNI_AITER_MHA_TRUSTED_COMMIT", "")
        if source.branch not in trusted_branches or source.pull_request:
            return False
        if not _SHA_PATTERN.fullmatch(trusted):
            return False
    return True


def download_from_source(
    source: BuildSource,
    job_dir: Path,
    identity: Mapping[str, str],
    env: Mapping[str, str] | None = None,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    current_checkout: str | None = None,
    ldd_ok: Callable[[Path], bool] = ldd_resolves,
) -> bool:
    prefix = artifact_prefix(identity)
    with tempfile.TemporaryDirectory(prefix="vllm-omni-aiter-download-") as temporary:
        root = Path(temporary)
        command = [
            "buildkite-agent",
            "artifact",
            "download",
            f"{prefix.as_posix()}/*",
            str(root),
            "--build",
            source.build_id,
        ]
        if source.job_id:
            # Buildkite scopes a download to a job ID through --step.
            command.extend(["--step", source.job_id])
        try:
            completed = runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log(f"could not download artifact from build #{source.number}: {exc}")
            return False
        if completed.returncode != 0:
            return False
        published = root / prefix
        if not publication_matches(published, identity, ldd_ok=ldd_ok):
            return False
        if not provenance_matches_source(
            published,
            source,
            identity,
            env,
            current_checkout=current_checkout,
        ):
            return False
        return restore_module(published, job_dir)


def emit_shell_exports(job_dir: Path) -> None:
    print(f"export AITER_JIT_DIR={shlex.quote(str(job_dir))}")
    print(f"export GPU_ARCHS={shlex.quote(ARCH)}")
    print(f"export VLLM_OMNI_AITER_MHA_JOB_DIR={shlex.quote(str(job_dir))}")


def upload_publication(
    staging_root: Path,
    identity: Mapping[str, str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    query = f"{artifact_prefix(identity).as_posix()}/*"
    try:
        completed = runner(
            ["buildkite-agent", "artifact", "upload", query],
            cwd=staging_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"artifact upload did not complete: {exc}")
        return False
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        log(f"artifact upload failed: {detail[-1] if detail else 'unknown error'}")
        return False
    return True


def prepare() -> int:
    job_dir = job_directory()
    job_dir.mkdir(parents=True, exist_ok=True)
    seed_prebuilt_modules(job_dir, installed_jit_dir())
    identity = collect_identity()
    restored_from: BuildSource | None = None
    if identity is not None:
        current_checkout = checkout_commit()
        for source in candidate_sources():
            if download_from_source(
                source,
                job_dir,
                identity,
                current_checkout=current_checkout,
            ):
                restored_from = source
                break
    if restored_from is not None:
        log(f"restored {module_filename()} from Buildkite build #{restored_from.number}")
    else:
        log(f"no verified Buildkite artifact; compiling into {job_dir}")
    emit_shell_exports(job_dir)
    return 0


def publish() -> int:
    job_value = os.environ.get("VLLM_OMNI_AITER_MHA_JOB_DIR", "")
    identity = collect_identity()
    if not job_value or identity is None:
        log("job-local AITER cache was not selected; skipping artifact upload")
        return 0
    provenance = provenance_from_env(identity)
    if provenance is None:
        log("Buildkite provenance is incomplete or untrusted; skipping artifact upload")
        return 0
    with tempfile.TemporaryDirectory(prefix="vllm-omni-aiter-upload-") as temporary:
        root = Path(temporary)
        published = stage_publication(Path(job_value), root, identity, provenance)
        if published is None:
            return 0
        if not upload_publication(root, identity):
            return 0
    log(f"published {module_filename()} as a build-scoped artifact")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "publish"))
    args = parser.parse_args(argv)
    if args.action == "prepare":
        return prepare()
    return publish()


if __name__ == "__main__":
    sys.exit(main())
