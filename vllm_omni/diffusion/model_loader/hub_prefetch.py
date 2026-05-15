# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Best-effort HuggingFace Hub prefetch for multi-subfolder pipelines.

This module exists to defend diffusion pipelines against a race condition we
hit after the `transformers` v5 rebase (see Buildkite vllm-omni-rebase
#1043 Qwen-Image-Edit-2509 failure):

When several diffusion worker processes start in parallel and each calls
``SomeModel.from_pretrained(model_id, subfolder="text_encoder", ...)`` with a
cold HuggingFace cache, transformers v5's cache-resolution (``cached_files``)
can observe a partially-written shard set written by a peer worker and raise
``OSError: <model_id> does not appear to have a file named
text_encoder/model-00002-of-00002.safetensors`` even though the peer will
eventually finish writing it.

Why ``origin/main`` does not need this helper
---------------------------------------------
The exact same ``__init__`` code lives on ``origin/main`` (e.g. the Qwen-Image
``pipeline_qwen_image_edit_plus.py`` in build vllm-omni#7412 passes without
any prefetch), so the race is NOT a behavioural change in vLLM-Omni itself.
Two environmental factors mask the race on main:

* ``origin/main`` is pinned (transitively, via vLLM main) to
  ``transformers`` 4.x. In 4.x the per-file ``cached_file`` path resolves
  shards **lazily**, one at a time, so each ``hf_hub_download`` blocks on its
  own single-file ``.lock`` and the second worker naturally waits for the
  first worker's atomic rename. ``transformers>=5.0`` rewrote this into
  ``cached_files`` (plural) which batch-resolves every shard listed in the
  index up-front via ``os.path.isfile`` and raises immediately if any shard
  is still sitting under its ``*.incomplete`` name. Same wave of v5 changes
  that introduced ``tie_weights(missing_keys=..., recompute_mapping=...)``
  (see the Dynin shim in ``dynin_omni_token2text.py``).
* CI shares ``HF_HOME=/fsx/hf_cache`` across pipelines (both the
  ``vllm-omni`` and ``vllm-omni-rebase`` pipelines mount the same FS). That
  cache is normally warm for long-lived repos like ``Qwen-Image-Edit-2509``,
  so most builds never go through the download path at all. Build 1043
  happened to hit a partially-evicted cache AND transformers v5's stricter
  resolver simultaneously, which is why the failure looks 'rebase-specific'
  but is really a latent race that main was absorbing via (1).

``huggingface_hub.snapshot_download`` does take per-blob ``.lock`` files,
but those locks are only acquired once each blob is mid-write - they do not
cover the surrounding ``cached_files`` shard-list resolution that
transformers v5 performs eagerly. So we additionally wrap the
``snapshot_download`` call in our own node-wide ``fcntl.flock`` keyed on the
repo id (see ``_repo_prefetch_lock``). The first concurrent worker / process
to reach the helper fully materialises the snapshot; subsequent entrants
block on the flock and then find a warm cache, so their ``from_pretrained``
calls never observe a half-written shard set. For a warm cache the snapshot
call is a near-noop (it only stat()s the files), so this is also cheap on
``origin/main`` should we ever backport it there.

The helper is intentionally best-effort: prefetch failures (offline, gated
repos, transient 5xx, missing flock support) are logged and swallowed so the
subsequent ``from_pretrained`` call can surface the real, specific error to
the user rather than being masked here.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)


def _node_lock_dir() -> str:
    """Return a node-local directory suitable for the prefetch lock files.

    We co-locate the lock file with the HF cache it is serialising
    (``$HF_HOME/locks/vllm-omni-prefetch/``) so they share a filesystem -
    ``fcntl.flock`` semantics across NFS / Lustre are only well-defined
    when the underlying FS supports POSIX advisory locking. CI's
    ``/fsx/hf_cache`` (FSx for Lustre) does. If ``HF_HOME`` is unwritable
    (read-only baked image, etc.) we fall back to ``XDG_CACHE_HOME``,
    ``~/.cache/huggingface``, then ``/tmp`` so unit tests outside CI keep
    working.
    """
    candidates: list[str] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "locks", "vllm-omni-prefetch"))
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(os.path.join(xdg_cache, "huggingface", "locks", "vllm-omni-prefetch"))
    candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "locks", "vllm-omni-prefetch"))
    candidates.append(os.path.join("/tmp", "vllm-omni-prefetch-locks"))

    for cand in candidates:
        try:
            os.makedirs(cand, exist_ok=True)
            probe = os.path.join(cand, ".write_check")
            with open(probe, "a"):
                pass
            with contextlib.suppress(OSError):
                os.remove(probe)
            return cand
        except OSError:
            continue
    fallback = os.path.join("/tmp", "vllm-omni-prefetch-locks")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _safe_repo_filename(model: str) -> str:
    """Translate an HF repo id into a filesystem-safe lock filename."""
    return model.replace("/", "__").replace(os.sep, "__") + ".lock"


@contextlib.contextmanager
def _repo_prefetch_lock(model: str) -> Iterator[None]:
    """Hold an exclusive ``fcntl.flock`` keyed on the HF repo id.

    Why we still need a node-wide lock on top of ``snapshot_download``'s
    own per-blob ``.lock`` files:

    * ``huggingface_hub.snapshot_download`` does serialise concurrent
      writers within the same call, but its lock lives in the per-blob
      cache directory and is only acquired once each blob is being
      written. With ``transformers`` v5 the multi-shard ``cached_files``
      resolver inspecting the cache between blob acquires is exactly
      the window that races - a peer worker's ``model-00001-of-00002.
      safetensors.incomplete`` makes ``cached_files`` raise ``OSError:
      <repo> does not appear to have a file named ...`` even though
      the peer is mid-download (Buildkite #8418).
    * Multiple unrelated processes on the same node (concurrent CI jobs,
      multiple ``OmniServer`` instances, every ``DiffusionWorker``
      subprocess from the multiproc executor) can enter this code path
      in parallel. ``flock`` on a per-repo file makes the whole
      ``snapshot_download(...)`` call a true node-wide critical
      section, so the *first* entrant fully populates the cache and
      every subsequent ``from_pretrained`` sees a warm, complete tree.

    Best-effort: if we cannot create / acquire the lock (no ``fcntl``,
    read-only FS, NFS without flock support) we log and run the download
    anyway. Worst case behaviour reverts to "snapshot_download +
    transformers v5 race" which is exactly the pre-fix state.
    """
    try:
        import fcntl  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - non-POSIX (Windows)
        logger.debug("fcntl unavailable on this platform; running prefetch unlocked")
        yield
        return

    try:
        lock_dir = _node_lock_dir()
    except OSError as exc:
        logger.warning("Could not allocate lock dir for prefetch of %s (%s); skipping flock", model, exc)
        yield
        return

    lock_path = os.path.join(lock_dir, _safe_repo_filename(model))

    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        logger.warning("Could not open prefetch lock %s (%s); proceeding without flock", lock_path, exc)
        yield
        return

    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            logger.debug("Acquired prefetch flock for %s at %s", model, lock_path)
        except OSError as exc:
            # FS without flock support (ENOLCK / EOPNOTSUPP / EINVAL).
            # EACCES on some Lustre configs without ``flock`` mount opt.
            if exc.errno in (errno.ENOLCK, errno.EOPNOTSUPP, errno.EACCES, errno.EINVAL):
                logger.warning(
                    "fcntl.flock not supported on '%s' (%s: %s); running prefetch unlocked",
                    lock_path,
                    type(exc).__name__,
                    exc,
                )
            else:
                raise
        yield
    finally:
        if locked:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)


def prefetch_subfolders(
    model: str,
    subfolders: Iterable[str],
    *,
    local_files_only: bool = False,
    include_root_metadata: bool = True,
) -> None:
    """Materialise ``model``'s ``subfolders`` in the HF cache before loading.

    Args:
        model: A HuggingFace Hub repo id (e.g. ``"Qwen/Qwen-Image-Edit-2509"``)
            or a local directory path. Local paths are a no-op.
        subfolders: Iterable of subfolder names (e.g. ``["text_encoder",
            "vae"]``) whose contents need to be fully present before any
            worker calls ``from_pretrained(subfolder=...)``.
        local_files_only: When True, skip the prefetch entirely. The caller
            has explicitly promised the cache is already populated (as happens
            for local model checkouts), so hitting the network would defeat
            the intent and may fail in air-gapped environments.
        include_root_metadata: When True, also pull ``*.json`` at the repo
            root so ``model_index.json`` / ``config.json`` resolution during
            ``from_pretrained`` also hits a warm cache.
    """
    if local_files_only or not model or os.path.isdir(model):
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        logger.debug("huggingface_hub unavailable; skipping prefetch of %s", model)
        return

    allow_patterns: list[str] = []
    for sub in subfolders:
        sub = (sub or "").strip("/")
        if not sub:
            continue
        # hf_hub globbing is shell-style: `text_encoder/*` catches the index +
        # any flat files, `text_encoder/**` catches nested safetensors shards
        # that some repos place under e.g. ``text_encoder/pytorch_model/``.
        allow_patterns.extend([f"{sub}/*", f"{sub}/**"])

    if include_root_metadata:
        allow_patterns.extend(["*.json", "*.txt"])

    if not allow_patterns:
        return

    # Wrap the download in a node-wide ``fcntl`` flock keyed on the repo
    # id. The first concurrent entrant fully populates the cache; every
    # subsequent worker / process blocks here, then no-ops because the
    # snapshot is now warm. This is what makes the prefetch race-free
    # even when many ``DiffusionWorker`` subprocesses (or multiple
    # OmniServer instances on the same node) hit this code in parallel.
    try:
        with _repo_prefetch_lock(model):
            snapshot_download(
                repo_id=model,
                allow_patterns=allow_patterns,
            )
    except Exception as exc:
        # Best-effort: propagate only via logging. The subsequent
        # ``from_pretrained`` call will raise a clearer, call-site-specific
        # error (auth, 404, disk full, ...) that we'd rather surface - EXCEPT
        # for auth/gating, which we escalate here with an explicit hint so
        # readers of CI logs don't have to correlate the generic "OSError:
        # <repo> does not appear to have a file named ..." that
        # ``from_pretrained`` would otherwise emit much later with an
        # unrelated-looking message.
        if _looks_like_auth_error(exc):
            logger.error(
                "Hub prefetch for '%s' failed with an authentication / gated "
                "repository error (%s: %s). The CI HF_TOKEN must (1) be set "
                "in the step env, (2) be valid, and (3) belong to an account "
                "that has accepted the model license on huggingface.co. See "
                "docs/contributing/ci/hf_credentials.md.",
                model,
                type(exc).__name__,
                exc,
            )
        else:
            logger.warning(
                "Hub prefetch for repo '%s' subfolders %s failed (%s: %s); "
                "falling back to on-demand download in from_pretrained",
                model,
                list(subfolders),
                type(exc).__name__,
                exc,
            )


def _looks_like_auth_error(exc: BaseException) -> bool:
    """Classify prefetch exceptions as auth/gating failures.

    ``huggingface_hub`` raises bespoke ``GatedRepoError`` /
    ``RepositoryNotFoundError`` subclasses when the token is missing or
    lacks license acceptance; but older releases (and third-party
    transport layers) sometimes only surface this as a generic
    ``HfHubHTTPError`` / ``requests.HTTPError`` with status 401/403. We
    check both code paths so the branch above is stable across
    ``huggingface_hub`` versions.
    """
    try:
        from huggingface_hub.errors import (  # type: ignore[import-not-found]
            GatedRepoError,
            RepositoryNotFoundError,
        )

        if isinstance(exc, GatedRepoError | RepositoryNotFoundError):
            return True
    except ImportError:  # pragma: no cover - very old huggingface_hub
        pass

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    # Last-resort string heuristic - ``snapshot_download`` on some
    # transports wraps the 401 as a plain OSError whose message is the
    # only load-bearing signal.
    msg = str(exc).lower()
    return "401 client error" in msg or "403 client error" in msg or "gatedrepo" in msg
