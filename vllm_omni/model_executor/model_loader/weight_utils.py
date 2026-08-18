import contextlib
import errno
import os
import time
from collections.abc import Iterator
from pathlib import Path

import huggingface_hub
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import DisabledTqdm, get_lock

if envs.VLLM_USE_MODELSCOPE:
    from modelscope.hub.snapshot_download import snapshot_download
else:
    from huggingface_hub import snapshot_download

logger = init_logger(__name__)

_DOWNLOAD_MAX_ATTEMPTS = 3
_DOWNLOAD_BACKOFF_BASE_S = 1.0
_FULL_SNAPSHOT_METADATA_MARKERS = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "model_index.json",
)


def _node_lock_dir() -> str:
    candidates: list[str] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "locks", "vllm-omni-weight-prefetch"))
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(os.path.join(xdg_cache, "huggingface", "locks", "vllm-omni-weight-prefetch"))
    candidates.append(
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "locks", "vllm-omni-weight-prefetch")
    )
    candidates.append(os.path.join("/tmp", "vllm-omni-weight-prefetch-locks"))

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
    fallback = os.path.join("/tmp", "vllm-omni-weight-prefetch-locks")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def _safe_repo_filename(model: str) -> str:
    return model.replace("/", "__").replace(os.sep, "__") + ".lock"


def _dotfile_lock_acquire(lock_dir: str, model: str, timeout: float = 300.0, poll_interval: float = 0.5) -> str | None:
    lock_path = os.path.join(lock_dir, _safe_repo_filename(model) + ".dir")
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.makedirs(lock_path, exist_ok=False)
            logger.info("Acquired dotfile weight prefetch lock for %s at %s", model, lock_path)
            return lock_path
        except FileExistsError:
            if time.monotonic() >= deadline:
                logger.warning(
                    "Timed out waiting for dotfile weight prefetch lock %s after %.0fs; proceeding unlocked",
                    lock_path,
                    timeout,
                )
                return None
            time.sleep(poll_interval)


@contextlib.contextmanager
def _repo_download_lock(model: str) -> Iterator[None]:
    """Serialize full-repo snapshot materialization across Omni processes."""
    lock_dir = None
    dotfile_held = None
    fd = None
    flock_held = False

    try:
        import fcntl  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - non-POSIX
        fcntl = None

    if fcntl is not None:
        try:
            lock_dir = _node_lock_dir()
            lock_path = os.path.join(lock_dir, _safe_repo_filename(model))
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            flock_held = True
            logger.info("Acquired flock weight prefetch lock for %s at %s", model, lock_path)
        except OSError as exc:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
                fd = None
            if exc.errno not in (errno.ENOLCK, errno.EOPNOTSUPP, errno.EACCES, errno.EINVAL):
                raise
            logger.warning("fcntl.flock unavailable for weight prefetch of %s (%s); using dotfile lock", model, exc)

    if not flock_held:
        try:
            lock_dir = lock_dir or _node_lock_dir()
            dotfile_held = _dotfile_lock_acquire(lock_dir, model)
        except OSError as exc:
            logger.warning("Could not allocate weight prefetch lock dir for %s (%s); proceeding unlocked", model, exc)

    try:
        yield
    finally:
        if flock_held and fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if dotfile_held is not None:
            with contextlib.suppress(OSError):
                os.rmdir(dotfile_held)


def _looks_like_auth_error(exc: BaseException) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError  # type: ignore[import-not-found]

        if isinstance(exc, GatedRepoError | RepositoryNotFoundError):
            return True
    except ImportError:  # pragma: no cover - very old huggingface_hub
        pass

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return "401 client error" in msg or "403 client error" in msg or "gatedrepo" in msg


def _is_full_snapshot_request(allow_patterns: list[str]) -> bool:
    return any(pattern.strip() in {"*", "**", "./*", "**/*"} for pattern in allow_patterns)


def _verify_snapshot_materialized(hf_folder: str, allow_patterns: list[str]) -> None:
    root = Path(hf_folder)
    if not root.exists():
        raise OSError(f"Downloaded snapshot folder does not exist: {hf_folder}")

    if not any(root.glob(pattern) for pattern in allow_patterns):
        raise OSError(f"Downloaded snapshot {hf_folder} has no files matching {allow_patterns}")

    # Full-repo downloads are used before transformers loads tokenizers,
    # processors, and feature extractors. A half-materialized shared cache can
    # contain weights while missing these small metadata files, which later
    # surfaces as "Can't load feature extractor ... preprocessor_config.json".
    if _is_full_snapshot_request(allow_patterns) and not any(
        (root / marker).exists() for marker in _FULL_SNAPSHOT_METADATA_MARKERS
    ):
        raise OSError(
            f"Downloaded full snapshot {hf_folder} is missing expected metadata files {_FULL_SNAPSHOT_METADATA_MARKERS}"
        )


def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    require_all: bool = False,
) -> str:
    """Download model weights from Hugging Face Hub. Users can specify the
    allow_patterns to download only the necessary weights.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.
        require_all (bool): If True, will iterate through and download files
            matching all patterns in allow_patterns. If False, will stop after
            the first pattern that matches any files.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    allow_patterns = list(allow_patterns)
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    download_kwargs = {"tqdm_class": DisabledTqdm} if not envs.VLLM_USE_MODELSCOPE else {}

    logger.info("Using model weights format %s", allow_patterns)
    start_time = time.perf_counter()
    last_exc: BaseException | None = None

    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            # Use both vLLM's cache lock and a repo-wide node lock. The latter
            # protects the transformers v5 eager metadata resolver from seeing
            # a peer process' half-written shared HF cache.
            with get_lock(model_name_or_path, cache_dir), _repo_download_lock(model_name_or_path):
                if require_all:
                    hf_folder = snapshot_download(
                        model_name_or_path,
                        allow_patterns=allow_patterns,
                        ignore_patterns=ignore_patterns,
                        cache_dir=cache_dir,
                        revision=revision,
                        local_files_only=local_only,
                        **download_kwargs,
                    )
                    _verify_snapshot_materialized(hf_folder, allow_patterns)
                    break

                hf_folder = None
                for allow_pattern in allow_patterns:
                    hf_folder = snapshot_download(
                        model_name_or_path,
                        allow_patterns=allow_pattern,
                        ignore_patterns=ignore_patterns,
                        cache_dir=cache_dir,
                        revision=revision,
                        local_files_only=local_only,
                        **download_kwargs,
                    )
                    _verify_snapshot_materialized(hf_folder, [allow_pattern])
                    # If we have downloaded weights for this allow_pattern,
                    # we don't need to check the rest, unless require_all is set.
                    if any(Path(hf_folder).glob(allow_pattern)):
                        break
                if hf_folder is None:
                    raise OSError(f"No snapshot downloaded for patterns {allow_patterns}")
                break
        except Exception as exc:
            last_exc = exc
            if _looks_like_auth_error(exc) or attempt == _DOWNLOAD_MAX_ATTEMPTS or local_only:
                raise
            backoff = _DOWNLOAD_BACKOFF_BASE_S * attempt
            logger.warning(
                "Downloading weights for %s with patterns %s failed on attempt %d/%d (%s: %s); retrying in %.1fs",
                model_name_or_path,
                allow_patterns,
                attempt,
                _DOWNLOAD_MAX_ATTEMPTS,
                type(exc).__name__,
                exc,
                backoff,
            )
            time.sleep(backoff)

    if last_exc is not None and "hf_folder" not in locals():
        raise last_exc

    time_taken = time.perf_counter() - start_time
    if time_taken > 0.5:
        logger.info(
            "Time spent downloading weights for %s: %.6f seconds",
            model_name_or_path,
            time_taken,
        )
    return hf_folder
