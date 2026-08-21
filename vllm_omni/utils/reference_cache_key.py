from __future__ import annotations

import base64
import binascii
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote_to_bytes, urlparse

import numpy as np
import torch
import xxhash
from PIL import Image


def _is_url_like(s: str) -> bool:
    """Quick check if a string is a URL (http, https, data, file)."""
    parsed = urlparse(s)
    return bool(parsed.scheme and parsed.scheme in ("http", "https", "data", "file"))


def _hash_joined(parts: list[str]) -> str:
    return xxhash.xxh3_64("|".join(parts).encode("utf-8")).hexdigest()


def hash_bytes(payload: bytes | bytearray | memoryview) -> str:
    return xxhash.xxh3_64(payload).hexdigest()


def hash_waveform(wav: torch.Tensor, sample_rate: int) -> str:
    """Generate a content key for a decoded waveform plus sample-rate metadata."""
    cpu = wav.detach().to(device="cpu", dtype=torch.float32).contiguous()
    meta = f"sr:{int(sample_rate)}|shape:{tuple(cpu.shape)}"
    return f"wav:{meta}:{hash_bytes(cpu.numpy().tobytes())}"


def hash_file_sampled(
    path: str | Path,
    head_size: int = 8192,
    tail_size: int = 8192,
) -> str:
    """Generate hash from file head + tail + size (fast sampling strategy).

    This avoids reading the entire file while still detecting most changes.
    For compressed formats (JPEG, PNG, WAV, MP4), any content change typically
    affects file size and/or head/tail bytes.
    """
    path = Path(path)
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        head = f.read(head_size)
        if file_size > head_size + tail_size:
            f.seek(-tail_size, 2)  # Seek from end
            tail = f.read(tail_size)
        else:
            tail = b""  # Small file, head already covers it

    payload = head + tail + f"|size:{file_size}".encode()
    return xxhash.xxh3_64(payload).hexdigest()


_REF_PATH_HASH_MEMO_MAX_ITEMS = 1024
_REF_PATH_HASH_SENTINEL_BYTES = 8192
_REF_PATH_HASH_MEMO: OrderedDict[str, tuple[str, str]] = OrderedDict()
_REF_PATH_HASH_MEMO_LOCK = threading.Lock()


def _reference_path_hash_memo_key(path: Path) -> tuple[str, int] | None:
    try:
        if not path.is_file():
            return None
        stat_result = path.stat()
        memo_key = (
            f"{path.resolve()}:"
            f"{stat_result.st_size}:"
            f"{stat_result.st_mtime_ns}:"
            f"{stat_result.st_ctime_ns}"
        )
        return memo_key, int(stat_result.st_size)
    except OSError:
        return None


def _reference_path_sentinel(path: Path, file_size: int) -> str | None:
    try:
        chunk_size = min(_REF_PATH_HASH_SENTINEL_BYTES, file_size)
        with path.open("rb") as f:
            chunks = [f.read(chunk_size)]
            if file_size > _REF_PATH_HASH_SENTINEL_BYTES:
                middle_offset = max((file_size - chunk_size) // 2, 0)
                f.seek(middle_offset)
                chunks.append(f.read(chunk_size))
            if file_size > 2 * _REF_PATH_HASH_SENTINEL_BYTES:
                f.seek(max(file_size - chunk_size, 0))
                chunks.append(f.read(chunk_size))
        return hash_bytes(b"".join(chunks) + f"|size:{file_size}".encode())
    except OSError:
        return None


def _get_reference_path_hash(memo_key: str, sentinel: str) -> str | None:
    with _REF_PATH_HASH_MEMO_LOCK:
        cached = _REF_PATH_HASH_MEMO.get(memo_key)
        if cached is None:
            return None
        cached_sentinel, digest = cached
        if cached_sentinel != sentinel:
            _REF_PATH_HASH_MEMO.pop(memo_key, None)
            return None
        _REF_PATH_HASH_MEMO.move_to_end(memo_key)
        return digest


def _get_reference_path_hash_by_memo_key(memo_key: str) -> str | None:
    # Memo lookup ignoring the sentinel; trust_stat=True callers only.
    with _REF_PATH_HASH_MEMO_LOCK:
        cached = _REF_PATH_HASH_MEMO.get(memo_key)
        if cached is None:
            return None
        _REF_PATH_HASH_MEMO.move_to_end(memo_key)
        return cached[1]


def _put_reference_path_hash(memo_key: str, sentinel: str, digest: str) -> None:
    with _REF_PATH_HASH_MEMO_LOCK:
        _REF_PATH_HASH_MEMO[memo_key] = (sentinel, digest)
        _REF_PATH_HASH_MEMO.move_to_end(memo_key)
        while len(_REF_PATH_HASH_MEMO) > _REF_PATH_HASH_MEMO_MAX_ITEMS:
            _REF_PATH_HASH_MEMO.popitem(last=False)


def reference_path_cache_key(
    path_like: str | Path, *, trust_stat: bool = False
) -> str | None:
    path = Path(str(path_like)).expanduser()
    memo = _reference_path_hash_memo_key(path)
    if memo is None:
        return None
    memo_key, file_size = memo

    if trust_stat:
        digest = _get_reference_path_hash_by_memo_key(memo_key)
        if digest is not None:
            return f"file:{digest}"

    sentinel = _reference_path_sentinel(path, file_size)
    if sentinel is None:
        return None

    if not trust_stat:
        digest = _get_reference_path_hash(memo_key, sentinel)
        if digest is not None:
            return f"file:{digest}"

    try:
        digest = hash_bytes(path.read_bytes())
    except OSError:
        return None
    if _reference_path_hash_memo_key(path) == memo:
        # Always store the sentinel so default callers still validate this entry.
        _put_reference_path_hash(memo_key, sentinel, digest)
    return f"file:{digest}"


def data_uri_content_key(uri: str) -> str | None:
    """Compute a content-addressed cache key for a ``data:`` URI.

    The payload is decoded (base64 or percent-encoded) and hashed with
    ``xxh3_64`` so that two ``data:`` URIs pointing at the same bytes share
    the same key even if the media type or attribute ordering differs.

    Returns ``None`` when ``uri`` is not a well-formed ``data:`` URI or when
    the payload cannot be decoded. Callers should fall back to a locator
    hash in that case.
    """
    if not isinstance(uri, str) or not uri.startswith("data:"):
        return None
    try:
        header, _, payload = uri[len("data:"):].partition(",")
    except ValueError:
        return None
    if not payload:
        return None
    is_base64 = any(part.strip().lower() == "base64" for part in header.split(";"))
    try:
        raw = base64.b64decode(payload, validate=False) if is_base64 else unquote_to_bytes(payload)
    except (binascii.Error, ValueError):
        return None
    if not raw:
        return None
    return f"bytes:{hash_bytes(raw)}"


def locator_content_key(locator: str) -> str:
    """Return a content-addressed cache key for a reference-audio *locator*.

    The key is computed straight from the locator string **without decoding
    the audio**, so a cache lookup can happen before the (expensive) resolve +
    codec-encode work. Strategy (first non-empty result wins):

    * ``data:`` URIs are keyed by the xxh3 of their decoded payload, so two
      requests carrying identical inline audio share the key even if the media
      type or attribute ordering differs.
    * Local paths and ``file://`` locators are keyed by
      :func:`reference_path_cache_key` (stat memo + head/mid/tail sentinel +
      full-content xxh3, ``trust_stat=False``), which detects overwrites that
      reuse the same path.
    * Everything else (http(s) URLs, opaque strings) falls back to a hash of
      the raw locator. This is best-effort and assumes the locator's contents
      are stable for the server lifetime.

    Unlike hashing the decoded waveform, every branch here is cheap on a cache
    hit (a few KiB of file reads at most), which is what lets the reference
    encoder skip resolve/encode entirely for a repeated speaker.
    """
    if not isinstance(locator, str):
        locator = str(locator)

    if locator.startswith("data:"):
        content_key = data_uri_content_key(locator)
        if content_key is not None:
            return f"src:{content_key}"

    candidate_path: str | None = None
    if locator.startswith("file://"):
        candidate_path = locator[len("file://"):]
    elif "://" not in locator:
        candidate_path = locator
    if candidate_path:
        file_key = reference_path_cache_key(candidate_path)
        if file_key is not None:
            return f"src:{file_key}"

    return f"str:{hash_bytes(locator.encode('utf-8'))}"


def hash_media_item(item: Any) -> str | None:
    """Generate hash for a single media item (unified logic for image/audio/video).

    Supported types:
    - str/Path: local file -> sampled hash; URL -> string hash
    - PIL.Image: mode + size + content hash
    - numpy.ndarray: dtype + shape + content hash
    - torch.Tensor: dtype + shape + content hash
    - bytes/bytearray: content hash

    Returns None for unsupported types (caller should skip caching).
    """
    # File path or URL
    if isinstance(item, (str, Path)):
        s = str(item)
        if _is_url_like(s):
            return f"url:{hash_bytes(s.encode())}"
        p = Path(s)
        if p.exists() and p.is_file():
            return f"file:{hash_file_sampled(p)}"
        return f"url:{hash_bytes(s.encode())}"

    # PIL Image
    if isinstance(item, Image.Image):
        meta = f"{item.mode}|{item.size}"
        content_hash = hash_bytes(item.tobytes())
        return f"pil:{meta}:{content_hash}"

    # numpy array
    if isinstance(item, np.ndarray):
        meta = f"{item.dtype}|{item.shape}"
        content_hash = hash_bytes(item.tobytes())
        return f"np:{meta}:{content_hash}"

    # torch Tensor
    if isinstance(item, torch.Tensor):
        cpu = item.detach().cpu()
        meta = f"{cpu.dtype}|{tuple(cpu.shape)}"
        content_hash = hash_bytes(cpu.numpy().tobytes())
        return f"pt:{meta}:{content_hash}"

    # Raw bytes
    if isinstance(item, (bytes, bytearray, memoryview)):
        return f"bytes:{hash_bytes(item)}"

    # Unsupported type
    return None


def compute_media_cache_key(items: Any, *, prefix: str) -> str | None:
    """Compute cache key for media items (image/audio/video).

    Args:
        items: Single item or list of items
        prefix: Type prefix (e.g., "image", "audio", "video")

    Returns:
        Cache key string or None if any item is unsupported.
    """
    if items is None:
        return None
    seq = items if isinstance(items, list) else [items]
    if not seq:
        return None

    parts: list[str] = []
    for item in seq:
        part = hash_media_item(item)
        if part is None:
            return None
        parts.append(part)

    return f"{prefix}:{_hash_joined(parts)}"


def compute_cache_key(
    items: Any, *, item_to_part: Callable[[Any], str | None]
) -> str | None:
    """Compute cache key from a list-like input.

    The item_to_part callback must return a string part or None to
    indicate the item type is unsupported (no cache key).

    Note: Prefer compute_media_cache_key() for new code.
    """
    if items is None:
        return None
    seq = items if isinstance(items, list) else [items]
    if not seq:
        return None

    parts: list[str] = []
    for item in seq:
        part = item_to_part(item)
        if part is None:
            return None
        parts.append(part)

    return _hash_joined(parts)
