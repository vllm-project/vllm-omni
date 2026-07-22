from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---- Module constants ----
_MB = 1024**2
_GB = 1024**3
_SIM_STATS_PATH = "/tmp/cache_sim_stats.json"
_SIM_STATS_FLUSH_INTERVAL = 50  # flush similarity stats to file every N searches


@dataclass(frozen=True)
class CacheKey:
    prompt: str
    negative_prompt: str
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    true_cfg_scale: float
    seed: int
    sigmas: tuple[float, ...] | None
    max_sequence_length: int | None
    num_images_per_prompt: int
    num_frames: int = 1

    def to_hash(self) -> str:
        data = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "true_cfg_scale": self.true_cfg_scale,
            "seed": self.seed,
            "sigmas": list(self.sigmas) if self.sigmas is not None else None,
            "max_sequence_length": self.max_sequence_length,
            "num_images_per_prompt": self.num_images_per_prompt,
            "num_frames": self.num_frames,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass
class StepLatentData:
    step_index: int
    timestep: float
    latent: torch.Tensor


@dataclass
class CacheEntry:
    latents: torch.Tensor | None
    cache_key_hash: str
    step_latents: list[StepLatentData] | None = None
    metadata: dict[str, Any] | None = None
    clip_embedding: torch.Tensor | None = None
    cache_key: CacheKey | None = None
    image_embedding: torch.Tensor | None = None


class DiTCacheStore:
    def __init__(
        self,
        max_entries: int = 100,
        max_memory_gb: float = 4.0,
        lmcache_engine: Any | None = None,
    ):
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_gb * _GB
        self._current_memory_bytes = 0
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._all_sims: list[float] = []  # all final_sim values across all queries
        self._all_t2t_sims: list[float] = []  # all t2t_sim values across all queries
        self._search_count: int = 0  # throttle for _flush_sim_stats_to_file
        self._use_t2i_penalty: bool = True  # enable t2i sigmoid penalty in hybrid matching

        # ---- LMCache-backed tiered storage ----
        # When lmcache_engine is set, put() also writes latents to LMCache (which
        # manages CPU→Disk tiering + LRU internally). On CPU eviction the entry's
        # latents are set to None (lightweight shell kept for semantic_search);
        # a subsequent get() recovers them from LMCache via engine.get().
        self._lmcache = lmcache_engine

        # ---- Pre-stacked embedding matrices for fast vectorized retrieval ----
        # Instead of torch.stack()-ing all embeddings on every semantic_search()
        # call (O(N) copy each time), we maintain two growable matrices that are
        # updated incrementally on put()/update/evict.  Rows are lazily freed:
        # evicted rows are marked invalid and skipped during search, avoiding
        # costly matrix rebuilds on every eviction.
        self._emb_dim: int | None = None  # embedding dim (set on first put)
        self._txt_matrix: torch.Tensor | None = None  # [rows, dim] text embeddings
        self._img_matrix: torch.Tensor | None = None  # [rows, dim] image embeddings (None row = absent)
        self._row_keys: list[str | None] = []  # row_idx -> key_hash (None = freed row)
        self._key_rows: dict[str, int] = {}  # key_hash -> row_idx
        self._next_row: int = 0  # next free row to append to
        self._capacity: int = 0  # allocated row capacity (grows in chunks)

    def set_t2i_penalty(self, enabled: bool) -> None:
        """Enable or disable t2i sigmoid penalty in hybrid matching."""
        self._use_t2i_penalty = enabled
        logger.info("t2i penalty %s", "enabled" if enabled else "disabled")

    # ---- Matrix maintenance helpers ----
    _ROW_CHUNK = 512  # rows to allocate per growth event

    def _ensure_capacity(self, extra: int = 1) -> None:
        """Grow the embedding matrices if needed to hold at least _next_row+extra rows."""
        needed = self._next_row + extra
        if needed <= self._capacity:
            return
        new_cap = max(needed, self._capacity + self._ROW_CHUNK)
        if self._txt_matrix is None:
            # first allocation
            self._txt_matrix = torch.zeros(new_cap, self._emb_dim)
            self._img_matrix = torch.zeros(new_cap, self._emb_dim)
        else:
            pad = new_cap - self._capacity
            self._txt_matrix = torch.cat([self._txt_matrix, torch.zeros(pad, self._emb_dim)], dim=0)
            self._img_matrix = torch.cat([self._img_matrix, torch.zeros(pad, self._emb_dim)], dim=0)
            # extend row_keys list
            self._row_keys.extend([None] * pad)
        self._capacity = new_cap

    def _matrix_add(self, key_hash: str, clip_embedding: torch.Tensor | None) -> None:
        """Add/overwrite a row in the text embedding matrix for key_hash."""
        if clip_embedding is None:
            self._key_rows.pop(key_hash, None)
            return
        emb = clip_embedding.detach().cpu()
        if emb.dim() == 2:
            emb = emb.squeeze(0)
        if self._emb_dim is None:
            self._emb_dim = emb.shape[0]
        elif emb.shape[0] != self._emb_dim:
            logger.warning("Embedding dim mismatch: got %d, expected %d, skipping", emb.shape[0], self._emb_dim)
            return
        self._ensure_capacity(1)
        row = self._next_row
        self._txt_matrix[row] = emb
        self._row_keys.append(key_hash)
        self._key_rows[key_hash] = row
        self._next_row += 1

    def _matrix_update_image(self, key_hash: str, image_embedding: torch.Tensor) -> bool:
        """Set the image embedding for an existing row. Returns False if key absent."""
        row = self._key_rows.get(key_hash)
        if row is None:
            return False
        emb = image_embedding.detach().cpu()
        if emb.dim() == 2:
            emb = emb.squeeze(0)
        self._img_matrix[row] = emb
        return True

    def _matrix_free(self, key_hash: str) -> None:
        """Lazily free a row (mark invalid); no matrix rebuild."""
        row = self._key_rows.pop(key_hash, None)
        if row is not None:
            self._row_keys[row] = None  # mark as freed

    def _estimate_tensor_bytes(self, tensor: torch.Tensor) -> int:
        return tensor.nelement() * tensor.element_size()

    def _estimate_step_latents_bytes(self, step_latents: list[StepLatentData]) -> int:
        return sum(self._estimate_tensor_bytes(s.latent) for s in step_latents)

    def _estimate_entry_bytes(self, entry: CacheEntry) -> int:
        total = 0
        if entry.latents is not None:
            total += self._estimate_tensor_bytes(entry.latents)
        if entry.step_latents is not None:
            total += self._estimate_step_latents_bytes(entry.step_latents)
        if entry.clip_embedding is not None:
            total += self._estimate_tensor_bytes(entry.clip_embedding)
        if entry.image_embedding is not None:
            total += self._estimate_tensor_bytes(entry.image_embedding)
        return total

    def _estimate_heavy_bytes(self, entry: CacheEntry) -> int:
        """Bytes of heavy tensors only (latents + step_latents), excluding embeddings."""
        total = 0
        if entry.latents is not None:
            total += self._estimate_tensor_bytes(entry.latents)
        if entry.step_latents is not None:
            total += self._estimate_step_latents_bytes(entry.step_latents)
        return total

    def _evict_if_needed(self, required_bytes: int):
        """Evict the oldest entries to make room.

        When LMCache is enabled, eviction clears the entry's heavy tensors
        (latents, step_latents) from CPU memory but keeps a lightweight shell
        (clip_embedding, cache_key) for semantic_search. The data was already
        written to LMCache at put() time, so a subsequent get() can recover it.
        Without LMCache, the entry is discarded entirely (original behaviour).
        """
        while len(self._store) >= self._max_entries or (
            self._current_memory_bytes + required_bytes > self._max_memory_bytes
            and len(self._store) > 0
        ):
            # Find the oldest entry that still has heavy data in CPU.
            oldest_key = None
            oldest_entry = None
            for kh, entry in self._store.items():
                if entry.latents is not None:
                    oldest_key = kh
                    oldest_entry = entry
                    break
            if oldest_key is None:
                # All entries are shells (latents=None). If still over budget,
                # purge the oldest shell entirely (frees its embedding bytes).
                # This handles the "max_memory_gb < total embedding bytes" edge case.
                if len(self._store) > 0:
                    purge_key = next(iter(self._store))
                    purged = self._store.pop(purge_key)
                    self._current_memory_bytes -= self._estimate_entry_bytes(purged)
                    self._matrix_free(purge_key)
                    continue
                break

            # Evict the heavy data. Use _estimate_entry_bytes so the freed
            # amount matches what was counted at put() time (including embeddings
            # that will remain in the shell — but we subtract the full amount and
            # re-add just the shell's embedding bytes below for LMCache path).
            full_bytes = self._estimate_entry_bytes(oldest_entry)

            if self._lmcache is not None:
                # Keep shell; clear heavy data (already safe in LMCache).
                heavy_bytes = self._estimate_heavy_bytes(oldest_entry)
                oldest_entry.latents = None
                oldest_entry.step_latents = None
                # Subtract only the heavy portion; shell's embedding bytes stay counted.
                self._current_memory_bytes -= heavy_bytes
                logger.debug(
                    "Evicted-to-lmcache %s, freed %.2f MB (shell kept)",
                    oldest_key[:8],
                    heavy_bytes / _MB,
                )
            else:
                # No LMCache: discard entirely.
                self._store.pop(oldest_key)
                self._matrix_free(oldest_key)
                self._current_memory_bytes -= full_bytes
                logger.debug(
                    "Evicted %s, freed %.2f MB",
                    oldest_key[:8],
                    full_bytes / _MB,
                )

    def put(
        self,
        key: CacheKey,
        latents: torch.Tensor,
        step_latents: list[StepLatentData] | None = None,
        metadata: dict[str, Any] | None = None,
        clip_embedding: torch.Tensor | None = None,
    ):
        key_hash = key.to_hash()
        # Full entry bytes including embeddings, so _current_memory_bytes
        # stays consistent with _estimate_entry_bytes used in evict.
        tensor_bytes = self._estimate_tensor_bytes(latents)
        if step_latents is not None:
            tensor_bytes += self._estimate_step_latents_bytes(step_latents)
        if clip_embedding is not None:
            tensor_bytes += self._estimate_tensor_bytes(clip_embedding)

        with self._lock:
            if key_hash in self._store:
                old_entry = self._store[key_hash]
                self._current_memory_bytes -= self._estimate_entry_bytes(old_entry)
                self._matrix_free(key_hash)
                del self._store[key_hash]

            cached_latents = latents.detach().clone().cpu()
            cached_step_latents = None
            if step_latents is not None:
                cached_step_latents = [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.detach().clone().cpu(),
                    )
                    for s in step_latents
                ]
            cached_clip = clip_embedding.detach().clone().cpu() if clip_embedding is not None else None

            # Persist to LMCache BEFORE inserting into OrderedDict, so that data
            # is safely on disk even if CPU eviction immediately clears it.
            if self._lmcache is not None:
                self._lmcache.put(f"{key_hash}:final", cached_latents)
                if cached_step_latents:
                    # Store step latents individually + a meta tensor carrying
                    # [step_index, timestep] pairs so recovery preserves the
                    # real diffusion timesteps (not just the step index).
                    meta_pairs = torch.tensor(
                        [[s.step_index, s.timestep] for s in cached_step_latents],
                        dtype=torch.float32,
                    )
                    self._lmcache.put(f"{key_hash}:steps_meta", meta_pairs)
                    for s in cached_step_latents:
                        self._lmcache.put(
                            f"{key_hash}:step_{s.step_index:04d}", s.latent
                        )

            self._evict_if_needed(tensor_bytes)

            entry = CacheEntry(
                latents=cached_latents,
                cache_key_hash=key_hash,
                step_latents=cached_step_latents,
                metadata=metadata,
                clip_embedding=cached_clip,
                cache_key=key,
            )
            self._store[key_hash] = entry
            self._matrix_add(key_hash, cached_clip)
            self._current_memory_bytes += tensor_bytes

            # If CPU budget is 0, evict this entry's heavy data immediately
            # (data is already safe in LMCache from the put above).
            if self._lmcache is not None:
                self._evict_if_needed(0)

            # Cached DiT state for key
            num_steps = len(cached_step_latents) if cached_step_latents is not None else 0
            logger.debug(
                "Cached DiT state for key %s, size %.2f MB (final + %d steps), total cache %.2f MB / %d entries",
                key_hash[:8],
                tensor_bytes / _MB,
                num_steps,
                self._current_memory_bytes / _MB,
                len(self._store),
            )

    def _recover_latents_from_lmcache(self, entry: CacheEntry, key_hash: str) -> bool:
        """Recover entry.latents from LMCache if it was evicted to None.
        Called inside self._lock. Returns True if latents are now available
        (either recovered or were already present), False if LMCache miss."""
        if entry.latents is not None:
            return True
        if self._lmcache is None:
            return False
        recovered = self._lmcache.get(f"{key_hash}:final", device="cpu")
        if recovered is None:
            return False
        entry.latents = recovered
        self._current_memory_bytes += self._estimate_tensor_bytes(recovered)
        logger.debug("Recovered %s final latent from LMCache", key_hash[:8])
        return True

    def get(self, key: CacheKey, target_device: torch.device | str | None = None) -> torch.Tensor | None:
        key_hash = key.to_hash()

        with self._lock:
            entry = self._store.get(key_hash)
            if entry is None:
                self._misses += 1
                logger.debug("Cache MISS for key %s", key_hash[:8])
                return None

            # LMCache recovery: if heavy tensors were evicted from CPU, recover.
            if not self._recover_latents_from_lmcache(entry, key_hash):
                self._misses += 1
                logger.warning("LMCache miss for key %s, treating as miss", key_hash[:8])
                return None

            self._store.move_to_end(key_hash)
            self._hits += 1

            latents = entry.latents
            if target_device is not None:
                latents = latents.to(device=target_device)
            else:
                latents = latents.clone()

            logger.debug(
                "Cache HIT for key %s (hits=%d, misses=%d, hit_rate=%.2f%%)",
                key_hash[:8],
                self._hits,
                self._misses,
                self.hit_rate * 100,
            )
            return latents

    def update_image_embedding(self, key_hash: str, image_embedding: torch.Tensor) -> bool:
        with self._lock:
            entry = self._store.get(key_hash)
            if entry is None:
                return False
            img_emb = image_embedding.detach().clone().cpu()
            entry.image_embedding = img_emb
            self._matrix_update_image(key_hash, img_emb)
            logger.debug("Updated image embedding for cache entry %s", key_hash[:8])
            return True

    def semantic_search(
        self,
        query_embedding: torch.Tensor,
        threshold: float = 0.75,
        target_device: torch.device | str | None = None,
        required_height: int | None = None,
        required_width: int | None = None,
        required_num_inference_steps: int | None = None,
        required_num_frames: int | None = None,
    ) -> tuple[torch.Tensor | None, list[StepLatentData] | None, float, str | None, str | None]:
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        query_norm = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        best_sim = 0.0
        best_key_hash = None
        best_t2t = 0.0
        best_t2i = 0.0
        best_penalty = 1.0

        with self._lock:
            # ---- Vectorized retrieval using pre-stacked matrices ----
            # The text embedding matrix (_txt_matrix) is maintained incrementally
            # on put()/evict(), so we avoid torch.stack() on every search.
            # We iterate over rows to build dimension/step-filtered index lists,
            # then slice the matrices and compute all similarities in batch.
            # Rows fall into:
            #   hybrid: has image embedding -> sim = t2t * sigmoid_penalty(t2i)
            #   text_only: no image embedding -> sim = t2t
            if self._txt_matrix is None or self._next_row == 0:
                # empty cache
                self._misses += 1
                logger.debug(
                    "CLIP semantic search: no match (cache empty, threshold=%.2f)",
                    threshold,
                )
                return None, None, 0.0, None, None

            # Build filtered index lists (row indices) + track which are hybrid
            hybrid_row_idxs: list[int] = []
            text_row_idxs: list[int] = []
            for row_idx in range(self._next_row):
                kh = self._row_keys[row_idx]
                if kh is None:
                    continue  # freed row
                entry = self._store.get(kh)
                if entry is None:
                    continue
                if entry.cache_key is not None:
                    if required_height is not None and entry.cache_key.height != required_height:
                        continue
                    if required_width is not None and entry.cache_key.width != required_width:
                        continue
                    if (
                        required_num_inference_steps is not None
                        and entry.cache_key.num_inference_steps < required_num_inference_steps
                    ):
                        continue
                    if (
                        required_num_frames is not None
                        and entry.cache_key.num_frames != required_num_frames
                    ):
                        continue
                if entry.image_embedding is not None and entry.clip_embedding is not None:
                    hybrid_row_idxs.append(row_idx)
                elif entry.clip_embedding is not None:
                    text_row_idxs.append(row_idx)

            q_cpu = query_norm.cpu()  # [1, dim]

            # --- Hybrid group: sim = t2t * sigmoid_penalty(t2i) ---
            # Embeddings already L2-normalized at encode time (backend.py),
            # so cosine similarity = dot product directly — no re-normalization.
            if hybrid_row_idxs:
                idx_t = torch.tensor(hybrid_row_idxs, dtype=torch.long)
                txt_sub = self._txt_matrix.index_select(0, idx_t)  # [M, dim]
                img_sub = self._img_matrix.index_select(0, idx_t)  # [M, dim]
                t2t = (q_cpu * txt_sub).sum(dim=-1)  # [M]
                t2i = (q_cpu * img_sub).sum(dim=-1)  # [M]
                if self._use_t2i_penalty:
                    penalty = torch.sigmoid((t2i - 0.10) * 10)  # [M]
                else:
                    penalty = torch.ones_like(t2t)  # no t2i penalty
                sims = t2t * penalty  # [M]

                self._all_sims.extend(sims.tolist())
                self._all_t2t_sims.extend(t2t.tolist())
                best_local = int(torch.argmax(sims).item())
                best_sim = float(sims[best_local].item())
                best_row = hybrid_row_idxs[best_local]
                best_key_hash = self._row_keys[best_row]
                best_t2t = float(t2t[best_local].item())
                best_t2i = float(t2i[best_local].item())
                best_penalty = float(penalty[best_local].item())

            # --- Text-only group: sim = t2t ---
            if text_row_idxs:
                idx_t = torch.tensor(text_row_idxs, dtype=torch.long)
                txt_sub = self._txt_matrix.index_select(0, idx_t)  # [K, dim]
                t2t = (q_cpu * txt_sub).sum(dim=-1)  # [K]
                sims = t2t  # sim == t2t for text-only

                self._all_sims.extend(sims.tolist())
                self._all_t2t_sims.extend(t2t.tolist())
                best_local = int(torch.argmax(sims).item())
                top_sim = float(sims[best_local].item())
                if top_sim > best_sim:
                    best_sim = top_sim
                    best_row = text_row_idxs[best_local]
                    best_key_hash = self._row_keys[best_row]
                    best_t2t = float(t2t[best_local].item())
                    best_t2i = 0.0
                    best_penalty = 1.0

            # Original loop initialized best_sim=0.0 and only updated on strictly
            # greater values, so an all-negative result left best_key_hash=None.
            # argmax can pick a negative winner; clamp to match original behavior
            # (all-negative -> no match, consistent best_sim=0.0 in logs/stats).
            if best_sim < 0.0:
                best_sim = 0.0
                best_key_hash = None
                best_t2t = 0.0
                best_t2i = 0.0
                best_penalty = 1.0

            # Throttle stats flushing: only every N searches to avoid O(total)
            # numpy computation + disk write on every single query.
            self._search_count += 1
            if self._search_count % _SIM_STATS_FLUSH_INTERVAL == 0:
                self._flush_sim_stats_to_file()

            if best_key_hash is not None and self._store[best_key_hash].image_embedding is not None:
                match_type = "hybrid"
            else:
                match_type = "text-text"

            if best_key_hash is None or best_sim < threshold:
                self._misses += 1
                logger.debug(
                    "CLIP semantic search: no match (t2t=%.4f, t2i=%.4f, penalty=%.4f, final=%.4f, threshold=%.2f)",
                    best_t2t,
                    best_t2i,
                    best_penalty,
                    best_sim,
                    threshold,
                )
                return None, None, best_sim, None, None

            self._store.move_to_end(best_key_hash)
            self._hits += 1
            entry = self._store[best_key_hash]

            # LMCache recovery: if heavy tensors were evicted from CPU, recover.
            if not self._recover_latents_from_lmcache(entry, best_key_hash):
                self._misses += 1
                logger.warning(
                    "LMCache miss for semantic hit %s, treating as miss",
                    best_key_hash[:8],
                )
                return None, None, best_sim, None, None

            cached_prompt = entry.cache_key.prompt if entry.cache_key is not None else None

            latents = entry.latents
            if target_device is not None:
                latents = latents.to(device=target_device)
            else:
                latents = latents.clone()

            step_latents = None
            if entry.step_latents is not None and target_device is not None:
                step_latents = [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.to(device=target_device),
                    )
                    for s in entry.step_latents
                ]
            elif entry.step_latents is not None:
                step_latents = [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.clone(),
                    )
                    for s in entry.step_latents
                ]

                logger.debug(
                    "CLIP semantic HIT [%s]: key=%s "
                    "t2t=%.4f t2i=%.4f penalty=%.4f final=%.4f "
                    "(threshold=%.2f, hits=%d, misses=%d)",
                    match_type,
                    best_key_hash[:8],
                    best_t2t,
                best_t2i,
                best_penalty,
                best_sim,
                threshold,
                self._hits,
                self._misses,
            )
            return latents, step_latents, best_sim, cached_prompt, match_type

    def get_step_latents(
        self, key: CacheKey, target_device: torch.device | str | None = None
    ) -> list[StepLatentData] | None:
        key_hash = key.to_hash()

        with self._lock:
            entry = self._store.get(key_hash)
            if entry is None:
                return None

            # LMCache recovery: if step_latents were evicted from CPU, recover.
            if entry.step_latents is None and self._lmcache is not None:
                # Recover final latent first (to ensure entry is warm).
                self._recover_latents_from_lmcache(entry, key_hash)
                # Recover step latents using the meta tensor that carries
                # [step_index, timestep] pairs (preserves real diffusion timesteps).
                meta = self._lmcache.get(f"{key_hash}:steps_meta", device="cpu")
                if meta is None:
                    return None
                recovered_steps = []
                for row in meta:
                    si = int(row[0].item())
                    ts = float(row[1].item())
                    latent = self._lmcache.get(f"{key_hash}:step_{si:04d}", device="cpu")
                    if latent is None:
                        break
                    recovered_steps.append(StepLatentData(step_index=si, timestep=ts, latent=latent))
                if recovered_steps:
                    entry.step_latents = recovered_steps
                else:
                    return None

            if entry.step_latents is None:
                return None

            self._store.move_to_end(key_hash)

            if target_device is not None:
                return [
                    StepLatentData(
                        step_index=s.step_index,
                        timestep=s.timestep,
                        latent=s.latent.to(device=target_device),
                    )
                    for s in entry.step_latents
                ]
            return [
                StepLatentData(
                    step_index=s.step_index,
                    timestep=s.timestep,
                    latent=s.latent.clone(),
                )
                for s in entry.step_latents
            ]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get_similarity_stats(self) -> dict:
        """Return distribution stats of all similarity values collected.
        Reads from a shared file so it can be called from any process."""
        try:
            with open(_SIM_STATS_PATH) as f:
                return json.load(f)
        except Exception:
            return {"final_sim": {"total_comparisons": 0}, "t2t_sim": {"total_comparisons": 0}}

    def reset_similarity_stats(self) -> None:
        """Clear collected similarity values and reset the shared file."""
        with self._lock:
            self._all_sims.clear()
            self._all_t2t_sims.clear()
        try:
            with open(_SIM_STATS_PATH, "w") as f:
                json.dump({"final_sim": {"total_comparisons": 0}, "t2t_sim": {"total_comparisons": 0}}, f)
        except Exception:
            pass

    def _flush_sim_stats_to_file(self) -> None:
        """Write current sim stats to a shared file (called from within lock)."""

        def _compute_stats(values):
            if not values:
                return {"total_comparisons": 0}
            arr = np.array(values)
            return {
                "total_comparisons": len(values),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "gte_0.9": int(np.sum(arr >= 0.9)),
                "gte_0.8": int(np.sum(arr >= 0.8)),
                "gte_0.7": int(np.sum(arr >= 0.7)),
                "gte_0.6": int(np.sum(arr >= 0.6)),
                "gte_0.5": int(np.sum(arr >= 0.5)),
            }

        result = {
            "final_sim": _compute_stats(self._all_sims),
            "t2t_sim": _compute_stats(self._all_t2t_sims),
        }
        try:
            with open(_SIM_STATS_PATH, "w") as f:
                json.dump(result, f)
        except Exception:
            pass

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self):
        with self._lock:
            self._store.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            # reset embedding matrices
            self._emb_dim = None
            self._txt_matrix = None
            self._img_matrix = None
            self._row_keys = []
            self._key_rows = {}
            self._next_row = 0
            self._capacity = 0
            logger.info("DiT cache store cleared")

    def stats(self) -> dict[str, Any]:
        with self._lock:
            shell_count = sum(1 for e in self._store.values() if e.latents is None)
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "memory_mb": self._current_memory_bytes / _MB,
                "max_memory_gb": self._max_memory_bytes / _GB,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
                "lmcache_enabled": self._lmcache is not None,
                "shells_latents_evicted": shell_count,
            }

    # ==================================================================
    # Persistent storage (save_to_disk / load_from_disk — for cross-process
    # reuse without LMCache; kept for backward compatibility)
    # ==================================================================

    def save_to_disk(self, cache_dir: str | Path) -> int:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        with self._lock:
            for key_hash, entry in self._store.items():
                # Skip shells whose heavy data was evicted to LMCache —
                # they have no latents to save (data lives in LMCache, not
                # persistent_cache_dir). Without this guard, entry.latents.cpu()
                # would raise AttributeError on None and be silently swallowed.
                if entry.latents is None:
                    continue
                entry_dir = cache_dir / key_hash
                try:
                    entry_dir.mkdir(parents=True, exist_ok=True)

                    torch.save(
                        entry.latents.cpu(),
                        entry_dir / "final_latent.pt",
                    )

                    meta = {
                        "cache_key_hash": entry.cache_key_hash,
                        "metadata": entry.metadata,
                    }

                    # Persist cache_key so that semantic_search dimension filters
                    # (height/width/steps/frames) survive a restart.
                    if entry.cache_key is not None:
                        ck = entry.cache_key
                        meta["cache_key"] = {
                            "prompt": ck.prompt,
                            "negative_prompt": ck.negative_prompt,
                            "height": ck.height,
                            "width": ck.width,
                            "num_inference_steps": ck.num_inference_steps,
                            "guidance_scale": ck.guidance_scale,
                            "true_cfg_scale": ck.true_cfg_scale,
                            "seed": ck.seed,
                            "sigmas": list(ck.sigmas) if ck.sigmas is not None else None,
                            "max_sequence_length": ck.max_sequence_length,
                            "num_images_per_prompt": ck.num_images_per_prompt,
                            "num_frames": ck.num_frames,
                        }

                    # Persist text/image embeddings so loaded entries remain
                    # searchable by semantic_search without re-encoding.
                    if entry.clip_embedding is not None:
                        torch.save(
                            entry.clip_embedding.cpu(),
                            entry_dir / "clip_embedding.pt",
                        )
                    if entry.image_embedding is not None:
                        torch.save(
                            entry.image_embedding.cpu(),
                            entry_dir / "image_embedding.pt",
                        )

                    if entry.step_latents is not None:
                        step_data = []
                        for s in entry.step_latents:
                            step_file = entry_dir / f"step_{s.step_index:04d}.pt"
                            torch.save(
                                {
                                    "step_index": s.step_index,
                                    "timestep": s.timestep,
                                    "latent": s.latent.cpu(),
                                },
                                step_file,
                            )
                            step_data.append(
                                {
                                    "step_index": s.step_index,
                                    "timestep": s.timestep,
                                    "file": step_file.name,
                                }
                            )
                        meta["step_latents"] = step_data
                        meta["num_steps"] = len(step_data)

                    with open(entry_dir / "meta.json", "w") as f:
                        json.dump(meta, f, indent=2)

                    saved_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to save cache entry %s to disk: %s",
                        key_hash[:8],
                        e,
                    )

        logger.info(
            "Saved %d cache entries to %s (%.2f MB)",
            saved_count,
            cache_dir,
            self._current_memory_bytes / _MB,
        )
        return saved_count

    def load_from_disk(self, cache_dir: str | Path) -> int:
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            logger.info("Cache directory %s does not exist, skipping load", cache_dir)
            return 0

        loaded_count = 0
        with self._lock:
            for entry_dir in sorted(cache_dir.iterdir()):
                if not entry_dir.is_dir():
                    continue
                meta_file = entry_dir / "meta.json"
                latent_file = entry_dir / "final_latent.pt"
                if not meta_file.exists() or not latent_file.exists():
                    continue

                try:
                    key_hash = entry_dir.name

                    with open(meta_file) as f:
                        meta = json.load(f)

                    latents = torch.load(latent_file, map_location="cpu", weights_only=True)

                    step_latents = None
                    if "step_latents" in meta and meta["step_latents"]:
                        step_latents = []
                        for step_info in meta["step_latents"]:
                            step_file = entry_dir / step_info["file"]
                            if step_file.exists():
                                step_data = torch.load(step_file, map_location="cpu", weights_only=True)
                                step_latents.append(
                                    StepLatentData(
                                        step_index=step_data["step_index"],
                                        timestep=step_data["timestep"],
                                        latent=step_data["latent"],
                                    )
                                )

                    if key_hash in self._store:
                        old_entry = self._store[key_hash]
                        self._current_memory_bytes -= self._estimate_entry_bytes(old_entry)

                    # Restore CacheKey so semantic_search dimension filters work.
                    cache_key = None
                    ck_meta = meta.get("cache_key")
                    if ck_meta is not None:
                        try:
                            cache_key = CacheKey(
                                prompt=ck_meta["prompt"],
                                negative_prompt=ck_meta.get("negative_prompt", ""),
                                height=ck_meta["height"],
                                width=ck_meta["width"],
                                num_inference_steps=ck_meta["num_inference_steps"],
                                guidance_scale=ck_meta["guidance_scale"],
                                true_cfg_scale=ck_meta["true_cfg_scale"],
                                seed=ck_meta["seed"],
                                sigmas=tuple(ck_meta["sigmas"]) if ck_meta.get("sigmas") is not None else None,
                                max_sequence_length=ck_meta.get("max_sequence_length"),
                                num_images_per_prompt=ck_meta["num_images_per_prompt"],
                                num_frames=ck_meta.get("num_frames", 1),
                            )
                        except (KeyError, TypeError) as e:
                            logger.debug("Could not restore CacheKey for %s: %s", key_hash[:8], e)

                    # Restore embeddings so loaded entries stay semantically searchable.
                    clip_embedding = None
                    image_embedding = None
                    clip_file = entry_dir / "clip_embedding.pt"
                    img_file = entry_dir / "image_embedding.pt"
                    if clip_file.exists():
                        clip_embedding = torch.load(clip_file, map_location="cpu", weights_only=True)
                    if img_file.exists():
                        image_embedding = torch.load(img_file, map_location="cpu", weights_only=True)

                    entry = CacheEntry(
                        latents=latents,
                        cache_key_hash=meta.get("cache_key_hash", key_hash),
                        step_latents=step_latents,
                        metadata=meta.get("metadata"),
                        clip_embedding=clip_embedding,
                        cache_key=cache_key,
                        image_embedding=image_embedding,
                    )
                    self._store[key_hash] = entry
                    # Register in the semantic-search matrices so loaded entries
                    # are visible to vectorized semantic_search.
                    self._matrix_add(key_hash, clip_embedding)
                    if image_embedding is not None:
                        self._matrix_update_image(key_hash, image_embedding)
                    self._current_memory_bytes += self._estimate_entry_bytes(entry)
                    loaded_count += 1

                except Exception as e:
                    logger.warning(
                        "Failed to load cache entry from %s: %s",
                        entry_dir,
                        e,
                    )

        logger.info(
            "Loaded %d cache entries from %s (%.2f MB)",
            loaded_count,
            cache_dir,
            self._current_memory_bytes / _MB,
        )
        return loaded_count


def build_cache_key_from_request(
    req: Any,
    pipeline: Any,
) -> CacheKey | None:
    try:
        prompt_item = getattr(req, "prompt", None)
        if prompt_item is None:
            return None

        prompt_text = ""
        negative_prompt_text = ""
        if isinstance(prompt_item, str):
            prompt_text = prompt_item
        elif isinstance(prompt_item, dict):
            prompt_text = prompt_item.get("prompt", "")
            negative_prompt_text = prompt_item.get("negative_prompt", "") or ""

        sampling = req.sampling_params

        height = sampling.height
        width = sampling.width
        if height is None and hasattr(pipeline, "default_sample_size"):
            vae_sf = getattr(pipeline, "vae_scale_factor", 8)
            height = pipeline.default_sample_size * vae_sf
        if width is None and hasattr(pipeline, "default_sample_size"):
            vae_sf = getattr(pipeline, "vae_scale_factor", 8)
            width = pipeline.default_sample_size * vae_sf

        num_inference_steps = sampling.num_inference_steps or 50
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        true_cfg_scale = sampling.true_cfg_scale or 1.0
        seed = sampling.seed if sampling.seed is not None else -1
        if seed == -1 and sampling.generator is not None:
            try:
                if isinstance(sampling.generator, torch.Generator):
                    seed = sampling.generator.initial_seed()
            except Exception:
                pass

        sigmas = tuple(sampling.sigmas) if sampling.sigmas is not None else None
        max_sequence_length = sampling.max_sequence_length
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        num_frames = getattr(sampling, "num_frames", 1) or 1

        return CacheKey(
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            sigmas=sigmas,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
        )
    except Exception as e:
        logger.warning("Failed to build cache key: %s", e)
        return None
