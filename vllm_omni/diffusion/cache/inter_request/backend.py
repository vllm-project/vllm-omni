from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.inter_request.cache_store import (
    DiTCacheStore,
    StepLatentData,
    build_cache_key_from_request,
)
from vllm_omni.diffusion.cache.inter_request.step_recorder import StepLatentsRecorder
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = logging.getLogger(__name__)


class InterRequestCacheBackend(CacheBackend):
    """
    Inter-request cache backend for DiT full reuse (Chorus Stage-1).

    This backend implements the Stage-1 caching strategy from the Chorus paper:
    when two requests have identical inputs (same prompt, dimensions, seed, etc.),
    the DiT computation can be entirely skipped by reusing cached latent features
    from a previous request.

    Unlike intra-request caching backends (cache_dit, TeaCache) that optimize
    within a single denoising process, this backend caches the final latents
    across different requests, enabling complete DiT computation reuse.

    The cache stores:
    - Key: Hash of all inputs that determine the DiT output (prompt, seed, etc.)
    - Value: Final latents after all denoising steps (before VAE decode)
    - Step latents: Intermediate latents at every denoising step (for future
      partial-resume capability)

    A :class:`StepLatentsRecorder` is always attached to the pipeline to capture
    intermediate latents during denoising.  These are stored in the cache
    alongside the final latent.

    Usage:
        omni = Omni(
            model="Qwen/Qwen-Image",
            cache_backend="inter_request",
            cache_config={
                "inter_request_max_entries": 100,
                "inter_request_max_memory_gb": 0.0,
            }
        )
    """

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        max_entries = getattr(config, "inter_request_max_entries", 100)
        max_memory_gb = getattr(config, "inter_request_max_memory_gb", 0.0)

        # Initialize LMCache ECCacheEngine for CPU→Disk tiered storage.
        # ECCacheEngine stores arbitrary tensors by string key with built-in
        # CPU→Disk layering, LRU eviction, and async persistence.
        lmcache_disk_dir = getattr(config, "inter_request_lmcache_disk_dir", None)
        if lmcache_disk_dir:
            import torch as _torch
            from lmcache.v1.cache_engine import LMCacheMetadata
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.ec_engine import ECCacheEngine

            cpu_gb = getattr(config, "inter_request_lmcache_max_cpu_gb", 5.0)
            disk_gb = getattr(config, "inter_request_lmcache_max_disk_gb", 100.0)

            lc_metadata = LMCacheMetadata(
                model_name="vllm_omni_diffusion",
                world_size=1,
                local_world_size=1,
                worker_id=0,
                local_worker_id=0,
                kv_dtype=_torch.float32,
                kv_shape=(1, 1, 1, 1, 1),
            )

            # Single LMCache engine for step latents only (8MB each).
            # Final latent (185MB) is stored via direct torch.save to avoid
            # LMCache CPU pool pressure and LRU-eviction-before-write data loss.
            steps_dir = os.path.join(lmcache_disk_dir, "steps")
            os.makedirs(steps_dir, exist_ok=True)

            self._lmcache_engine = ECCacheEngine(
                config=LMCacheEngineConfig.from_defaults(
                    local_cpu=True,
                    max_local_cpu_size=cpu_gb,
                    local_disk=steps_dir,
                    max_local_disk_size=disk_gb,
                    save_decode_cache=True,
                ),
                metadata=lc_metadata,
                encoder_dtype=_torch.float32,
            )
            logger.info(
                "LMCache ECCacheEngine initialized for step latents: steps_dir=%s, cpu_gb=%.1f",
                steps_dir,
                cpu_gb,
            )
        else:
            self._lmcache_engine = None

        # Final latent directory (torch.save direct to disk, bypasses LMCache)
        final_disk_dir = None
        if lmcache_disk_dir:
            final_disk_dir = os.path.join(lmcache_disk_dir, "final_direct")
            os.makedirs(final_disk_dir, exist_ok=True)

        self._cache_store = DiTCacheStore(
            max_entries=max_entries,
            max_memory_gb=max_memory_gb,
            lmcache_engine=self._lmcache_engine,
            max_stored_steps=getattr(config, "inter_request_max_stored_steps", 0),
            final_disk_dir=final_disk_dir,
        )
        # Fingerprint of model checkpoint + TP world size; included in every
        # cache key so persisted caches are namespaced per model/topology.
        self._model_digest = ""
        self._pipeline = None

        self._persistent_cache_dir = getattr(config, "inter_request_persistent_cache_dir", None)

        self._clip_model_path = getattr(config, "inter_request_clip_model_path", None)
        self._clip_threshold = float(getattr(config, "inter_request_clip_threshold", 0.75))
        self._clip_min_skip = int(getattr(config, "inter_request_clip_min_skip", 5))
        self._clip_max_skip_ratio = float(getattr(config, "inter_request_clip_max_skip_ratio", 0.5))
        self._use_t2i_penalty = bool(getattr(config, "inter_request_use_t2i_penalty", True))
        self._cache_store.set_t2i_penalty(self._use_t2i_penalty)
        self._clip_tokenizer = None
        self._clip_model = None
        self._clip_device = None
        # CLIP sub-attributes are only fully populated inside _init_clip_encoder().
        # Pre-initialize them here so that update_image_embedding / encode_image
        # can safely short-circuit when CLIP is not configured (otherwise they
        # would raise AttributeError on the never-set attributes).
        self._full_clip_model = None
        self._clip_processor = None
        self._clip_image_processor = None
        self._use_fgclip = False

        logger.info(
            "InterRequestCacheBackend initialized: "
            "max_entries=%d, max_memory_gb=%.1f, "
            "persistent_cache_dir=%s, clip_model_path=%s, "
            "clip_threshold=%.2f, clip_min_skip=%d, clip_max_skip_ratio=%.2f",
            max_entries,
            max_memory_gb,
            self._persistent_cache_dir,
            self._clip_model_path,
            self._clip_threshold,
            self._clip_min_skip,
            self._clip_max_skip_ratio,
        )

    def enable(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        self.enabled = True
        self._model_digest = self._compute_model_digest(pipeline)
        self._recorder = StepLatentsRecorder()
        # Register as a general-purpose denoising-step hook (see
        # ProgressBarMixin). Kept also as an attribute for backwards compat.
        pipeline.register_diffuse_step_hook(self._recorder)
        pipeline._step_latents_recorder = self._recorder

        if self._clip_model_path is not None:
            self._init_clip_encoder()

        # Restore persisted cache on startup.
        # When LMCache is enabled, latents are already on disk (managed by
        # LMCache) and will be recovered on first get(). We only need to
        # restore the embedding shells for semantic_search.
        # When persistent_cache_dir is set (without LMCache), do a full load.
        if self._persistent_cache_dir is not None and self._lmcache_engine is None:
            loaded = self._cache_store.load_from_disk(self._persistent_cache_dir)
            if loaded > 0:
                logger.info(
                    "Loaded %d cache entries from persistent storage %s",
                    loaded,
                    self._persistent_cache_dir,
                )

        logger.info(
            "InterRequestCacheBackend enabled on pipeline %s (model_digest=%s)",
            pipeline.__class__.__name__,
            self._model_digest[:8] or "n/a",
        )

    @staticmethod
    def _compute_model_digest(pipeline: Any) -> str:
        """Fingerprint the model checkpoint + TP topology for cache namespacing.

        Persisted caches written by a different model or world size must never
        be served: identical prompts across checkpoints would collide on hash
        keys otherwise.
        """
        import hashlib

        parts = [type(pipeline).__name__]
        for attr in ("name_or_path", "model_name", "_name_or_path"):
            val = getattr(pipeline, attr, None)
            if isinstance(val, str) and val:
                parts.append(val)
                break
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                parts.append(f"tp{dist.get_world_size()}")
        except Exception:
            pass
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _init_clip_encoder(self) -> None:
        try:
            self._clip_device = torch.device("cpu")
            logger.info("Loading CLIP model from %s on %s", self._clip_model_path, self._clip_device)

            # Verify the path exists before passing to from_pretrained,
            # otherwise transformers treats it as a repo id and fails confusingly.
            config_path = Path(self._clip_model_path)
            if not config_path.exists():
                logger.warning("CLIP model path does not exist: %s, semantic matching disabled", self._clip_model_path)
                self._clip_model = None
                self._clip_tokenizer = None
                return

            config_path = Path(self._clip_model_path)
            fgclip_config = config_path / "modeling_fgclip.py"
            if fgclip_config.exists():
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self._clip_tokenizer = AutoTokenizer.from_pretrained(self._clip_model_path)
                self._clip_model = AutoModelForCausalLM.from_pretrained(self._clip_model_path, trust_remote_code=True)
                self._clip_model.to(self._clip_device)
                self._clip_model.eval()
                self._use_fgclip = True
                self._clip_image_processor = None
                self._full_clip_model = None
                logger.info("FG-CLIP text encoder loaded successfully (text-only mode)")
            else:
                from transformers import CLIPModel, CLIPProcessor

                self._full_clip_model = CLIPModel.from_pretrained(self._clip_model_path)
                self._full_clip_model.to(self._clip_device)
                self._full_clip_model.eval()
                self._clip_processor = CLIPProcessor.from_pretrained(self._clip_model_path)
                self._clip_model = self._full_clip_model
                self._clip_tokenizer = self._clip_processor.tokenizer
                self._clip_image_processor = self._clip_processor.image_processor
                self._use_fgclip = False
                logger.info("CLIP model loaded successfully (text+image mode)")
        except Exception as e:
            logger.warning("Failed to load CLIP encoder: %s, semantic matching disabled", e)
            self._clip_model = None
            self._clip_tokenizer = None

    def encode_prompt(self, prompt: str) -> torch.Tensor | None:
        if self._clip_model is None or self._clip_tokenizer is None:
            return None
        try:
            if getattr(self, "_use_fgclip", False):
                inputs = self._clip_tokenizer(
                    [prompt], max_length=77, padding="max_length", truncation=True, return_tensors="pt"
                ).to(self._clip_device)
                with torch.no_grad():
                    text_features = self._clip_model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        walk_short_pos=True,
                    )
                feat = text_features.squeeze(0)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat
            else:
                inputs = self._clip_tokenizer([prompt], padding=True, return_tensors="pt").to(self._clip_device)
                with torch.no_grad():
                    text_features = self._full_clip_model.get_text_features(**inputs)
                # Handle both raw tensor (older transformers) and HF model output.
                # Newer transformers return BaseModelOutputWithPooling where the
                # text embedding is in pooler_output.
                if hasattr(text_features, "pooler_output"):
                    feat = text_features.pooler_output.squeeze(0)
                elif hasattr(text_features, "text_embeds"):
                    feat = text_features.text_embeds.squeeze(0)
                elif torch.is_tensor(text_features):
                    feat = text_features.squeeze(0)
                else:
                    feat = text_features.last_hidden_state.mean(dim=1).squeeze(0)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat
        except Exception as e:
            logger.warning("CLIP encoding failed: %s", e)
            return None

    def _encode_image_cpu(self, image_tensor: torch.Tensor) -> torch.Tensor | None:
        """Encode image on CPU to avoid NPU async issues."""
        if image_tensor.dim() != 4 or image_tensor.shape[1] != 3:
            return None
        try:
            from PIL import Image

            img = image_tensor[0].float().cpu()  # [3, H, W]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).clamp(0, 255).to(torch.uint8)
            pil_image = Image.fromarray(img.permute(1, 2, 0).numpy())
            inputs = self._clip_image_processor(images=[pil_image], return_tensors="pt")
            with torch.no_grad():
                feats = self._full_clip_model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feat = feats.pooler_output.squeeze(0)
            elif torch.is_tensor(feats):
                feat = feats.squeeze(0)
            else:
                feat = feats.last_hidden_state.mean(dim=1).squeeze(0)
            return feat / feat.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.debug("encode_image_cpu failed: %s", e)
            return None

    def _decode_latent_to_image(self, latent: torch.Tensor, height: int, width: int) -> torch.Tensor | None:
        """Best-effort decode of a final latent to a pixel image [B, 3, H, W].

        The t2i image-similarity penalty needs the *generated image*, but the
        runner only hands us the pre-VAE latent. Decode via the pipeline's VAE
        on its native device; on any failure return None (the entry then stays
        in the text-only matching group — no penalty, but matching still works).
        """
        if self._pipeline is None:
            return None
        try:
            decoded = None
            decode = getattr(self._pipeline, "_decode_latents", None)
            if decode is not None:
                decoded = decode(latent.to(next(self._pipeline.vae.parameters()).device), height, width, "pt")
            vae = getattr(self._pipeline, "vae", None)
            if (decoded is None) and (vae is not None):
                scale = getattr(getattr(vae, "config", None), "scaling_factor", None)
                scaled = latent / scale if scale else latent
                decoded = vae.decode(scaled.to(next(vae.parameters()).device)).sample
            if decoded is None:
                return None
            if decoded.dim() == 4 and decoded.shape[1] != 3:
                return None
            return decoded.detach()
        except Exception as e:
            logger.debug("latent decode for image embedding failed: %s", e)
            return None

    def update_image_embedding(self, cache_key_hash: str | None, image_or_latent: torch.Tensor) -> None:
        """Store the image embedding for a cache entry (t2i hybrid penalty).

        Accepts either a pixel image [B, 3, H, W] or a final latent; latents
        are decoded through the pipeline VAE first. Runs on the background
        final-latent executor to keep decode off the request path.
        """
        if cache_key_hash is None:
            return
        if getattr(self, "_use_fgclip", False) or self._full_clip_model is None:
            return
        t = image_or_latent
        is_pixel_image = t.dim() == 4 and t.shape[1] == 3
        if is_pixel_image:
            image = t
        else:
            # Latent: decode in the background so VAE cost stays off the
            # request path; dimensions come from the stored cache key.
            entry_key = self._cache_store._store.get(cache_key_hash)
            h = entry_key.cache_key.height if entry_key is not None and entry_key.cache_key else None
            w = entry_key.cache_key.width if entry_key is not None and entry_key.cache_key else None
            if h is None or w is None:
                return
            image = self._decode_latent_to_image(t.detach().cpu(), h, w)
            if image is None:
                logger.debug("no image embedding for %s: latent decode unavailable", cache_key_hash[:8])
                return

        def _encode_and_store():
            # Force CPU for image embedding to avoid NPU async issues in executor threads
            image_emb = self._encode_image_cpu(image)
            if image_emb is not None:
                self._cache_store.update_image_embedding(cache_key_hash, image_emb)
                logger.debug("stored image embedding for %s", cache_key_hash[:8])

        self._cache_store._final_write_executor.submit(_encode_and_store)

    def semantic_lookup(
        self, req: Any, target_device: torch.device | str | None = None
    ) -> tuple[torch.Tensor | None, list[StepLatentData] | None, float, str | None, str | None]:
        if not self.enabled or self._clip_model is None:
            return None, None, 0.0, None, None

        cache_key = build_cache_key_from_request(req, self._pipeline, model_digest=self._model_digest)
        if cache_key is None:
            return None, None, 0.0, None, None

        query_emb = self.encode_prompt(cache_key.prompt)
        if query_emb is None:
            return None, None, 0.0, None, None

        latents, step_latents, sim, cached_prompt, match_type = self._cache_store.semantic_search(
            query_emb,
            threshold=self._clip_threshold,
            target_device=target_device,
            required_height=cache_key.height,
            required_width=cache_key.width,
            required_num_inference_steps=cache_key.num_inference_steps,
            required_num_frames=cache_key.num_frames,
        )
        return latents, step_latents, sim, cached_prompt, match_type

    def compute_skip_steps(
        self,
        similarity: float,
        total_steps: int,
    ) -> int:
        if similarity < self._clip_threshold:
            return 0
        max_skip = int(total_steps * self._clip_max_skip_ratio)
        if max_skip <= self._clip_min_skip:
            return self._clip_min_skip

        ratio = (similarity - self._clip_threshold) / (1.0 - self._clip_threshold)
        ratio = min(max(ratio, 0.0), 1.0)

        skip = self._clip_min_skip + int(ratio * (max_skip - self._clip_min_skip))

        # Clamp to max_stored_steps: if we only stored the first N step latents,
        # resume beyond step N is impossible. Clamp to N so the runner can find
        # the step latent at index N-1.
        max_stored = self._cache_store._max_stored_steps
        if max_stored > 0 and skip > max_stored:
            skip = max_stored

        return skip

    @property
    def clip_enabled(self) -> bool:
        return self._clip_model is not None

    def shutdown(self) -> None:
        logger.info(
            "InterRequestCacheBackend shutdown: persistent_cache_dir=%s, lmcache_enabled=%s, cache_size=%d",
            self._persistent_cache_dir,
            self._lmcache_engine is not None,
            self._cache_store.size,
        )

        # Close LMCache engine (flushes async writes + stops background workers).
        if self._lmcache_engine is not None:
            self._lmcache_engine.close()
            logger.info("LMCache ECCacheEngine closed")
        # Flush pending final latent disk writes
        if hasattr(self._cache_store, "_final_write_executor"):
            self._cache_store._final_write_executor.shutdown(wait=True)
            logger.info("LMCache ECCacheEngine closed")

        # Persist hot (in-CPU) entries to persistent_cache_dir for cross-process
        # reuse (original behaviour, independent of LMCache).
        if self._persistent_cache_dir is not None and self._cache_store.size > 0:
            saved = self._cache_store.save_to_disk(self._persistent_cache_dir)
            logger.info(
                "Persisted %d cache entries to %s",
                saved,
                self._persistent_cache_dir,
            )

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        pass

    def before_diffuse(self, is_dummy: bool = False) -> None:
        if self._recorder is not None and not is_dummy:
            self._recorder.clear()
            self._recorder.enable()

    def after_diffuse(self, is_dummy: bool = False) -> None:
        """Reset the step-latents recorder after each forward pass."""
        if self._recorder is None or is_dummy:
            return
        self._recorder.disable()
        self._recorder.clear()

    def lookup(self, req: Any, target_device: torch.device | str | None = None) -> torch.Tensor | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline, model_digest=self._model_digest)
        if cache_key is None:
            return None

        return self._cache_store.get(cache_key, target_device=target_device)

    def lookup_step_latents(
        self, req: Any, target_device: torch.device | str | None = None
    ) -> list[StepLatentData] | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline, model_digest=self._model_digest)
        if cache_key is None:
            return None

        return self._cache_store.get_step_latents(cache_key, target_device=target_device)

    def store(
        self,
        req: Any,
        latents: torch.Tensor,
        step_latents: list[StepLatentData] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline, model_digest=self._model_digest)
        if cache_key is None:
            return None

        clip_emb = None
        if self._clip_model is not None:
            prompt = cache_key.prompt
            clip_emb = self.encode_prompt(prompt)

        self._cache_store.put(cache_key, latents, step_latents=step_latents, metadata=metadata, clip_embedding=clip_emb)
        return cache_key.to_hash()

    @property
    def cache_store(self) -> DiTCacheStore:
        return self._cache_store

    @property
    def recorder(self) -> StepLatentsRecorder | None:
        return self._recorder

    def stats(self) -> dict[str, Any]:
        return self._cache_store.stats()

    def similarity_stats(self) -> dict:
        return self._cache_store.get_similarity_stats()

    def reset_similarity_stats(self) -> None:
        self._cache_store.reset_similarity_stats()

    def clear(self) -> None:
        self._cache_store.clear()

    # ------------------------------------------------------------------
    # Cross-request polymorphic hooks (override CacheBackend no-ops).
    # These encapsulate the runner logic so the runner can call every
    # backend uniformly without isinstance() checks.
    # ------------------------------------------------------------------
    def short_circuit_requests(self, reqs: list, target_device: Any) -> tuple[list, list]:
        """Inspect requests before forward.

        Returns (hit_outputs, remaining_reqs) where hit_outputs is a list of
        (original_index, DiffusionOutput). Exact hits are short-circuited;
        semantic hits set ``resume_from_step`` on the request for the pipeline.
        """
        if not self.enabled or self._pipeline is None:
            return [], reqs

        from vllm_omni.diffusion.data import DiffusionOutput

        hit_outputs: list[tuple[int, Any]] = []
        remaining_reqs: list = []
        for idx, req in enumerate(reqs):
            # Path A: resume from a cached step (set by a prior semantic hit).
            resume_from_step = getattr(req.sampling_params, "resume_from_step", 0) or 0
            if resume_from_step > 0:
                step_latents_list = self.lookup_step_latents(req, target_device=target_device)
                if step_latents_list is not None and len(step_latents_list) >= resume_from_step:
                    resume_data = step_latents_list[resume_from_step - 1]
                    req.sampling_params.resume_latents = resume_data.latent
                    logger.info("Inter-request cache: resuming from step %d", resume_from_step)
                else:
                    req.sampling_params.resume_from_step = 0
                remaining_reqs.append(req)
                continue

            # Path B: exact hit -> skip forward entirely.
            cached_output = self.lookup(req, target_device=target_device)
            if cached_output is not None:
                logger.info("Inter-request cache HIT: skipping DiT computation entirely")
                hit_outputs.append((idx, DiffusionOutput(output=cached_output)))
                continue

            # Path C: semantic hit -> set resume_from_step for partial skip.
            if self.clip_enabled:
                clip_result = self.semantic_lookup(req, target_device=target_device)
                clip_latents, clip_step_latents, clip_sim, _, _ = clip_result
                if clip_latents is not None and clip_step_latents is None:
                    # The matched entry was evicted to a shell (heavy tensors
                    # dropped, embeddings kept). semantic_lookup cannot return
                    # step latents for shells — recover them from LMCache the
                    # same way the exact-hit path does.
                    clip_step_latents = self.lookup_step_latents(req, target_device=target_device)
                if clip_latents is not None and clip_step_latents is not None:
                    total_steps = req.sampling_params.num_inference_steps or len(clip_step_latents)
                    clip_resume_step = self.compute_skip_steps(clip_sim, total_steps)
                    if clip_resume_step > 0 and len(clip_step_latents) >= clip_resume_step:
                        resume_data = clip_step_latents[clip_resume_step - 1]
                        req.sampling_params.resume_latents = resume_data.latent
                        req.sampling_params.resume_from_step = clip_resume_step
                        logger.info(
                            "CLIP semantic match: similarity=%.4f, resuming from step %d/%d",
                            clip_sim,
                            clip_resume_step,
                            total_steps,
                        )
            remaining_reqs.append(req)
        return hit_outputs, remaining_reqs

    def post_forward_store(
        self,
        reqs: list,
        outputs: list,
        target_device: Any,
        runner: Any,
        is_dummy: bool = False,
    ) -> list:
        """Store computed outputs for future reuse; annotate outputs with cache hashes.

        For semantic-hit requests that resumed from a cached step, the final
        latent is complete and worth caching, but the step latents are partial
        (only the post-resume steps were recorded) so they are skipped.
        """
        if not self.enabled or self._pipeline is None:
            return outputs
        # The recorder's resume_from_step survives across forward(); use it to
        # detect whether this batch was a resume. req.sampling_params may have
        # been reset by the pipeline between short_circuit and here.
        recorder_resumed = self._recorder is not None and self._recorder.resume_from_step > 0
        for req, output in zip(reqs, outputs):
            if output.output is None or is_dummy:
                continue
            # Skip step latents when this was a resume (incomplete step history).
            step_latents_data = None
            if not recorder_resumed and self._recorder is not None and self._recorder.num_steps > 0:
                step_latents_data = [
                    StepLatentData(
                        step_index=r.step_index,
                        timestep=r.timestep,
                        latent=r.latent,
                    )
                    for r in self._recorder.records
                ]
            cache_key_hash = self.store(req, output.output, step_latents=step_latents_data)
            if recorder_resumed:
                steps_desc = "skipped(resumed)"
            else:
                steps_desc = f"{len(step_latents_data) if step_latents_data else 0} steps"
            logger.debug(
                "Inter-request cache: stored %s (step_latents=%s)",
                cache_key_hash[:8] if cache_key_hash else "n/a",
                steps_desc,
            )
            if cache_key_hash is not None and runner is not None and hasattr(runner, "_update_cache_image_embedding"):
                runner._update_cache_image_embedding(cache_key_hash, output.output)
        return outputs

    def merge_hit_outputs(self, outputs: list, hit_outputs: list) -> list:
        """Merge cache-hit outputs back into their original positions."""
        if not hit_outputs:
            return outputs
        merged: list = []
        computed_iter = iter(outputs)
        hit_map = dict(hit_outputs)
        total = max(max(hit_map.keys()) + 1, len(outputs) + len(hit_outputs))
        for i in range(total):
            if i in hit_map:
                merged.append(hit_map[i])
            else:
                merged.append(next(computed_iter))
        return merged
