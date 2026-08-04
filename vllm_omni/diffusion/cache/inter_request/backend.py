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
            from lmcache.v1.ec_engine import ECCacheEngine
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.cache_engine import LMCacheMetadata

            cpu_gb = getattr(config, "inter_request_lmcache_max_cpu_gb", 5.0)
            disk_gb = getattr(config, "inter_request_lmcache_max_disk_gb", 100.0)

            lc_metadata = LMCacheMetadata(
                model_name="vllm_omni_diffusion",
                world_size=1, local_world_size=1,
                worker_id=0, local_worker_id=0,
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
                    local_cpu=True, max_local_cpu_size=cpu_gb,
                    local_disk=steps_dir, max_local_disk_size=disk_gb,
                    save_decode_cache=True,
                ),
                metadata=lc_metadata,
                encoder_dtype=_torch.float32,
            )
            self._lmcache_steps_engine = self._lmcache_engine
            logger.info(
                "LMCache ECCacheEngine initialized for step latents: "
                "steps_dir=%s, cpu_gb=%.1f",
                steps_dir, cpu_gb,
            )
        else:
            self._lmcache_engine = None
            self._lmcache_steps_engine = None

        # Final latent directory (torch.save direct to disk, bypasses LMCache)
        final_disk_dir = None
        if lmcache_disk_dir:
            final_disk_dir = os.path.join(lmcache_disk_dir, "final_direct")
            os.makedirs(final_disk_dir, exist_ok=True)

        self._cache_store = DiTCacheStore(
            max_entries=max_entries,
            max_memory_gb=max_memory_gb,
            lmcache_engine=self._lmcache_engine,
            lmcache_steps_engine=self._lmcache_steps_engine,
            max_stored_steps=getattr(config, "inter_request_max_stored_steps", 0),
            final_disk_dir=final_disk_dir,
        )
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
        self._recorder = StepLatentsRecorder()
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
            "InterRequestCacheBackend enabled on pipeline %s",
            pipeline.__class__.__name__,
        )

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

    def update_image_embedding(self, cache_key_hash: str | None, image_tensor: torch.Tensor) -> None:
        if cache_key_hash is None:
            return
        if getattr(self, "_use_fgclip", False) or self._full_clip_model is None:
            return
        # Force CPU for image embedding to avoid NPU async issues in daemon threads
        image_emb = self._encode_image_cpu(image_tensor)
        logger.info("UPDATE_IMG: encode_image result is None=%s", image_emb is None)
        if image_emb is not None:
            self._cache_store.update_image_embedding(cache_key_hash, image_emb)
            logger.info("UPDATE_IMG: stored image embedding for %s", cache_key_hash[:8])

    def semantic_lookup(
        self, req: Any, target_device: torch.device | str | None = None
    ) -> tuple[torch.Tensor | None, list[StepLatentData] | None, float, str | None, str | None]:
        if not self.enabled or self._clip_model is None:
            return None, None, 0.0, None, None

        cache_key = build_cache_key_from_request(req, self._pipeline)
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
            "InterRequestCacheBackend shutdown: persistent_cache_dir=%s, "
            "lmcache_enabled=%s, cache_size=%d",
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

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True, **kwargs: Any) -> None:
        pass

    def before_forward(self, is_dummy: bool = False) -> None:
        if self._recorder is not None and not is_dummy:
            self._recorder.clear()
            self._recorder.enable()

    def after_forward(self, is_dummy: bool = False) -> None:
        """Reset the step-latents recorder after each forward pass."""
        if self._recorder is None or is_dummy:
            return
        self._recorder.disable()
        self._recorder.clear()

    def lookup(self, req: Any, target_device: torch.device | str | None = None) -> torch.Tensor | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline)
        if cache_key is None:
            return None

        return self._cache_store.get(cache_key, target_device=target_device)

    def lookup_step_latents(
        self, req: Any, target_device: torch.device | str | None = None
    ) -> list[StepLatentData] | None:
        if not self.enabled or self._pipeline is None:
            return None

        cache_key = build_cache_key_from_request(req, self._pipeline)
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

        cache_key = build_cache_key_from_request(req, self._pipeline)
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
