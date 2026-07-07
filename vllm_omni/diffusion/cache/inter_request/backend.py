from __future__ import annotations

import logging
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
    intermediate latents during denoising.  These are stored in the in-memory
    cache alongside the final latent.  When ``record_step_latents`` is enabled,
    the step latents are additionally saved to disk.

    Usage:
        omni = Omni(
            model="Qwen/Qwen-Image",
            cache_backend="inter_request",
            cache_config={
                "inter_request_max_entries": 100,
                "inter_request_max_memory_gb": 4.0,
                "inter_request_record_step_latents": True,
                "inter_request_step_latents_dir": "./step_latents",
            }
        )
    """

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        max_entries = getattr(config, "inter_request_max_entries", 100)
        max_memory_gb = getattr(config, "inter_request_max_memory_gb", 4.0)
        self._cache_store = DiTCacheStore(
            max_entries=max_entries,
            max_memory_gb=max_memory_gb,
        )
        self._pipeline = None

        self._record_step_latents = getattr(config, "inter_request_record_step_latents", False)
        self._step_latents_dir = getattr(config, "inter_request_step_latents_dir", "./step_latents")
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
            "max_entries=%d, max_memory_gb=%.1f, record_step_latents=%s, "
            "persistent_cache_dir=%s, clip_model_path=%s, "
            "clip_threshold=%.2f, clip_min_skip=%d, clip_max_skip_ratio=%.2f",
            max_entries,
            max_memory_gb,
            self._record_step_latents,
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

        if self._persistent_cache_dir is not None:
            loaded = self._cache_store.load_from_disk(self._persistent_cache_dir)
            if loaded > 0:
                logger.info(
                    "Loaded %d cache entries from persistent storage %s",
                    loaded,
                    self._persistent_cache_dir,
                )

        logger.info(
            "InterRequestCacheBackend enabled on pipeline %s (save_step_latents_to_disk=%s)",
            pipeline.__class__.__name__,
            self._record_step_latents,
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
                feat = text_features.squeeze(0)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                return feat
        except Exception as e:
            logger.warning("CLIP encoding failed: %s", e)
            return None

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor | None:
        # Video tensors are 5D [B, C, T, H, W] — skip image embedding for video
        if image_tensor.dim() == 5:
            return None
        if getattr(self, "_use_fgclip", False) or self._full_clip_model is None:
            return None
        if self._clip_image_processor is None:
            return None
        try:
            from PIL import Image

            img = image_tensor.float().cpu()
            if img.dim() == 4:
                img = img[0]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).clamp(0, 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img)
            inputs = self._clip_image_processor(images=[pil_image], return_tensors="pt").to(self._clip_device)
            with torch.no_grad():
                image_features = self._full_clip_model.get_image_features(**inputs)
            feat = image_features.squeeze(0)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat
        except Exception as e:
            logger.warning("CLIP image encoding failed: %s", e)
            return None

    def update_image_embedding(self, cache_key_hash: str | None, image_tensor: torch.Tensor) -> None:
        if cache_key_hash is None:
            return
        if getattr(self, "_use_fgclip", False) or self._full_clip_model is None:
            return
        image_emb = self.encode_image(image_tensor)
        if image_emb is not None:
            self._cache_store.update_image_embedding(cache_key_hash, image_emb)

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
        return skip

    @property
    def clip_enabled(self) -> bool:
        return self._clip_model is not None

    def shutdown(self) -> None:
        logger.info(
            "InterRequestCacheBackend shutdown: persistent_cache_dir=%s, cache_size=%d",
            self._persistent_cache_dir,
            self._cache_store.size,
        )
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
