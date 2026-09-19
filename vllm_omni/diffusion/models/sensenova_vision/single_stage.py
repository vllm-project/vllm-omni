# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage prompt, cache, and text-decode support for SenseNova-Vision."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import Any

import numpy as np
import PIL.Image
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
from vllm_omni.diffusion.models.sensenova_vision.transforms_sensenova_vision import max_long_edge_resize
from vllm_omni.model_executor.models.sensenova_vision.cfg_expand import IMG2IMG_PLACEHOLDER


class SenseNovaVisionSingleStageMixin:
    """Upstream-compatible local prefill for the one-stage SenseNova topology.

    Public prompts retain the two-stage vLLM transport format.  This mixin
    converts that format into the raw text/image terms expected by upstream
    ``InterleaveInferencer`` before constructing local BAGEL KV caches.
    """

    def _single_stage_autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        )

    def _single_stage_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def _new_single_stage_context(self) -> dict[str, Any]:
        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

    def _resize_context_image(self, image: PIL.Image.Image, *, num_images: int) -> PIL.Image.Image:
        if num_images > 1:
            return max_long_edge_resize(512, 256, 16)(image)
        return max_long_edge_resize(1024, 512, 16)(image)

    def _context_vit_transform(self, image: PIL.Image.Image, *, num_images: int) -> torch.Tensor:
        transform = max_long_edge_resize(448, 224, 14) if num_images > 1 else max_long_edge_resize(980, 224, 14)
        image = transform(image)
        return torch.from_numpy(np.array(image.convert("RGB"))).float().permute(2, 0, 1) / 127.5 - 1.0

    @staticmethod
    def _single_stage_raw_text(fragment: str) -> str:
        """Strip the standard two-stage transport wrapper from one text span."""
        text = fragment.strip()
        for prefix in ("<|im_start|>user", "<|im_start|>assistant", "<|im_start|>"):
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip()
                break
        for suffix in ("<|im_start|>assistant", "<|im_start|>", "<|im_end|>"):
            if text.endswith(suffix):
                text = text[: -len(suffix)].rstrip()
        if text.endswith("<|im_end|>"):
            text = text[: -len("<|im_end|>")].rstrip()
        return text

    def _single_stage_terms(self, first_prompt: Any) -> tuple[list[str], list[PIL.Image.Image], bool]:
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        data = {} if isinstance(first_prompt, str) else (first_prompt.get("multi_modal_data") or {})
        understanding_images = data.get("image")
        images = understanding_images if understanding_images is not None else (data.get("img2img") or [])
        if not isinstance(images, list):
            images = [images]
        images = [PIL.Image.open(image) if isinstance(image, str) else image for image in images]
        marker = "<|image_pad|>" if understanding_images is not None else IMG2IMG_PLACEHOLDER
        terms = [self._single_stage_raw_text(term) for term in prompt.split(marker)]
        if len(terms) != len(images) + 1:
            terms = [""] * len(images) + ["".join(terms)]
        return terms, images, understanding_images is not None

    def _update_single_stage_text_context(self, context: dict[str, Any], text: str) -> None:
        """Append one raw upstream text term, including its BAGEL BOS/EOS pair."""
        if not text.strip():
            return
        text_input, context["kv_lens"], context["ropes"] = self.bagel.prepare_prompts(
            curr_kvlens=context["kv_lens"],
            curr_rope=context["ropes"],
            prompts=[text.strip()],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        with self._single_stage_autocast():
            context["past_key_values"] = self.bagel.forward_cache_update_text(
                context["past_key_values"], **self._single_stage_to_device(text_input)
            )

    def _prefill_single_stage_image(
        self,
        context: dict[str, Any],
        image: PIL.Image.Image,
        *,
        num_images: int,
        understanding: bool,
    ) -> tuple[int, int]:
        image = self._resize_context_image(image, num_images=num_images)
        if not understanding:
            vae_input, context["kv_lens"], context["ropes"] = self.bagel.prepare_vae_images(
                curr_kvlens=context["kv_lens"],
                curr_rope=context["ropes"],
                images=[image],
                transforms=lambda item: torch.from_numpy(np.array(item.convert("RGB"))).float().permute(2, 0, 1) / 127.5
                - 1.0,
                new_token_ids=self.new_token_ids,
            )
            with self._single_stage_autocast():
                context["past_key_values"] = self.bagel.forward_cache_update_vae(
                    self.vae, context["past_key_values"], **self._single_stage_to_device(vae_input)
                )

        vit_input, context["kv_lens"], context["ropes"] = self.bagel.prepare_vit_images(
            curr_kvlens=context["kv_lens"],
            curr_rope=context["ropes"],
            images=[image],
            transforms=lambda item: self._context_vit_transform(item, num_images=num_images),
            new_token_ids=self.new_token_ids,
        )
        if not understanding:
            for key in ("packed_indexes", "packed_key_value_indexes", "key_values_lens"):
                vit_input.pop(key, None)
        with self._single_stage_autocast():
            context["past_key_values"] = self.bagel.forward_cache_update_vit(
                context["past_key_values"], **self._single_stage_to_device(vit_input)
            )
        return image.size[::-1]

    def _prepare_single_stage_contexts(
        self,
        first_prompt: Any,
        sampling: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[int, int]]:
        """Build upstream-style positive, text-CFG, and image-CFG contexts."""
        terms, images, understanding = self._single_stage_terms(first_prompt)
        gen_context = self._new_single_stage_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)
        image_shape = (int(self.bagel.max_latent_size * self.bagel.latent_downsample),) * 2

        for index, text in enumerate(terms):
            if text:
                cfg_text_context = deepcopy(gen_context)
                self._update_single_stage_text_context(gen_context, text)
            if index < len(images):
                image_shape = self._prefill_single_stage_image(
                    gen_context, images[index], num_images=len(images), understanding=understanding
                )
                cfg_text_context = deepcopy(gen_context)

        negative_prompt = first_prompt.get("negative_prompt") if isinstance(first_prompt, dict) else None
        if negative_prompt is None:
            negative_prompt = (sampling.extra_args or {}).get("negative_prompt", "")
        self._update_single_stage_text_context(cfg_text_context, negative_prompt)
        for text in terms:
            self._update_single_stage_text_context(cfg_img_context, text)
        return gen_context, cfg_text_context, cfg_img_context, image_shape

    def _decode_single_stage_text(self, gen_context: dict[str, Any], sampling: Any) -> str:
        """Decode on a copied cache, exactly like upstream ``gen_text``."""
        extra_args = getattr(sampling, "extra_args", None) or {}
        decode_context = deepcopy(gen_context)
        with self._single_stage_autocast():
            start_input = self.bagel.prepare_start_tokens(
                decode_context["kv_lens"], decode_context["ropes"], self.new_token_ids
            )
            token_ids = self.bagel.generate_text(
                past_key_values=decode_context["past_key_values"],
                max_length=int(extra_args.get("max_think_tokens", 500)),
                do_sample=bool(extra_args.get("do_sample", False)),
                temperature=float(extra_args.get("text_temperature", 0.3)),
                end_token_id=self.new_token_ids["eos_token_id"],
                **self._single_stage_to_device(start_input),
            )
        text = self.tokenizer.decode(token_ids[:, 0].tolist()).split("<|im_end|>")[0]
        return text.split("<|im_start|>")[-1]

    def _forward_single(self, first_prompt: Any, sampling: Any, *, prepare_only: bool = False):
        """Run the single-stage text/image split, then delegate DiT to BAGEL."""
        if sampling.past_key_values is not None:
            return super()._forward_single(first_prompt, sampling, prepare_only=prepare_only)
        gen_context, cfg_text_context, cfg_img_context, image_shape = self._prepare_single_stage_contexts(
            first_prompt, sampling
        )
        modalities = first_prompt.get("modalities", []) if isinstance(first_prompt, dict) else []
        if "text" in modalities:
            if prepare_only:
                raise NotImplementedError("SenseNovaVision text output is not supported by step execution.")
            text = self._decode_single_stage_text(gen_context, sampling)
            return DiffusionOutput(output={"payload": {"text": text}, "metadata": {"text": {"text_output": text}}})

        extra_args = getattr(sampling, "extra_args", None) or {}
        if extra_args.get("think"):
            text = self._decode_single_stage_text(gen_context, sampling)
            if text:
                self._update_single_stage_text_context(gen_context, text)
                sampling.extra_args = dict(extra_args)
                sampling.extra_args["text_output"] = text

        local_sampling = copy(sampling)
        local_sampling.past_key_values = gen_context["past_key_values"]
        local_sampling.kv_metadata = {"ropes": gen_context["ropes"], "image_shape": image_shape}
        for name, context in (("cfg_text", cfg_text_context), ("cfg_img", cfg_img_context)):
            if context["past_key_values"].seq_lens:
                setattr(local_sampling, f"{name}_past_key_values", context["past_key_values"])
                setattr(local_sampling, f"{name}_kv_metadata", {"ropes": context["ropes"]})
            else:
                setattr(local_sampling, f"{name}_past_key_values", None)
                setattr(local_sampling, f"{name}_kv_metadata", None)
        return super()._forward_single(first_prompt, local_sampling, prepare_only=prepare_only)
