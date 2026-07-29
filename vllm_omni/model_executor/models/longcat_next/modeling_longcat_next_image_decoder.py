"""LongCat-Next image decoder stage (visual codes -> RGB image).

Instantiates the checkpoint's remote-code ``LongcatNextVisualTokenizer`` and
drives its decode path: RQ codebook dequantisation -> 32-layer
VisionTransformerDecoder -> flow-matching refiner (DiT + VAE). Only the
``model.visual_tokenizer.*`` subtree of the sharded checkpoint is loaded
(shards 7/8/15), plus ``image_decoder/image_decoder.safetensors`` which the
remote code lazily pulls in on first decode.
"""

import os
import tempfile
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .longcat_next_utils import (
    NUM_CODEBOOKS,
    get_remote_attr,
    load_remote_hf_config,
    load_weight_subtree,
    resolve_checkpoint_relative_path,
)

logger = init_logger(__name__)

_DEFAULT_TOKEN_HW = 37  # generation_config.json visual custom_params


class LongcatNextImageDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True
        self.prefix = prefix

        self.model_path: str = vllm_config.model_config.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # The vllm-omni config shim lacks the nested visual configs; use the
        # checkpoint's own remote config and resolve its weight-path
        # placeholders against the local model directory (never editing the
        # checkpoint itself).
        self.hf_config = load_remote_hf_config(self.model_path)
        vdc = self.hf_config.visual_config.visual_decoder_config
        vdc.weight_path = resolve_checkpoint_relative_path(vdc.weight_path, self.model_path)
        if not os.path.isfile(vdc.weight_path):
            raise FileNotFoundError(
                f"Image decoder weights not found at {vdc.weight_path}; "
                "the checkpoint download may be incomplete."
            )

        tokenizer_cls = get_remote_attr(
            self.model_path, "modular_longcat_next_visual", "LongcatNextVisualTokenizer"
        )
        self.visual_tokenizer = tokenizer_cls(self.hf_config)
        self._weights_loaded = False

        default_h = _DEFAULT_TOKEN_HW
        default_w = _DEFAULT_TOKEN_HW
        gen_cfg_path = os.path.join(self.model_path, "generation_config.json")
        if os.path.isfile(gen_cfg_path):
            import json

            with open(gen_cfg_path) as f:
                custom = json.load(f).get("visual_generation_config", {}).get("custom_params", {})
            default_h = int(custom.get("token_h", default_h))
            default_w = int(custom.get("token_w", default_w))
        self.default_token_h = default_h
        self.default_token_w = default_w

    def _ensure_weights(self) -> None:
        if self._weights_loaded:
            return
        logger.info("Loading model.visual_tokenizer.* weights from %s", self.model_path)
        load_weight_subtree(
            self.visual_tokenizer,
            self.model_path,
            "model.visual_tokenizer",
            dtype=self.dtype,
        )
        self.visual_tokenizer.to(device=self.device, dtype=self.dtype)
        self.visual_tokenizer.to(device=self.device, dtype=self.dtype)
        self.visual_tokenizer.eval()
        self._weights_loaded = True

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> None:
        return None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds

        additional_info = kwargs.get("additional_information") or {}
        visual_codes = additional_info.get("visual_token_ids") or kwargs.get("visual_token_ids")
        if not visual_codes:
            logger.warning("No visual token IDs provided for image decoder")
            return OmniOutput(text_hidden_states=None, multimodal_outputs={})

        token_h = int(additional_info.get("token_h") or self.default_token_h)
        token_w = int(additional_info.get("token_w") or self.default_token_w)

        codes = torch.as_tensor(visual_codes, dtype=torch.long, device=self.device)
        if codes.ndim == 1:
            codes = codes.reshape(-1, NUM_CODEBOOKS)

        self._ensure_weights()

        # lazy_decode_and_save expects offset-carrying ids (it subtracts
        # visual_offset_vals itself); the stage input processor hands us raw
        # codebook indices, so re-apply the offsets here.
        offset_vals = torch.cumsum(
            torch.tensor(
                [self.hf_config.visual_offset]
                + list(self.hf_config.visual_config.vq_config.codebook_sizes[:-1]),
                dtype=torch.long,
                device=self.device,
            ),
            dim=0,
        )
        ids_with_offsets = codes + offset_vals

        with tempfile.TemporaryDirectory(prefix="longcat_imgdec_") as tmp_dir:
            save_prefix = os.path.join(tmp_dir, "out")
            with torch.inference_mode():
                image_paths = self.visual_tokenizer.lazy_decode_and_save(
                    ids_with_offsets, token_h, token_w, f"{save_prefix}_0.png"
                )
            from PIL import Image

            pil = Image.open(image_paths[0]).convert("RGB")
            image = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": image.unsqueeze(0),
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # This stage loads its own weight subtree (model.visual_tokenizer.* +
        # image_decoder.safetensors) lazily on first decode; the engine-side
        # loader has nothing to place here.
        consumed = {name for name, _ in weights}
        return consumed | {name for name, _ in self.named_parameters()}
