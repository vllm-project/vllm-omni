"""LongCat-Next image decoder stage (visual codes -> RGB image).

Instantiates the checkpoint's remote-code ``LongcatNextVisualTokenizer`` and
drives its decode path: RQ codebook dequantisation -> 32-layer
VisionTransformerDecoder -> flow-matching refiner (DiT + VAE). Only the
``model.visual_tokenizer.*`` subtree of the sharded checkpoint is loaded
(shards 7/8/15), plus ``image_decoder/image_decoder.safetensors`` which the
remote code lazily pulls in on first decode.
"""

import json
import os
import tempfile
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
        self.has_preprocess = False
        self.has_postprocess = False
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
                f"Image decoder weights not found at {vdc.weight_path}; the checkpoint download may be incomplete."
            )

        tokenizer_cls = get_remote_attr(self.model_path, "modular_longcat_next_visual", "LongcatNextVisualTokenizer")
        self.visual_tokenizer = tokenizer_cls(self.hf_config)
        self._weights_loaded = False

        default_h = _DEFAULT_TOKEN_HW
        default_w = _DEFAULT_TOKEN_HW
        gen_cfg_path = os.path.join(self.model_path, "generation_config.json")
        if os.path.isfile(gen_cfg_path):
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
        self.visual_tokenizer.eval()
        self._weights_loaded = True

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
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

        model_intermediate_buffer = (
            kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or {}
        )
        if isinstance(model_intermediate_buffer, dict):
            info_dicts = [info for info in model_intermediate_buffer.values() if isinstance(info, dict)]
        else:
            info_dicts = [info for info in model_intermediate_buffer if isinstance(info, dict)]
        if len(info_dicts) > 1:
            logger.warning(
                "LongcatNextImageDecoder got %d requests in one batch; only the "
                "first is decoded (max_num_seqs should be 1 for this stage).",
                len(info_dicts),
            )
        additional_info = info_dicts[0] if info_dicts else {}
        visual_codes = additional_info.get("visual_token_ids")
        if not visual_codes:
            logger.warning("No visual token IDs provided for image decoder")
            return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

        token_h = int(additional_info.get("token_h") or self.default_token_h)
        token_w = int(additional_info.get("token_w") or self.default_token_w)

        codes = torch.as_tensor(visual_codes, dtype=torch.long, device=self.device)
        if codes.ndim == 1:
            codes = codes.reshape(-1, NUM_CODEBOOKS)

        # The reference's own GEN_IMAGE_STAGE (output_processor.py:204-216)
        # reads token_h but never uses it -- only token_w gates per-row
        # IMAGE_NEWLINE forcing, so there is no analogous forced-termination
        # transition once token_h rows are complete (unlike audio's max_gen
        # cap). A thinker call with a loose max_tokens keeps sampling frames
        # past the intended grid until the token budget runs out or the
        # model happens to naturally sample IMAGE_END (observed NOT to
        # happen reliably -- job 15032548 produced 51 frames for a
        # requested 4x4=16 grid). VisionTransformerDecoder.forward's
        # positions_2d assert requires exactly token_h*s * token_w*s
        # positions, so passing every kept frame through unconditionally
        # crashes decoding (`AssertionError: positions_2d != L`) the moment
        # generation overruns. The intended image is always the *first*
        # token_h*token_w kept frames.
        expected_positions = token_h * token_w
        if codes.shape[0] > expected_positions:
            logger.warning(
                "Image decoder got %d code frames, expected token_h*token_w=%d "
                "-- generation ran past the intended grid; truncating to the "
                "first %d frames",
                codes.shape[0],
                expected_positions,
                expected_positions,
            )
            codes = codes[:expected_positions]
        elif codes.shape[0] < expected_positions:
            # The mirror-image failure: fewer real pixel frames than the
            # grid needs (e.g. the thinker's max_tokens cut generation short
            # before <longcat_img_end>, or any other frame-accounting bug).
            # VisionTransformerDecoder.forward's positions_2d assert requires
            # EXACTLY token_h*s * token_w*s positions -- handing it a short
            # `codes` crashes the assert, which kills this stage's whole GPU
            # worker process (and cascades into the scheduler losing track
            # of the request). Fail this one request cleanly instead: no
            # image is recoverable from a short, incomplete grid, so return
            # empty output rather than let the process die.
            logger.warning(
                "Image decoder got %d code frames, expected token_h*token_w=%d "
                "-- generation was cut short before the grid completed "
                "(e.g. max_tokens exhausted before <longcat_img_end>); "
                "skipping image output for this request instead of crashing "
                "the reference decoder.",
                codes.shape[0],
                expected_positions,
            )
            return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

        self._ensure_weights()

        # lazy_decode_and_save indexes each level's codebook directly
        # (embed[data[..., idx]] in modular_longcat_next_visual.py's
        # LongcatNextVisualTokenizer.lazy_decode_and_save) -- it wants raw
        # per-level codebook indices (0..codebook_size-1), not global-vocab
        # offset ids. Adding visual_offset_vals here (as an earlier version
        # of this code did) pushes indices past each level's embedding table
        # size and triggers an out-of-bounds device-side assert.
        with tempfile.TemporaryDirectory(prefix="longcat_imgdec_") as tmp_dir:
            save_prefix = os.path.join(tmp_dir, "out")
            with torch.inference_mode():
                image_paths = self.visual_tokenizer.lazy_decode_and_save(
                    codes, token_h, token_w, f"{save_prefix}_0.png"
                )
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
