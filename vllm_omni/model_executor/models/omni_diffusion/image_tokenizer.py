from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torchvision import transforms
from vllm.logger import init_logger

from vllm_omni.model_executor.models.omni_diffusion.third_party.magvit import MAGVITv2
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OmniDiffusionModelSpecialTokens,
    OmniDiffusionTokenizerBaseData,
    get_single_token_ids,
)

logger = init_logger(__name__)

_MAGVIT_IMAGE_TOKEN_COUNT = 256
_MAGVIT_MIN_CODEBOOK_ID = 0
_MAGVIT_MAX_CODEBOOK_ID = 8192 - 1


class OmniDiffusionImageTokenizer:
    """Encode images to Omni-Diffusion MagVIT tokens and decode tokens back to images."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        image_size: int = 512,
    ) -> None:
        logger.info(
            "Initializing OmniDiffusionImageTokenizer with model_path=%s, device=%s, image_size=%d.",
            model_path,
            device,
            image_size,
        )
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=True,
                ),
            ]
        )
        self.tokenizer = self._load_model(model_path, device)
        logger.info("Initialized OmniDiffusionImageTokenizer successfully.")

    def encode(self, images: Tensor) -> Tensor:
        """Convert CHW/BCHW image tensors into MagVIT codebook IDs."""
        # Match the tokenizer weights dtype. The MagVIT module may be loaded in
        # bfloat16 by the model runtime, while torchvision preprocessing returns
        # float32 tensors.
        images = self._preprocess(images).to(
            device=self.device,
            dtype=self.tokenizer.dtype,
        )
        return self.tokenizer.get_code(images)

    def decode(self, image_tokens: Tensor) -> Tensor:
        """Decode MagVIT codebook IDs into image tensors in [0, 1]."""
        match image_tokens.ndim:
            # [B, T]
            case 2:
                pass
            # [T] -> [B, T]
            case 1:
                image_tokens = image_tokens.unsqueeze(0)
            case _:
                raise ValueError(
                    f"Expected image token tensor with shape [T] or [B, T], got {tuple(image_tokens.shape)}."
                )
        # Empty token sequences cannot be decoded into an image.
        if image_tokens.shape[1] == 0:
            raise ValueError("Cannot decode an empty image token sequence.")

        # MagVIT expects exactly one fixed-size token grid per image. If the
        # model produced fewer tokens, match the official script and repeat the
        # final token until the grid is full.
        if image_tokens.shape[1] < _MAGVIT_IMAGE_TOKEN_COUNT:
            padding = image_tokens[:, -1:].repeat(1, _MAGVIT_IMAGE_TOKEN_COUNT - image_tokens.shape[1])
            image_tokens = torch.cat([image_tokens, padding], dim=1)

        # Keep exactly one image token grid and clamp generated IDs into the
        # MagVIT codebook range before looking up embeddings.
        image_tokens = image_tokens[:, :_MAGVIT_IMAGE_TOKEN_COUNT].clamp(
            min=_MAGVIT_MIN_CODEBOOK_ID,
            max=_MAGVIT_MAX_CODEBOOK_ID,
        )
        images = self.tokenizer.decode_code(image_tokens.to(self.device))
        # MagVIT decodes to the training range [-1, 1]; the API path expects
        # normalized image tensors in [0, 1].
        return images.add(1).div(2).clamp(0, 1)

    def prepare_image_token_inputs(
        self,
        input_ids: Sequence[int],
        images: Any,
        tokenizer: Any,
        tokenizer_base_data: OmniDiffusionTokenizerBaseData,
    ) -> list[int]:
        """Replace image placeholder tags with Omni-Diffusion MagVIT token IDs."""
        if isinstance(images, torch.Tensor):
            match images.ndim:
                # [C, H, W]
                case 3:
                    image_tensors = [images]
                # [N, C, H, W]
                case 4:
                    image_tensors = list(images)
                case _:
                    raise ValueError(
                        f"Expected images tensor with shape [C, H, W] or [N, C, H, W], got {tuple(images.shape)}."
                    )
        elif isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
            image_tensors = list(images)
            if not all(isinstance(item, torch.Tensor) for item in image_tensors):
                raise TypeError("Expected every image item to be a torch.Tensor.")
        else:
            raise TypeError(f"Expected images to be a tensor or sequence of tensors, got {type(images)!r}.")

        img_tag_id = tokenizer_base_data.get_token_id(OmniDiffusionModelSpecialTokens.IMG_TAG)
        img_positions = [idx for idx, token_id in enumerate(input_ids) if token_id == img_tag_id]
        if len(image_tensors) != len(img_positions):
            raise ValueError(
                f"Expected {len(img_positions)} image tensors to match prompt placeholders, got {len(image_tensors)}."
            )

        new_input_ids: list[int] = []
        start = 0
        for image_idx, img_pos in enumerate(img_positions):
            # Each <|image|> placeholder expands to:
            # <|begin_of_image|><|image_...|>...<|end_of_image|>.
            image_codes = self.encode(image_tensors[image_idx])
            if image_codes.ndim == 2 and image_codes.shape[0] == 1:
                image_codes = image_codes[0]
            if image_codes.ndim != 1:
                raise ValueError(f"Expected one-dimensional image codes, got shape={tuple(image_codes.shape)}.")

            replacement_tokens = [
                OmniDiffusionModelSpecialTokens.IMG_START.value,
                *(f"<|image_{int(image_code)}|>" for image_code in image_codes.tolist()),
                OmniDiffusionModelSpecialTokens.IMG_END.value,
            ]
            replacement_ids = get_single_token_ids(tokenizer, replacement_tokens)

            new_input_ids += input_ids[start:img_pos]
            new_input_ids += replacement_ids
            start = img_pos + 1

        new_input_ids += input_ids[start:]
        return new_input_ids

    def _load_model(self, model_path: str, device: torch.device | str) -> MAGVITv2:
        """Load the MagVIT image tokenizer used by Omni-Diffusion."""
        logger.info("Loading Omni-Diffusion image tokenizer from %s onto %s.", model_path, device)
        tokenizer: MAGVITv2 = MAGVITv2.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
        )
        tokenizer.to(device)
        tokenizer.eval()
        tokenizer.requires_grad_(False)
        logger.info("Loaded Omni-Diffusion image tokenizer successfully.")
        return tokenizer

    def _preprocess(self, images: Tensor) -> Tensor:
        """Preprocess images to prepare them for the Omni-Diffusion MagVIT tokenizer."""
        match images.ndim:
            # [B, C, H, W]
            case 4:
                pass
            # [C, H, W] -> [B, C, H, W]
            case 3:
                images = images.unsqueeze(0)
            case _:
                raise ValueError(f"Expected CHW or BCHW image tensor, got shape={tuple(images.shape)}.")

        if images.shape[1] != 3:
            raise ValueError(f"Expected RGB image tensor with 3 channels, got shape={tuple(images.shape)}.")

        # Match the official preprocessing path: PIL conversion, bicubic resize,
        # center crop, ToTensor, and normalization to [-1, 1].
        return torch.stack([self.transform(image.cpu()) for image in images]).contiguous()
