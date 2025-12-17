# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena
"""
import ast
import base64
import io
import json
import logging
import math
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import suppress
from typing import Any, cast, Dict

import cv2
import numpy as np
import torch
import torchaudio
from PIL import Image
from transformers import PreTrainedTokenizerBase
from vllm.benchmark.datasets import (RandomDataset, ShareGPTDataset, SpecBench,
                                     SonnetDataset, BurstGPTDataset, ConversationDataset,
                                     VisionArenaDataset, MMVUDataset, InstructCoderDataset, MTBenchDataset,
                                     BlazeditDataset, AIMODataset, NextEditPredictionDataset, ASRDataset, MLPerfDataset,
                                     PrefixRepetitionRandomDataset, CustomDataset, SampleRequest, _ValidateDatasetArgs,
                                     process_image)
from vllm.utils import PlaceholderModule

try:
    from datasets import load_dataset
except ImportError:
    datasets = PlaceholderModule("datasets")
    load_dataset = datasets.placeholder_attr("load_dataset")

try:
    import pandas as pd
except ImportError:
    pd = PlaceholderModule("pandas")

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

logger = logging.getLogger(__name__)


def process_video(video: Any) -> Mapping[str, Any]:
    """
    Process a single video input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw video bytes: - Expects a dict with a 'bytes' key
       containing raw video data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the image URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(video, dict) and 'bytes' in video:
        video_bytes = video['bytes']
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        return {
            "type": "video_url",
            "video_url": {
                "url": f"data:video/mp4;base64,{video_base64}"
            },
        }

    if isinstance(video, str):
        video_url = (video if video.startswith(
            ("http://", "https://", "file://")) else f"file://{video}")
        return {"type": "video_url", "video_url": {"url": video_url}}

    raise ValueError(
        f"Invalid video input {video}. Must be a string of local path/remote url, or a dictionary with raw video bytes in the form of `{{'bytes': raw_video_bytes}}`."  # noqa: E501
    )


def process_audio(audio: Any) -> Mapping[str, Any]:
    """
    Process a single audio input and return a multimedia content dictionary.

    Supports the following input types:

    1. Dictionary with raw audio bytes: - Expects a dict with a 'bytes' key
       containing raw audio data.

    2. String input: - Treats the string as a URL or local file path.  -
       Prepends "file://" if the string doesn't start with "http://" or
       "file://".  - Returns a dictionary with the audio URL.

    Raises:
        ValueError: If the input is not a supported type.
    """
    if isinstance(audio, dict) and 'bytes' in audio:
        audio_bytes = audio['bytes']
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/mpeg;base64,{audio_base64}"
            },
        }
    if isinstance(audio, str):
        audio_url = (audio if audio.startswith(
            ("http://", "https://", "file://")) else f"file://{audio}")
        return {"type": "audio_url", "audio_url": {"url": audio_url}}

    raise ValueError(f"Invalid audio input {audio}. Must be a string of local path/remote url, or a dictionary with raw audio bytes in the form of `{{'bytes': raw_audio_bytes}}`."
   )



# -----------------------------------------------------------------------------
# MultiModalDataset Implementation
# -----------------------------------------------------------------------------

class RandomMultiModalDataset(RandomDataset):
    """
    Synthetic multimodal dataset (text + images) that extends RandomDataset.

    Status:
    - Images: supported via synthetic RGB data.
    - Video: supported via synthetic bytes data.
    - Audio: supported via synthetic bytes data.

    Sampling overview:
    1) Number of items per request is sampled uniformly from the integer range
       [floor(n·(1−r)), ceil(n·(1+r))], where n is the base count and r is
       `num_mm_items_range_ratio` in [0, 1]. r=0 keeps it fixed; r=1 allows 0.
       The maximum is further clamped to the sum of per-modality limits.
    2) Each item’s modality and shape is sampled from `bucket_config`, a dict
       mapping (height, width, num_frames) → probability. We treat
       `num_frames`=1 as image and and `num_frames` > 1 as video.
       Entries with zero probability are removed and the rest are renormalized
       to sum to 1.
    3) Per-modality hard caps are enforced via `limit_mm_per_prompt`.
       When a modality reaches its cap, all of its buckets are excluded and the
       remaining probabilities are renormalized.

    Example bucket configuration:
    {(256, 256, 1): 0.5, (720, 1280, 1): 0.4, (720, 1280, 16): 0.1}
      - Two image buckets (`num_frames`=1) and one video bucket
      (`num_frames`=16).
    OBS.: Only image sampling is supported for now.
    """

    IS_MULTIMODAL = True
    # NOTE: video sampling is WIP. Setting it to 0.
    DEFAULT_LIMIT_MM_PER_PROMPT = {"image": 255, "video": 0}

    DEFAULT_BASE_ITEMS_PER_REQUEST = 1
    DEFAULT_NUM_MM_ITEMS_RANGE_RATIO = 0.0
    DEFAULT_MM_ITEM_BUCKET_CONFIG = {
        (256, 256, 1): 0.5,
        (720, 1280, 1): 0.5,
        (720, 1280, 16): 0.0,
    }
    DEFAULT_ENABLE_MULTIMODAL_CHAT = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    def generate_synthetic_image(self, width: int, height: int) -> Image.Image:
        """Generate synthetic PIL image with random RGB values.

        NOTE: iid pixel sampling results in worst-case compression
        (good for stressing I/O), but very unlike real photos.
        We could consider a “low-freq” mode (e.g., noise blur)
        to emulate network realism instead of max stress.
        """
        random_pixels = self._rng.integers(
            0,
            256,
            (height, width, 3),
            dtype=np.uint8,
        )
        return Image.fromarray(random_pixels)

    def generate_synthetic_video(self, width: int,
                                    height: int,
                                    num_frames: int) -> Any:
        """Generate synthetic video with random values.
        """
        video_data = self._rng.integers(
            0, 256,
            (num_frames, height, width, 3),
            dtype=np.uint8,
        )
        video_tensor = torch.from_numpy(video_data)
        with tempfile.NamedTemporaryFile(suffix=f".mp4", delete=False) as tmp:
            temp_path = tmp.name
        frames, height, width, channels = video_tensor.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 30, (width, height))

        for i in range(frames):
            frame = video_tensor[i].numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        out.release()

        with open(temp_path, 'rb') as f:
            video_bytes = f.read()

        os.unlink(temp_path)

        return {
            'bytes': video_bytes,
        }


    def generate_synthetic_audio(
        self,
        duration: int,  # seconds
        num_channels: int #1：Mono，2：Stereo 5：5.1 surround sound
    ) -> Dict[str, Any]:
        """Generate synthetic audio with random values.
           Default use 48000Hz.
        """
        sample_rate = 48000
        num_samples = int(sample_rate * duration)
        audio_data = self._rng.uniform(
            -0.5, 0.5,
            (num_samples, num_channels)
        )
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_tensor = torch.FloatTensor(audio_data.T)
        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate,
            format="mp3"
        )
        buffer.seek(0)
        audio_bytes = buffer.read()
        return {
            'bytes': audio_bytes,
        }

    def map_config_to_modality(self, config: tuple[int, int, int]) -> str:
        """Map the configuration to the modality."""
        if config[0] == 0:
            return "audio"
        elif config[-1] == 1:
            return "image"
        elif config[-1] > 1:
            return "video"
        else:
            raise ValueError(f"Invalid multimodal item configuration: {config}")

    def normalize_bucket_config(self, bucket_config: dict[tuple[int, int, int],
                                float]) -> dict[tuple[int, int, int], float]:
        """
        Remove zero probability entries
        and normalize the bucket config to sum to 1.
        """
        # Raise error if value is negative
        if any(v < 0 for v in bucket_config.values()):
            raise ValueError("Bucket config values must be non-negative.")
        # Remove zero probability entries
        bucket_config = {k: v for k, v in bucket_config.items() if v > 0}
        # if bucket config is empty, raise error
        if not bucket_config:
            raise ValueError("Got invalid bucket config. "
                             "Bucket config values must be non-zero.")
        # Normalize the remaining bucket config to sum to 1
        total = sum(bucket_config.values())
        return {k: v / total for k, v in bucket_config.items()}


    def generate_mm_item(self,
                         mm_item_config: tuple[int, int, int],
                         ) -> Mapping[str, Any]:
        """
        Create synthetic images and videos and
        apply process_image/process_video respectively.
        This follows the OpenAI API chat completions
        https://github.com/openai/openai-python
        """

        if self.map_config_to_modality(mm_item_config) == "image":
            return process_image(self.generate_synthetic_image(
                                                            mm_item_config[1],
                                                            mm_item_config[0]))
        elif self.map_config_to_modality(mm_item_config) == "video":
            return process_video(self.generate_synthetic_video(
                                                            mm_item_config[1],
                                                            mm_item_config[0],
                                                            mm_item_config[2]))
        elif self.map_config_to_modality(mm_item_config) == "audio":
            return process_audio(self.generate_synthetic_audio(
                                                            mm_item_config[1],
                                                            mm_item_config[2]))
        else:
            raise ValueError(f"Invalid multimodal item configuration: "
                             f"{mm_item_config}")


    def get_mm_item_sampling_params(
        self,
        base_items_per_request: int,
        num_mm_items_range_ratio: float,
        limit_mm_per_prompt: dict[str, int],
        bucket_config: dict[tuple[int, int, int], float],
    ) -> tuple[int, int, dict[str, int], dict[tuple[int, int, int], float]]:
        """
        Get the sampling parameters for the multimodal items.
        """
        # Enforce num_mm_items_range_ratio <= 1
        if not (0.0 <= num_mm_items_range_ratio <= 1.0):
            raise ValueError("num_mm_items_range_ratio must be in [0, 1].")

        # Ensure modalities to sample are in limit_mm_per_prompt
        for k, v in bucket_config.items():
            # get modality from bucket config
            modality = self.map_config_to_modality(k)
            if modality not in limit_mm_per_prompt:
                raise ValueError(f"Modality {modality} is not in "
                                 f"limit_mm_per_prompt: "
                                 f"{limit_mm_per_prompt.keys()}")

        # Remove zero probability entries
        # and normalize bucket config to sum to 1
        bucket_config = self.normalize_bucket_config(bucket_config)
        logger.info(
            "Normalized bucket config: %s", bucket_config,
        )
        # Only consider limit per prompt for modalities in bucket config
        allowed_modalities = {self.map_config_to_modality(cfg)
                              for cfg in bucket_config}
        limit_mm_per_prompt = {
            k: v for k, v in limit_mm_per_prompt.items()
            if k in allowed_modalities}
        if not limit_mm_per_prompt:
            raise ValueError("No valid limits for modalities present in "
                             "bucket_config.")

        logger.info(
            "Updated mm-limit-per-prompt: %s", limit_mm_per_prompt,
        )

        # Get max and min num mm items and ensure
        # it is at most the sum of limit_mm_per_prompt for all modalities
        max_num_mm_items = min(
            sum(limit_mm_per_prompt.values()),
            math.ceil(base_items_per_request * (1 + num_mm_items_range_ratio))
        )
        # Ensure min num mm items is at least 0
        min_num_mm_items = max(
            0,
            math.floor(base_items_per_request * (1 - num_mm_items_range_ratio))
        )
        # Raise error if min num mm items is greater than max num mm items
        if min_num_mm_items > max_num_mm_items:
            raise ValueError(f"Min num mm items is greater than max mm items: "
                             f"{min_num_mm_items} > {max_num_mm_items}")

        logger.info(
            "Sampling number of multimodal items from [%s, %s]",
            min_num_mm_items, max_num_mm_items,
        )

        return (
            min_num_mm_items,
            max_num_mm_items,
            limit_mm_per_prompt,
            bucket_config,
        )

    def get_mm_item_iterator(
        self,
        min_num_mm_items: int,
        max_num_mm_items: int,
        bucket_config: dict[tuple[int, int, int], float],
        limit_mm_per_prompt: dict[str, int],
    ) -> Iterator[tuple[int,int, int]]:
        """
        Iterator over the multimodal items for each request
        whose size is between min_num_mm_items and max_num_mm_items.

        Loop over the bucket config and sample a multimodal item.
        Loop until the number of multimodal items sampled is equal to
        request_num_mm_items or limit of multimodal items per prompt
        for all modalities is reached.

        Note:
        - This function operates on a per-request shallow copy of
          `bucket_config` (tuple->float). The original dict passed to
          `sample` is not mutated. If this ever changes, a test
          is implemented and will fail.
        """
        # Get the number of multimodal items to sample
        request_num_mm_items = int(
            self._rng.integers(min_num_mm_items, max_num_mm_items + 1)
        )
        # If request_num_mm_items is 0, yield an empty iterator
        if request_num_mm_items == 0:
            return
        # Initialize modality counters
        modality_counter = {self.map_config_to_modality(k): 0
                            for k in bucket_config}
        # Copy the bucket config to avoid modifying the original
        bucket_config_copy = bucket_config.copy()
        # Loop over the number of multimodal items to sample
        while sum(modality_counter.values()) < request_num_mm_items:
            # Sample a multimodal item config
            mm_item_config = self._rng.choice(list(bucket_config_copy.keys()),
                                                p=list(bucket_config_copy.values()))
            modality = self.map_config_to_modality(mm_item_config)
            # Check that modality count is less than limit per prompt
            if modality_counter[modality] < limit_mm_per_prompt[modality]:
                modality_counter[modality] += 1
                yield (
                    mm_item_config
                )
            else:
                # If the counter is greater than the limit per prompt
                # set all multimodal items of this modality to 0
                for k, v in bucket_config_copy.items():
                    if self.map_config_to_modality(k) == modality:
                        bucket_config_copy[k] = 0
                # If all configs are 0, break the loop
                # This should not happen as request_num_mm_items is at most
                # the sum of limit_mm_per_prompt for all modalities
                if all(v == 0 for v in bucket_config_copy.values()):
                    logger.warning("Exhausted all multimodal items "
                                   "of modality %s",
                                   modality)
                    break
                # Renormalize the bucket config
                bucket_config_copy = self.normalize_bucket_config(
                                        bucket_config_copy)


    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
        no_oversample: bool = False,
        prefix_len: int = RandomDataset.DEFAULT_PREFIX_LEN,
        range_ratio: float = RandomDataset.DEFAULT_RANGE_RATIO,
        input_len: int = RandomDataset.DEFAULT_INPUT_LEN,
        output_len: int = RandomDataset.DEFAULT_OUTPUT_LEN,
        limit_mm_per_prompt: dict[str, int] = DEFAULT_LIMIT_MM_PER_PROMPT,
        base_items_per_request: int = DEFAULT_BASE_ITEMS_PER_REQUEST,
        num_mm_items_range_ratio: float = DEFAULT_NUM_MM_ITEMS_RANGE_RATIO,
        bucket_config: dict[tuple[int, int, int], float] =
                                        DEFAULT_MM_ITEM_BUCKET_CONFIG,
        enable_multimodal_chat: bool = DEFAULT_ENABLE_MULTIMODAL_CHAT,
        **kwargs,
    ) -> list[SampleRequest]:

        # Get the sampling parameters for the dataset
        input_lens, output_lens, offsets = self.get_sampling_params(
            num_requests, range_ratio, input_len, output_len, tokenizer
        )

        (
            min_num_mm_items,
            max_num_mm_items,
            limit_mm_per_prompt,
            bucket_config,
        ) = self.get_mm_item_sampling_params(
            base_items_per_request,
            num_mm_items_range_ratio,
            limit_mm_per_prompt,
            bucket_config,
        )

        # Generate prefix once
        prefix_token_ids = self.get_prefix(tokenizer, prefix_len)
        vocab_size = tokenizer.vocab_size
        # Add synthetic multimodal items to each request
        mm_requests = []
        for i in range(num_requests):
            prompt, total_input_len = self.generate_token_sequence(
                tokenizer=tokenizer,
                prefix_token_ids=prefix_token_ids,
                prefix_len=prefix_len,
                vocab_size=vocab_size,
                input_len=int(input_lens[i]),
                offset=int(offsets[i]),
                index=i,
            )
            # Get multimodal item iterator for a given request
            mm_item_iterator = self.get_mm_item_iterator(
                min_num_mm_items,
                max_num_mm_items,
                bucket_config,
                limit_mm_per_prompt,
            )

            mm_content = cast(list[dict[str, Any]], [
                self.generate_mm_item(mm_item_config)
                for mm_item_config in mm_item_iterator
            ])

            if enable_multimodal_chat:
                # NOTE: For now this option is only provided for completeness
                # given that the serve.py benchmark currently does not use it.
                mm_chat_prompt: Any = prompt
                mm_chat_prompt = self.apply_multimodal_chat_transformation(
                    prompt, mm_content)
                sample_request = SampleRequest(
                    prompt=mm_chat_prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    multi_modal_data=None,
                    request_id=request_id_prefix + str(i),
                )
            else:
                sample_request = SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    multi_modal_data=mm_content,
                    request_id=request_id_prefix + str(i),
                )
            mm_requests.append(sample_request)
        return mm_requests



def add_dataset_parser(parser: FlexibleArgumentParser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        action=_ValidateDatasetArgs,
        choices=[
            "sharegpt", "burstgpt", "sonnet", "random", "random-mm", "hf",
            "custom", "prefix_repetition", "spec_bench"
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not load the dataset in streaming mode.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        action=_ValidateDatasetArgs,
        help="Path to the sharegpt/sonnet dataset. "
        "Or the huggingface dataset ID if using HF dataset.",
    )
    parser.add_argument(
        "--no-oversample",
        action="store_true",
        help="Do not oversample if the dataset has " \
        "fewer samples than num-prompts.",
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help=
        "Number of output tokens per request, used only for custom dataset.",
    )
    custom_group.add_argument(
        "--custom-skip-chat-template",
        action="store_true",
        help=
        "Skip applying chat template to prompt, used only for custom dataset.",
    )

    spec_bench_group = parser.add_argument_group("spec bench dataset options")
    spec_bench_group.add_argument(
        "--spec-bench-output-len",
        type=int,
        default=256,
        help=
        "Num of output tokens per request, used only for spec bench dataset.",
    )
    spec_bench_group.add_argument(
        "--spec-bench-category",
        type=str,
        default=None,
        help=
        "Category for spec bench dataset. If None, use all categories.",
    )

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument(
        "--sonnet-input-len",
        type=int,
        default=550,
        help=
        "Number of input tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-output-len",
        type=int,
        default=150,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    sonnet_group.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )

    sharegpt_group = parser.add_argument_group("sharegpt dataset options")
    sharegpt_group.add_argument(
        "--sharegpt-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length "
        "from the ShareGPT dataset.",
    )

    blazedit_group = parser.add_argument_group("blazedit dataset options")
    blazedit_group.add_argument(
        "--blazedit-min-distance",
        type=float,
        default=0.0,
        help=
        "Minimum distance for blazedit dataset. Min: 0, Max: 1.0",
    )
    blazedit_group.add_argument(
        "--blazedit-max-distance",
        type=float,
        default=1.0,
        help=
        "Maximum distance for blazedit dataset. Min: 0, Max: 1.0",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help=
        "Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help=
        "Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=("Number of fixed prefix tokens before the random context "
              "in a request. "
              "The total input length is the sum of `random-prefix-len` and "
              "a random "
              "context length sampled from [input_len * (1 - range_ratio), "
              "input_len * (1 + range_ratio)]."),
    )
    random_group.add_argument(
        "--random-batch-size",
        type=int,
        default=1,
        help=("Batch size for random sampling. "
              "Only used for embeddings benchmark."),
    )

    # random multimodal dataset options
    random_mm_group = parser.add_argument_group(
        "random multimodal dataset options extended from random dataset")
    random_mm_group.add_argument(
        "--random-mm-base-items-per-request",
        type=int,
        default=RandomMultiModalDataset.DEFAULT_BASE_ITEMS_PER_REQUEST,
        help=(
            "Base number of multimodal items per request for random-mm. "
            "Actual per-request count is sampled around this base using "
            "--random-mm-num-mm-items-range-ratio."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-num-mm-items-range-ratio",
        type=float,
        default=RandomMultiModalDataset.DEFAULT_NUM_MM_ITEMS_RANGE_RATIO,
        help=(
            "Range ratio r in [0, 1] for sampling items per request. "
            "We sample uniformly from the closed integer range "
            "[floor(n*(1-r)), ceil(n*(1+r))] "
            "where n is the base items per request. "
            "r=0 keeps it fixed; r=1 allows 0 items. The maximum is clamped "
            "to the sum of per-modality limits from "
            "--random-mm-limit-mm-per-prompt. "
            "An error is raised if the computed min exceeds the max."
        ),
    )
    random_mm_group.add_argument(
        "--random-mm-limit-mm-per-prompt",
        type=json.loads,
        default=RandomMultiModalDataset.DEFAULT_LIMIT_MM_PER_PROMPT,
        help=(
            "Per-modality hard caps for items attached per request, e.g. "
            "'{\"image\": 3, \"video\": 0}'. The sampled per-request item "
            "count is clamped to the sum of these limits. When a modality "
            "reaches its cap, its buckets are excluded and probabilities are "
            "renormalized."
            "OBS.: Only image sampling is supported for now."
        ),
    )

    def _parse_mm_bucket_config(v: object) -> dict[tuple[int, int, int], float]:
        # If already a dict (e.g., programmatic call), normalize keys
        def normalize(d: dict) -> dict[tuple[int, int, int], float]:
            out: dict[tuple[int, int, int], float] = {}
            for k, val in d.items():
                key = k
                if isinstance(key, str):
                    with suppress(Exception):
                        key = ast.literal_eval(key)
                if not (isinstance(key, tuple) and len(key) == 3
                        and all(isinstance(x, int) for x in key)):
                    raise ValueError(
                        f"Invalid bucket key {k!r}. Expected tuple (H, W, T)."
                    )
                out[(int(key[0]), int(key[1]), int(key[2]))] = float(val)
            return out

        if isinstance(v, dict):
            return normalize(v)
        if isinstance(v, str):
            # Python literal (supports tuple keys)
            parsed = ast.literal_eval(v)
            if not isinstance(parsed, dict):
                raise ValueError("Bucket config must parse to a dict.")
            return normalize(parsed)
        raise ValueError("Unsupported value for --random-mm-bucket-config.")

    random_mm_group.add_argument(
        "--random-mm-bucket-config",
        type=_parse_mm_bucket_config,
        default=RandomMultiModalDataset.DEFAULT_MM_ITEM_BUCKET_CONFIG,
        help=(
            "The bucket config is a dictionary mapping a multimodal item"
            "sampling configuration to a probability."
            "Currently allows for 2 modalities: images and videos. "
            "An bucket key is a tuple of (height, width, num_frames)"
            "The value is the probability of sampling that specific item. "
            "Example: "
            "--random-mm-bucket-config "
            "{(256, 256, 1): 0.5, (720, 1280, 1): 0.4, (720, 1280, 16): 0.10} "
            "First item: images with resolution 256x256 w.p. 0.5"
            "Second item: images with resolution 720x1280 w.p. 0.4 "
            "Third item: videos with resolution 720x1280 and 16 frames w.p. 0.1"
            "OBS.: If the probabilities do not sum to 1, they are normalized."
            "OBS bis.: Only image sampling is supported for now."
        ),
    )

    hf_group = parser.add_argument_group("hf dataset options")
    hf_group.add_argument("--hf-subset",
                          type=str,
                          default=None,
                          help="Subset of the HF dataset.")
    hf_group.add_argument("--hf-split",
                          type=str,
                          default=None,
                          help="Split of the HF dataset.")
    hf_group.add_argument(
        "--hf-name",
        type=str,
        default=None,
        help=(
            "Name of the dataset on HuggingFace "
            "(e.g., 'lmarena-ai/VisionArena-Chat'). "
            "Specify this if your dataset-path is a local path."
        ),
    )
    hf_group.add_argument(
        "--hf-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output lengths "
        "from the sampled HF dataset.",
    )

    prefix_repetition_group = parser.add_argument_group(
        "prefix repetition dataset options")
    prefix_repetition_group.add_argument(
        "--prefix-repetition-prefix-len",
        type=int,
        default=256,
        help="Number of prefix tokens per request, used only for prefix "
        "repetition dataset.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-suffix-len",
        type=int,
        default=256,
        help="Number of suffix tokens per request, used only for prefix "
        "repetition dataset. Total input length is prefix_len + suffix_len.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-num-prefixes",
        type=int,
        default=10,
        help="Number of prefixes to generate, used only for prefix repetition "
        "dataset. Prompts per prefix is num_requests // num_prefixes.",
    )
    prefix_repetition_group.add_argument(
        "--prefix-repetition-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for prefix "
        "repetition dataset.",
    )


def get_samples(args, tokenizer) -> list[SampleRequest]:

    if not hasattr(args, "request_id_prefix"):
        args.request_id_prefix = ""

    if args.dataset_name == "custom":
        dataset = CustomDataset(dataset_path=args.dataset_path)
        input_requests = dataset.sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.custom_output_len,
            skip_chat_template=args.custom_skip_chat_template,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
        )

    elif args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        # For the "sonnet" dataset, formatting depends on the backend.
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=False,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            )
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template, (
                "Tokenizer/model must have chat template for sonnet dataset.")
            input_requests = dataset.sample(
                num_requests=args.num_prompts,
                input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len,
                prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer,
                return_prompt_formatted=True,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            )

    elif args.dataset_name == "hf":
        # all following datasets are implemented from the
        # HuggingFaceDataset base class
        hf_kwargs = {}
        if (
            args.dataset_path in VisionArenaDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in VisionArenaDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = VisionArenaDataset
            args.hf_split = "train"
            args.hf_subset = None
        elif (
            args.dataset_path in MMVUDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MMVUDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MMVUDataset
            args.hf_split = "validation"
            args.hf_subset = None
        elif (
            args.dataset_path in InstructCoderDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in InstructCoderDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = InstructCoderDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in MTBenchDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MTBenchDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MTBenchDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in ConversationDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in ConversationDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = ConversationDataset
        elif (
            args.dataset_path in AIMODataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in AIMODataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = AIMODataset
            args.hf_split = "train"
        elif (
            args.dataset_path
            in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS  # noqa: E501
            or args.hf_name in NextEditPredictionDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = NextEditPredictionDataset
            args.hf_split = "train"
        elif (
            args.dataset_path in ASRDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in ASRDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = ASRDataset
            args.hf_split = "train"
        elif args.dataset_path in BlazeditDataset.SUPPORTED_DATASET_PATHS:
            dataset_class = BlazeditDataset
            args.hf_split = "train"
            hf_kwargs = {
                "min_distance": args.blazedit_min_distance,
                "max_distance": args.blazedit_max_distance,
            }
        elif (
            args.dataset_path in MLPerfDataset.SUPPORTED_DATASET_PATHS
            or args.hf_name in MLPerfDataset.SUPPORTED_DATASET_PATHS
        ):
            dataset_class = MLPerfDataset
            args.hf_split = "train"
        else:
            supported_datasets = set([
                dataset_name for cls in HuggingFaceDataset.__subclasses__()
                for dataset_name in cls.SUPPORTED_DATASET_PATHS
            ])
            raise ValueError(
                f"Unsupported dataset path: {args.dataset_path}. "
                "Huggingface dataset only supports dataset_path"
                f" from one of following: {supported_datasets}. "
                "Please consider contributing if you would "
                "like to add support for additional dataset formats.")

        if dataset_class.IS_MULTIMODAL and args.backend not in [
                "openai-chat",
                "openai-audio",
        ]:
            # multi-modal benchmark is only available on OpenAI Chat
            # endpoint-type.
            raise ValueError(
                "Multi-modal content is only supported on 'openai-chat' and "
                "'openai-audio' backends.")
        input_requests = dataset_class(
            dataset_path=args.dataset_path,
            dataset_subset=args.hf_subset,
            dataset_split=args.hf_split,
            random_seed=args.seed,
            no_stream=args.no_stream,
            hf_name=args.hf_name,
        ).sample(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            output_len=args.hf_output_len,
            request_id_prefix=args.request_id_prefix,
            no_oversample=args.no_oversample,
            **hf_kwargs
        )

    else:
        # For datasets that follow a similar structure, use a mapping.
        dataset_mapping = {
            "spec_bench":
            lambda: SpecBench(dataset_path=args.dataset_path,
                              category=args.spec_bench_category).sample(
                num_requests=args.num_prompts,
                tokenizer=tokenizer,
                output_len=args.spec_bench_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "sharegpt": lambda: ShareGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                output_len=args.sharegpt_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "burstgpt": lambda: BurstGPTDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "random": lambda: RandomDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                range_ratio=args.random_range_ratio,
                request_id_prefix=args.request_id_prefix,
                batchsize=args.random_batch_size,
                no_oversample=args.no_oversample,
            ),
            "random-mm":
            lambda: RandomMultiModalDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.random_prefix_len,
                range_ratio=args.random_range_ratio,
                input_len=args.random_input_len,
                output_len=args.random_output_len,
                base_items_per_request=args.random_mm_base_items_per_request,
                limit_mm_per_prompt=args.random_mm_limit_mm_per_prompt,
                num_mm_items_range_ratio=args.random_mm_num_mm_items_range_ratio,
                bucket_config=args.random_mm_bucket_config,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
            "prefix_repetition":
            lambda: PrefixRepetitionRandomDataset(
                random_seed=args.seed, dataset_path=args.dataset_path
            ).sample(
                tokenizer=tokenizer,
                num_requests=args.num_prompts,
                prefix_len=args.prefix_repetition_prefix_len,
                suffix_len=args.prefix_repetition_suffix_len,
                num_prefixes=args.prefix_repetition_num_prefixes,
                output_len=args.prefix_repetition_output_len,
                request_id_prefix=args.request_id_prefix,
                no_oversample=args.no_oversample,
            ),
        }

        try:
            # Enforce endpoint compatibility for multimodal datasets.
            if args.dataset_name == "random-mm" and args.backend not in [
                    "openai-chat"]:
                raise ValueError(
                    "Multi-modal content (images) is only supported on "
                    "'openai-chat' backend."
                )
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    return input_requests
































