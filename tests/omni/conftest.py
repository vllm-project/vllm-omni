# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pytest configuration and fixtures for vllm-omni tests.
"""
from typing import Any, Optional, Union

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni import Omni

# Type aliases for multimodal inputs
PromptAudioInput = Optional[Union[list[tuple[Any, int]], tuple[Any, int]]]
PromptImageInput = Optional[Union[list[Any], Any]]
PromptVideoInput = Optional[Union[list[Any], Any]]


class OmniRunner:
    """
    Simplified test runner for Omni models.

    This runner wraps the Omni entrypoint for easier testing with default
    configurations suitable for unit tests.

    Default values:
    - `seed`: Set to `0` for test reproducibility
    - `init_sleep_seconds`: Set to `5` to reduce test time
    - `batch_timeout`: Set to `5` seconds
    - `init_timeout`: Set to `60` seconds
    - `log_stats`: Set to `False` to reduce test output
    """

    def __init__(
        self,
        model_name: str,
        seed: int = 0,
        init_sleep_seconds: int = 5,
        batch_timeout: int = 5,
        init_timeout: int = 60,
        shm_threshold_bytes: int = 65536,
        log_stats: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize an OmniRunner for testing.

        Args:
            model_name: The model name or path
            seed: Random seed for reproducibility
            init_sleep_seconds: Sleep time after starting each stage
            batch_timeout: Timeout for batching in seconds
            init_timeout: Timeout for initializing stages in seconds
            shm_threshold_bytes: Threshold for using shared memory
            log_stats: Enable detailed statistics logging
            **kwargs: Additional arguments passed to Omni
        """
        self.model_name = model_name
        self.seed = seed

        self.omni = Omni(
            model=model_name,
            log_stats=log_stats,
            init_sleep_seconds=init_sleep_seconds,
            batch_timeout=batch_timeout,
            init_timeout=init_timeout,
            shm_threshold_bytes=shm_threshold_bytes,
            **kwargs,
        )

    def get_default_sampling_params_list(
        self,
        max_tokens: int = 128,
    ) -> list[SamplingParams]:
        """
        Get default sampling parameters for all three stages.

        Args:
            max_tokens: Maximum tokens to generate per stage

        Returns:
            List of SamplingParams for [thinker, talker, code2wav] stages
        """
        thinker_params = SamplingParams(
            temperature=0.0,  # Deterministic
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            seed=self.seed,
            detokenize=True,
            repetition_penalty=1.1,
        )

        talker_params = SamplingParams(
            temperature=0.9,
            top_p=0.8,
            top_k=40,
            max_tokens=max_tokens,
            seed=self.seed,
            detokenize=True,
            repetition_penalty=1.05,
            stop_token_ids=[8294],
        )

        code2wav_params = SamplingParams(
            temperature=0.0,  # Deterministic
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            seed=self.seed,
            detokenize=True,
            repetition_penalty=1.1,
        )

        return [thinker_params, talker_params, code2wav_params]

    def get_omni_inputs(
        self,
        prompts: Union[list[str], str],
        system_prompt: Optional[str] = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Construct Omni input format from prompts and multimodal data.

        This method formats inputs in the Qwen2.5-Omni style with proper
        multimodal placeholders.

        Args:
            prompts: Text prompt(s) - either a single string or list of strings
            system_prompt: Optional system prompt (defaults to Qwen system prompt)
            audios: Audio input(s) - tuple of (audio_array, sample_rate) or list of tuples
            images: Image input(s) - PIL Image or list of PIL Images
            videos: Video input(s) - numpy array or list of numpy arrays
            mm_processor_kwargs: Optional processor kwargs (e.g., use_audio_in_video)

        Returns:
            List of prompt dictionaries suitable for Omni.generate()
        """
        if system_prompt is None:
            system_prompt = (
                "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
                "Group, capable of perceiving auditory and visual inputs, as well as "
                "generating text and speech."
            )

        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Normalize multimodal inputs to lists
        def _normalize_mm_input(mm_input, num_prompts):
            if mm_input is None:
                return [None] * num_prompts
            if isinstance(mm_input, list):
                if len(mm_input) != num_prompts:
                    raise ValueError(
                        f"Multimodal input list length ({len(mm_input)}) "
                        f"must match prompts length ({num_prompts})"
                    )
                return mm_input
            # Single input - replicate for all prompts
            return [mm_input] * num_prompts

        num_prompts = len(prompts)
        audios_list = _normalize_mm_input(audios, num_prompts)
        images_list = _normalize_mm_input(images, num_prompts)
        videos_list = _normalize_mm_input(videos, num_prompts)

        omni_inputs = []
        for i, prompt_text in enumerate(prompts):
            # Build prompt with multimodal placeholders
            user_content = ""
            multi_modal_data = {}

            # Add audio placeholder and data
            audio = audios_list[i]
            if audio is not None:
                if isinstance(audio, list):
                    # Multiple audios
                    for _ in audio:
                        user_content += "<|audio_bos|><|AUDIO|><|audio_eos|>"
                    multi_modal_data["audio"] = audio
                else:
                    # Single audio
                    user_content += "<|audio_bos|><|AUDIO|><|audio_eos|>"
                    multi_modal_data["audio"] = audio

            # Add image placeholder and data
            image = images_list[i]
            if image is not None:
                if isinstance(image, list):
                    # Multiple images
                    for _ in image:
                        user_content += "<|vision_bos|><|IMAGE|><|vision_eos|>"
                    multi_modal_data["image"] = image
                else:
                    # Single image
                    user_content += "<|vision_bos|><|IMAGE|><|vision_eos|>"
                    multi_modal_data["image"] = image

            # Add video placeholder and data
            video = videos_list[i]
            if video is not None:
                if isinstance(video, list):
                    # Multiple videos
                    for _ in video:
                        user_content += "<|vision_bos|><|VIDEO|><|vision_eos|>"
                    multi_modal_data["video"] = video
                else:
                    # Single video
                    user_content += "<|vision_bos|><|VIDEO|><|vision_eos|>"
                    multi_modal_data["video"] = video

            # Add text prompt
            user_content += prompt_text

            # Build full prompt with chat template
            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            # Construct input dict
            input_dict: dict[str, Any] = {"prompt": full_prompt}
            if multi_modal_data:
                input_dict["multi_modal_data"] = multi_modal_data
            if mm_processor_kwargs:
                input_dict["mm_processor_kwargs"] = mm_processor_kwargs

            omni_inputs.append(input_dict)

        return omni_inputs

    def generate(
        self,
        prompts: list[dict[str, Any]],
        sampling_params_list: Optional[list[SamplingParams]] = None,
    ) -> list[Any]:
        """
        Generate outputs for the given prompts.

        Args:
            prompts: List of prompt dictionaries with 'prompt' and optionally
                    'multi_modal_data' keys
            sampling_params_list: List of sampling parameters for each stage.
                                 If None, uses default parameters.

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        if sampling_params_list is None:
            sampling_params_list = self.get_default_sampling_params_list()

        return self.omni.generate(prompts, sampling_params_list)

    def generate_multimodal(
        self,
        prompts: Union[list[str], str],
        sampling_params_list: Optional[list[SamplingParams]] = None,
        system_prompt: Optional[str] = None,
        audios: PromptAudioInput = None,
        images: PromptImageInput = None,
        videos: PromptVideoInput = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """
        Convenience method to generate with multimodal inputs.

        Args:
            prompts: Text prompt(s)
            sampling_params_list: List of sampling parameters for each stage
            system_prompt: Optional system prompt
            audios: Audio input(s)
            images: Image input(s)
            videos: Video input(s)
            mm_processor_kwargs: Optional processor kwargs

        Returns:
            List of OmniRequestOutput objects from stages with final_output=True
        """
        omni_inputs = self.get_omni_inputs(
            prompts=prompts,
            system_prompt=system_prompt,
            audios=audios,
            images=images,
            videos=videos,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        return self.generate(omni_inputs, sampling_params_list)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if hasattr(self.omni, 'close'):
            self.omni.close()

    def close(self):
        """Close and cleanup the Omni instance."""
        if hasattr(self.omni, 'close'):
            self.omni.close()


@pytest.fixture
def omni_runner():
    """
    Pytest fixture that provides an OmniRunner factory function.

    Usage:
        def test_example(omni_runner):
            with omni_runner("Qwen/Qwen2.5-Omni-7B") as runner:
                outputs = runner.generate(prompts)
    """
    runners = []

    def _omni_runner(model_name: str, **kwargs) -> OmniRunner:
        runner = OmniRunner(model_name, **kwargs)
        runners.append(runner)
        return runner

    yield _omni_runner

    # Cleanup all runners after test
    for runner in runners:
        try:
            runner.close()
        except Exception:
            pass


@pytest.fixture
def example_prompts():
    """Fixture providing example prompts for testing."""
    return [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
