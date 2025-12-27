# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/FunAudioLLM/Fun-Audio-Chat
#
# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for FunAudioChat.

This module provides the FunAudioChatProcessor for handling audio inputs
with the Fun-Audio-Chat model. It combines WhisperFeatureExtractor for
mel spectrograms and a speech tokenizer for discrete audio tokens.
"""

import json
import os
import warnings
from typing import BinaryIO

import librosa
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import (
    PROCESSOR_NAME,
    AudioKwargs,
    PreTrainedTokenizerBase,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    custom_object_save,
    logger,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import (
    CHAT_TEMPLATE_DIR,
    CHAT_TEMPLATE_FILE,
    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
)
from transformers.utils.deprecation import deprecate_kwarg


class FunAudioChatAudioKwargs(AudioKwargs, total=False):
    speech_kwargs: TextKwargs = {
        **TextKwargs.__annotations__,
    }


class FunAudioChatProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: FunAudioChatAudioKwargs = {
        **AudioKwargs.__annotations__,
    }
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {"speech_kwargs": {}},
    }


class FunAudioChatProcessor(ProcessorMixin):
    r"""
    Constructs a FunAudioChat processor which wraps a FunAudioChat feature extractor
    and a FunAudioChat tokenizer into a single processor.

    [`FunAudioChatProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`]
    and [`FunAudioChatTokenizerFast`]. See the [`~FunAudioChatProcessor.__call__`] and
    [`~FunAudioChatProcessor.decode`] for more information.

    Args:
        feature_extractor ([`WhisperFeatureExtractor`], *optional*):
            The feature extractor is a required input.
        tokenizer ([`FunAudioChatTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If not provided,
            the default chat template is used.
        audio_token (`str`, *optional*, defaults to `"<|AUDIO|>"`):
            The token to use for audio tokens.
        audio_bos_token (`str`, *optional*, defaults to `"<|audio_bos|>"`):
            The token to use for audio bos tokens.
        audio_eos_token (`str`, *optional*, defaults to `"<|audio_eos|>"`):
            The token to use for audio eos tokens.
    """

    attributes = ["feature_extractor", "speech_tokenizer", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "audio_token",
        "audio_bos_token",
        "audio_eos_token",
        "audio_pad_token",
        "audio_group_size",
    ]
    feature_extractor_class = "WhisperFeatureExtractor"
    speech_tokenizer_class = "AutoTokenizer"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        feature_extractor=None,
        speech_tokenizer=None,
        tokenizer=None,
        chat_template=None,
        audio_token="<|AUDIO|>",
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_pad_token="<|audio_pad|>",
        audio_group_size=5,
    ):
        if chat_template is None:
            chat_template = self.default_chat_template
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.audio_bos_token = tokenizer.audio_bos_token if hasattr(tokenizer, "audio_bos_token") else audio_bos_token
        self.audio_eos_token = tokenizer.audio_eos_token if hasattr(tokenizer, "audio_eos_token") else audio_eos_token
        self.audio_pad_token = tokenizer.audio_pad_token if hasattr(tokenizer, "audio_pad_token") else audio_pad_token
        if speech_tokenizer is not None:
            self.audio_group_size = speech_tokenizer.init_kwargs.get("audio_group_size", audio_group_size)
        else:
            self.audio_group_size = audio_group_size
        super().__init__(feature_extractor, speech_tokenizer, tokenizer, chat_template=chat_template)

    def _regularize_audios(self, audios: list, sampling_rate: float, **kwargs) -> dict[str, list | list[float]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        results, sampling_rates = [], []
        for audio in audios:
            if isinstance(audio, (str, BinaryIO)):
                audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

            if not isinstance(audio, np.ndarray):
                raise ValueError(f"Expect input is a list of audios, but got {type(audio)}.")

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    @deprecate_kwarg("audios", version="4.54.0", new_name="audio")
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        audio: np.ndarray | list[np.ndarray] = None,
        speech: PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        audios=None,  # kept for BC
        **kwargs: Unpack[FunAudioChatProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and audio(s).

        This method forwards the `text` and `kwargs` arguments to FunAudioChatTokenizerFast's
        [`~FunAudioChatTokenizerFast.__call__`] if `text` is not `None` to encode the text.
        To prepare the audio(s), this method forwards the `audios` and `kwargs` arguments to
        WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] if `audios` is not `None`.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string
                or a list of strings (pretokenized string).
            audio (`np.ndarray`, `List[np.ndarray]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array.
        """
        # Handle BC when user passes deprecated keyword argument
        if audios is not None and audio is None:
            audio = audios
            warnings.warn(
                "You may have used the keyword argument for the `audio` inputs. "
                "It is strongly recommended to pass inputs with keyword arguments "
                "with keys `audio` and `text`. From transformers v4.55 `audio` will "
                "be the only acceptable keyword argument.",
                FutureWarning,
            )

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            FunAudioChatProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            # Check if audio is a list of numpy arrays (wav_list) or JSON strings
            is_wav_list = isinstance(audio[0], np.ndarray) if isinstance(audio, list) and len(audio) > 0 else False

            if is_wav_list:
                # Handle wav_list input: list of numpy arrays
                audio_wavs = audio
                audio_path = []
                if output_kwargs["audio_kwargs"].get("sampling_rate", None) is not None:
                    # Resample audio to the specified sampling rate
                    audio_wavs = [
                        librosa.resample(
                            wav,
                            orig_sr=output_kwargs["audio_kwargs"]["sampling_rate"],
                            target_sr=getattr(self, "audio_sampling_rate", 16000),
                        )
                        for wav in audio_wavs
                    ]
                    output_kwargs["audio_kwargs"].pop("sampling_rate")

                # Calculate speech tokens based on 25Hz frame rate
                # Assuming audio_sampling_rate is 16000 Hz
                sampling_rate = getattr(self, "audio_sampling_rate", 16000)
                frame_rate = 25  # 25 Hz

                speech = []
                for wav in audio_wavs:
                    # Calculate number of frames at 25Hz
                    duration_seconds = len(wav) / sampling_rate
                    num_frames = int(duration_seconds * frame_rate)
                    # Construct speech token with audio_pad_token
                    speech_token = self.audio_pad_token * num_frames
                    speech.append(speech_token)

                parsed_audios = [{"path": "", "token": token} for token in speech]
            else:
                # Handle original JSON string format
                parsed_audios = [json.loads(au) for au in audio]
                audio_path = [audio_data["path"] for audio_data in parsed_audios if audio_data["path"] != ""]
                speech = [audio_data["token"] for audio_data in parsed_audios]
                audio_wavs = None

            # ensure we have as much audios as audio tokens
            num_audio_tokens = sum(sample.count(self.audio_token) for sample in text)
            num_audios = len(speech)
            if num_audio_tokens != num_audios:
                raise ValueError(
                    f"Found {num_audio_tokens} {self.audio_token} token"
                    f"{'s' if num_audio_tokens > 1 else ''} in provided text "
                    f"but received {num_audios} audio{'s' if num_audios > 1 else ''}"
                )

            # Some kwargs should not be changed so we can expand text with audio tokens below
            output_kwargs["audio_kwargs"]["speech_kwargs"]["return_attention_mask"] = True
            output_kwargs["audio_kwargs"]["speech_kwargs"]["return_token_type_ids"] = False
            output_kwargs["audio_kwargs"]["speech_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["speech_kwargs"]["pad_to_multiple_of"] = self.audio_group_size
            output_kwargs["audio_kwargs"]["speech_kwargs"]["return_tensors"] = "pt"
            audio_inputs = self.speech_tokenizer(speech, **output_kwargs["audio_kwargs"]["speech_kwargs"])
            # rename attention_mask to prevent conflicts later on
            audio_inputs["speech_ids"] = audio_inputs.pop("input_ids")
            audio_inputs["speech_attention_mask"] = audio_inputs.pop("attention_mask")

            speech_lengths = audio_inputs["speech_attention_mask"].sum(-1).tolist()
            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    speech_length = speech_lengths.pop(0)
                    num_audio_tokens = (speech_length + (self.audio_group_size - 1)) // self.audio_group_size

                    expanded_audio_token = self.audio_token * int(num_audio_tokens)

                    audio_token_start_idx = sample.find(self.audio_token)
                    audio_token_end_idx = audio_token_start_idx + len(self.audio_token)

                    has_bos = (
                        sample[audio_token_start_idx - len(self.audio_bos_token) : audio_token_start_idx]
                        == self.audio_bos_token
                    )
                    has_eos = (
                        sample[audio_token_end_idx : audio_token_end_idx + len(self.audio_eos_token)]
                        == self.audio_eos_token
                    )

                    # Check if this audio token is surrounded by bos/eos tokens
                    if not has_bos and not has_eos:
                        expanded_audio_token = self.audio_bos_token + expanded_audio_token + self.audio_eos_token

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)
            text = expanded_text

            # Process audio features
            if is_wav_list or len(audio_path) > 0:
                if not is_wav_list:
                    # Load audio from paths for JSON format
                    audio_wavs = self._regularize_audios(
                        audio_path,
                        sampling_rate=getattr(self, "audio_sampling_rate", 16000),
                    )["audios"]

                # Some kwargs should not be changed so we can expand text with audio tokens below
                output_kwargs["audio_kwargs"]["return_attention_mask"] = True
                output_kwargs["audio_kwargs"]["padding"] = "max_length"
                wav_inputs = self.feature_extractor(
                    audio_wavs,
                    sampling_rate=getattr(self, "audio_sampling_rate", 16000),
                    **output_kwargs["audio_kwargs"],
                )

                # rename attention_mask to prevent conflicts later on
                wav_inputs["feature_attention_mask"] = wav_inputs.pop("attention_mask")
                if is_wav_list:
                    # For wav_list, all audios exist
                    wav_inputs["feature_exist_mask"] = torch.ones(
                        len(audio_wavs),
                        dtype=torch.bool,
                        device=wav_inputs["feature_attention_mask"].device,
                    )
                else:
                    wav_inputs["feature_exist_mask"] = torch.tensor(
                        [audio_data["path"] != "" for audio_data in parsed_audios],
                        dtype=torch.bool,
                        device=wav_inputs["feature_attention_mask"].device,
                    )
                audio_inputs.update(wav_inputs)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, inputs, modalities=["audio"])

        if audio is not None:
            inputs.update(audio_inputs)

        return BatchFeature(data={**inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to FunAudioChatTokenizerFast's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of
        this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to FunAudioChatTokenizerFast's
        [`~PreTrainedTokenizer.decode`]. Please refer to the docstring of this
        method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Identify and instantiate the subcomponents of Processor classes, like image processors
        and tokenizers.
        """
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(cls.get_possibly_dynamic_module(n) if n is not None else None for n in class_name)
                if attribute_name == "image_processor":
                    use_fast = kwargs.get("use_fast", None)
                    if use_fast is None:
                        logger.warning_once(
                            "Using a slow image processor as `use_fast` is unset and a slow processor "
                            "was saved with this model. `use_fast=True` will be the default behavior "
                            "in v4.52, even if the model was saved with a slow processor."
                        )
                else:
                    use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = cls.get_possibly_dynamic_module(class_name)
            if attribute_name == "speech_tokenizer":
                extra_kwargs = {"subfolder": attribute_name}
            else:
                extra_kwargs = {}
            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs, **extra_kwargs))
        return args

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the
        specified directory so that it can be reloaded using the
        [`~ProcessorMixin.from_pretrained`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. "
                "Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.attributes]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
            configs.append(self)
            custom_object_save(self, save_directory, config=configs)

        save_jinja_files = kwargs.get("save_jinja_files", True)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            if attribute_name == "tokenizer":
                attribute.save_pretrained(save_directory, save_jinja_files=save_jinja_files)
            elif attribute_name == "speech_tokenizer":
                attribute.save_pretrained(os.path.join(save_directory, attribute_name))
            else:
                attribute.save_pretrained(save_directory)

        if self._auto_class is not None:
            for attribute_name in self.attributes:
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_chat_template_file_jinja = os.path.join(save_directory, CHAT_TEMPLATE_FILE)
        output_chat_template_file_legacy = os.path.join(save_directory, LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE)
        chat_template_dir = os.path.join(save_directory, CHAT_TEMPLATE_DIR)

        processor_dict = self.to_dict()

        if self.chat_template is not None:
            save_jinja_files = kwargs.get("save_jinja_files", True)
            is_single_template = isinstance(self.chat_template, str)
            if save_jinja_files and is_single_template:
                with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                    f.write(self.chat_template)
                logger.info(f"chat template saved in {output_chat_template_file_jinja}")
            elif save_jinja_files and not is_single_template:
                for template_name, template in self.chat_template.items():
                    if template_name == "default":
                        with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                            f.write(self.chat_template["default"])
                        logger.info(f"chat template saved in {output_chat_template_file_jinja}")
                    else:
                        os.makedirs(chat_template_dir, exist_ok=True)
                        template_filepath = os.path.join(chat_template_dir, f"{template_name}.jinja")
                        with open(template_filepath, "w", encoding="utf-8") as f:
                            f.write(template)
                        logger.info(f"chat template saved in {template_filepath}")
            elif is_single_template:
                chat_template_json_string = (
                    json.dumps({"chat_template": self.chat_template}, indent=2, sort_keys=True) + "\n"
                )
                with open(output_chat_template_file_legacy, "w", encoding="utf-8") as writer:
                    writer.write(chat_template_json_string)
                logger.info(f"chat template saved in {output_chat_template_file_legacy}")
            elif self.chat_template is not None:
                raise ValueError(
                    "Multiple chat templates are not supported in the legacy format. "
                    "Please save them as separate files using the `save_jinja_files` argument."
                )

        self.to_json_file(output_processor_file)
        logger.info(f"processor saved in {output_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        if set(processor_dict.keys()) == {"processor_class"}:
            return []
        return [output_processor_file]

    @property
    def default_chat_template(self):
        """
        Default chat template for Fun-Audio-Chat.

        This template formats inputs in the form of a chat history. For each message:
        * the template will output the role of the speaker followed by the content
        * content is a list of strings and audios
        * If the content element is an audio, outputs a sequence of <|AUDIO|> tokens
        """
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on


__all__ = ["FunAudioChatProcessor"]
