# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
"""BAGEL processor for image and text inputs."""

from transformers import AutoProcessor
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers import BatchFeature


class BagelProcessor(ProcessorMixin):
    """
    Constructs a BAGEL processor which wraps a
    SigLIP image processor and a Qwen2 tokenizer.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        images: ImageInput = None,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s) and image(s).
        """
        # 1. 处理图片
        if images is not None:
            image_kwargs = {**kwargs}
            if "return_tensors" not in image_kwargs:
                image_kwargs["return_tensors"] = "pt"

            # pixel_values 通常是一个 Tensor
            pixel_values_output = self.image_processor(images, **image_kwargs)
            pixel_values = pixel_values_output["pixel_values"]
        else:
            pixel_values = None

        # 2. 处理文本
        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
        else:
            text_inputs = BatchFeature()  # 空的容器

        # 3. 关键修改：合并并封装为 BatchFeature
        data = dict(text_inputs)  # 先转为普通字典

        if pixel_values is not None:
            data["pixel_values"] = pixel_values

        # 强制返回 BatchFeature 对象，消除 vLLM 的 Warning
        return BatchFeature(data=data, tensor_type="pt")

    # def __call__(
    #    self,
    #    text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
    #    images: ImageInput = None,
    #    **kwargs,
    # ):
    #    """
    #    Main method to prepare for the model one or several sequences(s) and image(s).
    #    """
    #    if images is not None:
    #        # Process images with the image processor
    #        # Ensure return_tensors is set to "pt" for PyTorch tensors
    #        image_kwargs = {**kwargs}
    #        if "return_tensors" not in image_kwargs:
    #            image_kwargs["return_tensors"] = "pt"
    #        pixel_values = self.image_processor(images, **image_kwargs)
    #    else:
    #        pixel_values = None

    #    text_inputs = self.tokenizer(text, **kwargs) if text is not None else None

    #    if pixel_values is not None and text_inputs is not None:
    #        text_inputs["pixel_values"] = pixel_values["pixel_values"]
    #        return text_inputs
    #    elif pixel_values is not None:
    #        return pixel_values
    #    else:
    #        return text_inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's batch_decode.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's decode.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


AutoProcessor.register("BagelProcessor", BagelProcessor)
