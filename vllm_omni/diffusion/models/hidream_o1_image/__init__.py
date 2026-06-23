# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

__all__ = [
    "HiDreamO1ImagePipeline",
    "get_hidream_o1_image_post_process_func",
]


def __getattr__(name: str):
    if name == "HiDreamO1ImagePipeline":
        from .pipeline_hidream_o1_image import HiDreamO1ImagePipeline

        return HiDreamO1ImagePipeline
    if name == "get_hidream_o1_image_post_process_func":
        from .pipeline_hidream_o1_image import get_hidream_o1_image_post_process_func

        return get_hidream_o1_image_post_process_func
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
