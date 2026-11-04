# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N999


class ABCSheetMusicPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"abc": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "audio/notation"
    DESCRIPTION = "Display ABC as sheet music and pass the original score to music generation."

    def preview(self, abc):
        return {"ui": {"abc": [abc]}, "result": (abc,)}


NODE_CLASS_MAPPINGS = {"ABCSheetMusicPreview": ABCSheetMusicPreview}
NODE_DISPLAY_NAME_MAPPINGS = {"ABCSheetMusicPreview": "Preview ABC Sheet Music"}
WEB_DIRECTORY = "./web"
