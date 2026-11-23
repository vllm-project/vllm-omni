# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request fields and example defaults for the Kandinsky 6 TI2VA pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Forwarded to ``OmniDiffusionSamplingParams.extra_args`` by the shared examples
# and the OpenAI video API's ``extra_body``.
KANDINSKY6_EXTRA_BODY_PARAMS = frozenset(
    {
        # bool, default True: also sample the audio track (TI2VA). False -> video only.
        "sample_audio",
        # "pretrain" (text-only) | "tail_cond_first_frame" (reference image).
        "visual_cond_scheme",
        "target_resolution",
        "sr_model",
        "sr_num_inference_steps",
        "sr_min_overlap",
        "sr_tiles_batch_size",
    }
)
KANDINSKY6_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Kandinsky6VideoGenerationDefaults:
    """Shared entrypoint metadata with generation settings deferred to K6.

    The shared defaults container requires concrete dimensions and counts.
    Kandinsky provides the same entrypoint interface locally so its loaded
    scheduler can resolve those settings for regular and Distil checkpoints.
    """

    width: None = None
    height: None = None
    num_frames: None = None
    num_inference_steps: None = None
    guidance_scale: None = None
    flow_shift: None = None
    duration_seconds: None = None
    default_negative_prompt: None = None
    fps: float = 24.0
    dimension_multiple: int = 2
    output: str = "kandinsky6_output.mp4"

    @property
    def max_area(self) -> None:
        return None

    def cli_defaults(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "fps": self.fps,
            "output": self.output,
        }


def get_kandinsky6_video_generation_defaults(
    extra_body: Mapping[str, Any] | None = None,
) -> Kandinsky6VideoGenerationDefaults:
    """Leave checkpoint-dependent settings to the Kandinsky pipeline's scheduler.

    Returning an entry prevents shared examples from applying another model's
    presets. Unspecified dimensions, frames, steps and guidance reach the
    pipeline unchanged, which resolves regular and Distil defaults from the
    checkpoint's scheduler configuration.
    """
    del extra_body
    return Kandinsky6VideoGenerationDefaults()


def preserve_kandinsky6_reference_image_size(*, model: str | None, revision: str | None = None) -> bool:
    """The pipeline resizes the reference image to the base generation canvas."""
    del model, revision
    return True
