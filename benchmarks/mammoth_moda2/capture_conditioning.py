# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Record the real AR->DiT payload while running the unchanged shared example.

This qualification-only wrapper observes the stage-input boundary; the original
processor produces the payload and receives its original arguments unchanged.
"""

import json
import os
import runpy
from pathlib import Path

from safetensors.torch import save_file

from vllm_omni.model_executor.stage_input_processors import mammoth_moda2


def main():
    destination = Path(os.environ["MAMMOTH_CONDITIONING_DIR"])
    destination.mkdir(parents=True, exist_ok=False)
    original = mammoth_moda2.ar2diffusion

    def record(*args, **kwargs):
        payload = original(*args, **kwargs)
        info = payload["additional_information"]
        target = destination / "conditioning.safetensors"
        assert not target.exists(), "Capture runner expects one real request"
        hidden = info["full_hidden_states"].detach().cpu().contiguous()
        save_file({"full_hidden_states": hidden}, str(target))
        metadata = {
            "full_token_ids": info["full_token_ids"],
            "answer_start_index": info["answer_start_index"],
            "height": payload["height"],
            "width": payload["width"],
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype),
        }
        (destination / "conditioning.json").write_text(json.dumps(metadata, indent=2))
        print("CAPTURED_REAL_AR_CONDITIONING " + str(target), flush=True)
        return payload

    mammoth_moda2.ar2diffusion = record
    try:
        runpy.run_path(
            "examples/offline_inference/text_to_image/text_to_image.py",
            run_name="__main__",
        )
    finally:
        mammoth_moda2.ar2diffusion = original


if __name__ == "__main__":
    main()
