"""Node B (image decoder) half of the 2-node LongCat-Next thinker+image e2e
test. Reads the thinker's real per-position visual codes -- produced by
talker_mtp's visual branch (modeling_longcat_next.py) and written by node A's
longcat_next_multinode_image_thinker.py to a shared scratch file -- and
decodes them with LongcatNextImageDecoder directly, no vLLM engine, no
server, same pattern as pbs/scripts/longcat_next_image_decoder_standalone.py,
just with real thinker codes instead of synthetic ones.

Run with: python longcat_next_multinode_image_decoder.py <model_path> <in_json> <out_png>
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm.config import VllmConfig

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.longcat_next.modeling_longcat_next_image_decoder import (
    LongcatNextImageDecoder,
)


def main() -> None:
    model_path = sys.argv[1]
    in_json = sys.argv[2]
    out_png = sys.argv[3]

    with open(in_json) as f:
        payload = json.load(f)
    codes = payload["visual_codes"]
    token_h = payload["token_h"]
    token_w = payload["token_w"]
    print(f"[image_decoder] read {len(codes)} visual code frames from {in_json} (token_h={token_h} token_w={token_w})")
    if not codes:
        verdict = (
            "FAIL: no visual codes in handoff file -- check that node A's talker_mtp "
            "actually ran (see '[thinker] talker_mtp produced N visual code frames' in its log)"
        )
        print(f"[image_decoder] VERDICT {verdict}")
        sys.exit(1)

    # The reference's own GEN_IMAGE_STAGE (output_processor.py:204-216) has no
    # analogous forced-termination transition tied to token_h -- it reads
    # token_h but never uses it; only token_w gates the per-row IMG_NEWLINE
    # forcing. Generation stops only on a naturally-sampled IMG_END or the
    # caller's token budget, so a thinker run with a loose max_tokens will
    # keep producing frames past the intended grid (observed: 51 frames for a
    # requested 4x4=16 grid). The intended image is always the *first*
    # token_h*token_w kept frames; anything after that is the model
    # continuing past the grid it was asked for, not part of this image.
    expected = token_h * token_w
    if len(codes) > expected:
        print(
            f"[image_decoder] got {len(codes)} frames, expected "
            f"token_h*token_w={expected} -- generation ran past the intended "
            f"grid (no auto-termination by row count, matching the "
            f"reference); truncating to the first {expected} frames"
        )
        codes = codes[:expected]
    elif len(codes) < expected:
        print(
            f"[image_decoder] WARNING: got {len(codes)} frames, expected "
            f"token_h*token_w={expected} -- generation stopped early "
            "(max_tokens hit before the grid completed); decoding the "
            "partial set anyway"
        )

    free_before, total = torch.cuda.mem_get_info()
    print(f"[image_decoder] before load: {(total - free_before) / 1e9:.2f} GB used / {total / 1e9:.2f} GB total")

    model_config = OmniModelConfig(
        model=model_path,
        model_arch="LongcatNextImageDecoder",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )
    vllm_config = VllmConfig(model_config=model_config)
    decoder = LongcatNextImageDecoder(vllm_config=vllm_config, prefix="")
    decoder._ensure_weights()

    free_after_load, _ = torch.cuda.mem_get_info()
    print(f"[image_decoder] after weight load: {(total - free_after_load) / 1e9:.2f} GB used")

    out = decoder.forward(
        input_ids=None,
        positions=None,
        additional_information={
            "visual_token_ids": codes,
            "token_h": token_h,
            "token_w": token_w,
        },
    )

    image = out.multimodal_outputs.get("model_outputs")
    if image is None:
        print("[image_decoder] VERDICT FAIL: decoder ran but produced no image")
        sys.exit(1)

    print(f"[image_decoder] OK: image tensor shape={tuple(image.shape)} dtype={image.dtype}")

    arr = (image.squeeze(0).clamp(0, 1) * 255).round().to(torch.uint8)
    arr = arr.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    from PIL import Image

    Image.fromarray(arr).save(out_png)
    print(f"[image_decoder] wrote {out_png}")
    print(
        f"[image_decoder] VERDICT PASS: decoded {len(codes)} frames into a "
        f"{arr.shape[1]}x{arr.shape[0]} image"
    )


if __name__ == "__main__":
    main()
