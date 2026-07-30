"""Standalone smoke test for LongcatNextImageDecoder — no vLLM engine, no
orchestrator, no thinker. Constructs the module directly, loads its weights
(model.visual_tokenizer.* + image_decoder/image_decoder.safetensors), and
decodes a short run of synthetic visual codes to verify the module works and
to measure its real VRAM footprint in isolation, mirroring the audio decoder
standalone test.

Run with: python pbs/scripts/longcat_next_image_decoder_standalone.py <model_path>
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import NUM_CODEBOOKS
from vllm_omni.model_executor.models.longcat_next.modeling_longcat_next_image_decoder import (
    LongcatNextImageDecoder,
)


def main() -> None:
    model_path = sys.argv[1]

    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    print(f"Before load: {(total - free_before) / 1e9:.2f} GB used / {total / 1e9:.2f} GB total")

    model_config = OmniModelConfig(
        model=model_path,
        model_arch="LongcatNextImageDecoder",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )
    from vllm.config import VllmConfig

    vllm_config = VllmConfig(model_config=model_config)

    decoder = LongcatNextImageDecoder(vllm_config=vllm_config, prefix="")
    decoder._ensure_weights()

    free_after_load, _ = torch.cuda.mem_get_info()
    print(f"After weight load: {(total - free_after_load) / 1e9:.2f} GB used")
    print(f"Peak allocated during load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Synthetic codes: token_h * token_w positions, NUM_CODEBOOKS levels each,
    # all sampled against the shared 16384 codebook size (unlike audio, every
    # visual VQ level uses the same vocab size).
    torch.manual_seed(0)
    codebook_sizes = [
        int(s) for s in decoder.hf_config.visual_config.vq_config.codebook_sizes
    ]
    token_h = decoder.default_token_h
    token_w = decoder.default_token_w
    num_positions = token_h * token_w

    levels = [
        torch.randint(0, codebook_sizes[lvl], (num_positions, 1), dtype=torch.long)
        for lvl in range(NUM_CODEBOOKS)
    ]
    visual_codes = torch.cat(levels, dim=1).flatten().tolist()

    out = decoder.forward(
        input_ids=None,
        positions=None,
        additional_information={
            "visual_token_ids": visual_codes,
            "token_h": token_h,
            "token_w": token_w,
        },
    )

    free_after_decode, _ = torch.cuda.mem_get_info()
    print(f"After decode: {(total - free_after_decode) / 1e9:.2f} GB used")
    print(f"Peak allocated overall: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    image = out.multimodal_outputs.get("model_outputs")
    if image is None:
        print("FAIL: no image produced")
        sys.exit(1)
    print(f"OK: image shape={tuple(image.shape)} dtype={image.dtype}")


if __name__ == "__main__":
    main()
