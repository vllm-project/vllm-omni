# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision-7B-MoT offline inference (full modality matrix).

SenseNova-Vision is a Bagel-fork MoT model; the diffuser reads the SenseNovaVision
per-mode defaults from the prompt dict (``mode`` / ``sensenova_vision_mode``) and
the pipeline routes outputs through the standard BAGEL modality contract:

    text2text        — text-only understanding (no image)
    img2text         — image captioning / VQA (text output)
    dense_detection  — structured detection text parsed with ``parse_bbox``
    dense_OCR        — structured OCR text
    text2img         — image generation from text
    img2img          — image editing from an input image
    img2dense        — depth / normal / segmentation decoded from the output
                       image with ``decode_depth`` / ``decode_normal`` /
                       ``decode_segmentation``
    multi-img2text   — multi-view camera pose estimation, parsed with
                       ``parse_camera_pose``
    recon3d          — multi-view 3D reconstruction (per-view point maps
                       decoded with ``decode_point_map``, optional text)
    mixed            — ``caption_generate``: returns an image plus the
                       intermediate caption text
    think-text2text  — ``think_understanding`` on the two-stage think topology
    think-text2img   — ``think_generate`` on the two-stage think topology
    think-img2img    — ``think_edit`` on the two-stage think topology

The ``think-*`` modalities wrap the user content with the BAGEL think system
prompt and require the ``sensenova_vision_think`` deploy YAML (selected
automatically unless ``--deploy-config`` is given).

Examples:

    # Text to image
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality text2img \
        --prompts "A cute corgi astronaut on the moon" \
        --output ./out

    # Image understanding (caption)
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality img2text \
        --image-path /path/to/photo.jpg \
        --prompts "What are the main objects in this scene and their relationships?"

    # Dense depth estimation
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality img2dense --dense-task depth \
        --image-path /path/to/photo.jpg \
        --output ./out

    # Multi-view camera pose estimation
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality multi-img2text \
        --image-path /path/to/view1.png /path/to/view2.png /path/to/view3.png \
        --output ./out

    # Multi-view 3D reconstruction
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality recon3d \
        --image-path /path/to/view1.png /path/to/view2.png /path/to/view3.png \
        --output ./out

    # Mixed text + image (caption_generate)
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality mixed \
        --image-path /path/to/photo.jpg \
        --output ./out

    # Think-mode image generation (two-stage think topology)
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality think-text2img \
        --prompts "A cute corgi astronaut on the moon" \
        --output ./out
"""

import argparse
import json
import os

from vllm_omni.entrypoints.omni import Omni

# SenseNova-Vision official prompts (from
# SenseNova-Vision/inference/sensenova_vision.py).
RECON3D_PROMPT = (
    "Reconstruct a scene from multiple input images and output one dense 3D "
    "coordinate map per view, all aligned to the first camera's perspective."
)
CAMERA_POSE_PROMPT = (
    "With the first frame as the reference frame, output the relative pose of"
    " all subsequent frames (excluding the first frame) with respect to the"
    " first frame, following the input order and adhering to the strict format"
    " below:Rotation: Represented by a quaternion in the format"
    " <quat>[x,y,z,w], enclosed in <quat> tags;Translation: Represented by a"
    " unit vector (direction) in the format <offset>[x,y,z], enclosed in"
    " <offset> tags (the vector has no absolute physical meaning, only"
    " directional information);Scale: Represented by a numerical value in the"
    " format <scale>value</scale> tags, where the value denotes the magnitude"
    " of translation (corresponding to the length of the translation unit"
    " vector);Enclose the result of each frame in <frame> tags, with no extra"
    " characters, spaces, or line breaks outside the tags."
)

UNDERSTANDING_PROMPT = "What are the main objects in this scene and their relationships?"
DEPTH_PROMPT = (
    "<image> Estimate relative depth for each pixel in the image, with closer "
    "objects appearing brighter and distant objects appearing darker. Output "
    "is a grayscale image with pixel values ranging from 0-255."
)
NORMAL_PROMPT = (
    "<image> Generate an RGB normal map where R, G, B channels represent X, Y, "
    "Z surface directions. The output should show continuous color variations "
    "with no discrete regions, unlike segmentation results."
)
SEGMENTATION_PROMPT = (
    "<image> Could you return the binary segmentation masks for the specified "
    "categories: <p>person furthest to the right</p>?"
)
DETECTION_PROMPT = (
    "<image> Please detect all instances of <p>bird</p>, <p>boat</p>, "
    "<p>person</p>, <p>cell phone</p>, <p>backpack</p>, <p>handbag</p> in the "
    "image. Output the results as a structured text list with each detection "
    "including category and bounding box coordinates in <bbox> format."
)
OCR_PROMPT = (
    "<image> Please recognize all the text in the image. Output the results as "
    "a structured text list with each detection including the recognized text "
    "and its bounding box coordinates in <bbox> format."
)
CAPTION_GENERATE_PROMPT = (
    "<image> Please briefly describe the contents of the image. Please respond "
    "with interleaved segmentation masks for the corresponding parts of the "
    "answer."
)
EDIT_PROMPT = "Turn this image into a vibrant cartoon-style illustration."

# recon3d output geometry (upstream ``recon3d_vae_transform`` =
# ``ImageTransform(512, 256, 16)``: long edge 512, short edge 256).
RECON3D_HEIGHT = 256
RECON3D_WIDTH = 512

# BAGEL/SenseNovaVision chat-scaffold markers.  The AR multimodal processor
# expands each <|image_pad|> into the ViT patch block of one ``image`` item and
# each <|fim_middle|> into the VAE+ViT conditioning block of one ``img2img``
# item, so the number of markers must equal the number of supplied items.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_FIM_MIDDLE = "<|fim_middle|>"
_IMAGE_PAD_BLOCK = "<|image_pad|>\n"
# Understanding tasks continue after an opened assistant turn.
_UNDERSTANDING_SUFFIX = f"{_IM_END}\n{_IM_START}assistant\n"

# ``--modality`` choices and the SenseNovaVision pipeline mode they map to.
# The ``think-*`` entries require the ``sensenova_vision_think`` topology
# (auto-selected in ``main()`` via ``THINK_MODALITIES``).
MODALITY_MODE = {
    "text2text": "understanding",
    "img2text": "understanding",
    "dense_detection": "dense_detection",
    "dense_OCR": "dense_OCR",
    "text2img": "generate",
    "img2img": "edit",
    "img2dense": "dense_perception",
    "multi-img2text": "understanding",
    "recon3d": "recon3d",
    "mixed": "caption_generate",
    "think-text2text": "think_understanding",
    "think-text2img": "think_generate",
    "think-img2img": "think_edit",
}

# Modalities that must run on the two-stage think topology: stage 0 decodes
# its <thinking> tokens to EOS before the KV cache transfers to the DiT.
THINK_MODALITIES = frozenset({"think-text2text", "think-text2img", "think-img2img"})

# Output-name prefixes per modality (deterministic filenames).
_MODALITY_PREFIX = {
    "text2text": "text2text",
    "img2text": "img2text",
    "dense_detection": "dense_detection",
    "dense_OCR": "dense_ocr",
    "text2img": "text2img",
    "img2img": "img2img",
    "img2dense": "img2dense",
    "multi-img2text": "multi_img2text",
    "recon3d": "recon3d",
    "mixed": "mixed",
    "think-text2text": "think_text2text",
    "think-text2img": "think_text2img",
    "think-img2img": "think_img2img",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SenseNova-Vision-7B-MoT offline inference (full modality matrix).")
    p.add_argument(
        "--model",
        default="RzZ/SenseNova-Vision-7B-MoT",
        help="HF repo or local path.",
    )
    p.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Input text prompts. Omitted prompts use the official SenseNovaVision defaults per mode.",
    )
    p.add_argument(
        "--modality",
        default="text2img",
        choices=sorted(MODALITY_MODE),
        help="SenseNova-Vision task modality.",
    )
    p.add_argument(
        "--image-path",
        nargs="+",
        default=None,
        help="Input image(s) for image-input modalities (repeatable / multiple values for multi-view).",
    )
    p.add_argument(
        "--dense-task",
        choices=["depth", "normal", "segmentation"],
        default="depth",
        help="Dense prediction task for --modality img2dense.",
    )
    p.add_argument("--height", type=int, default=None, help="Image height (image-output modes).")
    p.add_argument("--width", type=int, default=None, help="Image width (image-output modes).")
    p.add_argument("--output", type=str, default=".", help="Output directory.")
    p.add_argument("--steps", type=int, default=50, help="Denoising steps (SenseNovaVision default 50).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Path to the SenseNovaVision deploy YAML (two-stage thinker + DiT). "
        "Defaults to sensenova_vision.yaml, or sensenova_vision_think.yaml for the think-* modalities.",
    )
    p.add_argument(
        "--num-views",
        type=int,
        default=None,
        help="recon3d: number of output views to decode (default 4, upstream num_output_vae).",
    )
    p.add_argument(
        "--cfg-text-scale",
        type=float,
        default=None,
        help="Text CFG scale override (default from the per-mode BASE_PARAMS).",
    )
    p.add_argument(
        "--cfg-img-scale",
        type=float,
        default=None,
        help="Image CFG scale override (default from the per-mode BASE_PARAMS).",
    )
    p.add_argument(
        "--timestep-shift",
        type=float,
        default=None,
        help="Flow-match timestep shift override (default from the per-mode BASE_PARAMS).",
    )
    p.add_argument(
        "--max-think-tokens",
        type=int,
        default=None,
        help="Max think tokens for text-output modes (default from the per-mode BASE_PARAMS).",
    )
    p.add_argument(
        "--extra-args",
        type=str,
        default=None,
        help='Optional JSON object of extra pipeline args, e.g. \'{"cfg_renorm_type": "global"}\'.',
    )
    return p.parse_args()


def _load_images(image_paths):
    from PIL import Image

    if not image_paths:
        return []
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise ValueError(f"Image path does not exist: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def _single_image(image_paths):
    images = _load_images(image_paths)
    if not images:
        raise ValueError(f"--modality {args_modality()} requires at least one --image-path")
    return images[0]


# Small helper so the argument name is not confused with module-level ``args``.
def args_modality() -> str:
    return _ACTIVE_MODALITY


_ACTIVE_MODALITY = "text2img"


def _format_text2text_prompts(prompts):
    return [
        {
            "prompt": f"{_IM_START}user\n{p}{_UNDERSTANDING_SUFFIX}",
            "modalities": ["text"],
            "mode": "understanding",
        }
        for p in prompts
    ]


def _format_img2text_prompts(prompts, image):
    return [
        {
            "prompt": f"{_IM_START}user\n{_IMAGE_PAD_BLOCK}{p}{_UNDERSTANDING_SUFFIX}",
            "multi_modal_data": {"image": image},
            "modalities": ["text"],
            "mode": "understanding",
        }
        for p in prompts
    ]


def _format_dense_detection_prompts(prompts, image):
    return [
        {
            "prompt": f"{_IM_START}user\n{_IMAGE_PAD_BLOCK}{p}{_UNDERSTANDING_SUFFIX}",
            "multi_modal_data": {"image": image},
            "modalities": ["text"],
            "mode": "dense_detection",
        }
        for p in prompts
    ]


def _format_dense_ocr_prompts(prompts, image):
    return [
        {
            "prompt": f"{_IM_START}user\n{_IMAGE_PAD_BLOCK}{p}{_UNDERSTANDING_SUFFIX}",
            "multi_modal_data": {"image": image},
            "modalities": ["text"],
            "mode": "dense_OCR",
        }
        for p in prompts
    ]


def _format_text2img_prompts(prompts):
    # Generation scaffold: no user/assistant turns, <|im_start|> wraps the raw
    # prompt (matches ``build_text_to_image_prompt`` and the passing e2e test).
    return [{"prompt": f"{_IM_START}{p}{_IM_END}", "modalities": ["image"], "mode": "generate"} for p in prompts]


def _format_img2img_prompts(prompts, image):
    # img2img conditioning: <|fim_middle|> marks where the VAE+ViT block of the
    # input image is expanded; modalities must say "img2img" so the AR stage
    # routes embeddings through VAE+ViT (not ViT-only understanding).
    return [
        {
            "prompt": f"{_FIM_MIDDLE}{_IM_START}{p}{_IM_END}",
            "multi_modal_data": {"img2img": image},
            "modalities": ["img2img"],
            "mode": "edit",
        }
        for p in prompts
    ]


def _format_img2dense_prompts(prompts, image):
    return [
        {
            "prompt": f"{_FIM_MIDDLE}{_IM_START}{p}{_IM_END}",
            "multi_modal_data": {"img2img": image},
            "modalities": ["img2img"],
            "mode": "dense_perception",
        }
        for p in prompts
    ]


def _format_multi_img2text_prompts(prompts, images):
    # Multi-view understanding: one <|image_pad|> per input view, in input
    # order, before the prompt text (requires the SenseNova mm limit of 10).
    image_block = _IMAGE_PAD_BLOCK * len(images)
    return [
        {
            "prompt": f"{_IM_START}user\n{image_block}{p}{_UNDERSTANDING_SUFFIX}",
            "multi_modal_data": {"image": images},
            "modalities": ["text"],
            "mode": "understanding",
        }
        for p in prompts
    ]


def _format_recon3d_prompts(prompts, images):
    # Multi-view generation conditioning (mirrors upstream reconstruct_3d,
    # which interleaves [*images, prompt]): N <|fim_middle|> markers expand to
    # N VAE+ViT blocks, then the chat-wrapped prompt.  Order matters: the
    # first marker binds view 0.
    fim_block = _FIM_MIDDLE * len(images)
    return [
        {
            "prompt": f"{fim_block}{_IM_START}{p}{_IM_END}",
            "multi_modal_data": {"img2img": images},
            "modalities": ["img2img"],
            "mode": "recon3d",
        }
        for p in prompts
    ]


def _format_mixed_prompts(prompts, image):
    # caption_generate conditions on the image like edit/dense_perception
    # (<|fim_middle|> VAE+ViT block, modalities single-key img2img), but the
    # AR stage must decode a real caption that reaches EOS so the DiT is
    # conditioned on a COMPLETE interleaved caption.
    #
    # Marker parity (verified against SenseNova-Vision upstream): upstream
    # NEVER emits chat role words, and <|im_start|>/<|im_end|> ARE its bos/eos
    # special tokens.  Every training/inference text segment carries exactly
    # ONE pair ([bos]+ids+[eos], ``pack_sequence``/``prepare_prompts``), and
    # caption decoding STARTS from a freshly fed bos: ``prepare_start_tokens``
    # feeds [<|im_start|>] as the first decode step attending the whole
    # context (``Bagel.generate_text``).  So the AR must sample from a trailing
    # bare <|im_start|> opener:
    #   [fim block] [<|im_start|>]{p}[<|im_end|>] [<|im_start|>] -> caption -> [<|im_end|>]
    # Empirical GPU evidence for each variant: ending at [<|im_end|>] alone
    # degenerates to dot-fills (sampling after EOS is off-distribution);
    # a full "assistant" turn truncates early (role words are OOD); NO opener
    # at all degenerates to token loops.  Role words/newlines stay out.
    # Also strip the official ``<image>`` placeholder: upstream generate()
    # splits questions on it before tokenization (never reaches the model);
    # the image rides on multi_modal_data["img2img"] here.
    return [
        {
            "prompt": (f"{_FIM_MIDDLE}{_IM_START}{p.replace('<image>', '').strip()}{_IM_END}{_IM_START}"),
            "multi_modal_data": {"img2img": image},
            "modalities": ["img2img"],
            "mode": "caption_generate",
        }
        for p in prompts
    ]


def _format_think_text2text_prompts(prompts):
    from vllm_omni.model_executor.models.sensenova_vision.prompt_utils import build_think_prompt

    return [
        {
            "prompt": f"{build_think_prompt(p, mode='think_understanding')}{_UNDERSTANDING_SUFFIX}",
            "modalities": ["text"],
            "mode": "think_understanding",
        }
        for p in prompts
    ]


def _format_think_text2img_prompts(prompts):
    from vllm_omni.model_executor.models.sensenova_vision.prompt_utils import build_think_prompt

    # Think image generation: the think system+user wrapper replaces the plain
    # <|im_start|>{p}<|im_end|> generation scaffold; prepare_prompts adds the
    # port's single bos/eos pair around it (see prompt_utils docstring caveat).
    return [{"prompt": build_think_prompt(p), "modalities": ["image"], "mode": "think_generate"} for p in prompts]


def _format_think_img2img_prompts(prompts, image):
    from vllm_omni.model_executor.models.sensenova_vision.prompt_utils import build_think_prompt

    return [
        {
            "prompt": f"{_FIM_MIDDLE}{build_think_prompt(p)}",
            "multi_modal_data": {"img2img": image},
            "modalities": ["img2img"],
            "mode": "think_edit",
        }
        for p in prompts
    ]


def _extract_text(req_output) -> str:
    """Extract generated text from an OmniRequestOutput (lance end2end.py style).

    Diffusion-side outputs (mixed/dense_perception) surface text under
    ``multimodal_output["text"]`` / ``metadata.text.text_output``; pure-AR
    text modalities (img2text, text2text, dense_detection, dense_OCR,
    multi-img2text) run entirely on the AR stage and carry the decoded text
    on the vLLM completion output at ``outputs[0].text``.
    """
    multimodal_output = getattr(req_output, "multimodal_output", {}) or {}
    metadata = multimodal_output.get("metadata", {}) if isinstance(multimodal_output, dict) else {}
    text_metadata = metadata.get("text", {}) if isinstance(metadata, dict) else {}
    text = (multimodal_output.get("text") if isinstance(multimodal_output, dict) else None) or (
        text_metadata.get("text_output") if isinstance(text_metadata, dict) else None
    )
    if not text:
        # AR-only text stages surface the decoded generation on the completion
        # output rather than any multimodal payload.
        outputs = getattr(req_output, "outputs", None) or []
        if outputs:
            first = outputs[0]
            text = getattr(first, "text", None) or getattr(first, "cumulative_text", None)
    return text or getattr(req_output, "output", None) or getattr(req_output, "text", None)


def _write_text(output_dir, prefix, index, text) -> str:
    path = os.path.join(output_dir, f"{prefix}_{index}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text is not None else "")
    return path


def _write_image(output_dir, prefix, index, image) -> str:
    path = os.path.join(output_dir, f"{prefix}_{index}.png")
    image.save(path)
    return path


def _decode_and_write_dense(output_dir, prefix, index, image, dense_task):
    """Decode a dense prediction image with the SenseNovaVision decoders and save it."""
    import numpy as np

    from vllm_omni.model_executor.models.sensenova_vision.decoders import (
        decode_depth,
        decode_normal,
        decode_segmentation,
    )

    if dense_task == "depth":
        arr = decode_depth(image)
        path = os.path.join(output_dir, f"{prefix}_{index}_depth.npy")
        np.save(path, arr)
    elif dense_task == "normal":
        arr = decode_normal(image)
        path = os.path.join(output_dir, f"{prefix}_{index}_normal.npy")
        np.save(path, arr)
    else:
        arr = decode_segmentation(image)
        path = os.path.join(output_dir, f"{prefix}_{index}_segmentation.npy")
        np.save(path, arr)
    raw_path = _write_image(output_dir, prefix, index, image)
    print(f"  → decoded {dense_task} map to {path} (raw prediction: {raw_path})")
    return path


def _decode_and_write_recon3d(output_dir, prefix, index, images):
    """Decode each per-view point map and optionally save the intermediate text."""
    import numpy as np

    from vllm_omni.model_executor.models.sensenova_vision.decoders import decode_point_map

    paths = []
    for j, image in enumerate(images):
        arr = decode_point_map(image)
        path = os.path.join(output_dir, f"{prefix}_{index}_view{j}.npy")
        np.save(path, arr)
        paths.append(path)
    return paths


def main():
    global _ACTIVE_MODALITY
    args = parse_args()
    _ACTIVE_MODALITY = args.modality

    os.makedirs(args.output, exist_ok=True)
    os.environ.setdefault("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN")

    # Official default prompts per mode.
    default_prompts = {
        "text2text": ["What is the capital of France?"],
        "img2text": [UNDERSTANDING_PROMPT],
        "dense_detection": [DETECTION_PROMPT],
        "dense_OCR": [OCR_PROMPT],
        "text2img": ["A cute corgi astronaut on the moon, cinematic"],
        "img2img": [EDIT_PROMPT],
        "img2dense": {
            "depth": [DEPTH_PROMPT],
            "normal": [NORMAL_PROMPT],
            "segmentation": [SEGMENTATION_PROMPT],
        },
        "multi-img2text": [CAMERA_POSE_PROMPT],
        "recon3d": [RECON3D_PROMPT],
        "mixed": [CAPTION_GENERATE_PROMPT],
        "think-text2text": [UNDERSTANDING_PROMPT],
        "think-text2img": ["A cute corgi astronaut on the moon, cinematic"],
        "think-img2img": [EDIT_PROMPT],
    }
    if args.modality == "img2dense":
        prompts = args.prompts or default_prompts["img2dense"][args.dense_task]
    else:
        prompts = args.prompts or default_prompts[args.modality]

    images = _load_images(args.image_path)
    if args.modality in (
        "img2text",
        "dense_detection",
        "dense_OCR",
        "img2img",
        "img2dense",
        "mixed",
        "think-img2img",
    ):
        if not images:
            raise ValueError(f"--modality {args.modality} requires at least one --image-path")
        image = images[0]
    elif args.modality in ("multi-img2text", "recon3d"):
        if len(images) < 2:
            raise ValueError(f"--modality {args.modality} requires at least two --image-path values")
    elif args.modality in ("text2img", "think-text2img", "text2text", "think-text2text") and not args.image_path:
        pass

    # Think modalities need the two-stage think topology (stage 0 decodes its
    # <thinking> tokens before KV transfer); everything else uses the default
    # transfer-at-prefill topology.  An explicit --deploy-config always wins.
    if args.deploy_config is None:
        args.deploy_config = (
            "vllm_omni/deploy/sensenova_vision_think.yaml"
            if args.modality in THINK_MODALITIES
            else "vllm_omni/deploy/sensenova_vision.yaml"
        )

    omni_kwargs = {
        "model": args.model,
        "deploy_config": args.deploy_config,
    }
    omni = Omni(**omni_kwargs)

    if args.modality == "text2text":
        formatted = _format_text2text_prompts(prompts)
    elif args.modality == "img2text":
        formatted = _format_img2text_prompts(prompts, image)
    elif args.modality == "dense_detection":
        formatted = _format_dense_detection_prompts(prompts, image)
    elif args.modality == "dense_OCR":
        formatted = _format_dense_ocr_prompts(prompts, image)
    elif args.modality == "text2img":
        formatted = _format_text2img_prompts(prompts)
    elif args.modality == "img2img":
        formatted = _format_img2img_prompts(prompts, image)
    elif args.modality == "img2dense":
        formatted = _format_img2dense_prompts(prompts, image)
    elif args.modality == "multi-img2text":
        formatted = _format_multi_img2text_prompts(prompts, images)
    elif args.modality == "recon3d":
        formatted = _format_recon3d_prompts(prompts, images)
    elif args.modality == "think-text2text":
        formatted = _format_think_text2text_prompts(prompts)
    elif args.modality == "think-text2img":
        formatted = _format_think_text2img_prompts(prompts)
    elif args.modality == "think-img2img":
        formatted = _format_think_img2img_prompts(prompts, image)
    else:  # mixed
        formatted = _format_mixed_prompts(prompts, image)

    params_list = omni.default_sampling_params_list
    diffusion_params = params_list[0]  # single-stage: one param set
    diffusion_params.num_inference_steps = args.steps  # type: ignore
    if args.seed is not None:
        diffusion_params.seed = args.seed  # type: ignore

    extra = getattr(diffusion_params, "extra_args", {}) or {}

    # Output resolution: recon3d defaults to the recon3d_vae_transform target
    # (ImageTransform(512, 256, 16)); other modes use the checkpoint max.
    default_hw = (RECON3D_HEIGHT, RECON3D_WIDTH) if args.modality == "recon3d" else (None, None)
    if args.height is not None or default_hw[0] is not None:
        diffusion_params.height = args.height if args.height is not None else default_hw[0]  # type: ignore
    if args.width is not None or default_hw[1] is not None:
        diffusion_params.width = args.width if args.width is not None else default_hw[1]  # type: ignore

    if args.modality == "recon3d":
        if args.num_views is not None:
            extra["num_views"] = args.num_views
            if len(images) > args.num_views:
                raise ValueError(
                    f"recon3d got {len(images)} input views but --num-views {args.num_views}; "
                    "the output view count must cover every input view"
                )
        if args.cfg_text_scale is None:
            # Benchmark default (inference/benchmark/batch_recon3d.py:87-93):
            # overrides BASE_PARAMS' cfg_text_scale 1.0 with 4.0.
            extra["cfg_text_scale"] = 4.0

    if args.cfg_text_scale is not None:
        extra["cfg_text_scale"] = args.cfg_text_scale
    if args.cfg_img_scale is not None:
        extra["cfg_img_scale"] = args.cfg_img_scale
    if args.timestep_shift is not None:
        extra["timestep_shift"] = args.timestep_shift
    if args.max_think_tokens is not None:
        extra["max_think_tokens"] = args.max_think_tokens
    if args.extra_args:
        try:
            extra.update(json.loads(args.extra_args))
        except json.JSONDecodeError as e:
            raise ValueError(f"--extra-args must be valid JSON: {e}") from e
    diffusion_params.extra_args = extra  # type: ignore

    outputs = list(omni.generate(prompts=formatted, sampling_params_list=params_list))
    prefix = _MODALITY_PREFIX[args.modality]

    text_modalities = {
        "text2text",
        "img2text",
        "dense_detection",
        "dense_OCR",
        "multi-img2text",
        "think-text2text",
    }

    if args.modality in text_modalities:
        for i, req_output in enumerate(outputs):
            text = _extract_text(req_output)
            path = _write_text(args.output, prefix, i, text)
            print(f"[Output {i}] Saved text to {path}")
            if args.modality == "dense_detection":
                from vllm_omni.model_executor.models.sensenova_vision.decoders import parse_bbox

                parsed = parse_bbox(text or "")
                print(f"  → parsed detections: {json.dumps(parsed, indent=2)}")
            elif args.modality == "dense_OCR":
                from vllm_omni.model_executor.models.sensenova_vision.decoders import parse_points

                parsed = parse_points(text or "")
                print(f"  → parsed OCR boxes: {json.dumps(parsed, indent=2)}")
            elif args.modality == "multi-img2text":
                from vllm_omni.model_executor.models.sensenova_vision.decoders import parse_camera_pose

                parsed = parse_camera_pose(text or "")
                print(f"  → parsed camera pose: {json.dumps(parsed, indent=2) if parsed else 'unparsable'}")
        return

    if args.modality == "img2dense":
        for i, req_output in enumerate(outputs):
            images_out = getattr(req_output, "images", None) or []
            for j, img in enumerate(images_out):
                _decode_and_write_dense(args.output, prefix, i, img, args.dense_task)
        return

    if args.modality == "recon3d":
        for i, req_output in enumerate(outputs):
            images_out = getattr(req_output, "images", None) or []
            paths = _decode_and_write_recon3d(args.output, prefix, i, images_out)
            text = _extract_text(req_output)
            if text:
                text_path = _write_text(args.output, prefix, i, text)
                print(f"  → saved optional recon3d text to {text_path}")
            for p in paths:
                print(f"  → saved per-view point map to {p}")
        return

    if args.modality == "mixed":
        for i, req_output in enumerate(outputs):
            images_out = getattr(req_output, "images", None) or []
            text = _extract_text(req_output)
            if text:
                text_path = _write_text(args.output, prefix, i, text)
                print(f"[Output {i}] Saved intermediate text to {text_path}")
            for j, img in enumerate(images_out):
                path = _write_image(args.output, prefix, i, img)
                print(f"[Output {i}] Saved image to {path}")
        return

    # Plain image outputs (text2img / img2img).
    for i, req_output in enumerate(outputs):
        images_out = getattr(req_output, "images", None) or []
        for j, img in enumerate(images_out):
            path = _write_image(args.output, prefix, i, img)
            print(f"[Output {i}] Saved image to {path}")


if __name__ == "__main__":
    main()
