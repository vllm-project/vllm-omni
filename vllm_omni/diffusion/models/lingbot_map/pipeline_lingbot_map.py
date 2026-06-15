"""LingBot-MAP pipeline for vLLM-Omni: 3D reconstruction from images.

Takes a set of images via OmniDiffusionRequest.prompts[0]["multi_modal_data"]["image"]
and produces camera parameters (intrinsic, extrinsic), depth maps, and point maps.

Output is placed in DiffusionOutput.output as a dict, which flows into
OmniRequestOutput.multimodal_output.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as TF

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest


# ---------------------------------------------------------------------------
# In-memory image preprocessing (mirrors lingbot_map.utils.load_fn:
# load_and_preprocess_images in "crop" mode)
# ---------------------------------------------------------------------------

def _preprocess_images(
    images: list[Image.Image],
    image_size: int = 518,
    patch_size: int = 14,
) -> torch.Tensor:
    """Preprocess a list of PIL Images in memory into a batched tensor.

    Follows the same logic as ``load_and_preprocess_images(mode="crop")``:
    EXIF-correct, RGBA→RGB, resize width to *image_size*, scale height
    proportionally (rounded to *patch_size*), centre-crop height if taller,
    then pad (white) to unify shapes.
    """
    to_tensor = TF.ToTensor()
    processed = []

    for img in images:
        img = ImageOps.exif_transpose(img)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        new_width = image_size
        new_height = round(height * (new_width / width) / patch_size) * patch_size

        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)

        if new_height > image_size:
            start_y = (new_height - image_size) // 2
            img = img[:, start_y : start_y + image_size, :]

        processed.append(img)

    # Unify shapes (pad smaller images with white = 1.0)
    shapes = set((t.shape[1], t.shape[2]) for t in processed)
    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for t in processed:
            h_pad = max_h - t.shape[1]
            w_pad = max_w - t.shape[2]
            if h_pad > 0 or w_pad > 0:
                top = h_pad // 2
                bottom = h_pad - top
                left = w_pad // 2
                right = w_pad - left
                t = torch.nn.functional.pad(
                    t, (left, right, top, bottom), mode="constant", value=1.0
                )
            padded.append(t)
        processed = padded

    return torch.stack(processed)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LingbotMapPipeline(nn.Module):
    """LingBot-MAP 3D reconstruction pipeline for vLLM-Omni.

    Loads a GCTStream checkpoint and runs streaming inference over a set of
    input images, producing camera parameters and depth/point maps.

    Class attributes:
        support_image_input: True — declares this pipeline accepts image input
        color_format: "RGB" — expected input colour space
        dummy_run_num_frames: 0 — skip the engine dummy warmup (requires a
            real checkpoint on disk, so warming would add startup latency)
    """

    support_image_input: bool = True
    color_format: str = "RGB"
    dummy_run_num_frames: int = 0

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config
        self._execution_device = get_local_device()

        extras: dict[str, Any] = getattr(od_config, "extras", {}) or {}

        self.image_size: int = extras.get("image_size", 518)
        self.patch_size: int = extras.get("patch_size", 14)
        self.num_scale_frames: int = extras.get("num_scale_frames", 8)
        self.keyframe_interval: int | None = extras.get("keyframe_interval", None)
        self.mode: str = extras.get("mode", "streaming")
        self.window_size: int = extras.get("window_size", 64)
        self.enable_3d_rope: bool = extras.get("enable_3d_rope", True)
        self.max_frame_num: int = extras.get("max_frame_num", 1024)
        self.kv_cache_sliding_window: int = extras.get("kv_cache_sliding_window", 64)
        self.camera_num_iterations: int = extras.get("camera_num_iterations", 4)
        self.use_sdpa: bool = extras.get("use_sdpa", False)

        # Resolve checkpoint path: explicit extras key first, then look for
        # checkpoint files inside the model directory.
        model_path: str | None = extras.get("model_path")
        if not model_path:
            import os
            model_dir = od_config.model or ""
            candidates = [
                os.path.join(model_dir, "checkpoint.pt"),
                os.path.join(model_dir, "model.pt"),
                os.path.join(model_dir, "lingbot-map.pt"),
                os.path.join(model_dir, "lingbot-map-long.pt"),
                os.path.join(model_dir, "lingbot-map-stage1.pt"),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    model_path = cand
                    break
            if not model_path:
                raise FileNotFoundError(
                    f"Cannot find checkpoint in {model_dir}. "
                    "Place a .pt checkpoint alongside config.json, or set "
                    "extras.model_path."
                )
        self.model_path: str = model_path

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from vllm_omni.diffusion.models.lingbot_map._model.models.gct_stream import GCTStream

        device = self._execution_device

        if device.type == "cuda":
            self.dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability()[0] >= 8
                else torch.float16
            )
        else:
            self.dtype = torch.float32

        self.model = GCTStream(
            img_size=self.image_size,
            patch_size=self.patch_size,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
            camera_num_iterations=self.camera_num_iterations,
        )

        ckpt = torch.load(self.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # DiffusersPipelineLoader wraps this call inside
        # ``set_default_torch_dtype(bf16)``, so GCTStream parameters are
        # created as bf16.  load_state_dict does NOT restore fp32 dtype
        # (PyTorch silently truncates to the parameter's existing dtype).
        # Cast the whole model to fp32 first so the checkpoint loads at
        # full precision, then selectively cast aggregator to bf16 to
        # match demo.py's mixed-precision setup.
        self.model = self.model.float()
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device).eval()

        # Mirror demo.py: aggregator in bf16, heads stay fp32.
        if self.dtype != torch.float32:
            agg = getattr(self.model, "aggregator", None)
            if agg is not None:
                self.model.aggregator = agg.to(dtype=self.dtype)


        # Mirror demo.py: cast only the aggregator to the inference dtype.
        # The camera / depth / point heads stay at their checkpoint dtype
        # (fp32).  GCTBase internally upcasts aggregator output to fp32 and
        # disables autocast for the heads, so heads always see fp32.
        if self.dtype != torch.float32 and getattr(self.model, "aggregator", None) is not None:
            print(f"Casting aggregator to {self.dtype} (heads kept in fp32)")
            self.model.aggregator = self.model.aggregator.to(dtype=self.dtype)

    # ------------------------------------------------------------------
    # Input extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_images(req: OmniDiffusionRequest) -> list[Image.Image]:
        """Pull the list of input PIL Images from the request."""
        if not req.prompts:
            return []

        first = req.prompts[0]
        if isinstance(first, dict):
            mm = first.get("multi_modal_data", {})
            raw = mm.get("image", [])
        else:
            raw = []

        if isinstance(raw, Image.Image):
            return [raw]
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, req: OmniDiffusionRequest, **kwargs: Any) -> DiffusionOutput:
        device = self._execution_device

        # --- dummy run guard ---
        if req.is_dummy_run():
            return DiffusionOutput(output={
                "extrinsic": torch.zeros(0, 3, 4),
                "intrinsic": torch.zeros(0, 3, 3),
                "depth": torch.zeros(0, 1, 1, 1),
                "depth_conf": torch.zeros(0, 1, 1),
                "world_points": torch.zeros(0, 1, 1, 3),
                "world_points_conf": torch.zeros(0, 1, 1),
            })

        # --- extract & preprocess images ---
        pil_images = self._extract_images(req)
        if not pil_images:
            raise ValueError(
                "LingbotMapPipeline requires at least one image in "
                "prompt['multi_modal_data']['image']"
            )

        images = _preprocess_images(pil_images, self.image_size, self.patch_size)
        # Cast input to the same dtype as the aggregator so conv2d sees
        # matching dtypes without needing autocast.  GCTBase internally
        # upcasts aggregator output back to fp32 for the heads.
        images = images.to(device=device, dtype=self.dtype)
        num_frames = images.shape[0]

        # --- keyframe interval ---
        kf_interval = self.keyframe_interval
        if kf_interval is None:
            kf_interval = (
                (num_frames + 319) // 320 if num_frames > 320 else 1
            )

        # --- inference (no autocast — input already matches model dtype) ---
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.dtype):
            predictions = self.model.inference_streaming(
                images,
                num_scale_frames=self.num_scale_frames,
                keyframe_interval=kf_interval,
            )

        # --- post-process pose encoding → extrinsic / intrinsic ---
        self._attach_camera_params(predictions)

        # --- strip batch dim & move to CPU ---
        result: dict[str, Any] = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                if value.ndim >= 1 and value.shape[0] == 1:
                    value = value[0]
                result[key] = value.detach().cpu()

        # Post_process wraps result in {"video": [], "custom_output": result},
        # so the engine routes our 3D data into both OmniRequestOutput.custom_output
        # and OmniRequestOutput.multimodal_output (via engine merge), while
        # images stays empty.
        return DiffusionOutput(output=result)

    @staticmethod
    def _attach_camera_params(predictions: dict[str, Any]) -> None:
        """Convert ``pose_enc`` to extrinsic (c2w) + intrinsic in-place."""
        from vllm_omni.diffusion.models.lingbot_map._model.utils.pose_enc import pose_encoding_to_extri_intri
        from vllm_omni.diffusion.models.lingbot_map._model.utils.geometry import closed_form_inverse_se3_general

        pose_enc = predictions.get("pose_enc")
        if pose_enc is None:
            return

        h, w = predictions["images"].shape[-2:]
        extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(pose_enc, (h, w))

        # w2c → c2w
        ext_4x4 = torch.zeros(
            (*extrinsic_w2c.shape[:-2], 4, 4),
            device=extrinsic_w2c.device,
            dtype=extrinsic_w2c.dtype,
        )
        ext_4x4[..., :3, :4] = extrinsic_w2c
        ext_4x4[..., 3, 3] = 1.0
        ext_4x4 = closed_form_inverse_se3_general(ext_4x4)
        extrinsic = ext_4x4[..., :3, :4]

        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions.pop("pose_enc_list", None)
        predictions.pop("images", None)

    # ------------------------------------------------------------------
    # Weight loading — weights are already loaded in _load_model via
    # torch.load + load_state_dict, so we report all parameter names.
    # ------------------------------------------------------------------

    def load_weights(self, weights: Any) -> set[str]:
        # Weights already loaded in _load_model(); report all names with the
        # "model." prefix the loader expects (pipeline owns GCTStream as
        # self.model).
        return {f"model.{name}" for name, _ in self.model.named_parameters()}


# ---------------------------------------------------------------------------
# Post-process function (registered in registry.py)
# ---------------------------------------------------------------------------

def get_lingbot_map_post_process_func(od_config: OmniDiffusionConfig):
    """Return a post-process function that routes 3D data correctly.

    The engine looks for ``"video"`` (→ images), ``"audio"``,
    ``"actions"``, ``"custom_output"``, and ``"multimodal_output"`` keys
    in the returned dict.  By setting ``video=[]`` we produce empty images,
    and by placing the predictions in ``custom_output`` the engine merges
    them into ``OmniRequestOutput.multimodal_output`` (via the
    custom_output→mm_output merge).
    """

    def post_process(output: dict[str, Any]) -> dict[str, Any]:
        return {"video": [], "custom_output": output}

    return post_process
