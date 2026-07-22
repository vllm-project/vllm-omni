#!/usr/bin/env python3
"""
HTTP server for text-to-video generation with inter-request cache support.

Supports Wan2.2 and other T2V models. Returns base64-encoded MP4.

Usage:
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python3 run_server_with_cache_video.py

Endpoints:
    POST /v1/videos/generations  - Generate video (with cache support)
    GET  /health                 - Health check
"""

import argparse
import base64
import io
import json
import logging
import os
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

# Force unbuffered stdout
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
logger = logging.getLogger(__name__)

omni = None
SAMPLING_PARAMS_CLS = None


def _frames_to_mp4_base64(frames: np.ndarray, fps: int = 24) -> str:
    """Convert a list of frames (HWC uint8 numpy arrays) to base64-encoded MP4."""
    try:
        import imageio

        # Debug: log frame info
        logger.info("Encoding video: %d frames, shape=%s dtype=%s", len(frames), frames[0].shape if frames else None, frames[0].dtype if frames else None)

        buf = io.BytesIO()
        writer = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264")
        for frame in frames:
            # Ensure uint8 HWC with 3 channels
            if frame.dtype != np.uint8:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            writer.append_data(frame)
        writer.close()
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.error("Failed to encode video: %s", e)
        return ""


class VideoCacheHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info("%s - %s", self.client_address[0], format % args)

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/videos/generations":
            self._handle_generate()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_generate(self):
        global omni, SAMPLING_PARAMS_CLS
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return

        prompt = req.get("prompt", "")
        negative_prompt = req.get("negative_prompt")
        seed = req.get("seed", random.randint(0, 2**32 - 1))
        width = req.get("width", 832)
        height = req.get("height", 480)
        size_str = req.get("size")
        if size_str and "x" in str(size_str):
            parts = str(size_str).split("x")
            if width is None:
                width = int(parts[0])
            if height is None:
                height = int(parts[1])
        steps = req.get("num_inference_steps", 40)
        num_frames = req.get("num_frames", 81)
        fps = req.get("fps", 24)
        guidance_scale = req.get("guidance_scale", 4.0)
        guidance_scale_high = req.get("guidance_scale_high")
        flow_shift = req.get("flow_shift")
        resume_from_step = req.get("resume_from_step", 0) or 0

        prompt_dict = {"prompt": prompt}
        if negative_prompt:
            prompt_dict["negative_prompt"] = negative_prompt

        sampling_params = SAMPLING_PARAMS_CLS(
            height=height,
            width=width,
            seed=seed,
            num_inference_steps=steps,
            num_frames=num_frames,
            fps=fps,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_high,
            num_outputs_per_prompt=1,
            resume_from_step=resume_from_step,
        )

        start = time.perf_counter()
        logger.info("=" * 60)
        logger.info(
            "REQUEST: prompt=%r seed=%d steps=%d frames=%d %dx%d",
            prompt[:80],
            seed,
            steps,
            num_frames,
            width,
            height,
        )
        try:
            outputs = omni.generate(prompt_dict, sampling_params)
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            self._send_json(500, {"error": str(e)})
            return
        elapsed = time.perf_counter() - start
        logger.info("COMPLETED in %.2fs", elapsed)
        logger.info("=" * 60)

        videos = []
        for out in outputs:
            inner = out.request_output
            if inner and inner.images:
                # Video output: inner.images may contain a single 5D numpy array
                # [batch, frames, H, W, C] or a list of per-frame arrays.
                frames = []
                for img in inner.images:
                    if hasattr(img, "convert"):
                        # PIL Image
                        frames.append(np.array(img.convert("RGB")))
                    elif isinstance(img, np.ndarray):
                        if img.ndim == 5:
                            # [batch, frames, H, W, C] -> flatten batch, iterate frames
                            for b in range(img.shape[0]):
                                for f_idx in range(img.shape[1]):
                                    frames.append(img[b, f_idx])
                        elif img.ndim == 4:
                            # [frames, H, W, C]
                            for f_idx in range(img.shape[0]):
                                frames.append(img[f_idx])
                        elif img.ndim == 3:
                            frames.append(img)
                if frames:
                    b64 = _frames_to_mp4_base64(frames, fps=fps)
                    videos.append({"b64_json": b64})

        self._send_json(
            200,
            {
                "created": int(time.time()),
                "data": videos,
                "time_ms": elapsed * 1000,
            },
        )


def main():
    global omni, SAMPLING_PARAMS_CLS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/sdb/models/Wan2.2-T2V-A14B-Diffusers")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--cache-backend", default="inter_request")
    parser.add_argument("--persistent-cache-dir", default="./persistent_cache_video")
    parser.add_argument(
        "--lmcache-disk-dir", default=None,
        help="LMCache disk directory for CPU→Disk tiering. When set, latent "
             "tensors are stored via LMCache ECCacheEngine with built-in LRU.",
    )
    parser.add_argument("--lmcache-max-cpu-gb", type=float, default=5.0)
    parser.add_argument("--lmcache-max-disk-gb", type=float, default=100.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-entries", type=int, default=1000)
    parser.add_argument("--max-memory-gb", type=float, default=200.0)
    parser.add_argument("--clip-model-path", default=None, help="Path to CLIP model for semantic matching")
    parser.add_argument("--clip-threshold", type=float, default=0.75, help="CLIP similarity threshold")
    parser.add_argument(
        "--clip-min-skip", type=int, default=5, help="Minimum skip steps when similarity just exceeds threshold"
    )
    parser.add_argument(
        "--clip-max-skip-ratio", type=float, default=0.5, help="Max skip ratio of total steps when similarity=1.0"
    )
    parser.add_argument(
        "--no-t2i-penalty", action="store_true", help="Disable t2i sigmoid penalty (recommended for video)"
    )
    # cache_dit parameters
    parser.add_argument("--fn-compute-blocks", type=int, default=1)
    parser.add_argument("--bn-compute-blocks", type=int, default=0)
    parser.add_argument("--max-warmup-steps", type=int, default=4)
    parser.add_argument("--residual-diff-threshold", type=float, default=0.24)
    args = parser.parse_args()

    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    SAMPLING_PARAMS_CLS = OmniDiffusionSamplingParams

    cache_config = {
        "inter_request_max_entries": args.max_entries,
        "inter_request_max_memory_gb": args.max_memory_gb,
        "inter_request_persistent_cache_dir": args.persistent_cache_dir,
    }
    if args.lmcache_disk_dir:
        cache_config["inter_request_lmcache_disk_dir"] = args.lmcache_disk_dir
        cache_config["inter_request_lmcache_max_cpu_gb"] = args.lmcache_max_cpu_gb
        cache_config["inter_request_lmcache_max_disk_gb"] = args.lmcache_max_disk_gb
    if args.clip_model_path:
        cache_config["inter_request_clip_model_path"] = args.clip_model_path
        cache_config["inter_request_clip_threshold"] = args.clip_threshold
        cache_config["inter_request_clip_min_skip"] = args.clip_min_skip
        cache_config["inter_request_clip_max_skip_ratio"] = args.clip_max_skip_ratio
    cache_config["inter_request_use_t2i_penalty"] = False  # Video is 5D, no image embedding

    if "cache_dit" in args.cache_backend:
        cache_config["Fn_compute_blocks"] = args.fn_compute_blocks
        cache_config["Bn_compute_blocks"] = args.bn_compute_blocks
        cache_config["max_warmup_steps"] = args.max_warmup_steps
        cache_config["residual_diff_threshold"] = args.residual_diff_threshold

    logger.info("Initializing Omni engine (text-to-video)...")
    logger.info("  Model: %s", args.model)
    logger.info("  Cache backend: %s", args.cache_backend)
    logger.info("  Persistent cache dir: %s", args.persistent_cache_dir)

    omni = Omni(
        model=args.model,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        tensor_parallel_size=args.tensor_parallel_size,
        mode="text-to-video",
        init_timeout=3600,
        enable_cache_dit_summary=True,
    )

    server = HTTPServer((args.host, args.port), VideoCacheHandler)
    logger.info("Server listening on %s:%d", args.host, args.port)
    logger.info("POST /v1/videos/generations - Generate video")
    logger.info("GET  /health - Health check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.server_close()
        omni.shutdown()


if __name__ == "__main__":
    main()
