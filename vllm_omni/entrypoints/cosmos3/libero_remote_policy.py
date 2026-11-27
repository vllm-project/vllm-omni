# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared LIBERO frontend for a remote vLLM model or a local native model."""

import pybase64 as base64
import io
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vllm_omni.entrypoints.cosmos3.action_policy_common import (
    DEFAULT_LIBERO_STATS,
    LiberoActionNormalizer,
    decode_policy_image,
)
from vllm_omni.entrypoints.cosmos3.libero_policy_inputs import prepare_libero_policy_inputs
from vllm_omni.entrypoints.cosmos3.remote_action_stream import ActionStreamChannel, ActionStreamRegistry
from vllm_omni.entrypoints.cosmos3.vllm_omni_action import VllmOmniActionTransport

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemotePolicyConfig:
    action_stats_path: Path = DEFAULT_LIBERO_STATS
    dump_dir: Path | None = None
    dump_every: int = 1
    http_400_on_error: bool = True
    streaming_enabled: bool = False
    callback_base_url: str | None = None


class LiberoRemotePolicyService:
    """Reuse the native HTTP protocol, returning physical actions exactly once."""

    # 判断是否使用流式
    @property
    def supports_streaming(self) -> bool:
        return self.cfg.streaming_enabled and self.transport.action_time_schedule == "has_euler"

    # 初始化推理服务  VllmOmniActionTransport推理后端接口
    def __init__(self, transport: VllmOmniActionTransport, cfg: RemotePolicyConfig = RemotePolicyConfig()) -> None:
        if cfg.dump_every < 1:
            raise ValueError("dump_every must be positive")
        self.transport = transport
        self.cfg = cfg
        self.normalizer = LiberoActionNormalizer(cfg.action_stats_path)
        self._lock = threading.Lock()
        self._profile_confirmed = False
        self.streams = ActionStreamRegistry()  # 注册并生成token
        self._warmup_results: list[dict[str, Any]] = []

    def check_health(self) -> bool:
        with self._lock:
            return self.transport.check_health()

    # 向客户端返回服务配置和推理能力
    def get_info(self) -> dict[str, Any]:
        with self._lock:
            info = self.transport.get_info()
        return {
            **info,
            "backend": getattr(self.transport, "policy_backend", "vllm-omni"),
            "run_name": self.transport.policy_profile,
            "checkpoint": info["model"],
            "max_action_dim": 64,
            "action_stats_path": str(self.cfg.action_stats_path),
            "return_video": False,
            "video_decode_enabled": False,
            "early_stop_actions": self.transport.early_stop_actions,
            "profile_confirmed": self._profile_confirmed,
            "timing_schema": "pi_libero_v1",
            "synthetic_warmup": {
                "supported": True,
                "default_requests": 2,
                "seed_scope": "explicit_fixed_seed_per_job; synthetic_images_use_local_rng",
                "completed_requests": len(self._warmup_results),
            },
            "diagnostics": {"dump_enabled": self.cfg.dump_dir is not None, "video_enabled": False},
            "timing_scope": "complete_policy_call_wall_time",
            "measure_ttfa": True,
            "ttfa_scope": "policy_entry_to_first_ready_physical_action; nonstream_waits_for_full_chunk",
            "streaming": {
                "supported": self.supports_streaming,
                "protocol": "application/x-ndjson",
                "endpoints": ["/predict_stream", "/predict_batch_stream"] if self.supports_streaming else [],
            },
            "timing_scopes": {
                "server_infer_ms": "policy_entry_to_complete_physical_actions_before_response_serialization",
                "policy_infer_ms": "complete_policy_call_wall_time; same_boundary_as_pi_benchmark_policy_wall_ms",
                "policy_wall_ms": "complete_policy_call_including_preprocessing_queue_backend_callbacks_and_dumps",
                "backend_roundtrip_ms": "remote_jobs_and_stream_callbacks_after_policy_lock_acquisition",
                "worker_inference_ms": "vllm_video_job_generation_and_output_persistence_wall_time",
                "server_ttfa_ms": "policy_entry_to_first_physical_action_ready_for_http_delivery",
            },
        }

    # 预热
    def warmup(self, req: dict[str, Any]) -> dict[str, Any]:
        """Warm the selected request path without changing evaluation RNG state."""
        count = req.get("requests", 2)
        batch_size = req.get("batch_size", 1)
        streaming = req.get("streaming", False)
        if type(count) is not int or count < 0 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Warmup requires nonnegative requests and positive batch_size")
        if type(streaming) is not bool:
            raise ValueError("Warmup streaming must be a boolean")
        if self.cfg.dump_dir is not None:
            raise ValueError("Disable --dump-dir before benchmark warmup")
        rng = np.random.default_rng(self.transport.seed)
        results = []
        for _ in range(count):
            items = []
            for _ in range(batch_size):
                image = rng.integers(0, 256, size=(256, 512, 3), dtype=np.uint8)
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="PNG")
                items.append(
                    {
                        "image": base64.b64encode(buffer.getvalue()).decode(),
                        "prompt": "pick up the object",
                        "domain_name": "libero",
                        "image_size": 256,
                    }
                )
            result = self.predict_distributed(
                {"items": items} if batch_size > 1 else items[0],
                is_batch=batch_size > 1,
                on_actions_ready=(lambda event: None) if streaming else None,
            )
            results.append(result["benchmark"])
        self._warmup_results = results
        return {
            "requests": count,
            "batch_size": batch_size,
            "streaming": streaming,
            "seed": self.transport.seed,
            "seed_scope": "explicit_fixed_seed_per_job; synthetic_images_use_local_rng",
            "timings": results,
        }

    # 把客户端请求转换为后端需要的输入格式
    def _prepare(self, req: dict[str, Any]) -> tuple[np.ndarray, str, list[int], np.ndarray]:
        if not isinstance(req, dict):
            raise ValueError("Each policy item must be a JSON object")
        if req.get("domain_name") != "libero":
            raise ValueError("The remote profile requires domain_name=libero")
        prompt = req.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A non-empty policy prompt is required")
        image_size = req.get("image_size")
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size != 256:
            raise ValueError("The remote profile requires image_size=256")
        observation = decode_policy_image(req.get("image"))
        pixels, caption, frame_size = prepare_libero_policy_inputs(observation, prompt, image_size)
        return pixels, caption, frame_size, observation

    # 推理入口
    def predict_distributed(
        self, req: dict[str, Any], *, is_batch: bool, on_actions_ready: Any = None
    ) -> dict[str, Any]:
        """Measure the complete policy call, as the PI LIBERO benchmark does.

        This clock starts after the HTTP handler receives the request. Backend
        job time stays separate from preprocessing, lock waits and postprocessing.
        """
        request_start = time.perf_counter()
        result = self._predict_distributed(
            req, is_batch=is_batch, on_actions_ready=on_actions_ready, request_start=request_start
        )
        policy_wall_ms = (time.perf_counter() - request_start) * 1000
        result["benchmark"] = {"timing_schema": "pi_libero_v1", "policy_wall_ms": policy_wall_ms}
        result["policy_timing"] = {
            "infer_ms": policy_wall_ms,
            "backend_ms": result["timing_ms"]["backend_roundtrip"],
        }
        result["server_timing"] = {"infer_ms": policy_wall_ms}
        result["timing_ms"]["full_chunk"] = policy_wall_ms
        if on_actions_ready is None:  # 流式和非流式
            result["timing_ms"]["ttfa"] = policy_wall_ms
            for timing in result["performance"]["action_timings"]:
                timing["ready_ms"] = policy_wall_ms
        timeline = result["performance"]["timeline"]
        timeline["events"].append(
            {
                "stage": "server.policy",
                "start_s": request_start,
                "end_s": time.perf_counter(),
            }
        )
        if destination := os.environ.get("COSMOS3_SERVER_TIMELINE"):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(timeline, separators=(",", ":")) + "\n")
        return result

    # 推理核心逻辑
    def _predict_distributed(
        self,
        req: dict[str, Any],
        *,
        is_batch: bool,
        on_actions_ready: Any,
        request_start: float,
    ) -> dict[str, Any]:
        events = []
        if on_actions_ready is not None and not self.supports_streaming:
            raise ValueError("The remote backend does not support action streaming")
        items = req.get("items") if is_batch else [req]
        if not isinstance(items, list) or not items:
            raise ValueError("'items' must be a non-empty list")
        prepared = [self._prepare(item) for item in items]
        t0 = time.perf_counter()
        preprocess_ms = (t0 - request_start) * 1000
        first_ready_ms = None
        partial_events = 0
        normalized = []
        worker_metrics = []
        action_timings = []
        with self._lock:  # 推理锁
            backend_start = time.perf_counter()
            queue_ms = (backend_start - t0) * 1000
            for sample_index, (pixels, caption, frame_size, _) in enumerate(prepared):  # 逐个推理请求
                channel = None
                token = None
                action_stream = None
                if on_actions_ready is not None:
                    if not self.cfg.callback_base_url:
                        raise ValueError("Streaming requires a callback URL reachable from the vLLM worker")

                    # 处理提前就绪动作
                    def publish(event, index=sample_index):
                        nonlocal first_ready_ms, partial_events
                        partial_events += 1
                        item = {
                            "action_indices": event["action_indices"],
                            "actions": self.normalizer.denormalize(event["actions"]),
                        }
                        ready_ms = (time.perf_counter() - request_start) * 1000
                        callback_start = time.perf_counter()
                        if first_ready_ms is None:
                            first_ready_ms = ready_ms
                        action_timings.extend(
                            {
                                "sample_index": index,
                                "action_index": position,
                                "ready_ms": ready_ms,
                                "step": event["step"],
                            }
                            for position in event["action_indices"]
                        )
                        if is_batch:
                            payload = {"items": [{"sample_index": index, **item}]}
                        else:
                            payload = item
                        on_actions_ready(
                            {
                                "type": "partial",
                                "step": event["step"],
                                "total_steps": 10,
                                "timing_ms": {"action_ready": ready_ms},
                                **payload,
                            }
                        )
                        events.append(
                            {
                                "stage": "server.action_ready",
                                "start_s": callback_start,
                                "end_s": time.perf_counter(),
                                "sample_index": index,
                                "action_indices": event["action_indices"],
                                "denoise_step": event["step"],
                            }
                        )

                    # 流式
                    channel = ActionStreamChannel(publish, self.transport.early_stop_actions or 16)
                    token = self.streams.add(channel)  # 生成
                    action_stream = {
                        "version": 1,
                        "callback_url": f"{self.cfg.callback_base_url.rstrip('/')}/internal/action_events/{token}",
                    }
                try:
                    arguments = {"action_stream": action_stream} if action_stream is not None else {}
                    sample_metrics = {"sample_index": sample_index}
                    result = self.transport.predict_prepared(
                        pixels, prompt=caption, frame_size=frame_size, metrics=sample_metrics, **arguments
                    )  # 推理
                    worker_metrics.append(sample_metrics)
                    if channel is not None:  # 校验
                        channel.verify_final(result)
                    normalized.append(result)
                finally:
                    if token is not None:
                        self.streams.remove(token)
            self._profile_confirmed = True
            backend_ms = (time.perf_counter() - backend_start) * 1000
        denorm_start = time.perf_counter()
        actions = [self.normalizer.denormalize(action) for action in normalized]
        completed = time.perf_counter()
        elapsed_ms = (completed - t0) * 1000
        full_ms = (completed - request_start) * 1000
        if on_actions_ready is None:
            # All positions become observable together in a nonstream response.
            first_ready_ms = full_ms
            action_timings = [
                {
                    "sample_index": sample,
                    "action_index": index,
                    "ready_ms": full_ms,
                    "step": worker_metrics[sample].get("denoise_steps") or 10,
                }
                for sample, chunk in enumerate(actions)
                for index in range(len(chunk))
            ]
        result = {"actions": actions} if is_batch else {"action": actions[0], "video": []}
        result["timing_ms"] = {
            "remote_roundtrip": elapsed_ms,
            "ttfa": first_ready_ms,
            "full_chunk": full_ms,
            "preprocess": preprocess_ms,
            "queue_wait": queue_ms,
            "backend_roundtrip": backend_ms,
            "denormalization": (completed - denorm_start) * 1000,
            "partial_events": partial_events,
        }
        result["performance"] = {
            "timeline": {
                "schema": "cosmos3_overlap_v1",
                "trace_id": req.get("trace_id"),
                "request_id": req.get("request_id"),
                "host": socket.gethostname(),
                "clock_domain": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "clock": "CLOCK_MONOTONIC perf_counter",
                "pid": os.getpid(),
                "events": [
                    {"stage": "server.preprocess", "start_s": request_start, "end_s": t0},
                    {"stage": "server.queue", "start_s": t0, "end_s": backend_start},
                    {"stage": "server.backend_roundtrip", "start_s": backend_start, "end_s": denorm_start},
                    {"stage": "server.denormalize", "start_s": denorm_start, "end_s": completed},
                    *events,
                ],
            },
            "request_id": req.get("request_id"),
            "policy_profile": self.transport.policy_profile,
            "action_time_schedule": self.transport.action_time_schedule,
            "early_stop_actions": self.transport.early_stop_actions,
            "streaming": on_actions_ready is not None,
            "batch_size": len(items),
            "generated_chunks": len(actions),
            "generated_actions": sum(len(chunk) for chunk in actions),
            "worker_metrics": worker_metrics,
            "action_timings": action_timings,
        }
        log.info("policy request=%s batch=%s remote_roundtrip_ms=%.1f", req.get("request_id"), len(items), elapsed_ms)
        request_id = req.get("request_id", 0)
        if self.cfg.dump_dir is not None and (request_id == 1 or request_id % self.cfg.dump_every == 0):
            try:
                root = self.cfg.dump_dir / f"request_{request_id:06d}_{time.time_ns()}"
                root.mkdir(parents=True)
                for index, ((pixels, caption, frame_size, observation), raw, action) in enumerate(
                    zip(prepared, normalized, actions, strict=True)
                ):
                    Image.fromarray(observation).save(root / f"observation_{index:03d}.png")
                    Image.fromarray(pixels).save(root / f"prepared_{index:03d}.png")
                    (root / f"action_{index:03d}.json").write_text(
                        json.dumps(
                            {"prompt": caption, "frame_size": frame_size, "normalized": raw.tolist(), "action": action}
                        ),
                        encoding="utf-8",
                    )
            except Exception:
                log.exception("Could not save policy request %s", request_id)
        return result

    def close(self) -> None:
        self.transport.close()
