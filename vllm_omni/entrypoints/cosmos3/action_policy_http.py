# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HTTP policy protocol shared by the native and remote inference services."""

import datetime
import json
import logging
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image

from vllm_omni.entrypoints.cosmos3.action_policy_common import decode_policy_image

log = logging.getLogger(__name__)


class ActionPolicyHTTPServer(ThreadingHTTPServer):
    """Initialize request IDs before the first connection, avoiding a thread race."""

    def __init__(self, address: tuple[str, int], service: Any) -> None:
        self.service = service
        self._req_id_lock = threading.Lock()
        self._req_id = 0
        super().__init__(address, ActionPolicyHandler)


def _save_failed_request_dump(
    *,
    dump_root: Path,
    request_id: int,
    request_json: dict[str, Any],
    error: str,
) -> None:
    """
    Dump request + error even if inference fails.
    Best-effort: will try to decode and save observation image if present.
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    out_dir = dump_root / f"{ts}_req{request_id:06d}_ERROR"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save request JSON (without base64 image)
    request_json_copy = {k: v for k, v in request_json.items() if k != "image"}
    if "image" in request_json:
        request_json_copy["image"] = "<attempted to save as observation.png>"
    (out_dir / "request.json").write_text(json.dumps(request_json_copy, indent=2), encoding="utf-8")
    (out_dir / "error.txt").write_text(error, encoding="utf-8")

    try:
        image_b64 = request_json.get("image")
        if isinstance(image_b64, str):
            obs_hwc = decode_policy_image(image_b64)
            Image.fromarray(obs_hwc).save(out_dir / "observation.png")
    except Exception:
        # Ignore any dump failures.
        return


class ActionPolicyHandler(BaseHTTPRequestHandler):
    """
    基于 Python 标准库 http.server 的 HTTP 动作推理服务处理器
    整体设计围绕 单条/批量 -- 普通/流式 四个维度
    路由表：
        GET  /                      健康检查
        GET  /info                  运行时信息（含 streaming 能力自描述）
        POST /predict               单条，JSON 响应
        POST /predict_batch         批量，JSON 响应
        POST /predict_stream        单条，NDJSON 流
        POST /predict_batch_stream  批量，NDJSON 流

    流式与普通模式调用的是**同一个 service 入口**，差别只在 handler 的写回方式：
    ``is_stream`` 为真时先发 ``200 + application/x-ndjson`` 响应头，之后每收到一条
    事件就写一行并 flush，直到 ``final`` 结束；否则用 ``_send_json`` 一次性返回。
    客户端中途断开只会停掉后续写事件，**不会**中断模型前向（collective 已启动必须跑完）。
    """

    server: ThreadingHTTPServer  # type: ignore[assignment]

    def _start_stream(self) -> None:
        """
        发送响应头，用于流式，普通响应是直接发结果
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _write_stream_event(self, payload: dict[str, Any]) -> bool:
        """写一条 NDJSON 事件。

        Returns:
            ``False`` 表示客户端已断开（BrokenPipe/ConnectionReset/OSError），
            调用方应停止后续写事件（但不中断模型前向）
        """
        try:
            self.wfile.write(
                json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
            )  # 把字典序列化成 JSON 字符串  -- utf-8
            self.wfile.flush()  # 强制把缓冲里的数据推出去
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False
        return True

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        """非流式路径：一次性写出完整 JSON 响应（带 Content-Length）。"""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        # Avoid caches/proxies returning stale results.
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client closed the connection (often due to request timeout).
            # Avoid noisy tracebacks; nothing to do server-side.
            return

    def do_GET(self) -> None:  # noqa: N802
        """/info 返回 service.get_info()；/ 作为健康检查（不触模型）"""
        if self.path == "/info":
            service = getattr(self.server, "service")  # type: ignore[attr-defined]
            try:
                self._send_json(200, service.get_info())
            except Exception as exc:
                log.exception("Could not retrieve policy backend information")
                self._send_json(503, {"error": str(exc)})
        elif self.path == "/":
            service = getattr(self.server, "service")
            health = getattr(service, "check_health", lambda: True)()
            self._send_json(200 if health else 503, {"status": "ok" if health else "unavailable"})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/", "/predict", "/predict_batch", "/predict_stream", "/predict_batch_stream", "/warmup"):
            self._send_json(404, {"error": "Not found"})
            return

        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(415, {"error": "Content-Type must be application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return

        if length <= 0:
            self._send_json(400, {"error": "Content-Length must be positive"})
            return
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        if not isinstance(req, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return

        service = getattr(self.server, "service")  # type: ignore[attr-defined]
        if self.path == "/warmup":
            warmup = getattr(service, "warmup", None)
            if warmup is None:
                self._send_json(404, {"error": "This service does not support synthetic warmup"})
                return
            try:
                self._send_json(200, warmup(req))
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})
            return

        # Generate a per-request id at the HTTP layer to correlate logs and dumps.
        # 在 HTTP 层自增 request_id（而不是在 service 里），这样即使请求在预处理阶段
        # 就失败也能拿到一个可追踪的编号。
        req_id_lock: threading.Lock | None = getattr(self.server, "_req_id_lock", None)  # type: ignore[attr-defined]
        if req_id_lock is None:
            req_id_lock = threading.Lock()
            setattr(self.server, "_req_id_lock", req_id_lock)  # type: ignore[attr-defined]
            setattr(self.server, "_req_id", 0)  # type: ignore[attr-defined]
        with req_id_lock:
            next_id = int(getattr(self.server, "_req_id")) + 1  # type: ignore[attr-defined]
            setattr(self.server, "_req_id", next_id)  # type: ignore[attr-defined]
        req["request_id"] = next_id

        log.info(
            f"[action-server] HTTP request_id={next_id} from={self.client_address[0]}:{self.client_address[1]} "
            f"path={self.path} bytes={length}"
        )

        # 从 URL 同时推出“是否批量”与“是否流式”（批量请求体是 {"items": [...]}）。
        is_batch = self.path in ("/predict_batch", "/predict_batch_stream")
        is_stream = self.path in ("/predict_stream", "/predict_batch_stream")
        if is_stream and not getattr(service, "supports_streaming", True):
            self._send_json(400, {"error": "The remote policy backend supports nonstream requests only."})
            return
        stream_open = True

        def publish(event: dict[str, Any]) -> None:
            """把一条事件写成 NDJSON 行；一旦断开就置位，后续事件直接丢弃。"""
            nonlocal stream_open
            if stream_open:
                event["request_id"] = next_id
                stream_open = self._write_stream_event(event)

        if is_stream:
            # 响应头必须先于任何推理/事件发出，否则客户端会阻塞在等待状态码上。
            self._start_stream()
        try:
            out = service.predict_distributed(req, is_batch=is_batch, on_actions_ready=publish if is_stream else None)
        except Exception as e:
            err = str(e)
            traceback.print_exc()

            payload = (
                {"actions": [], "error": err}
                if is_batch
                else {"action": [], "error": err, "request_id": req.get("request_id")}
            )
            log.error(f"[action-server] request_id={req.get('request_id')} ERROR: {err}")

            # Dump failed request for offline debugging if enabled.
            if service.cfg.dump_dir is not None:
                try:
                    dump_root = Path(service.cfg.dump_dir)
                    dump_root.mkdir(parents=True, exist_ok=True)
                    _save_failed_request_dump(
                        dump_root=dump_root,
                        request_id=int(req.get("request_id") or 0),
                        request_json=req,
                        error=err,
                    )
                except Exception:
                    pass

            if is_stream:
                publish({"type": "error", "error": err})
            else:
                status = 400 if service.cfg.http_400_on_error else 200
                self._send_json(status, payload)
            return

        if is_stream:
            # 最后一条：完整响应（与 /predict 同构）。
            publish({"type": "final", **out})
        else:
            self._send_json(200, out)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence default request logging (the simulator can be chatty).
        # 网络请求日志由 do_POST 单条 log.info 统一输出，这里不再重复打印。
        return
