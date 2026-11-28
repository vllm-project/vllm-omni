"""168 EGL server: load self-contained model and draw Huawei-derived geometry."""

import argparse
import gzip
import hashlib
import io
import json
import socket
import time
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import mujoco
import numpy as np
from OpenGL import GL
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

from .protocol import DATA_FIELDS, MODEL_FIELDS, OPTION_FIELDS

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(exist_ok=True, parents=True)
models = OrderedDict()


def log_stages(row):
    with (args.output / "render_server_stages.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def reply(self, code, body, kind="application/json", extra=None):
        # Keep all EGL work on this server thread. Closing each HTTP connection
        # lets another vector-env client submit its next frame without waiting
        # for the first client's persistent connection to end.
        self.close_connection = True
        self.send_response(code)
        self.send_header("Connection", "close")
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/logs":
            path = args.output / "render_server_stages.jsonl"
            self.reply(200, path.read_bytes() if path.exists() else b"", "application/x-ndjson")
            return
        self.close_connection = True
        self.reply(
            200,
            json.dumps(
                {
                    "service": "render_only_v1",
                    "mujoco_version": mujoco.__version__,
                    "physical_gpu": args.gpu,
                    "physics_execution": False,
                    "stage_logs": True,
                    "gpu_proof": json.loads((args.output / "gpu_renderer_proof.json").read_text())
                    if (args.output / "gpu_renderer_proof.json").exists()
                    else None,
                }
            ).encode(),
        )

    def do_POST(self):
        request_start = time.perf_counter()
        previous = request_start
        stages = {}

        def mark(name):
            nonlocal previous
            current = time.perf_counter()
            stages[name] = (current - previous) * 1000
            previous = current

        try:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if len(body) != int(self.headers["Content-Length"]):
                self.reply(503, b'{"error":"incomplete_upload"}')
                return
            mark("receive_body_ms")
            if self.path == "/model":
                if self.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                digest = hashlib.sha256(body).hexdigest()
                received = self.headers.get("X-Mujoco-Version")
                assert received == mujoco.__version__, (received, mujoco.__version__)
                mark("model_digest_lookup_ms")
                cached = digest in models
                if not cached:
                    model = mujoco.MjModel.from_binary_path("scene.mjb", assets={"scene.mjb": body})
                    sim = MjSim(model)
                    mark("model_load_and_sim_init_ms")
                    context = MjRenderContextOffscreen(sim, device_id=args.gpu)
                    mark("gpu_context_init_ms")
                    renderer = GL.glGetString(GL.GL_RENDERER).decode()
                    assert "nvidia" in renderer.lower(), renderer
                    models[digest] = (sim, context)
                    while len(models) > 2:
                        _, (old_sim, old_context) = models.popitem(last=False)
                        old_context.gl_ctx.make_current()
                        old_sim._render_context_offscreen = None
                        del old_context, old_sim
                    context.gl_ctx.make_current()
                    (args.output / "gpu_renderer_proof.json").write_text(
                        json.dumps(
                            {
                                "renderer": renderer,
                                "vendor": GL.glGetString(GL.GL_VENDOR).decode(),
                                "mujoco_version": mujoco.__version__,
                                "physical_gpu": args.gpu,
                                "physics_execution": False,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                models.move_to_end(digest)
                mark("model_cache_and_proof_ms")
                self.reply(
                    200,
                    json.dumps(
                        {
                            "model_id": digest,
                            "mujoco_version": mujoco.__version__,
                            "cached": cached,
                            "registration_stages_ms": stages,
                        }
                    ).encode(),
                )
                mark("response_write_ms")
                log_stages(
                    {
                        "kind": "model_registration",
                        "model_id": digest,
                        "cached": cached,
                        "start_s": request_start,
                        "end_s": previous,
                        "stages_ms": stages,
                        "elapsed_ms": (previous - request_start) * 1000,
                        "clock": "168 process monotonic",
                    }
                )
                return
            assert self.path == "/render", self.path
            with np.load(io.BytesIO(body), allow_pickle=False) as payload:
                metadata = json.loads(payload["metadata"].tobytes())
                if metadata["model_id"] not in models:
                    self.reply(409, json.dumps({"error": "unknown_model"}).encode())
                    return
                sim, context = models[metadata["model_id"]]
                models.move_to_end(metadata["model_id"])
                # Copy the already-computed geometry; do not integrate or run
                # mj_forward on 168. All physical evolution remains on Huawei.
                for prefix, target, fields in [
                    ("d_", sim.data._data, DATA_FIELDS),
                    ("m_", sim.model._model, MODEL_FIELDS),
                    ("o_", context.vopt, OPTION_FIELDS),
                ]:
                    for field in fields:
                        key = prefix + field
                        if key in payload:
                            np.copyto(getattr(target, field), payload[key], casting="no")
                context.gl_ctx.make_current()
                camera = metadata["camera_name"]
                state_digest = hashlib.sha256(
                    payload["d_qpos"].tobytes() + payload["d_cam_xpos"].tobytes() + payload["d_geom_xpos"].tobytes()
                ).hexdigest()
                mark("decode_state_and_copy_ms")
                start = time.perf_counter()
                result = sim.render(
                    width=metadata["width"],
                    height=metadata["height"],
                    camera_name=camera,
                    depth=False,
                    segmentation=False,
                    device_id=args.gpu,
                )
                render_ms = (time.perf_counter() - start) * 1000
                mark("gpu_draw_readback_ms")
                buffer = io.BytesIO()
                np.save(buffer, result, allow_pickle=False)
                frame_hash = hashlib.sha256(result.tobytes()).hexdigest()
                mark("pack_pixels_and_hash_ms")
                self.reply(
                    200,
                    buffer.getvalue(),
                    "application/x-npy",
                    {
                        "X-Render-Ms": render_ms,
                        "X-State-SHA256": state_digest,
                        "X-Frame-SHA256": frame_hash,
                        "X-Render-Stages-Ms": json.dumps(stages),
                        "X-Render-Sequence": metadata["sequence"],
                    },
                )
                mark("response_write_ms")
                log_stages(
                    {
                        "kind": "frame",
                        "sequence": metadata["sequence"],
                        "model_id": metadata["model_id"],
                        "start_s": request_start,
                        "end_s": previous,
                        "stages_ms": stages,
                        "elapsed_ms": (previous - request_start) * 1000,
                        "clock": "168 process monotonic",
                    }
                )
        except Exception as exc:
            traceback.print_exc()
            self.reply(500, json.dumps({"error": repr(exc)}).encode())

    def log_message(self, fmt, *values):
        print(time.time(), fmt % values, flush=True)


HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
