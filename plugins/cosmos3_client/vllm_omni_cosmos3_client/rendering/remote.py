"""Draw Huawei-computed scene geometry on a remote EGL renderer.

Only camera drawing crosses the network. MuJoCo stepping, controllers, state,
observations and success checks remain in the client process.
"""

import functools
import gzip
import hashlib
import io
import json
import threading
from pathlib import Path

import mujoco
import numpy as np

from .protocol import DATA_FIELDS, MODEL_FIELDS, OPTION_FIELDS
from .transport import RenderSession


class RemoteContext:
    def __init__(self, sim, device_id=-1, max_width=640, max_height=480):
        self.sim = sim
        self.offscreen = True
        self.vopt = mujoco.MjvOption()
        self.model_id = None
        sim.add_render_context(self)


def install_renderer(url: str, output_dir: Path | None = None) -> None:
    """Select remote rendering explicitly, before creating any LIBERO envs."""
    import robosuite.environments.base as base
    import robosuite.utils.binding_utils as binding

    if getattr(binding.MjSim.render, "_cosmos3_remote_render", False):
        raise RuntimeError("Remote renderer is already installed in this process")
    session = RenderSession()
    url = url.rstrip("/")
    health = session.get(url + "/health")
    health.raise_for_status()
    info = health.json()
    if (
        info.get("service") != "render_only_v1"
        or info.get("physics_execution") is not False
        or info.get("mujoco_version") != mujoco.__version__
    ):
        raise ValueError(f"Incompatible render-only server: {info}")
    lock = threading.Lock()
    sequence = 0
    original = binding.MjSim.render

    @functools.wraps(original)
    def render(
        self,
        width=None,
        height=None,
        *,
        camera_name=None,
        depth=False,
        mode="offscreen",
        device_id=-1,
        segmentation=False,
    ):
        nonlocal sequence
        if mode != "offscreen" or depth or segmentation:
            raise ValueError("Remote rendering supports offscreen RGB cameras only")
        with lock:
            sequence += 1
            context = self._render_context_offscreen
            if not isinstance(context, RemoteContext):
                raise RuntimeError("Install the remote renderer before creating the environment")

            def register_model():
                binary = np.empty(mujoco.mj_sizeModel(self.model._model), dtype=np.uint8)
                mujoco.mj_saveModel(self.model._model, None, binary)
                response = session.post(
                    url + "/model",
                    data=gzip.compress(binary.tobytes(), compresslevel=1),
                    headers={"X-Mujoco-Version": mujoco.__version__, "Content-Encoding": "gzip"},
                )
                response.raise_for_status()
                context.model_id = response.json()["model_id"]
                if context.model_id != hashlib.sha256(binary.tobytes()).hexdigest():
                    raise ValueError("Remote model digest mismatch")

            if context.model_id is None:
                register_model()
            fields = {}
            for prefix, target, names in (
                ("d_", self.data._data, DATA_FIELDS),
                ("m_", self.model._model, MODEL_FIELDS),
                ("o_", context.vopt, OPTION_FIELDS),
            ):
                for name in names:
                    if hasattr(target, name):
                        fields[prefix + name] = np.asarray(getattr(target, name)).copy()
            state_hash = hashlib.sha256(
                fields["d_qpos"].tobytes() + fields["d_cam_xpos"].tobytes() + fields["d_geom_xpos"].tobytes()
            ).hexdigest()
            metadata = {
                "model_id": context.model_id,
                "camera_name": camera_name,
                "width": width,
                "height": height,
                "sequence": sequence,
            }
            fields["metadata"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
            payload = io.BytesIO()
            np.savez(payload, **fields)
            response = session.post(url + "/render", data=payload.getvalue())
            if response.status_code == 409 and response.json().get("error") == "unknown_model":
                # A renderer restart/eviction loses only its scene cache.
                # Register the same current model and repeat this camera draw.
                register_model()
                metadata["model_id"] = context.model_id
                fields["metadata"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
                payload = io.BytesIO()
                np.savez(payload, **fields)
                response = session.post(url + "/render", data=payload.getvalue())
            response.raise_for_status()
            result = np.load(io.BytesIO(response.content), allow_pickle=False)
            if (
                response.headers.get("X-State-SHA256") != state_hash
                or int(response.headers["X-Render-Sequence"]) != sequence
                or result.shape != (height, width, 3)
                or result.dtype != np.uint8
                or response.headers.get("X-Frame-SHA256") != hashlib.sha256(result.tobytes()).hexdigest()
            ):
                raise ValueError("Remote render state, sequence or image verification failed")
            return result

    render._cosmos3_remote_render = True
    binding.MjRenderContextOffscreen = RemoteContext
    base.MjRenderContextOffscreen = RemoteContext
    binding.MjSim.render = render
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "remote_render_backend.json").write_text(
            json.dumps(
                {
                    "backend": "remote EGL render-only",
                    "url": url,
                    "physics_host": "client",
                    "physics_execution_on_renderer": False,
                    "mujoco_version": mujoco.__version__,
                    "renderer_health": info,
                },
                indent=2,
            )
            + "\n"
        )
