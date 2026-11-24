"""Validate real-checkpoint baseline and HAS streaming, without Framework imports.

This checks the inference/protocol path on deterministic pixels. It does not
measure LIBERO task success or establish comparative performance.
"""

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from vllm_omni.entrypoints.cosmos3.action_policy_common import LiberoActionNormalizer
from vllm_omni.entrypoints.cosmos3.libero_policy_inputs import prepare_libero_policy_inputs
from vllm_omni.entrypoints.cosmos3.vllm_omni_action import VllmOmniActionTransport


def check_legacy(session, worker_url, pixels, caption, timeout):
    """Keep the pre-existing no-profile scheduler and video/action API usable."""
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    height, width = pixels.shape[:2]
    response = session.post(
        worker_url + "/v1/videos",
        data={
            "prompt": caption,
            "size": f"{width}x{height}",
            "num_frames": "17",
            "fps": "20",
            "num_inference_steps": "10",
            "guidance_scale": "1",
            "seed": "0",
            "extra_params": json.dumps(
                {
                    "action_mode": "policy",
                    "domain_name": "libero",
                    "raw_action_dim": 10,
                    "action_chunk_size": 16,
                    "image_size": 256,
                }
            ),
        },
        files={"input_reference": ("libero.png", buffer.getvalue(), "image/png")},
        timeout=60,
    )
    response.raise_for_status()
    job_url = worker_url + "/v1/videos/" + response.json()["id"]
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            response = session.get(job_url, timeout=30)
            response.raise_for_status()
            job = response.json()
            if job["status"] == "completed":
                action = VllmOmniActionTransport._decode_action(job["action"])
                assert job["action"].get("policy_profile") is None
                content = session.get(job_url + "/content", timeout=30)
                content.raise_for_status()
                assert len(content.content) > 0, "Legacy video output missing"
                return {
                    "shape": list(action.shape),
                    "finite": bool(np.isfinite(action).all()),
                    "default_scheduler": True,
                    "video_bytes": len(content.content),
                }
            if job["status"] in {"failed", "cancelled"}:
                raise RuntimeError(job.get("error"))
            time.sleep(0.05)
        raise TimeoutError("Legacy API smoke timed out")
    finally:
        session.delete(job_url, timeout=30).raise_for_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-url", default="http://127.0.0.1:8100")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8101")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/cosmos3_sync/smoke.json"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    image = (
        np.asarray(Image.open(args.image).convert("RGB")) if args.image else np.full((256, 512, 3), 30, dtype=np.uint8)
    )
    prompt = "pick up the object"
    pixels, caption, size = prepare_libero_policy_inputs(image, prompt, 256)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    payload = {
        "image": base64.b64encode(buffer.getvalue()).decode(),
        "prompt": prompt,
        "domain_name": "libero",
        "image_size": 256,
    }
    session = requests.Session()
    session.trust_env = False
    summary = {
        "seed": 0,
        "input": str(args.image) if args.image else "deterministic test pixels",
        "worker_url": args.worker_url,
        "policy_url": args.policy_url,
    }
    if args.legacy_only:
        summary["legacy_no_profile"] = check_legacy(session, args.worker_url, pixels, caption, args.timeout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        session.close()
        print(json.dumps(summary, indent=2))
        return
    with VllmTransportContext(args.worker_url, args.timeout, "constant_euler") as transport:
        started = time.perf_counter()
        baseline = transport.predict_prepared(pixels, prompt=caption, frame_size=size)
        summary["constant_euler"] = {
            "client_total_ms": (time.perf_counter() - started) * 1000,
            "shape": list(baseline.shape),
            "finite": bool(np.isfinite(baseline).all()),
        }
    response = session.post(args.policy_url + "/predict", json=payload, timeout=args.timeout)
    response.raise_for_status()
    expected = np.asarray(response.json()["action"], dtype=np.float32)
    events = []
    first_ms = None
    started = time.perf_counter()
    with session.post(args.policy_url + "/predict_stream", json=payload, stream=True, timeout=args.timeout) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "error":
                raise RuntimeError(event["error"])
            if event["type"] == "partial" and first_ms is None:
                first_ms = (time.perf_counter() - started) * 1000
            events.append(event)
    assert events and events[-1]["type"] == "final", "Missing final event"
    assert events[0]["type"] == "partial" and events[0]["step"] == 1
    assert events[0]["action_indices"] == [0]
    recovered = np.zeros((16, 10), dtype=np.float32)
    seen = set()
    for event in events[:-1]:
        assert event["type"] == "partial"
        indices = event["action_indices"]
        assert not seen.intersection(indices), "Duplicate action positions"
        seen.update(indices)
        recovered[indices] = event["actions"]
    assert seen == set(range(16)), "Missing action positions"
    final = np.asarray(events[-1]["action"], dtype=np.float32)
    assert final.shape == (16, 10) and np.isfinite(final).all()
    np.testing.assert_allclose(recovered, final, rtol=0, atol=1e-6)
    np.testing.assert_allclose(expected, final, rtol=0, atol=1e-6)
    summary["has_euler_stream"] = {
        "client_ttfa_ms": first_ms,
        "client_total_ms": (time.perf_counter() - started) * 1000,
        "first_step": events[0]["step"],
        "partial_events": len(events) - 1,
        "all_actions_received": True,
        "matches_nonstream": True,
        "timing_ms": events[-1]["timing_ms"],
    }
    summary["policy_info"] = session.get(args.policy_url + "/info", timeout=10).json()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(args.output.with_name("baseline_actions.npy"), LiberoActionNormalizer().denormalize(baseline))
    np.save(args.output.with_name("has_actions.npy"), final)
    session.close()
    print(json.dumps(summary, indent=2))


class VllmTransportContext:
    def __init__(self, url, timeout, schedule):
        self.transport = VllmOmniActionTransport(
            url, timeout=timeout, poll_interval=0.05, action_time_schedule=schedule
        )

    def __enter__(self):
        return self.transport

    def __exit__(self, *args):
        self.transport.close()


if __name__ == "__main__":
    main()
