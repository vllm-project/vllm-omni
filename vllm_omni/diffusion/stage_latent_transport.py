# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU shared-memory handoff for the opt-in H3 modality stages.

Control messages stay on the stage ZMQ channel. Each modality gets its own
single-consumer SHM key; request teardown unlinks unconsumed segments.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

_REFERENCE_KEY = "_minimax_h3_stage_latent_shm"
_VERSION = 1
_MODEL_MODALITIES = {
    "MiniMaxH3VideoDecoderPipeline": "video",
    "MiniMaxH3AudioDecoderPipeline": "audio",
}


class StageLatentTransport:
    def __init__(self, stage_id: int, modality: str):
        if modality not in {"video", "audio"}:
            raise ValueError(f"Invalid H3 latent modality: {modality}")
        self.stage_id = stage_id
        self.modality = modality
        self.connector = SharedMemoryConnector({"stage_id": stage_id})
        self.pending: dict[str, str] = {}
        self.metrics: dict[str, dict[str, float]] = {}

    def pack(self, request_id: str, prompt: Any) -> Any:
        if request_id in self.pending:
            raise ValueError(f"Duplicate H3 latent submission: {request_id}")
        if not isinstance(prompt, dict):
            raise ValueError("H3 modality stage requires a dictionary prompt")
        options = prompt.get("additional_information", {}).get("minimax_h3_decode", {})
        latents = options.get(f"{self.modality}_latents")
        if not isinstance(latents, (list, tuple)) or not latents:
            raise ValueError(f"Missing H3 {self.modality} latent list")
        envelope = {
            "request_id": request_id,
            "payload_version": _VERSION,
            "parent_stage_id": 1,
            "stage_id": self.stage_id,
            "modality": self.modality,
            "outputs": [{"output_index": i, "latent": latent} for i, latent in enumerate(latents)],
        }
        key = f"h3_latent_{self.stage_id}_{uuid.uuid4().hex}"
        started = time.perf_counter()
        try:
            success, size, _ = self.connector.put("1", str(self.stage_id), key, envelope)
            if not success:
                raise RuntimeError(f"Failed to write {self.modality} latents to shared memory")
        except BaseException:
            self.connector.cleanup(key)
            raise
        self.pending[request_id] = key
        self.metrics[request_id] = {
            "latent_shm_put_ms": (time.perf_counter() - started) * 1000,
            "latent_shm_bytes": size,
        }
        options = {k: v for k, v in options.items() if k not in {"video_latents", "audio_latents"}}
        additional = {**prompt.get("additional_information", {}), "minimax_h3_decode": options}
        return {
            **prompt,
            "additional_information": additional,
            _REFERENCE_KEY: {
                k: envelope[k] for k in ("request_id", "payload_version", "parent_stage_id", "stage_id", "modality")
            }
            | {"key": key, "output_count": len(latents)},
        }

    def release(self, request_id: str) -> dict[str, float]:
        key = self.pending.pop(request_id, None)
        if key is not None:
            self.connector.cleanup(key)
        return self.metrics.pop(request_id, {})

    def close(self) -> None:
        self.connector.close()
        self.pending.clear()
        self.metrics.clear()


def unpack_stage_latents(prompt: Any, request_id: str, stage_id: int, model_class_name: str) -> Any:
    if not isinstance(prompt, dict) or _REFERENCE_KEY not in prompt:
        return prompt
    reference = prompt[_REFERENCE_KEY]
    expected = {
        "request_id": request_id,
        "payload_version": _VERSION,
        "parent_stage_id": 1,
        "stage_id": stage_id,
        "modality": _MODEL_MODALITIES.get(model_class_name),
    }
    if expected["modality"] is None or not isinstance(reference, dict):
        raise ValueError("Invalid H3 latent stage reference")
    if any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError("H3 latent reference request/stage/modality/version mismatch")
    key = reference.get("key")
    if not isinstance(key, str) or not key.startswith(f"h3_latent_{stage_id}_"):
        raise ValueError("Invalid H3 latent shared-memory key")
    started = time.perf_counter()
    connector = SharedMemoryConnector({"stage_id": stage_id})
    try:
        received = connector.get("1", str(stage_id), key)
        if received is None:
            raise ValueError("H3 latent shared-memory payload is missing or already consumed")
        envelope, size = received
        if not isinstance(envelope, dict) or any(envelope.get(k) != v for k, v in expected.items()):
            raise ValueError("H3 latent payload request/stage/modality/version mismatch")
        outputs = envelope.get("outputs")
        if not isinstance(outputs, list) or not outputs or len(outputs) != reference.get("output_count"):
            raise ValueError("H3 latent output count mismatch")
        if any(not isinstance(item, dict) or item.get("output_index") != i for i, item in enumerate(outputs)):
            raise ValueError("H3 latent output order mismatch")
        options = dict(prompt["additional_information"]["minimax_h3_decode"])
        options[f"{expected['modality']}_latents"] = [item["latent"] for item in outputs]
        options["stage_latent_transfer"] = {
            "payload_version": _VERSION,
            "request_id": request_id,
            "output_indices": list(range(len(outputs))),
            "modality": expected["modality"],
            "parent_stage_id": 1,
            "latent_shm_get_ms": (time.perf_counter() - started) * 1000,
            "latent_shm_bytes": size,
        }
        return {k: v for k, v in prompt.items() if k != _REFERENCE_KEY} | {
            "additional_information": {**prompt["additional_information"], "minimax_h3_decode": options}
        }
    finally:
        connector.close()
