# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep model output PCM and text locally for speech-alignment diagnosis."""

import base64
import json
import time
import uuid
import wave
from pathlib import Path


class OutputAudit:
    def __init__(self):
        self.path = Path(__file__).parent / "recordings" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True)
        self.log = (self.path / "events.jsonl").open("w")
        self.waves = {}

    def write(self, event):
        saved = dict(event)
        if event.get("type") == "response.audio.delta":
            rid = event["response_id"]
            pcm = base64.b64decode(event["delta"])
            if rid not in self.waves:
                # Server-provided identifiers are metadata, not file names.
                filename = f"response-{len(self.waves)}.wav"
                w = wave.open(str(self.path / filename), "wb")
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(event.get("sample_rate_hz", 24000))
                self.waves[rid] = (w, filename)
            w, filename = self.waves[rid]
            w.writeframes(pcm)
            saved.pop("delta", None)
            saved.update(pcm_file=filename, pcm_bytes=len(pcm))
        # response.done can contain a duplicate PCM payload in metadata.
        if isinstance(saved.get("response"), dict):
            saved["response"] = dict(saved["response"])
            meta = saved["response"].get("metadata")
            if isinstance(meta, dict):
                saved["response"]["metadata"] = {k: v for k, v in meta.items() if k != "audio"}
        self.log.write(json.dumps(saved, ensure_ascii=False) + "\n")
        self.log.flush()

    def close(self):
        for w, _ in self.waves.values():
            w.close()
        self.log.close()
