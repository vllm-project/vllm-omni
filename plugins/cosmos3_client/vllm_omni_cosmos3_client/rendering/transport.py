"""Retry render-only RPCs without repeating any physics or policy actions."""

import json
import os
import sys
import time
from pathlib import Path

import requests


class RenderSession:
    def __init__(self, *, session_factory=requests.Session, clock=time.monotonic, sleep=time.sleep):
        self.factory = session_factory
        self.clock = clock
        self.sleep = sleep
        self.session = self._session()

    def _session(self):
        session = self.factory()
        session.trust_env = False
        return session

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", float(os.environ.get("COSMOS3_RENDER_REQUEST_TIMEOUT", "120")))
        timeout = float(os.environ.get("COSMOS3_RENDER_RECONNECT_TIMEOUT", "600"))
        interval = float(os.environ.get("COSMOS3_RENDER_RETRY_INTERVAL", "2"))
        start = self.clock()
        failures = 0
        while True:
            try:
                response = self.session.request(method, url, **kwargs)
                if response.status_code in {502, 503, 504}:
                    raise requests.ConnectionError(f"Render gateway unavailable: {response.status_code}")
                if failures:
                    row = {
                        "stage": "render.reconnect",
                        "start_s": start,
                        "end_s": self.clock(),
                        "attempts": failures + 1,
                        "method": method,
                        "url": url,
                    }
                    if output := os.environ.get("EVAL_OUT"):
                        path = Path(output) / "render_reconnect.jsonl"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                    print(f"Render connection recovered after {failures} retries", file=sys.stderr, flush=True)
                return response
            except (requests.ConnectionError, requests.Timeout) as exc:
                failures += 1
                if timeout > 0 and self.clock() - start >= timeout:
                    raise TimeoutError(f"Render service unavailable for {timeout}s: {url}") from exc
                if failures == 1:
                    print(f"Render connection interrupted; waiting to reconnect: {url}", file=sys.stderr, flush=True)
                self.session.close()
                self.session = self._session()
                self.sleep(interval)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)
