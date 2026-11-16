# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test-only observations at the real Mooncake restore and Talker boundaries.

The custom Thinker worker installs these observations inside its own process.
They deliberately serialize CPU tensors to disk and are not performance probes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from safetensors.torch import save_file

from vllm_omni.worker.gpu_ar_worker import GPUARWorker

AUDIT_ENV = "VLLM_OMNI_MOONCAKE_TEST_AUDIT_DIR"
# While this file exists in the audit directory, every HS read fails as a
# missing object would, after the native KV load has been submitted.
FAIL_HS_READ = "fail-hs-read"


def _record(directory: Path, kind: str, metadata: dict, tensors: dict | None = None) -> None:
    case = (directory / "case.txt").read_text().strip()
    name = f"{case}-{kind}-{uuid4().hex}"
    metadata = {"case": case, "kind": kind, **metadata}
    if tensors:
        tensor_file = directory / f"{name}.safetensors"
        save_file({key: tensor.detach().cpu().contiguous() for key, tensor in tensors.items()}, tensor_file)
        metadata["tensor_file"] = tensor_file.name
    (directory / f"{name}.json").write_text(json.dumps(metadata) + "\n")


class AuditedGPUARWorker(GPUARWorker):
    """Observe actual restoration and actual Qwen handoff without replacing either."""

    def init_device(self):
        import vllm_omni.model_executor.stage_input_processors.qwen3_omni as qwen_handoff
        from vllm_omni.distributed.kv_transfer.mooncake.connector import OmniMooncakeStoreConnector
        from vllm_omni.worker_v2.prefix_cache import MRv2PrefixCacheMixin

        directory = Path(os.environ[AUDIT_ENV])
        capture_step = OmniMooncakeStoreConnector.capture_step
        read_blocks = OmniMooncakeStoreConnector._read_hs_blocks
        merge_rows = qwen_handoff._merge_talker_prefill_rows
        save_local = MRv2PrefixCacheMixin._prefix_cache_save_mrv2

        def audited_read(connector, attempt, tokens):
            if (directory / FAIL_HS_READ).exists():
                _record(
                    directory,
                    "injected_failure",
                    {"request_id": attempt.req_id, "tokens": tokens, "local_tokens": attempt.local_tokens},
                )
                raise ValueError("injected HS read failure")
            return read_blocks(connector, attempt, tokens)

        def audited_capture(connector, req_id, attempt_id, start_token, prompt_len, tensors):
            merged = capture_step(connector, req_id, attempt_id, start_token, prompt_len, tensors)
            if merged is not None:
                # A confirmed-hit test needs writes published before the next
                # request. Drain the real save tasks, outside their owner lock;
                # ordinary serving remains asynchronous and best effort.
                with connector._lock:
                    saves = tuple(connector._saves)
                for future in saves:
                    future.result(timeout=connector.settings.load_timeout_seconds + 5)
                inserted = merged["embedding"].shape[0] - tensors["embedding"].shape[0]
                _record(
                    directory,
                    "capture",
                    {
                        "request_id": req_id,
                        "attempt_id": attempt_id,
                        "start_token": start_token,
                        "prompt_len": prompt_len,
                        "fresh_rows": tensors["embedding"].shape[0],
                        "inserted_rows": inserted,
                        **connector.request_cache_stats(req_id),
                    },
                )
            return merged

        def audited_local(runner, *args, **kwargs):
            rewrite = save_local(runner, *args, **kwargs)
            if rewrite is not None:
                _record(directory, "local_rewrite", {"requests": [req_id for _, req_id in rewrite.rows]})
            return rewrite

        def audited_merge(embed, hidden, prompt_token_ids, hf_config):
            packed = merge_rows(embed, hidden, prompt_token_ids, hf_config)
            prompt_len = len(prompt_token_ids)
            # Check both inputs before selecting one tensor per token. A correct
            # text-only transcript would not detect a misplaced hidden-state row.
            assert embed.shape[0] in (prompt_len, prompt_len + 1)
            assert hidden.shape[0] >= prompt_len
            assert packed.shape[0] == embed.shape[0]
            _record(
                directory,
                "handoff",
                {"prompt_token_ids": prompt_token_ids, "prompt_len": prompt_len},
                {
                    "embedding": embed[:prompt_len],
                    "hidden": hidden[:prompt_len],
                    "talker": packed[:prompt_len],
                },
            )
            return packed

        OmniMooncakeStoreConnector.capture_step = audited_capture
        OmniMooncakeStoreConnector._read_hs_blocks = audited_read
        setattr(MRv2PrefixCacheMixin, "_prefix_cache_save_mrv2", audited_local)
        qwen_handoff._merge_talker_prefill_rows = audited_merge
        super().init_device()
