# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporary logits instrumentation for issue 10417."""

import hashlib
import json
import os
import platform
from pathlib import Path

import torch

ENABLED = bool(os.environ.get("MINICPMO_DUPLEX_TRACE_DIR"))


def trace_path(name):
    root = Path(os.environ["MINICPMO_DUPLEX_TRACE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    return root / name


K = 8
WATCHED = [151717, 151718, 151705, 151706, 151703]


def snapshot(logits):
    values, ids = torch.topk(logits, K)
    eos = logits[151717]
    rank = 1 + (logits > eos).sum()
    watched = torch.stack([logits[token] for token in WATCHED])
    return torch.cat((ids.float(), values.float(), watched.float(), rank.reshape(1).float()))


def render(scores):
    return {
        "top": [[int(token), value] for token, value in zip(scores[:K], scores[K : 2 * K])],
        "watched": dict(zip(WATCHED, scores[2 * K : -1])),
        "eos_rank": int(scores[-1]),
    }


def begin(model, logits, metadata, rows):
    if not ENABLED:
        return
    flush(model)
    model._i417_current = {}
    if not getattr(model, "_i417_profile_written", False):
        write_profile(model)
        model._i417_profile_written = True
    for row in rows:
        if not row.session_id:
            continue
        state = model._minicpmo45_duplex_state_for_row(row.row_idx)
        if state is None or state.audio_chunk_idx > 50 or logits.shape[-1] <= max(WATCHED):
            continue
        payload = row.payload or {}
        history = getattr(metadata, "output_token_ids", None) or []
        record = {
            "session": row.session_id,
            "seq": row.seq,
            "unit": state.audio_chunk_idx,
            "window": state.window_enabled,
            "append": state.prepared_append_identity,
            "ended": state.current_turn_ended,
            "last_stop": state.last_terminator_token,
            "pending_speech": state.pending_speech_context,
            "speech": payload.get("is_speech"),
            "forced_payload": payload.get("force_listen"),
            "forced_mask": False,
            "temperature": row.temperature,
            "all_greedy": bool(getattr(metadata, "all_greedy", False)),
            "open_assistant_turn": payload.get("open_assistant_turn"),
            "history": list(history[row.row_idx]) if row.row_idx < len(history) else [],
            "generated": list(state.generated_tokens),
        }
        model._i417_current[row.row_idx] = (record, [snapshot(logits[row.row_idx])])


def force(model, row_idx):
    if not ENABLED:
        return
    row = getattr(model, "_i417_current", {}).get(row_idx)
    if row:
        row[0]["forced_mask"] = True


def adjusted(model, row_idx, effective, penalty_ids):
    row = getattr(model, "_i417_current", {}).get(row_idx)
    if row:
        row[0]["penalty_has_eos"] = 151717 in penalty_ids
        row[1].append(snapshot(effective))


def finish(model, sampled, pending):
    if not ENABLED:
        return
    current = getattr(model, "_i417_current", {})
    model._i417_current = {}
    if not current:
        return
    records = []
    for row_idx, (record, snapshots) in current.items():
        device_scores = torch.cat((*snapshots, sampled[row_idx].reshape(1).float()))
        host = torch.empty_like(device_scores, device="cpu", pin_memory=True)
        host.copy_(device_scores, non_blocking=True)
        records.append((row_idx, record, len(snapshots), host))
    event = torch.Event(device=sampled.device)
    event.record()
    model._i417_pending = (records, pending, event)


def flush(model):
    group = getattr(model, "_i417_pending", None)
    if group is None:
        return
    model._i417_pending = None
    records, pending, event = group
    event.synchronize()
    pending_rows = {} if pending is None else dict(zip(pending.row_idxs, pending.host.tolist()))
    size = 2 * K + len(WATCHED) + 1
    with open(trace_path("steps.log"), "a") as f:
        for row_idx, record, count, host in records:
            scores = host.tolist()
            record["model_logits"] = render(scores[:size])
            if count > 1:
                record["effective_logits"] = render(scores[size : 2 * size])
            record["sampled"] = int(scores[-1])
            if row_idx in pending_rows:
                final, raw, hit = pending_rows[row_idx]
                record.update(stage2_sample=raw, chunk_boundary=bool(hit), final_matches=(final == record["sampled"]))
            print(repr(record), file=f)


def write_profile(model):
    config = getattr(getattr(model, "vllm_config", None), "model_config", None)
    model_path = Path(str(getattr(config, "model", "")))
    files = {}
    for name in ("config.json", "preprocessor_config.json", "processing_minicpmo.py", "tokenizer.json"):
        path = model_path / name
        if path.is_file():
            files[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    weights = {path.name: str(path.resolve()) for path in model_path.glob("*.safetensors")}
    cpuinfo = Path("/proc/cpuinfo")
    profile = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cpu": platform.processor(),
        "cpuinfo": cpuinfo.read_text().split("\n\n", 1)[0] if cpuinfo.exists() else None,
        "torch_threads": torch.get_num_threads(),
        "model_path": str(model_path),
        "files": files,
        "weight_cache_paths": weights,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "NVIDIA_TF32_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
                "MKL_CBWR",
                "ATEN_CPU_CAPABILITY",
                "OMP_NUM_THREADS",
            )
        },
    }
    trace_path("runtime.json").write_text(json.dumps(profile, indent=2))
