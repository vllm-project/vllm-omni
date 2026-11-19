# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Temporary, recognition-fixture-only EOS diagnostics; no sampling changes."""

import hashlib
import json
import math
from functools import wraps
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.overrides import TorchFunctionMode

ROOT = Path("tests/e2e/online_serving/results/minicpmo_duplex_recognition/eos_trace")
EOS = 151717
SAVED = set()


def _hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _write(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / name).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def _scores(row):
    values, ids = torch.topk(row, 8)
    eos = row[EOS].float().item()
    top = [
        [int(token), float(value) if math.isfinite(value) else None]
        for token, value in zip(ids.tolist(), values.tolist())
    ]
    return {"top": top, "eos": eos if math.isfinite(eos) else None, "eos_rank": int((row > row[EOS]).sum().item()) + 1}


class _CaptureGreedy(TorchFunctionMode):
    def __init__(self, model, rows):
        super().__init__()
        self.model = model
        self.rows = rows

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func is torch.argmax and args[0].ndim == 2:
            assert len(args[0]) == len(self.rows)
            for index, row in enumerate(self.rows):
                record = getattr(self.model, "_eos_trace_rows", {}).get(row)
                if record is not None:
                    record["effective"] = _scores(args[0][index])
        return func(*args, **(kwargs or {}))


class MiniCPMDuplexEOSTrace:
    def init_worker_extension(self):
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
            MiniCPMO45OmniForConditionalGeneration as Model,
        )

        original_prepare = Model.prepare_duplex_sampling
        original_draw = Model._minicpmo45_duplex_stage2_draws
        original_sample = Model._sample_minicpmo45_native_duplex_stage0
        original_prefill = MiniCPMO45Stage0DuplexRuntime._stage_prefill_steps

        @wraps(original_prepare)
        def prepare(model, logits, sampling_metadata, rows):
            model._commit_minicpmo45_duplex_pending_samples()
            model._eos_trace_rows = {}
            if model.model_stage == "llm" and logits.shape[-1] > EOS:
                helper = getattr(model, "_minicpmo45_duplex_data_plane_helper", None)
                sessions = getattr(helper, "sessions", {})
                history = getattr(sampling_metadata, "output_token_ids", None) or []
                for row in rows:
                    state = sessions.get(row.session_id)
                    if state is None:
                        continue
                    model._eos_trace_rows[row.row_idx] = {
                        "session": row.session_id,
                        "request": row.request_id,
                        "unit": state.audio_chunk_idx,
                        "seq": row.seq,
                        "history": list(history[row.row_idx]) if row.row_idx < len(history) else [],
                        "generated": list(state.generated_tokens),
                        "ended": state.current_turn_ended,
                        "payload": {
                            key: (row.payload or {}).get(key)
                            for key in ("is_speech", "force_listen", "open_assistant_turn")
                        },
                        "temperature": row.temperature,
                        "raw": _scores(logits[row.row_idx]),
                    }
            return original_prepare(model, logits, sampling_metadata, rows)

        @wraps(original_draw)
        def draw(model, logits, rows, recent, token_ids, row_params, sampling_metadata, rewinds=None):
            greedy = [row for row in rows if getattr(sampling_metadata, "all_greedy", False) or row_params[row][0] <= 0]
            with _CaptureGreedy(model, greedy):
                return original_draw(model, logits, rows, recent, token_ids, row_params, sampling_metadata, rewinds)

        @wraps(original_sample)
        def sample(model, logits, sampling_metadata, **kwargs):
            result = original_sample(model, logits, sampling_metadata, **kwargs)
            if result is None:
                return result
            final = result.sampled_token_ids[:, 0].tolist()
            pending = getattr(model, "_minicpmo45_duplex_pending_samples", None)
            packed = {}
            if pending is not None:
                if pending.event is not None:
                    pending.event.synchronize()
                packed = dict(zip(pending.row_idxs, pending.host.tolist()))
            for row, record in getattr(model, "_eos_trace_rows", {}).items():
                record["final"] = final[row]
                record["deferred"] = packed.get(row)
                _write("steps.jsonl", record)
            model._eos_trace_rows = {}
            return result

        @wraps(original_prefill)
        def prefill(helper, state, waveform, **kwargs):
            result = yield from original_prefill(helper, state, waveform, **kwargs)
            key = (state.session_id, kwargs.get("epoch"), kwargs.get("seq"))
            if result and "inputs_embeds" in result and key not in SAVED:
                SAVED.add(key)
                folder = ROOT / state.session_id
                folder.mkdir(parents=True, exist_ok=True)
                tensor = result["inputs_embeds"].detach().cpu().contiguous()
                filename = folder / f"unit-{state.audio_chunk_idx:04d}-seq-{kwargs.get('seq')}.safetensors"
                save_file({"inputs_embeds": tensor}, str(filename))
                _write(
                    "prefills.jsonl",
                    {
                        "session": state.session_id,
                        "unit": state.audio_chunk_idx,
                        "seq": kwargs.get("seq"),
                        "token_ids": result["input_token_ids"],
                        "embedding_hash": _hash(tensor),
                        "wave_hash": hashlib.sha256(waveform.tobytes()).hexdigest() if waveform is not None else None,
                        "frames": [[_hash(t) for t in frame] for frame in (kwargs.get("encoded_frames") or [])],
                        "tensor": str(filename.relative_to(ROOT)),
                    },
                )
            return result

        Model.prepare_duplex_sampling = prepare
        Model._minicpmo45_duplex_stage2_draws = draw
        Model._sample_minicpmo45_native_duplex_stage0 = sample
        MiniCPMO45Stage0DuplexRuntime._stage_prefill_steps = prefill
