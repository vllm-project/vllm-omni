# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit general decisions on the resident Thinker, without chat interception."""

import asyncio
import json
import time
import uuid
from typing import Literal

import anyio
import torch
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool

from vllm_omni.model_executor.models.drvexo_speech_8b.general_input import BranchTokenizer


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=8192)


class GeneralRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    general_decision_head_enabled: StrictBool = True
    task: Literal["choice", "noul", "score"]
    instruction: str = Field(min_length=1, max_length=65536)
    state: JsonValue
    system_prompt: str = Field(default="", max_length=4096)
    candidates: list[Candidate] = Field(min_length=2, max_length=151)


def input_row(body):
    ids = [c.id for c in body.candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")
    state = body.state if isinstance(body.state, str) else json.dumps(body.state, ensure_ascii=False, sort_keys=True)
    if len(state) > 131072:
        raise ValueError("State exceeds the character limit")
    return {
        "task": body.task,
        "instruction": body.instruction,
        "state_json": state,
        "system_prompt": body.system_prompt,
        "candidate_ids": ids,
        "candidates": [f"Candidate: {c.id}: {c.text}" for c in body.candidates],
    }


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in tensors(item)]
    return []


async def general_decision(body: GeneralRequest, raw_request: Request):
    if not body.general_decision_head_enabled:
        return {"enabled": False, "probabilities": None, "executable": False}
    try:
        row = input_row(body)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    serving = raw_request.app.state.openai_serving_chat
    tokenizer = serving.renderer.get_tokenizer()
    if tokenizer is None:
        tokenizer = await serving.engine_client.get_tokenizer()
    branch = BranchTokenizer(tokenizer, 1536, 384)
    try:
        prefix = branch.prefix(row)
        suffixes = [branch.candidate(c) for c in row["candidates"]]
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    from vllm.sampling_params import RequestOutputKind, SamplingParams

    engine = serving.engine_client
    started = time.perf_counter()
    scores = []
    prompt_tokens = 0
    for suffix in suffixes:
        if await raw_request.is_disconnected():
            raise asyncio.CancelledError()
        ids = prefix + suffix
        prompt_tokens += len(ids)
        identifier = "drvexo-general-" + uuid.uuid4().hex
        prompt = {
            "prompt_token_ids": ids,
            "modalities": ["text"],
            "cache_salt": identifier,
            "additional_information": {"drvexo_general_decision": {"prompt_length": len(ids), "task": body.task}},
        }
        params = [p.clone() for p in engine.default_sampling_params_list]
        params[0] = SamplingParams(temperature=0, max_tokens=1, seed=20261019, output_kind=RequestOutputKind.FINAL_ONLY)
        generator = engine.generate(
            prompt=prompt, request_id=identifier, sampling_params_list=params, output_modalities=["text"]
        )
        final = None
        complete = False
        try:
            async for output in generator:
                if await raw_request.is_disconnected():
                    raise asyncio.CancelledError()
                final = output
            if final is None or final.error:
                raise RuntimeError("General-decision backbone request failed")
            payload = final.multimodal_output or {}
            values = [t.reshape(-1) for t in tensors(payload.get("drvexo_general_scores")) if t.numel()]
            statuses = [int(v) for t in tensors(payload.get("drvexo_general_status")) for v in t.flatten()]
            if len(values) != 1 or values[0].numel() != 1 or 1 not in statuses or -1 in statuses:
                raise RuntimeError("General head missing or native prefill incomplete")
            value = float(values[0][0])
            if not torch.isfinite(torch.tensor(value)):
                raise RuntimeError("Nonfinite candidate score")
            scores.append(value)
            complete = True
        finally:
            # Closing/aborting is scoped to this candidate's unique request ID.
            with anyio.CancelScope(shield=True):
                try:
                    await generator.aclose()
                finally:
                    if not complete:
                        await engine.abort(identifier)
    probabilities = torch.tensor(scores, dtype=torch.float32).softmax(0).double().tolist()
    return {
        "enabled": True,
        "task": body.task,
        "probabilities": dict(zip(row["candidate_ids"], probabilities, strict=True)),
        "scores": dict(zip(row["candidate_ids"], scores, strict=True)),
        "selected_id": row["candidate_ids"][max(range(len(scores)), key=scores.__getitem__)],
        "latency_ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": prompt_tokens,
        "internal_completion_tokens": len(scores),
        "language_generation_used_for_scoring": False,
        "shared_thinker": True,
        "input_scope": "explicit instruction/state/candidates only",
        "executable": False,
        "token_limits": {"prefix": 1536, "candidate": 384},
        "truncation_policy": (
            "fixed training protocol; instruction/state token budgets are shared; candidate is head-truncated"
        ),
    }


def validate_chat_request(request) -> None:
    """Keep internal stage metadata out of the public chat request."""
    if getattr(request, "entry_stage_id", None) is not None or getattr(request, "direct_stage_input", None) is not None:
        raise ValueError("Pipeline entry is selected internally")
    if getattr(request, "decision_head_enabled", False):
        raise ValueError("Use /v1/drvexo/decision for explicit candidate scoring")
    for name in ("additional_information", "model_intermediate_buffer"):
        values = getattr(request, name, None)
        if isinstance(values, dict) and any(
            str(key).startswith(("drvexo_", "_drvexo_")) or key == "direct_stage_input" for key in values
        ):
            raise ValueError("Internal DRVEXO metadata is not accepted through HTTP")
