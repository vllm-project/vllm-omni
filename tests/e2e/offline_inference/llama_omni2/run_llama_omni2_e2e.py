# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.omni import Omni


@dataclass
class RequestResult:
    request_id: str
    text_parts: list[str] = field(default_factory=list)
    text_token_ids: list[int] = field(default_factory=list)
    text_logprobs: list[list[dict[str, Any]]] = field(default_factory=list)
    codec_token_ids: list[int] = field(default_factory=list)
    audio_chunks: list[torch.Tensor] = field(default_factory=list)
    sample_rate: int | None = None
    sequence_indices: list[int] = field(default_factory=list)
    consumed_units: list[int] = field(default_factory=list)
    terminal_audio_chunks: int = 0
    output_events: list[str] = field(default_factory=list)

    def add_output(self, item: Any) -> None:
        self.output_events.append(str(item.final_output_type))
        request_output = item.request_output
        if request_output is None or not request_output.outputs:
            return
        output = request_output.outputs[0]
        if item.final_output_type == "text":
            self.text_parts.append(output.text)
            self.text_token_ids.extend(int(token_id) for token_id in output.token_ids)
            for candidates in output.logprobs or []:
                self.text_logprobs.append(
                    [
                        {
                            "token_id": int(token_id),
                            "logprob": float(logprob.logprob),
                            "rank": logprob.rank,
                            "decoded_token": logprob.decoded_token,
                        }
                        for token_id, logprob in sorted(
                            candidates.items(),
                            key=lambda candidate: (
                                candidate[1].rank is None,
                                candidate[1].rank,
                                candidate[0],
                            ),
                        )
                    ]
                )
            return
        if item.final_output_type != "audio":
            return

        multimodal_output = item.multimodal_output or {}
        sample_rate = _scalar_int(multimodal_output.get("sr"))
        if sample_rate is not None:
            self.sample_rate = sample_rate
        sequence_index = _scalar_int(multimodal_output.get("sequence_index"))
        if sequence_index is not None:
            self.sequence_indices.append(sequence_index)
        consumed_units = _scalar_int(multimodal_output.get("consumed_units"))
        if consumed_units is not None:
            self.consumed_units.append(consumed_units)
        if _scalar_bool(multimodal_output.get("finished")):
            self.terminal_audio_chunks += 1
        self.codec_token_ids.extend(_int_list(multimodal_output.get("codec_units")))

        audio = multimodal_output.get(
            "audio",
            multimodal_output.get("model_outputs"),
        )
        chunks = audio if isinstance(audio, list) else [audio]
        for chunk in chunks:
            if isinstance(chunk, torch.Tensor) and chunk.numel():
                self.audio_chunks.append(chunk.detach().to(torch.float32).cpu().reshape(-1))

    def validate(self, *, require_multiple_audio_chunks: bool) -> None:
        assert self.text_token_ids, f"{self.request_id}: no text token IDs"
        assert self.audio_chunks, f"{self.request_id}: no audio output"
        assert self.sample_rate == 24000, f"{self.request_id}: expected 24 kHz, got {self.sample_rate}"
        if require_multiple_audio_chunks:
            assert len(self.audio_chunks) > 1, (
                f"{self.request_id}: expected multiple audio chunks, got {len(self.audio_chunks)}"
            )
        if self.sequence_indices:
            assert self.sequence_indices == list(range(len(self.sequence_indices))), (
                f"{self.request_id}: non-contiguous sequence indices {self.sequence_indices}"
            )
        if self.consumed_units:
            assert all(
                current > previous
                for previous, current in zip(
                    self.consumed_units,
                    self.consumed_units[1:],
                    strict=False,
                )
            ), f"{self.request_id}: non-monotonic consumed units {self.consumed_units}"
        assert self.terminal_audio_chunks == 1, (
            f"{self.request_id}: expected exactly one terminal audio chunk, got {self.terminal_audio_chunks}"
        )
        waveform = self.waveform()
        assert waveform.numel() > 0
        assert torch.isfinite(waveform).all()
        assert float(waveform.abs().max()) > 0

    def waveform(self) -> torch.Tensor:
        return torch.cat(self.audio_chunks) if self.audio_chunks else torch.empty(0)

    def write_artifacts(self, output_dir: Path) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        waveform = self.waveform()
        wav_path = output_dir / f"{self.request_id}.wav"
        sf.write(wav_path, waveform.numpy(), self.sample_rate or 24000)
        return {
            "request_id": self.request_id,
            "events": self.output_events,
            "text": "".join(self.text_parts),
            "text_token_ids": self.text_token_ids,
            "text_logprobs": self.text_logprobs,
            "codec_token_ids": self.codec_token_ids,
            "audio_chunks": len(self.audio_chunks),
            "audio_samples": int(waveform.numel()),
            "audio_peak": float(waveform.abs().max()),
            "sample_rate": self.sample_rate,
            "sequence_indices": self.sequence_indices,
            "consumed_units": self.consumed_units,
            "terminal_audio_chunks": self.terminal_audio_chunks,
            "wav_path": str(wav_path),
        }


def _scalar_int(value: Any) -> int | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, torch.Tensor):
        if not value.numel():
            return None
        value = value.reshape(-1)[0].item()
    return int(value) if value is not None else None


def _scalar_bool(value: Any) -> bool:
    scalar = _scalar_int(value)
    return bool(scalar) if scalar is not None else False


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            result.extend(_int_list(item))
        return result
    return [int(value)]


def _text_prompt(text: str) -> dict[str, Any]:
    return {
        "prompt": (
            "<|im_start|>system\nYou are a concise helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "modalities": ["text", "audio"],
    }


def _speech_prompt() -> dict[str, Any]:
    sample_rate = 16000
    samples = np.arange(sample_rate, dtype=np.float32)
    waveform = 0.2 * np.sin(2 * np.pi * 440.0 * samples / sample_rate)
    return {
        "prompt": (
            "<|im_start|>system\nYou are a concise helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<speech>\n"
            "Briefly acknowledge the audio.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"audio": [(waveform, sample_rate)]},
        "modalities": ["text", "audio"],
    }


def _requested_text_prompts(args: argparse.Namespace) -> list[str]:
    return args.text_prompts or ["Answer with exactly one word: OK"]


async def _collect(
    engine: AsyncOmni,
    *,
    prompt: dict[str, Any],
    request_id: str,
) -> RequestResult:
    result = RequestResult(request_id=request_id)
    async for item in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=None,
        output_modalities=["text", "audio"],
    ):
        result.add_output(item)
    return result


def _collect_sync(
    engine: Omni,
    *,
    prompt: dict[str, Any],
    request_id: str,
) -> RequestResult:
    result = RequestResult(request_id=request_id)
    for item in engine.generate(
        [prompt],
        sampling_params_list=None,
        use_tqdm=False,
    ):
        result.add_output(item)
    return result


def _engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    worker_extension_cls = None
    if args.verify_tp_shards:
        worker_extension_cls = (
            "tests.e2e.offline_inference.llama_omni2.worker_extension.LlamaOmni2ValidationWorkerExtension"
        )
    kwargs: dict[str, Any] = {
        "model": args.model,
        "deploy_config": args.deploy_config,
        "stage_init_timeout": args.stage_init_timeout,
        "init_timeout": args.init_timeout,
        "trust_remote_code": False,
        "log_stats": False,
        "worker_extension_cls": worker_extension_cls,
    }
    if args.entrypoint == "sync":
        kwargs["async_chunk"] = False
    return kwargs


def _require_multiple_audio_chunks(args: argparse.Namespace) -> bool:
    return args.entrypoint == "async"


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    engine = Omni(**_engine_kwargs(args)) if args.entrypoint == "sync" else AsyncOmni(**_engine_kwargs(args))
    started = time.perf_counter()
    tp_shards = None
    try:
        if args.verify_tp_shards:
            assert isinstance(engine, AsyncOmni), "TP shard RPC verification requires the async entrypoint"
            tp_shards = {
                str(stage_id): await engine.collective_rpc(
                    method="llama_omni2_parameter_shapes",
                    stage_ids=[stage_id],
                )
                for stage_id in (0, 1)
            }
        if args.mode == "text":
            if isinstance(engine, AsyncOmni):
                results = [
                    await _collect(
                        engine,
                        prompt=_text_prompt(text_prompt),
                        request_id=f"{args.label}-text-{index}",
                    )
                    for index, text_prompt in enumerate(_requested_text_prompts(args))
                ]
            else:
                results = [
                    _collect_sync(
                        engine,
                        prompt=_text_prompt(text_prompt),
                        request_id=f"{args.label}-text-{index}",
                    )
                    for index, text_prompt in enumerate(_requested_text_prompts(args))
                ]
        elif args.mode == "speech":
            if isinstance(engine, AsyncOmni):
                results = [
                    await _collect(
                        engine,
                        prompt=_speech_prompt(),
                        request_id=f"{args.label}-speech",
                    )
                ]
            else:
                results = [
                    _collect_sync(
                        engine,
                        prompt=_speech_prompt(),
                        request_id=f"{args.label}-speech",
                    )
                ]
        else:
            assert isinstance(engine, AsyncOmni), "concurrent mode requires the async entrypoint"
            results = list(
                await asyncio.gather(
                    _collect(
                        engine,
                        prompt=_text_prompt("Say alpha in one short sentence."),
                        request_id=f"{args.label}-alpha",
                    ),
                    _collect(
                        engine,
                        prompt=_text_prompt("Say beta in one short sentence."),
                        request_id=f"{args.label}-beta",
                    ),
                )
            )
    finally:
        engine.shutdown()

    require_multiple = _require_multiple_audio_chunks(args)
    for result in results:
        result.validate(require_multiple_audio_chunks=require_multiple)
    if args.mode == "concurrent":
        assert results[0].text_token_ids != results[1].text_token_ids, (
            "concurrent requests unexpectedly produced identical text token streams"
        )
        assert not torch.equal(results[0].waveform(), results[1].waveform()), (
            "concurrent requests unexpectedly produced identical waveforms"
        )

    output_dir = Path(args.output_dir)
    payload = {
        "mode": args.mode,
        "label": args.label,
        "model": args.model,
        "deploy_config": args.deploy_config,
        "elapsed_s": time.perf_counter() - started,
        "tp_shards": tp_shards,
        "requests": [result.write_artifacts(output_dir) for result in results],
    }
    result_path = output_dir / f"{args.label}-{args.mode}.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    payload["result_path"] = str(result_path)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--mode",
        choices=("text", "speech", "concurrent"),
        required=True,
    )
    parser.add_argument("--stage-init-timeout", type=float, default=300)
    parser.add_argument("--init-timeout", type=float, default=600)
    parser.add_argument("--verify-tp-shards", action="store_true")
    parser.add_argument(
        "--entrypoint",
        choices=("async", "sync"),
        default="async",
    )
    parser.add_argument(
        "--text-prompt",
        action="append",
        dest="text_prompts",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = asyncio.run(_run(args))
    print("LLAMA_OMNI2_E2E_OK " + json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
