# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Spawn-safe real-weight offline/HTTP worker for ZONOS2 E2E tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

if os.environ.get("ZONOS2_E2E_TRACE_DIR"):
    from tests.e2e.zonos2.trace import install_trace

    install_trace()


async def offline(directory: Path, concurrent: bool):
    import numpy as np
    import soundfile as sf
    import torch
    from vllm import SamplingParams

    from tests.e2e.zonos2.audio_checks import check_audio, check_lifecycle, read_trace
    from tests.e2e.zonos2.runtime import deployment, model_path, reference_audio
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
    from vllm_omni.model_executor.models.zonos2.zonos2_processor import Zonos2Processor
    from vllm_omni.model_executor.models.zonos2.zonos2_speaker import Zonos2SpeakerEncoder

    processor = Zonos2Processor(Zonos2Config.from_pretrained(model_path(), local_files_only=True))
    cases = (
        [
            {"text": "Hello, this is the first Zonos2 baseline sample.", "language": "en_us", "seed": 42},
            {"text": "你好，这是第一条中文基线样本。", "language": "cmn", "seed": 43},
            {"text": "The quick brown fox jumps over the lazy dog.", "language": "en_us", "seed": 44},
            {
                "text": "This sentence should sound like the reference speaker. A second sentence makes it longer.",
                "language": "en_us",
                "seed": 45,
            },
        ]
        if concurrent
        else [{"text": "Hello, this is the first Zonos2 baseline sample.", "language": "en_us", "seed": 42}]
    )
    if concurrent:
        encoder = Zonos2SpeakerEncoder()
        embeddings = []
        for second in (False, True):
            wav, rate = sf.read(reference_audio(second), dtype="float32", always_2d=True)
            embeddings.append(encoder.encode(wav.T, rate))
        assert not torch.equal(*embeddings)
        for index, case in enumerate(cases):
            case["speaker_embedding"] = embeddings[index % 2]
    prompts = [
        processor.build_prompt(case["text"], language=case["language"], speaker_embedding=case.get("speaker_embedding"))
        for case in cases
    ]
    engine = AsyncOmni(model=model_path(), deploy_config=str(deployment(directory)), stage_init_timeout=900)
    reports = []
    comparison = []

    async def one(index, prefix, *, greedy=False):
        label = f"{prefix}_{index}"
        case = cases[index]
        sp = SamplingParams(
            temperature=0 if greedy else 1.15,
            top_k=106,
            min_p=0.18,
            top_p=1,
            repetition_penalty=1.2,
            max_tokens=48 if greedy else 1024,
            ignore_eos=greedy,
            seed=case["seed"],
            detokenize=False,
        )
        final = None
        events = 0
        async for output in engine.generate(
            prompts[index],
            request_id=label,
            sampling_params_list=[sp, SamplingParams(temperature=0, max_tokens=65536, detokenize=True)],
        ):
            final = output
            events += 1
        assert final is not None and final.finished and not final.error
        mm = final.multimodal_output
        audio = mm.get("audio", mm.get("model_outputs"))
        assert audio is not None, mm.keys()
        if isinstance(audio, list):
            audio = torch.cat([part.reshape(-1) for part in audio])
        audio = audio.float().cpu().numpy().reshape(-1)
        sr = mm.get("sr", 44100)
        if isinstance(sr, list):
            sr = sr[-1]
        sr = int(sr)
        # Let the finished-ID notification be scheduled before reading trace.
        for _ in range(100):
            rows = read_trace(directory / "trace")
            try:
                life = check_lifecycle(rows, label, allow_cap=greedy)
                break
            except AssertionError:
                await asyncio.sleep(0.05)
        else:
            life = check_lifecycle(rows, label, allow_cap=greedy)
        assert life["seed"] == case["seed"]
        report = check_audio(audio, sr, frames=life["decoded_samples"] // 512)
        report.update(label=label, events=events, lifecycle=life)
        np.save(directory / f"{label}.npy", audio)
        sf.write(directory / f"{label}.wav", audio, sr)
        reports.append(report)
        print("REAL_OFFLINE_PASS", label, report["samples"], life["frames"], flush=True)
        return life

    try:
        if concurrent:
            # Fixed-budget greedy trajectories test routing without promising
            # identical stochastic draws when batch-dependent logits differ.
            solo = [await asyncio.wait_for(one(i, "solo", greedy=True), 300) for i in range(4)]
            batched = await asyncio.wait_for(asyncio.gather(*(one(i, "batch", greedy=True) for i in range(4))), 600)
            for index, (a, b) in enumerate(zip(solo, batched, strict=True)):
                ca = torch.load(directory / "trace" / f"{a['request']}.pt", weights_only=True)
                cb = torch.load(directory / "trace" / f"{b['request']}.pt", weights_only=True)
                differences = (ca != cb).nonzero()
                comparison.append(
                    {
                        "case": index,
                        "code_agreement": float((ca == cb).float().mean()),
                        "first_difference": differences[0].tolist() if len(differences) else None,
                    }
                )
                if len(differences):
                    step = int(differences[0, 0])
                    la = torch.load(directory / "trace" / f"{a['request']}_step{step:02}.pt", weights_only=True)
                    lb = torch.load(directory / "trace" / f"{b['request']}_step{step:02}.pt", weights_only=True)
                    assert torch.equal(la["history"], lb["history"])
                    comparison[-1]["first_difference_logits_max_error"] = float(
                        (la["logits"] - lb["logits"]).abs().max()
                    )
                    comparison[-1]["first_difference_solo_top2_margin"] = float(
                        la["logits"].topk(2, dim=-1).values.diff(dim=-1).abs().min()
                    )
            await asyncio.wait_for(asyncio.gather(*(one(i, "default") for i in range(4))), 900)
        else:
            await asyncio.wait_for(one(0, "offline"), 600)
    finally:
        engine.close()
    if concurrent:
        from vllm_omni.model_executor.models.zonos2.zonos2_codec import DACStreamDecoder, LocalDAC

        rows = read_trace(directory / "trace")
        for kind in ("talker_cleanup", "dac_cleanup"):
            assert [row for row in rows if row["kind"] == kind][-1]["remaining"] == []
        reference = LocalDAC(device="cuda:0")
        for _ in range(3):
            reference.decode(torch.zeros((9, 16), dtype=torch.long))
        for report in reports:
            label = report["label"]
            index = int(label.rsplit("_", 1)[1])
            key = report["lifecycle"]["request"]
            conditioning = [row for row in rows if row["kind"] == "conditioning" and row["request"] == key]
            assert len(conditioning) == 1
            info = prompts[index]["additional_information"]

            def digest(tensor):
                return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

            assert conditioning[0]["prompt_sha"] == digest(info["zonos2_frames"])
            assert conditioning[0]["speaker_sha"] == digest(info["zonos2_speaker_embedding"])
            codes = torch.load(directory / "trace" / f"{key}.pt", weights_only=True).long()
            chunks = [row for row in rows if row["kind"] == "decode" and row["request"] == key]
            stream = DACStreamDecoder(reference.decode)
            reconstructed = []
            for sequence, row in enumerate(chunks):
                raw = codes[: row["raw_frames"]]
                assert row["raw_sha"] == digest(raw), (key, sequence)
                reconstructed.append(
                    stream.push(key, raw, final=row["last_chunk"], target=row["target"], sequence=sequence)
                )
            expected = torch.cat(reconstructed).numpy()
            actual = np.load(directory / f"{label}.npy")
            np.testing.assert_allclose(
                actual, expected, rtol=0, atol=2e-7, err_msg=f"Audio request routing mismatch for {label}"
            )
            stream.cleanup([key])
            assert not stream.states and not stream.closed
            report["request_routing_verified"] = True
            report["independent_dac_max_error"] = float(np.abs(actual - expected).max())
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "real_weights": True,
                "concurrency": 4 if concurrent else 1,
                "cases": reports,
                "solo_vs_batch_diagnostics": comparison,
            },
            indent=2,
        )
    )


def main():
    if sys.argv[1] == "--serve":
        from vllm_omni.entrypoints.cli.main import main as serve

        sys.argv = [sys.argv[0], "serve", *sys.argv[2:]]
        serve()
    else:
        directory = Path(sys.argv[2])
        asyncio.run(offline(directory, concurrent=sys.argv[1] == "--concurrent"))


if __name__ == "__main__":
    main()
