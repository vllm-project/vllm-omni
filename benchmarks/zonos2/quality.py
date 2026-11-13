# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Local Whisper-large-v3 WER/CER, native speaker cosine, UTMOS and failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import whisper
    import zhconv

    from benchmarks.zonos2.protocol import THRESHOLDS, edit_counts, failures, normalize_asr
    from vllm_omni.model_executor.models.zonos2.zonos2_speaker import Zonos2SpeakerEncoder
    from vllm_omni.model_executor.models.zonos2.zonos2_textnorm import Zonos2TextNormalizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    for asset in json.loads((args.assets / "manifest.json").read_text()):
        hasher = hashlib.sha256()
        with Path(asset["path"]).open("rb") as file:
            while data := file.read(8 * 1024 * 1024):
                hasher.update(data)
        assert hasher.hexdigest() == asset["sha256"]
    asr = whisper.load_model(str(args.assets / "large-v3.pt"), device="cuda:0")
    mos = torch.jit.load(str(args.assets / "utmos.jit"), map_location="cuda:0").eval()
    speaker = Zonos2SpeakerEncoder()
    tn = Zonos2TextNormalizer()
    reference, reference_rate = sf.read(manifest["reference_audio"], dtype="float32", always_2d=True)
    reference_embedding = speaker.encode(reference.T, reference_rate)
    rows = []
    for run in args.runs:
        result = json.loads((run / "result.json").read_text())
        for item in (row for row in result["rows"] if row["round"] == 0):
            bundle = next(row for row in manifest["cases"] if row["id"] == item["case"])
            samples, rate = sf.read(run / f"{item['label']}.wav", dtype="float32")
            assert rate == 44100 and samples.ndim == 1 and np.isfinite(samples).all() and len(samples)
            wave = torch.from_numpy(samples)
            audio16 = torchaudio.functional.resample(wave, rate, 16000).numpy()
            decoded = asr.transcribe(
                audio16,
                language="zh" if bundle["language"] == "cmn" else "en",
                task="transcribe",
                temperature=0,
                beam_size=5,
                condition_on_previous_text=False,
                initial_prompt="以下为普通话简体中文转写。" if bundle["language"] == "cmn" else None,
            )
            raw_hyp = decoded["text"].strip()
            if bundle["language"] == "cmn":
                raw_hyp = zhconv.convert(raw_hyp, "zh-cn")
            hypothesis = tn.normalize(raw_hyp, bundle["language"]) if raw_hyp else ""
            reference_units = normalize_asr(bundle["truth"], bundle["language"])
            hyp_units = normalize_asr(hypothesis, bundle["language"])
            counts = edit_counts(reference_units, hyp_units)
            with torch.inference_mode():
                score = float(mos(torch.from_numpy(audio16).unsqueeze(0).to("cuda:0"))[0].item())
            assert np.isfinite(score), "Non-finite UTMOS score"
            cosine = None
            if bundle["clone"]:
                embedding = speaker.encode(samples, rate)
                cosine = float(torch.nn.functional.cosine_similarity(embedding, reference_embedding, dim=0))
            metric = "zh_cer" if bundle["language"] == "cmn" else "en_wer"
            row = {
                "run": run.name,
                "case": item["case"],
                "language": bundle["language"],
                "truth": bundle["truth"],
                "asr_raw": raw_hyp,
                "asr_normalized": hypothesis,
                "error_counts": counts,
                metric: counts["rate"],
                "utmos": score,
                "speaker_cosine": cosine,
                "duration_s": len(samples) / rate,
                "rms": float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
                "reached_cap": item.get("reached_cap", False),
                "wave_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            }
            row["failures"] = failures(row)
            rows.append(row)
            print("QUALITY", run.name, row["case"], metric, row[metric], score, cosine, row["failures"], flush=True)
    aggregate = {}
    for name in sorted({row["run"] for row in rows}):
        group = [row for row in rows if row["run"] == name]
        summary = {
            "samples": len(group),
            "failed_samples": [row["case"] for row in group if row["failures"]],
            "utmos_mean": float(np.mean([row["utmos"] for row in group])),
            "duration_mean_s": float(np.mean([row["duration_s"] for row in group])),
        }
        for language, metric in (("en_us", "en_wer"), ("cmn", "zh_cer")):
            part = [row for row in group if row["language"] == language]
            if part:
                summary[metric] = sum(row["error_counts"]["errors"] for row in part) / sum(
                    row["error_counts"]["reference_units"] for row in part
                )
        clones = [row["speaker_cosine"] for row in group if row["speaker_cosine"] is not None]
        summary["speaker_cosine_mean"] = float(np.mean(clones)) if clones else None
        aggregate[name] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "thresholds": THRESHOLDS,
                "asr": "OpenAI Whisper large-v3, beam5, T0; EN/ZH",
                "normalization": "NFKC, punctuation, NeMo TN on both spoken truth and ASR; zhconv zh-cn",
                "speaker": "cosine of Qwen3 2048D voice embeddings (native encoder; not WavLM SV)",
                "utmos": "balacoon/utmos TorchScript, 16k float32",
                "rows": rows,
                "aggregate": aggregate,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
