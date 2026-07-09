# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex full-pipeline TTS-pass CER regression gate.

Generates the anchored en-24 corpus through the ``nemotron_labs_audex_full``
TTS pass, writes the WAVs + ``refs.tsv``, and (if an ASR transcript log is
supplied on a second invocation) compares corpus CER against the committed
anchor (``tests/assets/audex/tts_cer_anchor.json``: gate = anchor + 2 pp).

Two-step flow (ASR runs out-of-process; the repo does not vendor an ASR
model):

    # 1. Generate the corpus WAVs (writes <out>/refs.tsv too):
    python examples/offline_inference/speech_to_speech/audex/tts_cer_gate.py \\
        --output-dir results/audex_full_tts_gate

    # 2. Transcribe with your ASR of choice, producing a log where each
    #    clip line contains `expected:` and `asr:` rows (the
    #    transcribe-tts-output skill emits this format), then gate:
    python examples/offline_inference/speech_to_speech/audex/tts_cer_gate.py \\
        --check-asr-log results/audex_full_tts_gate/asr.log

Exit code 0 = gate met; 1 = gate exceeded or corpus incomplete.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
ANCHOR_PATH = _REPO_ROOT / "tests" / "assets" / "audex" / "tts_cer_anchor.json"
CORPUS_PATH = _REPO_ROOT / "tests" / "assets" / "audex" / "texts_en24.tsv"
SAMPLE_RATE = 16_000


def _load_anchor() -> dict:
    return json.loads(ANCHOR_PATH.read_text())


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _edit_distance(a: str, b: str) -> int:
    rows = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, rows[0] = rows[0], i
        for j, cb in enumerate(b, 1):
            cur = min(rows[j] + 1, rows[j - 1] + 1, prev + (ca != cb))
            prev, rows[j] = rows[j], cur
    return rows[-1]


def check_asr_log(log_path: Path) -> int:
    """Corpus CER from a transcribe-tts-output-style log vs the anchor gate."""
    anchor = _load_anchor()
    pairs: list[tuple[str, str]] = []
    expected: str | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("expected:"):
            expected = line[len("expected:") :].strip()
        elif line.startswith("asr") and ":" in line and expected is not None:
            pairs.append((expected, line.split(":", 1)[1].strip()))
            expected = None

    if len(pairs) < anchor["corpus_clips"]:
        print(f"GATE FAIL: only {len(pairs)}/{anchor['corpus_clips']} clips found in {log_path}")
        return 1

    edits = 0
    ref_chars = 0
    for ref, hyp in pairs:
        ref_n, hyp_n = _normalize(ref), _normalize(hyp)
        edits += _edit_distance(hyp_n, ref_n)
        ref_chars += len(ref_n)
    cer = 100.0 * edits / max(1, ref_chars)
    gate = float(anchor["gate_cer_percent"])
    verdict = "OK" if cer <= gate else "FAIL"
    print(
        f"GATE {verdict}: clips={len(pairs)} CER={cer:.2f}% (edits={edits}/{ref_chars}) "
        f"vs anchor {anchor['anchor_cer_percent']}% + 2pp gate {gate}%"
    )
    return 0 if cer <= gate else 1


def generate(output_dir: Path, model: str, deploy_config: str | None) -> int:
    import numpy as np
    import soundfile as sf
    import torch

    from vllm_omni import Omni
    from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

    corpus = [line.split("\t", 1) for line in CORPUS_PATH.read_text().splitlines() if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = Omni(
        model=model,
        deploy_config=deploy_config or str(_REPO_ROOT / "vllm_omni" / "deploy" / "nemotron_labs_audex_full.yaml"),
        trust_remote_code=True,
    )
    refs_lines = []
    for utt, text in corpus:
        outputs = engine.generate([{"prompt": build_cond_prompt(text.strip()), "modalities": ["audio"]}])
        chunks = []
        for req_output in outputs:
            mm = getattr(req_output, "multimodal_output", None) or {}
            if "audio" in mm:
                vals = mm["audio"] if isinstance(mm["audio"], list) else [mm["audio"]]
                chunks.extend(torch.as_tensor(v).float().cpu().reshape(-1) for v in vals if v is not None)
        pcm = torch.cat(chunks) if chunks else torch.empty(0)
        if pcm.numel() == 0:
            print(f"GATE FAIL: empty audio for {utt}")
            return 1
        arr = (np.clip(pcm.numpy(), -1.0, 1.0) * 32767.0).astype(np.int16)
        sf.write(str(output_dir / f"{utt}.wav"), arr, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        refs_lines.append(f"{utt}\t{text}\n")
        print(f"{utt} dur={pcm.numel() / SAMPLE_RATE:.2f}s", flush=True)
    (output_dir / "refs.tsv").write_text("".join(refs_lines))
    print(f"Generated {len(refs_lines)} clips; transcribe them and re-run with --check-asr-log.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audex full-pipeline TTS CER anchor gate")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-Labs-Audex-2B")
    parser.add_argument("--deploy-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/audex_full_tts_gate")
    parser.add_argument("--check-asr-log", type=str, default=None)
    args = parser.parse_args()

    if args.check_asr_log:
        return check_asr_log(Path(args.check_asr_log))
    return generate(Path(args.output_dir), args.model, args.deploy_config)


if __name__ == "__main__":
    sys.exit(main())
