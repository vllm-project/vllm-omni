#!/usr/bin/env python3
"""Serial (concurrency=1) runaway probe.

Sends one request at a time, so scheduling / batching / preemption are out of
the picture: anything that over-generates here is the sampler alone. Targets
the texts that were empirically enriched for runaway across two independent
17,920-request runs, and interleaves control texts for a baseline rate.

Env:
  REPS=30            repetitions per text
  ABS_S=20           runaway threshold (seconds of audio)
  REP_PENALTY=       override repetition_penalty (default: server yaml value)
  TEXTS=suspect|control|both
"""
import base64, io, json, os, sys, time, wave, datetime, urllib.request

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8091"))
MODEL = os.environ.get("MODEL", "/root/models/Qwen3-TTS-12Hz-0.6B-Base")
ROOT = os.environ.get("SEED_TTS_ROOT", "/root/datasets/seed-tts-eval")
LOCALE = os.environ.get("LOCALE", "zh")
REPS = int(os.environ.get("REPS", "30"))
ABS_S = float(os.environ.get("ABS_S", "20"))
REP_PENALTY = os.environ.get("REP_PENALTY", "")
WHICH = os.environ.get("TEXTS", "both")

# Texts that ran away in BOTH the PD 1p1d run and the single-card run.
SUSPECT = [
    "自动驾驶将大幅提升出行安全，效率。",
    "真正落地成为产品服务进入每个人的生活。",
    "打造更贴心，更细致的个性化服务。",
    "目前中国互联网国际化还在开拓阶段。",
]


def load_rows():
    rows = []
    with open(os.path.join(ROOT, LOCALE, "meta.lst")) as f:
        for line in f:
            p = line.strip().split("|")
            if len(p) < 4:
                continue
            wav = os.path.join(ROOT, LOCALE, p[2])
            if os.path.isfile(wav):
                rows.append({"utt": p[0], "ref_text": p[1], "wav": wav, "target": p[3]})
    return rows


def wav_dur(b):
    try:
        with wave.open(io.BytesIO(b), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return None


def send(row, text):
    with open(row["wav"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    payload = {
        "model": MODEL, "input": text, "task_type": "Base",
        "language": "Chinese" if LOCALE == "zh" else "English",
        "ref_audio": f"data:audio/wav;base64,{b64}", "ref_text": row["ref_text"],
    }
    if REP_PENALTY:
        payload["repetition_penalty"] = float(REP_PENALTY)
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/audio/speech",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = r.read()
    return wav_dur(data), time.time() - t0


def main():
    rows = load_rows()
    by_text = {r["target"]: r for r in rows}
    groups = []
    if WHICH in ("suspect", "both"):
        groups += [("SUSPECT", t) for t in SUSPECT if t in by_text]
    if WHICH in ("control", "both"):
        ctl = [r["target"] for r in rows if r["target"] not in set(SUSPECT)][:4]
        groups += [("CONTROL", t) for t in ctl]

    print(f"serial probe: reps={REPS} threshold={ABS_S}s rep_penalty={REP_PENALTY or 'yaml-default'} "
          f"groups={len(groups)}", flush=True)
    grand = {}
    for kind, text in groups:
        row = by_text[text]
        durs, bad = [], 0
        for i in range(REPS):
            try:
                d, lat = send(row, text)
            except Exception as e:
                print(f"    ERR {e}", flush=True); continue
            if d is None:
                continue
            durs.append(d)
            if d >= ABS_S:
                bad += 1
                print(f"    !!! runaway rep={i+1} dur={d:.1f}s lat={lat:.1f}s", flush=True)
        med = sorted(durs)[len(durs) // 2] if durs else 0
        print(f"  [{kind}] {text[:26]!r} n={len(durs)} median={med:.1f}s "
              f"max={max(durs) if durs else 0:.1f}s runaway={bad}/{len(durs)}", flush=True)
        grand.setdefault(kind, [0, 0])
        grand[kind][0] += bad
        grand[kind][1] += len(durs)
    print()
    for kind, (b, n) in grand.items():
        print(f"TOTAL {kind}: {b}/{n} = {b/n*100 if n else 0:.1f}% runaway", flush=True)


if __name__ == "__main__":
    main()
