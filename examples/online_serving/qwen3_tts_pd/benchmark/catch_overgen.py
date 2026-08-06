#!/usr/bin/env python3
"""Catcher for the PD never-stop / over-generation bug.

Sends seed-tts prompts concurrently to /v1/audio/speech, parses each returned
WAV's duration, and flags requests whose audio is wildly disproportionate to
the input text (the codec-EOS-miss signature: model keeps babbling past where
it should stop). Loops rounds until an outlier is caught. Prints the offending
utt/text and a wall-clock timestamp so the server log can be grepped.
"""
import asyncio, base64, io, json, os, sys, time, wave, datetime

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8091"))
MODEL = os.environ.get("MODEL", "/root/models/Qwen3-TTS-12Hz-0.6B-Base")
ROOT = os.environ.get("SEED_TTS_ROOT", "/root/datasets/seed-tts-eval")
LOCALE = os.environ.get("LOCALE", "zh")
CONC = int(os.environ.get("CONC", "64"))
ROUNDS = int(os.environ.get("ROUNDS", "20"))
# Over-generation flag: absolute seconds, and seconds-per-input-char ratio.
# Normal zh seed-tts ≈ 4-8s for 10-30 chars → ~0.3s/char. Flag >>that.
ABS_S = float(os.environ.get("ABS_S", "20"))
PER_CHAR_S = float(os.environ.get("PER_CHAR_S", "1.2"))

import aiohttp

def load_rows():
    meta = os.path.join(ROOT, LOCALE, "meta.lst")
    rows = []
    with open(meta) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            utt, ref_text, wav_rel, target = parts[0], parts[1], parts[2], parts[3]
            wav = os.path.join(ROOT, LOCALE, wav_rel)
            if os.path.isfile(wav):
                rows.append((utt, ref_text, wav, target))
    return rows

def wav_duration(b: bytes):
    try:
        with wave.open(io.BytesIO(b), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return None

async def one(session, sem, row):
    utt, ref_text, wav, target = row
    with open(wav, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    payload = {
        "model": MODEL, "input": target, "task_type": "Base",
        "language": "Chinese" if LOCALE == "zh" else "English",
        "ref_audio": f"data:audio/wav;base64,{b64}", "ref_text": ref_text,
    }
    async with sem:
        t0 = time.time()
        try:
            async with session.post(f"http://{HOST}:{PORT}/v1/audio/speech",
                                    json=payload) as r:
                data = await r.read()
        except Exception as e:
            return {"utt": utt, "err": str(e), "target": target}
        dt = time.time() - t0
    dur = wav_duration(data)
    return {"utt": utt, "target": target, "chars": len(target),
            "dur": dur, "lat": dt, "bytes": len(data),
            "is_wav": data[:4] == b"RIFF", "t0": t0}

async def run_round(rows):
    sem = asyncio.Semaphore(CONC)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[one(session, sem, r) for r in rows])
    return results

def analyze(rn, results):
    flagged = []
    durs = [x["dur"] for x in results if x.get("dur")]
    maxd = max(durs) if durs else 0
    med = sorted(durs)[len(durs)//2] if durs else 0
    nbad = sum(1 for x in results if not x.get("is_wav"))
    for x in results:
        d = x.get("dur")
        if d is None:
            continue
        if d >= ABS_S or (x["chars"] > 0 and d / x["chars"] >= PER_CHAR_S and d > 8):
            flagged.append(x)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] round {rn}: n={len(results)} non_wav={nbad} "
          f"dur median={med:.1f}s max={maxd:.1f}s flagged={len(flagged)}", flush=True)
    for x in flagged:
        tt = datetime.datetime.fromtimestamp(x["t0"]).strftime("%H:%M:%S")
        print(f"    !!! CAUGHT utt={x['utt']} chars={x['chars']} dur={x['dur']:.1f}s "
              f"lat={x['lat']:.1f}s sent_at={tt} text={x['target'][:40]!r}", flush=True)
    return flagged

async def main():
    rows = load_rows()
    print(f"loaded {len(rows)} {LOCALE} rows; conc={CONC} rounds={ROUNDS} "
          f"flag: abs>={ABS_S}s or >={PER_CHAR_S}s/char", flush=True)
    any_flag = []
    for rn in range(1, ROUNDS + 1):
        results = await run_round(rows)
        f = analyze(rn, results)
        any_flag += f
    print(f"DONE total_flagged={len(any_flag)}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
