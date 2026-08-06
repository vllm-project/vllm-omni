#!/usr/bin/env python3
"""Concurrency-threshold sweep for the runaway bug.

c=1 gave 0/240 runaways; the c=64 load test gave 26/17920. This finds where in
between it switches on, which discriminates between in-batch cross-contamination
(shows up at c=2) and pressure/preemption effects (needs high c).

Every level sends the SAME total number of requests from the SAME 4 texts that
ran away in both prior load tests, so rates are directly comparable.

Env:
  LEVELS=1,2,4,8,16,32   concurrency levels to sweep
  N_PER=128              requests per level
  ABS_S=20               runaway threshold (s)
"""
import asyncio, base64, io, os, time, wave, datetime
import aiohttp

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8091"))
MODEL = os.environ.get("MODEL", "/root/models/Qwen3-TTS-12Hz-0.6B-Base")
ROOT = os.environ.get("SEED_TTS_ROOT", "/root/datasets/seed-tts-eval")
LOCALE = os.environ.get("LOCALE", "zh")
LEVELS = [int(x) for x in os.environ.get("LEVELS", "1,2,4,8,16,32").split(",")]
N_PER = int(os.environ.get("N_PER", "128"))
ABS_S = float(os.environ.get("ABS_S", "20"))

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
                rows.append({"ref_text": p[1], "wav": wav, "target": p[3]})
    return rows


def wav_dur(b):
    try:
        with wave.open(io.BytesIO(b), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return None


async def one(session, sem, row, text):
    payload = {
        "model": MODEL, "input": text, "task_type": "Base",
        "language": "Chinese" if LOCALE == "zh" else "English",
        "ref_audio": f"data:audio/wav;base64,{row['b64']}", "ref_text": row["ref_text"],
    }
    async with sem:
        t0 = time.time()
        try:
            async with session.post(f"http://{HOST}:{PORT}/v1/audio/speech",
                                    json=payload) as r:
                data = await r.read()
        except Exception as e:
            return {"err": str(e), "text": text}
        lat = time.time() - t0
    return {"dur": wav_dur(data), "lat": lat, "text": text}


async def run_level(c, rows_by_text):
    sem = asyncio.Semaphore(c)
    work = [(rows_by_text[SUSPECT[i % len(SUSPECT)]], SUSPECT[i % len(SUSPECT)])
            for i in range(N_PER)]
    timeout = aiohttp.ClientTimeout(total=3600)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as s:
        res = await asyncio.gather(*[one(s, sem, r, t) for r, t in work])
    wall = time.time() - t0
    durs = [x["dur"] for x in res if x.get("dur")]
    bad = [x for x in res if x.get("dur") and x["dur"] >= ABS_S]
    errs = sum(1 for x in res if x.get("err"))
    med = sorted(durs)[len(durs) // 2] if durs else 0
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] c={c:<3} n={len(durs)} err={errs} wall={wall:.0f}s "
          f"median={med:.1f}s max={max(durs) if durs else 0:.1f}s "
          f"RUNAWAY={len(bad)}/{len(durs)} ({len(bad)/max(len(durs),1)*100:.1f}%)", flush=True)
    for b in bad:
        print(f"      !!! dur={b['dur']:.1f}s lat={b['lat']:.1f}s text={b['text'][:24]!r}", flush=True)
    return c, len(bad), len(durs)


async def main():
    rows = load_rows()
    by_text = {}
    for r in rows:
        if r["target"] in SUSPECT and r["target"] not in by_text:
            with open(r["wav"], "rb") as f:
                r["b64"] = base64.b64encode(f.read()).decode("ascii")
            by_text[r["target"]] = r
    print(f"sweep: levels={LEVELS} n_per_level={N_PER} threshold={ABS_S}s "
          f"texts={len(by_text)} (suspect-only)", flush=True)
    summary = []
    for c in LEVELS:
        summary.append(await run_level(c, by_text))
    print("\n=== SUMMARY ===")
    print(f"{'conc':>5} {'runaway':>9} {'n':>5}  rate")
    for c, b, n in summary:
        print(f"{c:>5} {b:>9} {n:>5}  {b/max(n,1)*100:5.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
