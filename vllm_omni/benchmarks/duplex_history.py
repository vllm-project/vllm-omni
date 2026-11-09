# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paired long-conversation experiments; no quality or latency pass threshold.

Generate/freeze a manifest, optionally synthesize CPU speech fixtures, then run
it unchanged against separately configured servers. No model judge is used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import time
import unicodedata
from collections import Counter
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.clients.duplex import ResponseHandle


def normalize(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKC", text)
        if not c.isspace() and not unicodedata.category(c).startswith("P")
    )


def score(answer: str, accepted: list[str], forbidden: list[str]) -> str:
    clean = normalize(answer)
    if clean in {normalize(v) for v in accepted}:
        return "correct"
    if not clean or clean in {normalize(v) for v in forbidden}:
        return "incorrect"
    # More elaborate paraphrases require an auditable review, not a substring
    # match that would accept 'it is not 3000' as the correct value 3000.
    return "needs_review"


def make_cases() -> list[dict]:
    evaluation = [
        ("旅行", "住宿预算", "三千元", "五千元"),
        ("配送", "收件城市", "杭州", "苏州"),
        ("会议", "开始时间", "上午九点", "上午十点"),
        ("烘焙", "面粉用量", "两百克", "三百克"),
        ("借书", "归还日期", "六月十二日", "六月十五日"),
        ("订餐", "主食", "米饭", "面条"),
        ("装修", "墙面颜色", "浅蓝色", "米白色"),
        ("露营", "集合地点", "北门", "东门"),
        ("种植", "浇水间隔", "三天", "五天"),
        ("接送", "车辆编号", "七号", "九号"),
    ]
    development = [
        ("健身", "训练时长", "二十分钟", "三十分钟"),
        ("电影", "座位", "第三排", "第五排"),
        ("洗衣", "水温", "三十度", "四十度"),
        ("游泳", "泳道", "第二道", "第四道"),
        ("舞蹈", "练习日", "星期二", "星期四"),
        ("采购", "茶叶品种", "龙井", "毛峰"),
        ("展览", "展厅", "红厅", "蓝厅"),
        ("音乐", "乐器", "钢琴", "小提琴"),
        ("摄影", "拍摄对象", "树林", "湖泊"),
        ("手工", "材料", "木头", "纸板"),
    ]
    fillers = [
        "接下来换一个话题。请用一句话解释为什么晴天的天空通常是蓝色的。这个问题与之前的安排无关。",
        "请简短说明整理桌面时怎样分类文具，只讨论分类方法，不修改我们之前确定的任何事项。",
        "请用一句话介绍散步的好处，不需要给我制定具体计划，也不要重复之前讨论过的安排。",
        "我们继续聊常识。请简短解释地图上的比例尺表示什么，不需要联系之前的任务。",
        "请简单说明阅读纸质书时怎样保护书页，这只是一个额外的问题，之前的安排保持有效。",
        "请用一句话描述晴天和阴天的区别，不需要查询天气，也不要重新确认此前已经说过的信息。",
    ]
    cases = []
    for split, topics in (("dev", development), ("eval", evaluation)):
        for category in ("fact",) if split == "dev" else ("fact", "instruction", "correction"):
            for index, (topic, field, old, new) in enumerate(topics):
                expected = new if category == "correction" else old
                suffix = "。这次对话最后回答我的安排时，请在答案末尾加上‘已确认’" if category == "instruction" else ""
                turns = [f"关于{topic}，我的{field}是{old}，请记住" + suffix + "。"]
                turns.extend(
                    fillers[i % len(fillers)] + "我们正在进行一段较长的交流。这一轮只讨论刚才提出的常识问题，"
                    "不用总结整段对话，不用复述最初的安排，也不要替我添加新的计划。"
                    "请简短作答，后面我还会继续提出其他问题。"
                    for i in range(18)
                )
                if category == "correction":
                    turns[5] = f"更正之前的信息：{topic}的{field}改为{new}，原来的{old}不再有效。"
                turns.append(f"我现在确定的{field}是什么？按之前约定的回答格式回答，不解释。")
                accepted = [expected + ("已确认" if category == "instruction" else "")]
                cases.append(
                    {
                        "id": f"{split}-{category}-{index:02}",
                        "split": split,
                        "category": category,
                        "turns": turns,
                        "accepted": accepted,
                        "forbidden": [old] if category == "correction" else [],
                    }
                )
    return cases


def read_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return cases


def audio_path(directory: Path, text: str) -> Path:
    return directory / (hashlib.sha256(text.encode()).hexdigest() + ".wav")


def synthesize(cases: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for text in dict.fromkeys(t for c in cases for t in c["turns"]):
        target = audio_path(directory, text)
        if not target.exists():
            subprocess.run(["espeak-ng", "-v", "cmn", "-s", "145", "-w", str(target), text], check=True)
    print("Synthetic speech created. Listen to key facts, negations and corrections before evaluation.")


async def next_answer(responses: AsyncIterator[ResponseHandle]) -> ResponseHandle:
    """Ignore standalone input-listening beats; await a generated response."""
    while True:
        handle = await anext(responses)
        if handle.response_id is not None or handle.decision != "listen":
            return handle


async def run_case(case: dict, args) -> dict:
    from vllm_omni.clients.duplex import (
        AudioDelta,
        DuplexClient,
        DuplexClientError,
        DuplexProtocolError,
        ErrorEvent,
        SessionConfig,
        TextDelta,
        TranscriptDelta,
    )

    config = SessionConfig(
        auto_response=False,
        turn_detection=None,
        temperature=0,
        max_output_tokens=args.max_output_tokens,
        playback_commit_policy="ack_only",
        instructions="你是中文对话助手。请遵守用户持续要求，使用最新更正，并简短回答。",
    )
    source_hash = hashlib.sha256()
    for text in case["turns"]:
        source_hash.update(text.encode())
        if args.modality == "audio":
            source_hash.update(audio_path(args.audio_dir, text).read_bytes())
    response_times: dict[str, dict] = {}
    errors = []
    turns = []
    index = None
    handle: ResponseHandle | None = None
    failed_turn = None
    try:
        async with DuplexClient(args.url, model=args.model, config=config, reconnect=None) as client:

            async def observe():
                async for event in client.events():
                    if isinstance(event, ErrorEvent):
                        errors.append(event.raw)
                    if event.response_id:
                        entry = response_times.setdefault(event.response_id, {})
                        if isinstance(event, (TextDelta, TranscriptDelta)):
                            entry.setdefault("first_text", time.monotonic())
                        if isinstance(event, AudioDelta):
                            entry.setdefault("first_audio", time.monotonic())

            observer = asyncio.create_task(observe())
            try:
                await asyncio.sleep(0)
                responses = client.responses()
                for index, text in enumerate(case["turns"]):
                    handle = None
                    if args.modality == "audio":
                        from math import gcd

                        import numpy as np
                        import soundfile as sf
                        from scipy.signal import resample_poly

                        raw, rate = await asyncio.to_thread(sf.read, audio_path(args.audio_dir, text), dtype="float32")
                        if raw.ndim > 1:
                            raw = raw.mean(axis=1)
                        divisor = gcd(rate, 16000)
                        raw = resample_poly(raw, 16000 // divisor, rate // divisor)
                        pcm = (np.clip(raw, -1, 1) * 32767).astype("<i2").tobytes()
                        for offset in range(0, len(pcm), 3200):
                            await client.append_audio(pcm[offset : offset + 3200])
                            if args.realtime_input:
                                await asyncio.sleep(len(pcm[offset : offset + 3200]) / 32000)
                        start = time.monotonic()
                        await client.commit(create_response=False)
                    else:
                        await client.send(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": text}],
                                },
                            }
                        )
                        start = time.monotonic()
                    await client.send({"type": "response.create"})

                    async def consume_response() -> ResponseHandle:
                        nonlocal handle
                        handle = await next_answer(responses)
                        async for chunk in handle.audio():
                            # Simulated playback at output speed makes full ACKs
                            # correspond to consumed audio, not mere generation.
                            await asyncio.sleep(len(chunk) / 48000)
                            await client.ack_playback(handle.played_ms, response_id=handle.response_id)
                        await handle.wait()
                        return handle

                    handle = await asyncio.wait_for(consume_response(), timeout=args.timeout)
                    if handle.played_ms:
                        await client.ack_playback(handle.played_ms, response_id=handle.response_id)
                    observed = response_times.get(handle.response_id or "", {})
                    turns.append(
                        {
                            "index": index,
                            "response_id": handle.response_id,
                            "text": handle.transcript or handle.text,
                            "ttft_ms": 1000 * (observed["first_text"] - start) if "first_text" in observed else None,
                            "ttfp_ms": 1000 * (observed["first_audio"] - start) if "first_audio" in observed else None,
                        }
                    )
                    if errors:
                        break
                    await asyncio.sleep(args.idle_seconds)
            finally:
                observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
    except (TimeoutError, DuplexClientError, StopAsyncIteration) as exc:
        # A failed conversation is an observation, not a reason to drop the
        # rest of the paired workload. Bugs and task cancellation still escape.
        if index is not None:
            failed_turn = {
                "index": index,
                "response_id": handle.response_id if handle is not None else None,
                "text": (handle.transcript or handle.text) if handle is not None else "",
            }
        error: dict[str, object] = {"type": type(exc).__name__, "message": str(exc), "turn_index": index}
        if isinstance(exc, DuplexProtocolError):
            error["raw"] = exc.raw
        errors.append(error)
    answer = turns[-1]["text"] if len(turns) == len(case["turns"]) and not errors else ""
    return {
        "case_id": case["id"],
        "input_sha256": source_hash.hexdigest(),
        "model": args.model,
        "max_output_tokens": args.max_output_tokens,
        "realtime_input": args.realtime_input,
        "category": case["category"],
        "policy": args.policy,
        "modality": args.modality,
        "idle_seconds": args.idle_seconds,
        "turns": turns,
        "errors": errors,
        "failed_turn": failed_turn,
        "expected": case["accepted"],
        "answer": answer,
        "score": "run_error"
        if errors or len(turns) != len(case["turns"])
        else score(answer, case["accepted"], case["forbidden"]),
    }


async def run(args) -> None:
    cases = [c for c in read_cases(args.manifest) if c["split"] == args.split]
    if args.limit:
        cases = cases[: args.limit]
    with args.output.open("x") as output:
        for case in cases:
            result = await run_case(case, args)
            result["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(case["id"], result["score"])


def report(paths: list[Path]) -> dict:
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    groups: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        key = (row["policy"], row["modality"], row["idle_seconds"])
        groups.setdefault(key, []).append(row)
    result = {}
    for key, values in groups.items():
        if len({v["case_id"] for v in values}) != len(values):
            raise ValueError("Duplicate case results; repeated runs must be reported separately")
        if len({v["manifest_sha256"] for v in values}) != 1:
            raise ValueError("Results use different manifests")
        metrics = {}
        for name in ("ttft_ms", "ttfp_ms"):
            measured = sorted(t[name] for v in values for t in v["turns"] if t[name] is not None)
            metrics[name] = {
                "count": len(measured),
                "p50": measured[int((len(measured) - 1) * 0.5)] if measured else None,
                "p95": measured[int((len(measured) - 1) * 0.95)] if measured else None,
            }
        result["/".join(map(str, key))] = {
            "cases": len(values),
            "scores": dict(Counter(v["score"] for v in values)),
            "latency": metrics,
            "by_category": {
                c: dict(Counter(v["score"] for v in values if v["category"] == c))
                for c in sorted({v["category"] for v in values})
            },
        }
    # Pair by scenario, never treat turns or modalities as independent cases.
    paired = {}
    for modality, idle in {(r["modality"], r["idle_seconds"]) for r in rows}:
        sides = {
            p: {
                r["case_id"]: r
                for r in rows
                if r["policy"] == p and r["modality"] == modality and r["idle_seconds"] == idle
            }
            for p in ("bounded_recent", "incremental")
        }
        if all(sides.values()):
            if sides["bounded_recent"].keys() != sides["incremental"].keys():
                raise ValueError("Paired comparison requires identical case sets")
            changes: Counter[str] = Counter()
            for case_id, baseline in sides["bounded_recent"].items():
                new = sides["incremental"][case_id]
                for setting in ("manifest_sha256", "input_sha256", "model", "max_output_tokens", "realtime_input"):
                    if baseline.get(setting) != new.get(setting):
                        raise ValueError(f"Paired comparison requires identical {setting}")
                a, b = baseline["score"], new["score"]
                label = (
                    "unresolved"
                    if a in {"needs_review", "run_error"} or b in {"needs_review", "run_error"}
                    else (
                        "improved"
                        if a != "correct" and b == "correct"
                        else "regressed"
                        if a == "correct" and b != "correct"
                        else "tied"
                    )
                )
                changes[label] += 1
            paired[f"{modality}/{idle}"] = dict(changes)
    return {"groups": result, "paired": paired, "thresholds": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    speech = sub.add_parser("audio")
    speech.add_argument("--manifest", type=Path, required=True)
    speech.add_argument("--audio-dir", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--url", required=True)
    execute.add_argument("--model", required=True)
    execute.add_argument("--policy", choices=("bounded_recent", "incremental"), required=True)
    execute.add_argument("--modality", choices=("text", "audio"), required=True)
    execute.add_argument("--audio-dir", type=Path)
    execute.add_argument("--split", choices=("dev", "eval"), default="eval")
    execute.add_argument("--limit", type=int, default=0)
    execute.add_argument("--idle-seconds", type=float, default=3.0)
    execute.add_argument("--max-output-tokens", type=int, default=128)
    execute.add_argument("--timeout", type=float, default=120)
    execute.add_argument("--realtime-input", action=argparse.BooleanOptionalAction, default=True)
    summary = sub.add_parser("report")
    summary.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "generate":
        with args.output.open("x") as output:
            for case in make_cases():
                output.write(json.dumps(case, ensure_ascii=False) + "\n")
    elif args.command == "audio":
        synthesize(read_cases(args.manifest), args.audio_dir)
    elif args.command == "run":
        if args.modality == "audio" and args.audio_dir is None:
            parser.error("audio runs require --audio-dir")
        asyncio.run(run(args))
    else:
        print(json.dumps(report(args.inputs), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
