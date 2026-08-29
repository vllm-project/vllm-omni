#!/usr/bin/env python3
"""把 seed-tts-eval 的短句 meta.lst 拼成「长文本」meta.lst，用于长上下文 TTS 压测。

原理：seed-tts loader 读 <root>/<locale>/meta.lst，每行格式
    utt_id|ref_text|prompt_wav_rel|target_text
长文本 = 把连续 N 条 target_text 拼成一条；ref_text / prompt_wav_rel 沿用第一条
（同一参考音做零样本克隆，只是合成长文本）。prompt-wavs 目录直接软链复用，不复制。

用法（在压测机上，数据集在 /root/datasets/seed-tts-eval）：

    python3 make_longtext_seedtts.py \
        --root /root/datasets/seed-tts-eval \
        --locale zh --group 20 --out-locale zh_long

生成 /root/datasets/seed-tts-eval/zh_long/meta.lst 与软链 prompt-wavs。
压测时换 locale 即可（数据集名仍是 seed-tts，无需改 bench 脚本）：

    DATASET=/root/datasets/seed-tts-eval LOCALE=zh_long \\
        TOPOS=2p2d CONCURRENCIES="8 32 64" NUM_PROMPTS=50 CHECK_AUDIO=0 \\
        bash run_all_bench.sh
"""
import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="seed-tts-eval 根目录（含 <locale>/meta.lst）")
    p.add_argument("--locale", default="zh", help="源 locale（默认 zh）")
    p.add_argument("--out-locale", default=None, help="输出 locale 目录名，默认 <locale>_long")
    p.add_argument("--group", type=int, default=20, help="最少拼几条源短句才收尾（--min-chars 优先）")
    p.add_argument("--sep", default=" ", help="拼接分隔符（en 用空格；zh 可用空格或 '。'）")
    p.add_argument("--max-chars", type=int, default=0, help="单条长文本最大字符数，超过则先停手（0=不限）")
    p.add_argument("--min-chars", type=int, default=0, help="单条长文本最小字符数，达标才收尾（如 1024）")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    locale = args.locale
    out_locale = args.out_locale or f"{locale}_long"
    src_meta = root / locale / "meta.lst"
    if not src_meta.is_file():
        raise SystemExit(f"找不到源 meta.lst: {src_meta}")

    rows: list[list[str]] = []
    for line in src_meta.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        rows.append(parts)

    if not rows:
        raise SystemExit(f"源 meta.lst 没有任何有效行: {src_meta}")

    # 分组拼接（以字符预算为主：--min-chars 达标即收尾，--max-chars 前先停手，
    # 避免把一句话从中间切断；--group 仅作「最少条数」兜底）。
    out_groups: list[list[list[str]]] = []
    buf: list[list[str]] = []
    for parts in rows:
        # 若加入这一条会超过 max_chars，先把当前 buf 收尾（句子边界不被切断）
        if buf and args.max_chars:
            proj = len(args.sep.join(r[3] for r in buf)) + len(args.sep) + len(parts[3])
            if proj > args.max_chars:
                out_groups.append(buf)
                buf = []
        buf.append(parts)
        text_len = sum(len(r[3]) for r in buf)
        reached_min = (args.min_chars == 0) or (text_len >= args.min_chars)
        reached_group = len(buf) >= args.group
        if reached_group and reached_min:
            out_groups.append(buf)
            buf = []
    if buf:
        out_groups.append(buf)

    # 写出
    out_dir = root / out_locale
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta = out_dir / "meta.lst"
    with out_meta.open("w", encoding="utf-8") as f:
        for gi, buf in enumerate(out_groups):
            utt_id = buf[0][0]
            ref_text = buf[0][1]
            wav_rel = buf[0][2]
            target = args.sep.join(r[3] for r in buf)
            # max_chars 作为安全上限（正常已在上面停手，不会触发硬切）
            if args.max_chars and len(target) > args.max_chars:
                target = target[: args.max_chars]
            f.write(f"{utt_id}_L{gi:04d}|{ref_text}|{wav_rel}|{target}\n")

    # 复用 prompt-wavs（软链，避免复制大文件）
    src_wavs = root / locale / "prompt-wavs"
    out_wavs = out_dir / "prompt-wavs"
    if src_wavs.is_dir():
        if out_wavs.is_symlink() or out_wavs.exists():
            print(f"[skip] 已存在: {out_wavs}")
        else:
            rel = os.path.relpath(src_wavs, out_dir)
            out_wavs.symlink_to(rel, target_is_directory=True)
            print(f"[link] {out_wavs} -> {rel}")
    else:
        print(f"[warn] 源 prompt-wavs 不存在: {src_wavs}（请确认参考音路径）")

    print(f"源行数: {len(rows)} -> 长文本行数: {len(out_groups)} （每组约 {args.group} 条）")
    print(f"输出 meta: {out_meta}")
    print(f"压测: DATASET={root} LOCALE={out_locale} bash bench_pd.sh <TOPO> <C> <N>")


if __name__ == "__main__":
    main()
