"""Metadata merge helpers for SoulX-Singer preprocess."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass
class SegmentMetadata:
    item_name: str
    wav_fn: str
    language: str
    start_time_ms: int
    end_time_ms: int
    note_text: list[str]
    note_dur: list[float]
    note_pitch: list[int]
    note_type: list[int]
    origin_wav_fn: str | None = None
    wav: np.ndarray | None = None

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)


def _merge_group(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[SegmentMetadata | dict[str, Any]],
    output_dir: Path | None,
    *,
    end_extension_ms: int = 0,
) -> SegmentMetadata:
    if not segments:
        raise ValueError("segments must not be empty")

    words: list[str] = []
    durs: list[float] = []
    pitches: list[int] = []
    types: list[int] = []

    for i, seg in enumerate(segments):
        if i > 0:
            prev_seg = segments[i - 1]
            gap_ms = seg.get("start_time_ms", 0) - prev_seg.get("end_time_ms", 0)
            if gap_ms > 0:
                words.append("<SP>")
                durs.append(gap_ms / 1000.0)
                pitches.append(0)
                types.append(1)

        words.extend(seg.get("note_text", []))
        durs.extend(seg.get("note_dur", []))
        pitches.extend(seg.get("note_pitch", []))
        types.extend(seg.get("note_type", []))

    if end_extension_ms > 0:
        words.append("<SP>")
        durs.append(end_extension_ms / 1000.0)
        pitches.append(0)
        types.append(1)

    merged_words, merged_durs, merged_pitches, merged_types = [], [], [], []
    for word, dur, pitch, note_type in zip(words, durs, pitches, types):
        if merged_words and word == "<SP>" and merged_words[-1] == "<SP>":
            merged_durs[-1] += dur
        else:
            merged_words.append(word)
            merged_durs.append(dur)
            merged_pitches.append(pitch)
            merged_types.append(note_type)

    languages = [s.get("language", "Mandarin") for s in segments if s.get("language")]
    language = max(languages, key=languages.count) if languages else "Mandarin"

    start_ms = int(segments[0].get("start_time_ms", 0))
    end_ms = int(segments[-1].get("end_time_ms", 0)) + end_extension_ms
    start_sample = int(start_ms * sample_rate // 1000)
    end_sample = int(end_ms * sample_rate // 1000)

    first_item_name = segments[0].get("item_name", "segment")
    song_prefix = "_".join(str(first_item_name).split("_")[:-1])
    item_name = f"{song_prefix}_{start_ms}_{end_ms}"

    wav_fn = ""
    segment_audio = audio[start_sample:end_sample]
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        wav_path = output_dir / f"{item_name}.wav"
        sf.write(wav_path, segment_audio, sample_rate)
        wav_fn = str(wav_path)

    return SegmentMetadata(
        item_name=item_name,
        wav_fn=wav_fn,
        language=language,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        note_text=merged_words,
        note_dur=merged_durs,
        note_pitch=merged_pitches,
        note_type=merged_types,
        origin_wav_fn=segments[0].get("origin_wav_fn", ""),
        wav=segment_audio if output_dir is None else None,
    )


def convert_metadata(item: SegmentMetadata, f0: np.ndarray | None = None) -> dict[str, Any]:
    """Convert merged segment metadata into SoulX JSON metadata format.

    Matches upstream ``convert_metadata``: F0 always comes from ``*_f0.npy`` written
    by ``extract_f0_file`` on the merged segment wav, not from an in-memory array.
    """
    from vllm_omni.diffusion.models.soulx_singer.preprocess.g2p import g2p_transform

    if item.wav_fn:
        f0_path = str(Path(item.wav_fn).with_suffix("")) + "_f0.npy"
        f0_arr = np.load(f0_path)
    elif f0 is not None:
        f0_arr = f0
    else:
        raise ValueError("convert_metadata requires item.wav_fn with sidecar f0 or an explicit f0 array")

    return {
        "index": item.item_name,
        "language": item.language,
        "time": [item.start_time_ms, item.end_time_ms],
        "duration": " ".join(f"{d:.2f}" for d in item.note_dur),
        "text": " ".join(item.note_text),
        "phoneme": " ".join(g2p_transform(item.note_text, item.language)),
        "note_pitch": " ".join(map(str, item.note_pitch)),
        "note_type": " ".join(map(str, item.note_type)),
        "f0": " ".join(f"{x:.1f}" for x in f0_arr),
    }
