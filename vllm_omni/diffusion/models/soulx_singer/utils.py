import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_MODEL_WEIGHTS_REPO = "Soul-AILab/SoulX-Singer"
_PREPROCESS_WEIGHTS_REPO = "Soul-AILab/SoulX-Singer-Preprocess"


def _patch_torchaudio_load() -> None:
    """Borrowed from MOSS-TTS-Nano. Patch torchaudio.load to use soundfile
    if torchcodec is unavailable.
    """
    try:
        import torchaudio

        torchaudio  # noqa
        import torchcodec  # noqa: F401

        return
    except Exception:
        pass

    import soundfile as sf

    def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        waveform = torch.from_numpy(data)
        if channels_first:
            waveform = waveform.T
        return waveform, sr

    def _soundfile_save(path, src, sample_rate, channels_first=True, **kwargs):
        wav = src.detach().cpu().float().numpy()
        if channels_first and wav.ndim == 2:
            wav = wav.T
        sf.write(str(path), wav, sample_rate)

    try:
        import torchaudio

        torchaudio.load = _soundfile_load
        torchaudio.save = _soundfile_save
        logger.info("Patched torchaudio.load/save to use soundfile (torchcodec unavailable)")
    except Exception as e:
        logger.warning("Could not patch torchaudio: %s", e)


def load_config(config_path: str | Path) -> DictConfig:
    """
    Load a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(str(config_path))

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(str(config.get("base_config")))
        config = OmegaConf.merge(base_config, config)

    return config


SoulXKind = Literal["svs", "svc"]

_SVS_ARCHITECTURE = "SoulXSingerPipeline"
_SVC_ARCHITECTURE = "SoulXSingerSVCPipeline"
_ARCHITECTURE_TO_KIND: dict[str, SoulXKind] = {
    _SVS_ARCHITECTURE: "svs",
    _SVC_ARCHITECTURE: "svc",
}


def load_model_config_json(model_dir: str | Path) -> dict[str, Any]:
    """Load ``config.json`` from a SoulX-Singer model view directory."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"SoulX-Singer config.json not found at {cfg_path}. "
            "Create one with architectures ['SoulXSingerPipeline'] (SVS) or "
            "['SoulXSingerSVCPipeline'] (SVC); see examples/offline_inference/text_to_speech/README.md."
        )
    with cfg_path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_soulx_kind(model_dir: str | Path) -> SoulXKind:
    """Resolve SVS vs SVC from ``config.json`` ``architectures[0]``."""
    cfg = load_model_config_json(model_dir)
    archs = cfg.get("architectures") or []
    if not archs:
        raise ValueError(
            f"SoulX-Singer config.json at {Path(model_dir) / 'config.json'} is missing 'architectures'. "
            "Set architectures to ['SoulXSingerPipeline'] for SVS or ['SoulXSingerSVCPipeline'] for SVC."
        )
    arch = archs[0]
    kind = _ARCHITECTURE_TO_KIND.get(arch)
    if kind is None:
        raise ValueError(
            f"Unknown SoulX-Singer architecture {arch!r} in config.json. Expected one of {list(_ARCHITECTURE_TO_KIND)}."
        )
    return kind


def validate_soulx_extra_args(kind: SoulXKind, extra_args: dict[str, Any] | None) -> dict[str, Any]:
    """Validate request ``extra_args`` against the model mode from config.json."""
    from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
        SOULX_PRECOMPUTED_KEYS_BY_KIND,
        has_precomputed,
    )

    extra_args = dict(extra_args or {})
    declared = extra_args.pop("kind", None)
    if declared is not None and declared != kind:
        logger.warning(
            "Ignoring extra_args kind=%r; model config.json declares %r.",
            declared,
            kind,
        )

    other_kind: SoulXKind = "svc" if kind == "svs" else "svs"
    if has_precomputed(extra_args, other_kind):
        raise ValueError(
            f"SoulX-Singer config.json declares {kind!r}, but request supplies "
            f"{other_kind} precomputed paths {list(SOULX_PRECOMPUTED_KEYS_BY_KIND[other_kind])}."
        )

    if kind == "svc":
        for key in ("language", "control"):
            if extra_args.get(key) is not None:
                logger.warning(
                    "Ignoring SVS-only extra_arg %r=%r on an SVC model (config.json).",
                    key,
                    extra_args.pop(key),
                )
        for key in SOULX_PRECOMPUTED_KEYS_BY_KIND["svs"]:
            if extra_args.get(key):
                raise ValueError(f"SoulX-Singer SVC model (config.json) does not accept SVS precomputed key {key!r}.")
    else:
        for key in SOULX_PRECOMPUTED_KEYS_BY_KIND["svc"]:
            if extra_args.get(key):
                raise ValueError(f"SoulX-Singer SVS model (config.json) does not accept SVC precomputed key {key!r}.")

    return extra_args


# ---------------- utils for model weights ----------------


def resolve_preprocess_weights_root(od_config: OmniDiffusionConfig) -> Path:
    """Locate preprocess weights on disk or download from Hugging Face."""

    # default local dir
    if Path(od_config.model).parent.is_dir():
        local_dir = Path(od_config.model).parent / "SoulX-Singer-Preprocess"
    else:
        local_dir = Path(__file__).parents[5] / "pretrained" / "SoulX-Singer-Preprocess"

    local_dir = local_dir.expanduser()
    if local_dir.is_dir():
        logger.info("Using SoulX preprocess weights from %s", local_dir)
        return local_dir

    logger.info(
        "SoulX preprocess weights not found locally; downloading %s",
        _PREPROCESS_WEIGHTS_REPO,
    )
    downloaded = snapshot_download(
        _PREPROCESS_WEIGHTS_REPO,
        allow_patterns=["*"],
        local_dir=local_dir,
    )
    return Path(downloaded)


def preprocess_weight_paths(weights_root: Path) -> dict[str, str]:
    """Map upstream relative paths to absolute paths under ``weights_root``."""
    root = weights_root.resolve()
    return {
        "rmvpe": str(root / "rmvpe/rmvpe.pt"),
        "sep_ckpt": str(root / "mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt"),
        "sep_config": str(root / "mel-band-roformer-karaoke/config_karaoke_becruily.yaml"),
        "asr_zh": str(root / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
        "asr_en": str(root / "parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"),
        "rosvot": str(root / "rosvot/rosvot/model.pt"),
    }


def resolve_phoneset_path(model_dir: str) -> str:
    """Resolve phoneme vocabulary; must be present in model directory."""
    candidates = (
        Path(model_dir) / "phoneme" / "phone_set.json",
        Path(model_dir) / "phone_set.json",
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "SoulX-Singer phoneset not found. Expected one of: "
        f"{[str(p) for p in candidates]}. "
        "Copy phone_set.json into the model directory. "
        "See `examples/offline_inference/text_to_speech/README.md` for instructions."
    )


# ---------------- utils for data processing ----------------


def load_wav(wav_path: str, sample_rate: int) -> torch.Tensor:
    """Load wav file and resample to target sample rate.

    Args:
        wav_path (str): Path to wav file.
        sample_rate (int): Target sample rate.

    Returns:
        torch.Tensor: Waveform tensor with shape (1, T).
    """
    _patch_torchaudio_load()
    waveform, sr = torchaudio.load(wav_path)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform


def f0_to_coarse(f0, f0_bin=361, f0_min=32.7031956625, f0_shift=0):
    """
    Convert continuous F0 values to discrete F0 bins (SIL and C1 - B6, 361 bins).
    args:
        f0: continuous F0 values
        f0_bin: number of F0 bins
        f0_min: minimum F0 value
        f0_shift: shift value for F0 bins
    returns:
        f0_coarse: discrete F0 bins
    """
    is_torch = isinstance(f0, torch.Tensor)
    uv_mask = f0 <= 0

    if is_torch:
        f0_min_tensor = f0.new_tensor(f0_min)
        f0_safe = torch.maximum(f0, f0_min_tensor)
        f0_cents = 1200 * torch.log2(f0_safe / f0_min_tensor)
    else:
        f0_safe = np.maximum(f0, f0_min)
        f0_cents = 1200 * np.log2(f0_safe / f0_min)

    f0_coarse = (f0_cents / 20) + 1

    if is_torch:
        f0_coarse = torch.round(f0_coarse).long()
        f0_coarse = torch.clamp(f0_coarse, min=1, max=f0_bin - 1)
    else:
        f0_coarse = np.rint(f0_coarse).astype(int)
        f0_coarse = np.clip(f0_coarse, 1, f0_bin - 1)

    f0_coarse[uv_mask] = 0

    if f0_shift != 0:
        if is_torch:
            voiced = f0_coarse > 0
            if voiced.any():
                shifted = f0_coarse[voiced] + f0_shift
                f0_coarse[voiced] = torch.clamp(shifted, 1, f0_bin - 1)
        else:
            voiced = f0_coarse > 0
            if np.any(voiced):
                shifted = f0_coarse[voiced] + f0_shift
                f0_coarse[voiced] = np.clip(shifted, 1, f0_bin - 1)

    return f0_coarse


def resolve_pitch_shift(
    *,
    auto_shift: bool,
    manual_shift: int,
    prompt_f0: torch.Tensor | None = None,
    target_f0: torch.Tensor | None = None,
    prompt_note_pitch: torch.Tensor | None = None,
    target_note_pitch: torch.Tensor | None = None,
) -> int:
    """Resolve semitone shift for auto_shift; ``f0_to_coarse(..., f0_shift=shift * 5)`` consumes it."""
    if not auto_shift or manual_shift != 0:
        return int(manual_shift)

    if prompt_note_pitch is not None and target_note_pitch is not None:
        target_pitched = target_note_pitch[target_note_pitch >= 1]
        prompt_pitched = prompt_note_pitch[prompt_note_pitch >= 1]
        if target_pitched.numel() > 0 and prompt_pitched.numel() > 0:
            shift = torch.round(prompt_pitched.median() - target_pitched.median())
            if torch.isfinite(shift):
                return int(shift.item())

    if prompt_f0 is not None and target_f0 is not None:
        target_voiced = target_f0[target_f0 > 0]
        prompt_voiced = prompt_f0[prompt_f0 > 0]
        if target_voiced.numel() > 0 and prompt_voiced.numel() > 0:
            shift = torch.round(torch.log2(prompt_voiced.median() / target_voiced.median()) * 1200 / 100)
            if torch.isfinite(shift):
                return int(shift.item())

    return 0


class MetadataProcessor:
    """Data processor for SoulX-Singer"""

    def __init__(
        self,
        hop_size: int,
        sample_rate: int,
        phoneset_path: str,
        device: str = "cuda",
    ):
        """Initialize data processor.

        Args:
            hop_size (int): Hop size in samples.
            sample_rate (int): Sample rate in Hz.
            phoneset_path (str): Path to phoneme set JSON file.
            device (str): Device to use for tensor operations.
        """
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.device = device
        self.load_phoneme_id_map(phoneset_path)

    def load_phoneme_id_map(self, phoneset_path: str):
        with open(phoneset_path, encoding="utf-8") as f:
            phoneset = json.load(f)
        self.phone2idx = {ph: idx for idx, ph in enumerate(phoneset)}

    def merge_phoneme(self, meta):
        merged_items = []

        duration = [float(x) for x in meta["duration"].split()]
        phoneme = [str(x).replace("<AP>", "<SP>") for x in meta["phoneme"].split()]
        note_pitch = [int(x) for x in meta["note_pitch"].split()]
        note_type = [int(x) if phoneme[i] != "<SP>" else 1 for i, x in enumerate(meta["note_type"].split())]

        for i in range(len(phoneme)):
            if (
                i > 0
                and phoneme[i] == phoneme[i - 1] == "<SP>"
                and note_type[i] == note_type[i - 1]
                and note_pitch[i] == note_pitch[i - 1]
            ):
                merged_items[-1][1] += duration[i]
            else:
                merged_items.append([phoneme[i], duration[i], note_pitch[i], note_type[i]])

        meta["phoneme"] = [x[0] for x in merged_items]
        meta["duration"] = [x[1] for x in merged_items]
        meta["note_pitch"] = [x[2] for x in merged_items]
        meta["note_type"] = [x[3] for x in merged_items]

        return meta

    def preprocess(
        self,
        note_duration: list[float],
        phonemes: list[str],
        note_pitch: list[int],
        note_type: list[int],
    ):
        """
        Insert <BOW> and <EOW> for each note.
        Get aligned indices for each frame.

        Args:
            note_duration: Duration of each note in seconds
            phonemes: Phoneme sequence for each note
            note_pitch: Pitch value for each note
            note_type: Type value for each note

        """
        sample_rate = self.sample_rate
        hop_size = self.hop_size
        duration = sum(note_duration) * sample_rate / hop_size
        mel2note = torch.zeros(int(duration), dtype=torch.long, device=self.device)

        ph_locations = []  # idx at mel scale and length
        new_phonemes = []
        dur_sum = 0

        note2origin = []

        for ph_idx in range(len(phonemes)):
            dur = int(np.round(dur_sum * sample_rate / hop_size))
            dur = min(dur, len(mel2note) - 1)
            new_phonemes.append("<BOW>")
            note2origin.append(ph_idx)
            if phonemes[ph_idx][:3] == "en_":
                en_phs = ["en_" + x for x in phonemes[ph_idx][3:].split("-")] + [
                    "<SEP>"
                ]  # <sep> between en words in one note
                ph_locations.append([dur, max(1, len(en_phs))])
                new_phonemes.extend(en_phs)
                note2origin.extend([ph_idx] * len(en_phs))
            else:
                ph_locations.append([dur, 1])
                new_phonemes.append(phonemes[ph_idx])
                note2origin.append(ph_idx)
            new_phonemes.append("<EOW>")
            note2origin.append(ph_idx)
            dur_sum += note_duration[ph_idx]

        ph_idx = 1
        for idx, (i, j) in enumerate(ph_locations):
            next_phoneme_start = ph_locations[idx + 1][0] if idx < len(ph_locations) - 1 else len(mel2note)
            if i >= len(mel2note) or i + j > len(mel2note):
                break
            if i < len(mel2note) and mel2note[i] > 0:
                logger.warning(f"warning: overlap of {idx}: {mel2note[i]}")
                while i < len(mel2note) and mel2note[i] > 0:
                    i += 1
            mel2note[i] = ph_idx
            k = i + 1
            while k + j < next_phoneme_start:
                mel2note[k : k + j] = torch.arange(ph_idx, ph_idx + j, device=self.device) + 1
                k += j
            mel2note[next_phoneme_start - 1] = ph_idx + j + 1
            ph_idx += j + 2  # <BOW> + ph repeats + <EOW>

        new_phonemes = ["<PAD>"] + new_phonemes
        new_note_pitch = [0] + [note_pitch[k] for k in note2origin]
        new_note_type = [1] + [note_type[k] for k in note2origin]

        return {
            "phoneme": torch.tensor([self.phone2idx[x] for x in new_phonemes], device=self.device).unsqueeze(0),
            "note_pitch": torch.tensor(new_note_pitch, device=self.device).unsqueeze(0),
            "note_type": torch.tensor(new_note_type, device=self.device).unsqueeze(0),
            "mel2note": mel2note.unsqueeze(0),
        }

    def process(self, meta: dict, wav_path: str | None = None) -> dict[str, torch.Tensor | None]:
        meta = self.merge_phoneme(meta)

        item = self.preprocess(
            meta["duration"],
            meta["phoneme"],
            meta["note_pitch"],
            meta["note_type"],
        )

        f0 = [float(x) for x in meta.get("f0", "").split()]
        min_frame = min(item["mel2note"].shape[1], len(f0)) if len(f0) > 0 else item["mel2note"].shape[1]
        item["f0"] = torch.tensor(f0, device=self.device)[:min_frame].unsqueeze(0).float() if len(f0) > 0 else None
        item["mel2note"] = item["mel2note"][:, :min_frame]

        if wav_path is not None:
            waveform = load_wav(wav_path, self.sample_rate)
            item["wav"] = waveform.to(self.device)[:, : min_frame * self.hop_size]

        return item


def build_vocal_segments(
    f0,
    *,
    f0_rate: int = 50,
    ignore_silent_frames_thresh: int = 5,
    min_duration_sec_per_segment: float = 5.0,
    max_duration_sec_per_segment: float = 30.0,
    num_overlaps: int = 1,
    ignore_silent_frames: bool = True,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Build vocal segments from an F0 contour for chunked SVC inference."""
    if isinstance(f0, torch.Tensor):
        f0_np = f0.detach().float().cpu().numpy()
    else:
        f0_np = np.asarray(f0, dtype=np.float32)
    f0_np = np.squeeze(f0_np)

    total_frames = int(f0_np.shape[0])
    if total_frames == 0:
        return [], []

    min_frames = max(1, int(round(min_duration_sec_per_segment * f0_rate)))
    max_frames = max(1, int(round(max_duration_sec_per_segment * f0_rate)))

    split_points = [0]

    def append_split_point(point: int):
        point = int(max(0, min(point, total_frames)))
        while point - split_points[-1] > max_frames:
            split_points.append(split_points[-1] + max_frames)
        if point > split_points[-1]:
            split_points.append(point)

    idx = 0
    while idx < total_frames:
        if f0_np[idx] == 0:
            run_start = idx
            while idx < total_frames and f0_np[idx] == 0:
                idx += 1
            run_end = idx
            if (run_end - run_start) >= ignore_silent_frames_thresh:
                split_point = max(run_end - 5, (run_start + run_end) // 2)
                append_split_point(split_point)
        else:
            idx += 1
    append_split_point(total_frames)

    segments: list[tuple[float, float]] = []
    overlap_segments: list[tuple[float, float]] = []

    def append_segment(start_idx: int, end_idx: int, overlaps: int = num_overlaps):
        segments.append((split_points[start_idx] / f0_rate, split_points[end_idx] / f0_rate))
        overlap_start_idx = start_idx
        if start_idx > 0 and (split_points[end_idx] - split_points[start_idx - overlaps]) <= max_frames:
            overlap_start_idx = start_idx - overlaps
        overlap_segments.append((split_points[overlap_start_idx] / f0_rate, split_points[end_idx] / f0_rate))

    segment_start, segment_end = 0, 1
    while segment_start < len(split_points) - 1:
        while (
            segment_end < len(split_points) and (split_points[segment_end] - split_points[segment_start]) < min_frames
        ):
            segment_end += 1

        if segment_end >= len(split_points):
            append_segment(segment_start, len(split_points) - 1, overlaps=num_overlaps)
            break
        append_segment(segment_start, segment_end, overlaps=num_overlaps)
        segment_start = segment_end
        segment_end = segment_start + 1

    if ignore_silent_frames:
        filtered_idx = []
        for i, seg in enumerate(overlap_segments):
            start_frame = int(seg[0] * f0_rate)
            end_frame = int(seg[1] * f0_rate)
            seg_frames = end_frame - start_frame
            voice_frames = np.sum(f0_np[start_frame:end_frame] > 0)
            if voice_frames / seg_frames > 0.05 and voice_frames >= 10:
                filtered_idx.append(i)

        overlap_segments = [overlap_segments[i] for i in filtered_idx]
        segments = [segments[i] for i in filtered_idx]

    return overlap_segments, segments
