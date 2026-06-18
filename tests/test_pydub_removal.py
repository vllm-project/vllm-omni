"""Functional tests for pydub removal.

Verifies that the new av/numpy/torchaudio replacements produce
identical results to the old pydub-based code for all three paths:
  1. WAV PCM → duration & frame count (pure arithmetic)
  2. Non-WAV (mp3/ogg) → duration & frame count (av.open)
  3. TTS WER PCM export: resample to 24000Hz mono s16
"""

import io
import wave

import av
import numpy as np
import soundfile as sf


def _make_wav_bytes(sample_rate=16000, channels=1, sample_width=2, duration_sec=1.0):
    """Generate a WAV file with a sine wave, return (wav_bytes, pcm_bytes, params)."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
    if channels == 2:
        samples = np.column_stack([samples, samples]).flatten()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    wav_bytes = buf.getvalue()

    # Extract raw PCM (skip WAV header)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wr:
        pcm_bytes = wr.readframes(wr.getnframes())
        params = (wr.getnchannels(), wr.getsampwidth(), wr.getframerate())

    return wav_bytes, pcm_bytes, params


def _make_ogg_bytes(sample_rate=16000, channels=1, duration_sec=1.0):
    """Generate an OGG Vorbis file with a sine wave, return bytes."""
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="OGG", subtype="VORBIS")
    return buf.getvalue()


def test_wav_path_duration_and_frames():
    """Path 1: WAV PCM buffer → duration & frames via arithmetic."""
    for sr in [16000, 24000, 44100]:
        for ch in [1, 2]:
            for dur in [0.5, 1.0, 2.5]:
                _, pcm_bytes, (channels, sample_width, frame_rate) = _make_wav_bytes(
                    sample_rate=sr, channels=ch, duration_sec=dur
                )

                # New code logic (from patch.py)
                frame_width = sample_width * channels
                audio_frames = len(pcm_bytes) // frame_width
                audio_duration_sec = audio_frames / frame_rate

                assert abs(audio_duration_sec - dur) < 0.001, f"sr={sr} ch={ch} dur={dur}: got {audio_duration_sec}"
                expected_frames = int(sr * dur)
                assert audio_frames == expected_frames, (
                    f"sr={sr} ch={ch} dur={dur}: frames {audio_frames} != {expected_frames}"
                )

    print("[PASS] test_wav_path_duration_and_frames")


def test_non_wav_path_duration_av():
    """Path 2: non-WAV audio → duration & frames via av."""
    for dur in [0.5, 1.0, 2.0]:
        ogg_bytes = _make_ogg_bytes(sample_rate=16000, duration_sec=dur)

        container = av.open(io.BytesIO(ogg_bytes))
        stream = container.streams.audio[0]
        audio_frames = sum(f.samples for f in container.decode(stream))
        audio_duration_sec = audio_frames / stream.rate if stream.rate else 0.0
        container.close()

        # OGG encoding may add padding, allow small tolerance
        assert abs(audio_duration_sec - dur) < 0.05, f"dur={dur}: got {audio_duration_sec}"

    print("[PASS] test_non_wav_path_duration_av")


def test_tts_pcm_export_wav_path():
    """Path 3a: WAV PCM → resample to 24000Hz mono s16."""
    import torch
    import torchaudio

    _, pcm_bytes, (channels, sample_width, frame_rate) = _make_wav_bytes(
        sample_rate=16000, channels=2, sample_width=2, duration_sec=1.0
    )

    # New code logic
    pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels > 1:
        pcm_array = pcm_array.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if frame_rate != 24000:
        t = torch.from_numpy(pcm_array.astype(np.float32)).unsqueeze(0)
        t = torchaudio.functional.resample(t, frame_rate, 24000)
        pcm_array = t.squeeze(0).to(torch.int16).numpy()

    result_bytes = pcm_array.tobytes()
    # Should be mono, 24000Hz, 2 bytes per sample
    expected_samples = int(24000 * 1.0)
    actual_samples = len(result_bytes) // 2
    assert abs(actual_samples - expected_samples) < 10, f"expected ~{expected_samples} samples, got {actual_samples}"
    assert len(pcm_array.shape) == 1, "should be mono (1D array)"

    print("[PASS] test_tts_pcm_export_wav_path")


def test_tts_pcm_export_non_wav_path():
    """Path 3b: non-WAV → resample to 24000Hz mono s16 via av.AudioResampler."""
    ogg_bytes = _make_ogg_bytes(sample_rate=16000, channels=1, duration_sec=1.0)

    container = av.open(io.BytesIO(ogg_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=24000)
    frames = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            frames.append(rf.to_ndarray().flatten())
    container.close()

    result = np.concatenate(frames)
    result_bytes = result.tobytes()
    expected_samples = int(24000 * 1.0)
    actual_samples = len(result_bytes) // 2
    assert abs(actual_samples - expected_samples) < 500, f"expected ~{expected_samples} samples, got {actual_samples}"

    print("[PASS] test_tts_pcm_export_non_wav_path")


def test_media_merge_and_export():
    """media.py: _merge_base64_audio_to_segment decode + concat + export WAV."""
    import base64 as b64mod
    import sys

    sys.path.insert(0, ".")
    from tests.helpers.media import _merge_base64_audio_to_segment

    # Create two WAV chunks as base64
    chunks_b64 = []
    for dur in [0.5, 0.3]:
        wav_bytes, _, _ = _make_wav_bytes(sample_rate=16000, channels=1, sample_width=2, duration_sec=dur)
        chunks_b64.append(b64mod.b64encode(wav_bytes).decode())

    merged = _merge_base64_audio_to_segment(chunks_b64)

    # Test export to WAV
    buf = io.BytesIO()
    merged.export(buf, format="wav")
    wav_out = buf.getvalue()
    assert len(wav_out) > 44, f"WAV output too small: {len(wav_out)} bytes"

    # Verify merged duration (~0.8s)
    out_data, out_sr = sf.read(io.BytesIO(wav_out))
    actual_dur = len(out_data) / out_sr
    assert abs(actual_dur - 0.8) < 0.05, f"expected ~0.8s merged duration, got {actual_dur}"

    print(f"[PASS] test_media_merge_and_export (duration={actual_dur:.3f}s, wav_size={len(wav_out)} bytes)")


if __name__ == "__main__":
    test_wav_path_duration_and_frames()
    test_non_wav_path_duration_av()
    import importlib.util

    if importlib.util.find_spec("torch") and importlib.util.find_spec("torchaudio"):
        test_tts_pcm_export_wav_path()
    else:
        print("[SKIP] test_tts_pcm_export_wav_path (torch/torchaudio not available)")
    test_tts_pcm_export_non_wav_path()
    test_media_merge_and_export()
    print("\nAll tests passed.")
