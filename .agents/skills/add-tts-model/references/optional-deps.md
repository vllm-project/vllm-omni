# Optional Dependency Handling

Models that rely on `torchaudio`, `torchcodec`, `soundfile`, or other optional
packages must handle the missing-package case at import time, not at call
time. Failing to do this causes cryptic errors only on environments without
the optional package — after the model is already deployed.

## Pattern (used in MOSS-TTS-Nano)

```python
def _patch_torchaudio_load() -> None:
    """Fallback torchaudio.load/save to soundfile if torchcodec is unavailable."""
    try:
        import torchcodec  # noqa: F401
        return  # torchcodec present, torchaudio works as-is
    except ImportError:
        pass

    import soundfile as sf
    import torchaudio

    def _sf_load(path, **kwargs):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data).T, sr

    torchaudio.load = _sf_load
    # patch .save similarly if needed
```

## Rules

- Mirror the full signature of the replaced function. `torchaudio.load`
  accepts `frame_offset`, `num_frames`, `normalize`, `channels_first`,
  `format` — missing any of them causes `TypeError` from calling code.
- Catch `except Exception`, not just `ImportError`. `import torchaudio`
  itself can fail with non-`ImportError` errors on broken installs. Log the
  exception type and message (`logger.warning("torchaudio probe failed: %s: %s",
  type(exc).__name__, exc)`) before falling back, so unrelated errors are not
  silently swallowed.
- Call the patch function at the top of `load_weights()` before loading any
  audio assets. Do not call it at module import time.

## Reference implementation

Any in-tree model that patches `torchaudio.load` in its `load_weights()` —
e.g. MOSS-TTS-Nano's `modeling_moss_tts_nano.py` once that integration
lands.
