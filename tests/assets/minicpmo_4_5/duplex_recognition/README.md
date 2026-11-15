# MiniCPM-o long-video recognition regression

This is the frozen **180.000-second accuracy fixture** used while investigating
[#8637](https://github.com/vllm-project/vllm-omni/issues/8637). It is not the
original 180.032-second residual-input timeout reproduction.

`inputs.tar.xz` contains 16 kHz mono PCM16 audio and 180 JPEG frames at 1 fps.
These are the exact decoded bytes used by the Transformers/Omni comparison,
not a fresh decode of the original video. Keeping them compressed in the test
assets avoids runtime network, ffmpeg, fonts or speech-synthesis dependencies.
The archive and each member have SHA256 digests in `manifest.json`.

The synthetic source contains 12 moving colored/numbered shapes, 15 seconds per
scene, and 28 Mandarin questions synthesized with espeak-ng. The source
[generator and timeline](https://github.com/amy-why-3459/vllm-omni/tree/47aa8f8141716a9456739e8653c75ac249afcfff)
are the artifacts linked from the issue. The manifest preserves the scene
labels and exact prompt. The reference voice is the checkpoint's
`assets/HT_ref_audio.wav`; it is not bundled. Its digest pins the same
96,256-sample conditioning waveform used in validation (including its tail).

The CUDA full-model test uses the existing three-stage `omni_server` fixture
and default `minicpmo_4_5.yaml`. It sends 200 ms chunks at realtime pace, window
off, temperature zero, one frame per second, followed by 8 seconds of silence
holding the last frame. The existing demo client handles the real WebSocket
session, playback acknowledgements, input commit and a bounded 90-second drain.
Only media decoding is replaced with the pinned bytes; inference is not mocked.

Pass criteria:

- All 12 color/shape/number descriptions appear together in a sentence. This
  allows delayed answers and is not a score on all 28 questions, direction or
  interruption compliance. Arabic and Chinese numbers are accepted without
  matching number 1 against 11 or 12.
- No client/server errors, a committed input and closed session, output audio,
  and matching response IDs for every created and completed response. Correct
  text does not excuse a missing final response completion.

Run from the repository root with the normal model cache configured:

```bash
pytest -sv tests/e2e/online_serving/test_minicpmo_4_5_duplex_recognition.py \
  --run-level full_model -m 'full_model and cuda'
```

The CUDA nightly MiniCPM-o duplex job selects the test on H100/B200 and uploads
`tests/e2e/online_serving/results/minicpmo_duplex_recognition/**/*`, including
transcript, events, recognition report, input timeline and generated audio.
The model comparison was measured on L20X; H100/B200 execution of this new test
has not yet been measured. This is a strict regression threshold, not a claim
of cross-hardware token equality.

`recorded_transcripts.json` contains the actual before/intermediate/after and
Transformers outputs used to test the scoring contract without a GPU. The
unfixed default covers 11/12, the repaired output and Transformers 4.51.3 cover
12/12. GPU runs used `c28741e0f75618ea700751af7038dd9a10711522` plus the repair;
the final repaired transcript repeated across three sessions. Replaying these
records tests the assertion, not the inference implementation. The earlier
Transformers 5.14 full-model reference with a zero position buffer is excluded.
