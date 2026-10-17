# Duplex turn detection: `server_vad` vs `semantic_vad`

`bench_vad_latency.py` measures the two engine-side turn detectors in isolation,
without model serving or a GPU, exactly as the duplex engine runs them: one shared
ONNX backend per model and one detector per session, fed 16 kHz chunks.

- `server_vad`: Silero v6.2, a fixed silence ends the turn.
- `semantic_vad`: Silero finds a pause, then Smart Turn v3.2 decides whether it ends the turn.

## What it reports

1. Per-call model latency: one Silero frame, and Smart Turn split into Whisper
   feature extraction and the ONNX run.
2. Smart Turn latency and throughput with 1/4/8 sessions calling the shared backend at once.
3. Detector cost per appended chunk, for ordinary chunks and for the chunk that
   runs Smart Turn.
4. Smart Turn's classification accuracy on labeled clips.
5. Streaming endpoint decisions in audio time:
   - **complete**: a complete utterance, then silence. Endpoint delay runs from
     the end of speech to the commit.
   - **incomplete → continue**: an incomplete utterance, a pause, then a complete
     utterance. Any commit before the end is a false cutoff.
6. Resident memory added by loading each model.

## Labeled clips

The endpoint scenarios need clips labeled complete/incomplete, for example a
sample of [`pipecat-ai/smart-turn-data-v3.2-test`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-test).
Write them as WAV files and list them in a manifest:

```json
[{"path": "clips/eng_c_0001.wav", "complete": true, "language": "eng"},
 {"path": "clips/eng_i_0002.wav", "complete": false, "language": "eng"}]
```

## Run

Both artifacts must already be in the Hugging Face cache (see the
[realtime web README](../../examples/online_serving/realtime_web/README.md#semantic-vad-silero--smart-turn)),
or pass `--silero-model` / `--smart-turn-model`.

```bash
OMP_NUM_THREADS=1 python benchmarks/duplex_vad/bench_vad_latency.py \
    --manifest clips/manifest.json --markdown --json vad_report.json
```

`--wav` adds unlabeled clips, which count as complete. `--chunk-ms` sets the
append size (default 100 ms) and `--pause-ms` the thinking pause (default 1000 ms).
