# Higgs-Audio V3 Online Serving

## Start the server

```bash
# Default: GPU 0, port 8095
./examples/online_serving/text_to_speech/higgs_audio_v3/run_server.sh

# Custom GPU / port
PORT=8096 GPUS=0,1 ./examples/online_serving/text_to_speech/higgs_audio_v3/run_server.sh
```

## Plain text TTS

```bash
python examples/online_serving/text_to_speech/higgs_audio_v3/batch_speech_client.py \
    --base-url http://localhost:8095 \
    --output-dir /tmp/higgs_v3_batch \
    --prompts "Hello world." "The quick brown fox jumps over the lazy dog."
```

## Voice clone

```bash
python examples/online_serving/text_to_speech/higgs_audio_v3/batch_speech_client.py \
    --base-url http://localhost:8095 \
    --output-dir /tmp/higgs_v3_clone \
    --ref-audio path/to/reference.wav \
    --ref-text "transcript of the reference clip" \
    --prompts "Text to synthesize in the cloned voice."
```

## curl example

```bash
curl -X POST http://localhost:8095/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model": "higgs_audio_v3", "input": "Hello world."}' \
    --output hello.wav
```
