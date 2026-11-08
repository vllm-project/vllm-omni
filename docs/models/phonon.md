# Phonon 31M EN

Single-stage MRV2 integration for `gradium/phonon-31m-en`.
The default deployment uses one request, TP=1, BF16 AR/Flow and FP32 Mimi.
The default pretrained voice produces 24 kHz mono audio in 1920-sample frames.

## Serve

Install the repository and its TTS dependencies, including `sentencepiece` and
`safetensors`, in a CUDA-enabled vLLM environment. The model ID uses the default
Hugging Face cache and downloads missing files as needed.

```bash
vllm serve gradium/phonon-31m-en --omni \
  --deploy-config vllm_omni/deploy/phonon.yaml
```

Use `/v1/audio/speech` with `voice="default"` and English text. The adapter
initializes the official SentencePiece tokenizer once and encodes request text.
Currently, only the checkpoint's default voice embedding from
`default-voice.safetensors` is supported. The worker loads and projects it once
during model-state initialization, then reuses the GPU tensor across requests.
The adapter reads its shape to determine prompt length. Requests carry text IDs
and tail-frame settings. The model checks the speaker file's `sig@epoch`
against `embeddings_ext`.
Whitespace, capitalization and sentence-final punctuation are normalized.
Full official English number/date/currency normalization is not ported.

## Execution and state

AR uses vLLM paged attention and runner CUDA graphs. The stateless LSD flow
head has a separate main-stream graph. Its normalized latent supplies the
next AR embedding through the model state's persistent feedback buffer.

Mimi consumes a denormalized latent through its quantizer output projection,
depthwise upsampling, two Transformer layers and SEANet decoder. Mimi owns
fixed input buffers and slot-indexed KV/conv pools. Attention writes into the
ring in place and applies the causal context mask in physical ring order.
Convolution updates read from separately materialized history/input storage.

Mimi decoding runs on the current CUDA stream using fixed-address buffers and
CUDA Graph replay. The decoder returns a completion event with each PCM chunk;
the MRV2 streaming-audio output path waits for that event before copying audio
to CPU. Decoded PCM is cloned before graph replay so later executions cannot
overwrite a chunk that is still being delivered.

AR and Mimi share `input_batch.idx_mapping`. Although tensor interfaces are
batch-shaped, admission explicitly requires B=1. The pipeline fails a request
on recompute preemption. Cancellation releases its shared slot.
EOS includes the triggering frame plus three tail frames for short text or
one for longer text, following the official prompt policy. Every frame is
emitted immediately, including the final frame.

## Limitations

- The initial implementation supports one concurrent request (`max_num_seqs=1`)
  and tensor parallelism size 1.
- Requests cannot be recompute-preempted; preemption fails the request.
- Only the checkpoint's default voice embedding (`voice="default"`) is supported.
  Custom speaker embeddings, online voice cloning and additional checkpoint
  variants are not supported.

## Default-voice benchmark

Start the Phonon service, then run the existing universal TTS benchmark against
it using the bundled English smoke dataset. Use concurrency 1 because the
initial deployment admits one request at a time.

```bash
python benchmarks/tts/bench_tts.py \
  --model gradium/phonon-31m-en \
  --task default_voice --locale en \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --concurrency 1 --num-prompts 20 --num-warmups 2 \
  --host 127.0.0.1 --port 8000 \
  --output-dir ./results/phonon
```

If the server was started with `--served-model-name phonon`, also pass
`--served-model-name phonon` to the benchmark. The registry sends `voice=default`
and English conditioning.
