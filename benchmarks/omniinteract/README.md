# OmniInteract MiniCPM Duplex Benchmark

Run `vllm bench serve --omni --backend minicpmo-realtime --endpoint /v1/realtime --model openbmb/MiniCPM-o-4_5 --trust-remote-code --dataset-name omniinteract --dataset-path lucky-lance/OmniInteract --omniinteract-subsets 1q1a,1q1a_math,1qna --omniinteract-realtime-ref-audio /path/to/ref.wav --omniinteract-official-output-dir ./omniinteract-output --no-oversample --num-prompts 2 --max-concurrency 2`.

Use `--omniinteract-root` for extracted data. Successful Sessions write `.done`, WAV, transcript, and manifest artifacts; failures write `.failed.json`. Final input processing is mandatory, and `speak` also requires a completed response.

Run `python benchmarks/omniinteract/run_official_eval.py ...` with the upstream repository, ASR/aligner checkpoints, and `JUDGE_API_KEY` to enable official accuracy scoring.
