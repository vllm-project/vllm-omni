# HSDP request concurrency

Enable `--use-hsdp --hsdp-shard-size 8 --hsdp-data-parallel` to process one
independent compatible request per HSDP rank. Set the shard size to match your
GPU allocation. The Python/stage parallel configuration flag is
`hsdp_data_parallel: true`.

Weights remain sharded and are gathered collectively. Each rank computes its
own activations. The scheduler admits up to the HSDP world size in one wave;
`request_batch_max_wait_ms` controls the wait for compatible requests. It
defaults to 500 ms in this mode, so a burst of requests forms one wave instead
of dispatching the first arrival alone. Set it to 0 to disable the wait.
Request metadata and KV state stay bound to the request assigned to each rank.
Short waves repeat requests on surplus ranks to keep collective execution
aligned; only the requested results are returned.

Tensor, sequence, pipeline, and CFG parallel sizes must all be one; expert
parallelism is unsupported with HSDP. `vae_patch_parallel_size` must also be
one, because patch-parallel VAE decode stitches tiles across all ranks and would
mix the different requests' latents. Requests in a wave must have compatible
shapes, guidance, denoising schedules, output counts, and LoRA settings, with
identical extra arguments and nonempty prompts. Step execution is unsupported.
MiniMax-H3 is unsupported because conditioning and output ownership are not
request-local.

`cache_backend` must be `none` or `sea_cache`. SeaCache synchronizes its skip
decision across the HSDP shard group, so a rank that needs a full forward makes
every rank run one; the other cache backends decide per rank and can skip
different weight collectives, so they are rejected at config time. Expect fewer
cache hits than with a single request because a step is skipped only when all
requests in the wave agree.

This feature works with the existing `full` HSDP loader and multiprocessing
executor. It does not require pre-sharded loading or compact video transport.
