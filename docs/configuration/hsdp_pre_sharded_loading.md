# Pre-sharded HSDP loading

Use `--use-hsdp --hsdp-shard-size 8 --hsdp-weight-load-strategy pre_sharded`
to shard transformer parameters before loading checkpoint values. Set the shard
size to the available GPU count (or configure replication explicitly).
The default strategy, `full`, retains the existing load-then-shard behavior.
The same `hsdp_weight_load_strategy` option is available through Python and stage
engine configuration.

The pre-sharded strategy reads rank-local slices from ordinary Hugging Face
safetensors checkpoints into FSDP-owned storage. No converted checkpoint is
required. Cosmos 3 and Cosmos 3 Edge transformer blocks release their initialized
parameter storage to meta as they are constructed; nonpersistent buffers retain
their values.
Encoders and VAEs follow their existing loading paths.

This strategy requires the default diffusion pipeline loader, complete dedicated
transformer checkpoint sources, and supported checkpoint key mappings. Tensor
layouts must match the runtime layout; tensor transforms are unsupported.
Quantization and LoRA are unsupported. Every worker needs checkpoint access.
`num_weight_load_threads` controls the safetensors reader's threads per rank.
