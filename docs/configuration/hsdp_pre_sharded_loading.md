# Pre-sharded HSDP loading

Use `--use-hsdp --hsdp-shard-size 8 --hsdp-weight-load-strategy pre_sharded`
to load checkpoint values into sharded proxies before preparing each real
transformer block. Set the shard size to the available GPU count (or configure replication explicitly).
The default strategy, `full`, retains the existing load-then-shard behavior.
The same `hsdp_weight_load_strategy` option is available through Python and stage
engine configuration.

Each rank reads rank-local slices of shardable checkpoint tensors; scalar and
integer tensors are replicated. This avoids materializing a full CPU copy of
the checkpoint, reduces CPU memory use and redundant checkpoint I/O,
and can improve loading speed. Real transformer blocks are materialized and
sharded one at a time, limiting temporary GPU memory use.

The pre-sharded strategy reads rank-local slices from ordinary Hugging Face
safetensors checkpoints into FSDP-owned proxy storage. No converted checkpoint is
required. Cosmos 3 and Cosmos 3 Edge transformer blocks release their initialized
parameter storage to meta as they are constructed; nonpersistent buffers retain
their values.
For construction-time memory optimization, use Cosmos 3/Edge's block-by-block
meta release as the reference implementation. This is the model's responsibility;
the loader does not release initial model storage before loading proxies.
Encoders and VAEs follow their existing loading paths.

Checkpoint tensors are loaded into sharded proxies in checkpoint layout. For each
block selected by `_hsdp_shard_conditions`, the loader gathers its proxy tensors,
applies the ordinary loader's checkpoint adapters, loads and postprocesses the
real block's weights, then applies FSDP sharding. Adapters use the component's
quantization configuration and run before `convert_checkpoint_tensor`.
Finally, it loads and processes the remaining DiT root weights and shards the root.

This strategy requires the default diffusion pipeline loader, complete dedicated
transformer checkpoint sources, and supported checkpoint key mappings.
Tensor parallelism and LoRA are unsupported. Every worker needs checkpoint access.
`num_weight_load_threads` controls the safetensors reader's threads per rank.

## Model-specific weight loading

Weights are loaded independently for each module selected by
`_hsdp_shard_conditions` (typically transformer blocks), then for the remaining
DiT root parameters. Keep model-specific loading logic local to these groups:
for example, QKV fusion and its scale checks belong in the attention module's
`load_weights()`, rather than only in the transformer root's loader.

Each group uses its own load_weights() when defined, otherwise AutoWeightsLoader. Custom loaders receive group-local names and return loaded parameter names. The DiT root receives only remaining non-block weights; its loader must not require or modify already-sharded blocks.

## Weight tensor transforms

Provide `convert_checkpoint_tensor(name, tensor)` on the pipeline when checkpoint
tensors need conversion to the model's weight format, such as transposing or
reshaping weights.
