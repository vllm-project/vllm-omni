# Bimanual CPU fixtures

`legacy_action_schema.json` and `cookbook_action_schema.json` contain v3 and
v4 action contracts and normalizer data, without model weights. The v4 fixture
records the statistics declaration, effective pose convention, and inference
profile authority under `normalizer.source.pose_convention_override`.

`camera_stone.npz` contains frozen regression vectors for four camera cases:
`stone_w` (`w-61`), `stone_s` (`s-61`), `stone_up` (`up-61`), and `stone_left`
(`left-61`). The vectors retain their original numerical values and archive
SHA-256. They are not generated from the implementation under test.

| NPZ key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `poses` | `4 x 61 x 4 x 4` | float64 | Absolute OpenCV camera-to-world poses, starting at identity |
| `raw_actions` | `4 x 60 x 9` | float32 | Translation and column-based rot6d deltas, anchored every 16 targets |
| `normalized_actions` | `4 x 60 x 9` | float32 | Global-asinh camera values before padding to 64 channels |

Each command produces 62 absolute poses (identity plus 61 targets). A 61-frame
request consumes the first 61 poses before computing 60 action rows. Global-asinh
normalization uses the canonical 59D metric-gripper statistics' first nine channels.

The companion `camera_stone.json` records parameters, generation environment,
statistics SHA-256, array shapes and dtypes, and archive hash. Tests load the NPZ
with `allow_pickle=False`, verify its hash and the normalizer's statistics hash,
and compare resolved poses, raw actions, normalized actions, and zero padding.
The test suite needs no external checkout or fixture-generation tool.

Comparison tolerances are `rtol=atol=1e-12` for FP64 poses, `rtol=1e-6,
atol=1e-7` for raw FP32 actions, and `rtol=atol=1e-6` for normalized FP32 actions.
These allow floating-point roundoff while detecting changes to conventions,
anchor boundaries, normalization, and rotation ordering. This fixture validates
conditioning only; it does not establish model or VAE output equivalence.

Run the checked-in vectors with:

```bash
python tools/run_bimanual_cpu_tests.py -q
```

Changes to these vectors require independent numerical validation and review of
the arrays, parameters, statistics hash, and updated archive hash. Never regenerate
expected values automatically from the camera implementation during tests.
