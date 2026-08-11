# Configuration Provenance

Read this reference when changed or newly load-bearing behavior reads config,
CLI, environment, legacy, alias, or default values. Always use it for endpoint,
host, port, address, connector metadata, and `*_extra_config` consumers, even
when the schema or documentation is unchanged.

## Trace the source of truth

For each semantic value, record one compact ledger:

```text
documented input -> parsed field -> competing sources and precedence
  -> runtime projection/transport -> final consumer
```

- Search the exact field, environment variable, default literal, and aliases in
  code, tests, docs, examples, and deployment files.
- Treat unchanged producers and context lines as review scope when the diff
  starts relying on them or makes their output authoritative.
- Resolve the intended precedence from active branch-local docs, enforced
  code/tests, and compatibility policy. Report a documented input that is
  silently ignored or reinterpreted at a live sink.
- Check every supported construction path and feature-off behavior without
  inventing parity for unsupported paths.

## Require discriminating evidence

Give every competing source a distinct value so the test reveals which source
wins. Prefer non-default sentinels; for example:

```text
documented config port = 25301
environment/default port = 8998
expected final endpoint = http://10.0.0.8:25301
```

Assert the final URL, connection target, runtime argument, or other live sink,
not only an intermediate object. A test that changes the environment value to
the expected configured value proves formatting, not configuration propagation.
For a bug fix, show the discriminating test fails on the frozen base and passes
on the reviewed head when the environment permits execution.

## Finding bar

A supported non-default value reaching the wrong endpoint, device, topology, or
runtime behavior is a correctness or compatibility finding. A missing
discriminating test alone is a validation gap unless repository policy requires
that regression coverage for the change.
