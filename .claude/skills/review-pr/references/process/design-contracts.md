# Resolve Design Contracts

Use after freezing the review head and before selecting module or feature
references. The maintained [design index](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/)
defines two routing axes:

- **Module designs** own code boundaries and invariants. Select one primary
  module and a second only when the live producer-consumer path crosses it.
- **Feature designs** describe behavior spanning modules. Load every feature
  page whose implementation, default, support, or contract changes.

## Resolution order

1. Open `docs/design/index.md` in the reviewed head, not the reviewer's checkout
   or published latest tree.
2. Match changed behavior against each module page's `primary_code_paths`,
   `primary_path_exceptions`, ownership boundary, and `depends_on` metadata.
3. Read `status`, `architecture_state`, `last_verified_commit`, and any linked
   in-flight work before treating prose as current behavior.
4. Use current code and tests when a page is draft, deferred, stale, absent, or
   conflicts with the frozen implementation. Report material documentation
   drift separately.
5. Treat candidate invariants and promotion gates as review questions until
   repository policy, current code/tests, or owner-approved normative text
   makes them enforceable. Do not turn a draft identifier into an automatic
   blocker.
6. Use published latest docs only for discovery when the reviewed head lacks a
   page; never import a newer contract into an older branch silently.

When behavior changes an active module or feature contract, require the same PR
to update that page. Require a new page only when repository policy, an accepted
RFC/issue, or explicit maintainer acceptance criteria do so; otherwise treat it
as a non-blocking design suggestion. Do not demand a second page when an existing
design already owns the exact contract.
