# Worked cases: choosing the next experiment

These cases use rounded measurements from earlier experiments to explain the decision at the time. Recompute the opportunity with the current request distribution, execution path, and acceptance criteria.

## Two callers of one function, one on the first-packet path

Reducing part of speech decoding by about a quarter barely moved service time. Tracing backwards from the first audio chunk exposed a second call: initialization ran encoding and diffusion to construct state for subsequent chunks, then discarded the generated audio. The decode probe missed this call. The first chunk had to wait for it.

Count calls by caller, purpose, and request, then measure reuse distance for reference inputs. In this workload, 32 requests used only 20 distinct references, often consecutively. Retaining reference features and reusing initialized state reduced low-concurrency RTF by about 18% in the historical comparison. Replay real arrival order to test cold and warm behavior. Include reference identity, batch shape, and padding policy in the cache key, invalidate derived state on eviction, and check isolation across simultaneous references.

## Full graph coverage with little change in warm service time

A vocoder received different widths from those registered at startup, so its target calls ran eagerly. Capturing the missing shapes removed the fallback. Warm service RTF differed by about 0.6%, within run-to-run variation. A separate cold run encountered successive captures during its first requests and delayed first audio. The cold stall warranted investigation even though the steady-state speedup remained unconfirmed.

Count startup captures, first occurrences in normal traffic, replays, and fallbacks separately. Use comparable warm windows for steady-state measurements. To assess early users' experience, submit requests through the actual startup path and measure readiness, first audio, and peak memory. Some historical runs were faster immediately after restart. Check run order and state before assigning a cause to a first-run outlier.

## An apparently idle device and an incomplete timeline

A report estimated that the union of busy intervals across processes covered about 30% of a window. One process alone was known to be busy for over 80% of that same window. The aggregation rounded event midpoints to whole seconds and counted distinct seconds as duration. Device operator details had also been exported for only some processes.

Clip raw start and end times to a common window and compute their interval union. The union must cover each member and fit within the window. Check the expected processes and stages against export coverage and mark missing intervals as unknown. If the first host read still looks expensive, time the wait for outstanding device work separately from the transfer after the producer is ready.

## Are the proposed batch rows actually compatible?

Serial groups were costly in a heavier workload. At dispatch on target traffic, the average batch contained only about two chunks, and continuation cache lengths often differed. Replacing a single-row concatenation with a copy gave no consistent end-to-end improvement. The encoder lacked a mask for padded cache entries, so padding with zeros would change attention.

Count ready rows, valid groups, and replays in one pre-dispatch snapshot. If enough compatible rows exist, test real caches and compare the next chunk's state as well as the current output. Otherwise save a trigger, such as an equivalent path for common length combinations, before investing in wider batches.

## Where early stopping first diverges

Long prefill mixed with active decode changed stopping behavior. Raw logits for the same input already differed before sampling. A fresh compile cache reproduced the problem. Running only target prefill eagerly, with compiled decode and draft paths retained, restored the expected stopping behavior in seven reproductions.

This narrowed the reproducer to compiled target prefill. The faulty operation remained unknown. Locate the first divergent tensor between serial and mixed execution before toggling a stage. Qualify the narrow workaround on triggering inputs, ordinary fast-path traffic, and peer requests in the mixed batch, then measure its cost.

## The first chunk looks correct. What state does the next one read?

An encoder graph expansion was tested on fixed real codec inputs and references of varying lengths. Baseline repeats matched on cross-chunk state hashes. Baseline versus candidate differed on held-out samples. The experiment required state equivalence, so testing a larger service load would have qualified the wrong implementation.

Restricting the new graph to exact batch shapes and taking the original path elsewhere restored matching hashes in the component comparison. Establish baseline repeatability, then exercise hits, fallback, and transitions within a request. Localize mismatches to live tensor elements to exclude padding-only differences. Recount eligible traffic after narrowing the guard before committing to serving A/B.

## The upstream stage improved. When should it be tested again?

In a diagnostic sample, text-to-last-codec time fell from about 560 to 430 ms, while last-codec-to-PCM-ready time rose from about 390 to 570 ms. Normal serving RTF differences lay within run variation. Stage means could not distinguish longer downstream queues, slower execution, or a change in overlap.

Split receipt, execution start, and PCM readiness for the same request and chunk. Save the local gain, unresolved downstream cost, and switch needed for a retest. Once the downstream wait shrinks and the upstream stage again affects completion, toggle the original change on the new baseline to measure its incremental effect.

## A secondary replica fails with OOM on its first trial

A secondary vocoder produced audio in isolation but failed during mixed-load warmup. The failing allocation was a temporary copy of history. Reducing the secondary workspace and copying history in bounded chunks provided a targeted way to retry the same peak.

After the repair, a comparison kept the hardware and both replicas fixed and varied only the fraction routed to the secondary vocoder. Routing one quarter of requests reduced mean warm RTF by about 8% against zero routing and passed the paired audio checks for that workload. Memory headroom remained small. This measures the routing gain with the repaired configuration. To isolate the performance contribution of bounded copying, hold routing and the other settings fixed. Recheck peak memory before extending the result to longer histories or higher concurrency.
