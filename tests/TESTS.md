# Test Log

## 2026-07-29: Front-padding fix deploy-test (commit 5e5ca301)

**What:** Deployed the context-only split fix (separate seq=1 for context, seq=2+ for audio) to eliminate front-padding. Ran `v1_realtime_test.py deploy-test`.

**Expected:** Eliminating front-padding would produce coherent audio output from the talker.

**Result:** Audio still garbled. STT transcription: "Anytime for immerse to anyone comes from to social media as in an entire family member."

**Findings:**
- Front-padding was NOT eliminated. Server logs show:
  - seq=1 (context-only): `prompt_len=19, embeds=15, pad_len=4`
  - seq=2 (audio): `prompt_len=117, embeds=45, pad_len=72`
- The budget formula `sample_count // 640 + 8` overestimates by ~2.7x vs actual Whisper encoder output (45 tokens).
- Thinker produced 36 output tokens (vs 16 pre-fix), talker ran 129 decode steps (vs 26 pre-fix). Model behavior changed significantly.
- `prompt_ids[:5]=[151675, 151675, 151675, 151675, 151644]` in talker FLUSH shows pad tokens leaking into talker input.
- Turn 2 crashed with keepalive ping timeout.

**Logs:** `tests/logs/2026-07-29_deploy-test_front-padding-fix.log`

## 2026-07-29: No-padding deploy-test (commits 6a14c99a, 29645e71)

**What:** Deployed exact Whisper output token formula (`_whisper_output_tokens`) and removed ALL padding from `thinker_preprocess`. Split prefix/suffix reserves for exact budget. Ran `v1_realtime_test.py deploy-test`.

**Expected:** With padding fully eliminated and budget exact, audio output should be coherent.

**Result:** Audio still garbled. Turn 1 produced 245205 samples (10.2s) — same duration as padded version. Turn 2 crashed with EngineCore error (stage2 WAITING_FOR_CHUNK).

**Findings:**
- Budget is now exact: no `budget_mismatch` warning in logs, `prompt_len=60` matches embeddings.
- `prompt_ids[:5]=[151644, 8948, 198, 2610, 525]` — real tokens (IM_START, system prompt), not pad tokens.
- Thinker text output appears correct (token IDs decoded to real words).
- Padding was definitively NOT the root cause of garbled audio.
- Turn 2 crash persists across all test runs — likely a separate issue in stage2 chunk handling.

**Logs:** `tests/logs/2026-07-29_deploy-test_no-padding.log`

## 2026-07-29: Flush fix deploy-test (commit 68db91b7)

**What:** Fixed t2t streaming handler to flush prefill when thinker finishes before a decode-only step can trigger the normal flush. Previously, when `_output_token_ids` was cleared by the scheduler before `save_async`, the handler misclassified prefill embeddings as decode. Ran `v1_realtime_test.py deploy-test`.

**Expected:** Turn 2 no longer crashes with `KeyError: 'prefill'`.

**Result:** `KeyError: 'prefill'` is fixed. Turn 1 produces audio (still garbled). Turn 2 no longer crashes with KeyError, but the server hangs after barge_in — epoch 1 request never starts on stage0. Websocket times out with keepalive ping timeout.

**Findings:**
- Turn 1 works identically to previous test: thinker generates correct text, talker produces 129 decode steps, audio is garbled.
- Barge_in at 20:40:11 aborts epoch 0 request. Stage1 frees the request.
- No epoch 1 activity appears in the log — engine core processes go completely silent after abort.
- The epoch transition mechanism fails to create a new stage0 request for epoch 1.
- This is a separate duplex session management issue, not a t2t handler issue.

**Logs:** `tests/logs/2026-07-29_deploy-test_flush-fix.log`

## 2026-07-29: response.done spec fix deploy-test (commit 9269a2f2)

**What:** Replaced all `audio.cancelled` internal events with `response.done{status:"cancelled"}` to conform to the OpenAI Realtime API spec. The spec requires `response.cancel` to produce `response.done` with `status=cancelled`. Ran `v1_realtime_test.py deploy-test`.

**Expected:** Client cancel drain completes properly (receives `response.done`). Turn 2 can proceed after barge_in.

**Result:** Cancel drain now works correctly — client receives `response.audio.done`, `response.content_part.done`, `response.output_item.done`, `conversation.item.done`, `response.done` in sequence. Turn 2 still fails with websocket keepalive ping timeout during the 10s silence period between turns.

**Findings:**
- The `response.done` spec fix works: cancel lifecycle completes cleanly on the client side.
- Turn 1 audio: 245205 samples (10.2s), STT: "Gonna some gasg Was success" (still garbled).
- After cancel drain, the test streams 10s of silence. With `auto_response=False`, silence is dropped by the server. The websocket keepalive times out (~20s) before turn 2's `response.create` is sent.
- The hang is not a server bug — it is the websocket library's keepalive timeout expiring during idle silence streaming. The fix is either to increase keepalive timeout on the client or shorten the inter-turn silence.

**Logs:** `tests/logs/2026-07-29_deploy-test_response-done-fix.log`

## 2026-07-30: eos_emitted fix deploy-test (commit a52a0313)

**What:** Used `eos_emitted` metadata from the Qwen3-Omni talker as the `tts_segment_end` signal in `data_plane.py`. Previously, `tts_segment_end` never triggered because `tts_is_last_chunk` is never set by Qwen3-Omni, and the fallback condition (`im_end_detected and not audio_chunks and finished`) requires no audio chunks in the same output. `eos_emitted` is set when the talker exhausts thinker embeddings and emits TTS EOS. Ran `v1_realtime_test.py deploy-test`.

**Expected:** Server emits `response.done{status:"completed"}` naturally when generation finishes, eliminating the 5s quiet timeout and websocket keepalive death on turn 2.

**Result:** All three turns completed without websocket timeout. Turn 1: 10.2s audio. Turn 2: 0.1s audio (suspiciously short). Turn 3: 5.0s audio. Chat fallback test also passed with coherent transcriptions.

**Findings:**
- The `eos_emitted` fix works: the server now terminates the response lifecycle naturally.
- The websocket keepalive timeout no longer kills the connection between turns.
- Turn 2 produced only 0.1s of audio, suggesting the barge_in epoch transition may be cutting off generation prematurely or the model produced very little output for that turn.
- Turn 1 audio still garbled (separate issue).

**Logs:** `tests/logs/2026-07-30_deploy-test_eos-emitted-fix.log`

## 2026-08-05: MiniCPM-o E2E full suite pass (branch feat/duplex-qwen3-omni)

**What:** Ran the full MiniCPM-o E2E test suite (`tests/e2e/online_serving/minicpmo/`) on AWS g7e.2xlarge. Fixed all test failures caused by removal of non-spec OpenAI Realtime WebSocket events (`response.speak`, `response.listen`, `session.closed`, `overlap.decision`, `playback.ack`).

**Expected:** All 85 tests pass after removing references to non-spec events from test assertions and demo driver code.

**Result:** 85/85 tests passed across multiple iterative runs.

**Findings:**
- Removed `response.speak` from event ordering checks in `minicpmo_realtime_duplex_scenarios.py`
- Removed `resume`/`takeover` assertions from `test_minicpmo_4_5_duplex.py` (session resume removed)
- Removed `expiry` assertions from `test_minicpmo_4_5_duplex_expansion.py` (expiry probe removed)
- Fixed `REPO_ROOT` depth in `run_minicpmo_realtime_duplex_soft_interrupt.py` (`parents[3]` to `parents[4]`)
- Rewrote `realtime_duplex_demo.py` post-commit model decision logic to use `response.done`/`response.created` instead of `response.listen`
- Removed `listen_between_responses`, `final_listen_after_commit`, `listen_after_response_before_commit` assertions from `test_minicpmo_realtime_duplex_drivers.py`
- Removed stale `validation_mode` parameter from `_send_clean_turn()` calls

**Logs:** `tests/logs/2026-08-05_minicpmo_e2e_all_pass.log`
