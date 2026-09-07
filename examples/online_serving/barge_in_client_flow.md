# `barge_in_client.py` — what happens, at a glance

The demo plays one everyday conversation pattern against the full-duplex
server: ask something, get talked-over mid-answer, and see the assistant
cope. High-level flow below — for the exact wire events, read the
docstring in `barge_in_client.py`.

```mermaid
sequenceDiagram
    autonumber
    participant U as user (client script)
    participant A as assistant (server + model)

    Note over U,A: GREET
    U->>A: start a voice session (with the reference voice)
    A-->>U: ready

    Note over U,A: ASK
    U->>A: ask a question (speak the question WAV)
    Note right of A: decides on its own to answer<br/>(no VAD — the model chooses)
    A-->>U: starts answering aloud

    Note over U,A: INTERRUPT
    Note over U: listens for ~2 seconds
    par both talk at once
        U->>A: interrupts with a follow-up
    and
        A-->>U: still speaking the first answer
    end

    alt the interruption is substantial
        Note right of A: stops mid-sentence (barge-in),<br/>first answer is cut off
    else it was just a short remark
        Note right of A: finishes the first answer,<br/>then takes the follow-up
    end

    Note over U,A: ANSWER AGAIN
    A-->>U: answers the follow-up aloud

    Note over U,A: HANG UP
    U->>A: goodbye (close the session)
    Note over U: saves one recording per answer<br/>plus a summary of what happened
```

Outcome: the output directory shows which branch ran — a cut-off first
answer is saved as `response_1_cancelled.wav`, a completed one as
`response_1_completed.wav`, and `summary.json` records status, text, and
duration per answer.

Companion: the runtime architecture lives in
[`docs/design/fullduplex.md`](../../docs/design/fullduplex.md); the client
API is `vllm_omni.clients.duplex.DuplexClient`, documented with the wire
protocol in
[`docs/serving/realtime_duplex_api.md`](../../docs/serving/realtime_duplex_api.md).
