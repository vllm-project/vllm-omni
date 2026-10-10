# Duplex session tool contract

A duplex function call has three owners:

| Owner | Job |
| --- | --- |
| Session (`DuplexToolLedger`) | `call_id`, epoch, duplicates, cancel, late and stale results |
| Model plugin | Recognize one call in model output, and turn an accepted result into model input |
| Application | Execute the tool. This stays outside the runtime |

The ledger does not run tools, does not start a timer, and does not block audio append while a result is outstanding. A timeout is recorded by `cancel(reason="timeout")`.

Model-specific grammars, prompts, and worker acknowledgement stay in that model's plugin. The Nemotron pending-batch acknowledgement added in #7884 still lives in the Nemotron plugin and runs only after the ledger accepts a result, through the existing `runtime_config_for_function_output` hook.

## Happy path

```mermaid
sequenceDiagram
    participant Model
    participant Plugin
    participant Ledger as Session ledger
    participant Client
    participant App as Application

    Model->>Plugin: model output
    Plugin->>Ledger: parse_function_call, or legacy function_call=True
    Ledger->>Ledger: open_call, status open, current epoch
    Ledger->>Client: function_call.done
    Client->>App: execute the tool
    App->>Ledger: conversation.item.create function_call_output
    Ledger->>Ledger: accept_result, status completed
    Ledger->>Plugin: runtime_config_for_function_output
    Plugin->>Model: maybe_continue_response
```

Audio append does not wait on this path. The ledger is not consulted when PCM is appended.

## Who does which step

```mermaid
flowchart LR
    subgraph plugin [Plugin]
        parse["parse_function_call"]
        feed["runtime_config_for_function_output"]
    end
    subgraph session [Session]
        open["open_call"]
        accept["accept_result"]
        done["function_call.done"]
    end
    subgraph outside [Outside the runtime]
        exec["Application executes the tool"]
    end
    parse --> open --> done --> exec --> accept --> feed
```

`parse_function_call` defaults to `None`. A plugin that does not emit tool calls leaves both hooks as no-ops. When the hook returns `None`, the session still accepts the legacy `function_call is True` shape (`call_id`, `name`, `arguments`). A recognized call is `{"call_id", "name", "arguments"}`.

`runtime_config_for_function_output` is unchanged. It runs only after `accept_result` succeeds. The default returns `None`, so a plugin with no tool-result encoding does not patch runtime config.

## Call states

```mermaid
stateDiagram-v2
    [*] --> open: open_call
    open --> completed: accept_result
    open --> cancelled: cancel
    open --> stale: barge_in retires older epoch
    cancelled --> cancelled: late result rejected
    completed --> completed: second result rejected
    stale --> stale: result rejected
```

| Event | Ledger | What the client sees |
| --- | --- | --- |
| First `open_call` | `open` | `function_call.done` |
| Same `call_id` opened again | unchanged | `duplicate_function_call` |
| Plugin parse is not a call | no row | `invalid_function_call` |
| `accept_result` on an open call at the current epoch | `completed` | item created, then `maybe_continue_response` |
| Unknown `call_id` | no row | `unknown_function_call` |
| Second result for a completed call | stays `completed` | `duplicate_function_call_output` |
| Result after `cancel` | stays `cancelled` | `late_function_call_output` |
| Result after `barge_in`, or epoch mismatch | `stale` | `stale_function_call_output` |

`barge_in` increments the session epoch and calls `retire_before`. Open calls from the previous epoch become `stale`. Completed and cancelled rows are left as they are. There is no background timer: a caller that wants a timeout calls `cancel(call_id, reason="timeout")`. A second cancel of an already cancelled call returns the same row.

A legacy `function_call is True` output that has no `call_id` and `name` still emits a raw `function_call.done` and does not open a ledger row. A plugin `parse_function_call` result that is not a complete call is `invalid_function_call` and does not emit `function_call.done`.

## Result that must not reach the model

```mermaid
flowchart TD
    item["function_call_output"] --> accept{"accept_result"}
    accept -->|open and current epoch| hook["runtime_config_for_function_output"]
    hook --> cont["maybe_continue_response"]
    accept -->|unknown id| unknown["unknown_function_call"]
    accept -->|already completed| dup["duplicate_function_call_output"]
    accept -->|cancelled| late["late_function_call_output"]
    accept -->|stale or older epoch| stale["stale_function_call_output"]
```

The realtime projector still checks conversation items on the wire. The session ledger is the additional check that a result belongs to a call this session opened and has not already finished, cancelled, or retired.
