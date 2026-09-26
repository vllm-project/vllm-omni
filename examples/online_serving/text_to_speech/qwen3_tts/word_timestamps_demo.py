"""Gradio demo for streaming TTS word-level timestamps.

Connects to the WebSocket endpoint ``/v1/audio/speech/stream`` with
``word_timestamps: true`` and visualizes the alignment: each sentence's
audio plays in a native ``<audio>`` element while its text is rendered as
inline word spans, the current word highlighting as ``audio.currentTime``
crosses each ``start_ms``.

A "Stop (barge-in)" button cuts playback and reports the last-spoken word,
demonstrating the voice-agent barge-in use case.

Timestamps are sentence-level: audio streams in real time, then the forced
aligner runs once per sentence and the complete word timestamps arrive in a
trailing frame. The demo therefore plays one sentence block at a time.

Usage:
    # Launch the server with a forced aligner first, e.g.:
    #   vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni \
    #       --deploy-config vllm_omni/deploy/qwen3_tts.yaml --trust-remote-code \
    #       --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B
    # Then:
    python word_timestamps_demo.py --api-base http://localhost:8000
"""

import argparse
import asyncio
import json
import logging

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install it with: pip install 'vllm-omni[demo]'") from None
try:
    import websockets
except ImportError:
    raise ImportError("websockets is required to run this demo. Install it with: pip install websockets") from None
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from tts_common import (
    PCM_SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    TASK_TYPES,
    add_common_args,
    fetch_voices,
)

logger = logging.getLogger(__name__)


def _upstream_ws_url(api_base: str) -> str:
    """Map an http(s) API base to its WebSocket streaming endpoint."""
    base = api_base.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    else:
        base = "ws://" + base
    return base + "/v1/audio/speech/stream"


# ── Front-end ────────────────────────────────────────────────────────
PLAYER_HTML = """
<div id="wt-root">
  <div style="display:flex; align-items:center; gap:10px;">
    <div id="wt-dot" style="width:10px;height:10px;border-radius:50%;background:#ccc;flex-shrink:0;"></div>
    <span id="wt-status" style="font-weight:600;font-size:1.05em;">Ready</span>
    <button id="wt-stop" onclick="window.wtStop()"
      style="display:none; margin-left:auto; padding:5px 16px; border-radius:6px; border:1px solid #EF5552;
             background:#fff; color:#EF5552; cursor:pointer; font-size:0.85em;">Stop (barge-in)</button>
  </div>
  <div id="wt-barge" style="display:none; margin-top:10px; padding:8px 12px; border-radius:6px;
       background:#fff5f5; border:1px solid #f3c2c2; color:#b02a2a; font-size:0.9em;"></div>
  <div id="wt-sentences" style="margin-top:12px; display:flex; flex-direction:column; gap:14px;"></div>
</div>
"""


def _build_head_js(sample_rate: int) -> str:
    """Browser-side player: WebSocket -> per-sentence <audio> + word highlight."""
    return f"""
<script>
const WT_SR = {sample_rate};
let wtWS = null, wtQueue = [], wtPlaying = false, wtCur = null, wtLastWord = "";

function wtSet(text, color) {{
    const s = document.getElementById('wt-status');
    const d = document.getElementById('wt-dot');
    if (s) s.textContent = text;
    if (d) d.style.background = color || '#ccc';
}}

function wtStopBtn(show) {{
    const b = document.getElementById('wt-stop');
    if (b) b.style.display = show ? 'inline-block' : 'none';
}}

// Build a WAV Blob (16-bit mono PCM) from concatenated Int16 samples.
function wtPcmToWav(int16, sr) {{
    const n = int16.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buf);
    const w = (off, s) => {{ for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); }};
    w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVE');
    w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, 'data'); v.setUint32(40, n * 2, true);
    let off = 44;
    for (let i = 0; i < n; i++) {{ v.setInt16(off, int16[i], true); off += 2; }}
    return new Blob([buf], {{ type: 'audio/wav' }});
}}

function wtConcat(chunks) {{
    let total = 0;
    for (const c of chunks) total += c.length;
    const out = new Int16Array(total);
    let off = 0;
    for (const c of chunks) {{ out.set(c, off); off += c.length; }}
    return out;
}}

// Render one finished sentence: <audio> + inline word spans.
function wtRenderSentence(sentence) {{
    const wrap = document.getElementById('wt-sentences');
    if (!wrap) return;
    const block = document.createElement('div');
    block.style.cssText = 'border:1px solid #e3e8ef;border-radius:8px;padding:10px 12px;';

    const head = document.createElement('div');
    head.style.cssText = 'font-size:0.75em;color:#888;margin-bottom:6px;';
    head.textContent = 'Sentence ' + sentence.index +
        (sentence.timestamps ? ' (' + sentence.timestamps.length + ' words)' : ' (no timestamps)');
    block.appendChild(head);

    const textEl = document.createElement('div');
    textEl.style.cssText = 'font-size:1.15em;line-height:1.8;margin-bottom:8px;';
    if (sentence.timestamps && sentence.timestamps.length) {{
        sentence.spans = [];
        for (const ts of sentence.timestamps) {{
            const span = document.createElement('span');
            span.textContent = ts.word;
            span.dataset.start = ts.start_ms;
            span.dataset.end = ts.end_ms;
            span.style.cssText = 'padding:1px 2px;border-radius:3px;transition:background 0.1s;';
            textEl.appendChild(span);
            // Keep spacing for spaced languages; CJK chars sit flush.
            textEl.appendChild(document.createTextNode(' '));
            sentence.spans.push(span);
        }}
    }} else {{
        textEl.textContent = sentence.text || '(timestamps unavailable)';
    }}
    block.appendChild(textEl);

    const audio = document.createElement('audio');
    audio.controls = true;
    audio.style.cssText = 'width:100%;';
    audio.src = URL.createObjectURL(wtPcmToWav(wtConcat(sentence.chunks), WT_SR));
    block.appendChild(audio);
    sentence.audio = audio;

    // Highlight the current word. Driven by requestAnimationFrame (~60fps)
    // rather than the <audio> 'timeupdate' event, which only fires ~4x/sec
    // and is too coarse for short words. We light the latest word whose
    // start_ms has passed, so there is always exactly one active word (no
    // inter-word flicker) and wtLastWord stays well-defined for barge-in.
    let raf = null;
    function highlight() {{
        if (!sentence.spans) return;
        const t = audio.currentTime * 1000;
        let idx = -1;
        for (let i = 0; i < sentence.spans.length; i++) {{
            if (t >= +sentence.spans[i].dataset.start) idx = i;
            else break;
        }}
        sentence.spans.forEach((sp, i) => {{
            const on = i === idx;
            sp.style.background = on ? '#ffe680' : 'transparent';
            sp.style.fontWeight = on ? '700' : '400';
        }});
        if (idx >= 0) wtLastWord = sentence.spans[idx].textContent;
    }}
    function loop() {{
        highlight();
        if (!audio.paused && !audio.ended) raf = requestAnimationFrame(loop);
    }}
    audio.addEventListener('play', () => {{
        wtCur = audio;
        wtSet('Playing sentence ' + sentence.index + '...', '#64dd17');
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(loop);
    }});
    audio.addEventListener('pause', () => cancelAnimationFrame(raf));
    audio.addEventListener('seeked', highlight);
    audio.addEventListener('ended', () => {{
        cancelAnimationFrame(raf);
        highlight();
        wtPlaying = false;
        wtPlayNext();
    }});

    wrap.appendChild(block);
    wtQueue.push(audio);
    wtPlayNext();
}}

function wtPlayNext() {{
    if (wtPlaying) return;
    const next = wtQueue.shift();
    if (!next) {{
        if (wtWS === null) {{ wtSet('Done', '#64dd17'); wtStopBtn(false); }}
        return;
    }}
    wtPlaying = true;
    next.play().catch(() => {{ wtPlaying = false; }});
}}

window.wtStop = function() {{
    // Barge-in: cut playback, stop generation, report the last-spoken word.
    if (wtCur) {{ try {{ wtCur.pause(); }} catch (e) {{}} }}
    wtQueue = [];
    wtPlaying = false;
    if (wtWS) {{ try {{ wtWS.close(); }} catch (e) {{}} wtWS = null; }}
    wtSet('Stopped (barge-in)', '#999');
    wtStopBtn(false);
    const b = document.getElementById('wt-barge');
    if (b) {{
        b.style.display = 'block';
        b.textContent = wtLastWord
            ? 'Barge-in at word: "' + wtLastWord + '"  (a voice agent would resume the conversation from here)'
            : 'Barge-in before any word was spoken.';
    }}
}};

window.wtGenerate = function(payload) {{
    if (wtWS) {{ try {{ wtWS.close(); }} catch (e) {{}} }}
    wtQueue = []; wtPlaying = false; wtCur = null; wtLastWord = "";
    document.getElementById('wt-sentences').innerHTML = '';
    const barge = document.getElementById('wt-barge');
    if (barge) {{ barge.style.display = 'none'; barge.textContent = ''; }}
    wtStopBtn(true);
    wtSet('Connecting...', '#4A90D9');

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(proto + '//' + location.host + '/proxy/stream');
    wtWS = ws;
    const sentences = {{}};

    ws.onopen = () => {{
        wtSet('Streaming...', '#4A90D9');
        ws.send(JSON.stringify(Object.assign({{ type: 'session.config' }}, payload.config)));
        ws.send(JSON.stringify({{ type: 'input.text', text: payload.text }}));
        ws.send(JSON.stringify({{ type: 'input.done' }}));
    }};

    ws.onmessage = (ev) => {{
        let msg;
        try {{ msg = JSON.parse(ev.data); }} catch (e) {{ return; }}
        if (msg.type === 'audio.start') {{
            sentences[msg.sentence_index] = {{
                index: msg.sentence_index, text: msg.sentence_text || '',
                chunks: [], timestamps: null
            }};
        }} else if (msg.type === 'audio.chunk') {{
            const s = sentences[msg.sentence_index];
            if (!s) return;
            if (msg.audio_b64) {{
                const bin = atob(msg.audio_b64);
                const bytes = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                const usable = bytes.length - (bytes.length % 2);
                if (usable > 0) s.chunks.push(new Int16Array(bytes.buffer, 0, usable / 2));
            }}
            if (msg.timestamps != null) s.timestamps = msg.timestamps;
        }} else if (msg.type === 'audio.done') {{
            const s = sentences[msg.sentence_index];
            if (s) wtRenderSentence(s);
        }} else if (msg.type === 'session.done') {{
            wtWS = null;
            try {{ ws.close(); }} catch (e) {{}}
            wtPlayNext();
        }} else if (msg.type === 'error') {{
            wtSet('Error: ' + (msg.message || 'unknown'), '#EF5552');
        }}
    }};

    ws.onerror = () => wtSet('WebSocket error', '#EF5552');
    ws.onclose = () => {{ if (wtWS === ws) wtWS = null; }};
}};
</script>
"""


def create_app(api_base: str) -> FastAPI:
    """FastAPI app with a same-origin WebSocket proxy + Gradio UI."""
    fastapi_app = FastAPI()
    upstream_url = _upstream_ws_url(api_base)

    @fastapi_app.websocket("/proxy/stream")
    async def proxy_stream(client_ws: WebSocket):
        await client_ws.accept()
        try:
            async with websockets.connect(upstream_url, max_size=None) as upstream:

                async def client_to_upstream():
                    try:
                        while True:
                            await upstream.send(await client_ws.receive_text())
                    except (WebSocketDisconnect, Exception):
                        await upstream.close()

                async def upstream_to_client():
                    try:
                        async for msg in upstream:
                            if isinstance(msg, bytes):
                                await client_ws.send_bytes(msg)
                            else:
                                await client_ws.send_text(msg)
                    except Exception:
                        pass
                    finally:
                        try:
                            await client_ws.close()
                        except Exception:
                            pass

                await asyncio.gather(client_to_upstream(), upstream_to_client())
        except Exception as exc:
            logger.exception("WS proxy error")
            try:
                await client_ws.send_text(json.dumps({"type": "error", "message": f"proxy: {exc}"}))
                await client_ws.close()
            except Exception:
                pass

    voices = fetch_voices(api_base)

    with gr.Blocks(title="Qwen3-TTS Word Timestamps") as demo:
        gr.HTML(f"""
        <div style="margin-bottom:8px;">
          <h1 style="margin:0; font-size:1.5em;">Qwen3-TTS Streaming Word Timestamps</h1>
          <span style="font-size:0.85em; color:#666;">
            Served by <a href="https://github.com/vllm-project/vllm-omni" target="_blank">vLLM-Omni</a>
            &nbsp;&middot;&nbsp; <code>{api_base}</code> &nbsp;&middot;&nbsp;
            launch the server with <code>--forced-aligner</code>.
          </span>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    value="Hello world. How are you today? I am doing great.",
                    lines=3,
                )
                with gr.Row():
                    task_type = gr.Radio(choices=TASK_TYPES, value="CustomVoice", label="Task Type", scale=2)
                    language = gr.Dropdown(choices=SUPPORTED_LANGUAGES, value="Auto", label="Language", scale=1)
                voice = gr.Dropdown(
                    choices=voices,
                    value=voices[0] if voices else None,
                    label="Speaker",
                    allow_custom_value=True,
                )
                instructions = gr.Textbox(
                    label="Instructions",
                    placeholder="Optional style/emotion (required for VoiceDesign)",
                    lines=1,
                )
                with gr.Column(visible=False) as ref_group:
                    ref_audio_url = gr.Textbox(
                        label="Reference Audio URL (Base task)",
                        placeholder="https://example.com/reference.wav",
                        lines=1,
                    )
                    ref_text = gr.Textbox(label="Reference Transcript", lines=1)
                generate_btn = gr.Button("Generate + Align", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.HTML(value=PLAYER_HTML)

        hidden_payload = gr.Textbox(visible=False, elem_id="wt-payload")

        def on_task_change(tt):
            return gr.update(visible=(tt == "Base"))

        task_type.change(fn=on_task_change, inputs=[task_type], outputs=[ref_group])

        def on_generate(text, tt, voice_v, lang_v, instr_v, ref_url, ref_t):
            if not text or not text.strip():
                raise gr.Error("Please enter text to synthesize.")
            config: dict = {
                "task_type": tt,
                "word_timestamps": True,
                "stream_audio": True,
                "response_format": "pcm",
            }
            if lang_v:
                config["language"] = lang_v
            if tt == "CustomVoice":
                if voice_v:
                    config["voice"] = voice_v
                if instr_v and instr_v.strip():
                    config["instructions"] = instr_v.strip()
            elif tt == "VoiceDesign":
                if not instr_v or not instr_v.strip():
                    raise gr.Error("VoiceDesign requires voice style instructions.")
                config["instructions"] = instr_v.strip()
            elif tt == "Base":
                if not ref_url or not ref_url.strip():
                    raise gr.Error("Base (voice clone) needs a reference audio URL.")
                config["ref_audio"] = ref_url.strip()
                if ref_t and ref_t.strip():
                    config["ref_text"] = ref_t.strip()
            return json.dumps({"config": config, "text": text.strip()})

        generate_btn.click(
            fn=on_generate,
            inputs=[text_input, task_type, voice, language, instructions, ref_audio_url, ref_text],
            outputs=[hidden_payload],
        ).then(
            fn=lambda p: p,
            inputs=[hidden_payload],
            outputs=[hidden_payload],
            js="(p) => { if (p && p.trim()) window.wtGenerate(JSON.parse(p)); return p; }",
        )

        demo.queue()

    return gr.mount_gradio_app(
        fastapi_app,
        demo,
        path="/",
        head=_build_head_js(PCM_SAMPLE_RATE),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for streaming TTS word-level timestamps.")
    add_common_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    print(f"Connecting to vLLM server at: {args.api_base}")
    app = create_app(args.api_base)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
