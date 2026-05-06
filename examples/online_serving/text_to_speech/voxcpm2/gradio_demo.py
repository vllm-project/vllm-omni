"""Gradio demo for VoxCPM2 TTS with gapless streaming audio playback.

Uses a custom AudioWorklet-based player for gap-free streaming
(adapted from the Qwen3-TTS demo). Audio is streamed from the vLLM
server through a same-origin proxy and played via the Web Audio API's
AudioWorklet, which maintains a FIFO buffer queue and plays samples at
the audio clock rate.

Usage:
    # Start the vLLM server first:
    vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000

    # Then launch the demo:
    python gradio_demo.py --api-base http://localhost:8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging

import gradio as gr
import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000

# ── AudioWorklet processor (loaded in browser via Blob URL) ──────────
WORKLET_JS = r"""
class TTSPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.queue = [];
        this.buf = null;
        this.pos = 0;
        this.playing = false;
        this.played = 0;
        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'clear') {
                this.queue = []; this.buf = null; this.pos = 0; this.played = 0;
                if (this.playing) { this.playing = false; this.port.postMessage({type:'stopped'}); }
                return;
            }
            this.queue.push(e.data);
        };
    }
    process(inputs, outputs) {
        const out = outputs[0][0];
        for (let i = 0; i < out.length; i++) {
            if (!this.buf || this.pos >= this.buf.length) {
                if (this.queue.length > 0) {
                    this.buf = this.queue.shift(); this.pos = 0;
                } else {
                    for (let j = i; j < out.length; j++) out[j] = 0;
                    if (this.playing) { this.playing = false; this.port.postMessage({type:'stopped', played:this.played}); }
                    return true;
                }
            }
            out[i] = this.buf[this.pos++] / 32768;
            this.played++;
        }
        if (!this.playing) { this.playing = true; this.port.postMessage({type:'started'}); }
        return true;
    }
}
registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
"""

PLAYER_HTML = """
<div id="tts-player">
  <div style="display:flex; align-items:center; gap:10px;">
    <div id="tts-status-dot" style="width:10px;height:10px;border-radius:50%;background:#ccc;flex-shrink:0;"></div>
    <span id="tts-status" style="font-weight:600;font-size:1.05em;">Ready</span>
    <button id="tts-stop-btn" onclick="window.ttsStop()"
      style="display:none; margin-left:auto; padding:5px 16px; border-radius:6px; border:1px solid #EF5552;
             background:#fff; color:#EF5552; cursor:pointer; font-size:0.85em;">Stop</button>
  </div>
  <div id="tts-metrics" style="display:none; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:12px;">
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">TTFP</div>
      <div id="tts-m-ttfp" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">RTF</div>
      <div id="tts-m-rtf" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">Audio</div>
      <div id="tts-m-dur" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">Speed</div>
      <div id="tts-m-speed" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
  </div>
  <div id="tts-rtf-bar-wrap" style="display:none; background:#e8ecf1; border-radius:4px; height:20px; overflow:hidden; position:relative; margin-top:10px;">
    <div id="tts-rtf-bar" style="height:100%; border-radius:4px; transition:width 0.3s ease, background 0.3s ease; width:0%;"></div>
    <span id="tts-rtf-label" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.75em;font-weight:600;color:#444;"></span>
  </div>
  <div id="tts-elapsed" style="display:none; margin-top:6px; font-size:0.8em; color:#999; text-align:right;"></div>
</div>
"""


def _build_player_js() -> str:
    return f"""
    <script>
    const SR = {SAMPLE_RATE};
    const WC = {json.dumps(WORKLET_JS)};
    let ctx = null, node = null, abort = null, gen = false, st = {{}};

    async function init() {{
        if (ctx) return;
        ctx = new AudioContext({{ sampleRate: SR }});
        const b = new Blob([WC], {{ type: 'application/javascript' }});
        const u = URL.createObjectURL(b);
        await ctx.audioWorklet.addModule(u);
        URL.revokeObjectURL(u);
        node = new AudioWorkletNode(ctx, 'tts-playback-processor');
        node.connect(ctx.destination);
        node.port.onmessage = (e) => {{
            if (e.data.type === 'started') setStatus('Playing...', '#64dd17');
            else if (e.data.type === 'stopped' && !gen) {{
                setStatus('Done', '#64dd17'); showStats(true);
                const btn = document.getElementById('tts-stop-btn');
                if (btn) btn.style.display = 'none';
            }}
        }};
    }}

    function setStatus(text, color) {{
        const s = document.getElementById('tts-status');
        const d = document.getElementById('tts-status-dot');
        if (s) s.textContent = text;
        if (d) d.style.background = color || '#ccc';
    }}

    function showStats(fin) {{
        if (!st.t0) return;
        const elapsed = (fin && st.streamEnd ? (st.streamEnd - st.t0) : (performance.now() - st.t0)) / 1000;
        const dur = st.samples / SR;
        const mTtfp = document.getElementById('tts-m-ttfp');
        const mRtf = document.getElementById('tts-m-rtf');
        const mDur = document.getElementById('tts-m-dur');
        const mSpeed = document.getElementById('tts-m-speed');
        const bar = document.getElementById('tts-rtf-bar');
        const barLabel = document.getElementById('tts-rtf-label');
        const elapsedEl = document.getElementById('tts-elapsed');
        if (mTtfp && st.ttfp != null) mTtfp.textContent = st.ttfp.toFixed(0) + 'ms';
        if (mDur) mDur.textContent = dur.toFixed(1) + 's';
        if (dur > 0 && elapsed > 0) {{
            const rtf = elapsed / dur;
            const speed = 1 / rtf;
            if (mRtf) {{
                mRtf.textContent = rtf.toFixed(2) + 'x';
                mRtf.style.color = rtf < 1 ? '#64dd17' : rtf < 1.5 ? '#e8a317' : '#EF5552';
            }}
            if (mSpeed) {{
                mSpeed.textContent = speed.toFixed(1) + 'x';
                mSpeed.style.color = speed > 1 ? '#64dd17' : speed > 0.7 ? '#e8a317' : '#EF5552';
            }}
            if (bar) {{
                const pct = Math.min(speed / 10 * 100, 100);
                bar.style.width = pct + '%';
                bar.style.background = speed > 1 ? 'linear-gradient(90deg,#4A90D9,#64dd17)' : 'linear-gradient(90deg,#EF5552,#f87171)';
            }}
            if (barLabel) barLabel.textContent = speed.toFixed(1) + 'x realtime';
        }}
        if (elapsedEl) {{
            elapsedEl.style.display = 'block';
            elapsedEl.textContent = fin ? 'Completed in ' + elapsed.toFixed(1) + 's  (' + st.chunks + ' chunks)' : elapsed.toFixed(1) + 's elapsed  (' + st.chunks + ' chunks)';
        }}
    }}

    window.ttsStop = function() {{
        if (abort) abort.abort();
        if (node) node.port.postMessage({{ type: 'clear' }});
        gen = false;
        setStatus('Stopped', '#999');
        const btn = document.getElementById('tts-stop-btn');
        if (btn) btn.style.display = 'none';
    }};

    window.ttsGenerate = async function(payload) {{
        try {{ await init(); if (ctx.state === 'suspended') await ctx.resume(); }}
        catch (e) {{ setStatus('Audio init error: ' + e.message, '#EF5552'); return; }}
        if (abort) abort.abort();
        node.port.postMessage({{ type: 'clear' }});
        await new Promise(r => setTimeout(r, 50));
        node.port.postMessage({{ type: 'clear' }});

        gen = true;
        st = {{ t0: null, chunks: 0, samples: 0, ttfp: null }};
        setStatus('Connecting...', '#4A90D9');
        const bEl = document.getElementById('tts-stop-btn');
        if (bEl) bEl.style.display = 'inline-block';
        const mp = document.getElementById('tts-metrics');
        if (mp) {{ mp.style.display = 'grid'; ['tts-m-ttfp','tts-m-rtf','tts-m-dur','tts-m-speed'].forEach(id => {{ const e = document.getElementById(id); if(e) {{ e.textContent = '\\u2014'; e.style.color = '#333'; }} }}); }}
        const bw = document.getElementById('tts-rtf-bar-wrap');
        if (bw) bw.style.display = 'block';
        const bar = document.getElementById('tts-rtf-bar');
        if (bar) bar.style.width = '0%';
        const bl = document.getElementById('tts-rtf-label');
        if (bl) bl.textContent = '';
        const ee = document.getElementById('tts-elapsed');
        if (ee) {{ ee.style.display = 'none'; ee.textContent = ''; }}
        abort = new AbortController();

        try {{
            st.t0 = performance.now();
            const r = await fetch('/proxy/v1/audio/speech', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload),
                signal: abort.signal,
            }});
            if (!r.ok) {{ const t = await r.text(); throw new Error('Server ' + r.status + ': ' + t.slice(0, 200)); }}
            setStatus('Streaming...', '#4A90D9');
            const reader = r.body.getReader();
            let left = new Uint8Array(0);
            while (true) {{
                const {{ done, value }} = await reader.read();
                if (done) break;
                let raw;
                if (left.length > 0) {{
                    raw = new Uint8Array(left.length + value.length);
                    raw.set(left); raw.set(value, left.length);
                }} else {{ raw = value; }}
                const usable = raw.length - (raw.length % 2);
                left = usable < raw.length ? raw.slice(usable) : new Uint8Array(0);
                if (usable > 0) {{
                    const ab = new ArrayBuffer(usable);
                    new Uint8Array(ab).set(raw.subarray(0, usable));
                    const pcm = new Int16Array(ab);
                    node.port.postMessage(pcm);
                    st.chunks++;
                    st.samples += pcm.length;
                    if (st.ttfp == null) st.ttfp = performance.now() - st.t0;
                    showStats(false);
                }}
            }}
        }} catch (e) {{
            if (e.name !== 'AbortError') {{
                setStatus('Error: ' + e.message, '#EF5552');
                console.error('TTS error:', e);
            }}
        }} finally {{
            st.streamEnd = performance.now();
            showStats(true);
            gen = false;
            if (st.samples > 0) setStatus('Finishing playback...', '#64dd17');
            else {{
                setStatus('No audio received', '#999');
                if (bEl) bEl.style.display = 'none';
            }}
        }}
    }};
    </script>
"""


def _encode_audio(audio_data: tuple) -> str:
    sr, audio_np = audio_data
    if audio_np.dtype in (np.float32, np.float64):
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_np = (audio_np * 32767).astype(np.int16)
    elif audio_np.dtype != np.int16:
        audio_np = audio_np.astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV")
    return f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}"


def create_app(api_base: str):
    app = FastAPI()
    _pending: dict[str, dict] = {}

    @app.post("/proxy/v1/audio/speech")
    async def proxy_speech(request: Request):
        body = await request.json()
        req_id = body.get("_req_id")
        if req_id and req_id in _pending:
            body = _pending.pop(req_id)
        logger.info("Proxy: %s", {k: (f"<{len(str(v))} chars>" if k == "ref_audio" else v) for k, v in body.items()})
        try:
            client = httpx.AsyncClient(timeout=300)
            resp = await client.send(
                client.build_request(
                    "POST",
                    f"{api_base}/v1/audio/speech",
                    json=body,
                    headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                ),
                stream=True,
            )
        except Exception as exc:
            logger.exception("Proxy connection error")
            await client.aclose()
            return Response(content=str(exc), status_code=502)
        if resp.status_code != 200:
            content = await resp.aread()
            await resp.aclose()
            await client.aclose()
            return Response(content=content, status_code=resp.status_code)

        async def relay():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(relay(), media_type="application/octet-stream")

    css = """
    #generate-btn button { width: 100%; }
    #streaming-player { border: 1px solid var(--border-color-primary) !important; border-radius: var(--block-radius) !important; padding: var(--block-padding) !important; }
    """
    theme = gr.themes.Default(
        primary_hue=gr.themes.Color(
            c50="#f0f5ff",
            c100="#dce6f9",
            c200="#b8cef3",
            c300="#8eb2eb",
            c400="#6496e0",
            c500="#4A90D9",
            c600="#3a7bc8",
            c700="#2d66b0",
            c800="#1f4f8f",
            c900="#163a6e",
            c950="#0e2650",
        ),
    )

    with gr.Blocks(title="VoxCPM2 TTS Demo") as demo:
        gr.HTML(f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
          <img src="https://raw.githubusercontent.com/vllm-project/vllm-omni/main/docs/source/logos/vllm-omni-logo.png"
               alt="vLLM-Omni" style="height:42px;">
          <div>
            <h1 style="margin:0; font-size:1.5em;">VoxCPM2 Streaming Demo</h1>
            <span style="font-size:0.85em; color:#666;">
              Served by <a href="https://github.com/vllm-project/vllm-omni" target="_blank"
              style="color:#4A90D9; text-decoration:none; font-weight:600;">vLLM-Omni</a>
              &middot; <code style="background:#eef2f7; padding:2px 6px; border-radius:4px; font-size:0.9em;">{api_base}</code>
              &middot; 48 kHz
            </span>
          </div>
        </div>
        """)

        gr.Markdown(
            "**Three modes:** "
            "**Voice Design** (control instruction only) &middot; "
            "**Controllable Cloning** (ref audio + optional style control) &middot; "
            "**Ultimate Cloning** (ref audio + transcript for audio continuation)"
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Target Text",
                    placeholder="Enter text to synthesize...",
                    lines=4,
                )
                control_instruction = gr.Textbox(
                    label="Control Instruction (optional)",
                    placeholder="e.g. A warm young woman / Excited and fast-paced",
                    lines=2,
                    info="Describe voice style, emotion, pace. Works for both Voice Design and Controllable Cloning.",
                )

                with gr.Accordion("Voice Cloning", open=False):
                    ref_audio = gr.Audio(
                        label="Reference Audio (upload for cloning)",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                    ref_audio_url = gr.Textbox(
                        label="or Reference Audio URL",
                        placeholder="https://example.com/reference.wav",
                    )
                    ultimate_clone = gr.Checkbox(
                        label="Ultimate Cloning Mode",
                        value=False,
                        info="Provide transcript of ref audio for audio continuation (disables control instruction)",
                    )
                    prompt_text = gr.Textbox(
                        label="Reference Audio Transcript",
                        placeholder="Transcript of your reference audio (for ultimate cloning)",
                        lines=2,
                        visible=False,
                    )

                with gr.Row():
                    stream_checkbox = gr.Checkbox(
                        label="Stream (gapless)",
                        value=True,
                        info="AudioWorklet streaming",
                    )
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate Speech",
                        variant="primary",
                        size="lg",
                        elem_id="generate-btn",
                        scale=3,
                    )
                    reset_btn = gr.Button("Reset", variant="secondary", size="lg", scale=1)

            with gr.Column(scale=2):
                player_html = gr.HTML(
                    value=PLAYER_HTML,
                    visible=True,
                    label="streaming player",
                    elem_id="streaming-player",
                )
                audio_output = gr.Audio(
                    label="generated audio",
                    interactive=False,
                    autoplay=True,
                    visible=False,
                )
                gr.Examples(
                    examples=[
                        ["Hello, this is a VoxCPM2 demo running on vLLM-Omni.", ""],
                        [
                            "I have a dream that my four little children will one day live in a nation "
                            "where they will not be judged by the color of their skin but by the content "
                            "of their character.",
                            "",
                        ],
                        [
                            "I never asked you to stay. It's not like I care or anything. "
                            "But why does it still hurt so much now that you're gone?",
                            "A young girl with a soft, sweet voice. Speaks slowly with a melancholic tone.",
                        ],
                    ],
                    inputs=[text_input, control_instruction],
                    label="examples",
                )
                gr.HTML("""
                <div style="text-align:center; padding:8px 0; margin-top:4px;">
                  <a href="https://github.com/vllm-project/vllm-omni" target="_blank">
                    <img src="https://raw.githubusercontent.com/vllm-project/vllm-omni/main/docs/source/logos/vllm-omni-logo.png"
                         alt="vLLM-Omni" style="height:28px; opacity:0.7;">
                  </a>
                </div>
                """)

        hidden_payload = gr.Textbox(visible=False, elem_id="tts-payload")

        def on_ultimate_toggle(checked):
            return (
                gr.update(visible=checked),  # prompt_text
                gr.update(interactive=not checked),  # control_instruction
            )

        ultimate_clone.change(
            fn=on_ultimate_toggle,
            inputs=[ultimate_clone],
            outputs=[prompt_text, control_instruction],
        )

        def on_stream_change(stream: bool):
            if stream:
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        stream_checkbox.change(
            fn=on_stream_change,
            inputs=[stream_checkbox],
            outputs=[player_html, audio_output],
        )

        def on_reset():
            return "", "", None, "", False, "", PLAYER_HTML

        reset_btn.click(
            fn=on_reset,
            outputs=[
                text_input,
                control_instruction,
                audio_output,
                hidden_payload,
                ultimate_clone,
                prompt_text,
                player_html,
            ],
            js="() => { if (window.ttsStop) window.ttsStop(); }",
        )

        def on_generate(stream_enabled, text, ctrl_instr, ref_a, ref_url, ult_clone, p_text):
            import time as _time

            if not text or not text.strip():
                raise gr.Error("Please enter text to synthesize.")

            # VoxCPM2 uses "(instruction)text" format for control
            ctrl = ctrl_instr.strip() if ctrl_instr and not ult_clone else ""
            final_text = f"({ctrl}){text.strip()}" if ctrl else text.strip()

            payload: dict = {
                "input": final_text,
                "voice": "default",
                "response_format": "pcm" if stream_enabled else "wav",
                "stream": stream_enabled,
            }

            # Reference audio for cloning
            ref_url_s = ref_url.strip() if ref_url else ""
            if ref_url_s:
                payload["ref_audio"] = ref_url_s
            elif ref_a is not None:
                payload["ref_audio"] = _encode_audio(ref_a)

            # Ultimate cloning: prompt_audio + prompt_text for continuation
            if ult_clone and p_text and p_text.strip():
                if ref_url_s:
                    payload["prompt_audio"] = ref_url_s
                elif ref_a is not None:
                    payload["prompt_audio"] = payload.get("ref_audio", "")
                payload["prompt_text"] = p_text.strip()

            if stream_enabled:
                if ref_a is not None and not ref_url_s:
                    req_id = f"req-{int(_time.time() * 1000)}"
                    _pending[req_id] = payload
                    browser_payload = {"_req_id": req_id, "_nonce": int(_time.time() * 1000)}
                    return json.dumps(browser_payload), gr.update()
                payload["_nonce"] = int(_time.time() * 1000)
                return json.dumps(payload), gr.update()
            else:
                try:
                    with httpx.Client(timeout=300.0) as client:
                        resp = client.post(
                            f"{api_base}/v1/audio/speech",
                            json=payload,
                            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
                        )
                except httpx.ConnectError:
                    raise gr.Error(f"Cannot connect to server at {api_base}.")
                if resp.status_code != 200:
                    raise gr.Error(f"Server error ({resp.status_code}): {resp.text[:200]}")
                audio_np, sr = sf.read(io.BytesIO(resp.content))
                if audio_np.ndim > 1:
                    audio_np = audio_np[:, 0]
                return "", (sr, audio_np.astype(np.float32))

        generate_btn.click(
            fn=on_generate,
            inputs=[
                stream_checkbox,
                text_input,
                control_instruction,
                ref_audio,
                ref_audio_url,
                ultimate_clone,
                prompt_text,
            ],
            outputs=[hidden_payload, audio_output],
        ).then(
            fn=lambda p: p,
            inputs=[hidden_payload],
            outputs=[hidden_payload],
            js="(p) => { if (p && p.trim()) { const d = JSON.parse(p); delete d._nonce; window.ttsGenerate(d); } return p; }",
        )

        demo.queue()

    return gr.mount_gradio_app(app, demo, path="/", css=css, theme=theme, head=_build_player_js())


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 streaming Gradio demo")
    parser.add_argument("--api-base", default="http://localhost:8000", help="vLLM API server URL")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Connecting to vLLM server at: {args.api_base}")

    import uvicorn

    uvicorn.run(create_app(args.api_base), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
