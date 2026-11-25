// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import { app } from "../../scripts/app.js";

let renderer;
function loadRenderer() {
    if (!renderer) {
        renderer = (async () => {
            const response = await fetch(new URL("./vendor/abcjs-basic-min.cjs", import.meta.url));
            if (!response.ok) throw new Error(`ABCJS: HTTP ${response.status}`);
            // Load the UMD bundle as a classic script, not a ComfyUI extension.
            const script = document.createElement("script");
            script.textContent = await response.text();
            document.head.append(script);
            script.remove();
            if (!window.ABCJS) throw new Error("ABCJS failed to load");
            return window.ABCJS;
        })().catch((error) => {
            renderer = null;
            throw error;
        });
    }
    return renderer;
}

async function renderScores(container, scores) {
    container.replaceChildren();
    if (!scores.some((abc) => abc.trim())) {
        container.textContent = "Run SheetSage2 or connect an ABC score to preview the sheet music.";
        return;
    }
    const status = document.createElement("div");
    status.textContent = "Rendering sheet music…";
    container.append(status);
    try {
        const abcjs = await loadRenderer();
        // A newer execution may have replaced this preview while loading.
        if (!container.contains(status)) return;
        status.remove();
        for (const abc of scores) {
            const paper = document.createElement("div");
            paper.className = "abc-sheet-paper";
            paper.style.cssText = "background:#fff;color:#111;min-width:300px;padding:8px;margin-bottom:12px;";
            container.append(paper);
            const tunes = abcjs.renderAbc(paper, abc, {
                responsive: "resize", staffwidth: 860, add_classes: true,
                wrap: { minSpacing: 1.8, maxSpacing: 2.7, preferredMeasuresPerLine: 4 },
            });
            if (!paper.querySelector("svg")) paper.textContent = "No music notation could be rendered. Check the ABC text below.";
            const warnings = tunes.flatMap((tune) => tune.warnings ?? []);
            if (warnings.length) {
                const warning = document.createElement("pre");
                warning.textContent = warnings.join("\n");
                warning.style.cssText = "white-space:pre-wrap;color:#8a4b00;font-size:12px;";
                paper.append(warning);
            }
        }
    } catch (error) {
        status.textContent = `Unable to render sheet music: ${error.message}`;
        container.append(status);
    }
}

function showLargePreview(scores) {
    const dialog = document.createElement("dialog");
    dialog.style.cssText = "width:min(1100px,92vw);height:88vh;padding:20px;border:1px solid #888;border-radius:8px;background:#fff;color:#111;";
    const close = document.createElement("button");
    close.textContent = "Close / 关闭";
    close.style.cssText = "position:sticky;top:0;float:right;z-index:1;padding:8px 16px;cursor:pointer;";
    close.onclick = () => dialog.close();
    const paper = document.createElement("div");
    paper.style.clear = "both";
    dialog.append(close, paper);
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    document.body.append(dialog);
    dialog.showModal();
    renderScores(paper, scores);
}

app.registerExtension({
    name: "local.abc_sheet_music_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ABCSheetMusicPreview") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const container = document.createElement("div");
            container.style.cssText = "overflow:auto;box-sizing:border-box;padding:8px;background:#f4f4f4;color:#111;border-radius:4px;height:100%;";
            container.addEventListener("wheel", (event) => event.stopPropagation());
            const button = document.createElement("button");
            button.textContent = "Enlarge sheet music / 放大五线谱";
            button.style.cssText = "padding:7px 12px;margin-bottom:8px;cursor:pointer;";
            button.onclick = () => showLargePreview(this.properties.abc_scores ?? []);
            const paper = document.createElement("div");
            const details = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = "ABC source / 原始乐谱文本";
            const source = document.createElement("pre");
            source.style.cssText = "white-space:pre-wrap;user-select:text;font-size:12px;";
            details.append(summary, source);
            container.append(button, paper, details);
            const widget = this.addDOMWidget("sheet_music", "ABC_SHEET", container, {
                serialize: false, hideOnZoom: false,
                getMinHeight: () => 300, getHeight: () => 440,
            });
            widget.serialize = false;
            this.updateScorePreview = () => {
                const scores = this.properties.abc_scores ?? [];
                source.textContent = scores.join("\n\n");
                renderScores(paper, scores);
            };
            this.setSize([560, 500]);
            this.updateScorePreview();
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            this.properties.abc_scores = message.abc ?? [];
            this.updateScorePreview();
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            configured?.apply(this, arguments);
            this.updateScorePreview();
        };
    },
});
