/**
 * @file Front-end extension for the vLLM-Omni nodes.
 *
 * Draws the vLLM-Omni mark in place of the default title dot.
 * Node colours follow the active ComfyUI theme.
 *
 * It also migrates Generate Video graphs saved before `duration` replaced the
 * `num_frames` widget. See migrateNumFramesToDuration below.
 *
 * And it upgrades the `model` field of every node that talks to a server into a
 * dropdown fed with the server's model list. See setupModelCombo below.
 *
 * This file also used to host dynamic fields on vLLM-Omni nodes, driven by
 * widget (in-node form fields) and input (connection link) values and changes.
 * That is deliberately absent: it introduced too much complexity, and it even
 * conflicted with the current backend (Python) validation for unknown reasons
 * (pending ComfyUI upstream fixes).
 */

import { app } from "../../scripts/app.js";

const MARK_URL = new URL("./vllm-omni-mark.svg", import.meta.url).href;
const MARK_ASPECT = 157 / 165;

/**
 * Same-origin proxy route served by the ComfyUI backend (see
 * comfyui_vllm_omni/routes.py). The browser must not fetch `{url}/models`
 * itself: for remote deployments a browser-side `localhost` resolves to the
 * user's own machine, not to the host running ComfyUI.
 */
const MODELS_ROUTE = "/vllm_omni/models";

/** Wait this long after the last `url` edit before refetching the model list. */
const URL_CHANGE_DEBOUNCE_MS = 600;

/**
 * Merge the fetched model ids with the widget's current value, so a value a
 * saved workflow carried in is never dropped just because the server does not
 * list it (renamed deployments, stale graphs, mid-edit values).
 */
function mergeModelValues(models, currentValue) {
    const values = models.map((id) => String(id));
    const current = String(currentValue ?? "").trim();
    if (current && !values.includes(current)) {
        values.push(current);
    }
    return values;
}

/**
 * Turns a node's free-text `model` widget into a dropdown of whatever the
 * server in its `url` widget currently serves.
 *
 * The widget stays a plain text field until a fetch succeeds, and returns to
 * one whenever a fetch fails: the dropdown is a convenience, and a server that
 * is down (or not started yet) must never make the node unfillable. `url`
 * edits refetch (debounced), the refresh button refetches immediately, and the
 * button is `serialize = false` so workflows saved before it existed still
 * restore their widget values positionally.
 */
class ModelComboController {
    constructor(node) {
        this.node = node;
        this.urlWidget = node.widgets.find((widget) => widget.name === "url");
        this.modelWidget = node.widgets.find((widget) => widget.name === "model");
        // Out-of-order guard: only the latest fetch may touch the widget.
        this.fetchTicket = 0;
        this.debounceTimer = null;
    }

    attach() {
        const originalUrlCallback = this.urlWidget.callback;
        this.urlWidget.callback = (...args) => {
            originalUrlCallback?.apply(this.urlWidget, args);
            clearTimeout(this.debounceTimer);
            this.debounceTimer = setTimeout(() => this.refreshModels(), URL_CHANGE_DEBOUNCE_MS);
        };

        const refreshButton = this.node.addWidget("button", "↻ refresh models", null, () => this.refreshModels());
        refreshButton.serialize = false;
        // addWidget appends; move the button right after the model widget.
        const widgets = this.node.widgets;
        widgets.splice(widgets.indexOf(this.modelWidget) + 1, 0, widgets.pop());
        this.node.setSize([this.node.size[0], this.node.computeSize()[1]]);

        // Defer past onConfigure so a value restored from a saved workflow is
        // the one merged into the fetched list.
        setTimeout(() => this.refreshModels(), 0);
    }

    // New-style litegraph widgets are class instances (TextWidget/ComboWidget
    // with their own draw/mouse handlers), so flipping `widget.type` never
    // re-renders a text field as a dropdown. Swap in a freshly created widget
    // of the right type instead, keeping name/value/position so workflows
    // serialize unchanged.
    _swapModelWidget(type, values) {
        const old = this.modelWidget;
        if (old.type === type) {
            if (type === "combo") {
                old.options.values = values;
            }
            return
        }
        const widgets = this.node.widgets;
        const index = widgets.indexOf(old);
        old.onRemove?.();
        widgets.splice(index, 1);
        const widget = this.node.addWidget(
            type,
            "model",
            old.value,
            old.callback?.bind(old),
            type === "combo" ? { values } : {},
        );
        // addWidget appends; move the new widget to the old one's slot.
        widgets.splice(widgets.indexOf(widget), 1);
        widgets.splice(index, 0, widget);
        this.modelWidget = widget;
        this.node.setSize([this.node.size[0], this.node.computeSize()[1]]);
        this.node.graph?.setDirtyCanvas?.(true, true);
    }

    setComboMode(values) {
        this._swapModelWidget("combo", values);
    }

    setTextMode() {
        this._swapModelWidget("text");
    }

    async refreshModels() {
        const ticket = ++this.fetchTicket;
        const url = String(this.urlWidget.value ?? "").trim();
        if (!url) {
            this.setTextMode();
            return
        }
        try {
            const response = await fetch(`${MODELS_ROUTE}?url=${encodeURIComponent(url)}`);
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !Array.isArray(data.models) || !data.models.length) {
                throw new Error(data.error ?? `HTTP ${response.status}`);
            }
            if (ticket === this.fetchTicket) {
                this.setComboMode(mergeModelValues(data.models, this.modelWidget.value));
            }
        } catch (e) {
            if (ticket === this.fetchTicket) {
                console.warn(`vLLM-Omni: could not fetch the model list from ${url}, model stays free-text:`, e);
                this.setTextMode();
            }
        }
    }
}

function setupModelCombo(node) {
    const urlWidget = node.widgets?.find((widget) => widget.name === "url");
    const modelWidget = node.widgets?.find((widget) => widget.name === "model");
    if (!urlWidget || !modelWidget) {
        return
    }
    new ModelComboController(node).attach();
}

/**
 * Rewrite a pre-rename Generate Video graph in place.
 *
 * `duration` (seconds) took over the slot `num_frames` used to occupy, and
 * ComfyUI restores widget values positionally, so a graph saved before the
 * rename reads its frame count as a duration: the workflow this repository
 * shipped stored 120 frames at 24 fps, which would come back as a 120-second
 * request. Saved files still name the old widget under `inputs`, and that is
 * what makes the rewrite both detectable and safe to apply exactly once -- once
 * rewritten the node serializes `duration`, so a later load finds nothing to do.
 */
function migrateNumFramesToDuration(node, info) {
    const savedBeforeRename = (info?.inputs ?? []).some((input) => input?.widget?.name === "num_frames");
    if (!savedBeforeRename) {
        return
    }
    const duration = node.widgets?.find((widget) => widget.name === "duration");
    const fps = node.widgets?.find((widget) => widget.name === "fps");
    if (!duration || !fps) {
        return
    }
    // configure() has already placed the stale frame count in the duration widget.
    const frames = Number(duration.value);
    const rate = Number(fps.value);
    if (!Number.isFinite(frames) || !Number.isFinite(rate) || rate <= 0) {
        return
    }
    const seconds = Math.max(duration.options?.min ?? 0.1, Math.round((frames / rate) * 1000) / 1000);
    console.info(
        `vLLM-Omni: migrated Generate Video "${node.title}" from ${frames} frames to ${seconds}s at ${rate} fps.`,
    );
    duration.value = seconds;
}

const mark = new Image();
let markReady = false;
mark.addEventListener("load", () => {
    markReady = true;
    // The image resolves after the first frames are already on screen.
    app.graph?.setDirtyCanvas?.(true, true);
});
mark.src = MARK_URL;

app.registerExtension({
    name: "vllm.vllm_omni",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name.startsWith("VLLMOmni")) {
            return
        }

        const inputs = { ...nodeData.input?.required, ...nodeData.input?.optional };
        if ("url" in inputs && "model" in inputs) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                setupModelCombo(this);
            };
        }

        if (nodeData.name === "VLLMOmniGenerateVideo") {
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, arguments);
                migrateNumFramesToDuration(this, info);
            };
        }

        // Defining this replaces the default dot outright — LiteGraph draws its
        // own square or circle only for nodes that leave onDrawTitleBox unset.
        nodeType.prototype.onDrawTitleBox = function (ctx, titleHeight, _size, scale) {
            // Zoomed out far enough that LiteGraph itself drops to flat shapes,
            // the mark is a few muddy pixels. Leave the title bar clean instead.
            if (!markReady || (scale !== undefined && scale < 0.5)) {
                return
            }
            const height = titleHeight * 0.62;
            const width = height * MARK_ASPECT;
            ctx.drawImage(
                mark,
                titleHeight * 0.5 - width * 0.5,
                titleHeight * -0.5 - height * 0.5,
                width,
                height,
            );
        };
    },
    async setup() {
        console.info("vLLM-Omni Setup complete!")
    },
})
