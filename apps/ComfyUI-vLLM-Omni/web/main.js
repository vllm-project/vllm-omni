// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/**
 * vLLM-Omni ComfyUI frontend extension.
 *
 * Automatically detects available models from the configured vLLM-Omni
 * server and replaces the manual model text input with a dropdown.
 *
 * Behaviour:
 *  - On node creation the extension fetches ``/vllm_omni/models`` with the
 *    current ``url`` widget value.
 *  - When the ``url`` widget changes the model list is re-fetched (debounced).
 *  - A "Refresh Models" right-click menu entry allows manual refresh.
 *  - If the server is unreachable the widget stays as a plain text input so
 *    users can still type a model name manually.
 */

import { app } from "../../scripts/app.js";

/** Node class names that have ``url`` + ``model`` widgets. */
const VLLM_NODE_NAMES = new Set([
    "VLLMOmniGenerateImage",
    "VLLMOmniGenerateVideo",
    "VLLMOmniUnderstanding",
    "VLLMOmniTTS",
    "VLLMOmniVoiceClone",
]);

// ---------------------------------------------------------------------------
// Model list cache (per URL, with TTL)
// ---------------------------------------------------------------------------

const modelCache = new Map();
const CACHE_TTL_MS = 30_000;

/**
 * Fetch available model IDs from the ComfyUI proxy route.
 * Returns ``null`` on any failure so the caller can fall back gracefully.
 */
async function fetchModels(url) {
    const now = Date.now();
    const cached = modelCache.get(url);
    if (cached && now - cached.ts < CACHE_TTL_MS) {
        return cached.models;
    }

    try {
        const resp = await fetch(
            `/vllm_omni/models?url=${encodeURIComponent(url)}`
        );
        if (!resp.ok) {
            return null;
        }
        const data = await resp.json();
        const models = data.models || [];
        if (models.length > 0) {
            modelCache.set(url, { models, ts: now });
        }
        return models;
    } catch (_err) {
        return null;
    }
}

// ---------------------------------------------------------------------------
// Widget helpers
// ---------------------------------------------------------------------------

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

/**
 * Convert the ``model`` STRING widget into a combo dropdown populated with
 * the models served by the vLLM-Omni instance at the current ``url``.
 *
 * If models cannot be fetched the widget is left unchanged (still editable
 * as a plain text field).
 */
async function updateModelWidget(node) {
    const urlWidget = findWidget(node, "url");
    const modelWidget = findWidget(node, "model");
    if (!urlWidget || !modelWidget) {
        return;
    }

    const url = urlWidget.value;
    if (!url) {
        return;
    }

    const models = await fetchModels(url);
    if (!models || models.length === 0) {
        return;
    }

    const currentValue = modelWidget.value;

    // Skip update if options haven't changed.
    if (modelWidget.type === "combo" && modelWidget.options?.values) {
        if (JSON.stringify(modelWidget.options.values) === JSON.stringify(models)) {
            return;
        }
    }

    // Convert STRING → combo.
    modelWidget.type = "combo";
    modelWidget.options = modelWidget.options || {};
    modelWidget.options.values = models;

    // Preserve current selection when possible.
    modelWidget.value = models.includes(currentValue)
        ? currentValue
        : models[0];

    // Redraw the node to reflect the new widget type.
    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "vllm.vllm_omni",

    async nodeCreated(node) {
        if (!VLLM_NODE_NAMES.has(node.comfyClass)) {
            return;
        }

        // Slight delay so that widget initialisation is complete.
        setTimeout(() => updateModelWidget(node), 500);

        // Re-fetch when the URL widget value changes (debounced).
        const urlWidget = findWidget(node, "url");
        if (urlWidget) {
            const origCallback = urlWidget.callback;
            urlWidget.callback = function (...args) {
                if (origCallback) {
                    origCallback.apply(this, args);
                }
                clearTimeout(node._vllmModelTimeout);
                node._vllmModelTimeout = setTimeout(
                    () => updateModelWidget(node),
                    800
                );
            };
        }

        // Right-click context menu entry for manual refresh.
        const origGetExtraMenuOptions = node.getExtraMenuOptions;
        node.getExtraMenuOptions = function (_, options) {
            if (origGetExtraMenuOptions) {
                origGetExtraMenuOptions.apply(this, arguments);
            }
            options.unshift({
                content: "🔄 Refresh Models",
                callback: () => {
                    const currentUrl = findWidget(node, "url")?.value;
                    if (currentUrl) {
                        modelCache.delete(currentUrl);
                    }
                    updateModelWidget(node);
                },
            });
        };
    },

    async setup() {
        console.info("vLLM-Omni: Model auto-detect extension loaded.");
    },
});
