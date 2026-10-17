/**
 * @file Front-end extension for the vLLM-Omni nodes.
 *
 * Draws the vLLM-Omni mark in place of the default title dot.
 * Node colours follow the active ComfyUI theme.
 *
 * It also migrates Generate Video graphs saved before `duration` replaced the
 * `num_frames` widget. See migrateNumFramesToDuration below.
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
