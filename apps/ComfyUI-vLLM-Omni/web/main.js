/**
 * @file This file is intended to add dynamic fields to vLLM-Omni nodes
 * based on widget (in-node form fields) and input (connection link) values and changes.
 * However, this functionality is currently disabled/commented out.
 * Because it introduces too much complexity,
 * and it may even conflict with the current backend (Python) validation for unknown reasons (pending ComfyUI upstream fixes).
 */

import { app } from "../../scripts/app.js";
app.registerExtension({
    name: "vllm.vllm_omni",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData.name.startsWith("VLLMOmni")) {
            return
        }
        // Stub frontend plugin for now
    },
    async setup() {
        console.info("vLLM-Omni Setup complete!")
    },
})
