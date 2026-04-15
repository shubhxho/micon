// Client entry point — bundles Three.js, DWV, Tween.js into a single module
// Exposes Brain3D and DwvController to the global scope for Alpine.js

import { Brain3D } from "./brain3d";
import { DwvController } from "./dwv-init";

// Expose to window for Alpine
(window as any).Brain3D = Brain3D;
(window as any).DwvController = DwvController;

console.log("[NeuroViz] Client modules loaded");
