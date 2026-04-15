// DWV Integration — DICOM parsing and 2D MPR views
// Wires DWV events to Three.js Brain3D renderer

import { App as DwvApp, ViewConfig, ToolConfig } from "dwv";

export class DwvController {
  private dwvApp: DwvApp;
  private worker: Worker;
  private onVolumeReady: ((data: Float32Array, sx: number, sy: number, sz: number) => void) | null = null;
  private onProgress: ((pct: number) => void) | null = null;
  private onWindowLevelChange: ((center: number, width: number) => void) | null = null;

  constructor() {
    this.dwvApp = new DwvApp();

    // Initialize DWV with multi-planar view configs
    const viewConfigs: Record<string, ViewConfig[]> = {
      "*": [
        { divId: "dwv-axial" },
        { divId: "dwv-sagittal" },
        { divId: "dwv-coronal" },
      ],
    };

    const tools: Record<string, ToolConfig> = {
      Scroll: {},
      WindowLevel: {},
      ZoomAndPan: {},
    };

    this.dwvApp.init({
      dataViewConfigs: viewConfigs,
      tools: tools,
    });

    // Web worker for volume assembly
    this.worker = new Worker("/public/js/worker.js", { type: "module" });
    this.worker.onmessage = ({ data }) => {
      if (data.type === "VOLUME_READY" && this.onVolumeReady) {
        this.onVolumeReady(
          new Float32Array(data.data),
          data.sizeX,
          data.sizeY,
          data.sizeZ
        );
      }
    };

    this.setupEvents();
  }

  private setupEvents() {
    // Load complete — extract volume data
    this.dwvApp.addEventListener("load", (event: any) => {
      try {
        const dataIds = this.dwvApp.getDataIds();
        if (dataIds.length === 0) return;

        const dataId = dataIds[0];
        const image = this.dwvApp.getImage(dataId);
        const geometry = image.getGeometry();
        const size = geometry.getSize();
        const spacing = geometry.getSpacing();

        const sizeX = size.get(0);
        const sizeY = size.get(1);
        const sizeZ = size.get(2);

        console.log(`[DWV] Volume loaded: ${sizeX}×${sizeY}×${sizeZ}`);
        console.log(`[DWV] Spacing: ${spacing.get(0)}×${spacing.get(1)}×${spacing.get(2)}`);

        // Get raw pixel buffer
        const buffer = image.getBuffer();

        // Send to worker for normalization
        const bufferCopy = new Int16Array(buffer.length);
        for (let i = 0; i < buffer.length; i++) {
          bufferCopy[i] = buffer[i];
        }

        this.worker.postMessage(
          { type: "VOLUME", buffer: bufferCopy, sizeX, sizeY, sizeZ },
          [bufferCopy.buffer]
        );

        // Set scroll tool active
        this.dwvApp.setTool("Scroll");

        // Update status
        this.updateStatus("Volume loaded — use scroll/window-level tools");
      } catch (e) {
        console.error("[DWV] Error processing load event:", e);
      }
    });

    // Load progress
    this.dwvApp.addEventListener("loadprogress", (event: any) => {
      const pct = event.loaded !== undefined ? Math.round((event.loaded / (event.total || 100)) * 100) : 0;
      if (this.onProgress) {
        this.onProgress(pct);
      }
      this.updateProgress(pct);
    });

    // Load error
    this.dwvApp.addEventListener("error", (event: any) => {
      console.error("[DWV] Load error:", event);
      this.updateStatus("Error loading DICOM data");
    });

    // Window/level change
    this.dwvApp.addEventListener("wlchange", (event: any) => {
      if (event.value && this.onWindowLevelChange) {
        const center = event.value[0] !== undefined ? event.value[0] : event.value.center;
        const width = event.value[1] !== undefined ? event.value[1] : event.value.width;
        if (center !== undefined && width !== undefined) {
          this.onWindowLevelChange(center, width);
        }
      }
    });
  }

  private updateProgress(pct: number) {
    const bar = document.getElementById("dwv-progress-bar");
    const text = document.getElementById("dwv-progress-text");
    if (bar) {
      (bar as HTMLElement).style.width = `${pct}%`;
    }
    if (text) {
      text.textContent = `${pct}%`;
    }
  }

  private updateStatus(msg: string) {
    const el = document.getElementById("dwv-status");
    if (el) el.textContent = msg;
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  async loadFromUrls(urls: string[]) {
    try {
      this.updateStatus(`Loading ${urls.length} DICOM files...`);
      await this.dwvApp.loadURLs(urls);
    } catch (e) {
      console.error("[DWV] Failed to load URLs:", e);
      this.updateStatus("Failed to load DICOM data");
    }
  }

  async loadFromFiles(files: File[]) {
    try {
      this.updateStatus(`Loading ${files.length} DICOM files...`);
      await this.dwvApp.loadFiles(files);
    } catch (e) {
      console.error("[DWV] Failed to load files:", e);
      this.updateStatus("Failed to load DICOM data");
    }
  }

  setTool(name: string) {
    try {
      this.dwvApp.setTool(name);
    } catch (e) {
      console.warn(`[DWV] Tool "${name}" not available`);
    }
  }

  onVolume(cb: (data: Float32Array, sx: number, sy: number, sz: number) => void) {
    this.onVolumeReady = cb;
  }

  onProgressUpdate(cb: (pct: number) => void) {
    this.onProgress = cb;
  }

  onWLChange(cb: (center: number, width: number) => void) {
    this.onWindowLevelChange = cb;
  }

  getApp(): DwvApp {
    return this.dwvApp;
  }
}
