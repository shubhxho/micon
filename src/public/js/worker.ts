// Web Worker: WASM-accelerated volume processing
// Falls back to JavaScript if WASM module is not available
// Normalizes Int16/Uint16 → Float32, applies ellipsoid brain mask

declare const self: DedicatedWorkerGlobalScope;

// ── WASM Module Loader ──────────────────────────────────────────────────────

interface VolumeWasm {
  memory: WebAssembly.Memory;
  processVolume(
    inPtr: number, outPtr: number, count: number,
    sizeX: number, sizeY: number, sizeZ: number, doBlur: number
  ): bigint;
  findMinMax(ptr: number, count: number): bigint;
  normalize(inPtr: number, outPtr: number, count: number, min: number, max: number): void;
  applyEllipsoidMask(dataPtr: number, sizeX: number, sizeY: number, sizeZ: number): void;
  computeHistogram(dataPtr: number, count: number, histPtr: number): void;
}

let wasmModule: VolumeWasm | null = null;
let wasmReady = false;

async function loadWasm(): Promise<boolean> {
  try {
    const response = await fetch("/public/wasm/volume.wasm");
    if (!response.ok) return false;

    const bytes = await response.arrayBuffer();
    const memory = new WebAssembly.Memory({ initial: 512, maximum: 2048 }); // 32MB–128MB

    const { instance } = await WebAssembly.instantiate(bytes, {
      env: {
        memory,
        abort: () => { throw new Error("WASM abort"); },
        "Math.exp": Math.exp,
      },
    });

    wasmModule = {
      memory: instance.exports.memory as WebAssembly.Memory || memory,
      processVolume: instance.exports.processVolume as any,
      findMinMax: instance.exports.findMinMax as any,
      normalize: instance.exports.normalize as any,
      applyEllipsoidMask: instance.exports.applyEllipsoidMask as any,
      computeHistogram: instance.exports.computeHistogram as any,
    };

    wasmReady = true;
    console.log("[Worker] WASM volume processor loaded");
    return true;
  } catch (e) {
    console.warn("[Worker] WASM not available, using JS fallback:", e);
    return false;
  }
}

// Try loading WASM on worker init
loadWasm();

// ── WASM-accelerated processing ─────────────────────────────────────────────

function processWithWasm(
  input: Int16Array,
  sizeX: number,
  sizeY: number,
  sizeZ: number
): { data: Float32Array; min: number; max: number } {
  const wasm = wasmModule!;
  const N = sizeX * sizeY * sizeZ;

  // Ensure WASM memory is large enough
  // Layout: [Int16 input | Float32 output | histogram]
  const inputBytes = N * 2;
  const outputBytes = N * 4;
  const histBytes = 256 * 4;
  const totalBytes = inputBytes + outputBytes + histBytes;
  const pagesNeeded = Math.ceil(totalBytes / 65536);

  const currentPages = wasm.memory.buffer.byteLength / 65536;
  if (pagesNeeded > currentPages) {
    wasm.memory.grow(pagesNeeded - currentPages + 16);
  }

  const inPtr = 0;
  const outPtr = inputBytes;

  // Copy input data to WASM memory
  const wasmInput = new Int16Array(wasm.memory.buffer, inPtr, N);
  wasmInput.set(input);

  // Run full pipeline in WASM
  const minMax = wasm.processVolume(inPtr, outPtr, N, sizeX, sizeY, sizeZ, 0);
  const min = Number(BigInt.asIntN(32, minMax & BigInt(0xFFFFFFFF)));
  const max = Number(BigInt.asIntN(32, minMax >> BigInt(32)));

  // Copy output from WASM memory
  const wasmOutput = new Float32Array(wasm.memory.buffer, outPtr, N);
  const result = new Float32Array(N);
  result.set(wasmOutput);

  return { data: result, min, max };
}

// ── JavaScript fallback processing ──────────────────────────────────────────

function processWithJS(
  input: Int16Array,
  sizeX: number,
  sizeY: number,
  sizeZ: number
): { data: Float32Array; min: number; max: number } {
  const N = sizeX * sizeY * sizeZ;
  const out = new Float32Array(N);

  // Find min/max
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < N; i++) {
    const v = input[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }

  const range = max - min || 1;

  // Normalize to [0, 1]
  for (let i = 0; i < N; i++) {
    out[i] = (input[i] - min) / range;
  }

  // Apply soft ellipsoid brain mask
  const cx = sizeX / 2;
  const cy = sizeY / 2;
  const cz = sizeZ / 2;
  const rx = sizeX * 0.45;
  const ry = sizeY * 0.45;
  const rz = sizeZ * 0.45;

  for (let z = 0; z < sizeZ; z++) {
    for (let y = 0; y < sizeY; y++) {
      for (let x = 0; x < sizeX; x++) {
        const idx = z * sizeY * sizeX + y * sizeX + x;
        const dx = (x - cx) / rx;
        const dy = (y - cy) / ry;
        const dz = (z - cz) / rz;
        const dist = dx * dx + dy * dy + dz * dz;

        if (dist > 1.0) {
          const falloff = Math.max(0, 1.0 - (dist - 1.0) * 3.0);
          out[idx] *= falloff;
        }
      }
    }
  }

  return { data: out, min, max };
}

// ── Message handler ─────────────────────────────────────────────────────────

self.onmessage = ({ data }) => {
  if (data.type === "VOLUME") {
    const { buffer, sizeX, sizeY, sizeZ } = data;
    const input = new Int16Array(buffer);

    const startTime = performance.now();

    let result: { data: Float32Array; min: number; max: number };

    if (wasmReady && wasmModule) {
      console.log("[Worker] Processing volume with WASM...");
      result = processWithWasm(input, sizeX, sizeY, sizeZ);
    } else {
      console.log("[Worker] Processing volume with JS fallback...");
      result = processWithJS(input, sizeX, sizeY, sizeZ);
    }

    const elapsed = performance.now() - startTime;
    console.log(`[Worker] Volume processed in ${elapsed.toFixed(1)}ms (${wasmReady ? "WASM" : "JS"})`);

    self.postMessage(
      {
        type: "VOLUME_READY",
        data: result.data,
        sizeX,
        sizeY,
        sizeZ,
        min: result.min,
        max: result.max,
        processingTime: elapsed,
        usedWasm: wasmReady,
      },
      [result.data.buffer] as any
    );
  }
};
