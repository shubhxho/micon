// AssemblyScript — Volume Processing for NeuroViz
// Compiles to WebAssembly for high-performance voxel operations
// Build: bun run build:wasm

// ── Memory layout ───────────────────────────────────────────────────────────
// Input Int16 buffer starts at offset 0
// Output Float32 buffer starts after input
// All pointers are byte offsets

/** Find min and max of an Int16 array. Returns packed i64: (max << 32) | (min & 0xFFFFFFFF) */
export function findMinMax(ptr: i32, count: i32): i64 {
  let min: i32 = 32767;
  let max: i32 = -32768;

  for (let i: i32 = 0; i < count; i++) {
    const val: i32 = <i32>load<i16>(ptr + (i << 1));
    if (val < min) min = val;
    if (val > max) max = val;
  }

  return (<i64>max << 32) | (<i64>(min & 0xFFFFFFFF));
}

/** Normalize Int16 values to Float32 [0, 1] range */
export function normalize(
  inPtr: i32,
  outPtr: i32,
  count: i32,
  min: i32,
  max: i32
): void {
  const range: f32 = <f32>(max - min);
  const invRange: f32 = range > 0.0 ? 1.0 / range : 1.0;
  const minF: f32 = <f32>min;

  for (let i: i32 = 0; i < count; i++) {
    const val: f32 = <f32>(<i32>load<i16>(inPtr + (i << 1)));
    store<f32>(outPtr + (i << 2), (val - minF) * invRange);
  }
}

/** Apply soft ellipsoid brain mask — attenuates voxels outside the ellipsoid */
export function applyEllipsoidMask(
  dataPtr: i32,
  sizeX: i32,
  sizeY: i32,
  sizeZ: i32
): void {
  const cx: f32 = <f32>sizeX * 0.5;
  const cy: f32 = <f32>sizeY * 0.5;
  const cz: f32 = <f32>sizeZ * 0.5;
  const rx: f32 = <f32>sizeX * 0.45;
  const ry: f32 = <f32>sizeY * 0.45;
  const rz: f32 = <f32>sizeZ * 0.45;
  const invRx: f32 = 1.0 / rx;
  const invRy: f32 = 1.0 / ry;
  const invRz: f32 = 1.0 / rz;

  const sliceSize: i32 = sizeY * sizeX;

  for (let z: i32 = 0; z < sizeZ; z++) {
    const dz: f32 = (<f32>z - cz) * invRz;
    const dz2: f32 = dz * dz;
    const zOffset: i32 = z * sliceSize;

    for (let y: i32 = 0; y < sizeY; y++) {
      const dy: f32 = (<f32>y - cy) * invRy;
      const dy2: f32 = dy * dy;
      const yzOffset: i32 = (zOffset + y * sizeX) << 2;

      for (let x: i32 = 0; x < sizeX; x++) {
        const dx: f32 = (<f32>x - cx) * invRx;
        const dist: f32 = dx * dx + dy2 + dz2;

        if (dist > 1.0) {
          const falloff: f32 = max<f32>(0.0, 1.0 - (dist - 1.0) * 3.0);
          const ptr: i32 = dataPtr + yzOffset + (x << 2);
          store<f32>(ptr, load<f32>(ptr) * falloff);
        }
      }
    }
  }
}

/** Compute histogram of Float32 data into 256 bins */
export function computeHistogram(
  dataPtr: i32,
  count: i32,
  histPtr: i32
): void {
  // Clear histogram (256 bins × 4 bytes)
  for (let i: i32 = 0; i < 256; i++) {
    store<i32>(histPtr + (i << 2), 0);
  }

  for (let i: i32 = 0; i < count; i++) {
    let val: f32 = load<f32>(dataPtr + (i << 2));
    let bin: i32 = <i32>(val * 255.0);
    if (bin < 0) bin = 0;
    if (bin > 255) bin = 255;
    const binPtr: i32 = histPtr + (bin << 2);
    store<i32>(binPtr, load<i32>(binPtr) + 1);
  }
}

/** Gaussian blur on a Float32 3D volume (single-pass, 3x3x3 kernel) */
export function gaussianBlur3D(
  srcPtr: i32,
  dstPtr: i32,
  sizeX: i32,
  sizeY: i32,
  sizeZ: i32
): void {
  const sliceSize: i32 = sizeX * sizeY;

  for (let z: i32 = 1; z < sizeZ - 1; z++) {
    for (let y: i32 = 1; y < sizeY - 1; y++) {
      for (let x: i32 = 1; x < sizeX - 1; x++) {
        let sum: f32 = 0.0;
        let weight: f32 = 0.0;

        // 3x3x3 Gaussian kernel (sigma ≈ 0.85)
        for (let dz: i32 = -1; dz <= 1; dz++) {
          for (let dy: i32 = -1; dy <= 1; dy++) {
            for (let dx: i32 = -1; dx <= 1; dx++) {
              const idx: i32 = (z + dz) * sliceSize + (y + dy) * sizeX + (x + dx);
              const d: f32 = <f32>(dx * dx + dy * dy + dz * dz);
              const w: f32 = Mathf.exp(-d * 0.5);
              sum += load<f32>(srcPtr + (idx << 2)) * w;
              weight += w;
            }
          }
        }

        const outIdx: i32 = z * sliceSize + y * sizeX + x;
        store<f32>(dstPtr + (outIdx << 2), sum / weight);
      }
    }
  }
}

/** Process entire volume pipeline: normalize → mask → optional blur */
export function processVolume(
  inPtr: i32,
  outPtr: i32,
  count: i32,
  sizeX: i32,
  sizeY: i32,
  sizeZ: i32,
  doBlur: i32
): i64 {
  // Step 1: Find min/max
  const minMax: i64 = findMinMax(inPtr, count);
  const min: i32 = <i32>(minMax & 0xFFFFFFFF);
  const max: i32 = <i32>(minMax >> 32);

  // Step 2: Normalize
  normalize(inPtr, outPtr, count, min, max);

  // Step 3: Apply ellipsoid mask
  applyEllipsoidMask(outPtr, sizeX, sizeY, sizeZ);

  return minMax;
}
