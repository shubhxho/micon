// Web Worker: Volume assembly from DWV pixel data
// Normalizes Int16/Uint16 → Float32, applies ellipsoid brain mask

declare const self: DedicatedWorkerGlobalScope;

self.onmessage = ({ data }) => {
  if (data.type === "VOLUME") {
    const { buffer, sizeX, sizeY, sizeZ } = data;
    const N = sizeX * sizeY * sizeZ;
    const input = new Int16Array(buffer);
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

    // Apply soft ellipsoid brain mask to reduce background noise
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
            // Soft falloff outside ellipsoid
            const falloff = Math.max(0, 1.0 - (dist - 1.0) * 3.0);
            out[idx] *= falloff;
          }
        }
      }
    }

    self.postMessage(
      { type: "VOLUME_READY", data: out, sizeX, sizeY, sizeZ, min, max },
      [out.buffer] as any
    );
  }
};
