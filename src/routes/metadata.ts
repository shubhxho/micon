import { Hono } from "hono";

const app = new Hono();

app.get("/metadata/:sessionId", async (c) => {
  const sessionId = c.req.param("sessionId");
  const tmpDir = `/tmp/neuroviz-${sessionId}`;

  const script = `
import json, sys, os
import pydicom

d = sys.argv[1]
files = sorted([f for f in os.listdir(d) if f.lower().endswith('.dcm')])
results = []
for fname in files:
    ds = pydicom.dcmread(os.path.join(d, fname), force=True)
    row = {'filename': fname}
    tags = [
        'PatientID','SeriesDescription','Modality','Rows','Columns',
        'SliceThickness','PixelSpacing','ImagePositionPatient',
        'WindowCenter','WindowWidth','RescaleSlope','RescaleIntercept',
        'BitsAllocated','RepetitionTime','EchoTime'
    ]
    for tag in tags:
        try:
            row[tag] = str(getattr(ds, tag))
        except AttributeError:
            pass
    # Pixel stats
    try:
        import numpy as np
        arr = ds.pixel_array.astype(float)
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        arr = arr * slope + intercept
        row['pixel_min'] = float(arr.min())
        row['pixel_max'] = float(arr.max())
        row['pixel_mean'] = float(arr.mean())
        row['pixel_std'] = float(arr.std())
    except Exception:
        pass
    results.append(row)
print(json.dumps(results))
`;

  const scriptPath = `/tmp/neuroviz-metadata-${sessionId}.py`;
  await Bun.write(scriptPath, script);

  try {
    const proc = Bun.spawn(["uv", "run", "--with", "pydicom", "--with", "numpy", "python3", scriptPath, tmpDir], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const stdout = await new Response(proc.stdout).text();
    await proc.exited;

    try {
      return c.json(JSON.parse(stdout.trim()));
    } catch {
      return c.json({ error: "Failed to parse metadata" }, 500);
    }
  } catch (e) {
    return c.json({ error: String(e) }, 500);
  }
});

export const metadataRoute = app;
