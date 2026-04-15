import { Hono } from "hono";
import { pushProgress, closeProgress } from "../server";

const app = new Hono();

// Store uploaded files in memory for the session
const uploadStore = new Map<
  string,
  { files: { name: string; data: ArrayBuffer }[]; metadata: Record<string, string> }
>();

export function getUploadedFiles(sessionId: string) {
  return uploadStore.get(sessionId);
}

app.post("/upload", async (c) => {
  const body = await c.req.parseBody({ all: true });
  const sessionId =
    (body["sessionId"] as string) || crypto.randomUUID().slice(0, 8);

  let rawFiles = body["files"];
  if (!rawFiles) {
    return c.html(
      `<div id="metadata-panel" class="p-4 text-red-400">No files received.</div>`
    );
  }

  const fileList: File[] = Array.isArray(rawFiles)
    ? (rawFiles as File[])
    : [rawFiles as File];

  const dcmFiles = fileList.filter(
    (f) => f instanceof File && (f.name.endsWith(".dcm") || f.name.endsWith(".DCM"))
  );

  if (dcmFiles.length === 0) {
    return c.html(
      `<div id="metadata-panel" class="p-4 text-red-400">No .dcm files found in upload.</div>`
    );
  }

  pushProgress(sessionId, 5, `Receiving ${dcmFiles.length} DICOM files...`);

  // Save files to temp directory for pydicom processing
  const tmpDir = `/tmp/neuroviz-${sessionId}`;
  await Bun.spawn(["mkdir", "-p", tmpDir]).exited;

  const storedFiles: { name: string; data: ArrayBuffer }[] = [];

  for (let i = 0; i < dcmFiles.length; i++) {
    const f = dcmFiles[i];
    const data = await f.arrayBuffer();
    const filePath = `${tmpDir}/${f.name}`;
    await Bun.write(filePath, data);
    storedFiles.push({ name: f.name, data });

    const pct = 5 + Math.round((i / dcmFiles.length) * 40);
    pushProgress(sessionId, pct, `Saving ${f.name}...`);
  }

  uploadStore.set(sessionId, { files: storedFiles, metadata: {} });

  // Extract metadata via pydicom subprocess
  pushProgress(sessionId, 50, "Extracting DICOM metadata via pydicom...");

  const metadata = await extractMetadataViaPydicom(tmpDir, sessionId);

  pushProgress(sessionId, 95, "Metadata extracted.");
  closeProgress(sessionId);

  // Store metadata
  const stored = uploadStore.get(sessionId);
  if (stored) stored.metadata = metadata;

  // Return HTML fragment
  return c.html(`
    <div id="metadata-panel" class="space-y-2 text-sm font-mono">
      <div class="flex items-center gap-2 mb-3">
        <div class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></div>
        <span class="text-emerald-400 font-semibold text-xs uppercase tracking-wider">Loaded</span>
      </div>
      <div class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
        <span class="text-zinc-500">Patient</span>
        <span class="text-zinc-200">${esc(metadata.PatientID || metadata.PatientName || "Anonymous")}</span>
        <span class="text-zinc-500">Series</span>
        <span class="text-zinc-200">${esc(metadata.SeriesDescription || "N/A")}</span>
        <span class="text-zinc-500">Modality</span>
        <span class="text-zinc-200">${esc(metadata.Modality || "N/A")}</span>
        <span class="text-zinc-500">Shape</span>
        <span class="text-zinc-200">${esc(metadata.Rows || "?")} × ${esc(metadata.Columns || "?")} × ${dcmFiles.length} slices</span>
        <span class="text-zinc-500">Spacing</span>
        <span class="text-zinc-200">${esc(metadata.PixelSpacing || "N/A")}</span>
        <span class="text-zinc-500">Thickness</span>
        <span class="text-zinc-200">${esc(metadata.SliceThickness || "N/A")} mm</span>
        <span class="text-zinc-500">TR / TE</span>
        <span class="text-zinc-200">${esc(metadata.RepetitionTime || "?")} / ${esc(metadata.EchoTime || "?")} ms</span>
        <span class="text-zinc-500">Field</span>
        <span class="text-zinc-200">${esc(metadata.MagneticFieldStrength || "?")} T</span>
        <span class="text-zinc-500">Sequence</span>
        <span class="text-zinc-200">${esc(metadata.ScanningSequence || "N/A")}</span>
        <span class="text-zinc-500">Window</span>
        <span class="text-zinc-200">C:${esc(metadata.WindowCenter || "?")} W:${esc(metadata.WindowWidth || "?")}</span>
        <span class="text-zinc-500">Bits</span>
        <span class="text-zinc-200">${esc(metadata.BitsAllocated || "?")} bit</span>
        <span class="text-zinc-500">Manufacturer</span>
        <span class="text-zinc-200">${esc(metadata.Manufacturer || "N/A")}</span>
      </div>
      <input type="hidden" id="upload-session" value="${sessionId}" />
      <input type="hidden" id="upload-dir" value="${tmpDir}" />
      <input type="hidden" id="upload-count" value="${dcmFiles.length}" />
    </div>
  `);
});

// Serve uploaded files back to the browser for DWV to parse
app.get("/files/:sessionId/:filename", async (c) => {
  const sessionId = c.req.param("sessionId");
  const filename = c.req.param("filename");
  const filePath = `/tmp/neuroviz-${sessionId}/${filename}`;

  const file = Bun.file(filePath);
  if (!(await file.exists())) {
    return c.notFound();
  }

  return new Response(file, {
    headers: {
      "Content-Type": "application/dicom",
      "Content-Disposition": `inline; filename="${filename}"`,
    },
  });
});

// List files for a session
app.get("/files/:sessionId", async (c) => {
  const sessionId = c.req.param("sessionId");
  const tmpDir = `/tmp/neuroviz-${sessionId}`;

  try {
    const proc = Bun.spawn(["ls", tmpDir]);
    const text = await new Response(proc.stdout).text();
    const files = text
      .trim()
      .split("\n")
      .filter((f) => f.endsWith(".dcm") || f.endsWith(".DCM"))
      .sort();
    return c.json({ files, sessionId });
  } catch {
    return c.json({ files: [], sessionId });
  }
});

function esc(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function extractMetadataViaPydicom(
  dirPath: string,
  sessionId: string
): Promise<Record<string, string>> {
  const script = `
import json, sys, os
import pydicom

d = sys.argv[1]
files = sorted([f for f in os.listdir(d) if f.lower().endswith('.dcm')])
if not files:
    print(json.dumps({}))
    sys.exit(0)

ds = pydicom.dcmread(os.path.join(d, files[0]), force=True)
tags = [
    'PatientID', 'PatientName', 'PatientSex',
    'StudyDate', 'StudyDescription',
    'SeriesDescription', 'SeriesNumber', 'Modality', 'BodyPartExamined',
    'Rows', 'Columns',
    'MagneticFieldStrength', 'ScanningSequence', 'SequenceVariant',
    'RepetitionTime', 'EchoTime', 'FlipAngle', 'InversionTime',
    'SliceThickness', 'SpacingBetweenSlices', 'PixelSpacing',
    'PhotometricInterpretation', 'BitsAllocated', 'HighBit',
    'WindowCenter', 'WindowWidth', 'RescaleSlope', 'RescaleIntercept',
    'Manufacturer', 'ManufacturerModelName', 'InstitutionName',
    'ImageOrientationPatient', 'ImagePositionPatient'
]
result = {}
for tag in tags:
    try:
        val = getattr(ds, tag)
        result[tag] = str(val)
    except AttributeError:
        pass
print(json.dumps(result))
`;

  const scriptPath = `/tmp/neuroviz-extract-${sessionId}.py`;
  await Bun.write(scriptPath, script);

  try {
    const proc = Bun.spawn(["uv", "run", "--with", "pydicom", "python3", scriptPath, dirPath], {
      stdout: "pipe",
      stderr: "pipe",
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    await proc.exited;

    if (stderr) {
      console.error("pydicom stderr:", stderr);
    }

    try {
      return JSON.parse(stdout.trim());
    } catch {
      console.error("Failed to parse pydicom output:", stdout);
      return {};
    }
  } catch (e) {
    console.error("pydicom subprocess failed:", e);
    return {};
  }
}

export const uploadRoute = app;
