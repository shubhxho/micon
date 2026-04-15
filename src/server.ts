import { Hono } from "hono";
import { serveStatic } from "hono/bun";
import { cors } from "hono/cors";
import { uploadRoute } from "./routes/upload";
import { metadataRoute } from "./routes/metadata";
import { panelRoute } from "./routes/fragments/panel";
import { slicesRoute } from "./routes/fragments/slices";

const app = new Hono();

app.use("*", cors());

// WASM files (explicit route with correct MIME type)
app.get("/public/wasm/:file", async (c) => {
  const filename = c.req.param("file");
  const file = Bun.file(`./src/public/wasm/${filename}`);
  if (!(await file.exists())) return c.notFound();
  return new Response(file, {
    headers: { "Content-Type": "application/wasm" },
  });
});

// Static files — JS and CSS from src/public
app.use("/public/*", serveStatic({ root: "./src" }));

// Routes
app.route("/", uploadRoute);
app.route("/", metadataRoute);
app.route("/", panelRoute);
app.route("/", slicesRoute);

// SSE progress stream — real progress pushed from upload handler
const sseClients = new Map<string, ReadableStreamDefaultController>();

export function pushProgress(sessionId: string, pct: number, message?: string) {
  const ctrl = sseClients.get(sessionId);
  if (ctrl) {
    try {
      ctrl.enqueue(
        `data: ${JSON.stringify({ pct, message: message ?? "" })}\n\n`
      );
    } catch {
      sseClients.delete(sessionId);
    }
  }
}

export function closeProgress(sessionId: string) {
  const ctrl = sseClients.get(sessionId);
  if (ctrl) {
    try {
      ctrl.close();
    } catch {}
    sseClients.delete(sessionId);
  }
}

app.get("/sse/progress/:sessionId", (c) => {
  const sessionId = c.req.param("sessionId");
  const stream = new ReadableStream({
    start(controller) {
      sseClients.set(sessionId, controller);
      controller.enqueue(
        `data: ${JSON.stringify({ pct: 0, message: "Waiting for upload..." })}\n\n`
      );
    },
    cancel() {
      sseClients.delete(sessionId);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
});

// Index page
app.get("/", async (c) => {
  const file = Bun.file("./src/public/index.html");
  return new Response(file, {
    headers: { "Content-Type": "text/html; charset=utf-8" },
  });
});

console.log("🧠 NeuroViz running at http://localhost:3000");

export default {
  port: 3000,
  fetch: app.fetch,
};
