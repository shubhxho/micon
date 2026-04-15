import { Hono } from "hono";

const app = new Hono();

app.get("/slices", (c) => {
  const x = c.req.query("slice-x") || "50";
  const y = c.req.query("slice-y") || "50";
  const z = c.req.query("slice-z") || "50";

  return c.html(`
    <div id="slice-stats" class="grid grid-cols-3 gap-2 text-xs font-mono">
      <div class="bg-zinc-800/50 rounded px-2 py-1">
        <span class="text-zinc-500">X</span>
        <span class="text-cyan-400 ml-1">${esc(x)}</span>
      </div>
      <div class="bg-zinc-800/50 rounded px-2 py-1">
        <span class="text-zinc-500">Y</span>
        <span class="text-cyan-400 ml-1">${esc(y)}</span>
      </div>
      <div class="bg-zinc-800/50 rounded px-2 py-1">
        <span class="text-zinc-500">Z</span>
        <span class="text-cyan-400 ml-1">${esc(z)}</span>
      </div>
    </div>
  `);
});

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export const slicesRoute = app;
