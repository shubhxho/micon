import { Hono } from "hono";

const app = new Hono();

// Colormap preview fragment
app.get("/panel/colormap", (c) => {
  const colormap = c.req.query("colormap") || "hot";

  const gradients: Record<string, string> = {
    hot: "linear-gradient(90deg, #000, #f00, #ff0, #fff)",
    plasma: "linear-gradient(90deg, #0d0887, #7e03a8, #cc4778, #f89540, #f0f921)",
    viridis: "linear-gradient(90deg, #440154, #31688e, #35b779, #fde725)",
    gray: "linear-gradient(90deg, #000, #fff)",
    pet: "linear-gradient(90deg, #000, #00f, #0ff, #0f0, #ff0, #f00, #fff)",
  };

  const grad = gradients[colormap] || gradients.hot;

  return c.html(`
    <div id="colormap-preview" class="mt-2">
      <div class="h-3 rounded" style="background: ${grad}"></div>
      <div class="flex justify-between text-[10px] text-zinc-500 mt-1">
        <span>0</span>
        <span class="text-zinc-400 uppercase tracking-wider">${colormap}</span>
        <span>max</span>
      </div>
    </div>
  `);
});

// Panel fragment — returns the full control panel HTML
app.get("/panel", (c) => {
  return c.html(`
    <div class="space-y-4 text-sm">
      <div class="text-zinc-500 text-xs uppercase tracking-wider">Controls</div>
      <div class="space-y-2">
        <label class="text-zinc-400 text-xs">Threshold</label>
        <input type="range" min="0" max="100" value="10"
               class="w-full accent-cyan-500"
               x-on:input="brain3d?.setThreshold($event.target.value / 100)" />
      </div>
      <div class="space-y-2">
        <label class="text-zinc-400 text-xs">Opacity</label>
        <input type="range" min="1" max="100" value="30"
               class="w-full accent-cyan-500"
               x-on:input="brain3d?.setOpacity($event.target.value / 1000)" />
      </div>
      <div class="space-y-2">
        <label class="text-zinc-400 text-xs">Ray Steps</label>
        <input type="range" min="32" max="512" value="256" step="32"
               class="w-full accent-cyan-500"
               x-on:input="brain3d?.setSteps(parseInt($event.target.value))" />
      </div>
    </div>
  `);
});

export const panelRoute = app;
