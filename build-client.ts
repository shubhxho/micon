async function build() {
  console.log("Building client bundles...");

  const appResult = await Bun.build({
    entrypoints: ["./src/public/js/app.ts"],
    outdir: "./src/public/js",
    format: "esm",
    target: "browser",
    naming: "[name].js",
  });

  if (!appResult.success) {
    console.error("App bundle failed:", appResult.logs);
    process.exit(1);
  }

  const workerResult = await Bun.build({
    entrypoints: ["./src/public/js/worker.ts"],
    outdir: "./src/public/js",
    format: "esm",
    target: "browser",
    naming: "[name].js",
  });

  if (!workerResult.success) {
    console.error("Worker bundle failed:", workerResult.logs);
    process.exit(1);
  }

  console.log("Client bundles built successfully.");
}

build();
