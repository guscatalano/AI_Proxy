#!/usr/bin/env node
"use strict";

// Assembles a platform-specific npm sub-package (e.g. guscatalano-ai-proxy-linux-x64)
// around a freshly-built PyInstaller binary. Run once per platform in CI, then publish
// the resulting directory.
//
// Usage:
//   node npm/build-platform-package.js <platformKey> <binaryPath> <outDir> [version]
//
//   platformKey : win32-x64 | darwin-x64 | darwin-arm64 | linux-x64 | linux-arm64
//   binaryPath  : path to the built `ai-proxy` (or ai-proxy.exe) binary
//   outDir      : directory to create the package in
//   version     : package version (defaults to the main package's version)

const fs = require("fs");
const path = require("path");

const PLATFORMS = {
  "win32-x64": { os: "win32", cpu: "x64", exe: "ai-proxy.exe" },
  "darwin-x64": { os: "darwin", cpu: "x64", exe: "ai-proxy" },
  "darwin-arm64": { os: "darwin", cpu: "arm64", exe: "ai-proxy" },
  "linux-x64": { os: "linux", cpu: "x64", exe: "ai-proxy" },
  "linux-arm64": { os: "linux", cpu: "arm64", exe: "ai-proxy" },
};

function main() {
  const [, , key, binaryPath, outDir, versionArg] = process.argv;
  if (!key || !binaryPath || !outDir) {
    console.error("usage: build-platform-package.js <platformKey> <binaryPath> <outDir> [version]");
    process.exit(2);
  }
  const meta = PLATFORMS[key];
  if (!meta) {
    console.error(`unknown platformKey '${key}'. known: ${Object.keys(PLATFORMS).join(", ")}`);
    process.exit(2);
  }
  const version =
    versionArg ||
    require(path.join(__dirname, "ai-proxy", "package.json")).version;

  const binDir = path.join(outDir, "bin");
  fs.mkdirSync(binDir, { recursive: true });
  fs.copyFileSync(binaryPath, path.join(binDir, meta.exe));
  if (meta.os !== "win32") {
    fs.chmodSync(path.join(binDir, meta.exe), 0o755);
  }

  const pkg = {
    name: `guscatalano-ai-proxy-${key}`,
    version,
    description: `Prebuilt ai-proxy binary for ${meta.os}/${meta.cpu}.`,
    homepage: "https://github.com/guscatalano/AI_Proxy",
    license: "MIT",
    author: "Gus Catalano",
    os: [meta.os],
    cpu: [meta.cpu],
    files: ["bin"],
  };
  fs.writeFileSync(
    path.join(outDir, "package.json"),
    JSON.stringify(pkg, null, 2) + "\n"
  );
  console.log(`wrote ${key} package (v${version}) -> ${outDir}`);
}

main();
