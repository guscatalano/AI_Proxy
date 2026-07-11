#!/usr/bin/env node
"use strict";

// Launcher: resolve the prebuilt `ai-proxy` binary for this platform (shipped as an
// optional, platform-gated dependency) and exec it, forwarding args, stdio and exit code.
// The heavy per-platform binaries are the optionalDependencies in package.json; npm's
// os/cpu gating installs only the one matching the current machine.

const { spawn } = require("child_process");
const path = require("path");

const PKG = "guscatalano-ai-proxy";
const SUPPORTED = ["win32-x64", "darwin-x64", "darwin-arm64", "linux-x64", "linux-arm64"];

function resolveBinary() {
  const key = `${process.platform}-${process.arch}`;
  const subpkg = `${PKG}-${key}`;
  const exe = process.platform === "win32" ? "ai-proxy.exe" : "ai-proxy";
  try {
    // Resolve via the sub-package's package.json so we don't depend on its "exports" map.
    const pkgJson = require.resolve(`${subpkg}/package.json`);
    return path.join(path.dirname(pkgJson), "bin", exe);
  } catch (_) {
    return null;
  }
}

function fail(msg) {
  process.stderr.write(msg + "\n");
  process.exit(1);
}

const bin = resolveBinary();
if (!bin) {
  fail(
    `${PKG}: no prebuilt binary found for ${process.platform}-${process.arch}.\n` +
      `Supported platforms: ${SUPPORTED.join(", ")}.\n` +
      `If your platform is supported, reinstall so the optional binary package is fetched ` +
      `(e.g. remove node_modules and 'npm install' again, without --no-optional).\n` +
      `Alternatively install the Python build:  pipx install ${PKG}`
  );
}

const child = spawn(bin, process.argv.slice(2), { stdio: "inherit" });
child.on("error", (err) => fail(`${PKG}: failed to launch binary at ${bin}\n${err.message}`));
child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
  } else {
    process.exit(code == null ? 1 : code);
  }
});
