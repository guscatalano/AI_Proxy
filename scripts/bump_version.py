#!/usr/bin/env python3
"""Bump the project version across Python and npm in one shot.

The canonical version lives in ai_proxy/_version.py (pyproject reads it dynamically).
This script rewrites it plus the npm launcher package.json (version + the pinned
optionalDependencies), rolls the CHANGELOG's Unreleased section into the new version,
and optionally commits and tags.

Usage:
    python scripts/bump_version.py <version|major|minor|patch> [--commit] [--tag]

Examples:
    python scripts/bump_version.py 0.2.0 --commit --tag
    python scripts/bump_version.py patch --commit --tag
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_PY = ROOT / "ai_proxy" / "_version.py"
NPM_PKG = ROOT / "npm" / "ai-proxy" / "package.json"
CHANGELOG = ROOT / "CHANGELOG.md"

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def current_version() -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', VERSION_PY.read_text(encoding="utf-8"))
    if not m:
        sys.exit("could not find __version__ in ai_proxy/_version.py")
    return m.group(1)


def resolve_target(spec: str, cur: str) -> str:
    if spec in ("major", "minor", "patch"):
        m = SEMVER_RE.match(cur)
        if not m:
            sys.exit(f"current version {cur!r} is not X.Y.Z; pass an explicit version")
        major, minor, patch = (int(x) for x in m.groups())
        if spec == "major":
            major, minor, patch = major + 1, 0, 0
        elif spec == "minor":
            minor, patch = minor + 1, 0
        else:
            patch += 1
        return f"{major}.{minor}.{patch}"
    if not SEMVER_RE.match(spec):
        sys.exit(f"{spec!r} is neither a bump level (major/minor/patch) nor an X.Y.Z version")
    return spec


def write_version_py(new: str) -> None:
    text = VERSION_PY.read_text(encoding="utf-8")
    text = re.sub(r'(__version__\s*=\s*)"[^"]+"', rf'\g<1>"{new}"', text)
    VERSION_PY.write_text(text, encoding="utf-8")


def write_npm(new: str) -> None:
    pkg = json.loads(NPM_PKG.read_text(encoding="utf-8"))
    pkg["version"] = new
    for dep in list(pkg.get("optionalDependencies", {})):
        pkg["optionalDependencies"][dep] = new
    NPM_PKG.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")


def roll_changelog(new: str) -> None:
    if not CHANGELOG.exists():
        return
    text = CHANGELOG.read_text(encoding="utf-8")
    heading = f"## [{new}] - {date.today().isoformat()}"
    if "## [Unreleased]" in text:
        text = text.replace("## [Unreleased]", f"## [Unreleased]\n\n{heading}", 1)
    CHANGELOG.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="explicit X.Y.Z, or one of: major, minor, patch")
    ap.add_argument("--commit", action="store_true", help="git commit the version changes")
    ap.add_argument("--tag", action="store_true", help="create an annotated git tag vX.Y.Z (implies --commit)")
    args = ap.parse_args()

    cur = current_version()
    new = resolve_target(args.version, cur)
    if new == cur and not args.tag:
        sys.exit(f"version is already {cur}; nothing to do")

    write_version_py(new)
    write_npm(new)
    roll_changelog(new)
    print(f"bumped {cur} -> {new}")
    print(f"  updated {VERSION_PY.relative_to(ROOT)}")
    print(f"  updated {NPM_PKG.relative_to(ROOT)}")
    if CHANGELOG.exists():
        print(f"  updated {CHANGELOG.relative_to(ROOT)}")

    if args.commit or args.tag:
        files = [str(VERSION_PY), str(NPM_PKG)]
        if CHANGELOG.exists():
            files.append(str(CHANGELOG))
        subprocess.run(["git", "add", *files], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", f"Release v{new}"], cwd=ROOT, check=True)
        print(f"  committed 'Release v{new}'")
    if args.tag:
        subprocess.run(["git", "tag", "-a", f"v{new}", "-m", f"v{new}"], cwd=ROOT, check=True)
        print(f"  tagged v{new}  (push with: git push origin v{new})")


if __name__ == "__main__":
    main()
