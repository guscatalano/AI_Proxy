"""Guards the version single-source invariant: Python and npm must never drift.

If this fails, run: python scripts/bump_version.py <version|major|minor|patch>
"""
import json
from pathlib import Path

import ai_proxy

ROOT = Path(__file__).resolve().parent.parent
NPM_PKG = ROOT / "npm" / "ai-proxy" / "package.json"


def test_python_and_npm_versions_match():
    npm = json.loads(NPM_PKG.read_text(encoding="utf-8"))
    assert npm["version"] == ai_proxy.__version__, (
        f"npm package.json is {npm['version']} but ai_proxy.__version__ is "
        f"{ai_proxy.__version__}"
    )


def test_platform_optional_deps_pinned_to_current_version():
    npm = json.loads(NPM_PKG.read_text(encoding="utf-8"))
    for name, pin in (npm.get("optionalDependencies") or {}).items():
        assert pin == ai_proxy.__version__, (
            f"optionalDependency {name} pinned to {pin}, expected {ai_proxy.__version__}"
        )


def test_version_is_semver():
    import re

    assert re.match(r"^\d+\.\d+\.\d+", ai_proxy.__version__)
