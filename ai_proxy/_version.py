# Single source of truth for the Python package version.
# pyproject.toml reads this via [tool.setuptools.dynamic]; scripts/bump_version.py
# rewrites it (and the npm package.json) in lockstep. Keep this module import-free.
__version__ = "0.1.0"
