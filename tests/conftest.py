"""Shared test fixtures.

Point all writable state at a throwaway temp dir *before* importing the app, so tests
never touch a real proxy.db or the per-user state dir. The app resolves DB_PATH /
GENERATED_DIR at import time, so these env vars must be set first.
"""
import os
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="aiproxy-tests-")
os.environ.setdefault("PROXY_STATE_DIR", _TMP)
os.environ.setdefault("PROXY_DB", str(Path(_TMP) / "test.db"))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import ai_proxy  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # The context manager drives the app's lifespan (init_db, http client setup).
    with TestClient(ai_proxy.app) as c:
        yield c
