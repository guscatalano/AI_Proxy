"""Tests for the install-safe writable-state path resolution."""
from pathlib import Path

from ai_proxy.proxy import _resolve_state_path, _user_state_dir


def test_env_var_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_PATH_VAR", str(tmp_path / "explicit.db"))
    got = _resolve_state_path("MY_PATH_VAR", "default.db")
    assert got == str(tmp_path / "explicit.db")


def test_existing_legacy_path_is_reused(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_PATH_VAR", raising=False)
    legacy = tmp_path / "legacy.db"
    legacy.write_text("x")
    got = _resolve_state_path("MY_PATH_VAR", "default.db", legacy)
    assert got == str(legacy)


def test_falls_back_to_user_state_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("MY_PATH_VAR", raising=False)
    monkeypatch.setenv("PROXY_STATE_DIR", str(tmp_path / "state"))
    missing_legacy = tmp_path / "does-not-exist.db"
    got = _resolve_state_path("MY_PATH_VAR", "default.db", missing_legacy)
    assert got == str(tmp_path / "state" / "default.db")


def test_proxy_state_dir_env_honored(monkeypatch, tmp_path):
    monkeypatch.setenv("PROXY_STATE_DIR", str(tmp_path / "custom"))
    assert _user_state_dir() == Path(str(tmp_path / "custom"))


def test_default_state_dir_ends_with_ai_proxy(monkeypatch):
    monkeypatch.delenv("PROXY_STATE_DIR", raising=False)
    d = _user_state_dir()
    assert d.is_absolute()
    assert d.name == "ai-proxy"
