"""Certificate expiry reporting, and the HTTPS startup ordering.

Expiry is the same failure shape as pool exhaustion: scheduled, silent, and total when it
lands. Nothing reported it, so health now does.

The fixtures are certificates only — no private keys — because they are parsed, never served.
Only two are checked in, both with stable verdicts: one valid until 2036, one that expired in
the past and always will have. A fixture that expires *soon* would quietly become an expired
one and start failing on a date, so that threshold is exercised by faking the clock instead.
"""
import asyncio
import time
from pathlib import Path

import pytest

import ai_proxy
from ai_proxy import proxy

FIXTURES = Path(__file__).parent / "fixtures"
LONGLIVED = str(FIXTURES / "cert_longlived.pem")
EXPIRED = str(FIXTURES / "cert_expired.pem")


def _configure(monkeypatch, cert, port="8443"):
    monkeypatch.setenv("PROXY_SSL_CERT", cert)
    monkeypatch.setenv("PROXY_HTTPS_PORT", port)
    proxy._TLS_CACHE["key"] = None          # fixtures differ by path, but be explicit


def test_no_https_configured_is_not_an_error(monkeypatch):
    monkeypatch.delenv("PROXY_SSL_CERT", raising=False)
    monkeypatch.delenv("PROXY_HTTPS_PORT", raising=False)
    assert proxy._tls_status() == {"enabled": False}


def test_a_cert_without_a_port_is_not_enabled(monkeypatch):
    _configure(monkeypatch, LONGLIVED, port="0")
    assert proxy._tls_status()["enabled"] is False


def test_a_valid_cert_reports_its_expiry(monkeypatch):
    _configure(monkeypatch, LONGLIVED)
    s = proxy._tls_status()
    assert s["enabled"] and s["exists"]
    assert s["days_remaining"] > 3000, "fixture is valid until 2036"
    assert s["expired"] is False
    assert s["expiring_soon"] is False


def test_an_expired_cert_says_so(monkeypatch):
    _configure(monkeypatch, EXPIRED)
    s = proxy._tls_status()
    assert s["expired"] is True
    assert s["days_remaining"] < 0
    assert s["expiring_soon"] is True, "expired is the extreme case of expiring"


def test_a_missing_cert_file_is_reported_not_hidden(monkeypatch):
    """Configured-but-absent means HTTPS never started; silence would read as healthy."""
    _configure(monkeypatch, str(FIXTURES / "does_not_exist.pem"))
    s = proxy._tls_status()
    assert s["enabled"] is True
    assert s["exists"] is False
    assert "days_remaining" not in s


def test_an_unparseable_cert_costs_the_metric_not_the_endpoint(monkeypatch, tmp_path):
    junk = tmp_path / "junk.pem"
    junk.write_text("-----BEGIN CERTIFICATE-----\nnope\n-----END CERTIFICATE-----\n")
    _configure(monkeypatch, str(junk))
    s = proxy._tls_status()                  # must not raise
    assert s["exists"] is True
    assert "days_remaining" not in s


def test_the_warning_threshold_does_not_depend_on_a_fixture_expiring(monkeypatch):
    """A cert 5 days out warns; one well clear does not."""
    _configure(monkeypatch, LONGLIVED)
    import time as _t

    monkeypatch.setattr(proxy, "_cert_not_after", lambda p: _t.time() + 5 * 86400)
    proxy._TLS_CACHE["key"] = None
    assert proxy._tls_status()["expiring_soon"] is True

    monkeypatch.setattr(proxy, "_cert_not_after", lambda p: _t.time() + 60 * 86400)
    proxy._TLS_CACHE["key"] = None
    assert proxy._tls_status()["expiring_soon"] is False


def test_health_carries_tls(client):
    body = client.get("/__proxy/api/health").json()
    assert "tls" in body
    assert "enabled" in body["tls"]


def test_tls_watch_warns_and_then_stays_quiet(monkeypatch, capsys):
    _configure(monkeypatch, EXPIRED)
    proxy._TLS_WARNED["at"] = 0.0
    proxy._tls_watch()
    assert "EXPIRED" in capsys.readouterr().out
    proxy._tls_watch()                       # hourly, so the second call says nothing
    assert capsys.readouterr().out == ""


def test_tls_watch_is_silent_when_https_is_off(monkeypatch, capsys):
    monkeypatch.delenv("PROXY_SSL_CERT", raising=False)
    proxy._TLS_WARNED["at"] = 0.0
    proxy._tls_watch()
    assert capsys.readouterr().out == ""


# --- HTTPS startup ordering ------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    """asyncio only. anyio would otherwise also run these under trio, where the
    asyncio.create_task calls below are meaningless — and the code under test is asyncio."""
    return "asyncio"


class _FakeServer:
    """Enough of uvicorn.Server for the ordering logic: a flag it sets when ready."""

    def __init__(self, started=False):
        self.started = started
        self.should_exit = False


@pytest.mark.anyio
async def test_https_waits_for_http_to_finish_starting():
    """The whole point: no HTTPS listener until app.state.client and the DB exist."""
    srv = _FakeServer(started=False)
    task = asyncio.create_task(asyncio.sleep(5))

    async def _start_later():
        await asyncio.sleep(0.15)
        srv.started = True

    flip = asyncio.create_task(_start_later())
    await proxy._wait_for_http_start(srv, task, timeout_s=5)
    assert srv.started, "returned before startup completed"
    flip.cancel()
    task.cancel()


@pytest.mark.anyio
async def test_a_slow_start_is_waited_out_not_raced():
    """Startup slower than the old hardcoded 5s must still be honoured, not overrun."""
    srv = _FakeServer(started=False)
    task = asyncio.create_task(asyncio.sleep(5))

    async def _start_later():
        await asyncio.sleep(0.3)
        srv.started = True

    flip = asyncio.create_task(_start_later())
    start = time.monotonic()
    await proxy._wait_for_http_start(srv, task, timeout_s=10)
    assert time.monotonic() - start >= 0.25
    flip.cancel()
    task.cancel()


@pytest.mark.anyio
async def test_it_refuses_rather_than_serving_uninitialised_state():
    srv = _FakeServer(started=False)
    task = asyncio.create_task(asyncio.sleep(5))
    with pytest.raises(RuntimeError, match="refusing to open HTTPS"):
        await proxy._wait_for_http_start(srv, task, timeout_s=0.2)
    assert srv.should_exit is True, "a proxy that cannot serve HTTPS should not linger"
    task.cancel()


@pytest.mark.anyio
async def test_an_http_listener_that_dies_does_not_yield_an_https_one():
    srv = _FakeServer(started=False)

    async def _die():
        raise OSError("address already in use")

    task = asyncio.create_task(_die())
    with pytest.raises(OSError, match="address already in use"):
        await proxy._wait_for_http_start(srv, task, timeout_s=5)


@pytest.mark.anyio
async def test_an_already_started_server_returns_immediately():
    srv = _FakeServer(started=True)
    task = asyncio.create_task(asyncio.sleep(5))
    start = time.monotonic()
    await proxy._wait_for_http_start(srv, task, timeout_s=5)
    assert time.monotonic() - start < 0.2
    task.cancel()
