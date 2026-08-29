"""Two vLLM slots, each with its own port, container and twins.

The second slot existed as a URL and a probe long before it could be controlled: every helper
resolved "the container publishing VLLM_URL", so starting the second slot's model stopped the
first slot's server. These tests pin the per-slot resolution.
"""
import asyncio

from ai_proxy import proxy as P


def _cfg(container, port_matches, running=False, model=None):
    return {"container": container, "image": "vllm/vllm-openai",
            "running": running, "serves_port": port_matches,
            "model": model or container, "checkpoint": None, "mounts": [],
            "max_model_len": "262144", "kv_cache_dtype": "fp8",
            "prefix_caching": True, "gpu_memory_utilization": 0.35, "quant": "nvfp4"}


def test_each_slot_resolves_its_own_container(monkeypatch):
    """The port in the slot's URL is what picks the container, not a module global."""
    seen = []

    async def run_cmd(args, *a, **k):
        if "ps" in args:
            return 0, "one-vllm\ntwo-vllm"
        if "inspect" in args:
            return 0, ('/one-vllm\ttrue\t{"8000/tcp":[{"HostPort":"8001"}]}\n'
                       '/two-vllm\ttrue\t{"8000/tcp":[{"HostPort":"8003"}]}')
        return 0, ""

    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(P, "_run_cmd", run_cmd)
    monkeypatch.setattr(P, "_upstream_is_local", lambda u: True)
    monkeypatch.setattr(P, "VLLM_URL", "http://localhost:8001")
    monkeypatch.setattr(P, "VLLM2_URL", "http://localhost:8003")

    assert asyncio.run(P._vllm_container("http://localhost:8001")) == "one-vllm"
    assert asyncio.run(P._vllm_container("http://localhost:8003")) == "two-vllm"
    # No argument still means the first slot, so every existing caller is unchanged.
    assert asyncio.run(P._vllm_container()) == "one-vllm"
    assert seen == []


def test_loading_one_slot_leaves_the_other_slots_server_up(client, monkeypatch):
    """The load path stops only the twins contending for ITS port.

    Before this, `configs` was every vLLM container on the box and the loop stopped each one
    that was running — so bringing up the second slot's model took down the first slot's.
    """
    stopped = []

    async def run_cmd(args, *a, **k):
        if args[1] == "stop":
            stopped.append(args[2])
        return 0, ""

    async def configs(url=None, *a, **k):
        # slot 2's view: its own twin publishes 8003, the other slot's container does not
        return [_cfg("two-vllm", True, running=False),
                _cfg("two-old-vllm", True, running=True),
                _cfg("one-vllm", False, running=True)]

    monkeypatch.setattr(P, "_docker_bin", lambda: "/usr/bin/docker")
    monkeypatch.setattr(P, "_run_cmd", run_cmd)
    monkeypatch.setattr(P, "_vllm_configs", configs)
    monkeypatch.setattr(P, "_vllm_boot_claim_mb", lambda *a, **k: _none())
    monkeypatch.setattr(P, "_free_mem_mb", lambda: 100000.0)

    async def ready(timeout_s, upstream="vllm"):
        return True
    monkeypatch.setattr(P, "_vllm_ready", ready)

    res = asyncio.run(P.PROVIDERS["vllm2"].load({"container": "two-vllm"}, "two-vllm"))
    assert res["ok"] is True
    assert res["started_container"] == "two-vllm"
    assert stopped == ["two-old-vllm"], f"stopped the wrong containers: {stopped}"
    assert "one-vllm" not in stopped


async def _none():
    return None


def test_the_second_slot_is_a_docker_twin_not_a_hand_started_orphan():
    """It was registered with control='none', so the UI offered no start/stop for it and
    auto-load skipped it entirely."""
    assert P.PROVIDERS["vllm2"].control == "docker"
    assert isinstance(P.PROVIDERS["vllm2"], P._VllmProvider)


def test_both_slots_answer_the_is_this_vllm_question():
    """Every `upstream == "vllm"` check meant "a vLLM server", not "the first one"."""
    assert P._is_vllm("vllm") and P._is_vllm("vllm2")
    assert not P._is_vllm("ollama")
    assert not P._is_vllm(None)


def test_slot_urls_are_read_live(monkeypatch):
    """Read through a function, not captured at import — tests patch these and rebuild."""
    monkeypatch.setattr(P, "VLLM_URL", "http://a:1")
    monkeypatch.setattr(P, "VLLM2_URL", "http://b:2")
    assert P._vllm_url("vllm") == "http://a:1"
    assert P._vllm_url("vllm2") == "http://b:2"
    assert P._vllm_url(None) == "http://a:1"


def test_a_boot_must_leave_the_driver_room(client, monkeypatch):
    """Utilization is not an upper bound.

    Two vLLMs at 0.45 and 0.35 on this box summed to 0.80 and still killed the NVIDIA
    driver's context allocator mid-request — coder-next's weights alone are 42.7 GiB against
    a 54.8 GB claim. Both engines exited 0 with OOMKilled=false, so nothing but the kernel
    log said what happened. The guard prices the claim PLUS a reserve.
    """
    async def configs(url=None, *a, **k):
        return [{"container": "big-vllm", "gpu_memory_utilization": 0.45}]

    monkeypatch.setattr(P, "_vllm_configs", configs)
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"total_mb": 121_700.0})

    need = asyncio.run(P._vllm_boot_claim_mb("big-vllm"))
    assert need == 121_700.0 * 0.45 + P._VLLM_BOOT_RESERVE_MB
    # The number that matters: a box with only the bare claim free is refused, not squeezed.
    assert need > 121_700.0 * 0.45


def test_the_reserve_is_configurable(client, monkeypatch):
    async def configs(url=None, *a, **k):
        return [{"container": "big-vllm", "gpu_memory_utilization": 0.5}]

    cfg = dict(P.load_rules_config())
    cfg["model_control"] = {**(cfg.get("model_control") or {}), "vllm_boot_reserve_mb": 4096}
    monkeypatch.setattr(P, "load_rules_config", lambda: cfg)
    monkeypatch.setattr(P, "_vllm_configs", configs)
    monkeypatch.setattr(P, "_mem_snapshot", lambda: {"total_mb": 100_000.0})

    assert asyncio.run(P._vllm_boot_claim_mb("big-vllm")) == 54_096.0
