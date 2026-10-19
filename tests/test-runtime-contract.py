"""Model-free process checks for the downloadable supervisor runtime."""
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

server, trainer, version = map(str, sys.argv[1:])
env = {k: v for k, v in os.environ.items() if not k.startswith("ACE_") and k != "GGML_BACKEND"}


def run(binary, *args, extra=None, ok=True):
    p = subprocess.run([binary, *args], env={**env, **(extra or {})}, cwd=scratch,
                       capture_output=True, text=True, timeout=15)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stderr)
    return p


def gguf(path, architecture):
    def string(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) +
                     string("general.architecture") + struct.pack("<I", 8) + string(architecture))


with tempfile.TemporaryDirectory(prefix="ace-runtime-contract-") as scratch:
    for binary in (server, trainer):
        assert run(binary, "--version").stdout.strip() == version
    p = run(server, "--props", "--device", "cpu", extra={"ACE_MODELS_DIR": "missing-models"})
    props = json.loads(p.stdout)
    assert props["version"] == version
    assert props["cli"]["adapter_mode"] == "functional"
    assert any(d["backend"] == "CPU" and d["type"] == "cpu" for d in props["devices"])
    assert props["capabilities"]["training_http"] is False
    assert props["training"]["xl_base_low_memory_encoding"] == "Q4_K_M"
    run(server, "--props", "--device", "missing-device", ok=False)
    run(server, "--props", extra={"ACE_DEVICE": "missing-device"}, ok=False)
    run(server, "--props", "--device", "cpu", extra={"ACE_DEVICE": "missing-device"})
    run(server, "--props", extra={"ACE_PORT": "bad"}, ok=False)
    run(server, "--props", "--port", "8080", extra={"ACE_PORT": "bad"})
    merged = json.loads(run(server, "--props", "--adapter-mode", "merge",
                            extra={"ACE_DEVICE": "cpu", "ACE_ADAPTER_MODE": "invalid"}).stdout)
    assert merged["cli"]["adapter_mode"] == "merge"

    models = Path(scratch) / "models"
    models.mkdir()
    gguf(models / "lm.gguf", "acestep-lm")
    # No CLI options: the supervisor configures the executable through env.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([server], cwd=scratch,
                            env={**env, "ACE_MODELS_DIR": str(models), "ACE_PORT": str(port), "ACE_DEVICE": "cpu"},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            assert proc.poll() is None, "environment-configured server exited before health"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2) as response:
                    assert response.status == 200
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("environment-configured server never became healthy")
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    for name in ("base", "xl-base"):
        gguf(models / f"{name}.gguf", "acestep-dit")
    gguf(models / "text.gguf", "acestep-text-enc")
    gguf(models / "vae.gguf", "acestep-vae")
    p = run(trainer, "--models", str(models), "--dataset", scratch, "--output", scratch, ok=False)
    assert "missing or ambiguous" in p.stderr, p.stderr
print("PASS: version, props, device/config precedence, health and explicit training model selection")
