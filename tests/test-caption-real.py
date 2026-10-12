"""Opt-in model-backed caption regression; supply your own representative WAV.

Exercises capture-enabled decoding, normalized results, determinism, queued
cancellation and actual backend selection. It does not grade semantic accuracy.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
for option in ("server", "models", "audio", "lm", "dit", "vae", "device", "output"):
    parser.add_argument("--" + option, required=True)
parser.add_argument("--no-fsm", action="store_true")
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=False)
env = {k: v for k, v in os.environ.items()
       if not k.startswith("ACE_") and k not in ("GGML_BACKEND", "GGML_CUDA_DISABLE_GRAPHS")}
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
url = f"http://127.0.0.1:{port}"
selection = {"lm_model": args.lm, "synth_model": args.dit, "vae": args.vae,
             "lm_seed": 42, "lm_temperature": .3}
boundary = "ace-caption-real-boundary"
body = (f'--{boundary}\r\nContent-Disposition: form-data; name="request"\r\n\r\n'
        + json.dumps(selection)
        + f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="fixture.wav"'
          '\r\nContent-Type: audio/wav\r\n\r\n').encode()
body += Path(args.audio).read_bytes() + f"\r\n--{boundary}--\r\n".encode()


def get(path):
    with urllib.request.urlopen(url + path, timeout=30) as response:
        return json.load(response)


def submit():
    request = urllib.request.Request(url + "/caption", data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["id"]


def wait(job):
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        status = get("/job?id=" + job)["status"]
        if status == "done":
            return get("/job?id=" + job + "&result=1")
        assert status not in ("failed", "cancelled"), status
        time.sleep(.1)
    raise AssertionError("caption timed out")


command = [str(Path(args.server).resolve()), "--models", str(Path(args.models).resolve()),
           "--device", args.device, "--port", str(port)]
if args.no_fsm:
    command.append("--no-fsm")
with (output / "server.log").open("w", encoding="utf-8") as log:
    process = subprocess.Popen(command, env=env, cwd=output, stdout=log, stderr=log)
    try:
        for _ in range(150):
            assert process.poll() is None, "server exited"
            try:
                get("/health")
                break
            except OSError:
                time.sleep(.1)
        else:
            raise AssertionError("server startup timed out")
        first = submit()
        cancelled = submit()
        request = urllib.request.Request(url + "/job?id=" + cancelled + "&cancel=1", data=b"")
        urllib.request.urlopen(request, timeout=30).close()
        result = wait(first)
        assert wait(submit()) == result, "same-seed replay changed"
        for _ in range(50):
            if get("/job?id=" + cancelled)["status"] == "cancelled":
                break
            time.sleep(.1)
        else:
            raise AssertionError("queued cancellation was not honored")
        caption = result["caption"]
        assert len(caption.split()) >= 8 and len(caption) < 16000, "empty/truncated caption"
        assert not re.search(r"<\|audio_code_|</?think>|\ufffd|(.)\1{11}", caption)
        assert result["schema_version"] == 1 and result["prompt"] == caption
        assert 30 <= result["bpm"] <= 300 and result["keyscale"] and result["time_signature"]
        assert result["duration"] > 0 and result["local_analysis"]["schema_version"] == 1
        for key in ("lm_model", "synth_model", "vae", "lm_seed"):
            assert result[key] == selection[key]
        assert "audio_codes" not in result and "src_latents" not in result
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        process.terminate()
        process.wait(timeout=30)
log_text = (output / "server.log").read_text(encoding="utf-8", errors="replace")
assert "[Load] LM backend: " + args.device + " " in log_text, "requested LM device did not execute"
if args.device.startswith("CUDA"):
    assert "CUDA graph warmup complete" in log_text, "CUDA capture was not exercised"
print("PASS: caption, same-seed replay, queued cancellation and " + args.device)
