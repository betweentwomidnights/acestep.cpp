"""Model-free HTTP validation and local-estimator parity with the real CLI."""
import io
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

server, analyzer = map(lambda p: str(Path(p).resolve()), sys.argv[1:])
env = {k: v for k, v in os.environ.items() if not k.startswith("ACE_") and k != "GGML_BACKEND"}


def gguf(path, architecture):
    def string(s):
        value = s.encode()
        return struct.pack("<Q", len(value)) + value
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("general.architecture") +
                     struct.pack("<I", 8) + string(architecture))


def post(endpoint, audio=None, request=None, status=200, latents=False):
    boundary = "ace-caption-contract-boundary"
    parts = []
    if request is not None:
        content = request if isinstance(request, bytes) else json.dumps(request).encode()
        parts.append(b'Content-Disposition: form-data; name="request"\r\n\r\n' + content)
    if audio is not None:
        parts.append(b'Content-Disposition: form-data; name="audio"; filename="test.wav"\r\n'
                     b'Content-Type: audio/wav\r\n\r\n' + audio)
    if latents:
        parts.append(b'Content-Disposition: form-data; name="src_latents"; filename="latents.bin"\r\n\r\n' + bytes(256))
    sep = ("--" + boundary).encode()
    body = b"".join(sep + b"\r\n" + p + b"\r\n" for p in parts) + sep + b"--\r\n"
    req = urllib.request.Request(url + endpoint, data=body,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        response = urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        result = json.load(response)
        assert response.status == status, (endpoint, response.status, result)
    return result


with tempfile.TemporaryDirectory(prefix="ace-caption-api-") as temp:
    root = Path(temp)
    models = root / "models"
    models.mkdir()
    for name, arch in [("lm", "acestep-lm"), ("dit", "acestep-dit"), ("second", "acestep-dit"), ("vae", "acestep-vae")]:
        gguf(models / (name + ".gguf"), arch)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen([server, "--models", str(models), "--device", "cpu", "--port", str(port)],
                                env=env, cwd=root, stdout=log, stderr=log)
        try:
            for _ in range(100):
                assert proc.poll() is None
                try:
                    with urllib.request.urlopen(url + "/health", timeout=.2):
                        break
                except OSError:
                    time.sleep(.05)
            else:
                raise AssertionError("server did not become healthy")
            props = json.load(urllib.request.urlopen(url + "/props"))
            assert props["capabilities"]["caption_http"] and props["capabilities"]["local_audio_analysis"]
            stream = io.BytesIO()
            rate = 11025
            with wave.open(stream, "wb") as file:
                file.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                file.writeframes(b"".join(struct.pack("<h", round(7000 * sum(math.sin(2 * math.pi * f * i / rate)
                    for f in (261.625565, 329.627557, 391.995436)))) for i in range(rate * 8)))
            audio = stream.getvalue()
            path = root / "chord.wav"
            path.write_bytes(audio)
            expected = json.loads(subprocess.check_output([analyzer, "--in", str(path)], text=True))
            actual = post("/analyze", audio)
            assert actual.pop("duration") == 8
            assert actual == expected, (actual, expected)
            assert actual["keyscale"] == "C major"
            for data in (b"", b"not WAV", audio[:-1]):
                assert "error" in post("/analyze", data, status=400)
            assert "error" in post("/analyze", request={}, status=400)
            selection = {"lm_model": "lm.gguf", "synth_model": "dit.gguf", "vae": "vae.gguf"}
            assert "error" in post("/caption", audio, status=400)  # ambiguous DiT
            assert "error" in post("/caption", audio, {**selection, "lm_model": "missing"}, status=400)
            assert "error" in post("/caption", audio, b"{", status=400)
            assert "error" in post("/caption", audio, selection, latents=True, status=400)
            assert "error" in post("/caption", audio, {**selection, "lm_top_p": 2}, status=400)
            assert "error" in post("/caption", b"not WAV", selection, status=400)
            payload = struct.pack("<f", float("nan")) * 9000
            nan = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate*4, 4, 32) + b"data" + struct.pack("<I", len(payload)) + payload
            assert "non-finite" in post("/analyze", nan, status=400)["error"]
            assert "non-finite" in post("/caption", nan, selection, status=400)["error"]
            payload = struct.pack("<d", 1e300) * 9000
            overflow = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate*8, 8, 64) + b"data" + struct.pack("<I", len(payload)) + payload
            assert "finite float range" in post("/analyze", overflow, status=400)["error"]
            assert "finite float range" in post("/caption", overflow, selection, status=400)["error"]
            stream = io.BytesIO()
            with wave.open(stream, "wb") as file:
                file.setparams((1, 1, 1000, 0, "NONE", "not compressed"))
                file.writeframes(bytes([128]) * 601000)
            assert "error" in post("/analyze", stream.getvalue(), status=413)
            assert "error" in post("/caption", stream.getvalue(), selection, status=413)
            assert path.read_bytes() == audio
        finally:
            proc.terminate()
            proc.wait(timeout=10)
print("PASS CPU HTTP/CLI metadata parity and caption model/input validation")
