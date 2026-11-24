"""Exercise the real CLI control contract without models or a GPU."""
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import wave

binary = Path(sys.argv[1]).resolve()
info = subprocess.run([binary, "--control-info"], capture_output=True, text=True, check=True)
capabilities = json.loads(info.stdout)
assert capabilities == {"schema_version": 1, "cooperative_cancel_file": True, "atomic_progress_file": True}
with tempfile.TemporaryDirectory(prefix="ace-cli-control-") as temporary:
    root = Path(temporary)
    dataset = root / "dataset"
    dataset.mkdir()
    status = root / "progress.json"
    cancel = root / "cancel.requested"
    common = [binary, "--dataset", dataset, "--models", root / "missing-models",
              "--output", root / "run", "--steps", "1",
              "--progress-file", status, "--cancel-file", cancel]
    failed = subprocess.run(common, capture_output=True, text=True)
    assert failed.returncode != 0
    state = json.loads(status.read_text())
    assert state["status"] == "failed" and state["error"], (state, failed.stderr)
    assert state["final_adapter"] == "" and state["step"] == 0
    cancel.write_text("stop")
    cancelled = subprocess.run(common, capture_output=True, text=True, check=True)
    state = json.loads(status.read_text())
    assert state["status"] == "cancelled" and state["step"] == 0, state
    assert state["final_adapter"] == "" and cancel.is_file()
    assert not list(root.glob("progress.json.tmp-*"))
    assert "Cancelled before loading" in state["message"]
print("PASS real trainer capabilities, terminal failure, and pre-load cooperative cancellation")

# Arguments and paths are UTF-8 end to end. On Windows the trainer used to read argv and file names in
# the system code page: a cp1252 name ("Jóhannsson") reached the progress JSON as invalid UTF-8 and
# ended the run with "cannot serialize training progress", and names outside cp1252 became "?".
name = "Jóhannsson- A Deal with Chaos データ"
with tempfile.TemporaryDirectory(prefix="ace-cli-Jóhannsson-データ-") as temporary:
    root = Path(temporary)
    dataset = root / "dataset ü データ"
    dataset.mkdir()
    with wave.open(str(dataset / f"{name}.wav"), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(b"".join(struct.pack("<hh", int(8000 * math.sin(i / 20)), 0) for i in range(48000)))
    status = root / "progress é.json"
    common = [binary, "--dataset", dataset, "--models", root / "modèles", "--output", root / "run ü",
              "--steps", "1", "--progress-file", status, "--validate-dataset"]
    failed = subprocess.run(common, capture_output=True)
    assert failed.returncode != 0
    assert status.is_file(), failed.stderr
    state = json.loads(status.read_bytes().decode("utf-8"))
    assert state["status"] == "failed", (state, failed.stderr)
    assert f"{name}.wav" in state["error"], state["error"]
    (dataset / f"{name}.txt").write_text(
        "caption: ambient drone\ngenre: ambient\nbpm: 90\nkey: A minor\nsignature: 4\n"
        "is_instrumental: true\ncustom_tag: drone\nlyrics:\n", encoding="utf-8")
    validated = subprocess.run(common, capture_output=True)
    state = json.loads(status.read_bytes().decode("utf-8"))
    assert validated.returncode == 0 and state["status"] == "completed", (state, validated.stderr)
print("PASS UTF-8 dataset, model, output and progress paths")
