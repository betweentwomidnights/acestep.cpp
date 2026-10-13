"""Opt-in XL Q4 training job control and exact continuation regression.

Creates its own dataset. Compare uninterrupted training, a mid-epoch step stop,
cooperative cancellation during accumulation, and explicit legacy migration.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import wave

p = argparse.ArgumentParser(description=__doc__)
for key in ("bin-dir", "models", "backend", "output"):
    p.add_argument("--" + key, required=True)
p.add_argument("--adapter-type", choices=("lora", "dora-rows"), default="lora")
p.add_argument("--dit", default="acestep-v15-xl-base-Q4_K_M.gguf")
p.add_argument("--rank", type=int, default=64)
args = p.parse_args()
root = Path(args.output).resolve()
root.mkdir(parents=True, exist_ok=False)
dataset = root / "dataset"
dataset.mkdir()
for index in range(3):
    with wave.open(str(dataset / f"{index}.wav"), "wb") as audio:
        audio.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(b"".join(struct.pack("<hh", round(1000 * math.sin(2 * math.pi * (220 + index*20) * i/48000)),
            round(900 * math.sin(2 * math.pi * 330 * i/48000))) for i in range(48000*4)))
    (dataset / f"{index}.txt").write_text("caption: Sustained instrumental tones.\ngenre: ambient\nbpm: 120\n"
        "key: C major\nsignature: 4\nis_instrumental: true\ncustom_tag: smoke\nlyrics: [Instrumental]\n")
env = {k: v for k, v in os.environ.items() if not k.startswith("ACE_")}
env["GGML_BACKEND"] = args.backend
exe = Path(args.bin_dir).resolve() / ("ace-train.exe" if os.name == "nt" else "ace-train")
common = [str(exe), "--models", str(Path(args.models).resolve()), "--dataset", str(dataset),
    "--dit", args.dit, "--text-encoder", "Qwen3-Embedding-0.6B-Q8_0.gguf", "--vae", "vae-BF16.gguf",
    "--adapter-type", args.adapter_type, "--profile", "balanced", "--rank", str(args.rank), "--alpha", str(args.rank * 2),
    "--epochs", "2", "--batch-size", "1", "--accumulation", "2", "--warmup-steps", "0",
    "--checkpoint-every", "1", "--window-min-seconds", "2", "--window-max-seconds", "2", "--seed", "42"]


def checkpoint(path):
    pointer = (path / "checkpoint.current").read_text().strip()
    resolved = (path / pointer).resolve()
    assert resolved.is_relative_to(path.resolve())
    return resolved


def run(label, extra=(), cancel=False, fail=None):
    status = root / (label + ".json")
    output = root / label
    command = common + ["--output", str(output), "--progress-file", str(status)] + list(map(str, extra))
    cancel_file = root / (label + ".cancel")
    if cancel:
        command += ["--cancel-file", str(cancel_file)]
    log_path = root / (label + ".log")
    print(label, flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 600
            while process.poll() is None:
                assert time.monotonic() < deadline, "trainer timeout"
                if status.exists():
                    # Repeated readers must never observe a partially written JSON.
                    try:
                        json.loads(status.read_text())
                    except PermissionError:
                        # MoveFileExW may briefly deny an open during replacement.
                        # This is not a partial JSON read; try the next snapshot.
                        pass
                if cancel and not cancel_file.exists() and "[Ace-Train] window epoch=1" in log_path.read_text():
                    cancel_file.write_text("stop at the next complete optimizer update")
                time.sleep(.01)
            terminal = json.loads(status.read_text())
            if fail:
                assert process.returncode != 0 and terminal["status"] == "failed"
                assert fail in terminal["error"], terminal
                assert "Preprocessing" not in log_path.read_text(), "incompatible resume reached preprocessing"
            else:
                assert process.returncode == 0, log_path.read_text()[-3000:]
                assert terminal["status"] == ("cancelled" if cancel else "completed"), terminal
                assert terminal["state_checkpoint"] == str(checkpoint(output)), terminal
                assert Path(terminal["final_adapter"]) == output
                assert f"backend: {args.backend}" in log_path.read_text(), "requested backend did not execute"
                if cancel:
                    assert terminal["step"] == 1 and cancel_file.exists(), terminal
            return output
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)


def same(left, right):
    for filename in ("adapter_model.safetensors", "adapter_config.json", "optimizer_state.safetensors", "trainer_state.json"):
        assert (checkpoint(left) / filename).read_bytes() == (checkpoint(right) / filename).read_bytes(), filename


full = run("full", ["--checkpoint-every-steps", "1"])
assert (full / "checkpoint-step-1" / "checkpoint.current").is_file()
partial = run("partial", ["--steps", "1"])
same(full / "checkpoint-step-1", partial)
partial_state = json.loads((checkpoint(partial) / "trainer_state.json").read_text())
assert partial_state["completed_epochs"] == 0 and partial_state["optimizer_step"] == 1
assert partial_state["continuation"]["next_batch"] == 2
resumed = run("resumed", ["--resume", partial])
same(full, resumed)
cancelled = run("cancelled", cancel=True)
same(partial, cancelled)
cancel_resumed = run("cancel-resumed", ["--resume", cancelled])
same(full, cancel_resumed)
run("changed-lr", ["--resume", partial, "--learning-rate", "0.0002"], fail="incompatible")
run("changed-budget", ["--resume", partial, "--epochs", "3"], fail="incompatible")
run("old-output", ["--output", full], fail="output already contains")
metadata = dataset / "0.txt"
original_metadata = metadata.read_bytes()
metadata.write_bytes(original_metadata.replace(b"ambient", b"changed"))
run("changed-data", ["--resume", partial], fail="incompatible")
metadata.write_bytes(original_metadata)

# Simulate a genuine older epoch-boundary checkpoint, preserving its weight and
# optimizer bytes, then recreate its integrity manifest after removing the plan.
legacy = root / "legacy"
shutil.copytree(checkpoint(full / "checkpoint-epoch-1"), legacy)
state = json.loads((legacy / "trainer_state.json").read_text())
state.pop("continuation")
(legacy / "trainer_state.json").write_text(json.dumps(state))


def fingerprint(path):
    payload = path.read_bytes()
    value = 14695981039346656037
    for byte in payload:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"fnv1a64:{value:016x}:{len(payload)}"


manifest = json.loads((legacy / "checkpoint_manifest.json").read_text())
manifest["files"]["trainer_state.json"] = fingerprint(legacy / "trainer_state.json")
(legacy / "checkpoint_manifest.json").write_text(json.dumps(manifest))
run("legacy-rejected", ["--resume", legacy], fail="predates resume-plan")
migrated = run("legacy-migrated", ["--resume", legacy, "--allow-legacy-resume"])
same(full, migrated)
(root / "summary.json").write_text(json.dumps({"passed": True, "backend": args.backend,
    "adapter_type": args.adapter_type, "dit": args.dit, "rank": args.rank, "accumulation": 2,
    "mid_epoch_bitexact": True, "cancel_bitexact": True, "legacy_epoch_migration_bitexact": True}, indent=2))
print("PASS exact training continuation, safe accumulation cancellation and resume-plan validation", flush=True)
