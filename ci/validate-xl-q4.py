#!/usr/bin/env python3
"""Real-model XL Q4 LoRA train/resume/inference smoke; never a quality gate.

Example: python ci/validate-xl-q4.py --bin-dir build-release-gpu-local/Release
  --models models --backend CUDA0 --output build-release-validation/cuda-xl
Outputs commands, logs, resolved checkpoint hashes and a JSON result. Use a new
output directory for each run. Only the small synthetic fixture is generated.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import wave


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checkpoint(root):
    pointer = root / "checkpoint.current"
    target = Path(os.readlink(pointer) if pointer.is_symlink() else pointer.read_text())
    assert not target.is_absolute() and target.parts[0] == ".checkpoint-generations" and len(target.parts) == 2
    assert target.name.startswith("generation-")
    return root / target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--backend", required=True, help="Exact GGML device, e.g. CUDA0, Vulkan1 or MTL0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--profile", choices=["attention", "balanced"], default="balanced")
    p.add_argument("--dit", default="acestep-v15-xl-base-Q4_K_M.gguf")
    p.add_argument("--text-encoder", default="Qwen3-Embedding-0.6B-Q8_0.gguf")
    p.add_argument("--vae", default="vae-BF16.gguf")
    a = p.parse_args()
    a.output = a.output.resolve()
    a.bin_dir = a.bin_dir.resolve()
    a.models = a.models.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    dataset = a.output / "dataset"
    dataset.mkdir()
    # A restricted registry makes both training and inference use the same
    # explicit files; hard links avoid duplicating multi-GB model weights.
    models = a.output / "models"
    models.mkdir()
    for name in [a.dit, a.text_encoder, a.vae]:
        os.link(a.models / name, models / name)
    dit_hash = sha256(a.models / a.dit)
    with wave.open(str(dataset / "smoke.wav"), "wb") as w:
        w.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        w.writeframes(b"".join(struct.pack("<hh", int(1200 * math.sin(2 * math.pi * 220 * i / 48000)),
                                         int(1000 * math.sin(2 * math.pi * 330 * i / 48000)))
                                for i in range(48000 * 4)))
    (dataset / "smoke.txt").write_text(
        "caption: Soft sustained instrumental tones.\ngenre: ambient\nbpm: 120\nkey: C major\n"
        "signature: 4\nis_instrumental: true\ncustom_tag: smoke\nlyrics: [Instrumental]\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACE_")}
    env["GGML_BACKEND"] = a.backend
    suffix = ".exe" if os.name == "nt" else ""
    repo = Path(__file__).resolve().parents[1]
    results = {"application_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
               "ggml_commit": subprocess.check_output(["git", "-C", str(repo / "ggml"), "rev-parse", "HEAD"], text=True).strip(),
               "backend": a.backend, "rank": a.rank, "profile": a.profile, "adapter_type": "lora",
               "dit": a.dit, "dit_sha256": dit_hash, "fixture_seconds": 4, "steps": {}}

    def run(label, tool, args):
        command = [str(a.bin_dir / (tool + suffix)), *map(str, args)]
        print(label, flush=True)
        start = time.monotonic()
        with (a.output / f"{label}.log").open("w") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=a.output, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        assert result.returncode == 0, f"{label} failed; see {a.output / (label + '.log')}"
        results["steps"][label] = {"seconds": round(time.monotonic() - start, 3), "command": command}

    run("props", "ace-server", ["--props", "--device", a.backend])
    # CLI logs the command on line 1; JSON document follows backend stderr.
    props_text = (a.output / "props.log").read_text()
    props = json.loads(props_text[props_text.index("{\n"):])
    assert props["hardware"]["selected_device"] == a.backend
    assert props["hardware"]["accelerator_available"], "CPU is not GPU validation"
    results["props"] = props
    common = ["--models", models, "--dataset", dataset, "--dit", a.dit, "--text-encoder", a.text_encoder,
              "--vae", a.vae, "--adapter-type", "lora", "--profile", a.profile, "--rank", a.rank,
              "--alpha", a.rank * 2, "--epochs", "2", "--batch-size", "1", "--accumulation", "1",
              "--warmup-steps", "0", "--checkpoint-every", "1", "--learning-rate", "0.0001",
              "--window-min-seconds", "2", "--window-max-seconds", "2", "--min-snr", "5", "--seed", "42"]
    full, resumed = a.output / "full", a.output / "resumed"
    run("train", "ace-train", [*common, "--output", full])
    run("resume", "ace-train", [*common, "--resume", full / "checkpoint-epoch-1", "--output", resumed])
    files = ["adapter_model.safetensors", "adapter_config.json", "optimizer_state.safetensors", "trainer_state.json"]
    results["checkpoint_hashes"] = {}
    for name in files:
        left, right = sha256(checkpoint(full) / name), sha256(checkpoint(resumed) / name)
        assert left == right, f"resume diverged: {name}"
        results["checkpoint_hashes"][name] = left
    assert json.loads((checkpoint(full) / "trainer_state.json").read_text())["optimizer_step"] == 2
    req = {"caption": "Soft sustained instrumental tones.", "lyrics": "[Instrumental]", "duration": 4,
           "bpm": 120, "seed": 42, "inference_steps": 2, "guidance_scale": 1.0, "shift": 1.0,
           "synth_model": a.dit, "output_format": "wav16"}
    for label in ["base", "adapted"]:
        if label == "adapted":
            req["adapter"] = "full"
            req["adapter_scale"] = 1.0
        path = a.output / f"{label}.json"
        path.write_text(json.dumps(req))
        run(label, "ace-synth", ["--models", models, "--adapters", a.output,
                                 "--adapter-mode", "functional", "--request", path])
        with wave.open(str(a.output / f"{label}0.wav")) as w:
            assert w.getframerate() == 48000 and w.getnchannels() == 2 and w.getnframes() == 4 * 48000
            assert any(w.readframes(w.getnframes())), "silent output"
    assert sha256(a.output / "base0.wav") != sha256(a.output / "adapted0.wav"), "adapter produced no output difference"
    assert sha256(a.models / a.dit) == dit_hash, "frozen Q4 base changed"
    for label in ["train", "resume", "base", "adapted"]:
        assert f"backend: {a.backend}" in (a.output / f"{label}.log").read_text(), f"device evidence missing: {label}"
    results["passed"] = True
    results["limits"] = "Synthetic two-step execution/resume smoke, not convergence, Torch parity or full-duration capacity."
    (a.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS: XL Q4 LoRA train/resume/functional inference on {a.backend}; {a.output / 'summary.json'}")


if __name__ == "__main__":
    main()
