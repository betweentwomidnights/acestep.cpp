#!/usr/bin/env python3
"""Opt-in shared-feature condition encoder replay against Carey (regular base).

Accepts a test-train-preprocess-real sample directory. Native features are kept
fixed while comparing arithmetic and silence/zero timbre policies separately.
Only the condition encoder executes on Torch CUDA; unused components are removed
after the production factory loads on CPU. No checkpoint/data is downloaded.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("step_parity", Path(__file__).with_name("test-train-step-parity.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def dump(path, value):
    value = np.ascontiguousarray(value, dtype=np.float32)
    path.write_bytes(struct.pack(f"{value.ndim + 1}i", value.ndim, *value.shape) + value.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["carey-root", "checkpoint-dir", "native-exe", "gguf", "features", "work-dir"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default="CUDA0")
    p.add_argument("--torch-attention", choices=["auto", "math"], default="auto")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--native-no-fa", action="store_true")
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    text = metrics.load_dump(args.features / "text_hidden.bin")
    lyric = metrics.load_dump(args.features / "lyric_embed.bin")
    timbre = metrics.load_dump(args.features / "context.bin")[:1, :64].copy()
    native = {}
    for policy in ["silence", "zero"]:
        inputs = args.work_dir / (policy + "-features")
        inputs.mkdir()
        dump(inputs / "text.bin", text)
        dump(inputs / "lyric.bin", lyric)
        dump(inputs / "timbre.bin", timbre if policy == "silence" else np.zeros_like(timbre))
        output = args.work_dir / (policy + "-native")
        with (args.work_dir / (policy + "-native.log")).open("w") as log:
            subprocess.run([str(args.native_exe), str(args.gguf), str(inputs), str(output)] + (["no-fa"] if args.native_no_fa else []),
                           env=dict(os.environ, GGML_BACKEND=args.backend), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        native[policy] = metrics.load_dump(output / "encoder.bin")
    sys.path.insert(0, str(args.carey_root))
    import torch
    from acestep.training_v2 import model_loader
    from acestep.training.dataset_builder_modules.preprocess_encoder import run_encoder
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    model = model_loader.load_decoder_for_training(args.checkpoint_dir, "base", "cpu", "fp32")
    del model.decoder, model.tokenizer, model.detokenizer
    gc.collect()
    original = metrics.load_dump(args.features / "encoder.bin")
    report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "torch": torch.__version__, "residency_override": "production CPU factory; condition encoder only on CUDA",
              "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "ggml_environment": {key: value for key, value in os.environ.items() if key.startswith("GGML_")},
              "native_features": {"text_tokens": len(text), "lyric_tokens": len(lyric)},
              "native_production_replay_bitexact": bool(np.array_equal(original, native["silence"])),
              "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          [Path(model_loader.__file__), Path(sys.modules[run_encoder.__module__].__file__), Path(__file__)]},
              "modes": {}}
    cuts = {"lyric": slice(0, len(lyric)), "timbre": slice(len(lyric), len(lyric) + 1), "text": slice(len(lyric) + 1, None)}
    report["native_zero_vs_silence"] = {key: metrics.flat_metrics(native["silence"][part], native["zero"][part]) for key, part in cuts.items()}
    for precision, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        model.to(device=args.device, dtype=dtype)
        outputs = {}
        for policy in ["silence", "zero"]:
            def tensor(value):
                return torch.from_numpy(value).unsqueeze(0).to(device=args.device, dtype=dtype)
            arguments = (model, tensor(text), torch.ones(1, len(text), device=args.device, dtype=dtype),
                tensor(lyric), torch.ones(1, len(lyric), device=args.device, dtype=dtype), args.device, dtype,
                tensor(timbre if policy == "silence" else np.zeros_like(timbre)))
            def attention_context():
                return torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if args.torch_attention == "math" else nullcontext()
            with attention_context():
                encoded, mask = run_encoder(*arguments)
                output = encoded[0].float().cpu().numpy()
                repeated, _ = run_encoder(*arguments)
                repeat = repeated[0].float().cpu().numpy()
            assert bool(mask.all()) and output.shape == native[policy].shape
            outputs[policy] = output
            row = {"shape": list(output.shape), "all_finite": bool(np.isfinite(output).all()),
                   "repeat_bitexact": bool(np.array_equal(output, repeat)),
                   "repeat": metrics.flat_metrics(output, repeat),
                   "shared_input_cast": precision, "full": metrics.flat_metrics(native[policy], output)}
            row.update({key: metrics.flat_metrics(native[policy][part], output[part]) for key, part in cuts.items()})
            report["modes"][precision + "-" + policy] = row
            np.save(args.work_dir / (precision + "-" + policy + ".npy"), output)
            print(precision, policy, json.dumps(row), flush=True)
        report[precision + "_zero_vs_silence"] = {key: metrics.flat_metrics(outputs["silence"][part], outputs["zero"][part]) for key, part in cuts.items()}
        (args.work_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
