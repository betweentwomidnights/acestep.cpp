#!/usr/bin/env python3
"""Compare one deterministic native ACE-Step training step with PyTorch.

The native fixture executable owns randomness and writes its exact noisy input,
conditioning, target, initial DoRA adapter, gradients, and AdamW result. This
script feeds those tensors to the full-precision PyTorch decoder and compares
the forward result, loss, every adapter gradient, and every updated parameter.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Callable

import numpy as np


def load_dump(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        ndim_raw = handle.read(4)
        if len(ndim_raw) != 4:
            raise RuntimeError(f"invalid dump header: {path}")
        ndim = struct.unpack("i", ndim_raw)[0]
        shape = struct.unpack(f"{ndim}i", handle.read(4 * ndim))
        data = np.frombuffer(handle.read(), dtype=np.float32).copy()
    expected = math.prod(shape)
    if data.size != expected:
        raise RuntimeError(f"dump shape/data mismatch: {path}: {shape} vs {data.size}")
    return data.reshape(shape)


def canonical_adapter_key(name: str) -> str:
    name = name.removeprefix("decoder.")
    name = name.replace(".lora_A.default.weight", ".lora_A.weight")
    name = name.replace(".lora_B.default.weight", ".lora_B.weight")
    name = name.replace(".lora_magnitude_vector.default.weight", ".lora_magnitude_vector")
    name = name.replace(".lora_magnitude_vector.default", ".lora_magnitude_vector")
    name = name.replace(".lora_magnitude_vector.weight", ".lora_magnitude_vector")
    return name


def flat_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    a = reference.astype(np.float64, copy=False).reshape(-1)
    b = candidate.astype(np.float64, copy=False).reshape(-1)
    if a.shape != b.shape:
        raise RuntimeError(f"tensor shape mismatch: {reference.shape} vs {candidate.shape}")
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    diff_norm = np.linalg.norm(a - b)
    cosine = float(np.dot(a, b) / (a_norm * b_norm)) if a_norm and b_norm else float(a_norm == b_norm)
    return {
        "cosine": cosine,
        "relative_l2": float(diff_norm / max(a_norm, 1e-30)),
        "max_abs": float(np.max(np.abs(a - b))) if a.size else 0.0,
        "reference_norm": float(a_norm),
        "candidate_norm": float(b_norm),
    }


def aggregate_metrics(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
                      predicate: Callable[[str], bool]) -> dict[str, float]:
    keys = sorted(key for key in reference if key in candidate and predicate(key))
    if not keys:
        return {"count": 0, "cosine": 0.0, "relative_l2": None, "max_abs": None}
    a = np.concatenate([reference[key].astype(np.float32, copy=False).reshape(-1) for key in keys])
    b = np.concatenate([candidate[key].astype(np.float32, copy=False).reshape(-1) for key in keys])
    result = flat_metrics(a, b)
    result["count"] = len(keys)
    return result


def run_native(args: argparse.Namespace) -> None:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["GGML_BACKEND"] = args.backend
    command = [str(args.native_exe), str(args.gguf), str(args.work_dir), args.adapter_type]
    if args.nonzero_b:
        command.append("nonzero-b")
    print("[Parity] Native:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True, env=env)


def load_pytorch_model(args: argparse.Namespace):
    if args.carey_root:
        sys.path.insert(0, str(args.carey_root))
        from acestep.training_v2.model_loader import load_decoder_for_training
        return load_decoder_for_training(
            checkpoint_dir=args.checkpoint_root, variant=args.variant,
            device=args.device, precision=args.precision,
        )
    sys.path.insert(0, str(args.sidestep_root))
    from sidestep_engine.models.loader import load_decoder_for_training

    return load_decoder_for_training(
        checkpoint_dir=args.checkpoint_root,
        variant=args.variant,
        device=args.device,
        precision=args.precision,
    )


def adapter_tensors(model) -> dict[str, object]:
    result = {}
    for name, parameter in model.decoder.named_parameters():
        if parameter.requires_grad:
            result[canonical_adapter_key(name)] = parameter
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-exe", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sidestep-root", type=Path,
                        help="Side-Step checkout containing sidestep_engine")
    source.add_argument("--carey-root", type=Path, help="Gary services/carey containing acestep")
    parser.add_argument("--adapter-type", choices=("lora", "dora-rows"), default="dora-rows")
    parser.add_argument("--adapter-precision", choices=("fp32", "model"), default="fp32",
                        help="FP32 isolates native math; model reproduces Carey's post-injection dtype cast")
    parser.add_argument("--nonzero-b", action="store_true", help="Exercise nonzero A gradients after adapter initialization")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--backend", default="CUDA0")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()

    args.native_exe = args.native_exe.resolve()
    args.gguf = args.gguf.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    if args.sidestep_root:
        args.sidestep_root = args.sidestep_root.resolve()
    if args.carey_root:
        args.carey_root = args.carey_root.resolve()
    args.work_dir = args.work_dir.resolve()
    if not args.skip_native:
        run_native(args)

    import torch
    import torch.nn.functional as functional
    from peft import PeftModel
    from safetensors.torch import load_file

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    print(f"[Parity] Loading PyTorch {args.variant} decoder", flush=True)
    model = load_pytorch_model(args)
    initial_dir = args.work_dir / "initial"
    model.decoder = PeftModel.from_pretrained(
        model.decoder,
        str(initial_dir),
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    model.decoder.train()
    if hasattr(model.decoder, "config"):
        model.decoder.config.use_cache = False

    trainable = adapter_tensors(model)
    native_initial = {key: value.float().cpu().numpy() for key, value in load_file(
        str(initial_dir / "adapter_model.safetensors"), device="cpu").items()}
    missing = sorted(set(native_initial) - set(trainable))
    extra = sorted(set(trainable) - set(native_initial))
    if missing or extra:
        raise RuntimeError(f"adapter inventory mismatch; missing={missing[:5]}, extra={extra[:5]}")
    # PEFT first casts loaded weights to the base dtype. Restore the exact native
    # initial values for a controlled FP32 comparison; keep model precision when
    # measuring Carey's actual post-injection model.to(module.dtype) policy.
    if args.adapter_precision == "fp32":
        for key, parameter in trainable.items():
            parameter.data = torch.from_numpy(native_initial[key].copy()).to(device=parameter.device, dtype=torch.float32)
    initial = {key: value.detach().float().cpu().numpy().copy() for key, value in trainable.items()}

    combined = torch.from_numpy(load_dump(args.work_dir / "input_latents.bin")).to(args.device)
    encoder_hidden = torch.from_numpy(load_dump(args.work_dir / "encoder_hidden.bin")).to(args.device)
    target = torch.from_numpy(load_dump(args.work_dir / "target_velocity.bin")).to(args.device)
    timesteps = torch.from_numpy(load_dump(args.work_dir / "timesteps.bin")).to(args.device)
    native_velocity = load_dump(args.work_dir / "native_velocity.bin")
    native_loss = float(load_dump(args.work_dir / "native_loss.bin").reshape(-1)[0])

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    output_channels = target.shape[-1]
    context_channels = combined.shape[-1] - output_channels
    context = combined[..., :context_channels].to(dtype=dtype)
    noisy = combined[..., context_channels:].to(dtype=dtype)
    encoder_hidden = encoder_hidden.to(dtype=dtype)
    target = target.to(dtype=dtype)
    timesteps = timesteps.to(dtype=dtype)
    attention_mask = torch.ones(noisy.shape[:2], device=args.device, dtype=dtype)
    encoder_attention_mask = torch.ones(encoder_hidden.shape[:2], device=args.device, dtype=dtype)

    autocast = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if args.device.startswith("cuda") and dtype in (torch.bfloat16, torch.float16)
        else nullcontext()
    )
    with autocast:
        prediction = model.decoder(
            hidden_states=noisy,
            timestep=timesteps,
            timestep_r=timesteps,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context,
        )[0]
        per_sample = functional.mse_loss(prediction, target, reduction="none").mean(dim=(-1, -2)).float()
        if args.carey_root:
            from acestep.training_v2.loss_weighting import flow_min_snr_weights
            weights = flow_min_snr_weights(timesteps, gamma=5.0)
        else:
            t_f32 = timesteps.float().clamp(min=1e-4, max=1.0 - 1e-4)
            snr = (((1.0 - t_f32) / t_f32) ** 2).clamp(max=1e6)
            weights = torch.clamp(snr, max=5.0) / snr.clamp(min=1e-6)
        loss = (weights * per_sample).mean().float()

    loss.backward()
    python_velocity = prediction.detach().float().cpu().numpy()
    python_gradients = {
        key: parameter.grad.detach().float().cpu().numpy().copy()
        for key, parameter in trainable.items()
    }
    native_gradients = {
        key: value.float().cpu().numpy()
        for key, value in load_file(str(args.work_dir / "native_gradients.safetensors"), device="cpu").items()
    }

    native_metrics = json.loads((args.work_dir / "native_metrics.json").read_text(encoding="utf-8"))
    parameters = list(trainable.values())
    python_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, native_metrics["max_gradient_norm"]))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=native_metrics["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=native_metrics["weight_decay"],
        foreach=False,
        fused=False,
    )
    if args.carey_root:
        from acestep.training_v2.optim import build_optimizer
        optimizer = build_optimizer(parameters, lr=native_metrics["learning_rate"],
                                    weight_decay=native_metrics["weight_decay"],
                                    device_type=torch.device(args.device).type)
    optimizer.step()
    python_updated = {key: value.detach().float().cpu().numpy().copy() for key, value in trainable.items()}
    native_updated = {
        key: value.float().cpu().numpy()
        for key, value in load_file(
            str(args.work_dir / "native_updated" / "adapter_model.safetensors"), device="cpu").items()
    }

    groups = {
        "lora_A": lambda key: ".lora_A." in key,
        "lora_B": lambda key: ".lora_B." in key,
        "magnitude": lambda key: "magnitude" in key,
        "all": lambda _key: True,
    }
    report = {
        "reference": "carey" if args.carey_root else "sidestep",
        "adapter_type": args.adapter_type,
        "adapter_precision": args.adapter_precision,
        "nonzero_b": args.nonzero_b,
        "torch_version": torch.__version__,
        "torch_device": str(next(iter(trainable.values())).device),
        "trainable_dtypes": sorted({str(parameter.dtype) for parameter in trainable.values()}),
        "precision": args.precision,
        "native_backend": args.backend,
        "initial": {name: aggregate_metrics(native_initial, initial, pred) for name, pred in groups.items()},
        "forward": flat_metrics(native_velocity, python_velocity),
        "loss": {
            "native": native_loss,
            "pytorch": float(loss.detach().cpu()),
            "relative_error": abs(native_loss - float(loss.detach().cpu())) / max(abs(native_loss), 1e-30),
        },
        "gradients": {name: aggregate_metrics(native_gradients, python_gradients, pred) for name, pred in groups.items()},
        "gradient_norm": {
            "native": native_metrics["gradient_norm"],
            "pytorch": python_gradient_norm,
            "relative_error": abs(native_metrics["gradient_norm"] - python_gradient_norm)
                              / max(abs(native_metrics["gradient_norm"]), 1e-30),
        },
        "updated": {name: aggregate_metrics(native_updated, python_updated, pred) for name, pred in groups.items()},
        "parameter_deltas": {name: aggregate_metrics(
            {key: native_updated[key] - native_initial[key] for key in native_initial},
            {key: python_updated[key] - initial[key] for key in initial}, pred)
            for name, pred in groups.items()},
    }
    if args.carey_root:
        names = ['training_v2/model_loader.py', 'training_v2/loss_weighting.py',
                 'training_v2/optim.py', 'training_v2/trainer_fixed.py']
        report['carey_sources'] = {name: hashlib.sha256((args.carey_root / 'acestep' / name).read_bytes()).hexdigest()
                                   for name in names}
    # Full-model BF16 implementations need not be bit-identical, but a real
    # parity pass must preserve direction through forward and non-zero grads.
    passed = (
        report["forward"]["cosine"] >= 0.99
        and report["initial"]["all"]["max_abs"] == 0
        and report["loss"]["relative_error"] <= 0.10
        and report["gradients"]["lora_B"]["cosine"] >= 0.95
        and (not args.nonzero_b or report["gradients"]["lora_A"]["cosine"] >= 0.95)
        and report["parameter_deltas"]["lora_B"]["cosine"] >= 0.90
        and (args.adapter_type == "lora" or (
            report["gradients"]["magnitude"]["cosine"] >= 0.95
            and report["parameter_deltas"]["magnitude"]["cosine"] >= 0.90))
    )
    report["passed"] = bool(passed)
    (args.work_dir / "parity_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[Parity] Report: {args.work_dir / 'parity_report.json'}")
    print("PASS" if passed else "FAIL", "PyTorch/native one-step parity")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
