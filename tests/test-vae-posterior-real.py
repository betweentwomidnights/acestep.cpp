#!/usr/bin/env python3
"""Opt-in real-audio VAE comparison; no model or dataset is downloaded.

Runs the native emitter and Carey's actual production tiler with identical
retained-core posterior noise. Effective-native modes are diagnostic overrides,
not stock Carey precision. Reports descriptive errors, not a training parity gate.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import types
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location("step_parity", Path(__file__).with_name("test-train-step-parity.py"))
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--carey-root", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--native-exe", type=Path, required=True)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=375)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--legacy-seed", type=int, default=42)
    p.add_argument("--legacy-target", type=Path)
    p.add_argument("--backend", default="CUDA0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--modes", nargs="+", default=["bf16", "fp32", "effective", "effective-f16-input"])
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.carey_root))
    import torch
    import torch.nn.functional as F
    from diffusers.models.autoencoders.autoencoder_oobleck import Snake1d
    from acestep.training_v2 import model_loader, preprocess_vae
    from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(6)
    audio, _ = load_audio_stereo(str(args.audio), 48000, 600)
    audio = audio.unsqueeze(0)
    epsilon = torch.from_numpy(np.random.default_rng(20261015).standard_normal(
        (audio.shape[-1] // 1920 + 64, 64)).astype(np.float32)).bfloat16().float().numpy()
    noise_file = args.work_dir / "noise.bin"
    noise_file.write_bytes(struct.pack("3i", 2, *epsilon.shape) + epsilon.tobytes())
    env = dict(os.environ, GGML_BACKEND=args.backend)
    native = args.work_dir / "native"
    with (args.work_dir / "native.log").open("w") as log:
        subprocess.run([str(args.native_exe), str(args.gguf), str(args.audio), str(noise_file), str(native),
                        str(args.chunk), str(args.overlap), str(args.legacy_seed)], env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = json.loads((native / "posterior.json").read_text())
    assert metadata["audio_samples"] == audio.shape[-1]
    native_samples = metrics.load_dump(native / "sampled.bin")
    native_parameters = [metrics.load_dump(native / tile["file"]) for tile in metadata["tiles"]]
    report = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "ggml_environment": {key: value for key, value in env.items() if key.startswith("GGML_")},
              "torch": torch.__version__, "native": metadata,
              "epsilon_policy": "PCG64 seed 20261015, BF16-representable values, native retained-core order",
              "epsilon_sha256": hashlib.sha256(epsilon.tobytes()).hexdigest(),
              "audio_sha256": hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
              "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                          [Path(model_loader.__file__), Path(preprocess_vae.__file__), Path(__file__)]},
              "modes": {}}
    if args.legacy_target:
        old = metrics.load_dump(args.legacy_target)
        legacy = metrics.load_dump(native / "legacy-seeded.bin")
        report["legacy_seeded_target_bitexact"] = bool(np.array_equal(old[:len(legacy)], legacy))
        assert report["legacy_seeded_target_bitexact"], "default RNG/output changed"

    for mode in args.modes:
        if mode not in {"bf16", "fp32", "effective", "effective-f16-input", "effective-f16-conv"}:
            raise ValueError(mode)
        start = time.monotonic()
        vae = model_loader.load_vae(args.checkpoint_dir, args.device, "bf16" if mode == "bf16" else "fp32")
        if mode.startswith("effective"):
            for name, module in vae.named_modules():
                if not name.startswith("encoder."):
                    continue
                prefix = native / "weights" / name
                if isinstance(module, torch.nn.Conv1d):
                    torch.nn.utils.remove_weight_norm(module)
                    with torch.no_grad():
                        for key in ["weight", "bias"]:
                            getattr(module, key).copy_(torch.from_numpy(metrics.load_dump(Path(str(prefix) + f".{key}.bin"))).to(args.device))
                    if mode in {"effective-f16-input", "effective-f16-conv"}:
                        round_output = mode == "effective-f16-conv"
                        def conv_forward(self, x):
                            if self.reference_round_output:
                                value = F.conv1d(x.half(), self.weight.half(), None,
                                                 self.stride, self.padding, self.dilation, self.groups).float()
                                return value + self.bias[None, :, None]
                            return F.conv1d(x.half().float(), self.weight, self.bias,
                                            self.stride, self.padding, self.dilation, self.groups)
                        module.reference_round_output = round_output
                        module.forward = types.MethodType(conv_forward, module)
                elif isinstance(module, Snake1d):
                    for key in ["alpha_effective", "inverse_beta_effective"]:
                        value = torch.from_numpy(metrics.load_dump(Path(str(prefix) + f".{key}.bin"))).reshape(1, -1, 1).to(args.device)
                        module.register_buffer(key, value)
                    def snake_forward(self, x):
                        return x + self.inverse_beta_effective * torch.sin(self.alpha_effective * x).square()
                    module.forward = types.MethodType(snake_forward, module)
        original_encode = vae.encode
        captured = []
        def encode(chunk):
            index = len(captured)
            tile = metadata["tiles"][index]
            assert chunk.shape[-1] == tile["audio_count"], (index, chunk.shape, tile)
            expected_audio = audio[:, :, tile["audio_offset"]:tile["audio_offset"] + tile["audio_count"]].to(args.device, dtype=vae.dtype)
            assert torch.equal(chunk, expected_audio), "tile audio/precision mismatch"
            result = original_encode(chunk)
            dist = result.latent_dist
            assert dist.mean.shape[-1] == tile["raw_frames"], (index, dist.mean.shape, tile)
            captured.append({"parameters": dist.parameters.float().cpu().numpy()[0],
                             "std": dist.std.float().cpu().numpy()[0]})
            record = captured[-1]
            original_sample = dist.sample
            def capture_sample(*a, **kw):
                value = original_sample(*a, **kw)
                record["sample"] = value.float().cpu().numpy()[0]
                return value
            dist.sample = capture_sample
            return result
        vae.encode = encode
        def shared_noise(shape, generator=None, device=None, dtype=None, **kwargs):
            tile = metadata["tiles"][len(captured) - 1]
            value = torch.zeros(shape, device=device, dtype=dtype)
            start, count, trim = tile["write_start"], tile["core_frames"], tile["trim_start"]
            value[0, :, trim:trim + count] = torch.from_numpy(epsilon[start:start + count].T.copy()).to(device=device, dtype=dtype)
            return value
        torch.cuda.reset_peak_memory_stats() if args.device.startswith("cuda") else None
        with patch("diffusers.models.autoencoders.autoencoder_oobleck.randn_tensor", shared_noise), torch.inference_mode():
            production_result = preprocess_vae.tiled_vae_encode(vae, audio, vae.dtype,
                chunk_size=args.chunk * 1920, overlap=args.overlap * 1920)[0].float().cpu().numpy()
        assert len(captured) == len(metadata["tiles"])
        # Preserve/report production's result even when its legacy rounded
        # overlap trims lose a frame. Align retained cores separately, using
        # actual distribution.sample() results and native's observed bounds.
        result = np.concatenate([pt["sample"][:, tile["trim_start"]:tile["trim_start"] + tile["core_frames"]].T
                                 for tile, pt in zip(metadata["tiles"], captured)])
        assert result.shape == native_samples.shape, (result.shape, native_samples.shape)
        nmean, pmean, nstd, pstd, pscale, nscale = [], [], [], [], [], []
        seams = np.zeros(len(result), dtype=bool)
        for tile, raw, pt in zip(metadata["tiles"], native_parameters, captured):
            lo, count, at = tile["trim_start"], tile["core_frames"], tile["write_start"]
            core = slice(lo, lo + count)
            nmean.append(raw[:64, core].T)
            pmean.append(pt["parameters"][:64, core].T)
            nscale.append(raw[64:, core].T)
            pscale.append(pt["parameters"][64:, core].T)
            nstd.append((np.logaddexp(0, raw[64:, core]) + np.float32(1e-4)).T)
            pstd.append(pt["std"][:, core].T)
            if at:
                seams[max(0, at - 8):min(len(result), at + 8)] = True
        nmean, pmean, nstd, pstd, nscale, pscale = [np.concatenate(v) for v in [nmean,pmean,nstd,pstd,nscale,pscale]]
        row = {"frames": len(result), "seconds": time.monotonic() - start,
               "policy": "stock Carey" if mode in {"bf16", "fp32"} else "diagnostic: loaded native effective weights/Snake; FP32 pointwise ops" + ("; F16 convolution before F32 bias" if mode.endswith("conv") else "; F16-rounded convolution inputs" if mode.endswith("input") else ""),
               "mean": metrics.flat_metrics(nmean, pmean), "scale": metrics.flat_metrics(nscale, pscale),
               "std": metrics.flat_metrics(nstd, pstd), "sample": metrics.flat_metrics(native_samples, result),
               "sample_shared_math_fp32": metrics.flat_metrics(native_samples, pmean + pstd * epsilon[:len(result)]),
               "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else None,
               "all_finite": bool(np.isfinite(result).all())}
        row["production_frames"] = len(production_result)
        row["production_sample"] = metrics.flat_metrics(native_samples, production_result) if production_result.shape == native_samples.shape else None
        row["production_reassembly_matches_aligned"] = bool(np.array_equal(production_result, result))
        for label, mask in [("seams", seams), ("interior", ~seams)]:
            if mask.any():
                row[label] = metrics.flat_metrics(native_samples[mask], result[mask])
        np.savez(args.work_dir / f"{mode}.npz", sample=result, production_sample=production_result, mean=pmean, std=pstd, scale=pscale)
        report["modes"][mode] = row
        (args.work_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(mode, json.dumps(row), flush=True)
        del vae, original_encode, result, captured
        gc.collect()
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None


if __name__ == "__main__":
    main()
