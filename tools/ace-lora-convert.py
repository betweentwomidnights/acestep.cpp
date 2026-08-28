#!/usr/bin/env python3
"""Package a PEFT LoRA/DoRA directory as one ACE-Step safetensors file.

The native runtime already accepts PEFT directories. This converter is for
portable, config-free distribution: it writes the single-file layout consumed
by adapter-merge.h and serializes each module's effective alpha as a scalar.
That preserves heterogeneous rank/alpha maps such as Fisher-ranked adapters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


FACTOR_RE = re.compile(
    r"^(?:base_model\.model\.|diffusion_model\.)?"
    r"(?P<module>.+)\.(?P<kind>lora_A|lora_B)"
    r"(?:\.default)?\.weight$"
)
MAGNITUDE_RE = re.compile(
    r"^(?:base_model\.model\.|diffusion_model\.)?"
    r"(?P<module>.+)\.lora_magnitude_vector"
    r"(?:\.default)?(?:\.weight)?$"
)


def pattern_matches(module: str, pattern: str) -> bool:
    if not pattern:
        return False
    if pattern.startswith("^"):
        return module == pattern[1:]
    return module == pattern or module.endswith("." + pattern)


def pattern_value(pattern: dict[str, int], module: str, fallback: int) -> int:
    for key, value in pattern.items():
        if pattern_matches(module, key):
            return int(value)
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PEFT LoRA/DoRA directory to portable ACE-Step safetensors."
    )
    parser.add_argument("--input", required=True, type=Path, help="PEFT adapter directory")
    parser.add_argument("--output", required=True, type=Path, help="output .safetensors file")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.input.resolve()
    output_path = args.output.resolve()
    config_path = source_dir / "adapter_config.json"
    weights_path = source_dir / "adapter_model.safetensors"

    if not source_dir.is_dir():
        raise SystemExit(f"input is not a directory: {source_dir}")
    if not config_path.is_file() or not weights_path.is_file():
        raise SystemExit("input must contain adapter_config.json and adapter_model.safetensors")
    if output_path.suffix.lower() != ".safetensors":
        raise SystemExit("output must end in .safetensors")
    if output_path.exists() and not args.force:
        raise SystemExit(f"output already exists (pass --force to replace it): {output_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_rank = int(config.get("r", 0))
    base_alpha = int(config.get("lora_alpha", 0))
    use_dora = bool(config.get("use_dora", False))
    rank_pattern = {str(k): int(v) for k, v in (config.get("rank_pattern") or {}).items()}
    alpha_pattern = {str(k): int(v) for k, v in (config.get("alpha_pattern") or {}).items()}
    target_modules = [str(value) for value in (config.get("target_modules") or [])]
    if base_rank <= 0 or base_alpha <= 0 or not target_modules:
        raise SystemExit("adapter config must define positive r/lora_alpha and target_modules")

    source = load_file(str(weights_path), device="cpu")
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in source.items():
        factor = FACTOR_RE.match(key)
        if factor:
            grouped.setdefault(factor.group("module"), {})[factor.group("kind")] = tensor
            continue
        magnitude = MAGNITUDE_RE.match(key)
        if magnitude:
            grouped.setdefault(magnitude.group("module"), {})["magnitude"] = tensor
            continue
        raise SystemExit(f"unsupported adapter tensor key: {key}")

    if not grouped:
        raise SystemExit("adapter contains no LoRA tensors")
    for target in target_modules:
        if not any(pattern_matches(module, target) for module in grouped):
            raise SystemExit(f"target module has no serialized tensors: {target}")

    converted: dict[str, torch.Tensor] = {}
    dora_count = 0
    for module in sorted(grouped):
        tensors = grouped[module]
        if "lora_A" not in tensors or "lora_B" not in tensors:
            raise SystemExit(f"module is missing lora_A or lora_B: {module}")
        a = tensors["lora_A"]
        b = tensors["lora_B"]
        if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
            raise SystemExit(f"invalid LoRA factor shapes for {module}: A={tuple(a.shape)} B={tuple(b.shape)}")

        saved_rank = pattern_value(rank_pattern, module, base_rank)
        if a.shape[0] != saved_rank:
            raise SystemExit(f"rank mismatch for {module}: tensor={a.shape[0]} config={saved_rank}")
        magnitude = tensors.get("magnitude")
        if use_dora != (magnitude is not None):
            raise SystemExit(f"DoRA magnitude/config mismatch for {module}")
        if magnitude is not None:
            if magnitude.numel() != b.shape[0]:
                raise SystemExit(
                    f"DoRA magnitude shape mismatch for {module}: magnitude={magnitude.numel()} out={b.shape[0]}"
                )
            dora_count += 1

        stem = f"diffusion_model.{module}"
        converted[f"{stem}.lora_A.weight"] = a.contiguous()
        converted[f"{stem}.lora_B.weight"] = b.contiguous()
        if magnitude is not None:
            converted[f"{stem}.lora_magnitude_vector"] = magnitude.contiguous()
        converted[f"{stem}.alpha"] = torch.tensor(
            float(pattern_value(alpha_pattern, module, base_alpha)), dtype=torch.float32
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        converted,
        str(output_path),
        metadata={
            "format": "pt",
            "ace_adapter_format": "portable-single-file",
            "source_layout": "peft",
            "use_dora": "true" if use_dora else "false",
        },
    )
    print(
        f"wrote {output_path} ({len(grouped)} modules, {dora_count} DoRA, "
        f"{len(converted)} tensors)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
