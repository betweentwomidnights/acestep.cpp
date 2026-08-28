# Adapters

Place your adapter files or folders here. Three adapter formats are supported
and automatically detected from the safetensors payload. PEFT directories can
contain LoRA or DoRA-row adapters emitted by `ace-train`.

| Trainer | Layout                | Alpha source                          | Example                                |
|---------|-----------------------|---------------------------------------|----------------------------------------|
| PEFT    | folder                | `lora_alpha` in `adapter_config.json` | `ACE-Step-v1.5-chinese-new-year-LoRA/` |
| ComfyUI | single `.safetensors` | per tensor `.alpha` scalar            | `turbo_v9_1850_comfyui.safetensors`    |
| LyCORIS | single `.safetensors` | per module `.alpha` scalar            | `acestep-qinglong-lokr.safetensors`    |

LyCORIS LoKr handles factorized (`lokr_w2_a` + `lokr_w2_b`) or monolithic
(`lokr_w2`) weights, with optional DoRA via `dora_scale`.

Point the server at this folder:

```bash
./build/ace-server --models ./models --adapters ./adapters
```

## Functional LoRA / DoRA inference

For quantized bases, use `ace-synth --adapter-mode functional` to retain the
adapter correction in floating point instead of merging and requantizing it:

```bash
./build/ace-synth --models models --adapters adapters \
  --adapter-mode functional --request request.json
```

`ace-server` accepts the same `--adapter-mode functional` option. Its mode is
server-wide and is reported by `GET /props` as `cli.adapter_mode`; requests still
select the adapter and scale. The supplied server launch scripts opt into this mode.

This supports PEFT LoRA/DoRA directories and compatible single-file LoRA/DoRA
safetensors, including heterogeneous ranks and alphas. Base weights remain
unchanged. DoRA normalization and scaled low-rank factors are prepared once at
load time; each denoising step adds only low-rank matrix products and row scaling.
By default (`--adapter-strength-mode delta`), the `adapter_scale` request field retains merge-mode semantics:
`W_effective = W_base + strength * (W_DoRA - W_base)`. Zero is a no-op; strengths
above one extrapolate the full adapter update, including the magnitude change.

For SA3/PEFT-style DoRA strength, opt into normalized mode:

```bash
./build/ace-synth --models models --adapters adapters \
  --adapter-mode functional --adapter-strength-mode normalized --request request.json
```

With `delta = (alpha / rank) * B @ A`, normalized mode computes
`W_effective = magnitude * normalize_rows(W_base + strength * delta)`.
Strength is inside normalization, so nonzero strengths retain the learned row
magnitudes. Both modes are identical at strength one. Exactly zero explicitly
bypasses the adapter; normalized mode is not a continuous dry/wet fade at zero
when learned magnitudes differ from base norms. Plain LoRA is unchanged by this
option. Normalization is still prepared only once, not on every denoising step.
This is an opt-in inference experiment, not a guarantee that strengths above one
will sound better. It does not change training or add multi-adapter composition.

The default full-delta interpolation should not be confused with PEFT's runtime
`set_scale` semantics: PEFT scales the low-rank update before DoRA normalization.
Normalized mode currently requires functional inference; merge mode rejects it.
The C++ API exposes `AceSynthParams::normalized_adapter_strength`, which is also
included in the model-store cache key. CLI/API support only; no HTTP control yet.

The synthesis API exposes `AceSynthParams::functional_adapter`. The mode is part
of the model-store cache key and its adapter memory is counted in VRAM accounting.
`--adapter-mode merge` remains the default for compatibility and comparison.
LoKr/LyCORIS weight-decomposition adapters still require merge mode; functional
mode rejects them explicitly rather than silently falling back to a lossy merge.

## Portable single-file conversion

PEFT directories work directly and are the preferred editable/training format.
To package one as a config-free single file, preserving per-module rank/alpha
patterns and DoRA magnitudes:

```bash
python tools/ace-lora-convert.py \
  --input path/to/peft-adapter \
  --output adapters/my-adapter.safetensors
```

The output uses `diffusion_model.*` tensor names and embeds a scalar alpha for
every adapted module. It can be selected from `--adapters` like any other
single-file adapter.
