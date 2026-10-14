# Carey training comparison, 2026-10-14

This is focused execution evidence and an explicit list of remaining policy
differences. It does **not** close the README's full gary4local/PyTorch parity
checkbox, demonstrate convergence, or validate Metal training.

References: Gary `79ff69b93e77d8abe573043e07d0628afcc7a5d9`, using its existing
Carey environment (Torch `2.7.1+cu128`, PEFT `0.18.1`); shared ggml remains
`4ad3b30bfe452efc5f4f9c0e111e017d9970292a`. The previously checked balanced
profile builder/injector is unchanged from Gary `238549ab`.
No Gary or ggml files are modified by this work.

## Controlled production-math fixtures

`test-train-carey-parity.cpp` calls native production batch preparation,
the shared full/checkpointed decoder loss graph, accumulated AdamW and LR
schedule functions. Its Python comparator imports Carey's actual
`FixedLoRAModule.training_step`, timestep/CFG helpers, optimizer/scheduler
factories and collator. Only decoder predictions and random inputs are
controlled: no model is loaded in this fixture.

The CPU FP32 checks pass for context/noisy-latent packing, CFG kept/dropped
conditions, two logit-normal draws followed by max and `r=t`, flat/Min-SNR
unmasked loss and prediction derivatives, and three AdamW updates including
global clipping, A/B/magnitude parameters and first/second moments. Constant,
linear and cosine schedules match when given the same effective positive
warmup. A scalar model also executes Carey's actual Fabric training loop with
a synchronous Fabric stand-in to measure final accumulation-window gradients;
checkpoint/TensorBoard/distributed side effects are excluded.

```powershell
& C:/dev/gary-localhost-installer/services/carey/env/Scripts/python.exe `
  tests/test-train-carey-parity.py `
  --native-exe build-release-cpu/Release/test-train-carey-parity.exe `
  --carey-root C:/dev/gary-localhost-installer/services/carey `
  --output build-release-validation/carey-focused-complete
```

Use a fresh output directory. The report includes hashes of every compared
Carey source file. The executable is built with the ordinary test targets;
the Python comparison is opt-in because CI does not contain Gary/Torch/models.

## Full base-decoder replay on CUDA

The real-model comparator now accepts `--carey-root` and both adapter types.
It uses Carey's model loader, loss-weight helper, PEFT implementation and AdamW
factory, replaying native input/target tensors with explicit BF16 compute casts
and controlled adapter tensors. Native and Torch
both actually execute on the RTX 5070 Laptop (8151 MiB, driver 616.56).
Native build: Windows MSVC Release, CUDA 12.8, dynamic backends, native CUDA
architecture, `GGML_MAX_NAME=128`, `GGML_NATIVE=OFF`.

Two modes distinguish math from storage precision:

- `--adapter-precision fp32` restores the exact native FP32 initial adapter
  after PEFT loading, while keeping the frozen base and Torch compute BF16.
- `--adapter-precision model` retains BF16 adapters, reproducing Carey's
  `self.module.model.to(self.module.dtype)` after injection. This mode is
  expected to fail the equivalence gate below; it measures the actual policy.

`--nonzero-b` seeds a small nonzero B so A gradients are exercised; a zero-B
initialization alone gives zero A gradients and cannot establish their parity.
The native fixtures use balanced reference rank 8/alpha 16, temporal length 8,
encoder length 8, Min-SNR gamma 5, LR `3e-5`, weight decay `0.01`, clip `1.0`.

| BF16 frozen base, FP32 adapters | Forward cosine | Relative loss error | A/B/magnitude gradient cosine | A/B/magnitude parameter-delta cosine |
| --- | --- | --- | --- | --- |
| LoRA, nonzero B | 0.9994682 | 0.0002490 | 0.998052 / 0.997694 / n/a | 0.961450 / 0.950826 / n/a |
| DoRA-row, nonzero B | 0.9994909 | 0.0011412 | 0.998083 / 0.997711 / 0.998324 | 0.961748 / 0.950845 / 0.950419 |

Both pass the directional gate (forward cosine >=0.99, loss relative error
<=0.10, nonzero gradient cosines >=0.95, B/magnitude delta cosine >=0.90),
with exact initial adapter values. These are deliberately **directional**
tolerances: delta relative L2 is approximately 0.28-0.32, so this is not tight
numerical equivalence. AdamW is sensitive to sign changes in small gradients.
The original zero-B DoRA fixture also passes with FP32 adapters.

With the real Carey BF16 adapter policy, both zero/nonzero-B DoRA fixtures have close
forward/gradient directions but **all magnitude updates round to zero** at
this fixture's learning rate; native FP32 magnitude delta norm is about `0.023738`.
Initial A/magnitude values are also rounded when loaded into BF16 storage.
The comparator reports this as a failure rather than hiding it behind the
large unchanged parameter norm. This is a precision-policy difference, not
evidence that native magnitude updates should be removed.

Example (fresh output):

```powershell
& C:/dev/gary-localhost-installer/services/carey/env/Scripts/python.exe `
  tests/test-train-step-parity.py `
  --native-exe build-release-gpu-local/Release/test-train-step-parity.exe `
  --gguf models/acestep-v15-base-BF16.gguf `
  --checkpoint-root C:/dev/gary-localhost-installer/services/carey/checkpoints `
  --carey-root C:/dev/gary-localhost-installer/services/carey `
  --work-dir build-release-validation/carey-full-base-bf16-dora-nonzero `
  --variant base --device cuda --precision bf16 --backend CUDA0 `
  --adapter-type dora-rows --adapter-precision fp32 --nonzero-b
```

Native base BF16 GGUF SHA256:
`ef880d77ffec8e6638f268c6237b73a8965af24bdd10fd5cfc27432c9d19ee57`.
Torch base safetensors SHA256:
`4177f600501a6d4bd81cadaa0abac557ffd15c54e5c8cb52053cdb24a0844d6b`.
Local reports/logs are under `build-release-validation/carey-full-base-bf16-*`.

## Policy differences retained explicitly

| Area | Actual difference | Native decision for this change |
| --- | --- | --- |
| Padding loss | Carey averages errors over every padded frame. Native weights only real audio frames, normalized per example. The controlled mixed-length fixture gives native `1.9590331` vs Carey `1.3914309`. | Preserve masked loss; do not train on padding merely to match the Python implementation. |
| Padding values | Both zero-pad targets and preserve real context/text regions. Carey zero-pads context and encoder features; native uses silence/context-mask ones and null condition. | Preserve native conditioning/masking; full-model variable-length parity remains open. |
| Warmup | Carey clamps requested warmup to at most `max(1, total_steps//10)`. Native clamps to total steps. | The host must pass an explicit effective budget. Do not change persisted native resume schedules. |
| Zero warmup | Carey's `LinearLR` construction with zero warmup still starts at 10% LR; native starts at the requested peak. At peak `0.001`, the first applied rates are `0.0001` vs `0.001`. | Preserve native zero-warmup semantics; record Python behavior rather than silently reproduce it. |
| Cosine restarts | Carey floors cycle length to an integer; native divides the remaining budget evenly among cycles. They differ when the budget is not divisible. | Keep the documented native schedule; migration must decide effective cycle semantics. |
| Accumulation | Native averages by actual example count. Carey divides every microbatch loss by the configured accumulation count, including an incomplete final window; mixed microbatch sizes also differ. The actual scalar Fabric loop gives preclip norms `[1, 0.5]` for three unit gradients at accumulation 2, instead of native `[1, 1]`. | Preserve native normalization. |
| Adapter storage | Native parameters and moments are FP32 over quantized or unquantized bases. Carey casts adapters to model precision before Fabric setup. | Preserve FP32 trainable state; downstream precision policy remains explicit. |
| Metrics LR | Carey reports scheduler LR after `scheduler.step()`. Native/SA3 history reports the LR actually applied. | Keep applied-LR history; compare scheduling at optimizer boundaries. |
| Checkpoint ranking | Carey selects MA5-based `best/`. | Intentionally omitted per user direction. Periodic and completed outputs remain. |

Raw RNG streams and initialization are different across C++/Torch; shared
seeds alone do not imply identical batches or adapters. Fixtures share actual
tensors/random draws, rather than assuming seeds provide cross-runtime parity.

## Remaining validation

- Compare actual audio decoding/resampling, metadata/tokenization, sampled VAE
  posterior, condition-encoder outputs and masking on shared source fixtures.
- Measure variable-duration/conditioning behavior through full decoders and
  longer trajectories; decide which downstream policies migrate explicitly.
- XL Q4 native training is separately validated; the full Torch replay here is
  the unquantized **base** model. Full XL Q4 vs dequantized-reference behavior,
  representative-dataset capacity, convergence and musical acceptance remain.
- Current-revision Metal LoRA/DoRA/resume/inference and BF16 frozen-DiT backward
  remain separate gates. No Metal result is claimed here.

## Validation provenance

Runtime source: `dcbe1785e756a49ce752f2ca83bb28851b799e6e`. Later changes to
this evidence/history documentation do not change runtime or test sources.
The full base-model nonzero-B LoRA/DoRA replays and the BF16-storage negative
check were repeated at that exact revision. Their reports are in:

- `build-release-validation/carey-dcbe-focused/report.json`
- `build-release-validation/carey-dcbe-base-bf16-lora/parity_report.json`
- `build-release-validation/carey-dcbe-base-bf16-dora/parity_report.fp32.json`
- `build-release-validation/carey-dcbe-base-bf16-dora/parity_report.json`
  (model-precision mode, intentionally fails equivalence).

All 20 CTests pass in both CPU-only and dynamic CUDA/Vulkan Windows Release
builds at `dcbe1785`. Final local trainer binary SHA256:
`9441946a2c9debe5eb531377ec54b6c6cb0df251106fc46c59ad0216d7a936c2`.

Balanced rank-8 XL Q4 LoRA/DoRA cold/warm/streaming/resume matrices remain
byte-identical to the previous uncached `2d140f7d` binary on CUDA0 and NVIDIA
Vulkan1. They verify finite metrics, actual accumulation sizes `[2,1,2,1]`,
mean loss against persisted continuation, resumed update numbering and
append-on-resume session boundaries. CUDA matrices ran during implementation
before the source commit (`history-cuda-lora`, `history-cuda-dora`); the final
Vulkan matrices ran at `dcbe1785` (`history-dcbe-vulkan-lora`,
`history-dcbe-vulkan-dora`). Moving the identical loss graph builder into
`train-loss.h` introduced no math change. Each directory's `summary.json`
records commands, hardware and results. CUDA used Gary's shared CUDA 12.8
runtime on PATH; Vulkan used only System32. Trainer logs confirm the actual
requested DiT backend.

At this update, CI Linux/macOS builds, lint, ggml checks and macOS universal
tests/signing/notarization pass. CUDA 12.6 compilation and portable Windows
packaging are still running; the live results are
[CI Build](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38620329939),
[ggml checks](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38620329933),
and [release dry-run](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38620329996).
These build checks do not validate Metal training or a clean Windows machine.

See `TRAINING_HISTORY.md` for the persistent metric contract.
