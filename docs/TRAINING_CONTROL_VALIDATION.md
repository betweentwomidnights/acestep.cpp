# Training control validation — 2026-10-13

Implementation: `2d140f7d46ebd037ea162e68a8a1a6673b1d39b4`, atop merged
`c4e96d6db58d4ae0e6b738f6ab263c3cfdd192ec`.
Shared ggml remains `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`.
SA3's control reference is `e2f6d1039fabb50ad27729d3cc2013ac9c8002b1`.
The local builds were made from the implementation's runtime/test source;
documentation and the recorded evidence are a separate follow-up.

Windows 11, RTX 5070 Laptop GPU (8151 MiB), NVIDIA driver 616.56. Release
MSVC builds use `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`,
`GGML_NATIVE=OFF` and `GGML_MAX_NAME=128`. The GPU build enables CUDA and
Vulkan with local `CMAKE_CUDA_ARCHITECTURES=native`; this is not the portable
CI architecture build. GPU tests run sequentially with exact backend names.
CUDA uses Gary's shared 12.8.0 runtime with the child PATH limited to
`C:/dev/gary-localhost-installer/native-runtimes/cudart-12.8;C:/Windows/System32`.
Without CUDA on PATH, props reports the NVIDIA GPU as `Vulkan1` and Intel
graphics as `Vulkan0`; Vulkan validation uses `Vulkan1` with the child PATH
limited to `C:/Windows/System32`.

| Model | SHA256 |
| --- | --- |
| `acestep-v15-xl-base-Q4_K_M.gguf` | `13b08ac84e0cdca8ac770d6bc12842046113c9479ddcedd2672ac87ac458ff78` |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | `972f23255e46adfe744a0eb9a0039f3c63988f65753b0968d776e8b27168c321` |
| `vae-BF16.gguf` | `0599862ac5d15cd308e1d2e368373aea6c02e25ebd1737ad4a4562a0901b0ef8` |

## Results

| Gate | Result |
| --- | --- |
| CPU-only model-free CTests | All 18 pass. |
| Dynamic CUDA/Vulkan model-free CTests | All 18 pass. |
| CPU packaging / app-local VC guard | Pass; staged server loads all four VC/OpenMP DLLs from its own directory. |
| Local CUDA/Vulkan split + standalone packaging | Pass; all 18 CTests, staged startup and actual loaded VC/OpenMP DLL paths pass. Native CUDA architectures were used for this local build. |
| CUDA0 rank-64 balanced LoRA | Pass: uninterrupted vs mid-epoch resume, accumulation cancellation/resume and explicit legacy migration match all four checkpoint payloads byte for byte. Step checkpoint matches the one-step stop. Changed LR/data/epoch budget and unapproved legacy resume fail before preprocessing. |
| CUDA0 rank-64 balanced DoRA-row | Pass: the same complete continuation/control suite, also rejecting reuse of an existing output without resuming its current checkpoint. |
| NVIDIA Vulkan1 rank-64 balanced LoRA | Pass: all the same continuation/cancellation/legacy checks are byte exact within Vulkan; changed-input and output-reuse guards pass. |
| NVIDIA Vulkan1 rank-64 balanced DoRA-row | Pass: the same complete suite, with byte-exact mid-epoch and cancelled continuation and legacy epoch migration. |

For the exact runtime commit, Linux/macOS source builds, both ggml checks and
lint pass in [CI Build](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38482225376)
and [ggml check](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38482225469).
The [universal macOS package](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38482225299/job/115587268320)
also passes tests, signing and notarization. At this evidence update, portable
Windows packaging and CUDA 12.6 compilation are still running; their CI links
retain the final result. The follow-up only changes documentation and the
opt-in test's device probe, leaving that runtime source unchanged.

`tests/test-training-control-real.py` creates three synthetic four-second stereo
WAVs with canonical metadata. Each job uses two epochs, batch one, accumulation
two (including the epoch's partial tail accumulation), two-second rotating
windows, rank 64/alpha 128, seed 42 and warmup zero. The original schedule is four
optimizer updates. Cancellation is requested after the first window is logged,
so it exercises finishing the second accumulated microbatch before saving step
one with `next_batch=2` and the shuffled order. Resume restores the remaining
batch of that epoch, then the next epoch. The comparison covers PEFT weights,
config, optimizer moments and trainer continuation, with no filename normalization
inside the original checkpoint layout.

Local evidence is ignored under `build-release-validation/`:

- `training-control-cpu-ctest.log`, `training-control-gpu-final-ctest.log` and
  `training-control-cpu-package.log`, `training-control-gpu-package.log`.
- `training-control-cuda-lora-final/`: full, partial, resumed, cancelled,
  cancel-resumed and legacy-migrated logs/progress/checkpoints plus summary JSON.
- `training-control-cuda-dora-final/`: the same suite for DoRA-row.
- `training-control-vulkan1-lora-final/`: NVIDIA LoRA suite with the actual
  device description recorded in `props.json` and `summary.json`.
- `training-control-vulkan1-dora-final/`: the same suite for NVIDIA DoRA-row.

Reproduce using the command in `TRAINING_JOB_CONTROL.md`, selecting a fresh
results directory. These runs establish deterministic continuation within a
backend/build. They do not establish cross-backend equality, convergence,
Torch parity, full-song memory capacity or downstream application integration.
No Apple Silicon hardware was available; current Metal training and BF16
frozen-DiT backward remain unvalidated/unimplemented as documented in
`METAL_TRAINING_HANDOFF.md`. CPU XL training was not run.

## Existing rank-4 CUDA limitation

The same fixture with balanced rank 4/alpha 8, XL-base Q4, batch one,
accumulation two and two-second windows fails in CUDA OUT_PROD with
`cublasSgemm_v2` reporting an unsupported value/parameter. This also reproduces
with the old portable application `27aeabf880a18f6078fb2ca020fe99510403988f` at
the same ggml pin, before the new controls/plan persistence. Logs are
`training-control-cuda-lora/full.log` and
`training-control-rank4-baseline.log` under the local evidence root.

The root cause and other affected ranks/shapes/backends are not established.
Rank-64 passing results must not be generalized to all ranks. No ggml patch or
pin change was made here; a shared-fork fix requires the coordinated consumer
validation workflow. This native abort also demonstrates why a host must treat
a nonzero child exit and stale `running` progress as failure.

An initial Vulkan run used `Vulkan0` after removing CUDA from PATH, which
changed enumeration order and selected Intel graphics. It failed at the first
training loss with a nonfinite value (`training-control-vulkan-lora-final/full.log`).
No Intel XL training success or root cause is claimed. The test now probes and
records physical device descriptions, and supports `--expected-device NVIDIA`
to reject that mismatch before training.
