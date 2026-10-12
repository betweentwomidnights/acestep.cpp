# Release readiness validation — 2026-10-11

The release preparation branch starts at application `76981988` and pins
ggml `60f49e09ce67df544831a357b654318ec2722fb0`, the already-merged fork default.
No new ggml candidate or consumer pin change is part of this work.

## Windows environment

- Windows x64, Visual Studio 2022 Community / MSVC 14.44.
- NVIDIA GeForce RTX 5070 Laptop GPU, 8151 MiB, driver 616.56.
- Local CUDA toolkit 12.8.0; CI installs 12.8.1 to match the shared runtime.
- Vulkan SDK 1.4.350.0. Use the NVIDIA Vulkan device explicitly for fleet evidence.
- Release flags: `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`,
  `GGML_CPU_ALL_VARIANTS=ON`, `GGML_MAX_NAME=128`, `BUILD_TESTING=ON`.
  The local GPU packaging smoke uses `CMAKE_CUDA_ARCHITECTURES=native` in its
  own build directory. That archive is a local validation build, not a portable
  release artifact. CI's default architecture list is the portable release gate.

## Evidence

| Check | Result |
| --- | --- |
| `ci/package-windows.ps1 -Version v0.1.0 -CpuOnly -BuildDir build-release-cpu -OutDir dist-cpu` | Passed on `60f49e09`; all 13 registered tests passed, including runtime process contract and CPU training/adapter tests. Staged core startup outside the checkout passed. |
| `bash -n ci/package-macos.sh` and workflow YAML parse | Passed; syntax checks are not macOS execution/signing evidence. |
| CUDA/Vulkan packaging and GPU tests | In progress; results will be recorded after completion. |
| Current-revision Metal LoRA | Not run on this Windows machine; remains an explicit README gate. |
| Carey/PyTorch parity and application migration | Not complete; see `GARY4LOCAL_READINESS.md`. |

Local logs and generated artifacts are under `build-release-*`,
`dist-cpu/`, and `dist-gpu-local/`. They are intentionally excluded from source
control. `ci/validate-xl-q4.py` makes the real-model smoke reproducible, resolves
the immutable checkpoint pointer, compares adapter and optimizer hashes after
resume, checks output audio and records the exact selected GPU. Its synthetic
fixture is a two-step execution test, not convergence or full-duration memory
acceptance. Production rank/duration/dataset coverage still needs application
testing with other services stopped.

The first signed/notarized portable CI dry run and published installation path
must be accepted before a stable release is claimed. No release/tag or
gary4local manifest pin is created by the local packaging scripts.
