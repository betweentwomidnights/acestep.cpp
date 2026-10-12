# Release readiness validation — 2026-10-12

At the user's request, [ACE PR #9](https://github.com/betweentwomidnights/acestep.cpp/pull/9)
was merged as `e93ee47f8b266d10458cafba8495a4d2d3752d4a`. The release branch
incorporates it and pins ggml `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`.
No new ggml patch is part of this work. Earlier `60f49e09` results remain
baseline evidence, not validation of the final pin.

Local final-code tests below use application
`27aeabf880a18f6078fb2ca020fe99510403988f`, with a clean tracked tree.
Both CI packages use GitHub's corresponding PR merge ref
`bdd2aed82ed0adac2f1d2dd5d4edaca4c4638b85`. Its tree matches the runtime
commit exactly (`9d7ca54a698b150efd2729391aaeba7ce2eafde2`), with the same ggml pin.
Documentation-only follow-ups do not change the tested runtime implementation.

## Windows environment

- Windows x64, Visual Studio 2022 Community / MSVC 14.44.
- NVIDIA GeForce RTX 5070 Laptop GPU, approximately 8 GiB, driver 616.56;
  Intel Core Ultra 9 275HX with integrated Intel graphics.
- Local CUDA toolkit/shared runtime 12.8.0; CI installs 12.8.1 to match
  the published shared runtime convention.
- Vulkan SDK 1.4.350.0.
- Release flags: `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=ON`,
  `GGML_CPU_ALL_VARIANTS=ON`, `GGML_MAX_NAME=128`, `BUILD_TESTING=ON`.
  The local GPU packaging smoke uses `CMAKE_CUDA_ARCHITECTURES=native` in its
  own build directory. Those archives are local validation builds, not portable
  release artifacts. CI's default architecture list is the portable release gate.

## Evidence

| Check | Result |
| --- | --- |
| CPU-only and CUDA/Vulkan Windows packaging | Passed; all 13 registered tests passed in each build. Staged-core startup outside the checkout passed. |
| Archive checks | All local ZIP SHA256SUMS entries and flat layouts passed. Extracted core, core + CUDA, core + Vulkan, and standalone startup/version/build metadata passed outside the package directory. Four missing-backend cases failed explicitly without CPU fallback, as required. |
| Shared CUDA | Extracted core + CUDA executed XL Q4 training/resume/inference using `gary-localhost-installer/native-runtimes/cudart-12.8` (12.8.0) on PATH, with toolkit directories removed. |
| CUDA XL Q4 LoRA | Rank 64, balanced profile, alpha 128, batch 1, checkpointed backward, two-second windows over a four-second synthetic WAV: two optimizer steps passed on CUDA0. Resume from epoch 1 reproduced adapter, optimizer, config and trainer-state files bitexactly. |
| Vulkan XL Q4 LoRA | The same train/resume/functional inference smoke passed on the NVIDIA GPU using the extracted core + Vulkan package. Checkpoint files matched bitexactly within Vulkan. CUDA and Vulkan are not claimed to be bitexact to each other. |
| Functional adapter inference | Base and trained-adapter synthesis each returned valid four-second stereo 48 kHz audio, with different output hashes and an unchanged frozen Q4 GGUF. Passed on CUDA and Vulkan. |
| DoRA-row operator / functional adapter suites | F32, Q8_0, Q4_K, Q5_K and Q6_K forward/backward operator tests passed on CUDA and NVIDIA Vulkan. Functional LoRA/DoRA inference tests passed on both. The Intel Vulkan device additionally passed these small operator/inference suites; no Intel XL training claim is made. |
| GPU alias selection | Extracted Vulkan-only and standalone packages both chose NVIDIA for `--device vulkan`, although NVIDIA was Vulkan1 in the former and Vulkan0 in the latter. Exact device names remain exact. |
| Standalone HTTP | Environment-only startup, health/props, asynchronous `/synth` job, trained functional adapter and multipart WAV result passed with bundled CUDA DLLs and no toolkit/shared-runtime PATH. Loaded-module paths confirmed CUDA/ggml DLLs came from the standalone directory. |
| macOS universal package | [CI job](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38195497934/job/114637101424) passed 13/13 tests for each arm64 and Rosetta x86_64 slice, staged CPU probes, Developer ID signing and accepted Apple notarization. Downloaded archive checksum, flat layout and universal executable architectures passed inspection. |
| Portable Windows CI | [Build/test/package job](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38195497934/job/114637101318) passed all 13 tests and package probes using CUDA 12.8.1 and the nine-entry portable architecture list. Downloaded checksums, flat layouts and extracted core/CUDA/Vulkan/standalone device probes passed on this machine. |
| Downloaded portable binaries on the 5070 | The same XL-base Q4 rank-64 balanced LoRA train/resume/functional inference smoke passed on CUDA0 with gary4local's existing shared CUDA 12.8.0, and on NVIDIA Vulkan1 without CUDA on PATH. Both reproduced all four checkpoint files bitexactly within their backend. Standalone HTTP generation with the trained adapter also passed using only its bundled CUDA runtime. |
| Linux/macOS source builds, lint and CUDA 12.6 compatibility | All [CI Build](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38195497977) and [ggml check](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38195497897) jobs passed. CUDA compilation on a hosted runner is not GPU execution evidence. |
| Current-revision Metal LoRA | Not run: no Apple Silicon training hardware was available in this Windows session. The signed package/CPU tests do not close the README Metal LoRA gate. |
| Carey/PyTorch parity and application migration | Not complete; see `GARY4LOCAL_READINESS.md`. |
| Windows prerequisite / clean-machine install | PE imports confirm shared MSVC/OpenMP runtime dependencies. This machine has Visual Studio installed; VC Redistributable preflight and startup on a machine without build tools remain application acceptance gates. |

The macOS ZIP SHA256 is
`91a3f3923ca7c31e007da04fc5605b095d957d1f6088403e14e9eff043cf3070`;
Apple accepted notarization `35f709ae-03e3-4c6a-b124-918b05c17593`.
This dry-run archive is a workflow artifact, not a published release asset.

Downloaded portable Windows artifact SHA256 values:

| Archive | SHA256 |
| --- | --- |
| core | `d0855a1b8c419a62f892daba80c85c6b3a36c0070d4dbd388f68eba138e13c32` |
| CUDA | `f7e67f1854218e31d81a7c9b233942001447ca776ffa6a71c580ee14ffdf3d1f` |
| Vulkan | `8e0652a5992c413617cf51d237fcd9474a50ed8f4fd92ef0cbdeae044f50a8ef` |
| standalone | `b6f69b19bebbc16cf4969c297c7c793dd8e9e509e03f8484272d5f75844b652f` |

Local commands:

```powershell
./ci/package-windows.ps1 -Version v0.1.0 -CpuOnly -BuildDir build-release-cpu -OutDir dist-cpu
./ci/package-windows.ps1 -Version v0.1.0 -CudaArch native -BuildDir build-release-gpu-local -OutDir dist-gpu-local
python ci/validate-xl-q4.py --bin-dir <extracted-core-and-backend> --models models --backend <exact-device> --output <new-results-directory>
```

Local logs, summaries and extracted archives are under `build-release-validation/`,
`build-release-*`, `dist-cpu/`, and `dist-gpu-local/`, intentionally excluded
from source control. The final smoke summaries are
`cuda-shared-xl-final/summary.json`, `vulkan-xl-final/summary.json`,
`extracted-27aeabf8/summary.json`, and `standalone-http-summary.json` under
`build-release-validation/`. Portable CI artifact results are
`portable-cuda-xl/summary.json`, `portable-vulkan-xl/summary.json`,
`extracted-portable-ci/summary.json` and `portable-standalone-http-summary.json`.
For those training runs the checkout temporarily selected the exact CI merge
commit, then restored the release branch. Device numbering must be discovered from props
in the actual package/environment, not copied from another build's log.

`ci/validate-xl-q4.py` resolves the immutable checkpoint pointer and records
commands, times, selected devices and checkpoint hashes. Its synthetic fixture
is a two-step execution/resume test, not convergence, Torch parity or
full-duration memory acceptance. Production rank/duration/dataset coverage still
needs application testing with competing services stopped. BF16 VAE inference
does not validate BF16 frozen-DiT backward; the pinned CPU/Metal OUT_PROD guards
still omit BF16 support.

The published download/install/update path and gary4local acceptance remain open.
No release/tag or gary4local manifest pin was created by this preparation work.
