# Release readiness validation — 2026-10-12

The later SA3-aligned training controls and persisted mid-epoch continuation
are validated separately in [TRAINING_CONTROL_VALIDATION.md](TRAINING_CONTROL_VALIDATION.md).
Earlier archive hashes below describe their named revisions, not that follow-up.

At the user's request, [ACE PR #9](https://github.com/betweentwomidnights/acestep.cpp/pull/9)
was merged as `e93ee47f8b266d10458cafba8495a4d2d3752d4a`. The release branch
incorporates it and pins ggml `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`.
No new ggml patch is part of this work. Earlier `60f49e09` results remain
baseline evidence, not validation of the final pin.

The original release-package tests below use application
`27aeabf880a18f6078fb2ca020fe99510403988f`, with a clean tracked tree.
Both CI packages use GitHub's corresponding PR merge ref
`bdd2aed82ed0adac2f1d2dd5d4edaca4c4638b85`. Its tree matches the runtime
commit exactly (`9d7ca54a698b150efd2729391aaeba7ce2eafde2`), with the same ggml pin.
Those results describe that source revision. The caption/analysis implementation
at `be413acf9c99a05bb11c309639f6fdce55c338e1` changes runtime source and needs
fresh portable packaging CI; earlier archive hashes do not describe its binaries.

For that final runtime revision, Linux/macOS source builds and ggml checks pass.
The [universal macOS package job](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38225739764/job/114732678789)
also passes build/tests, signing and notarization. Portable Windows packaging
and the remaining source checks are still running in
[release CI](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38225739764)
and [CI Build](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38225739686).

## Caption, analysis and prerequisite follow-up

On the same PC and unchanged ggml `4ad3b30b`, the caption implementation passes
all 15 CTests in both CPU-only and dynamic CUDA/Vulkan builds. A local CPU package
also passes staged startup and includes `ace-audio-analyze.exe` and both imported
MIT licenses. The analysis CLI and HTTP responses agree on the same PCM fixture;
tests cover Unicode paths, silence, invalid/truncated/empty WAVs, non-finite
samples, duration bounds and model/sampling selection errors without model loads.
The tokenizer-backed understand prompt test passes with the installed LM GGUF.
Small-cache model-backed tests verify that both a prompt that cannot fit and a
decode that exhausts its KV budget fail cleanly, leaving the server healthy.
Float64 sample overflow is rejected before it reaches either estimator/encoder.
The final range guard checks the double downmix before narrowing to float;
all 15 CPU CTests pass after that portability adjustment.

The opt-in `tests/test-caption-real.py` passes on actual CUDA0 for base Q4 with
FSM enabled and disabled, XL-base Q4 with FSM enabled, and on NVIDIA Vulkan0 for
base Q4 with FSM enabled. Each run checks two identical same-seed captions in
the same server process, queued cancellation and normalized JSON. CUDA uses only
Gary's shared CUDA 12.8.0 directory plus Windows System32 on PATH; capture is
enabled and confirmed in logs. Typical complete requests take about 3–6 seconds
on this fixture. These are execution/structural checks, not a production caption
accuracy evaluation or backend numerical parity claim.

The 30-second PCM16 stereo, 44.1 kHz fixture is a local excerpt (offset 30s) from
Gary's existing `keygen_music_for_installer.wav`, SHA256
`2a652992877dce20ba0fdc056720de3302e1b94c7924b3f7de3a6a1ef7405d32`.
It is not committed. Model SHA256 values:

| Model | SHA256 |
| --- | --- |
| LM 0.6B Q8_0 | `bdaf9e292d4470f31c19cafeaca1b74936a114667e3a85e5d33b65247e9908ec` |
| Base Q4_K_M tokenizer | `942ec6a7bc11a40809af5f337063f2af2acc63e4fcd4a92b0cb42dc2bc7c41e8` |
| XL-base Q4_K_M tokenizer | `13b08ac84e0cdca8ac770d6bc12842046113c9479ddcedd2672ac87ac458ff78` |
| BF16 VAE | `0599862ac5d15cd308e1d2e368373aea6c02e25ebd1737ad4a4562a0901b0ef8` |

The old understand single-token path produced missing/repetitive text when CUDA
capture engaged. With capture disabled it produced coherent text. Reusing the
existing generation pipeline's persistent batched decode graph (batch size one)
restores coherent capture-enabled results, matching the capture-disabled result
for the same tokenizer/FSM setting. No shared ggml change was made. The prompt
also now closes the user turn immediately after codes and conditions constrained
metadata decoding on `<think>\n`, matching the Python template/FSM opener.
Incomplete token-budget responses and malformed caption text fail rather than
being offered as successful dataset suggestions.

Caption semantics vary by tokenizer and backend: the same excerpt produced
Eurodance, rock and J-core descriptions, with differing predicted keys. The API
retains low-confidence local key suggestions separately; it does not claim that
any of these labels are authoritative. Pin and evaluate the application caption
bundle before enabling bulk sidecar writes. Actual WAV duration overrides the
LM's estimated duration.

Local native-architecture GPU packaging also passes all 15 tests and staged-core
probes. Capture-enabled captions pass from the staged split core + CUDA package
with Gary's shared CUDA runtime, and from the standalone package with only
Windows System32 on PATH. Those local packages are not portable release assets.

CUDA Q4 DoRA-row forward/backward and the functional LoRA/DoRA inference suites
were rerun successfully after the change. The owner additionally reports an
earlier successful PC DoRA adapter trained on Q4 XL-base. That is separate from
the automated LoRA checkpoint evidence below and was not rerun as a full dataset
training job in this follow-up. Metal LoRA and BF16 frozen-DiT backward remain
open; see `METAL_TRAINING_HANDOFF.md`.

[Gary PR #8](https://github.com/betweentwomidnights/gary-localhost-installer/pull/8)
at `11138dd` adds the shared VC prerequisite preflight/repair. Five focused tests
and 18 native-runtime tests pass (one real-download test remains ignored).
The official pinned Microsoft installer passes SHA256, actual Authenticode and
version verification; a wrong expected version is rejected without executing it.
The current-system test confirms no download/cache/elevation when already ready.
No VC installation/uninstallation was performed here. Clean Windows installation,
UAC cancellation/retry and reboot acceptance remain required.

Follow-up logs/results are under `build-release-validation/caption-*-final/`
and `caption-376-*/`, `caption-bounds-ctest.log`,
`caption-cpu-package-final.log`, `caption-prompt-test.log`,
`caption-dora-q4.log`, `caption-adapter-functional.log`, `vc-cargo-tests-final.log`
and `vc-native-runtime-tests.log`.

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
| Windows prerequisite / clean-machine install | Superseded prerequisite finding: merged PR #11 packages the four VC/OpenMP DLLs app-local and checks their loaded paths. No central VC install or elevation is required. Startup on a clean machine without build tools remains an acceptance gate. |

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
