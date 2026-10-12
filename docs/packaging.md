# Native runtime releases

Version is declared once in `project(VERSION)` in CMakeLists.txt. `ace-server
--version` and `ace-train --version` print that bare version. The git revision
remains available as `build_revision` in props and in other CLI banners.

The release workflow follows sa3.cpp's packaging scripts: Windows 2022 with
CUDA 12.8.1/Vulkan SDK 1.4.350.0, plus universal macOS on macos-15 with arm64
Metal/CPU and x86_64 CPU. Each package records application/ggml hashes and
toolchains in `BUILD-INFO.json`. Linux source builds remain supported by the
existing build scripts and CI; the new release workflow packages Windows/macOS.

## Windows

Run `ci/package-windows.ps1 -Version v0.1.0`. It builds portable dynamic
backends, runs the self-contained CTests, checks the staged CPU startup probe,
and emits flat archives plus LF/no-BOM `SHA256SUMS`:

| Archive | Contents |
| --- | --- |
| `acestep-vX.Y.Z-windows-x64-core.zip` | Server, trainer, synthesis/LM/analysis/codec/quantizer tools, ggml core/CPU DLLs, notices and build metadata |
| `acestep-vX.Y.Z-windows-x64-cuda.zip` | Only `ggml-cuda.dll` |
| `acestep-vX.Y.Z-windows-x64-vulkan.zip` | Only `ggml-vulkan.dll` |
| `acestep-vX.Y.Z-windows-x64-standalone.zip` | Core + CUDA + Vulkan + CUDA runtime DLLs and NVIDIA EULA |

gary4local installs core plus its selected backend and shares `cudart-12.8`
with yuey and sa3. It must keep ACE's ggml DLLs in ACE's own directory:
ACE uses `GGML_MAX_NAME=128`; sharing CUDA does not mean sharing ggml DLLs.

Windows also needs the x64 Microsoft Visual C++ v14 Redistributable, at least
as recent as the compiler used for the package. See Microsoft's
[supported runtime downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
The archives use the shared MSVC/OpenMP runtimes (`MSVCP140`, `VCRUNTIME140`,
`VCRUNTIME140_1`, `VCOMP140`); they do not bundle those DLLs. The installer must
cover this prerequisite and test startup on a Windows machine without build tools.

`-CpuOnly -BuildDir build-release-cpu -OutDir dist-cpu` is a local contract
check, not a release GPU validation. `-CudaArch native` requires a separate
build directory and produces a local machine build, not a distributable one.
`-CudaRuntime` emits an optional runtime pack for local installer exercises;
gary4local owns publication of the shared CUDA runtime. Avoid `-SkipTests` for
release evidence. Model-free tests do not prove real-model GPU training.

## macOS

On Apple Silicon run `ci/package-macos.sh --version v0.1.0`. The script builds
and tests both slices (x86_64 execution needs Rosetta), merges tools and dylibs,
rewrites imports to the flat archive, and probes startup outside that directory.
`acestep-vX.Y.Z-macos-universal.zip` and `SHA256SUMS-macos` are the outputs.
The macOS deployment target is 13.3. Metal is arm64-only in this package.

Published releases require Developer ID signing and accepted notarization.
The workflow reads the same `MACOS_CERT_P12`, `MACOS_CERT_PASSWORD`,
`APPLE_TEAM_ID`, `APPLE_NOTARY_KEY_P8`, `APPLE_NOTARY_KEY_ID`, and
`APPLE_NOTARY_ISSUER_ID` secrets as the sibling repositories. Local scripts use
`ACE_SIGN_IDENTITY`, `ACE_NOTARY_KEY` (P8 file path), `ACE_NOTARY_KEY_ID`, and
`ACE_NOTARY_ISSUER`. Dry runs may use ad-hoc signing. A signed ZIP is notarized;
bare binaries cannot be stapled. Actual current-revision Metal LoRA training
still requires a separate real-model acceptance test.

## Supervisor contract

`ace-server --props` prints JSON to stdout, loads backend libraries and
initializes the selected device, then exits without binding a port or loading
model weights. `devices` reports backend names, device names/types and
free/total bytes. Metal is reported as `Metal` for the installer, with raw `registry_backend=MTL`
and devices such as `MTL0`;
`--device metal` and `ACE_DEVICE=metal` resolve that alias. Explicit unavailable
devices fail. Backend aliases prefer discrete GPUs over integrated GPUs;
exact device names preserve the caller's choice. Discover indices from props
rather than assuming `Vulkan0` always names the same GPU. Device aliases also
work in `GGML_BACKEND` for the CLI tools.

| Environment | CLI override |
| --- | --- |
| `ACE_MODELS_DIR` | `--models` |
| `ACE_ADAPTERS_DIR` | `--adapters` |
| `ACE_HOST` | `--host` |
| `ACE_PORT` | `--port` |
| `ACE_DEVICE` (then `GGML_BACKEND`) | `--device` |
| `ACE_ADAPTER_MODE` | `--adapter-mode` |

`GET /health` is the liveness endpoint. Serving requires a model registry;
the installation probe does not. CLI options override environment settings.
The server defaults to functional adapters so Q4 bases remain untouched;
`--adapter-mode merge` preserves the historical opt-in merge path. Synthesis
CLI behavior is unchanged; pass its existing functional adapter option explicitly.

`training.xl_base_low_memory_encoding=Q4_K_M` advertises the intended small-GPU
training path. This is not an automatic memory-fit guarantee. We deliberately
do not reuse yuey's model-specific VRAM thresholds as ACE training thresholds.
The application must select family and exact GGUF explicitly and account for
rank, windows, microbatch size, resident services and current free memory.

## First release gate

1. Package-script/CMake/workflow pull requests automatically build dry-run
   artifacts. Alternatively, dispatch `release` with an empty tag to build
   the selected branch as workflow artifacts. No release is published by a dry run.
2. Inspect both platforms' packages, provenance and checksum files. Exercise
   core + CUDA with the shared runtime, core + Vulkan, and standalone extraction
   outside the repository. Confirm the selected device really runs the workload.
3. Complete current-revision Metal LoRA/Q4/resume/inference acceptance and
   gary4local migration gates described in `GARY4LOCAL_READINESS.md` before
   claiming those integrations or backend paths as validated.
4. Create the matching version tag/release only after accepting the dry run.
   Published-release jobs sign/notarize macOS and attach build attestations.
5. Pin real URLs/hashes in gary4local with its pin script only after adding a
   Carey native candidate and API/job adaptation. The generic installer contract
   alone does not replace Carey's Python endpoints. Do not overwrite pinned assets.

Model downloads are separate: install one text encoder and VAE, the desired
DiT family/encoding, and an LM for LM/understand tasks. Native training accepts
paired WAV/metadata files and requires `--dit`, `--text-encoder`, or `--vae` when
that bucket has multiple installed variants. See `ace-train --help` for options.
