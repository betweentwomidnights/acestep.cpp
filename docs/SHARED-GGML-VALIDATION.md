# Shared GGML candidate: M4 validation

The draft PRs form one testable set:

- [GGML #5](https://github.com/betweentwomidnights/ggml/pull/5)
- [ACE #1](https://github.com/betweentwomidnights/acestep.cpp/pull/1)
- [SA3 #34](https://github.com/betweentwomidnights/sa3.cpp/pull/34)

Both applications pin **`19c5421c9314893db83ab93575d34b9cb828516f`** from
`betweentwomidnights/ggml`. No local patches, bundles, Python runtime, or submodule
branch overrides are needed to build the native engines. The optional conversion
and PyTorch parity tools still have their documented Python dependencies.

## Checkout

Use a clean dedicated clone for each application; do not replace an active M4
development checkout. With GitHub CLI installed, inside the respective clone:

```sh
# Inside acestep.cpp
gh pr checkout 1 --repo betweentwomidnights/acestep.cpp
git submodule sync --recursive
git submodule update --init --recursive
git -C ggml rev-parse HEAD

# Inside the separate sa3.cpp clone
gh pr checkout 34 --repo betweentwomidnights/sa3.cpp
git submodule sync --recursive
git submodule update --init --recursive
git -C ggml rev-parse HEAD
```

Both hashes must match the candidate above. Do not use `submodule update --remote`:
that follows a branch instead of the reviewable application pin. If GGML fixes are
needed on M4, push them to GGML #5 and update both application PR pins together.

## ACE Metal

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DBUILD_TESTING=ON
cmake --build build-metal --parallel 4
ctest --test-dir build-metal --output-on-failure
./build-metal/test-adapter-functional MTL0
for type in F32 Q8_0 Q4_K Q5_K Q6_K; do
  ./build-metal/test-dora-backend MTL0 "$type" || exit 1
done
```

The backend argument is the GGML *device* name, which is `MTL0` under GGML 0.17, not
`Metal`; passing the wrong name fails backend init rather than testing anything. Confirm the
registered device is the Metal one; do not count CPU fallback as a Metal pass.
The explicit graph tests above complement the CPU-oriented CTests.

Then validate real models and adapters using local M4 model/data paths:

- XL-base Q4 synthesis without an adapter and with functional DoRA at scale 1.
  Use 50 steps, CFG 7, shift 1 for base models, not turbo defaults. No LM is required.
- Use `ace-synth --adapter-mode functional` and confirm that all expected adapter
  projections load and the base weights remain unchanged. Keep normalized-strength
  mode experimental; it is not the production default or an established quality gain.
- Start `ace-server --adapter-mode functional` on loopback; verify `/props` reports
  `cli.adapter_mode` as `functional` and complete a model-backed generation request.
- Run a short native checkpointed DoRA training job and resume its full training
  state in a separate output directory. Check finite loss/gradients, changed adapter
  tensors, optimizer/step continuity, and inference reload of the resulting adapter.

The one-step native/PyTorch parity harness is available separately:
`tests/test-train-step-parity.py --help`. It requires explicit native executable,
GGUF, full-precision checkpoint, Side-Step checkout, and output paths. It is not a
dependency of native training or inference and is not automatically run by CTest.

## SA3 Metal

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DSA3_METAL=ON -DSA3_CUDA=OFF -DSA3_VULKAN=OFF -DBUILD_TESTING=ON
cmake --build build-metal --parallel 4
ctest --test-dir build-metal --output-on-failure
```

Test small-music and medium, F16/Q4, fixed-seed inference, functional adapters and a
blend, native preprocessing/training, checkpoint resume, C-ABI training/callbacks,
and trained-adapter reload. Confirm the wide-row regression actually executes on
Metal. Compare with a separate old-pin checkout using the same local data and seeds;
cross-backend waveform identity is not expected or required.

## Existing Windows evidence

SA3 main `378c269` on CUDA and NVIDIA Vulkan: complete builds, all 18 CTests per
version/backend, and 52 model-backed runs per backend. Within each backend, all 15
matched waveform pairs and all compared adapter/optimizer tensors matched the old
0.16 pin exactly. Both F16 and Q4 training/resume and the C ABI passed. No meaningful
speed gain was measured.

ACE Vulkan: full build, nine CTests, 60 functional-adapter cases, DoRA forward/backward
for all five weight types above, and XL-base Q4 synthesis with/without the known-good
DoRA adapter (20 seconds, 50 steps, CFG 7, scale 1). Existing CUDA inference/training
checks also passed. This is bounded correctness evidence, not a new long-run quality
claim. AMD/Intel Vulkan and M4 Metal remain separate hardware gates.

The published GGML computational source exactly matches the Windows-validated
`d1bead82` plus the latest Metal OUT_PROD capability guard. History reconciliation
and patch documentation do not change its kernels. The Metal guard still requires
compilation/runtime validation on M4.

Report Metal results on the corresponding application PR with both application and
GGML hashes, build flags, device, tests, and any failures. Keep all three PRs unmerged
until the required Metal checks pass. Twilwa's original implementation commits are
preserved in ACE's existing PR; use a history-preserving merge when it is ready.
