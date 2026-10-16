# Preprocessing cache validation — 2026-10-14

Implementation `bc47ed55ea18d946212775b832d93a115eb7fa6b`; allocated-capacity
accounting fix `f3a8df0f460eccdff261bb815d48ccd1de76c0ca`.
Shared ggml stays `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`.
The uncached reference is the previous local split Windows package recording
runtime source `2d140f7d46ebd037ea162e68a8a1a6673b1d39b4`.

Windows, RTX 5070 Laptop (8151 MiB), NVIDIA driver 616.56. Release MSVC builds
use dynamic backends, all portable CPU variants, `GGML_NATIVE=OFF`,
`GGML_MAX_NAME=128` and local native CUDA architectures. These local GPU builds
are not the portable CI architecture matrix. CUDA children use only Gary's
shared CUDA 12.8.0 runtime and System32 on PATH; Vulkan children use only
System32, with props and trainer logs confirming NVIDIA `Vulkan1` execution.
GPU runs are sequential. No M4/Metal or CPU XL model-training result is claimed.

| Model | SHA256 |
| --- | --- |
| XL-base Q4_K_M | `13b08ac84e0cdca8ac770d6bc12842046113c9479ddcedd2672ac87ac458ff78` |
| Qwen3 embedding Q8_0 | `972f23255e46adfe744a0eb9a0039f3c63988f65753b0968d776e8b27168c321` |
| BF16 VAE | `0599862ac5d15cd308e1d2e368373aea6c02e25ebd1737ad4a4562a0901b0ef8` |

## Results

- All 19 model-free tests pass in CPU-only and dynamic CUDA/Vulkan Windows
  builds. Coverage includes lossless F32 storage, restored global padding,
  allocated-capacity LRU bounds, zero residency/oversized bypass, invalid
  identities, malformed/truncated/corrupt/nonfinite entries, atomic publication,
  Windows replacement failure preserving old data and staging cleanup.
- CLI identity tests cover model contents, seed, VAE tiling and selective
  metadata changes, while excluding optimizer/adapter/cache/output controls.
  Changing cache controls remains compatible with full-state continuation.
- CPU packaging passes all 19 tests, staged startup and the actual loaded
  app-local VC/OpenMP DLL-origin guard.
- At `bc47ed55`, CUDA0 and NVIDIA Vulkan1 LoRA pass the full real-model cache
  matrix against the uncached reference. Cold, warm, chunk-size-two and resumed
  runs match all four checkpoint payloads byte for byte: adapter weights/config,
  AdamW state and trainer continuation. Cached models skip VAE/text/condition
  encoder loads on warm starts.
- Both LoRA matrices also pass corrupt-entry regeneration, single-entry
  metadata invalidation, optimizer-only cache reuse, and a changed longest
  caption compared against freshly uncached preprocessing. Cancellation after
  the first complete cache entry yields no adapter; the next run reuses that
  entry, computes the missing two, and matches the uncached reference.
- At `bc47ed55`, DoRA-row on CUDA0 and NVIDIA Vulkan1 passes the shorter
  baseline/cold/warm/mid-epoch-resume matrix, with byte-exact payloads throughout.
- At final runtime `f3a8df0f`, all four CUDA/Vulkan LoRA/DoRA-row
  baseline/cold/warm/mid-epoch-resume matrices pass again against the old
  uncached reference. Zero-residency streaming and the one-MiB warm LRU match
  all four checkpoint payloads byte for byte; warm runs load no preprocessing
  encoders. Props and actual DiT logs confirm the requested NVIDIA device.
- The final cache files are byte-identical between LoRA and DoRA for relocated
  copies of the same dataset, and CUDA/Vulkan cache keys are disjoint. The
  final legacy all-RAM path also matches streamed training byte for byte. An
  unusable cache root produces a terminal `failed` progress file/nonzero exit
  with no adapter, rather than silently switching to unbounded memory.
- Final local core/CUDA/Vulkan/standalone packaging passes all 19 tests and
  staged startup/actual app-local DLL-origin checks. The backend ZIP hashes
  remain unchanged; no shared ggml source or pin changed.

The capacity fix changes only LRU memory accounting from logical vector size to
allocated capacity; it does not change preprocessing, padding, random streams
or optimizer math. Its regression pads a nearly full vector to force geometric
allocation growth, then verifies that the actual allocation is counted or
bypasses residency. The final four GPU matrices above exercise the updated
accounting; the broader invalidation/cancellation matrix at the preceding
implementation revision remains separately identified.

## Fixture and evidence

The opt-in script makes three four-second stereo WAVs with different caption
lengths. It uses balanced rank 8/alpha 16, two epochs, batch 1, accumulation 2,
two-second windows, seed 42 and four total optimizer updates. The changed-caption
case lengthens the longest caption, so previously cached shorter examples must
receive new dataset-wide null padding. These checks establish deterministic
execution/continuation, not convergence, Carey numerical parity or realistic
full-song capacity. BF16 VAE inference is not BF16 frozen-DiT backward coverage.

```powershell
python tests/test-training-cache-real.py --bin-dir build-release-gpu-local/Release --baseline-bin-dir build-release-validation/cache-baseline2d --models models --backend CUDA0 --adapter-type lora --output <fresh-results>
```

Use `Vulkan1` only after probing current hardware enumeration. Repeat with
`--adapter-type dora-rows --quick` for the shorter matrix; `--quick` can also
validate the final revision without repeating all invalidation/cancellation cases.

Ignored local evidence under `build-release-validation/`:

- `cache-bc47-{cuda,vulkan}-{lora,dora}/`: per-job commands/times, logs,
  props, checkpoints and `summary.json` for the four matrices above.
- `cache-bc47-ctest.log`, `cache-bc47-cpu-package.log`,
  `cache-f3-cpu-package.log`: model-free/packaging evidence.
- `cache-baseline2d/BUILD-INFO.json`: the uncached baseline package metadata.
- `cache-f3-{cuda,vulkan}-{lora,dora-rows}/`: final runtime matrices.
- `cache-final-checks.py/.json`, `cache-f3-legacy-memory.log`,
  `cache-f3-blocked.log`: cross-adapter/backend, legacy-mode and I/O checks.
- `cache-f3-gpu-package.log`: final local split/standalone packaging.

Source/packaging CI for the final runtime is in
[CI Build](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38527199008),
[release dry run](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38527198941)
and [ggml check](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38527198919).
The earlier `bc47ed55` release run was superseded by the capacity fix; its Windows
job was cancelled rather than treated as final release evidence. No release/tag,
Gary change or shared ggml bump is published by this cache work.
At this update, Linux/macOS source builds, lint, both ggml checks and signed/
notarized universal macOS packaging pass. Portable Windows CI and CUDA 12.6
compilation are still running. Later documentation/test-probe updates leave the
runtime source above unchanged; earlier package hashes do not describe it.
