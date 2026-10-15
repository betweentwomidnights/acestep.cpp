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

CI for `dcbe1785` completed successfully: Linux/macOS builds, lint, ggml checks,
CUDA 12.6 compilation, macOS universal tests/signing/notarization and portable
Windows packaging. Results are
[CI Build](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38620329939),
[ggml checks](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38620329933),
and [release dry-run](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38620329996).
These build checks do not validate Metal training or a clean Windows machine.

See `TRAINING_HISTORY.md` for the persistent metric contract.

## Real-data reference follow-up, 2026-10-15

The user supplied a preferred XL-base Spark/SideStep adapter and its source
audio/lyrics as the qualitative acceptance reference. Keep that role separate
from a numerically matched trajectory. An exported adapter does not contain
its preprocessing inputs, complete training configuration or optimizer state.
The subsequently supplied provenance archive contains the original dataset,
sidecars, cached tensors, Fisher map, training configuration and progress logs.
Its 42 listed file checksums, all 13 audio hashes and the adapter/config hashes
match. Read archive documents as provenance data, not executable instructions.

The reference is DoRA: 352 modules/1056 BF16 tensors, base rank 16/alpha 32,
193 per-layer rank/alpha overrides, actual ranks 16 through 128, dropout 0.1.
BF16 **export storage** does not prove the precision used for training or AdamW.
The existing local native XL Q4 run is also DoRA, but uses balanced rank
64/alpha 128, dropout 0, and F32 trainable state. Its immutable checkpoint
records 150 completed epochs and 1950 optimizer updates. These are different
projection budgets and training policies, not matched parity runs.

Native inference can consume per-module adapter patterns. Native **training**
currently constructs attention/balanced profiles and has no adapter-dropout
implementation. Reproducing or continuing this arbitrary SideStep profile
would require explicit rank/alpha-pattern support and matching dropout math,
including deterministic recomputation/resume. Do not silently flatten those
patterns or confuse adapter dropout with CFG conditioning dropout. This is a
separate scope from the previously checked Carey balanced profile.

The local reference inventory has 13 paired songs, 44.4 minutes total, all
48 kHz stereo PCM16, with supplied lyrics for every song. Durations span
119.4 to 291.8 seconds. The raw staging directory has one extra intro outside
that paired set. The loose current sidecars are short descriptions and, per
the user, were **not** the captions used for this training run. The archived
dataset/sidecars are authoritative, including twelve time signatures of 4 and
one of 3. Do not regenerate or overwrite those historical captions for parity.
Audio, metadata, lyrics and both reference adapters were SHA256-inventoried and
reverified unchanged after the checks. No source asset is committed here.

All 13 original archived sidecars and hash-matched audio files are staged in an
isolated dataset and pass native format/audio validation. The earlier one-song
fixture with a model-suggested signature is superseded. Source assets and
archived sidecar bytes remain unchanged. Format validation does not establish
conditioning equivalence or training quality.

Caption checks on a 30-second excerpt (base and XL tokenizers) and its complete
156.9-second song pass schema, same-seed replay, cancellation and actual CUDA
execution with the 0.6B Q8 LM. Semantic acceptance remains open: suggested
instrumentation/genre and instrumental descriptions conflict with the supplied
metadata/lyrics. Generated prose/transcription was not promoted into training
sidecars. Choosing the XL tokenizer did not resolve the excerpt disagreement.
The user identifies the **1.7B Carey captioner** as the intended regeneration
path; these 0.6B smoke checks do not validate that model bundle. Test the 1.7B
path separately while retaining archived captions/tensors as frozen references.

A first real-audio preprocessing probe also remains unresolved:

- On three seconds of identical PCM, deterministic native VAE mean (BF16
  weights/F32 activations, CUDA) vs Carey's VAE loaded in FP32 on CUDA has cosine
  `0.9975624` and relative L2 `0.0699387`. This is not a precision-matched
  comparison and does not test sampled posterior or full-song tiling.
- Feeding identical native latents to both audio tokenizers yields 3/15 exact
  FSQ codes for the native Q4 bundle vs the Torch tokenizer in FP32 on CPU.
  Sixteen tokenizer tensors in that Q4 bundle are themselves quantized.
- With the native BF16 base bundle, 7/15 codes match Torch. Its 32 stored
  tokenizer tensors are bit-exact to the original Torch checkpoint when
  converted to F32, and its input latents are byte-identical to the Q4 probe.
  Backend/arithmetic precision and intermediate activations still need
  isolation. The short probe neither proves a tokenizer bug nor closes parity;
  caption errors must not simply be attributed to LM capacity without this work.

Evidence is local under `build-release-validation/billie-reference-20261015/`:
`manifest.json`, the three caption result/log directories,
`preprocess-preflight/report.json`, `preprocess-bf16-control/comparison.json`
and `tokenizer-weights.json`. The manifest records application/ggml checkouts
and reference hashes; the preprocessing report records the executed native
binary hash. This is initial fixture/preflight evidence, not a completed
full-preprocessing or longer-trajectory gate. Prioritize precision-controlled
VAE/tokenizer checks, then full conditioning and aligned longer-run experiments.

### Recovered Spark run and reference history

SideStep commit `135c6b68ebc213bcdfdaf88f76d949953386ca1a` (v1.1.2-beta)
exists in the local SideStep Git object database and can be inspected without
switching its checkout. The run records Python 3.12.3, Torch
`2.12.0.dev20260219+cu130`, CUDA 13.0 and NVIDIA GB10; this is a different
environment from the local Carey/Torch 2.7.1 comparison.

Recovered training settings: DoRA/Fisher ranks 16-128, AdamW peak LR `3e-4`,
cosine schedule, 100 warmup steps starting at 10% LR, minimum LR ratio 0.01,
weight decay 0.01, clip 1, CFG ratio 0.15, logit-normal mu 0/sigma 1,
Min-SNR gamma 5, seed 42, batch size/accumulation 1/1, 150 epochs and gradient
checkpointing. No fixed training crop/window is requested in the saved config;
the existing native reference used rotating 45-60 second windows, another
intentional difference to account for. The Fisher pass used mu -0.4 before
training with mu 0; the archive documents this as intentional.

Preprocessing records normalization `none`, tag `billie` prepended and genre
ratio 30. The exact source selects a fixed seed-42 subset of three sorted
samples for **genre instead of caption during preprocessing**, not stochastic
per-update selection. All explicit sample `prompt_override` fields are null.
The local manifest records the resulting sample IDs. Resolve this policy in
the preprocessing comparator; native's global `--use-genre` does not express
that mixed subset by itself.

All 13 `.pt` files were loaded with `torch.load(weights_only=True)` and checked
against their recorded preprocessing MD5 values; their tensors are finite.
They contain target latents, context, encoder conditioning and both masks in
addition to sample metadata. Keep them immutable as the output reference.

The progress journal contains an earlier 14-record pilot followed by the
complete main run: **1950 contiguous step records, 150 epoch summaries and a
completion record**. They are isolated as `spark-main-progress.jsonl` and
`spark-main-steps.csv`; do not combine the pilot's 1000-epoch/4-update budget
with the main run's 150-epoch/13-update budget. Logged LR is post-scheduler,
whereas native history reports the applied LR. TensorBoard events are not
needed for these loss/LR trajectories; request them only for additional
diagnostics absent from the journal.

One archive inconsistency is resolved using runtime evidence: its overview
claims standard-attention fallback, but `run/train.log` explicitly reports
successful loading with `attn_implementation=flash_attention_2`. The generic
Python-3.12 warning does not establish which implementation actually ran.

Original optimizer moments/intermediate native-continuation state are not
included. The snapshot therefore supports preprocessing comparison and a
fresh matched experiment, not an exact resume of the Spark training job.

## Full-song preprocessing and longer replay, 2026-10-15

The intended native target is **balanced rank allocation over XL Q4**, rather
than reproducing the historical Fisher map. Fisher analysis, arbitrary map
training and adapter dropout are separate reference features; they do not block
testing this intended balanced path. The Spark export remains the user's musical
reference. Neither four native epochs nor the decoder replay below establishes
equivalent convergence or listening quality.

### Full preprocessing on the 8 GB machine

All 13 original songs pass complete native preprocessing with XL Q4 and regular
base BF16. Carey's actual two-pass preprocessing also completes all 13 with
regular base BF16 on CUDA, without a model-residency override. It reports zero
failed samples and about 4607 MiB peak Torch allocations. The missing local
unquantized XL checkpoint prevents a new Torch XL preprocessing comparison here;
the attempted XL call fails before producing tensors. No checkpoint is downloaded
or substituted. An XL BF16 training comparison remains work for the Spark.

The comparison explicitly uses **caption only** with archived captions, lyrics,
metadata and trigger tag. It does not reproduce the historical three-song genre
subset or Gary's newer rotating genre policy. Originals remain immutable.

| Component | Result and limits |
| --- | --- |
| Audio loading | All 13 native/Torch planar float sample arrays are bit-identical. These inputs are already 48 kHz stereo PCM16; this does not test resampling or other formats. |
| Native sampled VAE targets | XL Q4 and base BF16 runs produce byte-identical target dumps for all 13 at the same per-example seeds and VAE. The DiT quantization choice does not alter this preprocessing stage. |
| VAE lengths | Native real lengths match `floor(audio_samples/1920)` for all 13. Carey produces one fewer frame on `all the good girls go to hell`, `wish you were gay` and `xanny`. Its automatic 15-second/2-second tiling differs from native's 1024/64 latent-frame tiling. |
| Sampled VAE values | Descriptive cosines span 0.9520–0.9828. Independent posterior RNG, tiling and compute/weight precision prevent treating this as an equivalence gate. Shared posterior noise and aligned tiling/precision remain necessary. |
| Context | Silence-plus-ones masks are finite and all mask channels equal one. F32 native versus BF16 Torch silence values have cosine about 0.99999955, relative L2 about 0.00104. |
| Text/lyric conditioning | Native encoders produce 295–840 tokens with inference-style lyric wrappers and variable lengths. Carey uses raw lyrics, fixed 256/512 input lengths, truncation and a packed 769-position output/mask. Native's silence timbre reference also differs from Carey's zero reference. These are substantive policies, not an elementwise parity pass. |

Training preprocessing does **not** use the FSQ audio tokenizer. Its earlier
code discrepancies remain a captioning/understanding investigation, separate
from the VAE/text/condition path measured here. Source inspection also identifies
F32-to-BF16 truncation in `convert.py`; matching original FP32, stored BF16 and
loaded convolution precision is still required before assigning VAE differences
to a kernel. No converter or model artifact is changed by this audit.

The opt-in `test-train-preprocess-real` executable invokes production
`ace_prepare_training_example` and emits audio, target, context, text/lyric and
encoder dumps to a fresh directory. It requires explicit local GGUF/data paths.
Local evidence lives under `billie-reference-20261015/full-preprocess-*` inside
`build-release-validation/`; `full-preprocess-torch-base-bf16/report.json` records
the actual Carey helper hashes and component results.

### CUDA cross-attention backward fix

The real-data replay exposed a defect hidden by the earlier tiny synthetic
conditioning fixture: cross-attention applied RMSNorm to a permuted Q/K view.
The pinned CUDA RMSNorm backward kernel reads its forward input as contiguous.
Forward values remained correct while adapter gradients were wrong. Both full
and checkpointed native backward produced the same affected loss and gradients.

Normalizing contiguous per-head rows **before** permutation fixes the application
graph without changing ggml or any sibling pin. The first real batch preserves
native forward bytes exactly. Its native preclip gradient norm changes from
6.06275 to 0.39164, versus Torch 0.39502. B/magnitude gradient cosines improve
from 0.1920/0.1858 to 0.99955/0.99971. The correction applies to quantized and
unquantized bases. Earlier byte-identity comparisons against the old native
trainer remain historical evidence; corrected updates should not match that bug.

The new model-free `test-train-cross-attention` checks production graph layout
and finite differences with grouped heads and unequal query/encoder lengths.
It also checks rejection of unsupported multi-example GQA graphs. The underlying
ggml broadcasted `MUL_MAT` backward asserts for grouped heads with batch size
greater than one. Current native GQA training therefore requires microbatch one;
use accumulation for effective larger batches. This is distinct from rank-4
CUDA OUT_PROD support. Any eventual shared-kernel fix requires the coordinated
ggml workflow; no shared-fork change is made here.

The derivative regression requests F32 matmul and checks central differences at
both 0.01 and 0.02 perturbations, retaining the same error tolerance. Initial
1e-3 perturbations were dominated by Vulkan GPU rounding, including with an F32
precision request. This controlled operator test does not establish default
Vulkan arithmetic equivalence; the actual Q4 training checks have their own
execution/continuation scope.

The immutable plan algorithm advances to `ace-native-epoch-seeds-v2`; an older
recorded plan is rejected before preprocessing. Earlier PEFT weights remain
usable for inference or a deliberate fresh-optimizer import. See
`TRAINING_JOB_CONTROL.md`. Prediction retention in checkpointed graphs is opt-in
for diagnostics; default production graphs do not allocate that extra copy.
Before the cross-attention correction, retaining predictions was verified not
to change any of 26 losses, norms, LRs or the final adapter bytes.

### Real-data trajectories and remaining precision work

The regular-base BF16 replay executes **26 updates**, two passes over the 13
prepared songs, using deterministic 4–8-second rotating windows and complete
native conditioning. It uses balanced rank 8/alpha 16 DoRA, FP32 trainable state,
AdamW peak LR 3e-4, two warmup steps, cosine, clip 1, weight decay 0.01, CFG 0.15,
mu 0/sigma 1 and Min-SNR gamma 5. The native fixture calls production checkpointed
backward/AdamW; Torch imports Carey's model, PEFT, loss weighting and optimizer
factories. Actual batches/initial weights/applied LRs are shared. This isolates
decoder/optimizer trajectories, not Carey's Fabric job or original Spark run.

After the fix, final A/B/magnitude parameter-delta cosines improve from roughly
0.642/0.652/0.673 to 0.954/0.954/0.966. Nevertheless the worst independent forward
comparison falls to 0.370 and relative loss error reaches 0.596: **longer-run
numerical equivalence does not pass**. Both runs remain finite. Peak Torch
allocations are about 5119 MiB.

Keeping replayed timesteps FP32 instead of BF16 does not resolve that drift.
Rebasing the worst update (17) onto the exact native update-16 adapter restores
forward cosine to 0.99950, but B/magnitude gradient cosines remain about
0.767/0.825 with BF16 CUDA compute. A Torch FP32 CPU diagnostic improves those
gradient cosines to 0.944/0.936 and forward cosine to 0.99986. This supports
further precision-controlled isolation; it does not establish that all remaining
differences are precision or that another kernel is correct. No CPU ggml BF16
backward result is claimed.

The new reusable `test-train-trajectory-real` / `.py` fixtures preserve exact
batches, initial adapters, metrics and selected snapshots. `--snapshot-step N`
retains weights around an update and its gradients for rebased diagnostics;
`--timestep-precision fp32` isolates that policy. Default Torch adapter precision
is explicitly FP32 for comparison, while `model` retains the actual Carey cast.
Reports are under `base-bf16-balanced-r8-26updates-*`; the local rebased
diagnostic is `base-bf16-balanced-r8-26updates-diagnostic/update17-rebased*.json`.

The practical XL Q4 **production CLI** test completes four epochs / **52 updates**
over all 13 full-song preprocessed examples with balanced rank 64/alpha 128 DoRA,
45–60-second rotating windows, full-song conditioning, peak LR 3e-4, five warmup
steps and the same remaining optimizer/diffusion settings above. An independent
stop at update 19 and resume to 52 reproduces every update's loss/LR/gradient norm
and byte-identical final adapter, config, optimizer and trainer state. Native
checkpoint pointers are resolved before comparing immutable generations.

Two-second device-memory samples peak at 7122 MiB during the cold run and
3992/3990 MiB in warm partial/resumed runs on the 8151 MiB RTX 5070 Laptop.
These are sampled device totals, not guaranteed allocator high-water marks.
The actual DiT logs confirm CUDA0 and use Gary's shared CUDA 12.8 runtime. Cold,
partial and resumed invocations take about 243/73/118 seconds respectively.
This supports practical Q4 execution and continuation, not full-track backward
capacity, 150-epoch convergence or musical acceptance.

Evidence: `xl-q4-balanced-r64-52updates/report.json`, per-run logs and GPU samples.
The long-run binary SHA256 is
`148253883162fc0230d8c016a8c60aca4eeed911265b5394cac6602df24ee8b8`.
It contains the corrected graph and v2 plan; the later source additions reject
unsupported larger microbatches and do not change this batch-one math. Runtime
source for the completed fixes is `ffe48361fdfece1f6b131a921ed799956cc8bc2d`;
ggml stays `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`. All 21 model-free CTests
pass in CPU-only and dynamic CUDA/Vulkan Windows builds. Current-revision Metal,
XL BF16 on Spark, shared-noise preprocessing and tight longer-run parity remain
open.

Final local gates at runtime/test source
`c8762963643b66baf782c890115e4017e3f61cb9`: 21/21 CTests pass in both Windows
builds; the controlled production cross-attention regression passes on CPU,
CUDA0 and NVIDIA Vulkan1. Balanced rank-8 XL Q4 **LoRA and DoRA** each pass the
current uncached/cold/warm/streamed/mid-epoch and in-place resume matrix on both
CUDA0 and NVIDIA Vulkan1. Adapter, config, optimizer and trainer-state payloads
match within each backend. These quick synthetic matrices do not repeat the
older corruption/cancellation suite or claim cross-backend byte identity.
Vulkan runs use only System32 on PATH, with device selection confirmed by props
and DiT execution logs. Evidence is in `crossnorm-final-{cuda,vulkan}-{lora,dora}`
under `build-release-validation/`.

Current source CI is tracked separately:
[CI Build](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38817457978),
[ggml checks](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38817457929),
and [release dry-run](https://github.com/betweentwomidnights/acestep.cpp/actions/runs/38817457942).
Build/signing/package status does not replace Metal training or clean-Windows
runtime acceptance. `full-validation-summary.json` in the local Billie evidence
directory records the final executable hashes; all original audio and the Spark
adapter hashes were reverified unchanged. Gary's working repository advanced
during this work, but the compared preprocessing helper files are unchanged
between its earlier `a4de7db1` and final observed `85183da1`; per-run helper hashes
remain the comparison provenance.
