# gary4local readiness audit — 2026-10-12

This audit compares the working source in `C:/dev/gary-localhost-installer`,
`C:/dev/sa3.cpp`, `C:/dev/yue2.cpp`, and `C:/dev/stems.cpp`. It prepares the
upstream runtime contract; it does not activate a native Carey service or
claim that the existing Python APIs have been replaced. Local consumer edits
were inspected without changing them.

| Repository | Application HEAD inspected | Actual pinned ggml |
| --- | --- | --- |
| acestep.cpp | `e93ee47f` (merged bump PR #9) plus this readiness change | `4ad3b30bfe452efc5f4f9c0e111e017d9970292a` |
| gary-localhost-installer | `eae118fafecaa7c3364564eff53ac2b8ced02f97` plus local edits | Separate native service packages |
| sa3.cpp | `b844016d49628ec730c273b78c4e9e18ecca5fef` | `60f49e09ce67df544831a357b654318ec2722fb0` |
| yuey.cpp (`yue2.cpp` checkout) | `82ed90333f7fb299459b965ff1e75a5383d784a3` | `4ad3b30bfe452efc5f4f9c0e111e017d9970292a` |
| stems.cpp | `d423a4863a1851b826d5be28c988e46c925cefcc` | `4ad3b30bfe452efc5f4f9c0e111e017d9970292a` |

The starting ACE checkout was `ed00def1`/`fff93d27`. After initial checks at
`76981988`/`60f49e09`, PR #9 was merged at the user's request and the release
branch incorporated `e93ee47f`/`4ad3b30b` before final validation. Earlier
`60f49e09` smoke results are baseline evidence only. No new ggml patch is
introduced by the release branch.

These pins are intentionally reported from each checkout rather than inferred
from historical shared-fork notes. This change neither modifies ggml nor bumps
any consumer. Shared CUDA compatibility is independent of the ggml pin; every
service still needs its own ggml binaries and matching compile definitions.

## Release blockers addressed here

- Replaced the old monolithic/unchecked release workflow with the sa3.cpp
  Windows split/core/backend/standalone pattern and universal signed/notarized
  macOS packaging. Added reproducible entry points, checksum files, license
  notices, build metadata, tag/version checks and an artifact-only dry run.
- Added bare semantic `--version` to the server and trainer and model-free
  `--props` with devices/capabilities. Environment settings cover model/adapter
  paths, port, host, device and adapter mode; CLI overrides them.
- Added exact device names and `cuda`, `vulkan`, `cpu`, `metal` aliases. An
  explicit unavailable device fails. Props canonicalizes `MTL` to `Metal`,
  because the current installer checks backend names by substring.
  Backend aliases prefer a discrete GPU over an integrated GPU; exact names
  preserve the selected device. Vulkan indices must be discovered from props,
  because their ordering can change with the runtime environment.
- The server defaults to functional adapters, preserving a quantized frozen
  base rather than merging and requantizing its adapter. Historical merge
  mode remains available explicitly; the synthesis CLI retains its own default.
- Training refuses ambiguous model buckets rather than using alphabetical
  order. A directory containing base, turbo and XL variants therefore requires
  exact `--dit` selection; multiple encoders/VAEs also require explicit names.
- Converted tests that called CPU functions directly to the backend registry
  and proc-address interface. They now link and run with dynamically loaded
  CPU variants, which is the actual downloadable Windows configuration.

## Training comparison and remaining work

The key application source is `services/carey/train_lora_job.py`. It owns
captioning, canonical sidecars, two-pass preprocessing, child-process tracking,
atomic status files, cancellation, result selection and LoRA registry updates.
The UI/Tauri lifecycle currently launches this Python job, not `ace-train`.
Carey's production path explicitly disables model quantization and uses BF16
Torch; native XL Q4 is a new memory-saving path, not a claim of bitwise parity.

| Area | Native implementation / action for migration |
| --- | --- |
| LoRA / DoRA | `--adapter-type lora` / `dora-rows`; Carey exposes `dora` and translates it to PEFT `use_dora`. Map the name explicitly; do not send `dora` to the native CLI. |
| Balanced ranks | Native attention/balanced profiles cover all 24 base / 32 XL layers. Saved native rank/alpha patterns and tensor ranks match Carey's balanced reference-rank-8 configuration across all 352 XL projections. See the low-rank follow-up below. Torch/native RNG initialization and numerical training parity remain open. |
| Optimizer/loss | AdamW, accumulation, clipping, CFG dropout, two-draw logit-normal timesteps and Min-SNR exist natively. Map every option explicitly; native Min-SNR defaults off, whereas the Carey job defaults to `min_snr`. |
| Defaults | Both use rank 64/alpha 128 in the job-facing defaults, but epochs, LR, warmup, checkpoint interval and adapter kind must come from the saved application plan, not implicit CLI defaults. |
| Training windows | Native `--window-min-seconds`/`--window-max-seconds` bound DiT backward activation memory while retaining full-song conditioning. The application max-duration/preprocessing policy is a different operation; do not equate them. |
| Training device | Export `GGML_BACKEND` with the selected exact device or backend alias for the `ace-train` child. `ACE_DEVICE`/`--device` configure the server; do not assume that service environment variable also forces the trainer. |
| Dataset | Native accepts flat paired WAV/required canonical metadata. Carey discovers nested WAV/MP3/FLAC/OGG/Opus/M4A, captions and normalizes metadata. Application staging must decode/flatten safely, avoid basename collisions and produce every required field, with `lyrics:` last. |
| Preprocessing | Native now caches full-song sampled latents/context/conditioning on disk, prepares bounded chunks and loads training examples through a byte-budget LRU (zero streams). Model/data/seed/settings/build/backend identities and checksums guard reuse; complete entries survive cancellation. CUDA/Vulkan Q4 LoRA/DoRA cold/warm/resume match the uncached baseline byte for byte; see `TRAINING_PREPROCESS_CACHE.md` and `TRAINING_CACHE_VALIDATION.md`. Carey `.pt` caches are a separate format. Gary owns location/quota/cleanup/staging; representative-dataset capacity and numerical parity remain open. |
| Full resume | Native now persists the complete immutable input/config/scheduler plan and mid-epoch shuffle/cursor/loss state alongside moments and PEFT artifacts. Resume validates compatibility before preprocessing. Keep the original epoch budget; `--steps` can extend the execution target within it. Resolve `checkpoint.current` before reading its immutable generation. Older state requires explicit `--allow-legacy-resume`; PEFT-only import is a weight-only restart. See `TRAINING_JOB_CONTROL.md`. |
| Output selection | Native saves periodic checkpoints and the adapter at completion/cancellation. Best-loss ranking is intentionally excluded: training loss does not establish musical quality. Gary should register the chosen completed output rather than require Python's `best/` convention. |
| Training history | Native appends `metrics.jsonl` with SA3-compatible metric names, one row per optimizer update, including mean accumulated loss, applied LR, pre-clip gradient norm, timing and resume-session identifiers. See `TRAINING_HISTORY.md`. Run registry/UI/retention remain application responsibilities. |
| Status/cancel | Native now mirrors SA3's schema-1 `--control-info`, atomic `--progress-file` and cooperative `--cancel-file` contract, with checkpoint-on-cancel at a complete optimizer update. Optional step-based checkpoints complement the existing epoch cadence. The supervisor must check exit status and terminal JSON; kills/backend aborts cannot promise a final snapshot. Application lifecycle/registration still needs adaptation. |
| Auto captioning | Native `/caption` now exposes normalized queued JSON suggestions and `/analyze` / `ace-audio-analyze` port SA3's model-free BPM/key helpers. Both trainers can consume the shared caption provider once Gary adapts the native job API. Caption pools, confidence/filename precedence, triggers, genre/instrumental policy and sidecar writes remain application responsibilities. See `TRAINER_CAPTION_API.md`. |
| Adapter registration | Map model family, allowed inference variants, adapter type/profile, checkpoint path and caption pool into Carey's catalog. Validate incompatible XL/base shapes before allowing selection. Restart/rescan is currently needed to refresh the native server's adapter registry. |
| Torch parity | Focused fixtures import the actual Carey batch/loss/optimizer/scheduler code; known padding, warmup, restart and partial-accumulation policies require explicit decisions. The real-model comparator now accepts Carey sources and both LoRA/DoRA. See `TRAINING_CAREY_PARITY.md` for evidence and remaining full-preprocessing/XL Q4 coverage. CPU fixtures or audible output do not close the full gate. |
| Backend coverage | Validate LoRA and DoRA separately, by base tensor type and device, including full resume and functional inference. Current-revision Metal LoRA remains open. At `4ad3b30b`, CPU/Metal OUT_PROD guards still omit BF16; do not offer BF16 frozen-DiT backward there. BF16 VAE inference does not validate that path. Unquantized F16/F32 and quantized bases are separate coverage. |

These are visible migration gaps, not reasons to claim the existing Carey
installation can switch runtimes immediately. This release prepares downloadable
CLI/server components that can be tested through a native candidate first.

## Low-rank balanced profiles — follow-up, 2026-10-13

Keep the shared ggml CUDA low-rank issue in this audit until it is fixed and
validated. It is a known limitation of the affected configurations, not a
reason to change Carey's projection budgets or reject all rank-4 adapters.
Do not silently clamp projection ranks or substitute the attention profile.

Carey at `238549ab0052bc514a781e62e7a6fef6020dc710` passes the selected rank and
balanced profile from `services/carey/train_lora_job.py` into
`acestep/training_v2/cli/config_builder.py`. Its
`adapter_profiles.py::build_balanced_projection_profile` computes
`max(1, round(base_rank * multiplier))`, then supplies the per-family rank and
alpha patterns to PEFT through `training/lora_utils.py`. DoRA uses the same
LoRA configuration with `use_dora=True`, so it has the same projection ranks.
The current `CareyAceTrainingModal.svelte` already accepts integer reference
ranks starting at 1; the Python job validates positivity without clamping to 8.

| Projection family | Balanced rank 4 | Balanced rank 8 |
| --- | --- | --- |
| Self attention Q / K / V / O | 1 / 2 / 5 / 4 | 2 / 3 / 10 / 7 |
| Cross attention Q / K / V / O | 4 / 2 / 2 / 3 | 8 / 5 / 4 / 6 |
| MLP gate / up / down | 2 / 3 / 3 | 5 / 6 / 6 |

The native implementation uses the same multipliers and ties-to-even rounding.
The current Carey environment (Torch 2.7.1+cu128, PEFT 0.18.1) was exercised with
the actual profile builder and adapter injector on a tiny two-layer CPU model:
all 22 injected projections had the expected A/B ranks for reference ranks 4
and 8, for both LoRA and DoRA, and finite forward/backward results. This confirms
that Carey really allocates rank-1 self-attention Q adapters at balanced rank 4;
it is not a complete Carey GPU-training or Torch/native parity test.

On the 5070, native ACE balanced rank 4 / XL Q4 aborts in CUDA OUT_PROD with
an invalid/unsupported cuBLAS parameter; the older portable ACE binary also
reproduces it. ACE attention rank 4 passes a bounded training run. Downloaded
SA3 v0.1.4 (`da74bc15`, shared ggml `4ad3b30b`) passes one medium-base Q4 LoRA
update at uniform rank 4, while uniform rank 1 aborts at the same OUT_PROD call.
This points to shared rank-1/layout handling; the exact failing operator still
needs isolation before a kernel patch is claimed.

Balanced rank 8 / alpha 16 was also trained against XL-base Q4 on CUDA0 and
NVIDIA Vulkan1 with both LoRA and DoRA-row. Each uninterrupted synthetic run
completed four optimizer updates: three four-second WAVs, two epochs, batch 1,
accumulation 2 (including an epoch's partial tail), checkpointed backward and
two-second rotating windows. Inspecting all 352 saved XL projection pairs in
each run confirms a minimum actual rank of 2 and rank/alpha patterns identical
to Carey's helper. All four suites also pass byte-exact comparison of adapter,
config, optimizer and trainer-state payloads after mid-epoch stop/resume,
accumulation cancellation/resume and explicit legacy epoch migration. Changed
LR/data/epoch budget, unapproved legacy resume and accidental output reuse are
rejected before preprocessing. The balanced rank-8 continuation gate therefore
passes on this NVIDIA GPU; the balanced rank-4 CUDA fix remains a deferred
follow-up before exposing that native configuration as supported downstream.
Runtime source is `2d140f7d46ebd037ea162e68a8a1a6673b1d39b4`,
with shared ggml `4ad3b30bfe452efc5f4f9c0e111e017d9970292a`, using the same
local Release/dynamic-backend build and 5070 as the rank-64 checks. CUDA uses
Gary's shared 12.8.0 runtime. These synthetic checks establish execution and
profile mapping, not convergence, full-song memory capacity or Metal support.

Local investigation logs, exact commands, package checksums and actual PEFT
tensor-rank results are under `build-release-validation/rank-investigation/`.
Any future shared ggml fix needs low-rank operator coverage, LoRA/DoRA
training/resume checks and coordinated exact-candidate consumer gates before
adoption. No GitHub issue or shared ggml change is required to retain this note.

## GPU detection and quantization

`control-center/src-tauri/src/native_runtime.rs` already owns GPU detection,
backend preference, split-package installation, shared runtime PATH setup and
post-extraction props probing. Preserve that owner. It tries automatic GPU
candidates in order and rejects a failed explicit selection; copying a second
independent hardware detector into Carey would create conflicting decisions.

`lib.rs::choose_yuey_encoding` selects an installed explicit tier, then the
probe's installed recommended tier, then the smallest installed tier. Yuey's
`runtime_info.cpp::recommended_quantization` uses thresholds of 16 GiB BF16,
12 GiB Q8, 10 GiB Q5, otherwise Q4. Those are yuey inference policy, not measured
ACE XL training budgets. ACE props reports memory and its low-memory XL Q4
path without inventing an inference/training memory-fit heuristic.

For this RTX 5070 Laptop GPU (8151 MiB), the initial native training plan should
require `acestep-v15-xl-base-Q4_K_M.gguf`, batch 1, checkpointed backward and
bounded windows. Family and precision must be independent selections. Never
silently replace XL-base with smaller base/turbo or dequantize the entire base.
Fail with a useful memory/configuration error if the requested plan cannot fit.
Rank 64 balanced/full-song workloads need their own measured acceptance run;
a tiny smoke run establishes execution, not useful convergence or capacity.
The low-rank follow-up above records passing balanced rank-8 LoRA/DoRA checks
on NVIDIA CUDA/Vulkan and the deferred balanced rank-4 CUDA OUT_PROD failure.
Do not generalize these measured ranks to every configuration; the broader
rank-64 validation is recorded in `TRAINING_CONTROL_VALIDATION.md`.

The downloaded portable CI packages passed XL-base Q4 LoRA training, full-state
resume and functional inference on this GPU through CUDA and Vulkan. CUDA used
gary4local's existing shared 12.8.0 runtime; standalone HTTP synthesis passed
with bundled CUDA. See `RELEASE_VALIDATION.md` for the exact revisions and the
two-step fixture's limits. This does not activate the native application path.

## Complete remains application continuation

`carey_wrapper.py::_build_release_task_data` translates application `complete`
to backend `repaint`, preserves the prefix, and sets the repaint interval from
the source duration to the requested output duration. Preserve upstream task
names and implement this mapping in the native Carey adapter. Native `complete`
has its own task meaning and must not be globally renamed.

Migration tests must cover source prefix preservation, target duration,
source-end/target-end bounds, seed repeatability, model-family compatibility,
adapter selection/scales, extension beyond the original waveform and sample
rate. Exercise the actual DAW-facing request and status/result payloads.

## Application handoff gates

PE imports confirm that the Windows server and CPU backend depend on the shared
MSVC/OpenMP runtimes (`MSVCP140`, `VCRUNTIME140`, `VCRUNTIME140_1`, `VCOMP140`).
The core archive carries them beside the executables, as every sibling's does,
so gary4local needs no VC Redistributable prerequisite and never elevates for
one. This replaces the preflight and Microsoft-installer repair first drafted in
[Gary PR #8](https://github.com/betweentwomidnights/gary-localhost-installer/pull/8).
Startup on a clean Windows machine without build tools is still an acceptance
gate; startup on this Visual Studio machine does not establish it.

1. Add a Carey native **candidate** with `acestep` asset names; keep Python
   active while preparing models/runtime. No hashes should be invented before
   publication; use local package overrides for initial installation exercises.
2. Implement Carey job/API adaptation for generation, cover, extract, lego,
   complete-as-repaint and progress/results/audio transport. ACE's `/synth`
   JSON/multipart and `/job` contracts differ from Carey's endpoints.
3. Add a native training job owner, explicit option mapping, dataset staging,
   status/cancel handling, resume plan validation, best/final registration and
   model-family checks. Stop generation and competing services before training.
4. Run CUDA and Vulkan through core + one backend and the shared CUDA runtime,
   then standalone. Confirm the requested GPU in logs and test unavailable
   backend/error paths; CPU fallback is not GPU validation.
5. Run current-revision Metal LoRA unquantized/Q4, resume and trained-adapter
   inference on Apple Silicon before checking the README Metal TODO. The planned
   BF16 backward implementation and shared-fork gates are listed in
   `METAL_TRAINING_HANDOFF.md`.
6. Test published download/hash/install/update/restart and a representative
   XL Q4 dataset inside gary4local. Only then activate native Carey and remove
   its Python payload, following the SA3 staged migration pattern.

See `packaging.md` for archive and CI details. Local validation evidence for
this change is recorded separately in `RELEASE_VALIDATION.md`.
