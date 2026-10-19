# Current-revision Metal training handoff

The PC release work leaves ggml pinned at
`4ad3b30bfe452efc5f4f9c0e111e017d9970292a`. The owner reports a successful
DoRA adapter trained on Q4 XL-base on this PC. Preserve that result separately
from the recorded automated LoRA execution/resume and DoRA operator checks;
none of those closes current-revision Metal validation.

BF16 frozen-DiT backward is still unsupported by the pinned Metal OUT_PROD
guard. BF16 VAE inference and older Metal smoke results do not establish that
training path. Implement and validate the shared ggml change on the M4, following
the shared-fork workflow, before offering BF16 bases there.

1. Record the M4 hardware, OS/toolchain, application hash, exact shared baseline,
   candidate hash and build flags. Fetch and inspect the current shared tip;
   do not assume an older historical note is still the agreed baseline.
2. Add focused BF16 OUT_PROD numerical/backward tests, including accumulation,
   representative training shapes and non-contiguous inputs where supported.
   Compare against a CPU/PyTorch reference; verify `MTL0` actually executed.
3. On the exact combined candidate, run ggml-bump-check's Metal fleet across
   ACE, SA3, audiocraft, yuey and stems. Read its current README/config first;
   force reused workdirs because the cache does not key successful runs by ggml
   commit. Match `GGML_MAX_NAME` across loaded libraries and executables.
4. Exercise ACE LoRA and DoRA separately on BF16, existing F16/F32 and Q4 frozen
   bases: forward/loss/gradient/optimizer parity, train/export, interrupted versus
   uninterrupted full-state resume, and functional adapter inference without
   mutating the base. Resolve `checkpoint.current` before inspecting saved state.
   Record missing models, skipped workloads and any CPU fallback explicitly.
5. Test a representative XL Q4 dataset and production rank/window settings with
   competing applications stopped; tiny execution tests do not prove memory
   capacity or convergence. Check full-song export and resume through the trainer
   settings that Gary will expose.
6. Run CUDA/Vulkan regression gates on the exact adoption candidate before
   merging the shared change and coordinating all five consumer submodule bumps.
   Application CI must then rebuild/sign/notarize the adopted revision.
7. Exercise the macOS package's analysis/caption endpoints and the actual Gary
   training job owner, cancellation, registration and restart/install flow.

Only mark the README Metal LoRA TODO complete after current-revision unquantized
and quantized training, resume and trained-adapter inference have passed on
Apple Silicon. Keep BF16 coverage explicit rather than extending a claim from
another tensor type.
