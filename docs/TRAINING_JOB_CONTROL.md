# Native training job control and continuation

`ace-train` follows sa3.cpp's schema-1 file controls, adapted from SA3
`e2f6d1039fabb50ad27729d3cc2013ac9c8002b1`. Hosts can share their supervisor
logic across the sibling trainers. Dataset staging, caption policy, job ownership,
adapter catalog registration and complete-as-repaint remain application work.

## Controls

```sh
ace-train --control-info
# {"schema_version":1,"cooperative_cancel_file":true,"atomic_progress_file":true}

GGML_BACKEND=CUDA0 ace-train --models models --dataset training-data \
  --output adapter --dit acestep-v15-xl-base-Q4_K_M.gguf \
  --text-encoder Qwen3-Embedding-0.6B-Q8_0.gguf --vae vae-BF16.gguf \
  --adapter-type lora --profile balanced --rank 64 --alpha 128 \
  --epochs 10 --batch-size 1 --accumulation 2 \
  --window-min-seconds 2 --window-max-seconds 2 \
  --progress-file job/progress.json --cancel-file job/cancel
```

`--control-info` needs no models. Use different progress/cancel paths and one
writer per job. Duplicate control flags and aliases of the same file are rejected.
The host creates the cancellation file to request a stop and removes it before
launching another job. The trainer leaves it in place.

Progress replaces the entire JSON atomically, using a process-specific staging
file; Windows replacement retries temporary reader locks. A reader may itself
need to retry a transient Windows sharing error. Schema 1 has these fields:

| Fields | Meaning |
| --- | --- |
| `schema_version` | Integer `1`. |
| `status` | `starting`, `running`, `completed`, `cancelled`, `failed`. |
| `phase` | `loading`, `validating`, `pre-encode`, `training`, `saving`, or the terminal status. |
| `message`, `error` | Human-readable context; error is populated for a failed job. |
| `step`, `epoch`, `max_steps` | Completed optimizer updates, current training epoch (one-based during updates), and total execution target. Before an update, epoch may be zero or the restored completed-epoch count. |
| `loss`, `learning_rate`, `step_seconds` | Last update's last-microbatch loss, LR and elapsed time; nonfinite values serialize as `null`. Initial values are zero. |
| `final_adapter` | Absolute PEFT output **directory**, populated after a successful final save, including checkpoint-on-cancel. |
| `adapter_checkpoint`, `state_checkpoint` | Absolute immutable generation **directories**, populated after a successful save. The state path can be passed directly to `--resume`. |

Unlike SA3's GGUF file paths, ACE artifacts are directories containing PEFT
weights/config and optimizer/trainer state. Check `status` before registering a
result: a cancelled job may have a usable adapter, but is not completed training.
Exit codes follow SA3: `0` for completed or cooperative cancellation, `1` for
runtime failure and `2` for usage errors. The host must inspect both exit status
and terminal JSON. Process kills, native backend aborts and power loss cannot
promise terminal JSON or a new checkpoint; stale `running` JSON is not success.

Cancellation before model loading or the first optimizer update creates no
adapter. Preprocessing checks between VAE examples and before text encoding;
active model loading, hashing and GPU operations finish before a safe check.
During training, the trainer finishes the current accumulated optimizer update
before saving. No half-applied update or partial gradient accumulator is saved.
SA3's sample boundary therefore becomes an optimizer-update boundary in ACE;
larger accumulation settings can make a cancellation take longer.

## Persisted plan and resume

Every new native checkpoint includes `trainer_state.json.continuation` inside
the existing integrity-checked immutable generation selected by
`checkpoint.current`. It records:

- Versioned algorithm/settings, all three model content identities, ordered
  dataset IDs and WAV/metadata content identities.
- Epoch, next microbatch, shuffle order and running epoch loss/count.
- The original scheduler budget, optimizer hyperparameters, seed, profile,
  adapter type/ranks/scales, preprocessing, windows and diffusion options.

Model and dataset identities use the existing FNV-1a content fingerprint; they
detect accidental changes and are not cryptographic authenticity checks. Input
files must remain unchanged during a job. Moving unchanged files is supported.
Resume validates the plan and cursor before preprocessing, then restores PEFT
weights, AdamW moments and optimizer step. It rejects changed inputs/settings,
invalid permutations, inconsistent steps and partial accumulation cursors.
Seeds are derived from epoch/batch/example identities, so the saved cursor and
order reproduce the sequence without importing SA3's stateful RNG format.

Keep the original `--epochs` on resume: ACE's linear/cosine schedules depend on
the total epoch-derived step budget. `--steps N` is an optional **total optimizer
step stop target** within that original budget, rather than additional steps.
For example, `--epochs 10 --steps 20`, followed by the same arguments with
`--resume adapter --steps 40`, continues the original ten-epoch schedule to step
40. Omitting `--steps` runs to the original budget. A target below the checkpoint
step or above the budget is rejected. This differs from SA3's budget-independent
scheduler and avoids silently changing ACE's existing training math.

Output/control paths, checkpoint cadence and the execution stop target can
change on resume. Backend selection is not part of compatibility: changing
backends is allowed, but bitwise equality across backends is not promised.
An existing native output can only be reused when resuming its current immutable
checkpoint; resume to a fresh output directory to branch an older checkpoint.

`--checkpoint-every N` retains its existing **epoch** meaning.
`--checkpoint-every-steps N` additionally saves `checkpoint-step-N` directories
at complete optimizer updates. Either cadence can be disabled with zero. Final
and cooperative-cancel saves use the output root. Hosts should resolve the
pointer before reading files and retain complete generations when moving state.

Older full-state checkpoints lack the immutable plan. They are rejected unless
`--allow-legacy-resume` is explicit, require epoch-boundary counters compatible
with the supplied settings, and warn that original settings cannot be validated.
Use the original inputs/options/budget; the next save upgrades the format.
Ordinary PEFT-only adapters remain accepted as **weight-only restarts**: moments,
cursor and schedule start at zero, and the trainer logs this distinction.

## Validation and remaining gates

Model-free CTests cover control parsing, JSON replacement/Windows lock behavior,
capability probing, startup failure/pre-load cancellation and real CLI plan/cursor
compatibility. The opt-in real-model regression creates three four-second WAVs,
uses rank 64, two-second windows, accumulation two and four optimizer updates:

```sh
python tests/test-training-control-real.py --bin-dir build/Release --models models \
  --backend CUDA0 --expected-device NVIDIA --adapter-type lora --output new-control-results
```

Repeat with `dora-rows` and each supported exact backend. It compares every
adapter/config/optimizer/trainer-state file after uninterrupted training,
mid-epoch stop/resume, cancellation during accumulation and legacy epoch migration.
It also rejects changed LR, epoch budget and data before preprocessing. These
synthetic checks establish execution/continuation, not convergence or full-song
memory acceptance. See `TRAINING_CONTROL_VALIDATION.md` for measured results.
The script probes and records physical hardware before training;
`--expected-device` rejects a mismatched description. Rediscover Vulkan indices
after changing the runtime PATH or package composition.

Disk preprocessing cache and bounded tensor loading are described in
`TRAINING_PREPROCESS_CACHE.md`. Resumed best-loss history/selection, WebUI training,
Carey/PyTorch parity and downstream integration remain open. Current Metal LoRA
and BF16 frozen-DiT backward gates remain in `METAL_TRAINING_HANDOFF.md`.
