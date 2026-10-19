# Native training history

`ace-train` appends `OUTPUT/metrics.jsonl`, following the metric names in
sa3.cpp's `src/train_job.h` at `b844016d49628ec730c273b78c4e9e18ecca5fef`.
Each complete optimizer update produces one JSON object followed by a newline,
flushed before publishing live progress. No best-adapter ranking is performed.
Periodic checkpoints and the output saved at completion/cancellation retain
their existing meanings.

| Field | Meaning |
| --- | --- |
| `epoch` | One-based epoch, matching ACE's CLI progress. |
| `update` | Global completed optimizer-update count. |
| `split` | `train`; no validation-quality metric is implied. |
| `lr` | Learning rate actually used for this update, before advancing the schedule. |
| `loss` | Example-weighted mean microbatch loss over this accumulation window. The live progress/log now reports this same mean rather than only the last microbatch. |
| `grad_norm` | Global L2 norm of the example-averaged accumulated gradients **before clipping**, including DoRA magnitudes. |
| `step_s` | Wall time for this accumulation window; excludes preceding model loading/preprocessing and checkpoint saves. |
| `microbatches`, `examples` | Actual accumulation sizes, including short final windows. |
| `session` | Unique invocation identifier; groups one uninterrupted invocation's samples. |
| `start_update` | Restored optimizer step at invocation start; zero for new/weight-only training. |

SA3 logs per example, potentially repeating an update number while accumulating;
ACE logs once per completed optimizer update. ACE therefore omits SA3's single
example `id`, scalar timestep and CFG flag, which would be ambiguous for a batch.
The shared `update/lr/loss/grad_norm/step_s` fields support the same chart reader.
This does not implement SA3's WebUI run registry or its HTTP history endpoints;
the application owns job/run metadata and presentation.

## Resume and failure behavior

Resuming into the same output directory appends a new session. Existing complete
records remain, including updates later than a rolled-back checkpoint. Readers
must use `session` and `start_update` to distinguish replayed steps. Never treat
all rows as one monotonic series without considering session boundaries.

Resuming into a different directory creates a new journal starting at the
restored step; earlier history stays in the source run. The journal is a
diagnostic artifact, separate from immutable checkpoint generations, and is
not required for deterministic resume. Copying an adapter or checkpoint alone
does not copy history. The native continuation continues to persist epoch loss
and cursor state independently.

On open, an incomplete final line from an interrupted writer is removed before
appending. Complete newline-terminated records are preserved. Live readers
should ignore an unterminated final line and retry. Use one writer
per output directory. A storage/serialization failure fails the job explicitly;
nonfinite samples are never emitted as invalid JSON. The file is flushed per
update, but is not a transactional or power-loss-durable optimizer journal.
The application owns retention and disk quota. No moving average, automatic
checkpoint ranking or claims about musical quality are included.

## Validation

`test-train-history` covers append/resume sessions, torn-tail repair, nonfinite
rejection and storage failures. `tests/test-training-cache-real.py` checks real
GPU journal fields, accumulation counts and restored update numbering alongside
byte-exact adapter/optimizer/native-state comparisons.

See `TRAINING_CAREY_PARITY.md` for numerical scope and remaining differences.
