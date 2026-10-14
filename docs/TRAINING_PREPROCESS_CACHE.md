# Disk-backed training preprocessing

`ace-train` caches full-song sampled VAE latents, source context and text/lyric
conditioning before loading the frozen DiT for training. This follows SA3's
identity-checked pre-encode workflow while retaining ACE's own tensor layout,
posterior sampling and conditioning. It does not claim `.pt`/SA3 cache compatibility.

```sh
ace-train --models models --dataset training-data --output adapter \
  --dit acestep-v15-xl-base-Q4_K_M.gguf \
  --text-encoder Qwen3-Embedding-0.6B-Q8_0.gguf --vae vae-BF16.gguf \
  --preprocess-cache-dir training-cache --cache-memory-mb 256 \
  --preprocess-batch-size 1
```

## Controls and memory

| Option | Default / behavior |
| --- | --- |
| `--preprocess-cache true` | Enabled. Default directory is `DATASET/.ace-train-cache`. |
| `--preprocess-cache-dir DIR` | Override the cache root, including for read-only datasets. |
| `--cache-memory-mb N` | 256 MiB of prepared-tensor LRU residency. Zero streams from disk. |
| `--preprocess-batch-size N` | One cold-preprocessing example at a time. Larger chunks trade host RAM for fewer model reloads. |
| `--preprocess-cache false` | Explicit legacy compatibility/debug mode: preprocess and retain the whole dataset in RAM. |

Enabled caching retains only metadata for the entire dataset. Cold preprocessing
holds at most the configured chunk of prepared examples, plus one decoded audio
file and the encoder's working memory. Training holds an LRU within the specified
byte budget, plus the active example, microbatch/window/collation buffers and
model/optimizer state. An example larger than the budget is read transiently and
not retained in the LRU. Accounting includes allocated vector capacity after
padding. The budget is not a limit on whole-process RAM or VRAM.

Cold chunks preserve model phases and `EVICT_STRICT`: the VAE, text encoder and
condition encoder load sequentially, with one GPU module resident at a time.
The default of one example minimizes preprocessing tensor memory; larger chunks
can amortize repeated model loads. Cached complete chunks skip those encoder
loads entirely. With a partially missing chunk, the whole chunk is recomputed;
choose chunk size one for per-example invalidation granularity.

## Identity and integrity

Each entry is a versioned binary F32 file with its full JSON identity, lengths,
tensor sizes and a checksum. Its name is derived from the identity rather than
user filenames. It includes:

- Content fingerprints of all three GGUFs and that example's WAV/metadata,
  matching the resume plan's conservative file-content identities.
- Example identity and original `seed + dataset_index` posterior seed.
- Caption/genre/tag settings, VAE tiling and attention/clamp settings.
- Application/release and ggml revisions, numerical build configuration,
  selected physical backend/device, CPU thread count, runtime GGML environment
  controls and byte order.

Training rank, adapter kind, optimizer/scheduler options, output paths, cache
location, chunk size and RAM budget do not invalidate preprocessing. Relocating
unchanged data with unchanged example IDs is supported. Changing dataset order
can change posterior seeds and therefore correctly invalidates affected entries.
Different builds/backends receive separate identities; caches are local prepared
tensors rather than a portable exchange format. Uncommitted development changes
to preprocessing should use a fresh cache directory.

Entries contain both audio and conditioning, so a metadata edit rebuilds that
example's complete entry. Other entries remain reusable. The real encoder length
is stored without padding; loading restores the longest sequence length across
the **current complete dataset**, using the original null-conditioning vector.
This preserves the old trainer's attention shapes, including after the longest
caption changes or when training uses only an epoch subset. Full-song latents
remain intact; rotating windows are selected later using the original identities.

Writes stage beside their destination, flush and atomically replace it. Windows
publication retries transient reader locks. Failed writes preserve the previous
complete entry and remove their staging file. Missing, truncated, wrong-identity,
nonfinite or checksum-invalid entries are regenerated during preprocessing.
After training begins, a missing/damaged disk entry fails explicitly; it never
loads preprocessing models over an active DiT to repair it. A write failure is
also explicit: select a writable cache root or intentionally use legacy mode.

The checksum/content fingerprint uses FNV-1a for accidental corruption detection,
not authentication. Header sizes, file lengths and model tensor dimensions are
checked before allocation. Cache files are trusted local application data.

## Cancellation, resume and ownership

Cooperative cancellation preserves complete entries already published. The next
job reuses them and computes missing entries. Temporary files are never cache
hits. Native optimizer checkpoints remain independent of the cache: deleting
cache entries between jobs causes deterministic preprocessing to run again.
Cache controls can change on full-state resume without changing the immutable
training plan. Changing actual models/data/preprocessing still fails resume
compatibility before preprocessing starts.

Old identities remain on disk for reuse by other jobs/builds. There is no
automatic disk-quota eviction or purge of another job's cache. Gary owns the
cache location and storage/cleanup policy; remove unused entries with training
stopped. Dataset discovery/staging, caption policy and adapter catalog ownership
remain downstream.

## Verification

`test-train-preprocess-cache` covers lossless tensors/lengths, global null padding,
byte-budget eviction/zero residency, checksum/truncation/size bounds, wrong
identities, nonfinite values and failed publication preserving old data. The real
CLI plan test checks model/seed/tiling/metadata identity changes and excludes
training/output/cache controls from tensor identity and resume compatibility.

`tests/test-training-cache-real.py` creates variable-length captions and compares
all four checkpoint payloads between uncached, cold, warm, chunked and resumed
runs. It checks corrupt-entry repair, selective metadata invalidation (including
changing the longest caption), optimizer-only reuse, skipped encoder loads on
warm starts and cancellation/reuse of a partial preprocessing cache. An older
binary can be supplied with `--baseline-bin-dir` for algorithm regression checks.
`--quick` restricts the run to baseline/cold/warm and mid-epoch resume.

These are deterministic execution tests, not convergence, Carey/PyTorch numerical
parity or full-song capacity acceptance. Current-revision Metal training remains
the separate `METAL_TRAINING_HANDOFF.md` gate.
