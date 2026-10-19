# Caption and genre mixing

`ace-train --genre-ratio 30` substitutes a tagged genre prompt for roughly
30 percent of tracks each epoch. Each selected track still contributes one
example; this does not duplicate audio or increase the update budget. Default
zero retains caption-only training. `--use-genre` keeps its existing global
genre behavior and cannot be combined with a positive ratio.

This follows current Carey's rotating policy, reviewed at
`gary-localhost-installer` commit `f0ff85ae286556dfbb2ea5fb88128dadd1327ecb`,
in `services/carey/acestep/training/data_module.py::PromptVariantSampler`.
Native retains its own seeded shuffle/RNG; identical seeds do not imply the
same Python and C++ track order. The archived Spark run's fixed subset is a
different policy.

Counts use cumulative rounding: three tracks at 30 percent produce nine genre
selections over ten epochs. Genre-capable tracks are selected from a fixed
seeded shuffled cycle, using each before repeating. Missing genres retain the
caption, so the achieved ratio is capped by availability. With
`--examples-per-epoch`, counts and eligibility refer to the selected subset;
the cycle is stable when that population is stable.

Both prompts apply `custom_tag` with the selected `--tag-position` (prepend,
append or replace), and retain the same full-song lyrics, duration and musical
metadata. A replacement tag can intentionally make the two prompts identical.
The application supplies these metadata fields and the desired ratio.

Preprocessing runs the VAE once per track and encodes both conditioning variants
for tracks with genres. A mixed cache entry stores shared audio/context tensors
and two conditioning tensors; its identity and binary format distinguish it from
the existing single-prompt cache. Both conditioning allocations count toward the
LRU byte budget. Changing a positive ratio can reuse those tensors, but changing
the ratio rejects full-state resume because it changes the training trajectory.
Existing zero-ratio/global-genre plans and caches keep their prior representation.

The immutable plan records `genre_ratio` and `ace-rotating-prompts-v1`. The saved
epoch/order/cursor, seed and fingerprinted metadata reconstruct the selection
without an extra mutable RNG stream, including a resume inside an epoch.
Periodic history and checkpoint selection are unchanged.

Validation on 2026-10-18: model-free checks cover fractional counts, fair cycle
coverage, missing genres, subsets, tag modes, variant cache integrity/padding,
memory accounting and ratio compatibility. CUDA0 and NVIDIA Vulkan0 XL-base Q4 balanced rank-8
LoRA and DoRA each pass uncached/cold/warm-streamed and mid-epoch resume
comparisons with identical adapter, optimizer and trainer-state payloads. The
three-track fixture includes a track without a genre and unequal caption
lengths. This is functional/resume coverage, not a new long-run musical result.
Local evidence: `build-release-validation/trainer-gaps-20261018/`.
