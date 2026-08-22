# ACE-Step native adapter-training research

## Outcome

Native ACE-Step LoRA and DoRA-row training is feasible without replacing ACE-Step's inference graph.
The narrow integration is a functional linear-transform hook in the existing DiT graph plus a port of
the GGML training kernels from the SA3 fork.

## Compared revisions

- `ServeurpersoCom/acestep.cpp` at `9761469d95fc204b5468623c68a1a2203e50b1f9`
- ACE-Step's `ServeurpersoCom/ggml` submodule at `c044c6f03892f9d5e98213b05f8afea1f8b0d3c9`
- `betweentwomidnights/sa3.cpp` at `0942dc320d49dadc947b7a2e17c0b2b97966dae1`
- SA3's `betweentwomidnights/ggml` submodule at `ba817aa16736bd7a5bca61e7e3ac446b26a266b0`
- `betweentwomidnights/gary-localhost-installer` at `89ae662a37d875428e1a34fc7f13540e129dee5f`

The two GGML histories share
`524f974bb21a1013408f76d71c15732482c0c3fe` as their relevant base.
Replacing ACE-Step's GGML with SA3's older pin would discard later ACE-Step
backend work, so the spike rebased 17 focused training commits onto ACE-Step's
newer GGML revision instead.

## GGML capability port

The port covers the backward graph and the backend operations required by functional adapters:

- generic backward support for concatenation, strided tensors, and contiguous gradients;
- allocator capacity for large adapter graphs;
- F16 and quantized frozen weights in `OUT_PROD` backward;
- CUDA strided unary and quantized `OUT_PROD` support;
- Vulkan tiled and quantized `OUT_PROD` support;
- Metal backward operations plus tiled and quantized `OUT_PROD` support.

On an Apple M3 Max, the unmodified ACE-Step GGML pin reports zero supported Metal `OUT_PROD` cases.
The rebased port passes 128 of 128 selected Metal cases, including F32, F16, Q5_K, and Q6_K source
weights. The complete ACE-Step CMake build also succeeds against the port. A production Vulkan build
passes the same 128 of 128 `OUT_PROD` cases on an Apple M3 Max through MoltenVK, including the explicit
Q5_K and Q6_K training cases. The original SA3 CUDA commit records end-to-end LoRA and DoRA-row training
on a Q4_K_M base, with losses matching its Vulkan implementation to the fifth decimal. That proves the
source algorithm before the rebase, but the rebased ACE-Step head still needs build and execution on a
matching CUDA runner; inherited results will not be treated as validation of the integration.

## Model-side training seam

SA3's trainer applies adapters functionally:

```text
y = W x + scale B(Ax)
```

DoRA-row training then scales each output row by its trainable magnitude divided by the norm of the
effective row. The norm is expanded into base, cross, and low-rank-square terms, so the graph never
materializes `W + scale AB`. This is essential for quantized bases and avoids retaining full F32 copies
of every adapted projection through the backward pass.

The ACE-Step spike routes every DiT linear projection through an optional model callback. Frozen
inference leaves the callback unset and continues to use `ggml_mul_mat`. Training installs the functional
adapter transform. The transform also resolves GGML weight views, which is necessary because ACE-Step's
fused cross-attention uses Q and KV slices of one stored projection.

The real CPU integration test verifies:

- callback dispatch and unchanged frozen inference fallback;
- numerical DoRA-row forward output;
- adapters attached to sliced fused weights;
- gradients for LoRA A, LoRA B, and DoRA magnitude;
- an analytical adapter gradient against a finite-difference result.

The subsequent trainer spike also implements Gary's `attention` and `balanced` module profiles. The
balanced profile preserves the family-specific rank multipliers and proportional alpha values used by
Gary. Training loads projections without inference fusion so Q, K, V, O, gate, up, and down modules
retain independent PEFT factors. A reusable GGML graph now constructs the complete ACE-Step DiT
forward pass, temporally masked flow-matching loss, and backward gradients while leaving inference fusion
enabled by default. A tiny one-layer real graph executes on the CPU backend with finite loss and nonzero
adapter gradients.

The native state and optimizer layer now also provides:

- Kaiming-uniform LoRA A initialization, zero LoRA B initialization, and identity-preserving DoRA-row
  magnitude initialization from frozen weight norms;
- AdamW with bias correction, decoupled weight decay, and global gradient-norm clipping;
- PEFT-compatible `adapter_model.safetensors` and `adapter_config.json` output, including heterogeneous
  `rank_pattern` and `alpha_pattern` values;
- checkpoint resume with exact tensor-name, shape, and data round-trip coverage;
- DoRA magnitude loading in ACE-Step's native inference adapter merge path.

The diffusion batch path matches Gary's core fixed-training semantics: it samples two logit-normal
timesteps per item, retains their maximum, sets the reference timestep equal to it, interpolates
`x_t = t * noise + (1 - t) * data`, targets `noise - data`, applies per-sample classifier-free
condition dropout, and packs model input channels as `[context_latents, x_t]`. The DiT graph now accepts
per-sample timestep vectors rather than a batch-shared scalar. A deterministic two-item CPU test uploads
the prepared batch, executes the complete forward/backward graph, and performs a real AdamW replay.

Optional Min-SNR weighting matches Gary's flow-timestep convention. Loss weights are defined per latent
frame, normalized by each example's real duration, and zeroed over padding; padded attention keys are also
masked without producing all-masked query rows. This preserves variable-duration batches without training
on padding or weighting long songs more heavily merely because they contain more frames.

Training-only VAE encoding now samples the diagonal Gaussian posterior using the model's softplus scale
parameter while inference retains deterministic mean extraction. The sampled path has deterministic seed
replay coverage and is available in both direct and tiled VAE encoding.

The native preprocessing pipeline decodes and resamples real audio, samples tiled VAE targets, applies
Gary-compatible caption/genre and custom-tag placement, runs ACE-Step's text and condition encoders, and
constructs silence-plus-mask context tensors. It processes the dataset in model phases so strict eviction
loads the VAE once for all songs, then the text and condition models once each, instead of reloading weights
per example. Per-request duration is preserved when multiple songs are conditioned together.

A backend-resident DoRA-row test independently allocates every tensor on the selected GGML device and
checks its analytical forward loss plus nonzero finite gradients for LoRA A, LoRA B, and magnitude. F32,
Q8_0, Q4_K, Q5_K, and Q6_K frozen-weight graphs pass on CPU, Metal (`MTL0`, Apple M3 Max), and Vulkan
(`Vulkan0`, Apple M3 Max through MoltenVK). These include the tensor types used by Q4_K_M models. Each
quantized case verifies the model transform's forward matrix multiplication and the quantized `OUT_PROD`
backward in one graph. The Vulkan
validation also caught an incomplete rebase resolution: SA3's expanded typed `OUT_PROD` implementation
had been added beside upstream's later basic-F32 implementation. Removing only the stale duplicate
registration and backend path restored one authoritative F32/F16/quantized implementation. The validated
integration heads are GGML
`7ccc6dfc08d7ec5952fa933a20b596c0d4f382d3` and ACE-Step
`5a47b5b7108c106232ca64c79112124dd415ccc8`. Sawhee merged the
complete ACE-Step history at `3fc27a5`. CUDA remains unexecuted until matching
hardware is available.

Sawhee is a public personal-account repository with no registered self-hosted
runner. Standard public GitHub-hosted runners are free but do not expose an
NVIDIA GPU. GitHub's T4 runner is a paid larger runner restricted to Team and
Enterprise organizations. The CI contract therefore performs a bounded CUDA
12.6 compile on a free standard runner. It does not mislabel compilation as
hardware execution.

## Gary compatibility target

Gary currently invokes a Python ACE-Step trainer and expects PEFT-compatible artifacts:

- `adapter_model.safetensors`;
- `adapter_config.json`;
- LoRA or DoRA selection, rank, alpha, and a balanced module profile;
- continuous flow-matching with logit-normal timestep sampling centered at `-0.4`;
- classifier-free condition dropout and optional Min-SNR weighting.

Sawhee should implement that behavior and artifact contract in native code. No Gary files should change
until Sawhee can demonstrate compatible training and inference independently.

The native dataset contract discovers basename-matched `.wav`/`.txt` pairs in stable order and parses
caption, genre, BPM, key, time signature, instrumental state, custom tag, and multiline lyrics. Missing
pairs and incomplete conditioning metadata fail explicitly. The `ace-train` executable now provides model
selection, LoRA or DoRA-row configuration, PEFT weight resume, variable-size batching, gradient
accumulation, warmup plus constant/linear/cosine scheduling, epoch checkpoints, and final PEFT output.
Native checkpoints also persist and restore every AdamW first/second moment tensor, optimizer step, and
completed epoch. A PEFT-only directory remains a valid weight-only resume. The remaining trainer surface
is real-checkpoint end-to-end training/inference validation.

## Real-data acceptance

The supplied [Drive dataset](https://drive.google.com/drive/folders/152WEfkTQuyb78dNWejyvgdoGbJkf-xoP)
contains 13 basename-matched WAV/metadata pairs. Its text files follow the native dataset contract above.
The first acceptance run should use a tiny fixed slice, prove loss reduction and checkpoint reload, then
compare base and adapted generations with fixed prompts and seeds. Lyrics remain input data and are not
copied into this repository.

One supplied pair has also passed the executable dataset journey: the 31.7 MB WAV decoded as 8,313,677
stereo samples at 48 kHz, its sidecar metadata parsed, and the validator measured 2.89 minutes. The full
13-song corpus is intentionally deferred until model-backed training can run from the chosen Sawhee layout.

## Repository layout

Sawhee is a full MIT-licensed ACE-Step fork at the repository root. The merge
preserves the prior Sawhee research, Beads, and OpenSpec history alongside the
complete upstream ACE-Step history. The rebased GGML branch is stored as the
unrelated `ggml/acestep-training` branch in the same public repository, and the
`ggml` submodule pins its exact tested commit. This keeps recursive clones
reproducible without requiring write access to the SA3 GGML fork or creating a
second repository.
