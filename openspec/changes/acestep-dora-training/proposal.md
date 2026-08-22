# Proposal: native ACE-Step LoRA and DoRA-row training

## Motivation

ACE-Step has native GGML inference but no native adapter trainer. SA3 has the required optimizer,
checkpoint, functional-adapter, and multi-backend GGML work on an older GGML base. Sawhee will combine
those capabilities while targeting Gary-compatible adapter outputs.

## Proposed change

1. Base Sawhee on ACE-Step's current source and GGML revision.
2. Carry the focused SA3 GGML training patches on top of that revision.
3. Add an optional functional linear transform to ACE-Step's existing DiT graph.
4. Implement LoRA and DoRA-row parameter initialization, forward/backward graphs, optimizer state,
   checkpoint resume, and adapter export.
5. Implement dataset preparation and ACE-Step flow-matching loss consistent with Gary's current trainer.
6. Load exported adapters through native inference and preserve unadapted inference behavior.
7. Validate CPU, CUDA, Vulkan, and Metal, then perform a fixed-slice overfit and ear test.

## Compatibility contract

The trainer emits `adapter_model.safetensors` and `adapter_config.json` with
sufficient metadata for native reload and PEFT-compatible consumers. The
`ace-train` CLI covers adapter type, rank, alpha, module profile, seed,
optimizer settings, checkpoint resume, and output directory.

## Acceptance gates

- Unit: adapter initialization, DoRA norm algebra, target selection, config parsing, and checkpoint schema.
- Integration: real GGML forward/backward graph, numerical gradients, optimizer step, save/reload parity.
- Backend: executed training graph on CPU, CUDA, Vulkan, and Metal, including quantized frozen weights.
- End to end: a tiny real dataset slice overfits, exports, reloads, and changes fixed-seed inference.
- Regression: the full ACE-Step build and existing inference checks remain green without adapters.

## Approved source layout

Sawhee is a full ACE-Step fork at the repository root under ACE-Step's MIT
license. The repository retains both the original Sawhee project history and
the upstream ACE-Step history. The exact rebased GGML commit is pinned as a
submodule and published on the repository's `ggml/acestep-training` branch.
