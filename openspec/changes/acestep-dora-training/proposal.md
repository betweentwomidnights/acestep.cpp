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

The trainer will emit `adapter_model.safetensors` and `adapter_config.json` with sufficient metadata for
native reload and PEFT-compatible consumers. CLI details will be specified after the source layout is
approved, but must cover adapter type, rank, alpha, module profile, seed, optimizer settings, checkpoint
interval, resume, and output directory.

## Acceptance gates

- Unit: adapter initialization, DoRA norm algebra, target selection, config parsing, and checkpoint schema.
- Integration: real GGML forward/backward graph, numerical gradients, optimizer step, save/reload parity.
- Backend: executed training graph on CPU, CUDA, Vulkan, and Metal, including quantized frozen weights.
- End to end: a tiny real dataset slice overfits, exports, reloads, and changes fixed-seed inference.
- Regression: the full ACE-Step build and existing inference checks remain green without adapters.

## Decision required before implementation lands

Approve one source layout and its license handling:

- **Recommended:** full ACE-Step fork at the Sawhee root, replacing the template AGPL license with MIT and
  retaining upstream attribution/history.
- AGPL meta-repository with ACE-Step and GGML as submodules plus maintained patch series.

Approval of this proposal authorizes the OpenSpec delta and task list, followed by the TDD implementation.
