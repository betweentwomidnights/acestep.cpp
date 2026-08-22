# Sawhee project specification

## Purpose

Sawhee provides native ACE-Step LoRA and DoRA-row training and inference on GGML backends.

## Product boundaries

- ACE-Step model loading, inference, training, and adapter checkpoint handling are in scope.
- CPU, CUDA, Vulkan, and Metal are supported targets.
- Compatibility with Gary's current ACE-Step adapter artifacts and training behavior is in scope.
- Editing or packaging `gary-localhost-installer` is not in scope for this change.

## Engineering constraints

- Preserve the existing ACE-Step inference path when no adapter transform is installed.
- Keep frozen base weights in their stored precision during training.
- Apply LoRA and DoRA functionally; do not materialize full effective weights.
- Validate backend behavior with executed tests on matching hardware.
- Use real checkpoints and real dataset examples for integration and end-to-end acceptance.

## Repository decision

The source layout and license remain pending. See the
[training proposal](changes/acestep-dora-training/proposal.md) for the requested decision.
