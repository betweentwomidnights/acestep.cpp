acestep.cpp native runtime

Download GGUF model files separately; none are bundled here.
See RUNTIME_README.md for model types, training, backend selection and limits.

Windows: ace-server.exe --models C:\path\to\models --adapters C:\path\to\adapters
macOS:   ./ace-server --models /path/to/models --adapters /path/to/adapters
Open http://127.0.0.1:8080 . Use --device cuda, vulkan, metal or cpu to select
a backend. An unavailable explicit device fails instead of falling back.

The Windows standalone archive includes NVIDIA's CUDA runtime and both GPU
backends. Applications should install core + one backend, and supply their
shared CUDA 12.8 runtime on PATH instead. Each service keeps its own ggml DLLs.

For native LoRA training on an 8 GB GPU, select the XL-base Q4_K_M GGUF
explicitly with ace-train --dit. Use --adapter-type lora for LoRA; the trainer's
historical default is dora-rows. Start with batch size 1, checkpointed backward
and bounded training windows. Training memory depends on rank and duration;
Q4 alone does not guarantee that an arbitrary dataset or configuration fits.

The server defaults to functional adapter inference, preserving quantized base
weights. Training runs through ace-train; there is no training HTTP/WebUI API.
This runtime is not yet a drop-in replacement for the gary4local Carey API.
