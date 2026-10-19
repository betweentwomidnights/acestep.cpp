// vae-tile.h: choose a VAE tile size that fits the device's free memory.
//
// A tile's decode or encode graph grows with its length, and a backend that
// cannot allocate it mid-compute aborts the process (CUDA's memory pool
// does). A fixed 1024-frame tile needs more than 7 GiB to decode, so it fails
// on 8 GB GPUs that run every DiT comfortably. Instead of a fixed size, measure
// each candidate's graph without allocating it and take the largest that fits
// what the device has free right now, which also accounts for whatever else
// (a DAW, another model) shares the GPU.

#pragma once
#include "backend.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

// Tile lengths in latent frames, largest first. 1024 was the fixed default;
// 128 is the smallest Python ACE-Step uses. A tile size of 0 means "choose".
static const int VAE_TILE_CANDIDATES[] = { 1024, 512, 256, 128 };

// Kept free beyond a tile's measured needs: the backend's own bookkeeping and
// rounding in its pools, which grow with the tile. On an 8 GB RTX 5070 a tile
// used 0.1-0.5 GiB more than measured (more for larger tiles), so the margin is
// a fixed amount plus a tenth of the measurement.
static const size_t VAE_TILE_HEADROOM = (size_t) 256 * 1024 * 1024;

static size_t vae_tile_with_headroom(size_t measured) {
    return measured + measured / 10 + VAE_TILE_HEADROOM;
}

// Bytes the graph `build(ctx, length)` needs on `backend`: its compute buffer,
// plus the largest temporary a backend makes while computing it, the
// half-precision copy of a matrix multiply's F32 input. CUDA's memory pool and
// Vulkan's staging buffers hold that copy outside the graph's allocation.
template <typename Build>
static size_t vae_tile_bytes(ggml_backend_sched_t sched, ggml_backend_t backend, Build build, int length) {
    const size_t            ctx_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t>    ctx_buf(ctx_size);
    struct ggml_init_params params = { ctx_size, ctx_buf.data(), true };
    struct ggml_context *   ctx    = ggml_init(params);
    struct ggml_tensor *    output = build(ctx, length);
    struct ggml_cgraph *    graph  = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, output);

    std::vector<size_t> sizes(ggml_backend_sched_get_n_backends(sched), 0);
    ggml_backend_sched_reserve_size(sched, graph, sizes.data());
    size_t compute = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); i++) {
        if (ggml_backend_sched_get_backend(sched, i) == backend) {
            compute = sizes[i];
        }
    }
    size_t temporary = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
        const struct ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT && node->src[1]->type == GGML_TYPE_F32 && node->src[0]->type != GGML_TYPE_F32) {
            temporary = std::max(temporary, (size_t) ggml_nelements(node->src[1]) * sizeof(ggml_fp16_t));
        }
    }
    ggml_free(ctx);
    return compute + temporary;
}

// The largest candidate tile whose graph fits on `backend` now. `full_length`
// is the whole input in latent frames: a tile never needs to be longer.
// `to_graph_length` converts a tile in latent frames to the graph's input
// length (audio samples for the encoder). Candidates are measured on a
// scratch scheduler over the same devices: measuring replans a scheduler's
// allocations, and the caller's `sched` must keep the plan for its graph.
template <typename Build, typename ToGraphLength>
static int vae_tile_choose(ggml_backend_sched_t sched,
                           ggml_backend_t       backend,
                           ggml_backend_t       cpu_backend,
                           int                  full_length,
                           Build                build,
                           ToGraphLength        to_graph_length,
                           const char *         tag) {
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_bytes, &total_bytes);
    // The scheduler's compute buffer is ours to reuse, so it counts as free.
    const size_t available = free_bytes + ggml_backend_sched_get_buffer_size(sched, backend);

    const BackendPair    pair     = { backend, cpu_backend, backend != cpu_backend };
    ggml_backend_sched_t scratch  = backend_sched_new(pair, 8192);
    const int            smallest = VAE_TILE_CANDIDATES[sizeof(VAE_TILE_CANDIDATES) / sizeof(int) - 1];
    for (int tile : VAE_TILE_CANDIDATES) {
        const int    length = std::min(tile, full_length);
        const size_t needed = vae_tile_with_headroom(vae_tile_bytes(scratch, backend, build, to_graph_length(length)));
        if (needed <= available || tile == smallest) {
            ggml_backend_sched_free(scratch);
            fprintf(stderr, "[%s] Auto tile: %d frames (needs %zu MiB, %zu MiB available)%s\n", tag, tile, needed >> 20,
                    available >> 20, needed <= available ? "" : " -- may not fit");
            return tile;
        }
    }
    ggml_backend_sched_free(scratch);
    return smallest;
}
