// ABOUTME: Merges LoRA, DoRA, and LoKr adapters into GGUF weights on the selected backend.
// ABOUTME: Loads PEFT and LyCORIS metadata before pending weights are uploaded or fused.

#pragma once
// adapter-merge.h: runtime adapter merge into GGUF weights before QKV fusion.
//
// Supported algorithms:
//   LoRA (low rank adaptation):  delta = (alpha / rank) * B @ A
//   LoKr (low rank Kronecker):   delta = (alpha / rank) * kron(w1, w2_a @ w2_b)
//                                optional DoRA magnitude rescale per output row
//                                when a dora_scale tensor is present.
//
// Called after individual GGUF projection tensors are loaded into WeightCtx
// but BEFORE wctx_alloc uploads to GPU and BEFORE QKV fusion concatenation.
//
// Each projection (q_proj, k_proj, v_proj, o_proj) has its own PendingCopy
// even when destined for a fused QKV tensor. We patch each one separately,
// so fusion proceeds normally on already merged data.
//
// Performance: one backend graph per tensor runs the full pipeline. The base
// weight is uploaded in its native GGUF type directly from the mmap, the
// backend dequantizes it to F32 via ggml_cast, the adapter delta is built
// from adapter factors at their serialized precision, added to base, DoRA
// rescaled when requested, then cast back to the native GGUF type when the
// backend supports that encode direction. LyCORIS deltas retain their BF16
// weight-decomposition rule. The result is downloaded into PendingCopy staging.
//
// ACE-Step GGUFs ship in BF16, Q8_0, Q4_K_M, Q5_K_M, and Q6_K. On CUDA the
// backend encode cast handles F32 -> BF16 and F32 -> Q8_0 directly; the
// K-quants have no F32 -> native kernel so the graph terminates at F32 and
// ggml_quantize_chunk completes the job on host. Either way the base upload
// PCIe is cut vs a prior F32 upload, and the dequant, add, optional BF16 round and
// DoRA rescale all run on the backend.
// PendingCopy lookup is O(1) via hashmap.

#include "adapter-checkpoint-directory.h"
#include "adapter-config.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "safetensors.h"
#include "weight-ctx.h"
#include "yyjson.h"

#include <sys/stat.h>
#ifdef _WIN32
#    ifndef S_ISDIR
#        define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#    endif
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Convert safetensors tensor data to F32 based on dtype string.
// Handles "F32", "BF16", "F16". Returns false for unknown dtypes.
static bool adapter_to_f32(const void * src, float * dst, int64_t n, const std::string & dtype) {
    if (dtype == "F32") {
        memcpy(dst, src, (size_t) n * sizeof(float));
    } else if (dtype == "BF16") {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
    } else if (dtype == "F16") {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
    } else {
        return false;
    }
    return true;
}

// Map a LoRA safetensors key to the GGUF base tensor name.
//
// Supported key formats (all map to GGUF "decoder.layers.0.self_attn.q_proj.weight"):
//
//   PEFT adapter_model.safetensors:
//     base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
//     base_model.model.layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (no prefix):
//     layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (diffusion_model prefix):
//     diffusion_model.layers.0.self_attn.q_proj.lora_A.weight
//
// Steps: strip known prefix, extract module path before ".lora_",
// prepend "decoder." if needed, append ".weight".
static std::string lora_base_name(const std::string & key) {
    std::string s = key;

    // strip known prefixes (PEFT, ComfyUI)
    static const char * prefixes[] = {
        "base_model.model.",  // PEFT
        "diffusion_model.",   // ComfyUI official ACE-Step format
    };
    for (const char * pfx : prefixes) {
        size_t pfx_len = strlen(pfx);
        if (s.compare(0, pfx_len, pfx) == 0) {
            s = s.substr(pfx_len);
            break;
        }
    }

    // everything before ".lora_" is the module path
    size_t pos = s.find(".lora_");
    if (pos == std::string::npos) {
        return "";
    }
    s = s.substr(0, pos);

    // ensure decoder prefix (PEFT wraps the decoder directly,
    // so the internal path starts at "layers." not "decoder.layers.")
    if (s.compare(0, 8, "decoder.") != 0) {
        s = "decoder." + s;
    }

    return s + ".weight";
}

// Check whether a safetensors key is a lora_A/down or lora_B/up weight.
// PEFT uses .lora_A. / .lora_B., ComfyUI single-file uses .lora_down. / .lora_up.
static bool lora_is_a(const std::string & key) {
    return key.find(".lora_A.") != std::string::npos || key.find(".lora_down.") != std::string::npos;
}

static bool lora_is_b(const std::string & key) {
    return key.find(".lora_B.") != std::string::npos || key.find(".lora_up.") != std::string::npos;
}

static bool lora_is_magnitude(const std::string & key) {
    const std::string marker = ".lora_magnitude_vector";
    const size_t      position = key.find(marker);
    if (position == std::string::npos) {
        return false;
    }
    const size_t end = position + marker.size();
    return end == key.size() || key[end] == '.';
}

static std::string adapter_module_for_weight(const std::string & weight_name) {
    std::string module = weight_name;
    if (module.compare(0, 8, "decoder.") == 0) {
        module.erase(0, 8);
    }
    if (module.size() >= 7 && module.compare(module.size() - 7, 7, ".weight") == 0) {
        module.resize(module.size() - 7);
    }
    return module;
}

static bool adapter_config_targets_weight(const adapter_config & config, const std::string & weight_name) {
    const std::string module = adapter_module_for_weight(weight_name);
    for (const std::string & target : config.target_modules) {
        if (adapter_module_matches_pattern(module, target)) {
            return true;
        }
    }
    return false;
}

// Read linear_dim from LyCORIS __metadata__.lokr_config for LoKr payloads.
// LyCORIS stores lokr_config as a serialized JSON string inside the safetensors
// header __metadata__ object. This dim is the LoRA rank used for the w2
// factorization, and is the only place it lives when w2 is stored as a single
// monolithic tensor (LyCORIS use_w2 path, factor=-1 with linear_dim bigger
// than the factorized budget). Returns 0 when absent or unparseable.
static int adapter_read_lokr_dim(const STFile & st) {
    if (st.data_offset <= 8) {
        return 0;
    }
    size_t       hdr_len = st.data_offset - 8;
    const char * hdr     = (const char *) st.mapping + 8;

    yyjson_doc * doc = yyjson_read(hdr, hdr_len, 0);
    if (!doc) {
        return 0;
    }
    int          dim  = 0;
    yyjson_val * root = yyjson_doc_get_root(doc);
    yyjson_val * meta = yyjson_obj_get(root, "__metadata__");
    if (meta && yyjson_is_obj(meta)) {
        yyjson_val * cfg = yyjson_obj_get(meta, "lokr_config");
        if (cfg && yyjson_is_str(cfg)) {
            yyjson_doc * sub = yyjson_read(yyjson_get_str(cfg), yyjson_get_len(cfg), 0);
            if (sub) {
                yyjson_val * ld = yyjson_obj_get(yyjson_doc_get_root(sub), "linear_dim");
                if (ld && yyjson_is_int(ld)) {
                    dim = (int) yyjson_get_int(ld);
                }
                yyjson_doc_free(sub);
            }
        }
    }
    yyjson_doc_free(doc);
    return dim;
}

// Requant F32 data back to original type. Writes into dst buffer.
// Returns the number of bytes written. Used only as host fallback when the
// backend lacks an F32 -> native cast kernel (K-quants on CUDA today).
static size_t adapter_requant(const float * src, void * dst, int64_t nel, int64_t n_per_row, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        size_t nb = (size_t) nel * sizeof(float);
        memcpy(dst, src, nb);
        return nb;
    }

    const struct ggml_type_traits * traits = ggml_get_type_traits(type);

    if (traits->is_quantized) {
        // quantized types: use ggml_quantize_chunk (handles block alignment)
        int64_t nrows = nel / n_per_row;
        size_t  qsize = ggml_row_size(type, n_per_row) * (size_t) nrows;
        ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, NULL);
        return qsize;
    }

    // non quantized (BF16, F16): use from_float_ref
    if (traits->from_float_ref) {
        size_t nb = (size_t) nel * traits->type_size;
        traits->from_float_ref(src, dst, nel);
        return nb;
    }

    fprintf(stderr, "[Adapter] WARNING: no requant for type %d\n", type);
    return 0;
}

// True when the backend has an F32 -> native encode cast kernel for this type.
// Probed via ggml_backend_supports_op on a throwaway GGML_OP_CPY tensor: the
// backend only inspects src[0]->type and src[1]->type so no buffer allocation
// or data upload is needed. When false, the caller downloads F32 and calls
// adapter_requant on host.
static bool adapter_backend_can_encode(ggml_backend_t backend, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        return true;
    }
    size_t                  meta   = ggml_tensor_overhead() * 4 + 1024;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    struct ggml_tensor *    src    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    struct ggml_tensor *    dst    = ggml_cast(ctx, src, type);
    bool                    ok     = ggml_backend_supports_op(backend, dst);
    ggml_free(ctx);
    return ok;
}

// Build the reverse map from LyCORIS key prefix to GGUF tensor name.
// LyCORIS stores adapter tensors as "lycoris_<path_with_underscores>.<suffix>",
// where the torch module path has all dots flattened to underscores. We cannot
// safely reverse that transform blindly (names like "cross_attn" contain real
// underscores), so we enumerate the GGUF decoder .weight tensors we already
// have and build the mapping from them.
//
// Example:
//   GGUF tensor "decoder.layers.0.self_attn.q_proj.weight"
//   -> "lycoris_layers_0_self_attn_q_proj" -> "decoder.layers.0.self_attn.q_proj.weight"
static std::unordered_map<std::string, std::string> lokr_build_reverse_map(const GGUFModel & gf) {
    std::unordered_map<std::string, std::string> out;
    int                                          n_tensors = (int) gguf_get_n_tensors(gf.gguf);
    static const char *                          suffix    = ".weight";
    size_t                                       slen      = strlen(suffix);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gf.gguf, i);
        std::string  s    = name;
        // only decoder.*.weight tensors qualify as LoKr targets
        if (s.size() <= slen || s.compare(s.size() - slen, slen, suffix) != 0) {
            continue;
        }
        if (s.compare(0, 8, "decoder.") != 0) {
            continue;
        }
        // strip "decoder." prefix and ".weight" suffix, flatten dots to underscores
        std::string path = s.substr(8, s.size() - 8 - slen);
        for (char & c : path) {
            if (c == '.') {
                c = '_';
            }
        }
        out["lycoris_" + path] = s;
    }
    return out;
}

// Split a safetensors key on its last dot. Returns false when no dot exists.
// Example: "lycoris_layers_0_self_attn_q_proj.lokr_w1"
//   -> prefix "lycoris_layers_0_self_attn_q_proj", suffix "lokr_w1"
static bool adapter_split_suffix(const std::string & key, std::string * prefix, std::string * suffix) {
    size_t dot = key.rfind('.');
    if (dot == std::string::npos) {
        return false;
    }
    *prefix = key.substr(0, dot);
    *suffix = key.substr(dot + 1);
    return true;
}

// Detect whether the safetensors payload is a LyCORIS LoKr pack.
// Any tensor named "*.lokr_w1" or "*.lokr_w2*" is a LoKr marker that LoRA
// payloads never produce, so a single match is sufficient.
static bool adapter_detect_lokr(const STFile & st) {
    for (const auto & e : st.entries) {
        if (e.name.find(".lokr_w1") != std::string::npos || e.name.find(".lokr_w2") != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Merge one base tensor with an adapter delta entirely inside a backend graph.
//
// Responsibilities split between caller and helper:
//   caller: provide the delta builder lambda that, given the graph ctx,
//           returns a tdelta tensor of shape ggml ne=(in, out) in F32 row
//           major (out_feat, in_feat). The lambda is also responsible for
//           uploading its own adapter factors to the backend inside the
//           helper's alloc_ctx_tensors sweep (see pattern below).
//   helper: upload base in native type from mmap, dequant to F32 on backend,
//           apply payload-specific delta semantics, add base + delta, encode
//           back to native when supported, and download into staging.
//
// Upload pattern for caller-owned tensors: the lambda allocates tensors in
// the shared ctx. The helper calls alloc_ctx_tensors once after the lambda
// runs, then the lambda's returned populator function is invoked to do the
// actual ggml_backend_tensor_set calls.
//
// build_delta signature expanded: pair<tdelta, populator> so the caller can
// defer its uploads until after the single alloc sweep. This keeps one graph,
// one buffer alloc, one compute, one download per merge.
struct adapter_delta_build {
    struct ggml_tensor *  tdelta;
    std::function<void()> upload;
};

enum class adapter_merge_semantics {
    lora,
    peft_dora,
    lokr,
    lycoris_dora,
};

// Execute the unified merge graph for one tensor. Returns true on success.
// See adapter_delta_build for the caller contract.
static bool adapter_merge_on_backend(WeightCtx *                                                       wctx,
                                     std::unordered_map<const void *, size_t> &                        pending_idx,
                                     const void *                                                      base_ptr,
                                     enum ggml_type                                                    ttype,
                                     int64_t                                                           ne0,
                                     int64_t                                                           ne1,
                                     const float *                                                     ds,
                                     float                                                             user_scale,
                                     adapter_merge_semantics                                           merge_semantics,
                                     ggml_backend_t                                                    backend,
                                     const char *                                                      gguf_name,
                                     const std::function<adapter_delta_build(struct ggml_context *)> & build_delta) {
    // locate the pending copy upfront: no point computing if we can't apply
    auto pc_it = pending_idx.find(base_ptr);
    if (pc_it == pending_idx.end()) {
        fprintf(stderr, "[Adapter] FATAL: no PendingCopy for %s\n", gguf_name);
        return false;
    }
    WeightCtx::PendingCopy * pc = &wctx->pending[pc_it->second];

    int64_t nel       = ne0 * ne1;
    size_t  base_nb   = ggml_row_size(ttype, ne0) * (size_t) ne1;
    bool    encode_ok = adapter_backend_can_encode(backend, ttype);

    // slack for the largest graph (DoRA + optional BF16 round + cast in/out + caller subgraph)
    size_t                  meta   = ggml_tensor_overhead() * 64 + ggml_graph_overhead() + 32 * 1024;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);
    if (!ctx) {
        return false;
    }

    // base uploaded in native type, dequant to F32 on backend via ggml_cast
    struct ggml_tensor * tbase_native = ggml_new_tensor_2d(ctx, ttype, ne0, ne1);
    struct ggml_tensor * tbase_f32    = ggml_cast(ctx, tbase_native, GGML_TYPE_F32);

    // DoRA scale vector, one F32 per output row when dora_scale is set
    struct ggml_tensor * tds = ds ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, ne1) : NULL;

    // caller builds the delta subgraph and returns its upload closure
    adapter_delta_build db = build_delta(ctx);

    const bool lokr_semantics = merge_semantics == adapter_merge_semantics::lokr ||
                                merge_semantics == adapter_merge_semantics::lycoris_dora;
    struct ggml_tensor * tdelta_f = lokr_semantics
                                        ? ggml_cast(ctx, ggml_cast(ctx, db.tdelta, GGML_TYPE_BF16), GGML_TYPE_F32)
                                        : db.tdelta;

    // merge: plain scaled add, PEFT DoRA interpolation, or LyCORIS weight decomposition
    struct ggml_tensor * tmerged;
    if (!tds) {
        struct ggml_tensor * td_u = (user_scale != 1.0f) ? ggml_scale(ctx, tdelta_f, user_scale) : tdelta_f;
        tmerged                   = ggml_add(ctx, tbase_f32, td_u);
    } else {
        struct ggml_tensor * tm_pre  = ggml_add(ctx, tbase_f32, tdelta_f);
        struct ggml_tensor * tsq     = ggml_sqr(ctx, tm_pre);
        struct ggml_tensor * tss     = ggml_sum_rows(ctx, tsq);  // ne=(1, out)
        struct ggml_tensor * trn     = ggml_sqrt(ctx, tss);
        if (merge_semantics == adapter_merge_semantics::peft_dora) {
            struct ggml_tensor * tscale = ggml_div(ctx, tds, trn);
            struct ggml_tensor * tfull  = ggml_mul(ctx, tm_pre, tscale);
            if (user_scale == 1.0f) {
                tmerged = tfull;
            } else {
                struct ggml_tensor * tdifference = ggml_sub(ctx, tfull, tbase_f32);
                tmerged = ggml_add(ctx, tbase_f32, ggml_scale(ctx, tdifference, user_scale));
            }
        } else {
            const float          eps      = 7.8125e-3f;
            struct ggml_tensor * trn_eps  = ggml_scale_bias(ctx, trn, 1.0f, eps);
            struct ggml_tensor * tscale   = ggml_div(ctx, tds, trn_eps);
            struct ggml_tensor * tscale_m =
                (user_scale != 1.0f) ? ggml_scale_bias(ctx, tscale, user_scale, 1.0f - user_scale) : tscale;
            tmerged = ggml_mul(ctx, tm_pre, tscale_m);
        }
    }

    // output: native type when the backend can encode, else F32 for host requant
    struct ggml_tensor * tout = encode_ok ? ggml_cast(ctx, tmerged, ttype) : tmerged;

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, tout);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }

    // helper-owned uploads: base in native type from mmap, ds if DoRA
    ggml_backend_tensor_set(tbase_native, base_ptr, 0, base_nb);
    if (tds) {
        ggml_backend_tensor_set(tds, ds, 0, (size_t) ne1 * sizeof(float));
    }
    // caller-owned uploads for adapter factors
    db.upload();

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[Adapter] WARNING: backend merge graph failed for %s\n", gguf_name);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // allocate a staging slot sized for the native encoded weight, then download
    size_t n_floats    = (base_nb + sizeof(float) - 1) / sizeof(float);
    auto   staging_buf = std::make_unique<float[]>(n_floats);
    void * merged_buf  = staging_buf.get();

    if (encode_ok) {
        // download straight into staging in native type, zero host postprocess
        ggml_backend_tensor_get(tout, merged_buf, 0, base_nb);
        pc->src    = merged_buf;
        pc->nbytes = base_nb;
    } else {
        // download F32 then requant on host (K-quants on CUDA: Q4_K_M, Q5_K_M, Q6_K)
        std::vector<float> merged_f32((size_t) nel);
        ggml_backend_tensor_get(tout, merged_f32.data(), 0, (size_t) nel * sizeof(float));
        size_t merged_bytes = adapter_requant(merged_f32.data(), merged_buf, nel, ne0, ttype);
        if (merged_bytes == 0) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }
        pc->src    = merged_buf;
        pc->nbytes = merged_bytes;
    }
    wctx->staging.push_back(std::move(staging_buf));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// LoRA merge path. Matches PEFT merge_and_unload for PEFT payloads and ComfyUI
// merge semantics for single file LoRA:
//   delta = (alpha / rank) * scale * B @ A
// Applied to base weights in place. Alpha is read per tensor if present
// (ComfyUI baked), else from adapter_config.json, else defaults to rank.
static bool adapter_merge_lora(WeightCtx *         wctx,
                               const GGUFModel &   gf,
                               const STFile &      st,
                               const std::string & cfg_dir,
                               bool                peft_directory,
                               float               scale,
                               ggml_backend_t      backend) {
    adapter_config config;
    const bool config_loaded = adapter_read_config(cfg_dir.c_str(), config);
    if (peft_directory && !config_loaded) {
        fprintf(stderr, "[Adapter] PEFT adapter_config.json is missing or invalid\n");
        return false;
    }

    // group lora_A and lora_B entries by their GGUF base tensor name.
    // also collect per tensor alpha scalars (ComfyUI baked format).
    std::map<std::string, const STEntry *> a_map, b_map, magnitude_map;
    std::map<std::string, float>           alpha_map;
    bool                                   duplicate_entry = false;
    for (const auto & e : st.entries) {
        // per tensor alpha: "base_model.model.layers.0.self_attn.q_proj.alpha"
        // scalar F32 with shape [] containing the baked alpha value
        const char * alpha_suffix = ".alpha";
        size_t       slen         = strlen(alpha_suffix);
        if (e.name.size() > slen && e.name.compare(e.name.size() - slen, slen, alpha_suffix) == 0 && e.dtype == "F32" &&
            e.n_dims == 0) {
            std::string fake_key = e.name.substr(0, e.name.size() - slen) + ".lora_.x";
            std::string base     = lora_base_name(fake_key);
            if (!base.empty()) {
                float val = 0.0f;
                memcpy(&val, st_data(st, e), sizeof(float));
                duplicate_entry = !alpha_map.emplace(base, val).second || duplicate_entry;
            }
            continue;
        }

        std::string base = lora_base_name(e.name);
        if (base.empty()) {
            continue;
        }
        if (lora_is_magnitude(e.name)) {
            duplicate_entry = !magnitude_map.emplace(base, &e).second || duplicate_entry;
        } else if (lora_is_a(e.name)) {
            duplicate_entry = !a_map.emplace(base, &e).second || duplicate_entry;
        } else if (lora_is_b(e.name)) {
            duplicate_entry = !b_map.emplace(base, &e).second || duplicate_entry;
        }
    }

    if (duplicate_entry) {
        fprintf(stderr, "[Adapter] adapter contains duplicate tensor aliases\n");
        return false;
    }

    std::unordered_map<const void *, size_t> pending_idx;
    pending_idx.reserve(wctx->pending.size());
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        pending_idx[wctx->pending[i].src] = i;
    }

    if (peft_directory) {
        if (a_map.empty() || a_map.size() != b_map.size() ||
            (config.use_dora ? magnitude_map.size() != a_map.size() : !magnitude_map.empty()) ||
            config.target_modules.empty()) {
            fprintf(stderr, "[Adapter] PEFT adapter tensor inventory does not match adapter_config.json\n");
            return false;
        }
        for (const auto & entry : b_map) {
            if (a_map.count(entry.first) == 0) {
                fprintf(stderr, "[Adapter] PEFT adapter contains lora_B without lora_A for %s\n", entry.first.c_str());
                return false;
            }
        }
        for (const auto & entry : magnitude_map) {
            if (a_map.count(entry.first) == 0) {
                fprintf(stderr, "[Adapter] PEFT adapter contains magnitude without factors for %s\n", entry.first.c_str());
                return false;
            }
        }
        for (const auto & entry : a_map) {
            const std::string & gguf_name = entry.first;
            const STEntry *     a = entry.second;
            auto                b = b_map.find(gguf_name);
            auto                magnitude = magnitude_map.find(gguf_name);
            if (b == b_map.end() || (config.use_dora && magnitude == magnitude_map.end()) || a->n_dims != 2 ||
                b->second->n_dims != 2 || a->shape[0] <= 0 || a->shape[1] <= 0 || b->second->shape[0] <= 0 ||
                b->second->shape[1] != a->shape[0] ||
                a->shape[0] != adapter_config_value_for_weight(config.rank_pattern, gguf_name, config.rank) ||
                (config.use_dora && (magnitude->second->n_dims != 1 || magnitude->second->shape[0] != b->second->shape[0]))) {
                fprintf(stderr, "[Adapter] PEFT adapter tensor shapes are invalid for %s\n", gguf_name.c_str());
                return false;
            }
            int64_t tensor_index = gguf_find_tensor(gf.gguf, gguf_name.c_str());
            struct ggml_tensor * tensor = tensor_index >= 0 ? ggml_get_tensor(gf.meta, gguf_name.c_str()) : nullptr;
            if (!adapter_config_targets_weight(config, gguf_name) || !tensor || tensor->ne[0] != a->shape[1] ||
                tensor->ne[1] != b->second->shape[0]) {
                fprintf(stderr, "[Adapter] PEFT adapter target does not match GGUF tensor %s\n", gguf_name.c_str());
                return false;
            }
            const size_t offset = gguf_get_tensor_offset(gf.gguf, tensor_index);
            const void * base = gf.mapping + gf.data_offset + offset;
            if (pending_idx.count(base) == 0) {
                fprintf(stderr, "[Adapter] PEFT adapter target is not staged for merge: %s\n", gguf_name.c_str());
                return false;
            }
        }
        std::set<std::string> serialized_targets;
        for (const auto & entry : a_map) {
            serialized_targets.insert(entry.first);
        }
        std::set<std::string> staged_targets;
        const int64_t         tensor_count = gguf_get_n_tensors(gf.gguf);
        for (int64_t tensor_index = 0; tensor_index < tensor_count; ++tensor_index) {
            const char * name = gguf_get_tensor_name(gf.gguf, tensor_index);
            if (!name || !adapter_config_targets_weight(config, name)) {
                continue;
            }
            const size_t offset = gguf_get_tensor_offset(gf.gguf, tensor_index);
            const void * base = gf.mapping + gf.data_offset + offset;
            if (pending_idx.count(base) != 0) {
                staged_targets.insert(name);
            }
        }
        if (serialized_targets != staged_targets) {
            fprintf(stderr, "[Adapter] PEFT adapter module-instance inventory does not match staged GGUF targets\n");
            return false;
        }
        for (const std::string & target : config.target_modules) {
            bool target_present = false;
            for (const auto & entry : a_map) {
                target_present = target_present ||
                                 adapter_module_matches_pattern(adapter_module_for_weight(entry.first), target);
            }
            if (!target_present) {
                fprintf(stderr, "[Adapter] PEFT target module has no serialized tensors: %s\n", target.c_str());
                return false;
            }
        }
    }

    int merged     = 0;
    int skipped    = 0;
    int dora_count = 0;

    for (const auto & kv : a_map) {
        const std::string & gguf_name = kv.first;
        const STEntry *     ea        = kv.second;

        auto it = b_map.find(gguf_name);
        if (it == b_map.end()) {
            fprintf(stderr, "[Adapter] WARNING: no lora_B for %s, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        const STEntry * eb = it->second;

        int64_t tidx = gguf_find_tensor(gf.gguf, gguf_name.c_str());
        if (tidx < 0) {
            fprintf(stderr, "[Adapter] WARNING: tensor %s not in GGUF, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        struct ggml_tensor * tmeta = ggml_get_tensor(gf.meta, gguf_name.c_str());
        enum ggml_type       ttype = tmeta->type;
        int64_t              ne0   = tmeta->ne[0];
        int64_t              ne1   = tmeta->ne[1];

        size_t       toff     = gguf_get_tensor_offset(gf.gguf, tidx);
        const void * base_ptr = gf.mapping + gf.data_offset + toff;

        // LoRA shapes (safetensors PyTorch convention, row major):
        //   A: [rank, in_features]
        //   B: [out_features, rank]
        int64_t rank     = ea->shape[0];
        int64_t in_feat  = ea->shape[1];
        int64_t out_feat = eb->shape[0];

        if (eb->shape[1] != rank) {
            fprintf(stderr, "[Adapter] WARNING: rank mismatch A=%lld vs B=%lld for %s\n", (long long) rank,
                    (long long) eb->shape[1], gguf_name.c_str());
            skipped++;
            continue;
        }
        if (in_feat != ne0 || out_feat != ne1) {
            fprintf(stderr, "[Adapter] WARNING: shape mismatch for %s: LoRA [%lld,%lld] vs GGUF [%lld,%lld]\n",
                    gguf_name.c_str(), (long long) out_feat, (long long) in_feat, (long long) ne1, (long long) ne0);
            skipped++;
            continue;
        }

        // alpha: prefer per tensor (ComfyUI baked), then module config, then global config, fallback to rank
        float alpha;
        auto  alpha_it = alpha_map.find(gguf_name);
        if (alpha_it != alpha_map.end()) {
            alpha = alpha_it->second;
        } else {
            alpha = (float) adapter_config_value_for_weight(config.alpha_pattern, gguf_name,
                                                            config.lora_alpha > 0 ? config.lora_alpha : (int) rank);
        }
        float scaling = alpha / (float) rank;

        // load A and B to F32 while preserving the values stored by the adapter
        int64_t            a_nel = rank * in_feat;
        int64_t            b_nel = out_feat * rank;
        std::vector<float> a_f32((size_t) a_nel);
        std::vector<float> b_f32((size_t) b_nel);
        if (!adapter_to_f32(st_data(st, *ea), a_f32.data(), a_nel, ea->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported dtype %s for lora_A\n", ea->dtype.c_str());
            skipped++;
            continue;
        }

        std::vector<float> magnitude;
        auto               magnitude_it = magnitude_map.find(gguf_name);
        if (magnitude_it != magnitude_map.end()) {
            const STEntry * em            = magnitude_it->second;
            int64_t         magnitude_nel = 1;
            for (int dimension = 0; dimension < em->n_dims; ++dimension) {
                magnitude_nel *= em->shape[dimension];
            }
            if (magnitude_nel != out_feat) {
                fprintf(stderr,
                        "[Adapter] WARNING: DoRA magnitude shape mismatch for %s: got %lld values, expected %lld\n",
                        gguf_name.c_str(),
                        (long long) magnitude_nel,
                        (long long) out_feat);
                skipped++;
                continue;
            }
            magnitude.resize((size_t) magnitude_nel);
            if (!adapter_to_f32(st_data(st, *em), magnitude.data(), magnitude_nel, em->dtype)) {
                fprintf(stderr, "[Adapter] WARNING: unsupported dtype %s for DoRA magnitude\n", em->dtype.c_str());
                skipped++;
                continue;
            }
        }
        if (!adapter_to_f32(st_data(st, *eb), b_f32.data(), b_nel, eb->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported dtype %s for lora_B\n", eb->dtype.c_str());
            skipped++;
            continue;
        }

        // delta = scaling * B @ A, built inside the unified merge graph.
        // A row major (rank, in_feat) stored as ggml ne=(in_feat, rank), transposed
        // so rank sits on ne[0] for mul_mat contraction. Result ne=(in_feat, out_feat).
        auto build = [&](struct ggml_context * ctx) {
            struct ggml_tensor * ta     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, rank);
            struct ggml_tensor * tb     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out_feat);
            struct ggml_tensor * ta_t   = ggml_cont(ctx, ggml_transpose(ctx, ta));
            struct ggml_tensor * tdelta = ggml_scale(ctx, ggml_mul_mat(ctx, ta_t, tb), scaling);

            adapter_delta_build db;
            db.tdelta = tdelta;
            db.upload = [=]() {
                ggml_backend_tensor_set(ta, a_f32.data(), 0, (size_t) a_nel * sizeof(float));
                ggml_backend_tensor_set(tb, b_f32.data(), 0, (size_t) b_nel * sizeof(float));
            };
            return db;
        };

        const float * magnitude_data = magnitude.empty() ? nullptr : magnitude.data();
        const adapter_merge_semantics merge_semantics =
            magnitude_data ? adapter_merge_semantics::peft_dora : adapter_merge_semantics::lora;
        if (!adapter_merge_on_backend(wctx, pending_idx, base_ptr, ttype, ne0, ne1, magnitude_data, scale,
                                      merge_semantics, backend, gguf_name.c_str(), build)) {
            return false;
        }
        if (magnitude_data) {
            dora_count++;
        }
        merged++;
    }

    fprintf(stderr, "[Adapter] LoRA merged %d pairs (%d DoRA, skipped %d), scale=%.2f\n", merged, dora_count,
            skipped, scale);
    return merged > 0;
}

// LoKr merge path. Matches the LyCORIS runtime forward for LoKr at
// lycoris/modules/lokr.py:551..566 (non bypass mode, scalar=1):
//   delta       = (alpha / rank) * kron(w1, W2)
//   no DoRA     : merged = base + delta * multiplier
//   DoRA present: merged = apply_weight_decompose(base + delta, multiplier)
//
// Two W2 variants are handled, selected per module by the suffixes present:
//   factorized  : W2 = w2_a @ w2_b, rank = w2_a.shape[1] = w2_b.shape[0]
//   monolithic  : W2 = w2 directly, rank = __metadata__.lokr_config.linear_dim
//
// LyCORIS picks monolithic when the factorized rank would not shrink the
// tensor (use_w2 path). qinglong SFT is fully factorized, garage-band is
// fully monolithic. w1 is always whole, not factorized (no w1_a). Tucker
// decomposition and convolutional LoKr are not implemented.
//
// Kron layout on GGML (same for both variants):
//   w1     row major (a, b)     -> ggml ne=(b, a)
//   W2     row major (c, d)     -> ggml ne=(d, c)
//   delta  row major (a*c, b*d) -> ggml ne=(b*d, a*c)
//
// Graph, factorized:
//   tw2 = mul_mat(cont(transpose(tw2b)), tw2a) = ne=(d, c)
// Graph, monolithic:
//   tw2 uploaded directly, ne=(d, c)
// Shared downstream:
//   tw1_s = scale(tw1, alpha / rank), scaling on the tiny side.
//   touter = mul_mat(reshape(tw1_s, 1, a*b), reshape(tw2, 1, c*d)) = ne=(a*b, c*d)
//   kron_p = permute(reshape_4d(touter, b, a, d, c), 1, 3, 0, 2) = ne=(d, b, c, a)
//   delta  = reshape_2d(cont(kron_p), b*d, a*c)
//
// ggml_permute axis_i positions src axis i AT new ne index axis_i
// (ggml.c:3781 sets ne[axis_i] = src.ne[i]), so mapping src (b, a, d, c)
// -> new (d, b, c, a) needs src axes (0, 1, 2, 3) to land at new positions
// (1, 3, 0, 2). The fast pair (d, b) then collapses into in_feat and the
// slow pair (c, a) into out_feat under reshape_2d. Net effect:
//   delta_rm[aa*c + cc, bb*d + dd] = W1[aa, bb] * W2[cc, dd]
static bool adapter_merge_lokr(WeightCtx *       wctx,
                               const GGUFModel & gf,
                               const STFile &    st,
                               float             user_scale,
                               ggml_backend_t    backend) {
    // group the per module tensors by LyCORIS prefix. Each module has either
    // w2 alone (monolithic) or w2_a + w2_b (factorized), never both.
    struct LoKrEntry {
        const STEntry * w1         = nullptr;
        const STEntry * w2         = nullptr;
        const STEntry * w2_a       = nullptr;
        const STEntry * w2_b       = nullptr;
        const STEntry * alpha      = nullptr;
        const STEntry * dora_scale = nullptr;
    };

    std::map<std::string, LoKrEntry> modules;

    for (const auto & e : st.entries) {
        std::string prefix, suffix;
        if (!adapter_split_suffix(e.name, &prefix, &suffix)) {
            continue;
        }
        if (prefix.compare(0, 8, "lycoris_") != 0) {
            continue;
        }
        LoKrEntry & m = modules[prefix];
        if (suffix == "lokr_w1") {
            m.w1 = &e;
        } else if (suffix == "lokr_w2") {
            m.w2 = &e;
        } else if (suffix == "lokr_w2_a") {
            m.w2_a = &e;
        } else if (suffix == "lokr_w2_b") {
            m.w2_b = &e;
        } else if (suffix == "alpha") {
            m.alpha = &e;
        } else if (suffix == "dora_scale") {
            m.dora_scale = &e;
        }
    }

    std::unordered_map<std::string, std::string> name_map = lokr_build_reverse_map(gf);

    std::unordered_map<const void *, size_t> pending_idx;
    pending_idx.reserve(wctx->pending.size());
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        pending_idx[wctx->pending[i].src] = i;
    }

    // linear_dim from __metadata__.lokr_config, only needed by monolithic modules.
    // Factorized modules derive rank from w2_a / w2_b shapes and ignore this value.
    int lokr_dim = adapter_read_lokr_dim(st);

    int merged     = 0;
    int skipped    = 0;
    int dora_count = 0;
    int mono_count = 0;

    for (const auto & kv : modules) {
        const std::string & lyc_prefix = kv.first;
        const LoKrEntry &   m          = kv.second;

        // per module variant: factorized (w2_a + w2_b) XOR monolithic (w2)
        bool has_factor = (m.w2_a && m.w2_b);
        bool has_mono   = (m.w2 != nullptr);
        if (!m.w1 || !m.alpha || has_factor == has_mono) {
            fprintf(stderr, "[Adapter] WARNING: incomplete or ambiguous LoKr module %s, skipping\n",
                    lyc_prefix.c_str());
            skipped++;
            continue;
        }

        auto nm_it = name_map.find(lyc_prefix);
        if (nm_it == name_map.end()) {
            fprintf(stderr, "[Adapter] WARNING: no GGUF tensor for %s, skipping\n", lyc_prefix.c_str());
            skipped++;
            continue;
        }
        const std::string & gguf_name = nm_it->second;

        int64_t tidx = gguf_find_tensor(gf.gguf, gguf_name.c_str());
        if (tidx < 0) {
            fprintf(stderr, "[Adapter] WARNING: tensor %s not in GGUF, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        struct ggml_tensor * tmeta = ggml_get_tensor(gf.meta, gguf_name.c_str());
        enum ggml_type       ttype = tmeta->type;
        int64_t              ne0   = tmeta->ne[0];
        int64_t              ne1   = tmeta->ne[1];

        size_t       toff     = gguf_get_tensor_offset(gf.gguf, tidx);
        const void * base_ptr = gf.mapping + gf.data_offset + toff;

        // LoKr shapes (safetensors row major):
        //   w1 : (a, b)
        //   factorized : w2_a (c, rank), w2_b (rank, d)
        //   monolithic : w2 (c, d), rank read from lokr_config.linear_dim
        // Kronecker product yields (a*c, b*d) = (out_feat, in_feat) = (ne1, ne0).
        int64_t a = m.w1->shape[0];
        int64_t b = m.w1->shape[1];
        int64_t c;
        int64_t d;
        int64_t r;

        if (has_factor) {
            c          = m.w2_a->shape[0];
            r          = m.w2_a->shape[1];
            d          = m.w2_b->shape[1];
            int64_t r2 = m.w2_b->shape[0];
            if (r != r2) {
                fprintf(stderr, "[Adapter] WARNING: LoKr rank mismatch w2_a=%lld vs w2_b=%lld for %s\n", (long long) r,
                        (long long) r2, lyc_prefix.c_str());
                skipped++;
                continue;
            }
        } else {
            c = m.w2->shape[0];
            d = m.w2->shape[1];
            if (lokr_dim <= 0) {
                fprintf(stderr,
                        "[Adapter] WARNING: monolithic LoKr %s needs __metadata__.lokr_config.linear_dim, skipping\n",
                        lyc_prefix.c_str());
                skipped++;
                continue;
            }
            r = lokr_dim;
        }

        if (a * c != ne1 || b * d != ne0) {
            fprintf(stderr,
                    "[Adapter] WARNING: LoKr shape mismatch for %s: kron(%lldx%lld, %lldx%lld) = %lldx%lld vs GGUF "
                    "out=%lld in=%lld\n",
                    gguf_name.c_str(), (long long) a, (long long) b, (long long) c, (long long) d, (long long) (a * c),
                    (long long) (b * d), (long long) ne1, (long long) ne0);
            skipped++;
            continue;
        }

        // alpha scalar, shape [] varies across trainers in dtype: F32, BF16, F16
        float alpha = 0.0f;
        if (!adapter_to_f32(st_data(st, *m.alpha), &alpha, 1, m.alpha->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported alpha dtype %s for %s, skipping\n", m.alpha->dtype.c_str(),
                    lyc_prefix.c_str());
            skipped++;
            continue;
        }

        // load w1 always to F32
        int64_t            w1_nel = a * b;
        std::vector<float> w1_f32((size_t) w1_nel);
        if (!adapter_to_f32(st_data(st, *m.w1), w1_f32.data(), w1_nel, m.w1->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported dtype %s for lokr_w1 %s\n", m.w1->dtype.c_str(),
                    lyc_prefix.c_str());
            skipped++;
            continue;
        }

        // load w2 payload: two buffers for factorized, one for monolithic
        int64_t            w2_nel  = 0;
        int64_t            w2a_nel = 0;
        int64_t            w2b_nel = 0;
        std::vector<float> w2_f32;
        std::vector<float> w2a_f32;
        std::vector<float> w2b_f32;
        if (has_factor) {
            w2a_nel = c * r;
            w2b_nel = r * d;
            w2a_f32.resize((size_t) w2a_nel);
            w2b_f32.resize((size_t) w2b_nel);
            if (!adapter_to_f32(st_data(st, *m.w2_a), w2a_f32.data(), w2a_nel, m.w2_a->dtype) ||
                !adapter_to_f32(st_data(st, *m.w2_b), w2b_f32.data(), w2b_nel, m.w2_b->dtype)) {
                fprintf(stderr, "[Adapter] WARNING: unsupported dtype in LoKr module %s, skipping\n",
                        lyc_prefix.c_str());
                skipped++;
                continue;
            }
        } else {
            w2_nel = c * d;
            w2_f32.resize((size_t) w2_nel);
            if (!adapter_to_f32(st_data(st, *m.w2), w2_f32.data(), w2_nel, m.w2->dtype)) {
                fprintf(stderr, "[Adapter] WARNING: unsupported dtype %s for lokr_w2 %s\n", m.w2->dtype.c_str(),
                        lyc_prefix.c_str());
                skipped++;
                continue;
            }
        }

        // optional DoRA scale vector, one F32 per output row
        const float *      ds_ptr = nullptr;
        std::vector<float> ds_f32;
        if (m.dora_scale) {
            int64_t ds_out = m.dora_scale->shape[0];
            if (ds_out != ne1) {
                fprintf(stderr, "[Adapter] WARNING: dora_scale dim0 %lld != out_feat %lld for %s\n", (long long) ds_out,
                        (long long) ne1, gguf_name.c_str());
                skipped++;
                continue;
            }
            ds_f32.resize((size_t) ds_out);
            if (!adapter_to_f32(st_data(st, *m.dora_scale), ds_f32.data(), ds_out, m.dora_scale->dtype)) {
                fprintf(stderr, "[Adapter] WARNING: unsupported dora_scale dtype %s for %s\n",
                        m.dora_scale->dtype.c_str(), gguf_name.c_str());
                skipped++;
                continue;
            }
            ds_ptr = ds_f32.data();
        }

        float scaling = alpha / (float) r;

        auto build = [&](struct ggml_context * ctx) {
            struct ggml_tensor * tw1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b, a);

            // tw2 ne=(d, c): w2_a @ w2_b for factorized, direct upload for monolithic
            struct ggml_tensor * tw2;
            struct ggml_tensor * tw2_src = nullptr;
            struct ggml_tensor * tw2a    = nullptr;
            struct ggml_tensor * tw2b    = nullptr;
            if (has_factor) {
                tw2a                        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, c);
                tw2b                        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, r);
                // transpose w2_b so rank sits on ne[0] contraction dim
                struct ggml_tensor * tw2b_T = ggml_cont(ctx, ggml_transpose(ctx, tw2b));
                tw2                         = ggml_mul_mat(ctx, tw2b_T, tw2a);
            } else {
                tw2_src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, c);
                tw2     = tw2_src;
            }

            // scale alpha / rank on the tiny (typically 4x4) side, cheapest placement
            struct ggml_tensor * tw1_s = ggml_scale(ctx, tw1, scaling);

            // kron via outer product + axis swap
            struct ggml_tensor * tw1_flat  = ggml_reshape_2d(ctx, tw1_s, 1, a * b);
            struct ggml_tensor * tw2_flat  = ggml_reshape_2d(ctx, tw2, 1, c * d);
            struct ggml_tensor * touter    = ggml_mul_mat(ctx, tw1_flat, tw2_flat);
            struct ggml_tensor * touter_4d = ggml_reshape_4d(ctx, touter, b, a, d, c);
            struct ggml_tensor * tkron_p   = ggml_permute(ctx, touter_4d, 1, 3, 0, 2);
            struct ggml_tensor * tkron_c   = ggml_cont(ctx, tkron_p);
            struct ggml_tensor * tdelta    = ggml_reshape_2d(ctx, tkron_c, b * d, a * c);

            adapter_delta_build db;
            db.tdelta = tdelta;
            db.upload = [=]() {
                ggml_backend_tensor_set(tw1, w1_f32.data(), 0, (size_t) w1_nel * sizeof(float));
                if (has_factor) {
                    ggml_backend_tensor_set(tw2a, w2a_f32.data(), 0, (size_t) w2a_nel * sizeof(float));
                    ggml_backend_tensor_set(tw2b, w2b_f32.data(), 0, (size_t) w2b_nel * sizeof(float));
                } else {
                    ggml_backend_tensor_set(tw2_src, w2_f32.data(), 0, (size_t) w2_nel * sizeof(float));
                }
            };
            return db;
        };

        const adapter_merge_semantics merge_semantics =
            ds_ptr ? adapter_merge_semantics::lycoris_dora : adapter_merge_semantics::lokr;
        if (!adapter_merge_on_backend(wctx, pending_idx, base_ptr, ttype, ne0, ne1, ds_ptr, user_scale,
                                      merge_semantics, backend, gguf_name.c_str(), build)) {
            return false;
        }
        if (ds_ptr) {
            dora_count++;
        }
        if (!has_factor) {
            mono_count++;
        }
        merged++;
    }

    fprintf(stderr,
            "[Adapter] LoKr merged %d modules (%d factorized, %d monolithic, %d with DoRA, skipped %d), scale=%.2f\n",
            merged, merged - mono_count, mono_count, dora_count, skipped, user_scale);
    return merged > 0;
}

// Main adapter merge entry point.
//
// Call after all GGUF tensors are loaded into wctx->pending but before wctx_alloc.
// Detects the adapter algorithm from the safetensors payload and dispatches to
// the matching merge path. The adapter_path points to either:
//   PEFT directory  : a folder with adapter_model.safetensors + adapter_config.json
//   LyCORIS file    : a flat .safetensors file (LoRA ComfyUI or LoKr)
// Directories exist only for PEFT. LyCORIS ships as a single file for both LoRA
// and LoKr payloads.
static bool adapter_merge(WeightCtx *       wctx,
                          const GGUFModel & gf,
                          const char *      adapter_path,
                          float             scale,
                          ggml_backend_t    backend) {
    std::string           sf_path;
    std::string           cfg_dir;
    bool                  peft_directory = false;
    std::filesystem::path resolved_path;
    std::string           directory_error;
    if (!adapter_checkpoint_directory(adapter_path, resolved_path, directory_error)) {
        fprintf(stderr, "[Adapter] %s\n", directory_error.c_str());
        return false;
    }
    const std::string resolved_adapter_path = resolved_path.string();

    struct stat sb;
    if (stat(resolved_adapter_path.c_str(), &sb) != 0) {
        fprintf(stderr, "[Adapter] path does not exist: %s\n", adapter_path);
        return false;
    }

    if (S_ISDIR(sb.st_mode)) {
        peft_directory = true;
        sf_path = resolved_adapter_path + "/adapter_model.safetensors";
        cfg_dir = resolved_adapter_path;
        if (stat(sf_path.c_str(), &sb) != 0) {
            fprintf(stderr, "[Adapter] directory %s is not a PEFT layout, missing adapter_model.safetensors\n",
                    adapter_path);
            return false;
        }
    } else {
        // LyCORIS flat file, LoRA or LoKr
        sf_path    = resolved_adapter_path;
        size_t sep = sf_path.find_last_of("/\\");
        cfg_dir    = (sep != std::string::npos) ? sf_path.substr(0, sep) : ".";
    }

    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        return false;
    }

    bool ok;
    if (adapter_detect_lokr(st)) {
        ok = !peft_directory && adapter_merge_lokr(wctx, gf, st, scale, backend);
    } else {
        ok = adapter_merge_lora(wctx, gf, st, cfg_dir, peft_directory, scale, backend);
    }

    st_close(&st);
    return ok;
}
