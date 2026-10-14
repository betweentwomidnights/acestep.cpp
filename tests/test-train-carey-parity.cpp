// Emits small fixtures from production batch/loss/optimizer/scheduler code.
// No model weights are loaded; the Python side imports Carey's actual helpers.
#include "test-backend.h"
#include "train-diffusion.h"

#include <fstream>
#include <iomanip>
#include <iostream>

template <class T> static void array(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

static void check(bool ok, const std::string & error) {
    if (!ok) {
        std::cerr << error << '\n';
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "Usage: test-train-carey-parity output.json\n";
        return 1;
    }
    std::ofstream out(argv[1]);
    check(bool(out), "cannot open fixture output");
    out << std::setprecision(17);
    ggml_context * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, false });
    check(ctx != nullptr, "cannot allocate fixture context");
    DiTGGML model          = {};
    model.cfg.out_channels = 2;
    model.cfg.in_channels  = 6;
    model.cfg.patch_size   = 2;
    model.cond_emb_w       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    std::string                           error;
    std::vector<ACETrainDiffusionExample> sources(2);
    for (size_t s = 0; s < sources.size(); ++s) {
        auto & e = sources[s];
        e.target_latents.resize(8);
        e.context_latents.resize(16);
        e.encoder_hidden.resize(12);
        for (size_t i = 0; i < e.target_latents.size(); ++i) {
            e.target_latents[i] = float(i + s) * 0.03f;
        }
        for (size_t i = 0; i < e.context_latents.size(); ++i) {
            e.context_latents[i] = float(i + s) * 0.02f;
        }
        for (size_t i = 0; i < e.encoder_hidden.size(); ++i) {
            e.encoder_hidden[i] = float(i + s) * 0.01f;
        }
        e.real_temporal_length         = 4;
        e.real_encoder_sequence_length = 3;
    }
    const std::vector<float> null_cond = { -1, -2, -3, -4 };
    out << "{\"batches\":[";
    ggml_backend_t backend = test_cpu_backend();
    check(backend != nullptr, "no CPU backend");
    for (int c = 0; c < 4; ++c) {
        if (c) {
            out << ',';
        }
        sources[1].real_temporal_length = c == 3 ? 2 : 4;
        ACETrainDiffusionConfig config;
        config.cfg_dropout = c == 1 ? 1.0f : 0.0f;
        config.min_snr     = c != 2;
        ACETrainDiffusionBatch batch;
        check(ace_prepare_train_diffusion_batch(model, sources, 4, 3, null_cond, 456, config, batch, error), error);
        std::vector<float> data, context, encoder;
        for (const auto & e : sources) {
            data.insert(data.end(), e.target_latents.begin(), e.target_latents.end());
            context.insert(context.end(), e.context_latents.begin(), e.context_latents.end());
            encoder.insert(encoder.end(), e.encoder_hidden.begin(), e.encoder_hidden.end());
        }
        std::vector<float> prediction(16, 0.125f), derivative(16);
        ggml_tensor *      p       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 4, 2);
        ggml_tensor *      target  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 4, 2);
        ggml_tensor *      weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        ggml_set_param(p);
        std::memcpy(p->data, prediction.data(), prediction.size() * sizeof(float));
        std::memcpy(target->data, batch.target_velocity.data(), prediction.size() * sizeof(float));
        std::memcpy(weights->data, batch.loss_weights.data(), batch.loss_weights.size() * sizeof(float));
        ggml_tensor * loss = ace_train_flow_loss(ctx, p, target, weights);
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, 1024, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "fixture loss graph failed");
        ggml_tensor * grad = ggml_graph_get_grad(graph, p);
        check(grad != nullptr, "missing loss gradient");
        std::memcpy(derivative.data(), grad->data, derivative.size() * sizeof(float));
        out << "{\"cfg\":" << config.cfg_dropout << ",\"min_snr\":" << (config.min_snr ? "true" : "false")
            << ",\"padded\":" << (c == 3 ? "true" : "false") << ",\"loss\":" << *static_cast<float *>(loss->data);
        for (const auto & pair : std::vector<std::pair<const char *, const std::vector<float> *>>{
                 { "data",      &data                      },
                 { "context",   &context                   },
                 { "encoder",   &encoder                   },
                 { "null",      &null_cond                 },
                 { "input",     &batch.input_latents       },
                 { "condition", &batch.encoder_hidden      },
                 { "target",    &batch.target_velocity     },
                 { "t",         &batch.timesteps           },
                 { "r",         &batch.reference_timesteps },
                 { "weights",   &batch.loss_weights        },
                 { "gradient",  &derivative                }
        }) {
            out << ",\"" << pair.first << "\":";
            array(out, *pair.second);
        }
        out << '}';
    }
    ggml_backend_free(backend);
    out << "],\"timesteps\":[";
    const std::vector<float> logits = { -10, -2, -0.4f, 0, 2, 10 };
    for (size_t i = 0; i < logits.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << ace_train_timestep_from_logits(logits[i], logits[logits.size() - 1 - i]);
    }
    out << "],\"optimizer\":[";
    for (int mode = 0; mode < 2; ++mode) {
        if (mode) {
            out << ',';
        }
        ACETrainAdapterState state = {};
        state.params.resize(1);
        state.params[0].a         = { 0.125f, -0.25f, 0.5f };
        state.params[0].b         = { 0.0f, 0.2f };
        state.params[0].magnitude = mode ? std::vector<float>{ 1.0f, 0.75f } : std::vector<float>{};
        ACETrainAdapterGraphState graph;
        ACETrainAdapterGraphParam param = {};
        param.a                         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        param.b                         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        param.magnitude                 = mode ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2) : nullptr;
        graph.params.push_back(param);
        ACETrainAdapterOptimizer optimizer;
        ACETrainAdamWConfig      config;
        config.learning_rate     = 0.001f;
        config.max_gradient_norm = 0.5f;
        out << "{\"dora\":" << (mode ? "true" : "false") << ",\"steps\":[";
        for (int step = 0; step < 3; ++step) {
            if (step) {
                out << ',';
            }
            ACETrainAdapterGradientAccumulator accum;
            accum.microbatch_count = 2;
            accum.example_count    = 3;
            accum.params.resize(1);
            accum.params[0].a = { 0.3f + step, -0.6f, 0.9f };
            accum.params[0].b = { 0.6f, -0.3f - step };
            if (mode) {
                accum.params[0].magnitude = { -0.15f, 0.45f };
            }
            out << "{\"sum_a\":";
            array(out, accum.params[0].a);
            out << ",\"sum_b\":";
            array(out, accum.params[0].b);
            out << ",\"sum_m\":";
            array(out, accum.params[0].magnitude);
            double norm = 0;
            check(ace_train_adapter_adamw_step_accumulated(graph, state, optimizer, config, accum, error, &norm),
                  error);
            out << ",\"norm\":" << norm << ",\"a\":";
            array(out, state.params[0].a);
            out << ",\"b\":";
            array(out, state.params[0].b);
            out << ",\"m\":";
            array(out, state.params[0].magnitude);
            out << ",\"first_a\":";
            array(out, optimizer.params[0].a.first_moment);
            out << ",\"second_a\":";
            array(out, optimizer.params[0].a.second_moment);
            out << '}';
        }
        out << "]}";
    }
    out << "],\"schedules\":[";
    for (int s = 0; s < 4; ++s) {
        if (s) {
            out << ',';
        }
        out << '[';
        for (int step = 0; step < 23; ++step) {
            if (step) {
                out << ',';
            }
            out << ace_train_learning_rate(0.001f, step, 23, 2, ACETrainSchedule(s), 4);
        }
        out << ']';
    }
    out << "],\"long_warmup\":[";
    for (int step = 0; step < 20; ++step) {
        if (step) {
            out << ',';
        }
        out << ace_train_learning_rate(0.001f, step, 20, 50, ACE_TRAIN_SCHEDULE_COSINE);
    }
    out << "],\"zero_warmup\":[";
    for (int step = 0; step < 20; ++step) {
        if (step) {
            out << ',';
        }
        out << ace_train_learning_rate(0.001f, step, 20, 0, ACE_TRAIN_SCHEDULE_COSINE);
    }
    sources[1].target_latents.resize(4);
    sources[1].context_latents.resize(8);
    sources[1].encoder_hidden.resize(8);
    sources[1].real_temporal_length         = 2;
    sources[1].real_encoder_sequence_length = 2;
    std::vector<ACETrainDiffusionExample> collated;
    int                                   temporal = 0, encoder_len = 0;
    const std::vector<float>              silence(8, -0.25f);
    check(ace_collate_training_examples(model, sources, silence, null_cond, collated, temporal, encoder_len, error),
          error);
    out << "],\"collation\":[";
    for (size_t i = 0; i < sources.size(); ++i) {
        if (i) {
            out << ',';
        }
        out << '{';
        int n = 0;
        for (const auto & pair : std::vector<std::pair<const char *, const std::vector<float> *>>{
                 { "data",             &sources[i].target_latents   },
                 { "context",          &sources[i].context_latents  },
                 { "encoder",          &sources[i].encoder_hidden   },
                 { "collated_data",    &collated[i].target_latents  },
                 { "collated_context", &collated[i].context_latents },
                 { "collated_encoder", &collated[i].encoder_hidden  }
        }) {
            if (n++) {
                out << ',';
            }
            out << '"' << pair.first << "\":";
            array(out, *pair.second);
        }
        out << '}';
    }
    out << "]}\n";
    ggml_free(ctx);
    out.close();
    check(bool(out), "failed to write fixture");
    return 0;
}
