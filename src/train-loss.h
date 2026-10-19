// Flow loss shared by full/checkpointed decoder graphs and parity fixtures.
#pragma once

#include "ggml.h"

static ggml_tensor * ace_train_flow_loss(ggml_context * ctx,
                                         ggml_tensor *  velocity,
                                         ggml_tensor *  target,
                                         ggml_tensor *  weights) {
    ggml_tensor * squared = ggml_sqr(ctx, ggml_sub(ctx, velocity, target));
    ggml_tensor * frames  = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, squared), velocity->ne[1], velocity->ne[2]);
    frames                = ggml_scale(ctx, frames, 1.0f / float(velocity->ne[0]));
    return ggml_scale(ctx, ggml_sum(ctx, ggml_mul(ctx, frames, weights)), 1.0f / float(velocity->ne[2]));
}
