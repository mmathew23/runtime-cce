// Copyright © 2026

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/bf16.h"

using namespace metal;

namespace mlx_cce_runtime_ext {

template <typename T>
METAL_FUNC float apply_softcap(float x, float softcap) {
  if (softcap <= 0.0f) {
    return x;
  }
  return softcap * fast::tanh(x / softcap);
}

template <typename T>
METAL_FUNC float softcap_grad(float x, float softcap) {
  if (softcap <= 0.0f) {
    return 1.0f;
  }
  float t = fast::tanh(x / softcap);
  return 1.0f - t * t;
}

template <typename T, int N_READS = 4>
[[kernel]] void runtime_cce_compute_d_logits(
    const device T* logits [[buffer(0)]],
    const device float* lse [[buffer(1)]],
    const device int32_t* targets [[buffer(2)]],
    const device float* grad_output [[buffer(3)]],
    device T* d_logits [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& chunk_V [[buffer(6)]],
    constant int& v_start [[buffer(7)]],
    constant int& V [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    constant int& ignore_index [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  const int base_idx = tid * N_READS;
  const int total_elements = N * chunk_V;

  if (base_idx >= total_elements) {
    return;
  }

  #pragma unroll
  for (int i = 0; i < N_READS; i++) {
    const int idx = base_idx + i;
    if (idx >= total_elements) {
      continue;
    }

    const int row = idx / chunk_V;
    const int col = idx % chunk_V;
    const int global_v = v_start + col;
    if (global_v >= V) {
      d_logits[idx] = static_cast<T>(0.0f);
      continue;
    }

    const int target = targets[row];
    if (target == ignore_index) {
      d_logits[idx] = static_cast<T>(0.0f);
      continue;
    }

    const float token_lse = lse[row];
    const float grad_scale = grad_output[row];

    float raw_logit = float(logits[idx]);
    float capped_logit = apply_softcap<T>(raw_logit, softcap);
    float prob = fast::exp(capped_logit - token_lse);
    prob = clamp(prob, 0.0f, 1.0f);

    float d_capped = (prob - float(global_v == target)) * grad_scale;
    float d_logit = d_capped * softcap_grad<T>(raw_logit, softcap);
    d_logits[idx] = static_cast<T>(d_logit);
  }
}

} // namespace mlx_cce_runtime_ext

#define instantiate_runtime_cce_dlogits(name, type)                                 \
  template [[host_name("runtime_cce_compute_d_logits_" #name)]]                     \
  [[kernel]] void mlx_cce_runtime_ext::runtime_cce_compute_d_logits<type>(          \
      const device type* logits [[buffer(0)]],                                      \
      const device float* lse [[buffer(1)]],                                        \
      const device int32_t* targets [[buffer(2)]],                                  \
      const device float* grad_output [[buffer(3)]],                                \
      device type* d_logits [[buffer(4)]],                                          \
      constant int& N [[buffer(5)]],                                                \
      constant int& chunk_V [[buffer(6)]],                                          \
      constant int& v_start [[buffer(7)]],                                          \
      constant int& V [[buffer(8)]],                                                \
      constant float& softcap [[buffer(9)]],                                        \
      constant int& ignore_index [[buffer(10)]],                                    \
      uint tid [[thread_position_in_grid]]);

instantiate_runtime_cce_dlogits(float32, float)
instantiate_runtime_cce_dlogits(float16, half)
instantiate_runtime_cce_dlogits(bfloat16, bfloat16_t)
