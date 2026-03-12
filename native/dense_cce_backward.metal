// Copyright © 2026

#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"

using namespace metal;

namespace mlx_cce_runtime_ext {

template <typename T>
METAL_FUNC float safe_exp_diff(float a, float b) {
  if (a <= -INFINITY) {
    return 0.0f;
  }
  return fast::exp(a - b);
}

METAL_FUNC float apply_softcap(float x, float softcap) {
  if (softcap <= 0.0f) {
    return x;
  }
  return softcap * fast::tanh(x / softcap);
}

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
  for (int i = 0; i < N_READS; ++i) {
    const int idx = base_idx + i;
    if (idx >= total_elements) {
      continue;
    }

    const int row = idx / chunk_V;
    const int col = idx % chunk_V;
    const int global_v = v_start + col;
    if (global_v >= V) {
      d_logits[idx] = T(0.0f);
      continue;
    }

    const int target = targets[row];
    if (target == ignore_index) {
      d_logits[idx] = T(0.0f);
      continue;
    }

    float raw_logit = float(logits[idx]);
    float capped_logit = apply_softcap(raw_logit, softcap);
    float prob = fast::exp(capped_logit - lse[row]);
    prob = clamp(prob, 0.0f, 1.0f);
    float grad = (prob - float(global_v == target)) * grad_output[row];
    grad *= softcap_grad(raw_logit, softcap);
    d_logits[idx] = T(grad);
  }
}

template <typename T, int N_READS = 4>
[[kernel]] void runtime_cce_chunk_logsumexp(
    const device T* logits [[buffer(0)]],
    const device int32_t* targets [[buffer(1)]],
    device float* running_max [[buffer(2)]],
    device float* running_sum_exp [[buffer(3)]],
    device float* target_logit [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& chunk_V [[buffer(6)]],
    constant int& v_start [[buffer(7)]],
    constant int& V [[buffer(8)]],
    constant float& softcap [[buffer(9)]],
    threadgroup float* smem [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int threads_per_tg = 256;
  constexpr int simd_size = 32;
  constexpr int num_simdgroups = threads_per_tg / simd_size;

  const int row = tgid.x;
  if (row >= N) {
    return;
  }

  const int target = targets[row];
  const device T* row_logits = logits + row * chunk_V;
  threadgroup float* smem_max = smem;
  threadgroup float* smem_sum = smem + num_simdgroups;

  if (lid < num_simdgroups) {
    smem_max[lid] = -INFINITY;
    smem_sum[lid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int valid_chunk_v = min(chunk_V, V - v_start);
  const int iterations = (valid_chunk_v + threads_per_tg * N_READS - 1) / (threads_per_tg * N_READS);

  float maxval = -INFINITY;
  float normalizer = 0.0f;
  float local_target = 0.0f;
  bool found_target = false;

  for (int r = 0; r < iterations; ++r) {
    int offset = r * threads_per_tg * N_READS + lid * N_READS;
    float vals[N_READS];

    #pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      float raw = (offset + i < valid_chunk_v) ? float(row_logits[offset + i]) : -INFINITY;
      vals[i] = (raw <= -INFINITY) ? raw : apply_softcap(raw, softcap);
      int global_v = v_start + offset + i;
      if (global_v == target && offset + i < valid_chunk_v) {
        local_target = vals[i];
        found_target = true;
      }
    }

    float prevmax = maxval;
    #pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      maxval = max(maxval, vals[i]);
    }
    normalizer *= safe_exp_diff<T>(prevmax, maxval);
    #pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      normalizer += safe_exp_diff<T>(vals[i], maxval);
    }
  }

  float prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= safe_exp_diff<T>(prevmax, maxval);
  normalizer = simd_sum(normalizer);

  if (simd_lid == 0) {
    smem_max[simd_gid] = maxval;
    smem_sum[simd_gid] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float chunk_max = smem_max[0];
  float chunk_sum_exp = smem_sum[0];
  if (simd_gid == 0 && simd_lid == 0) {
    for (int i = 1; i < num_simdgroups; ++i) {
      float sg_max = smem_max[i];
      float sg_sum = smem_sum[i];
      float new_max = max(chunk_max, sg_max);
      chunk_sum_exp = chunk_sum_exp * safe_exp_diff<T>(chunk_max, new_max) +
          sg_sum * safe_exp_diff<T>(sg_max, new_max);
      chunk_max = new_max;
    }
  }

  if (lid == 0) {
    float old_max = running_max[row];
    float old_sum_exp = running_sum_exp[row];
    float new_max = max(old_max, chunk_max);
    float new_sum_exp = old_sum_exp * safe_exp_diff<T>(old_max, new_max) +
        chunk_sum_exp * safe_exp_diff<T>(chunk_max, new_max);
    running_max[row] = new_max;
    running_sum_exp[row] = new_sum_exp;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  float simd_target_val = simd_sum(local_target);
  bool any_found = simd_any(found_target);
  if (simd_lid == 0) {
    smem_max[simd_gid] = simd_target_val;
    smem_sum[simd_gid] = any_found ? 1.0f : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid == 0) {
    for (int i = 0; i < num_simdgroups; ++i) {
      if (smem_sum[i] != 0.0f) {
        target_logit[row] = smem_max[i];
        break;
      }
    }
  }
}

METAL_FUNC float cce_compute_lse(float running_max_val, float running_sum_exp_val) {
  return running_max_val + log(running_sum_exp_val + 1e-9f);
}

[[host_name("runtime_cce_finalize_lse")]]
[[kernel]] void runtime_cce_finalize_lse(
    const device float* running_max [[buffer(0)]],
    const device float* running_sum_exp [[buffer(1)]],
    device float* logsumexp [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(N)) {
    return;
  }
  logsumexp[tid] = cce_compute_lse(running_max[tid], running_sum_exp[tid]);
}

[[host_name("runtime_cce_init_running_values")]]
[[kernel]] void runtime_cce_init_running_values(
    device float* running_max [[buffer(0)]],
    device float* running_sum_exp [[buffer(1)]],
    device float* target_logit [[buffer(2)]],
    constant int& N [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(N)) {
    return;
  }
  running_max[tid] = -INFINITY;
  running_sum_exp[tid] = 0.0f;
  target_logit[tid] = 0.0f;
}

[[host_name("runtime_cce_finalize_loss")]]
[[kernel]] void runtime_cce_finalize_loss(
    const device float* running_max [[buffer(0)]],
    const device float* running_sum_exp [[buffer(1)]],
    const device float* target_logit [[buffer(2)]],
    const device int32_t* targets [[buffer(3)]],
    device float* loss [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& ignore_index [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(N)) {
    return;
  }
  if (targets[tid] == ignore_index) {
    loss[tid] = 0.0f;
    return;
  }
  float lse = cce_compute_lse(running_max[tid], running_sum_exp[tid]);
  loss[tid] = (lse - target_logit[tid]) * scale;
}

[[host_name("runtime_cce_finalize_loss_with_lse")]]
[[kernel]] void runtime_cce_finalize_loss_with_lse(
    const device float* running_max [[buffer(0)]],
    const device float* running_sum_exp [[buffer(1)]],
    const device float* target_logit [[buffer(2)]],
    const device int32_t* targets [[buffer(3)]],
    device float* loss [[buffer(4)]],
    device float* logsumexp_out [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& ignore_index [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(N)) {
    return;
  }
  float lse = cce_compute_lse(running_max[tid], running_sum_exp[tid]);
  logsumexp_out[tid] = lse;
  if (targets[tid] == ignore_index) {
    loss[tid] = 0.0f;
    return;
  }
  loss[tid] = (lse - target_logit[tid]) * scale;
}

} // namespace mlx_cce_runtime_ext

#define instantiate_runtime_cce_dlogits(name, type)                               \
  template [[host_name("runtime_cce_compute_d_logits_" #name)]]                   \
  [[kernel]] void mlx_cce_runtime_ext::runtime_cce_compute_d_logits<type>(        \
      const device type* logits [[buffer(0)]],                                    \
      const device float* lse [[buffer(1)]],                                      \
      const device int32_t* targets [[buffer(2)]],                                \
      const device float* grad_output [[buffer(3)]],                              \
      device type* d_logits [[buffer(4)]],                                        \
      constant int& N [[buffer(5)]],                                              \
      constant int& chunk_V [[buffer(6)]],                                        \
      constant int& v_start [[buffer(7)]],                                        \
      constant int& V [[buffer(8)]],                                              \
      constant float& softcap [[buffer(9)]],                                      \
      constant int& ignore_index [[buffer(10)]],                                  \
      uint tid [[thread_position_in_grid]]);

#define instantiate_runtime_cce_chunk_logsumexp(name, type)                       \
  template [[host_name("runtime_cce_chunk_logsumexp_" #name)]]                    \
  [[kernel]] void mlx_cce_runtime_ext::runtime_cce_chunk_logsumexp<type>(         \
      const device type* logits [[buffer(0)]],                                    \
      const device int32_t* targets [[buffer(1)]],                                \
      device float* running_max [[buffer(2)]],                                    \
      device float* running_sum_exp [[buffer(3)]],                                \
      device float* target_logit [[buffer(4)]],                                   \
      constant int& N [[buffer(5)]],                                              \
      constant int& chunk_V [[buffer(6)]],                                        \
      constant int& v_start [[buffer(7)]],                                        \
      constant int& V [[buffer(8)]],                                              \
      constant float& softcap [[buffer(9)]],                                      \
      threadgroup float* smem [[threadgroup(0)]],                                 \
      uint3 tgid [[threadgroup_position_in_grid]],                                \
      uint lid [[thread_index_in_threadgroup]],                                   \
      uint simd_lid [[thread_index_in_simdgroup]],                                \
      uint simd_gid [[simdgroup_index_in_threadgroup]]);

instantiate_runtime_cce_dlogits(float32, float)
instantiate_runtime_cce_dlogits(float16, half)
instantiate_runtime_cce_dlogits(bfloat16, bfloat16_t)

instantiate_runtime_cce_chunk_logsumexp(float32, float)
instantiate_runtime_cce_chunk_logsumexp(float16, half)
instantiate_runtime_cce_chunk_logsumexp(bfloat16, bfloat16_t)
