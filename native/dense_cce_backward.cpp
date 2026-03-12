// Copyright © 2026

#include <dlfcn.h>

#include <filesystem>
#include <optional>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/utils.h"

#include "dense_cce_backward.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace mlx_cce_runtime_ext {

namespace {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to determine current binary path.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

constexpr int MAX_CHUNK_V = 16384;
constexpr int MIN_CHUNK_V = 1024;

struct PreparedArray {
  mx::array value;
  bool temporary{false};
};

int get_adaptive_chunk_v(
    int n,
    int v,
    size_t bytes_per_element,
    int scratch_arrays = 1) {
  if (n <= 2048) {
    return std::min(MAX_CHUNK_V, v);
  }

  size_t system_memory = 64ULL * 1024 * 1024 * 1024;
#if defined(__APPLE__)
  size_t size = sizeof(system_memory);
  if (sysctlbyname("hw.memsize", &system_memory, &size, nullptr, 0) != 0) {
    system_memory = 64ULL * 1024 * 1024 * 1024;
  }
#endif

  size_t chunk_budget = system_memory / 200;
  size_t denom = static_cast<size_t>(std::max(n, 1)) *
      std::max<size_t>(bytes_per_element, 1) *
      static_cast<size_t>(std::max(scratch_arrays, 1));
  int max_chunk_from_memory = static_cast<int>(chunk_budget / denom);
  int chunk_v = std::min({MAX_CHUNK_V, v, std::max(MIN_CHUNK_V, max_chunk_from_memory)});
  chunk_v = (chunk_v / 256) * 256;
  if (chunk_v < MIN_CHUNK_V) {
    chunk_v = std::min(MIN_CHUNK_V, v);
  }
  return chunk_v;
}

mx::array apply_softcap(const mx::array& logits, float logit_softcap, mx::Stream s) {
  if (logit_softcap <= 0.0f) {
    return logits;
  }
  auto softcap = mx::array(logit_softcap, mx::float32);
  return mx::multiply(mx::tanh(mx::divide(logits, softcap, s), s), softcap, s);
}

bool is_supported_compute_dtype(mx::Dtype dtype) {
  return dtype == mx::float16 || dtype == mx::bfloat16 || dtype == mx::float32;
}

PreparedArray materialize_input(const mx::array& arr, mx::Dtype dtype, mx::Stream s) {
  PreparedArray prepared{
      (arr.dtype() == dtype) ? arr : mx::astype(arr, dtype, s),
      arr.dtype() != dtype};
  if (!prepared.value.flags().row_contiguous) {
    prepared.value = mx::contiguous(prepared.value, false, s);
    prepared.temporary = true;
  }
  if (prepared.value.has_primitive()) {
    prepared.value.eval();
  }
  return prepared;
}

void add_if_temporary(mx::metal::Device& d, const PreparedArray& prepared, const mx::Stream& s) {
  if (prepared.temporary) {
    d.add_temporary(prepared.value, s.index);
  }
}

void launch_regular_gemm(
    const mx::Stream& s,
    mx::metal::Device& d,
    MTL::Library* lib,
    const mx::array& a,
    const mx::array& b,
    mx::array& out,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldd,
    bool transpose_a,
    bool transpose_b,
    std::optional<std::reference_wrapper<const mx::array>> c = std::nullopt,
    float alpha = 1.0f,
    float beta = 0.0f) {
  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int wm = 2;
  constexpr int wn = 2;
  const bool has_batch = false;
  const bool use_out_source = c.has_value();
  const bool do_axpby = use_out_source && (alpha != 1.0f || beta != 1.0f);
  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  std::ostringstream base_name;
  base_name << "steel_gemm_fused_" << (transpose_a ? 't' : 'n')
            << (transpose_b ? 't' : 'n') << "_" << mx::type_to_name(a) << "_"
            << mx::type_to_name(out) << "_bm" << bm << "_bn" << bn << "_bk" << bk
            << "_wm" << wm << "_wn" << wn;

  std::ostringstream hash_name;
  hash_name << base_name.str() << "_has_batch_" << (has_batch ? 't' : 'n')
            << "_use_out_source_" << (use_out_source ? 't' : 'n') << "_do_axpby_"
            << (do_axpby ? 't' : 'n') << "_align_M_" << (align_M ? 't' : 'n')
            << "_align_N_" << (align_N ? 't' : 'n') << "_align_K_"
            << (align_K ? 't' : 'n');

  mx::metal::MTLFCList func_consts = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  auto kernel =
      d.get_kernel(base_name.str(), lib, hash_name.str(), func_consts, {});
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);

  int tiles_n = (N + bn - 1) / bn;
  int tiles_m = (M + bm - 1) / bm;
  int swizzle_log = 0;

  mlx::steel::GEMMParams params{
      M,
      N,
      K,
      lda,
      ldb,
      ldd,
      tiles_n,
      tiles_m,
      0,
      0,
      0,
      swizzle_log,
      (K / bk),
      0,
  };

  MTL::Size group_dims = MTL::Size(32, wn, wm);
  MTL::Size grid_dims = MTL::Size(tiles_n, tiles_m, 1);

  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(b, 1);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(params, 4);

  if (use_out_source) {
    const auto& c_ref = c->get();
    mlx::steel::GEMMAddMMParams addmm_params{
        static_cast<int>(c_ref.strides()[c_ref.ndim() - 2]),
        static_cast<int>(c_ref.strides()[c_ref.ndim() - 1]),
        0,
        alpha,
        beta,
    };
    compute_encoder.set_input_array(c_ref, 2);
    compute_encoder.set_bytes(addmm_params, 5);
  }

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

std::function<std::vector<mx::array>(std::vector<mx::array>)> make_forward_fallback(
    int ignore_index,
    float logit_softcap,
    bool output_logsumexp,
    mx::Stream s) {
  return [ignore_index, logit_softcap, output_logsumexp, s](std::vector<mx::array> inputs) {
    auto& hidden = inputs[0];
    auto& weight = inputs[1];
    auto& targets = inputs[2];

    auto logits = mx::matmul(hidden, mx::transpose(weight, s), s);
    logits = mx::astype(logits, mx::float32, s);
    logits = apply_softcap(logits, logit_softcap, s);

    auto lse = mx::logsumexp(logits, -1, false, s);
    auto t_clamped = mx::clip(
        targets,
        mx::array(0, mx::int32),
        mx::array(static_cast<int>(weight.shape(0) - 1), mx::int32),
        s);
    auto target_logits = mx::reshape(
        mx::take_along_axis(logits, mx::expand_dims(t_clamped, -1, s), 1, s),
        {hidden.shape(0)},
        s);
    auto valid = mx::not_equal(targets, mx::array(ignore_index, mx::int32), s);
    auto loss = mx::where(
        valid,
        mx::subtract(lse, target_logits, s),
        mx::zeros_like(lse, s),
        s);
    if (output_logsumexp) {
      return std::vector<mx::array>{loss, lse};
    }
    return std::vector<mx::array>{loss};
  };
}

std::function<std::vector<mx::array>(std::vector<mx::array>)> make_backward_fallback(
    int ignore_index,
    float logit_softcap,
    bool has_logsumexp,
    mx::Stream s) {
  return [ignore_index, logit_softcap, has_logsumexp, s](std::vector<mx::array> inputs) {
    auto& hidden = inputs[0];
    auto& weight = inputs[1];
    auto& targets = inputs[2];
    auto& grad_output = inputs[3];

    auto logits = mx::matmul(hidden, mx::transpose(weight, s), s);
    logits = mx::astype(logits, mx::float32, s);
    logits = apply_softcap(logits, logit_softcap, s);

    auto lse = (has_logsumexp && inputs.size() > 4)
        ? mx::expand_dims(inputs[4], -1, s)
        : mx::logsumexp(logits, -1, true, s);

    int n = hidden.shape(0);
    int v = weight.shape(0);
    auto softmax_probs = mx::exp(mx::subtract(logits, lse, s), s);

    auto one_hot_arr = mx::zeros({n, v}, mx::float32, s);
    auto valid_mask = mx::logical_and(
        mx::greater_equal(targets, mx::array(0, mx::int32), s),
        mx::less(targets, mx::array(v, mx::int32), s),
        s);
    valid_mask = mx::logical_and(
        valid_mask,
        mx::not_equal(targets, mx::array(ignore_index, mx::int32), s),
        s);
    auto t_clamped = mx::clip(
        targets,
        mx::array(0, mx::int32),
        mx::array(v - 1, mx::int32),
        s);
    auto vocab_idx = mx::reshape(mx::arange(v, mx::int32, s), {1, v}, s);
    auto target_idx = mx::reshape(t_clamped, {n, 1}, s);
    one_hot_arr = mx::astype(mx::equal(target_idx, vocab_idx, s), mx::float32, s);
    one_hot_arr = mx::multiply(
        one_hot_arr,
        mx::reshape(mx::astype(valid_mask, mx::float32, s), {n, 1}, s),
        s);

    auto grad_logits = mx::subtract(softmax_probs, one_hot_arr, s);
    grad_logits = mx::multiply(grad_logits, mx::expand_dims(grad_output, -1, s), s);
    auto ignore_mask = mx::equal(targets, mx::array(ignore_index, mx::int32), s);
    grad_logits = mx::where(
        mx::expand_dims(ignore_mask, -1, s),
        mx::zeros_like(grad_logits, s),
        grad_logits,
        s);

    auto compute_type = hidden.dtype();
    auto grad_hidden = mx::matmul(
        mx::astype(grad_logits, compute_type, s),
        mx::astype(weight, compute_type, s),
        s);
    auto grad_weight = mx::matmul(
        mx::transpose(mx::astype(grad_logits, compute_type, s), s),
        mx::astype(hidden, compute_type, s),
        s);
    return std::vector<mx::array>{grad_hidden, grad_weight};
  };
}

} // namespace

std::pair<mx::array, mx::array> dense_cce_loss(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index,
    float logit_softcap,
    mx::StreamOrDevice s) {
  if (hidden.ndim() != 2 || weight.ndim() != 2 || targets.ndim() != 1) {
    throw std::invalid_argument("dense_cce_loss expects hidden/weight 2D and targets 1D.");
  }
  if (hidden.shape(0) != targets.shape(0) || hidden.shape(1) != weight.shape(1)) {
    throw std::invalid_argument("dense_cce_loss input shapes do not align.");
  }

  auto stream = mx::to_stream(s);
  auto outputs = mx::array::make_arrays(
      {{hidden.shape(0)}, {hidden.shape(0)}},
      {mx::float32, mx::float32},
      std::make_shared<DenseCCELoss>(
          stream,
          make_forward_fallback(ignore_index, logit_softcap, true, stream),
          ignore_index,
          logit_softcap,
          true),
      {hidden, weight, targets});
  return {outputs[0], outputs[1]};
}

mx::array dense_cce_loss_single(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index,
    float logit_softcap,
    mx::StreamOrDevice s) {
  if (hidden.ndim() != 2 || weight.ndim() != 2 || targets.ndim() != 1) {
    throw std::invalid_argument("dense_cce_loss_single expects hidden/weight 2D and targets 1D.");
  }
  if (hidden.shape(0) != targets.shape(0) || hidden.shape(1) != weight.shape(1)) {
    throw std::invalid_argument("dense_cce_loss_single input shapes do not align.");
  }

  auto stream = mx::to_stream(s);
  return mx::array(
      {hidden.shape(0)},
      mx::float32,
      std::make_shared<DenseCCELoss>(
          stream,
          make_forward_fallback(ignore_index, logit_softcap, false, stream),
          ignore_index,
          logit_softcap,
          false),
      {hidden, weight, targets});
}

std::pair<mx::array, mx::array> dense_cce_loss_custom(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index,
    float logit_softcap,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  auto fun = [ignore_index, logit_softcap, stream](const std::vector<mx::array>& inputs) {
    auto [loss, lse] = dense_cce_loss(
        inputs[0], inputs[1], inputs[2], ignore_index, logit_softcap, stream);
    return std::vector<mx::array>{loss, lse};
  };

  auto fun_vjp =
      [ignore_index, logit_softcap, stream](
          const std::vector<mx::array>& primals,
          const std::vector<mx::array>& cotangents,
          const std::vector<mx::array>& outputs) {
        auto grad_output = cotangents[0];
        if (grad_output.has_primitive()) {
          grad_output.eval();
        }
        auto saved_lse = outputs[1];
        if (saved_lse.has_primitive()) {
          saved_lse.eval();
        }
        auto [grad_hidden, grad_weight] = dense_cce_backward(
            primals[0],
            primals[1],
            primals[2],
            grad_output,
            saved_lse,
            ignore_index,
            logit_softcap,
            stream);
        grad_hidden.eval();
        grad_weight.eval();
        return std::vector<mx::array>{
            grad_hidden,
            grad_weight,
            mx::zeros_like(primals[2], stream)};
      };

  auto wrapped = mx::custom_vjp(fun, fun_vjp);
  auto outs = wrapped({hidden, weight, targets});
  return {outs[0], outs[1]};
}

mx::array dense_cce_loss_single_custom(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index,
    float logit_softcap,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  auto fun = [ignore_index, logit_softcap, stream](const std::vector<mx::array>& inputs) {
    auto [loss, _] = dense_cce_loss(
        inputs[0], inputs[1], inputs[2], ignore_index, logit_softcap, stream);
    return std::vector<mx::array>{loss};
  };

  auto fun_vjp =
      [ignore_index, logit_softcap, stream](
          const std::vector<mx::array>& primals,
          const std::vector<mx::array>& cotangents,
          const std::vector<mx::array>& /* outputs */) {
        auto grad_output = cotangents[0];
        if (grad_output.has_primitive()) {
          grad_output.eval();
        }

        auto logits = mx::matmul(primals[0], mx::transpose(primals[1], stream), stream);
        auto lse = mx::logsumexp(mx::astype(logits, mx::float32, stream), -1, false, stream);
        if (lse.has_primitive()) {
          lse.eval();
        }

        auto [grad_hidden, grad_weight] = dense_cce_backward(
            primals[0],
            primals[1],
            primals[2],
            grad_output,
            lse,
            ignore_index,
            logit_softcap,
            stream);
        grad_hidden.eval();
        grad_weight.eval();
        return std::vector<mx::array>{
            grad_hidden,
            grad_weight,
            mx::zeros_like(primals[2], stream)};
      };

  auto wrapped = mx::custom_vjp(fun, fun_vjp);
  return wrapped({hidden, weight, targets})[0];
}

std::pair<mx::array, mx::array> dense_cce_backward(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    const mx::array& grad_output,
    const mx::array& logsumexp,
    int ignore_index,
    float logit_softcap,
    mx::StreamOrDevice s) {
  if (hidden.ndim() != 2 || weight.ndim() != 2 || targets.ndim() != 1 ||
      grad_output.ndim() != 1 || logsumexp.ndim() != 1) {
    throw std::invalid_argument(
        "dense_cce_backward expects hidden/weight 2D and targets/grad_output/logsumexp 1D.");
  }
  auto stream = mx::to_stream(s);
  auto outputs = mx::array::make_arrays(
      {hidden.shape(), weight.shape()},
      {hidden.dtype(), weight.dtype()},
      std::make_shared<DenseCCELossVJP>(
          stream,
          make_backward_fallback(ignore_index, logit_softcap, true, stream),
          ignore_index,
          logit_softcap,
          true,
          true,
          true),
      {hidden, weight, targets, grad_output, logsumexp});
  return {outputs[0], outputs[1]};
}

void DenseCCELoss::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto results = fallback_(inputs);
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].copy_shared_buffer(results[i]);
  }
}

void DenseCCELoss::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto& hidden_in = inputs[0];
  auto& weight_in = inputs[1];
  auto& targets_in = inputs[2];

  if (!is_supported_compute_dtype(hidden_in.dtype()) || hidden_in.dtype() != weight_in.dtype()) {
    eval_cpu(inputs, outputs);
    return;
  }

  int n = hidden_in.shape(0);
  int h_dim = hidden_in.shape(1);
  int vocab = weight_in.shape(0);

  auto hidden_prepared = materialize_input(hidden_in, hidden_in.dtype(), s);
  auto weight_prepared = materialize_input(weight_in, weight_in.dtype(), s);
  auto targets_prepared = materialize_input(targets_in, mx::int32, s);

  auto& hidden = hidden_prepared.value;
  auto& weight = weight_prepared.value;
  auto& targets = targets_prepared.value;

  auto& loss = outputs[0];
  loss.set_data(mx::allocator::malloc(loss.nbytes()));
  if (output_logsumexp_ && outputs.size() > 1) {
    outputs[1].set_data(mx::allocator::malloc(outputs[1].nbytes()));
  }

  int adaptive_chunk_v = get_adaptive_chunk_v(n, vocab, hidden.itemsize(), 1);
  int num_chunks = (vocab + adaptive_chunk_v - 1) / adaptive_chunk_v;
  int max_chunk_v = std::min(adaptive_chunk_v, vocab);

  mx::array logits_chunk({n, max_chunk_v}, hidden.dtype(), nullptr, {});
  logits_chunk.set_data(mx::allocator::malloc(logits_chunk.nbytes()));

  mx::array running_state({3 * n}, mx::float32, nullptr, {});
  running_state.set_data(mx::allocator::malloc(running_state.nbytes()));

  mx::array running_max({n}, mx::float32, nullptr, {});
  running_max.copy_shared_buffer(running_state, {1}, running_state.flags(), n, 0);

  mx::array running_sum_exp({n}, mx::float32, nullptr, {});
  running_sum_exp.copy_shared_buffer(running_state, {1}, running_state.flags(), n, n);

  mx::array target_logit({n}, mx::float32, nullptr, {});
  target_logit.copy_shared_buffer(running_state, {1}, running_state.flags(), n, 2 * n);

  auto lib = d.get_library("mlx_cce_runtime_native", current_binary_dir());
  auto init_kernel = d.get_kernel("runtime_cce_init_running_values", lib);
  auto lse_kernel = d.get_kernel(
      "runtime_cce_chunk_logsumexp_" + mx::type_to_name(hidden.dtype()), lib);
  auto& compute_encoder = d.get_command_encoder(s.index);

  compute_encoder.set_compute_pipeline_state(init_kernel);
  compute_encoder.set_output_array(running_max, 0);
  compute_encoder.set_output_array(running_sum_exp, 1);
  compute_encoder.set_output_array(target_logit, 2);
  compute_encoder.set_bytes(n, 3);
  compute_encoder.dispatch_threadgroups(MTL::Size((n + 255) / 256, 1, 1), MTL::Size(256, 1, 1));

  constexpr int threads_per_tg = 256;
  constexpr int num_simdgroups = threads_per_tg / 32;
  size_t smem_size = 2 * num_simdgroups * sizeof(float);
  float scale = 1.0f;

  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    int v_start = chunk_idx * max_chunk_v;
    int v_end = std::min(v_start + max_chunk_v, vocab);
    int current_chunk_v = v_end - v_start;

    mx::array logits_view({n, current_chunk_v}, hidden.dtype(), nullptr, {});
    logits_view.copy_shared_buffer(
        logits_chunk,
        {static_cast<int64_t>(current_chunk_v), 1},
        logits_chunk.flags(),
        static_cast<size_t>(n * current_chunk_v),
        0);

    mx::array weight_chunk({current_chunk_v, h_dim}, weight.dtype(), nullptr, {});
    int64_t w_offset = static_cast<int64_t>(v_start) * h_dim;
    weight_chunk.copy_shared_buffer(
        weight,
        {static_cast<int64_t>(h_dim), 1},
        weight.flags(),
        static_cast<size_t>(current_chunk_v * h_dim),
        w_offset);

    launch_regular_gemm(
        s,
        d,
        lib,
        hidden,
        weight_chunk,
        logits_view,
        n,
        current_chunk_v,
        h_dim,
        static_cast<int>(hidden.strides()[hidden.ndim() - 2]),
        static_cast<int>(weight_chunk.strides()[weight_chunk.ndim() - 2]),
        static_cast<int>(logits_view.strides()[logits_view.ndim() - 2]),
        false,
        true);

    compute_encoder.set_compute_pipeline_state(lse_kernel);
    compute_encoder.set_input_array(logits_view, 0);
    compute_encoder.set_input_array(targets, 1);
    compute_encoder.set_output_array(running_max, 2);
    compute_encoder.set_output_array(running_sum_exp, 3);
    compute_encoder.set_output_array(target_logit, 4);
    compute_encoder.set_bytes(n, 5);
    compute_encoder.set_bytes(current_chunk_v, 6);
    compute_encoder.set_bytes(v_start, 7);
    compute_encoder.set_bytes(vocab, 8);
    compute_encoder.set_bytes(logit_softcap_, 9);
    compute_encoder.set_threadgroup_memory_length(smem_size, 0);
    compute_encoder.dispatch_threadgroups(MTL::Size(n, 1, 1), MTL::Size(threads_per_tg, 1, 1));
  }

  if (output_logsumexp_ && outputs.size() > 1) {
    auto final_kernel = d.get_kernel("runtime_cce_finalize_loss_with_lse", lib);
    compute_encoder.set_compute_pipeline_state(final_kernel);
    compute_encoder.set_input_array(running_max, 0);
    compute_encoder.set_input_array(running_sum_exp, 1);
    compute_encoder.set_input_array(target_logit, 2);
    compute_encoder.set_input_array(targets, 3);
    compute_encoder.set_output_array(loss, 4);
    compute_encoder.set_output_array(outputs[1], 5);
    compute_encoder.set_bytes(n, 6);
    compute_encoder.set_bytes(ignore_index_, 7);
    compute_encoder.set_bytes(scale, 8);
  } else {
    auto final_kernel = d.get_kernel("runtime_cce_finalize_loss", lib);
    compute_encoder.set_compute_pipeline_state(final_kernel);
    compute_encoder.set_input_array(running_max, 0);
    compute_encoder.set_input_array(running_sum_exp, 1);
    compute_encoder.set_input_array(target_logit, 2);
    compute_encoder.set_input_array(targets, 3);
    compute_encoder.set_output_array(loss, 4);
    compute_encoder.set_bytes(n, 5);
    compute_encoder.set_bytes(ignore_index_, 6);
    compute_encoder.set_bytes(scale, 7);
  }
  compute_encoder.dispatch_threadgroups(MTL::Size((n + 255) / 256, 1, 1), MTL::Size(256, 1, 1));

  add_if_temporary(d, hidden_prepared, s);
  add_if_temporary(d, weight_prepared, s);
  add_if_temporary(d, targets_prepared, s);
  d.add_temporary(logits_chunk, s.index);
  d.add_temporary(running_state, s.index);
}

std::vector<mx::array> DenseCCELoss::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
  auto s = stream();
  bool has_logsumexp = output_logsumexp_ && outputs.size() > 1;
  std::vector<mx::array> returned_vjps;
  auto backward_inputs = std::vector<mx::array>{primals[0], primals[1], primals[2], cotangents[0]};
  if (has_logsumexp) {
    backward_inputs.push_back(outputs[1]);
  }
  auto backward_results =
      make_backward_fallback(ignore_index_, logit_softcap_, has_logsumexp, s)(backward_inputs);
  for (int arg : argnums) {
    if (arg >= 2) {
      returned_vjps.push_back(mx::zeros_like(primals[arg], s));
    } else {
      returned_vjps.push_back(std::move(backward_results[arg]));
    }
  }
  return returned_vjps;
}

std::vector<mx::Shape> DenseCCELossVJP::output_shapes(const std::vector<mx::array>& inputs) {
  std::vector<mx::Shape> shapes;
  if (need_hidden_) {
    shapes.push_back(inputs[0].shape());
  }
  if (need_weight_) {
    shapes.push_back(inputs[1].shape());
  }
  return shapes;
}

void DenseCCELossVJP::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto results = fallback_(inputs);
  size_t result_idx = 0;
  if (need_hidden_) {
    outputs[result_idx].copy_shared_buffer(results[0]);
    result_idx++;
  }
  if (need_weight_) {
    outputs[result_idx].copy_shared_buffer(results[1]);
  }
}

void DenseCCELossVJP::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto s = stream();
  auto& d = mx::metal::device(s.device);
  auto& hidden_in = inputs[0];
  auto& weight_in = inputs[1];
  auto& targets_in = inputs[2];
  auto& grad_output_in = inputs[3];

  if (!is_supported_compute_dtype(hidden_in.dtype()) || hidden_in.dtype() != weight_in.dtype()) {
    eval_cpu(inputs, outputs);
    return;
  }

  int n = hidden_in.shape(0);
  int h_dim = hidden_in.shape(1);
  int vocab = weight_in.shape(0);

  auto hidden_prepared = materialize_input(hidden_in, hidden_in.dtype(), s);
  auto weight_prepared = materialize_input(weight_in, weight_in.dtype(), s);
  auto targets_prepared = materialize_input(targets_in, mx::int32, s);
  auto grad_output_prepared = materialize_input(grad_output_in, mx::float32, s);
  std::optional<PreparedArray> grad_output_broadcast_prepared;
  if (grad_output_prepared.value.size() == 1) {
    grad_output_broadcast_prepared = materialize_input(
        mx::broadcast_to(grad_output_prepared.value, {n}, s),
        mx::float32,
        s);
  }

  auto& hidden = hidden_prepared.value;
  auto& weight = weight_prepared.value;
  auto& targets = targets_prepared.value;
  auto& grad_output = grad_output_broadcast_prepared.has_value()
      ? grad_output_broadcast_prepared->value
      : grad_output_prepared.value;

  size_t output_idx = 0;
  mx::array* grad_hidden = nullptr;
  mx::array* grad_weight = nullptr;
  if (need_hidden_) {
    grad_hidden = &outputs[output_idx++];
    grad_hidden->set_data(mx::allocator::malloc(grad_hidden->nbytes()));
  }
  if (need_weight_) {
    grad_weight = &outputs[output_idx++];
    grad_weight->set_data(mx::allocator::malloc(grad_weight->nbytes()));
  }

  int adaptive_chunk_v = get_adaptive_chunk_v(
      n,
      vocab,
      hidden.itemsize(),
      (has_logsumexp_ && inputs.size() > 4) ? 2 : 3);
  int num_chunks = (vocab + adaptive_chunk_v - 1) / adaptive_chunk_v;
  int max_chunk_v = std::min(adaptive_chunk_v, vocab);

  auto lib = d.get_library("mlx_cce_runtime_native", current_binary_dir());
  auto& compute_encoder = d.get_command_encoder(s.index);

  mx::array logsumexp({n}, mx::float32, nullptr, {});
  bool owns_logsumexp = false;
  std::optional<PreparedArray> provided_logsumexp;

  if (has_logsumexp_ && inputs.size() > 4) {
    provided_logsumexp = materialize_input(inputs[4], mx::float32, s);
    logsumexp = provided_logsumexp->value;
  } else {
    owns_logsumexp = true;
    logsumexp.set_data(mx::allocator::malloc(logsumexp.nbytes()));

    mx::array running_state({2 * n}, mx::float32, nullptr, {});
    running_state.set_data(mx::allocator::malloc(running_state.nbytes()));

    mx::array running_max({n}, mx::float32, nullptr, {});
    running_max.copy_shared_buffer(running_state, {1}, running_state.flags(), n, 0);

    mx::array running_sum_exp({n}, mx::float32, nullptr, {});
    running_sum_exp.copy_shared_buffer(running_state, {1}, running_state.flags(), n, n);

    mx::array lse_logits_chunk({n, max_chunk_v}, hidden.dtype(), nullptr, {});
    lse_logits_chunk.set_data(mx::allocator::malloc(lse_logits_chunk.nbytes()));

    auto init_kernel = d.get_kernel("runtime_cce_init_running_values", lib);
    compute_encoder.set_compute_pipeline_state(init_kernel);
    compute_encoder.set_output_array(running_max, 0);
    compute_encoder.set_output_array(running_sum_exp, 1);
    compute_encoder.set_output_array(logsumexp, 2);
    compute_encoder.set_bytes(n, 3);
    compute_encoder.dispatch_threadgroups(MTL::Size((n + 255) / 256, 1, 1), MTL::Size(256, 1, 1));

    auto lse_kernel = d.get_kernel(
        "runtime_cce_chunk_logsumexp_" + mx::type_to_name(hidden.dtype()), lib);
    constexpr int lse_threads_per_tg = 256;
    constexpr int lse_num_simdgroups = lse_threads_per_tg / 32;
    size_t lse_smem_size = 2 * lse_num_simdgroups * sizeof(float);

    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      int v_start = chunk_idx * max_chunk_v;
      int v_end = std::min(v_start + max_chunk_v, vocab);
      int current_chunk_v = v_end - v_start;

      mx::array logits_view({n, current_chunk_v}, hidden.dtype(), nullptr, {});
      logits_view.copy_shared_buffer(
          lse_logits_chunk,
          {static_cast<int64_t>(current_chunk_v), 1},
          lse_logits_chunk.flags(),
          static_cast<size_t>(n * current_chunk_v),
          0);

      mx::array weight_chunk({current_chunk_v, h_dim}, weight.dtype(), nullptr, {});
      int64_t w_offset = static_cast<int64_t>(v_start) * h_dim;
      weight_chunk.copy_shared_buffer(
          weight,
          {static_cast<int64_t>(h_dim), 1},
          weight.flags(),
          static_cast<size_t>(current_chunk_v * h_dim),
          w_offset);

      launch_regular_gemm(
          s,
          d,
          lib,
          hidden,
          weight_chunk,
          logits_view,
          n,
          current_chunk_v,
          h_dim,
          static_cast<int>(hidden.strides()[hidden.ndim() - 2]),
          static_cast<int>(weight_chunk.strides()[weight_chunk.ndim() - 2]),
          static_cast<int>(logits_view.strides()[logits_view.ndim() - 2]),
          false,
          true);

      compute_encoder.set_compute_pipeline_state(lse_kernel);
      compute_encoder.set_input_array(logits_view, 0);
      compute_encoder.set_input_array(targets, 1);
      compute_encoder.set_output_array(running_max, 2);
      compute_encoder.set_output_array(running_sum_exp, 3);
      compute_encoder.set_output_array(logsumexp, 4);
      compute_encoder.set_bytes(n, 5);
      compute_encoder.set_bytes(current_chunk_v, 6);
      compute_encoder.set_bytes(v_start, 7);
      compute_encoder.set_bytes(vocab, 8);
      compute_encoder.set_bytes(logit_softcap_, 9);
      compute_encoder.set_threadgroup_memory_length(lse_smem_size, 0);
      compute_encoder.dispatch_threadgroups(
          MTL::Size(n, 1, 1),
          MTL::Size(lse_threads_per_tg, 1, 1));
    }

    auto finalize_lse_kernel = d.get_kernel("runtime_cce_finalize_lse", lib);
    compute_encoder.set_compute_pipeline_state(finalize_lse_kernel);
    compute_encoder.set_input_array(running_max, 0);
    compute_encoder.set_input_array(running_sum_exp, 1);
    compute_encoder.set_output_array(logsumexp, 2);
    compute_encoder.set_bytes(n, 3);
    compute_encoder.dispatch_threadgroups(MTL::Size((n + 255) / 256, 1, 1), MTL::Size(256, 1, 1));

    d.add_temporary(running_state, s.index);
    d.add_temporary(lse_logits_chunk, s.index);
  }

  mx::array logits_chunk({n, max_chunk_v}, hidden.dtype(), nullptr, {});
  logits_chunk.set_data(mx::allocator::malloc(logits_chunk.nbytes()));

  mx::array d_logits_chunk({n, max_chunk_v}, hidden.dtype(), nullptr, {});
  d_logits_chunk.set_data(mx::allocator::malloc(d_logits_chunk.nbytes()));

  auto d_logits_kernel = d.get_kernel(
      "runtime_cce_compute_d_logits_" + mx::type_to_name(hidden.dtype()), lib);

  for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    int v_start = chunk_idx * max_chunk_v;
    int v_end = std::min(v_start + max_chunk_v, vocab);
    int current_chunk_v = v_end - v_start;

    mx::array logits_view({n, current_chunk_v}, hidden.dtype(), nullptr, {});
    logits_view.copy_shared_buffer(
        logits_chunk,
        {static_cast<int64_t>(current_chunk_v), 1},
        logits_chunk.flags(),
        static_cast<size_t>(n * current_chunk_v),
        0);

    mx::array d_logits_view({n, current_chunk_v}, hidden.dtype(), nullptr, {});
    d_logits_view.copy_shared_buffer(
        d_logits_chunk,
        {static_cast<int64_t>(current_chunk_v), 1},
        d_logits_chunk.flags(),
        static_cast<size_t>(n * current_chunk_v),
        0);

    mx::array weight_chunk({current_chunk_v, h_dim}, weight.dtype(), nullptr, {});
    int64_t w_offset = static_cast<int64_t>(v_start) * h_dim;
    weight_chunk.copy_shared_buffer(
        weight,
        {static_cast<int64_t>(h_dim), 1},
        weight.flags(),
        static_cast<size_t>(current_chunk_v * h_dim),
        w_offset);

    launch_regular_gemm(
        s,
        d,
        lib,
        hidden,
        weight_chunk,
        logits_view,
        n,
        current_chunk_v,
        h_dim,
        static_cast<int>(hidden.strides()[hidden.ndim() - 2]),
        static_cast<int>(weight_chunk.strides()[weight_chunk.ndim() - 2]),
        static_cast<int>(logits_view.strides()[logits_view.ndim() - 2]),
        false,
        true);

    compute_encoder.set_compute_pipeline_state(d_logits_kernel);
    compute_encoder.set_input_array(logits_view, 0);
    compute_encoder.set_input_array(logsumexp, 1);
    compute_encoder.set_input_array(targets, 2);
    compute_encoder.set_input_array(grad_output, 3);
    compute_encoder.set_output_array(d_logits_view, 4);
    compute_encoder.set_bytes(n, 5);
    compute_encoder.set_bytes(current_chunk_v, 6);
    compute_encoder.set_bytes(v_start, 7);
    compute_encoder.set_bytes(vocab, 8);
    compute_encoder.set_bytes(logit_softcap_, 9);
    compute_encoder.set_bytes(ignore_index_, 10);
    int total_threads = (n * current_chunk_v + 3) / 4;
    compute_encoder.dispatch_threadgroups(
        MTL::Size((total_threads + 255) / 256, 1, 1),
        MTL::Size(256, 1, 1));

    if (need_hidden_ && chunk_idx == 0) {
      launch_regular_gemm(
          s,
          d,
          lib,
          d_logits_view,
          weight_chunk,
          *grad_hidden,
          n,
          h_dim,
          current_chunk_v,
          static_cast<int>(d_logits_view.strides()[d_logits_view.ndim() - 2]),
          static_cast<int>(weight_chunk.strides()[weight_chunk.ndim() - 2]),
          static_cast<int>(grad_hidden->strides()[grad_hidden->ndim() - 2]),
          false,
          false);
    } else if (need_hidden_) {
      launch_regular_gemm(
          s,
          d,
          lib,
          d_logits_view,
          weight_chunk,
          *grad_hidden,
          n,
          h_dim,
          current_chunk_v,
          static_cast<int>(d_logits_view.strides()[d_logits_view.ndim() - 2]),
          static_cast<int>(weight_chunk.strides()[weight_chunk.ndim() - 2]),
          static_cast<int>(grad_hidden->strides()[grad_hidden->ndim() - 2]),
          false,
          false,
          std::cref(*grad_hidden),
          1.0f,
          1.0f);
    }

    if (need_weight_) {
      mx::array grad_weight_chunk({current_chunk_v, h_dim}, weight.dtype(), nullptr, {});
      int64_t gw_offset = static_cast<int64_t>(v_start) * h_dim;
      grad_weight_chunk.copy_shared_buffer(
          *grad_weight,
          {static_cast<int64_t>(h_dim), 1},
          grad_weight->flags(),
          static_cast<size_t>(current_chunk_v * h_dim),
          gw_offset);

      launch_regular_gemm(
          s,
          d,
          lib,
          d_logits_view,
          hidden,
          grad_weight_chunk,
          current_chunk_v,
          h_dim,
          n,
          static_cast<int>(d_logits_view.strides()[d_logits_view.ndim() - 2]),
          static_cast<int>(hidden.strides()[hidden.ndim() - 2]),
          static_cast<int>(grad_weight_chunk.strides()[grad_weight_chunk.ndim() - 2]),
          true,
          false);
    }
  }

  add_if_temporary(d, hidden_prepared, s);
  add_if_temporary(d, weight_prepared, s);
  add_if_temporary(d, targets_prepared, s);
  add_if_temporary(d, grad_output_prepared, s);
  if (grad_output_broadcast_prepared.has_value()) {
    add_if_temporary(d, *grad_output_broadcast_prepared, s);
  }
  if (provided_logsumexp.has_value()) {
    add_if_temporary(d, *provided_logsumexp, s);
  } else if (owns_logsumexp) {
    d.add_temporary(logsumexp, s.index);
  }
  d.add_temporary(logits_chunk, s.index);
  d.add_temporary(d_logits_chunk, s.index);
}

bool DenseCCELoss::is_equivalent(const mx::Primitive& other) const {
  const auto& rhs = static_cast<const DenseCCELoss&>(other);
  return ignore_index_ == rhs.ignore_index_ && logit_softcap_ == rhs.logit_softcap_ &&
      output_logsumexp_ == rhs.output_logsumexp_;
}

bool DenseCCELossVJP::is_equivalent(const mx::Primitive& other) const {
  const auto& rhs = static_cast<const DenseCCELossVJP&>(other);
  return ignore_index_ == rhs.ignore_index_ && logit_softcap_ == rhs.logit_softcap_ &&
      has_logsumexp_ == rhs.has_logsumexp_ && need_hidden_ == rhs.need_hidden_ &&
      need_weight_ == rhs.need_weight_;
}

} // namespace mlx_cce_runtime_ext
