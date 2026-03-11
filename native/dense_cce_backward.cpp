// Copyright © 2026

#include <dlfcn.h>

#include <filesystem>
#include <optional>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"

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

int get_adaptive_chunk_v(int n, int v) {
  if (n <= 2048) {
    return std::min(MAX_CHUNK_V, v);
  }

  constexpr size_t BYTES_PER_ELEMENT = 4;
  size_t system_memory = 64ULL * 1024 * 1024 * 1024;
#if defined(__APPLE__)
  size_t size = sizeof(system_memory);
  if (sysctlbyname("hw.memsize", &system_memory, &size, nullptr, 0) != 0) {
    system_memory = 64ULL * 1024 * 1024 * 1024;
  }
#endif

  size_t chunk_budget = system_memory / 200;
  int max_chunk_from_memory =
      static_cast<int>(chunk_budget / (static_cast<size_t>(n) * BYTES_PER_ELEMENT));
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

void eval_generic(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs,
    mx::Stream s,
    int ignore_index,
    float logit_softcap) {
  auto& hidden = inputs[0];
  auto& weight = inputs[1];
  auto& targets = inputs[2];
  auto& grad_output = inputs[3];
  auto& lse = inputs[4];

  auto compute_type = hidden.dtype();
  auto logits = mx::matmul(
      mx::astype(hidden, compute_type, s),
      mx::transpose(mx::astype(weight, compute_type, s), s),
      s);
  logits = mx::astype(logits, mx::float32, s);
  logits = apply_softcap(logits, logit_softcap, s);

  int n = hidden.shape(0);
  int v = weight.shape(0);
  auto softmax_probs = mx::exp(mx::subtract(logits, mx::expand_dims(lse, -1, s), s), s);

  auto one_hot_arr = mx::zeros({n, v}, mx::float32, s);
  auto valid_mask = mx::logical_and(
      mx::greater_equal(targets, mx::array(0, mx::int32), s),
      mx::less(targets, mx::array(v, mx::int32), s),
      s);
  valid_mask = mx::logical_and(
      valid_mask,
      mx::not_equal(targets, mx::array(ignore_index, mx::int32), s),
      s);
  auto t_clamped =
      mx::clip(targets, mx::array(0, mx::int32), mx::array(v - 1, mx::int32), s);
  auto row_idx = mx::expand_dims(mx::arange(n, mx::int32, s), -1, s);
  auto col_idx = mx::expand_dims(t_clamped, -1, s);
  auto updates = mx::expand_dims(
      mx::where(
          valid_mask,
          mx::ones({n}, mx::float32, s),
          mx::zeros({n}, mx::float32, s),
          s),
      -1,
      s);
  one_hot_arr = mx::scatter(one_hot_arr, {row_idx, col_idx}, updates, {0, 1}, s);

  auto grad_logits = mx::subtract(softmax_probs, one_hot_arr, s);
  grad_logits = mx::multiply(grad_logits, mx::expand_dims(grad_output, -1, s), s);
  auto ignore_mask = mx::equal(targets, mx::array(ignore_index, mx::int32), s);
  grad_logits = mx::where(
      mx::expand_dims(ignore_mask, -1, s),
      mx::zeros_like(grad_logits, s),
      grad_logits,
      s);

  outputs[0].copy_shared_buffer(
      mx::matmul(mx::astype(grad_logits, compute_type, s), mx::astype(weight, compute_type, s), s));
  outputs[1].copy_shared_buffer(
      mx::matmul(
          mx::transpose(mx::astype(grad_logits, compute_type, s), s),
          mx::astype(hidden, compute_type, s),
          s));
}

bool is_supported_compute_dtype(mx::Dtype dtype) {
  return dtype == mx::float16 || dtype == mx::bfloat16 || dtype == mx::float32;
}

bool is_fast_path_eligible(const std::vector<mx::array>& inputs) {
  const auto& hidden = inputs[0];
  const auto& weight = inputs[1];
  return is_supported_compute_dtype(hidden.dtype()) && hidden.dtype() == weight.dtype();
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

} // namespace

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
  if (hidden.shape(0) != targets.shape(0) || hidden.shape(0) != grad_output.shape(0) ||
      hidden.shape(0) != logsumexp.shape(0)) {
    throw std::invalid_argument("dense_cce_backward input sizes do not align.");
  }
  if (hidden.shape(1) != weight.shape(1)) {
    throw std::invalid_argument("dense_cce_backward hidden dim must match weight dim.");
  }

  auto stream = mx::to_stream(s);
  auto outputs = mx::array::make_arrays(
      {hidden.shape(), weight.shape()},
      {hidden.dtype(), weight.dtype()},
      std::make_shared<DenseCCEBackward>(stream, ignore_index, logit_softcap),
      {hidden, weight, targets, grad_output, logsumexp});
  return {outputs[0], outputs[1]};
}

void DenseCCEBackward::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  eval_generic(inputs, outputs, stream(), ignore_index_, logit_softcap_);
}

void DenseCCEBackward::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto s = stream();
  if (!is_fast_path_eligible(inputs)) {
    eval_generic(inputs, outputs, s, ignore_index_, logit_softcap_);
    return;
  }

  auto materialize = [&s](const mx::array& arr, mx::Dtype dtype) {
    mx::array out = (arr.dtype() == dtype) ? arr : mx::astype(arr, dtype, s);
    if (!out.flags().row_contiguous) {
      out = mx::contiguous(out, false, s);
    }
    if (out.has_primitive()) {
      out.eval();
    }
    return out;
  };

  auto hidden = materialize(inputs[0], inputs[0].dtype());
  auto weight = materialize(inputs[1], inputs[1].dtype());
  auto targets = materialize(inputs[2], mx::int32);
  auto grad_output = materialize(inputs[3], mx::float32);
  auto lse = materialize(inputs[4], mx::float32);

  auto& d = mx::metal::device(s.device);

  int n = hidden.shape(0);
  int h_dim = hidden.shape(1);
  int vocab = weight.shape(0);
  int adaptive_chunk_v = get_adaptive_chunk_v(n, vocab);
  int num_chunks = (vocab + adaptive_chunk_v - 1) / adaptive_chunk_v;
  int max_chunk_v = std::min(adaptive_chunk_v, vocab);

  mx::array& grad_hidden = outputs[0];
  mx::array& grad_weight = outputs[1];
  grad_hidden.set_data(mx::allocator::malloc(grad_hidden.nbytes()));
  grad_weight.set_data(mx::allocator::malloc(grad_weight.nbytes()));

  mx::array logits_chunk({n, max_chunk_v}, hidden.dtype(), nullptr, {});
  logits_chunk.set_data(mx::allocator::malloc(logits_chunk.nbytes()));

  mx::array d_logits_chunk({n, max_chunk_v}, hidden.dtype(), nullptr, {});
  d_logits_chunk.set_data(mx::allocator::malloc(d_logits_chunk.nbytes()));

  auto lib = d.get_library("mlx_cce_runtime_native", current_binary_dir());
  auto d_logits_kernel =
      d.get_kernel("runtime_cce_compute_d_logits_" + mx::type_to_name(hidden.dtype()), lib);

  for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
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

    auto& compute_encoder = d.get_command_encoder(s.index);
    compute_encoder.set_compute_pipeline_state(d_logits_kernel);
    compute_encoder.set_input_array(logits_view, 0);
    compute_encoder.set_input_array(lse, 1);
    compute_encoder.set_input_array(targets, 2);
    compute_encoder.set_input_array(grad_output, 3);
    compute_encoder.set_output_array(d_logits_view, 4);
    compute_encoder.set_bytes(n, 5);
    compute_encoder.set_bytes(current_chunk_v, 6);
    compute_encoder.set_bytes(v_start, 7);
    compute_encoder.set_bytes(vocab, 8);
    compute_encoder.set_bytes(logit_softcap_, 9);
    compute_encoder.set_bytes(ignore_index_, 10);

    constexpr int n_reads = 4;
    int total_elements = n * current_chunk_v;
    int total_threads = (total_elements + n_reads - 1) / n_reads;
    int threads_per_tg = 256;
    int num_tgs = (total_threads + threads_per_tg - 1) / threads_per_tg;
    compute_encoder.dispatch_threadgroups(
        MTL::Size(num_tgs, 1, 1),
        MTL::Size(threads_per_tg, 1, 1));

    if (chunk_idx == 0) {
      launch_regular_gemm(
          s,
          d,
          lib,
          d_logits_view,
          weight_chunk,
          grad_hidden,
          n,
          h_dim,
          current_chunk_v,
          static_cast<int>(d_logits_view.strides()[d_logits_view.ndim() - 2]),
          static_cast<int>(weight_chunk.strides()[weight_chunk.ndim() - 2]),
          static_cast<int>(grad_hidden.strides()[grad_hidden.ndim() - 2]),
          false,
          false);
    } else {
      launch_regular_gemm(
          s,
          d,
          lib,
          d_logits_view,
          weight_chunk,
          grad_hidden,
          n,
          h_dim,
          current_chunk_v,
          static_cast<int>(d_logits_view.strides()[d_logits_view.ndim() - 2]),
          static_cast<int>(weight_chunk.strides()[weight_chunk.ndim() - 2]),
          static_cast<int>(grad_hidden.strides()[grad_hidden.ndim() - 2]),
          false,
          false,
          std::cref(grad_hidden),
          1.0f,
          1.0f);
    }

    mx::array grad_weight_chunk({current_chunk_v, h_dim}, weight.dtype(), nullptr, {});
    int64_t gw_offset = static_cast<int64_t>(v_start) * h_dim;
    grad_weight_chunk.copy_shared_buffer(
        grad_weight,
        {static_cast<int64_t>(h_dim), 1},
        grad_weight.flags(),
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

  d.add_temporary(logits_chunk, s.index);
  d.add_temporary(d_logits_chunk, s.index);
}

bool DenseCCEBackward::is_equivalent(const mx::Primitive& other) const {
  const auto& rhs = static_cast<const DenseCCEBackward&>(other);
  return ignore_index_ == rhs.ignore_index_ && logit_softcap_ == rhs.logit_softcap_;
}

} // namespace mlx_cce_runtime_ext
