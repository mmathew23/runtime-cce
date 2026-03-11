// Copyright © 2026

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mlx_cce_runtime_ext {

std::pair<mx::array, mx::array> dense_cce_backward(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    const mx::array& grad_output,
    const mx::array& logsumexp,
    int ignore_index = -100,
    float logit_softcap = 0.0f,
    mx::StreamOrDevice s = {});

class DenseCCEBackward : public mx::Primitive {
 public:
  explicit DenseCCEBackward(mx::Stream stream, int ignore_index, float logit_softcap)
      : mx::Primitive(stream),
        ignore_index_(ignore_index),
        logit_softcap_(logit_softcap) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "DenseCCEBackward";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape(), inputs[1].shape()};
  }

 private:
  int ignore_index_;
  float logit_softcap_;
};

} // namespace mlx_cce_runtime_ext
