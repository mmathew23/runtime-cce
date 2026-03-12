// Copyright © 2026

#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"

namespace mx = mlx::core;

namespace mlx_cce_runtime_ext {

std::pair<mx::array, mx::array> dense_cce_loss(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index = -100,
    float logit_softcap = 0.0f,
    mx::StreamOrDevice s = {});

mx::array dense_cce_loss_single(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index = -100,
    float logit_softcap = 0.0f,
    mx::StreamOrDevice s = {});

std::pair<mx::array, mx::array> dense_cce_loss_custom(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index = -100,
    float logit_softcap = 0.0f,
    mx::StreamOrDevice s = {});

mx::array dense_cce_loss_single_custom(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    int ignore_index = -100,
    float logit_softcap = 0.0f,
    mx::StreamOrDevice s = {});

std::pair<mx::array, mx::array> dense_cce_backward(
    const mx::array& hidden,
    const mx::array& weight,
    const mx::array& targets,
    const mx::array& grad_output,
    const mx::array& logsumexp,
    int ignore_index = -100,
    float logit_softcap = 0.0f,
    mx::StreamOrDevice s = {});

class DenseCCELoss : public mx::Primitive {
 public:
  DenseCCELoss(
      mx::Stream stream,
      std::function<std::vector<mx::array>(std::vector<mx::array>)> fallback,
      int ignore_index,
      float logit_softcap,
      bool output_logsumexp)
      : mx::Primitive(stream),
        fallback_(std::move(fallback)),
        ignore_index_(ignore_index),
        logit_softcap_(logit_softcap),
        output_logsumexp_(output_logsumexp) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "DenseCCELoss";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

  auto state() const {
    return std::make_tuple(nullptr, ignore_index_, logit_softcap_, output_logsumexp_);
  }

 private:
  std::function<std::vector<mx::array>(std::vector<mx::array>)> fallback_;
  int ignore_index_;
  float logit_softcap_;
  bool output_logsumexp_;
};

class DenseCCELossVJP : public mx::Primitive {
 public:
  DenseCCELossVJP(
      mx::Stream stream,
      std::function<std::vector<mx::array>(std::vector<mx::array>)> fallback,
      int ignore_index,
      float logit_softcap,
      bool has_logsumexp,
      bool need_hidden,
      bool need_weight)
      : mx::Primitive(stream),
        fallback_(std::move(fallback)),
        ignore_index_(ignore_index),
        logit_softcap_(logit_softcap),
        has_logsumexp_(has_logsumexp),
        need_hidden_(need_hidden),
        need_weight_(need_weight) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "DenseCCELossVJP";
  }

  bool is_equivalent(const mx::Primitive& other) const override;

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override;

  auto state() const {
    return std::make_tuple(
        nullptr, ignore_index_, logit_softcap_, has_logsumexp_, need_hidden_, need_weight_);
  }

 private:
  std::function<std::vector<mx::array>(std::vector<mx::array>)> fallback_;
  int ignore_index_;
  float logit_softcap_;
  bool has_logsumexp_;
  bool need_hidden_;
  bool need_weight_;
};

} // namespace mlx_cce_runtime_ext
