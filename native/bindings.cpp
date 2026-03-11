// Copyright © 2026

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>

#include "dense_cce_backward.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native helpers for mlx-cce-runtime";

  m.def(
      "dense_cce_backward",
      &mlx_cce_runtime_ext::dense_cce_backward,
      "hidden"_a,
      "weight"_a,
      "targets"_a,
      "grad_output"_a,
      "logsumexp"_a,
      nb::kw_only(),
      "ignore_index"_a = -100,
      "logit_softcap"_a = 0.0f,
      "stream"_a = nb::none());
}
