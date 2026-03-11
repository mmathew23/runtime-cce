import unittest

import mlx.core as mx

from mlx_cce_runtime import (
    install_mlx_fast_cce_loss,
    make_chunked_cross_entropy_loss,
)
from mlx_cce_runtime.runtime_cce import _native_ext


def _reference_loss(hidden, weight, targets, *, ignore_index=-100, logit_softcap=0.0):
    logits = hidden @ weight.T
    if logit_softcap > 0.0:
        softcap = mx.array(logit_softcap, dtype=mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

    lse = mx.logsumexp(logits, axis=-1)
    local_targets = mx.clip(targets, 0, weight.shape[0] - 1)
    target_logits = mx.take_along_axis(logits, mx.expand_dims(local_targets, -1), axis=1).squeeze(-1)
    valid = targets != ignore_index
    return mx.where(valid, lse - target_logits, mx.zeros_like(lse))


def _reference_grads(hidden, weight, targets, *, ignore_index=-100):
    logits = hidden @ weight.T
    probs = mx.softmax(logits, axis=-1)
    one_hot = (mx.arange(weight.shape[0], dtype=mx.int32)[None, :] == targets[:, None]).astype(
        probs.dtype
    )
    valid = (targets != ignore_index).astype(probs.dtype)[:, None]
    d_logits = (probs - one_hot) * valid
    return d_logits @ weight, d_logits.T @ hidden


class TestRuntimeCCE(unittest.TestCase):
    def setUp(self):
        self.hidden = mx.array(
            [
                [0.1, -0.2, 0.3, 0.5],
                [-0.4, 0.2, 0.1, -0.3],
                [0.7, -0.1, 0.2, -0.5],
            ],
            dtype=mx.float32,
        )
        self.weight = mx.array(
            [
                [0.2, -0.1, 0.4, 0.3],
                [-0.3, 0.6, -0.2, 0.1],
                [0.5, 0.2, -0.4, 0.7],
                [-0.6, 0.1, 0.3, -0.2],
                [0.4, -0.5, 0.2, 0.6],
            ],
            dtype=mx.float32,
        )
        self.targets = mx.array([1, 3, -100], dtype=mx.int32)

    def test_dense_forward_matches_reference(self):
        runtime_cce, _ = make_chunked_cross_entropy_loss(ignore_index=-100, chunk_size=2)
        actual = runtime_cce(self.hidden, self.weight, self.targets)
        expected = _reference_loss(self.hidden, self.weight, self.targets)
        mx.eval(actual, expected)
        self.assertTrue(mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item())

    def test_dense_gradients_match_reference(self):
        if _native_ext is None:
            self.skipTest("native extension not built")

        logits = self.hidden @ self.weight.T
        lse = mx.logsumexp(logits, axis=-1).astype(mx.float32)
        grad_output = mx.ones((self.hidden.shape[0],), dtype=mx.float32)
        runtime_hidden_grad, runtime_weight_grad = _native_ext.dense_cce_backward(
            self.hidden,
            self.weight,
            self.targets,
            grad_output,
            lse,
            ignore_index=-100,
            logit_softcap=0.0,
        )
        ref_hidden_grad, ref_weight_grad = _reference_grads(self.hidden, self.weight, self.targets)
        mx.eval(runtime_hidden_grad, runtime_weight_grad, ref_hidden_grad, ref_weight_grad)

        self.assertTrue(mx.allclose(runtime_hidden_grad, ref_hidden_grad, atol=1e-5, rtol=1e-5).item())
        self.assertTrue(mx.allclose(runtime_weight_grad, ref_weight_grad, atol=1e-5, rtol=1e-5).item())

    def test_quantized_path_runs_and_produces_hidden_grad(self):
        runtime_cce, _ = make_chunked_cross_entropy_loss(
            ignore_index=-100,
            chunk_size=2,
            quantized=True,
            group_size=32,
            bits=4,
        )
        weight = mx.arange(64 * 32, dtype=mx.float32).reshape(64, 32) / 100.0
        hidden = mx.arange(3 * 32, dtype=mx.float32).reshape(3, 32) / 50.0
        targets = mx.array([1, 7, -100], dtype=mx.int32)
        q_weight, scales, biases = mx.quantize(weight, group_size=32, bits=4)

        def total(hidden_input):
            return runtime_cce(hidden_input, q_weight, scales, biases, targets).sum()

        grad_fn = mx.grad(total)
        hidden_grad = grad_fn(hidden)
        losses = runtime_cce(hidden, q_weight, scales, biases, targets)
        mx.eval(losses, hidden_grad)

        self.assertEqual(losses.shape, targets.shape)
        self.assertGreater(float(mx.abs(hidden_grad).max()), 0.0)

    def test_install_is_idempotent(self):
        first = install_mlx_fast_cce_loss(override=True)
        second = install_mlx_fast_cce_loss()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
