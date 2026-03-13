import argparse
import json

import mlx.core as mx
import mlx.nn as nn

from mlx_cce_runtime import LEGACY_RUNTIME_VARIANT_ALIASES, SUPPORTED_RUNTIME_VARIANTS


def bytes_per_dtype(dtype):
    return mx.array(0, dtype=dtype).itemsize


def snapshot(label):
    return {
        "label": label,
        "active_gb": mx.get_active_memory() / 1e9,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "cache_gb": mx.get_cache_memory() / 1e9,
    }


def estimate_runtime_dense(args):
    from mlx_cce_runtime.runtime_cce import _resolve_chunk_size

    n_tokens = args.tokens
    hidden_size = args.hidden_size
    vocab_size = args.vocab_size
    compute_bytes = bytes_per_dtype(mx.bfloat16)
    chunk_v = _resolve_chunk_size(
        args.chunk_size,
        n_tokens,
        vocab_size,
        bytes_per_element=compute_bytes,
    )
    dlogits_bytes = n_tokens * chunk_v * 4
    logits_bytes = n_tokens * chunk_v * compute_bytes
    grad_weight_chunk_bytes = chunk_v * hidden_size * compute_bytes
    grad_weight_bytes = vocab_size * hidden_size * compute_bytes
    grad_hidden_bytes = n_tokens * hidden_size * compute_bytes
    return {
        "resolved_chunk_size": chunk_v,
        "logits_chunk_gb": logits_bytes / 1e9,
        "dlogits_chunk_gb": dlogits_bytes / 1e9,
        "grad_weight_chunk_gb": grad_weight_chunk_bytes / 1e9,
        "grad_weight_full_gb": grad_weight_bytes / 1e9,
        "grad_hidden_full_gb": grad_hidden_bytes / 1e9,
        "slice_update_output_gb": grad_weight_bytes / 1e9,
    }


def estimate_baseline_dense(args):
    compute_bytes = bytes_per_dtype(mx.bfloat16)
    logits_bytes = args.tokens * args.vocab_size * compute_bytes
    return {
        "full_logits_gb": logits_bytes / 1e9,
    }


def make_loss(impl, chunk_size, runtime_variant):
    if impl == "baseline":
        def loss_fn(hidden, weight, targets):
            logits = hidden @ weight.T
            per_token = nn.losses.cross_entropy(logits, targets)
            return per_token.astype(mx.float32).mean()

        return loss_fn

    if impl == "runtime":
        from mlx_cce_runtime import make_chunked_cross_entropy_loss

        runtime_cce, _ = make_chunked_cross_entropy_loss(
            ignore_index=-100,
            logit_softcap=0.0,
            chunk_size=chunk_size,
            runtime_variant=runtime_variant,
        )

        def loss_fn(hidden, weight, targets):
            per_token = runtime_cce(hidden, weight, targets)
            return per_token.astype(mx.float32).mean()

        return loss_fn

    if impl == "fork-fast":
        if not hasattr(mx.fast, "cce_loss"):
            raise RuntimeError("mx.fast.cce_loss is not available in this environment.")

        def loss_fn(hidden, weight, targets):
            per_token = mx.fast.cce_loss(
                hidden,
                weight,
                targets,
                ignore_index=-100,
                logit_softcap=0.0,
            )
            return per_token.astype(mx.float32).mean()

        return loss_fn

    raise ValueError(f"Unsupported impl: {impl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["baseline", "runtime", "fork-fast"], required=True)
    parser.add_argument("--grads", choices=["hidden", "full"], default="full")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument(
        "--runtime-variant",
        choices=[
            *SUPPORTED_RUNTIME_VARIANTS,
            *LEGACY_RUNTIME_VARIANT_ALIASES,
            "native",
            "native_bridge",
            "native_custom_vjp",
        ],
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    mx.clear_cache()
    mx.reset_peak_memory()

    hidden = mx.random.normal((args.tokens, args.hidden_size), dtype=mx.bfloat16)
    weight = mx.random.normal((args.vocab_size, args.hidden_size), dtype=mx.bfloat16)
    targets = mx.random.randint(
        0,
        args.vocab_size,
        shape=(args.tokens,),
        dtype=mx.int32,
    )
    mx.eval(hidden, weight, targets)

    result = {
        "impl": args.impl,
        "grads": args.grads,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "vocab_size": args.vocab_size,
        "dtype": "bfloat16",
        "runtime_variant": args.runtime_variant,
        "snapshots": [snapshot("inputs_eval")],
    }

    if args.impl == "runtime":
        result["runtime_estimate"] = estimate_runtime_dense(args)
    if args.impl == "baseline":
        result["baseline_estimate"] = estimate_baseline_dense(args)

    loss_fn = make_loss(args.impl, args.chunk_size, args.runtime_variant)
    loss = loss_fn(hidden, weight, targets)
    mx.eval(loss)
    result["loss"] = float(loss.item())
    result["snapshots"].append(snapshot("forward_eval"))

    if args.grads == "hidden":
        grad_fn = mx.value_and_grad(loss_fn, argnums=0)
        value, grad_hidden = grad_fn(hidden, weight, targets)
        mx.eval(value, grad_hidden)
        result["snapshots"].append(snapshot("backward_eval"))
        result["grad_shapes"] = {"hidden": list(grad_hidden.shape)}
    else:
        grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
        value, grads = grad_fn(hidden, weight, targets)
        grad_hidden, grad_weight = grads
        mx.eval(value, grad_hidden, grad_weight)
        result["snapshots"].append(snapshot("backward_eval"))
        result["grad_shapes"] = {
            "hidden": list(grad_hidden.shape),
            "weight": list(grad_weight.shape),
        }

    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
