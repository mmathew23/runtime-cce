import json
import os
import time

import mlx
import mlx.core as mx
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth_zoo.mlx_trainer import MLXTrainer, MLXTrainingConfig
from unsloth_zoo.mlx_utils import create_batches, has_cce_kernel


MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-bf16"
SEQ_LEN = 8192
BATCH_SIZE = 1
N_STEPS = 1
RUNTIME_VARIANT = os.environ.get("MLX_CCE_RUNTIME_VARIANT", "clean")


def make_dataset():
    text = (
        "The quick brown fox jumps over the lazy dog while a unicorn writes Metal kernels "
        "for chunked cross entropy. " * 300
    )
    return Dataset.from_dict({"text": [text for _ in range(4)]})


def extract_loss(stats):
    if isinstance(stats, dict):
        value = stats.get("train_loss")
        if isinstance(value, list):
            return float(value[-1]) if value else None
        if value is not None:
            return float(value)
    if isinstance(stats, (int, float)):
        return float(stats)
    return None


def make_batches(dataset, tokenizer):
    return create_batches(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_seq_length=SEQ_LEN,
        num_batches=N_STEPS,
        seed=3407,
        dataset_text_field="text",
        formatting_func=lambda x: x["text"],
    )


def run(use_cce, model, tokenizer, dataset):
    model.value_and_grad = None
    mx.clear_cache()
    mx.reset_peak_memory()
    batches = make_batches(dataset, tokenizer)
    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=MLXTrainingConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            max_steps=N_STEPS,
            logging_steps=1,
            learning_rate=1e-5,
            save_steps=0,
            report_to="none",
            max_seq_length=SEQ_LEN,
            dataset_text_field="text",
            use_cce=use_cce,
        ),
    )
    trainer._batches = batches
    start = time.time()
    stats = trainer.train()
    end = time.time()
    return {
        "ms_per_step": (end - start) * 1000.0 / N_STEPS,
        "peak_gb": mx.get_peak_memory() / 1e9,
        "final_loss": extract_loss(stats),
    }


def main():
    dataset = make_dataset()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=SEQ_LEN,
        load_in_4bit=False,
        full_finetuning=True,
    )

    result = {
        "model": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "runtime_variant": RUNTIME_VARIANT,
        "mlx_path": list(getattr(mlx, "__path__", [])),
        "core_file": mx.__file__,
        "has_cce_kernel_before": hasattr(mx.fast, "cce_loss"),
        "has_cce_kernel_probe": has_cce_kernel(),
        "has_cce_kernel_after": hasattr(mx.fast, "cce_loss"),
    }
    result["baseline"] = run(False, model, tokenizer, dataset)
    result["cce"] = run(True, model, tokenizer, dataset)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
