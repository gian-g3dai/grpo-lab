"""GRPO training entry point.

Usage:
    python -m grpo_lab.train --config configs/qwen2.5-0.5b-gsm8k.yaml [--max-steps N]

The YAML has four sections:

    model / run_name        which HF model, name of the output folder
    task                    'gsm8k' (default) or 'arc' (see arc.py)
    data                    dataset split + optional subset size (gsm8k) / task selection (arc)
    lora                    PEFT LoRA hyperparameters (omit for full fine-tuning)
    quantization            4-bit QLoRA loading (for big models on small GPUs)
    grpo                    passed straight into ``trl.GRPOConfig``

Everything under ``grpo`` is a real GRPOConfig field, so the TRL docs apply.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Less fragmentation of the CUDA caching allocator: generation and the loss pass have very
# different allocation patterns, and on a small GPU the fragmentation alone can push the
# reserved pool to the VRAM ceiling. Must be set before torch initialises CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import yaml
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from .arc import load_arc_train
from .arc_rewards import ARC_REWARD_FUNCS
from .data import load_gsm8k
from .rewards import REWARD_FUNCS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override grpo.max_steps (smoke tests)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_name = cfg["run_name"]
    output_dir = args.output_dir or f"outputs/{run_name}"

    grpo_kwargs = dict(cfg["grpo"])
    if args.max_steps is not None:
        grpo_kwargs["max_steps"] = args.max_steps
    grpo_kwargs.setdefault("report_to", "none")
    grpo_kwargs.setdefault("bf16", True)
    grpo_kwargs.setdefault("logging_steps", 1)
    grpo_config = GRPOConfig(output_dir=output_dir, run_name=run_name, **grpo_kwargs)

    data_cfg = cfg.get("data", {})
    task = cfg.get("task", "gsm8k")
    if task == "gsm8k":
        train_ds = load_gsm8k(data_cfg.get("split", "train"), limit=data_cfg.get("limit"), seed=grpo_config.seed)
        all_rewards = REWARD_FUNCS
    elif task == "arc":
        train_ds = load_arc_train(data_cfg, seed=grpo_config.seed)
        all_rewards = ARC_REWARD_FUNCS
    else:
        raise ValueError(f"unknown task {task!r}")
    print(f"task: {task}  train examples: {len(train_ds)}")

    reward_names = cfg.get("rewards", list(all_rewards))
    reward_funcs = [all_rewards[n] for n in reward_names]

    peft_config = None
    if "lora" in cfg:
        peft_config = LoraConfig(task_type="CAUSAL_LM", **cfg["lora"])

    quant_config = None
    if cfg.get("quantization", {}).get("load_in_4bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        peft_config=peft_config,
        quantization_config=quant_config,
    )
    trainer.train()

    final = Path(output_dir) / "final"
    trainer.save_model(str(final))  # LoRA adapter (or full weights if no lora section)
    trainer.state.save_to_json(str(final / "trainer_state.json"))
    (final / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"saved to {final}")


if __name__ == "__main__":
    main()
