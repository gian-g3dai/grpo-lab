"""Supervised fine-tuning on augmented re-arc examples: the stage before RL on ARC.

Usage:
    python -m grpo_lab.sft --config configs/qwen2.5-7b-arc-sft.yaml [--max-steps N]

Why a separate stage: GRPO only amplifies behaviour the sampler already produces. A stock
instruct model almost never writes an exact ARC output grid, so nearly every group of
rollouts has zero reward variance and no gradient. SFT on re-arc examples, shown in random
rotations / mirrors / colour relabellings (``arc_augment.py``), is the cheap, dense-signal
way to get pass@k off the floor. The LoRA this produces is what ``train.py`` should start
from (``init_adapter:`` in the GRPO config).

The YAML mirrors ``train.py``:

    model / run_name        which HF model, name of the output folder
    task                    must be 'arc'
    data                    task selection, samples_per_task, n_demos (int or [lo, hi]), augment: {...}
    lora                    PEFT LoRA hyperparameters (omit for full fine-tuning)
    quantization            4-bit QLoRA loading
    sft                     passed straight into ``trl.SFTConfig``

The dataset is TRL's conversational prompt-completion format (``prompt`` = system + user
messages, ``completion`` = the assistant message holding the gold grid), so the loss is on
the completion only and the chat template matches what the GRPO trainer and the evaluator
apply at inference. The completion is just the grid in a ```` ```grid ```` block, no
reasoning: direct output is what every strong open-weights ARC entry trains.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import yaml
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .arc import answer_to_completion, load_arc_train


def to_prompt_completion(ds):
    """``prompt``/``answer`` rows (what the GRPO trainer consumes) -> TRL prompt-completion rows."""
    return ds.map(
        lambda ex: {"completion": [{"role": "assistant", "content": answer_to_completion(ex["answer"])}]},
        remove_columns=[c for c in ds.column_names if c not in ("prompt",)],
    )


def drop_too_long(ds, tokenizer, max_length: int | None):
    """Remove rows whose chat-templated prompt + completion would be truncated.

    TRL truncates from the right, which would cut the grid (the only part with a loss), so a
    row that does not fit is useless and misleading. The ``data.max_prompt_cells`` filter
    bounds this roughly (one token per cell plus one per row); this is the exact check.
    """
    if max_length is None:
        return ds

    def n_tokens(ex):
        text = tokenizer.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False)
        return {"n_tokens": len(tokenizer(text, add_special_tokens=False)["input_ids"])}

    with_len = ds.map(n_tokens)
    kept = with_len.filter(lambda ex: ex["n_tokens"] <= max_length)
    print(
        f"sequence length: max {max(with_len['n_tokens'])} tokens, "
        f"{len(ds) - len(kept)} of {len(ds)} rows exceed max_length={max_length} and are dropped"
    )
    return kept.remove_columns(["n_tokens"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="override sft.max_steps (smoke tests)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if cfg.get("task", "gsm8k") != "arc":
        raise SystemExit("grpo_lab.sft only implements the ARC task")
    run_name = cfg["run_name"]
    output_dir = args.output_dir or f"outputs/{run_name}"

    sft_kwargs = dict(cfg["sft"])
    if args.max_steps is not None:
        sft_kwargs["max_steps"] = args.max_steps
    sft_kwargs.setdefault("report_to", "none")
    sft_kwargs.setdefault("bf16", True)
    sft_kwargs.setdefault("logging_steps", 10)
    sft_kwargs.setdefault("completion_only_loss", True)
    sft_config = SFTConfig(output_dir=output_dir, run_name=run_name, **sft_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    train_ds = to_prompt_completion(load_arc_train(cfg.get("data", {}), seed=sft_config.seed))
    train_ds = drop_too_long(train_ds, tokenizer, sft_config.max_length)
    print(f"task: arc  SFT examples: {len(train_ds)}  augment: {cfg.get('data', {}).get('augment')}")

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

    trainer = SFTTrainer(
        model=cfg["model"],
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
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
